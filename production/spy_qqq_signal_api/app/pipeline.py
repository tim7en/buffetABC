"""Signal computation engine.

Flow (runs daily after market close):
  1. refresh_data()        — download latest QQQ, SPY, macro, and FRED data
  2. compute_signals()     — build walk-forward ML signals for SPY (gate) and
                             QQQ (expression), apply SPY-gate policy, compute
                             VIX-acceleration trigger, persist state.json

The walk-forward classifier reproduces the exact same methodology as the
research backtests (monthly refit, logistic + random-forest ensemble blend,
1-day lag, same features) so the live signal is consistent with the backtested
performance.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Tool imports (mounted at /app/tools inside the container)
# --------------------------------------------------------------------------- #
TOOLS_DIR = Path("/app/tools")
CACHE_SCRIPTS_DIR = Path("/app/cache_scripts")
for _p in [str(TOOLS_DIR), str(CACHE_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qqq_macro_ml_regime_analysis as ra          # noqa: E402
import qqq_macro_walkforward_model_compare as mc   # noqa: E402

# --------------------------------------------------------------------------- #
# Paths  (override DATA_DIR env-var for local testing)
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CACHE_DIR = DATA_DIR / "cache"
FRED_CACHE_DIR = DATA_DIR / "fred"
QQQ_PARQUET = CACHE_DIR / "QQQ_daily.parquet"
SPY_PARQUET = CACHE_DIR / "SPY_daily.parquet"
MACRO_PARQUET = CACHE_DIR / "macro_daily_1999.parquet"

# --------------------------------------------------------------------------- #
# Model hyper-parameters  (mirrors the backtested configuration)
# --------------------------------------------------------------------------- #
RISK_OFF_THRESHOLD = 0.45
JUMP_IN_THRESHOLD = 0.55
MIN_TRAIN_MONTHS = 96
RF_ESTIMATORS = 300
TARGET_HORIZON = 63         # 63 trading-day (~3-month) forward return target
PRICE_START = "1999-03-10"
MACRO_START = "1999-01-01"
MONTHLY_LAG_DAYS = 45       # CPI / unemployment publication lag
QUARTERLY_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 365

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; spy-qqq-signal-api/1.0)"}

log = logging.getLogger(__name__)


# =========================================================================== #
# Data download
# =========================================================================== #

def _yahoo_get(session: requests.Session, url: str, params: dict[str, Any]) -> str:
    """GET with exponential-backoff retry (mirrors existing download logic)."""
    last: Exception | None = None
    for attempt in range(6):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=45)
            if r.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last = exc
            if attempt < 5:
                time.sleep(min(2.0 * (attempt + 1), 20.0))
    raise RuntimeError(f"Yahoo request failed: {last}") from last


def _save_price_parquet(ticker: str, output_path: Path) -> None:
    """Download Yahoo Finance daily history for *ticker* and save as parquet.

    The saved columns match what ``ra.load_qqq()`` expects:
      time (UTC datetime), adj_c (float), v (int/float)
    """
    session = requests.Session()
    start_dt = datetime(1999, 3, 10, tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) + timedelta(days=1)
    text = _yahoo_get(
        session,
        YAHOO_URL.format(ticker=quote(ticker, safe="")),
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        },
    )
    payload = json.loads(text)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"No Yahoo chart data returned for {ticker}")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_data = (indicators.get("quote") or [{}])[0]
    adjclose_list = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s", utc=True),
        "c": pd.to_numeric(quote_data.get("close"), errors="coerce"),
        "adj_c": (
            pd.to_numeric(adjclose_list, errors="coerce")
            if adjclose_list else pd.Series(dtype=float)
        ),
        "v": pd.to_numeric(quote_data.get("volume"), errors="coerce"),
    })
    # Fallback: if adjusted close is all-NaN, use unadjusted
    if "adj_c" not in df.columns or df["adj_c"].isna().all():
        df["adj_c"] = df["c"]

    df = df.dropna(subset=["c"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Saved %s: %d rows → %s", ticker, len(df), output_path)


def refresh_data() -> None:
    """Download the latest QQQ, SPY, and macro data.

    Price data is downloaded directly via the Yahoo Finance API.
    Macro data is downloaded by calling the existing cache script as a
    subprocess so we don't have to duplicate its FRED + scraping logic.
    """
    log.info("=== Data refresh started ===")

    log.info("Downloading QQQ prices …")
    _save_price_parquet("QQQ", QQQ_PARQUET)

    log.info("Downloading SPY prices …")
    _save_price_parquet("SPY", SPY_PARQUET)

    log.info("Downloading macro data (FRED + Yahoo cross-assets) …")
    macro_script = CACHE_SCRIPTS_DIR / "download_daily_macro_data.py"
    result = subprocess.run(
        [sys.executable, str(macro_script),
         "--start", MACRO_START,
         "--cache-dir", str(CACHE_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Macro download failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    log.info("=== Data refresh complete ===")


# =========================================================================== #
# Signal computation
# =========================================================================== #

def _build_signals_for(
    price_path: Path,
    label: str,
    refresh_fred: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (daily_signal, blend_monthly, daily_dataset) for a price series.

    *price_path* must be a parquet with columns compatible with ``ra.load_qqq``
    (i.e. ``time``/``date``, ``adj_c``/``c``, ``v``/``volume``).

    The ML pipeline is identical to the research backtest:
      • 40-feature walk-forward monthly logistic + random-forest classifiers
      • Ensemble probability blend (avg risk-off and jump-in probs)
      • Daily signal with 1-trading-day lag to prevent look-ahead
    """
    log.info("Building signals for %s …", label)

    # ---- Load price data ------------------------------------------ #
    price_data = ra.load_qqq(price_path, start=PRICE_START, end=None)

    # ---- Align macro to this price series' trading days ------------ #
    macro = ra.load_macro(
        MACRO_PARQUET,
        price_data.index,
        monthly_release_lag_days=MONTHLY_LAG_DAYS,
        quarterly_release_lag_days=QUARTERLY_LAG_DAYS,
        annual_release_lag_days=ANNUAL_LAG_DAYS,
    )

    # ---- FRED stress proxies (VIX, HY OAS, NFCI, T10Y3M) ---------- #
    stress, status = ra.load_stress_proxies(
        price_data.index, FRED_CACHE_DIR, refresh=refresh_fred
    )
    for k, v in status.items():
        if v != "loaded":
            log.warning("  %s stress series '%s': %s", label, k, v)

    # ---- Feature engineering + forward-return targets -------------- #
    # build_dataset() includes: momentum, macro, cross-asset, sentiment PCA,
    # shock scores, and forward-return targets for the classifier.
    dataset = ra.build_dataset(price_data, macro, stress, target_horizon=TARGET_HORIZON)

    # ---- Monthly end-of-month sample for ML ------------------------ #
    monthly = ra.month_end_sample(dataset)

    # ---- Walk-forward logistic regression -------------------------- #
    log.info("  Walk-forward logistic regression (%s) …", label)
    logistic_monthly = mc.build_walkforward_monthly_classifier_regimes(
        monthly,
        ra.MODEL_FEATURES,
        model_name="logistic",
        target_horizon=TARGET_HORIZON,
        min_train_months=MIN_TRAIN_MONTHS,
        risk_off_threshold=RISK_OFF_THRESHOLD,
        jump_in_threshold=JUMP_IN_THRESHOLD,
        random_state=ra.RANDOM_STATE,
        rf_estimators=RF_ESTIMATORS,
    )

    # ---- Walk-forward random forest -------------------------------- #
    log.info("  Walk-forward random forest (%s) …", label)
    rf_monthly = mc.build_walkforward_monthly_classifier_regimes(
        monthly,
        ra.MODEL_FEATURES,
        model_name="random_forest",
        target_horizon=TARGET_HORIZON,
        min_train_months=MIN_TRAIN_MONTHS,
        risk_off_threshold=RISK_OFF_THRESHOLD,
        jump_in_threshold=JUMP_IN_THRESHOLD,
        random_state=ra.RANDOM_STATE,
        rf_estimators=RF_ESTIMATORS,
    )

    # ---- Ensemble blend (avg logistic + RF probabilities) ---------- #
    blend = mc.build_probability_blend_regimes(
        logistic_monthly,
        rf_monthly,
        risk_off_threshold=RISK_OFF_THRESHOLD,
        jump_in_threshold=JUMP_IN_THRESHOLD,
    )

    # ---- Convert monthly signal → daily with 1-day lag ------------ #
    daily_signal = mc.monthly_signal_to_daily(blend["regime"], dataset.index)
    log.info("  %s signal range: %s … %s", label,
             daily_signal.first_valid_index(), daily_signal.index[-1])

    return daily_signal, blend, dataset


def _apply_spy_gate(spy_daily: pd.Series, qqq_daily: pd.Series) -> pd.Series:
    """Combine SPY gate and QQQ expression into the policy signal.

    Rules (1-day lagged, matches the backtest):
      - spy == risk_off                       → risk_off  (circuit breaker)
      - spy == risk_on AND qqq != risk_off    → risk_on   (full deployment)
      - everything else                        → neutral   (1× unlevered)
    """
    joined = (
        pd.concat([spy_daily.rename("spy"), qqq_daily.rename("qqq")], axis=1, join="inner")
        .dropna()
    )
    policy = pd.Series(index=joined.index, dtype=object)
    policy[joined["spy"] == "risk_off"] = "risk_off"
    policy[(joined["spy"] == "risk_on") & (joined["qqq"] != "risk_off")] = "risk_on"
    policy[policy.isna()] = "neutral"
    return policy


def _latest_prob(blend: pd.DataFrame, col: str) -> Optional[float]:
    valid = blend[col].dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def compute_signals() -> dict[str, Any]:
    """Run the full SPY-gate × QQQ-ensemble pipeline and return a state dict.

    This function is called:
      • Once at container startup (if the cached state is stale / missing)
      • Every trading day at ~17:30 ET by the scheduler

    Returns a dict that matches the ``SignalState`` dataclass fields so it can
    be persisted directly to ``signal_state.json``.
    """
    computed_at = datetime.now(timezone.utc).isoformat()
    log.info("=== Signal computation started at %s ===", computed_at)

    try:
        # SPY is the gate (broad-market circuit breaker)
        # QQQ is the expression (leveraged upside vehicle)
        spy_daily, spy_blend, spy_dataset = _build_signals_for(
            SPY_PARQUET, "SPY", refresh_fred=True
        )
        qqq_daily, qqq_blend, qqq_dataset = _build_signals_for(
            QQQ_PARQUET, "QQQ", refresh_fred=False   # FRED already fresh from SPY pass
        )

        # ---- SPY-gate policy ----------------------------------------- #
        policy = _apply_spy_gate(spy_daily, qqq_daily)
        valid_policy = policy.dropna()
        if valid_policy.empty:
            raise RuntimeError("Policy signal is empty — insufficient data overlap.")

        today_ts: pd.Timestamp = valid_policy.index[-1]
        today_signal = str(valid_policy.iloc[-1])
        today_spy = str(spy_daily.reindex([today_ts]).iloc[0]) if today_ts in spy_daily.index else "unknown"
        today_qqq = str(qqq_daily.reindex([today_ts]).iloc[0]) if today_ts in qqq_daily.index else "unknown"

        # ---- Probabilities from last fitted month -------------------- #
        spy_ro_prob = _latest_prob(spy_blend, "risk_off_probability")
        spy_ji_prob = _latest_prob(spy_blend, "jump_in_probability")
        qqq_ro_prob = _latest_prob(qqq_blend, "risk_off_probability")
        qqq_ji_prob = _latest_prob(qqq_blend, "jump_in_probability")

        # ---- Model training coverage --------------------------------- #
        spy_trained_through = (
            spy_blend.dropna(subset=["risk_off_probability"]).index[-1].isoformat()
            if not spy_blend.dropna(subset=["risk_off_probability"]).empty else None
        )
        qqq_trained_through = (
            qqq_blend.dropna(subset=["risk_off_probability"]).index[-1].isoformat()
            if not qqq_blend.dropna(subset=["risk_off_probability"]).empty else None
        )

        # ---- VIX-acceleration DCA trigger (uses lagged values) ------- #
        # When triggered the DCA contribution is doubled (2× multiplier).
        # Conditions (all lagged 1 day to match the backtest):
        #   policy == risk_on  AND  VIX ≥ 20  AND  VIX rising  AND  QQQ 21d < 0
        def _safe_float(ds: pd.Series, ts: pd.Timestamp) -> Optional[float]:
            val = ds.shift(1).reindex([ts])
            if val.empty or pd.isna(val.iloc[0]):
                return None
            return float(val.iloc[0])

        vix_lag = _safe_float(spy_dataset["vix_level"], today_ts)
        vix_chg_lag = _safe_float(spy_dataset["vix_21d_change"], today_ts)
        qqq_ret_lag = _safe_float(qqq_dataset["qqq_21d_return"], today_ts)

        vix_accel = bool(
            today_signal == "risk_on"
            and vix_lag is not None and vix_lag >= 20.0
            and vix_chg_lag is not None and vix_chg_lag > 0.0
            and qqq_ret_lag is not None and qqq_ret_lag < 0.0
        )

        # Current (unlagged) VIX for display
        vix_now_series = spy_dataset["vix_level"].reindex([today_ts])
        vix_now: Optional[float] = (
            float(vix_now_series.iloc[0])
            if not vix_now_series.empty and pd.notna(vix_now_series.iloc[0])
            else None
        )

        # ---- 90-day rolling history ---------------------------------- #
        history_policy = valid_policy.tail(90)
        history = [
            {
                "date": d.date().isoformat(),
                "policy_signal": str(s),
                "spy_signal": (
                    str(spy_daily.reindex([d]).iloc[0])
                    if d in spy_daily.index else "unknown"
                ),
                "qqq_signal": (
                    str(qqq_daily.reindex([d]).iloc[0])
                    if d in qqq_daily.index else "unknown"
                ),
            }
            for d, s in history_policy.items()
        ]

        state: dict[str, Any] = {
            "computed_at_utc": computed_at,
            "as_of_date": today_ts.date().isoformat(),
            "policy_signal": today_signal,
            "spy_signal": today_spy,
            "qqq_signal": today_qqq,
            "spy_risk_off_prob": spy_ro_prob,
            "spy_jump_in_prob": spy_ji_prob,
            "qqq_risk_off_prob": qqq_ro_prob,
            "qqq_jump_in_prob": qqq_ji_prob,
            "vix_accel_trigger": vix_accel,
            "vix_level": vix_now,
            "spy_model_trained_through": spy_trained_through,
            "qqq_model_trained_through": qqq_trained_through,
            "history": history,
            "error": None,
        }
        log.info(
            "=== Signal computation complete: %s (SPY=%s, QQQ=%s, VIX=%.1f) ===",
            today_signal, today_spy, today_qqq, vix_now or 0.0,
        )

    except Exception as exc:
        log.exception("Signal computation failed: %s", exc)
        state = {
            "computed_at_utc": computed_at,
            "as_of_date": date.today().isoformat(),
            "policy_signal": "unknown",
            "spy_signal": "unknown",
            "qqq_signal": "unknown",
            "spy_risk_off_prob": None,
            "spy_jump_in_prob": None,
            "qqq_risk_off_prob": None,
            "qqq_jump_in_prob": None,
            "vix_accel_trigger": False,
            "vix_level": None,
            "spy_model_trained_through": None,
            "qqq_model_trained_through": None,
            "history": [],
            "error": str(exc),
        }

    return state


def run_full_refresh() -> dict[str, Any]:
    """Convenience wrapper: refresh data then compute signals."""
    refresh_data()
    return compute_signals()
