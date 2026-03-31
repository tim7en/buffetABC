"""
Fund Manager Backtest — Session Turtle Core x3
===============================================
Production-grade rolling moderator implementing the operational fund-manager
logic used for this repo's strategy moderator research. It is the current
reference runner for moderation decisions, but some scoring inputs remain
proxies for the higher-level playbook described in strategy-moderator.jsx.

Architecture
------------
The backtest engine runs in a single pass over precomputed candidates.
To simulate the fund manager we pre-filter the candidate list BEFORE
handing it to the engine:

  1. Run BASELINE (no moderation) to collect full trade history + equity curve.
  2. Build monthly review decisions using 13-week rolling lookback:
       - Phase W1-4  (days 0-27)  : observation only — zero filtering.
       - Phase W5-8  (days 28-55) : reduce weights only — no removal.
       - Phase W9+   (days 56+)   : full rotation — keep/reduce/remove.
  3. Per-asset state machine:
       - Probation: C-tier → 4-week hold before reduction applied.
       - Min tenure: 8 weeks live before removal eligible.
  4. Circuit breakers (ref max DD = 30.36%):
       - DD > 60% of ref (18.22%) → reduce ALL new entries by 50%.
       - DD > 80% of ref (24.29%) → halt ALL new entries.
       - DD < 40% of ref (12.14%) → resume normal sizing.
  5. Pre-filter candidates incorporating all decisions + circuit breaker state.
  6. Run MODERATED backtest with filtered candidates.

Scoring — 4 operational proxy pillars × 25 pts = 100
-----------------------------------------------------
  Pillar 1  Risk-Adjusted Return  : Sharpe approx (annualized from trade P&L)
  Pillar 2  Stability             : Win-rate × Profit-factor composite
  Pillar 3  Trade Quality         : Sortino approx + MAR/Calmar approx
  Pillar 4  Expectancy            : (WR × avg_win) − (LR × avg_loss) + net return sign

Tier → Action
-------------
  A (≥ 75) : Full weight
  B (55-74) : Standard weight
  C (40-54) : Reduce 50%, 4-week probation
  D (25-39) : Reduce 25% (diversification hold)
  F (< 25)  : Remove (blocked)
  INSUFFICIENT : Keep (benefit of the doubt)

Forward-look bias guards
------------------------
  * Scoring window uses only trades with exit_ts < review_date.
  * Circuit breaker uses baseline equity at time of each candidate entry.
  * All decisions apply only to candidates with entry_ts ≥ review_date.
  * Probation / min-tenure clocks start from strategy start_ts (W1 clock).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django
django.setup()

from edgar.services.session_turtle_portfolio import (
    CORE_SESSION_TURTLE_UNIVERSE,
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)

# ── Phase-gate config (aligned to strategy-moderator.jsx Decision Gate) ─────────
OBSERVE_ONLY_DAYS = 28    # W1-W4  : no changes whatsoever
REDUCE_ONLY_DAYS  = 56    # W5-W8  : reduce weights, never remove
FULL_ROTATION_DAYS = 56   # W9+    : same as REDUCE_ONLY end boundary

# ── Scoring config ──────────────────────────────────────────────────────────────
SCORE_LOOKBACK_DAYS = 91  # 13 rolling weeks
REVIEW_FREQ_DAYS    = 30  # monthly scoring pass
MIN_TRADES_TO_SCORE = 3   # fewer → INSUFFICIENT → keep at full weight

# ── Tier thresholds (aligned to strategy-moderator.jsx) ─────────────────────────
TIER_A = 75   # Full weight
TIER_B = 55   # Standard weight
TIER_C = 40   # Reduce 50% — enters probation
TIER_D = 25   # Reduce 25% — diversification hold
# below TIER_D → Tier F (remove if eligible)

# ── Asset lifecycle rules ───────────────────────────────────────────────────────
PROBATION_DAYS = 28   # 4 weeks — C-tier must remain C before 50% kicks in
MIN_TENURE_DAYS = 56  # 8 weeks live before removal is permitted

# ── Diversification hold rules ──────────────────────────────────────────────────
# D/F-tier assets are only kept at 25% when they still offer diversification value
# versus the currently retained portfolio.
DIVERSIFICATION_CORR_THRESHOLD = 0.30
MIN_DIVERSIFICATION_OBS = 20

# ── Circuit-breaker thresholds (aligned to strategy-moderator.jsx HALT/CIRCUIT BREAKER) ───
# Reference max DD = 30.36% from production backtest
REF_MAX_DD_PCT   = 30.36
CB_REDUCE_DD_PCT = REF_MAX_DD_PCT * 0.60   # 18.22% → reduce all entries 50%
CB_HALT_DD_PCT   = REF_MAX_DD_PCT * 0.80   # 24.29% → halt all new entries
CB_RESUME_DD_PCT = REF_MAX_DD_PCT * 0.40   # 12.14% → resume normal sizing

# ── Output ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("reports/fund_manager_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy params (identical to production backtest) ─────────────────────────
_BASKET        = "core"
_LOOKBACK_YRS  = 4.1
_CHANNEL       = 20
_EXPOSURE_MULT = 3.0
_BASE_RISK     = 0.05
_FIXED_STOP    = 0.10
_DIR_VOL_RISK  = 0.07
_TREND_FAST    = 55
_TREND_SLOW    = 200
_EMA_PERIOD    = 200

_OVERLAY_KWARGS = dict(
    use_extended_hours_proxy=True,
    extended_hours_proxy_lag_days=1,
    extended_hours_vix_risk_on_threshold=15.0,
    extended_hours_vix_risk_off_threshold=25.0,
    extended_hours_fg_greed_threshold=60.0,
    extended_hours_fg_fear_threshold=30.0,
    extended_hours_long_risk_on_mult=1.0,
    extended_hours_long_neutral_mult=1.0,
    extended_hours_long_risk_off_mult=0.5,
    extended_hours_short_risk_on_mult=1e-9,
    extended_hours_short_neutral_mult=1.0,
    extended_hours_short_risk_off_mult=1.0,
    use_per_asset_technical_overlay=True,
    per_asset_ema_lag_days=1,
    per_asset_ema_above_long_mult=1.0,
    per_asset_ema_above_short_mult=0.0,
    per_asset_ema_below_long_mult=0.0,
    per_asset_ema_below_short_mult=1.0,
    per_asset_use_adx_gate=False,
)

_BASE_KWARGS = dict(
    basket=_BASKET,
    exposure_mult=_EXPOSURE_MULT,
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    base_risk_pct=_BASE_RISK,
    fixed_stop_pct=_FIXED_STOP,
    directional_volume_risk_pct=_DIR_VOL_RISK,
)


# ── Date helpers ───────────────────────────────────────────────────────────────

def _add_days(dt: datetime, days: int) -> datetime:
    return dt + timedelta(days=days)


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    """Simple Pearson correlation with minimal dependencies."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    return cov / math.sqrt(var_x * var_y)


def _build_daily_return_state(technical_state: dict) -> dict[str, dict[datetime.date, float]]:
    """
    Build daily close-to-close returns from the already-computed daily EMA state.

    Returns:
        {ticker: {date: daily_return}}
    """
    out: dict[str, dict[datetime.date, float]] = {}
    daily_ema = technical_state.get("daily_ema", {}) if technical_state else {}
    for ticker, rows in daily_ema.items():
        ordered = sorted(
            (
                datetime.fromisoformat(day).date(),
                float(payload.get("close", 0.0) or 0.0),
            )
            for day, payload in rows.items()
        )
        ticker_returns: dict[datetime.date, float] = {}
        prev_close: float | None = None
        for day, close in ordered:
            if prev_close is not None and prev_close > 0:
                ticker_returns[day] = (close / prev_close) - 1.0
            prev_close = close
        out[str(ticker)] = ticker_returns
    return out


def _avg_abs_corr_to_portfolio(
    *,
    ticker: str,
    peer_tickers: list[str],
    review_dt: datetime,
    daily_return_state: dict[str, dict[datetime.date, float]],
    lookback_days: int = SCORE_LOOKBACK_DAYS,
    min_obs: int = MIN_DIVERSIFICATION_OBS,
) -> float | None:
    """
    Average absolute correlation of one asset to the current retained peer set.

    Uses prior daily returns only, over the same 13-week lookback window as the
    scoring pass. Returns None when correlation cannot be estimated reliably.
    """
    own = daily_return_state.get(ticker, {})
    if not own or not peer_tickers:
        return None

    end_day = review_dt.date()
    start_day = (review_dt - timedelta(days=lookback_days)).date()
    corrs: list[float] = []
    for peer in peer_tickers:
        peer_series = daily_return_state.get(peer, {})
        if not peer_series:
            continue
        common_days = sorted(
            d for d in own.keys() & peer_series.keys()
            if start_day <= d < end_day
        )
        if len(common_days) < min_obs:
            continue
        xs = [own[d] for d in common_days]
        ys = [peer_series[d] for d in common_days]
        corr = _pearson_corr(xs, ys)
        if corr is not None:
            corrs.append(abs(corr))

    if not corrs:
        return None
    return sum(corrs) / len(corrs)


# ── Equity-curve builder ───────────────────────────────────────────────────────

def _build_equity_curve(trades: list[dict]) -> list[tuple[datetime, float]]:
    """
    Returns sorted list of (exit_ts, equity_after_exit) from baseline trades.
    Equity is the running equity stamped at each trade exit.
    """
    valid = [t for t in trades if t.get("equity_after_exit") is not None]
    valid.sort(key=lambda t: datetime.fromisoformat(t["exit_ts"]))
    return [(datetime.fromisoformat(t["exit_ts"]), float(t["equity_after_exit"])) for t in valid]


def _portfolio_dd_at(equity_curve: list[tuple[datetime, float]], at_ts: datetime) -> float:
    """
    Returns the realised drawdown % (0-100) at a given timestamp, computed
    from the equity high-water-mark up to (but not including) that timestamp.
    Uses the last known equity value before at_ts.
    """
    relevant = [(ts, eq) for ts, eq in equity_curve if ts < at_ts]
    if not relevant:
        return 0.0
    eq_values = [eq for _, eq in relevant]
    peak = max(eq_values)
    current = eq_values[-1]
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - current) / peak * 100.0)


# ── Multi-metric scoring ───────────────────────────────────────────────────────

def _score_asset(
    trades: list[dict],
    window_days: int = SCORE_LOOKBACK_DAYS,
) -> tuple[int, str, dict]:
    """
    Compute a composite 0-100 score across 4 proxy pillars (25 pts each).
    This is an operational scoring proxy, not a literal implementation of every
    narrative metric listed in strategy-moderator.jsx.

    Returns (score, tier, metrics_dict).
    Returns (-1, 'INSUFFICIENT', {}) when too few trades.
    """
    if len(trades) < MIN_TRADES_TO_SCORE:
        return -1, "INSUFFICIENT", {}

    pnls   = [float(t["net_pnl"]) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(pnls)

    total_return = sum(pnls)
    avg_pnl      = total_return / n
    wr           = len(wins) / n
    gross_win    = sum(wins)           if wins   else 0.0
    gross_loss   = abs(sum(losses))    if losses else 1e-9
    pf           = gross_win / gross_loss
    avg_win      = gross_win / len(wins)  if wins   else 0.0
    avg_loss     = gross_loss / len(losses) if losses else 0.0
    expectancy   = wr * avg_win - (1 - wr) * avg_loss

    # ── Sharpe approx ──────────────────────────────────────────────────────────
    # Annualise using average holding period derived from window length / n_trades
    if n >= 2:
        mean_p = avg_pnl
        std_p  = math.sqrt(sum((p - mean_p) ** 2 for p in pnls) / (n - 1))
        # Annualisation factor: sqrt(trades_per_year)
        trades_per_year = n / max(window_days, 1) * 252
        sharpe = (mean_p / std_p) * math.sqrt(trades_per_year) if std_p > 1e-12 else 0.0
    else:
        sharpe = 0.0

    # ── Sortino approx ─────────────────────────────────────────────────────────
    downside_pnls = [p - avg_pnl for p in pnls if p < avg_pnl]
    if downside_pnls and n >= 2:
        down_std = math.sqrt(sum(d ** 2 for d in downside_pnls) / len(downside_pnls))
        trades_per_year = n / max(window_days, 1) * 252
        sortino = (avg_pnl / down_std) * math.sqrt(trades_per_year) if down_std > 1e-12 else 0.0
    else:
        sortino = sharpe

    # ── MAR / Calmar approx ────────────────────────────────────────────────────
    running_total = 0.0
    peak_total    = 0.0
    max_dd_abs    = 0.0
    for p in pnls:
        running_total += p
        if running_total > peak_total:
            peak_total = running_total
        dd = peak_total - running_total
        if dd > max_dd_abs:
            max_dd_abs = dd

    # Calmar = annualised return / max_dd (as %)
    ann_return    = total_return / max(window_days, 1) * 365
    max_dd_pct    = (max_dd_abs / max(abs(peak_total), abs(pnls[0]), 1e-9)) * 100
    mar           = ann_return / max(max_dd_pct, 1e-9)

    # ── Pillar 1: Risk-Adjusted Return (25 pts) — Sharpe-based ────────────────
    if   sharpe >= 0.8:   p1 = 25
    elif sharpe >= 0.3:   p1 = 17
    elif sharpe >= 0.0:   p1 = 8
    else:                 p1 = 0

    # ── Pillar 2: Stability (25 pts) — win-rate + profit-factor ───────────────
    if   wr >= 0.45 and pf >= 1.5:  p2 = 25
    elif wr >= 0.35 and pf >= 1.0:  p2 = 17
    elif wr >= 0.25 and pf >= 0.7:  p2 = 8
    else:                           p2 = 0

    # ── Pillar 3: Trade Quality (25 pts) — Sortino + MAR ─────────────────────
    if   sortino >= 1.0 and mar >= 0.5:  p3 = 25
    elif sortino >= 0.5 and mar >= 0.2:  p3 = 17
    elif sortino >= 0.0 and mar >= 0.0:  p3 = 8
    else:                                p3 = 0

    # ── Pillar 4: Expectancy (25 pts) ─────────────────────────────────────────
    if   expectancy > 0 and total_return > 0:    p4 = 25
    elif expectancy > 0:                          p4 = 17
    elif expectancy > -abs(avg_pnl) * 0.5:       p4 = 8
    else:                                         p4 = 0

    score = p1 + p2 + p3 + p4
    tier  = ("A" if score >= TIER_A else
             "B" if score >= TIER_B else
             "C" if score >= TIER_C else
             "D" if score >= TIER_D else "F")

    return score, tier, {
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "mar":           round(mar, 3),
        "max_dd_pct":    round(max_dd_pct, 2),
        "expectancy":    round(expectancy, 2),
        "win_rate":      round(wr, 3),
        "profit_factor": round(pf, 3),
        "total_return":  round(total_return, 2),
        "n_trades":      n,
        "p1": p1, "p2": p2, "p3": p3, "p4": p4,
    }


# ── Per-asset state machine ────────────────────────────────────────────────────

class AssetState:
    """
    Tracks each asset's lifecycle: phase gate, probation, min tenure, current tier.
    """
    __slots__ = ("ticker", "start_ts", "probation_since", "tier_history")

    def __init__(self, ticker: str, start_ts: datetime) -> None:
        self.ticker         = ticker
        self.start_ts       = start_ts
        self.probation_since: datetime | None = None
        self.tier_history: list[tuple[datetime, str]] = []

    def tenure_days(self, at_ts: datetime) -> float:
        return (at_ts - self.start_ts).total_seconds() / 86_400

    def is_min_tenure_met(self, at_ts: datetime) -> bool:
        return self.tenure_days(at_ts) >= MIN_TENURE_DAYS

    def update(self, review_dt: datetime, new_tier: str) -> None:
        """
        Record new tier and manage probation clock.

        Any sub-55 tier (C/D/F) starts a 4-week probation clock. Returning to A/B
        or insufficient-data status clears it.
        """
        self.tier_history.append((review_dt, new_tier))
        if new_tier in {"C", "D", "F"}:
            if self.probation_since is None:
                self.probation_since = review_dt
        else:
            self.probation_since = None

    def probation_completed(self, at_ts: datetime) -> bool:
        """True when asset has been in C-tier for >= PROBATION_DAYS."""
        if self.probation_since is None:
            return False
        return (at_ts - self.probation_since).total_seconds() / 86_400 >= PROBATION_DAYS

    def effective_keep_ratio(
        self,
        review_dt: datetime,
        tier: str,
        phase: str,  # "observe" | "reduce_only" | "full"
        diversifies: bool | None = None,
    ) -> float:
        """
        Derive keep-ratio from tier, phase, and state.

        strategy-moderator.jsx rules:
          KEEP Full  (A)         → 1.0
          KEEP Std   (B)         → 1.0
          REDUCE 50% (C)         → 0.5 immediately, with 4-week probation
          SCORE < 55 after probation:
            - diversifies        → 0.25 (diversification hold)
            - no diversification → 0.0 once min tenure is met
          INSUFFICIENT           → 1.0  (benefit of the doubt)

          Phase gates:
            observe_only  → always 1.0 (no action)
            reduce_only   → max reduction is 50%; never 25% hold or full removal
            full          → all actions allowed
        """
        if tier == "INSUFFICIENT" or phase == "observe":
            return 1.0

        if tier in ("A", "B"):
            return 1.0

        # Any persistent sub-55 state starts with an immediate 50% reduction.
        if not self.probation_completed(review_dt):
            return 0.5 if phase in ("reduce_only", "full") else 1.0

        if tier == "C":
            return 0.5

        # D/F after probation: keep only if diversification value remains or the
        # asset is too new to remove.
        keep_for_diversification = diversifies is not False
        if keep_for_diversification or not self.is_min_tenure_met(review_dt):
            return 0.25 if phase == "full" else 0.5

        if phase != "full":
            return 0.5

        if tier in {"D", "F"}:
            return 0.0

        return 1.0  # safety fallback


# ── Rolling decision map ───────────────────────────────────────────────────────

def _build_rolling_decisions(
    baseline_trades:  list[dict],
    asset_states:     dict[str, AssetState],
    all_tickers:      set[str],
    strategy_start:   datetime,
    last_entry:       datetime,
    daily_return_state: dict[str, dict[datetime.date, float]],
) -> dict[datetime, dict[str, tuple[int, str, float, dict]]]:
    """
    For each monthly review date, compute:
      - tier score and tier label for each ticker
      - effective keep_ratio given phase and asset state

    Returns: {review_dt: {ticker: (score, tier, keep_ratio, metrics)}}
    """
    # Index baseline trades by ticker with exit_ts datetime
    by_ticker: dict[str, list] = defaultdict(list)
    for t in baseline_trades:
        by_ticker[t["ticker"]].append(
            (datetime.fromisoformat(t["exit_ts"]), t)
        )

    decisions: dict[datetime, dict[str, tuple[int, str, float, dict]]] = {}

    # First review starts after W4 (OBSERVE_ONLY_DAYS)
    review_dt = _add_days(strategy_start, OBSERVE_ONLY_DAYS)

    while review_dt <= last_entry + timedelta(days=32):
        window_start = _add_days(review_dt, -SCORE_LOOKBACK_DAYS)
        days_live    = (review_dt - strategy_start).total_seconds() / 86_400

        # Determine phase
        if days_live < OBSERVE_ONLY_DAYS:
            phase = "observe"
        elif days_live < REDUCE_ONLY_DAYS:
            phase = "reduce_only"
        else:
            phase = "full"

        raw_scores: dict[str, tuple[int, str, dict]] = {}
        for ticker in all_tickers:
            window_trades = [
                td for (exit_dt, td) in by_ticker.get(ticker, [])
                if window_start <= exit_dt < review_dt   # strict: no look-ahead
            ]
            raw_scores[ticker] = _score_asset(window_trades)

        tick_decisions: dict[str, tuple[int, str, float, dict]] = {}
        retained_peer_candidates = {
            ticker for ticker, (_score, tier, _metrics) in raw_scores.items()
            if tier in {"A", "B", "C", "INSUFFICIENT"}
        }
        for ticker in all_tickers:
            score, tier, metrics = raw_scores[ticker]
            state   = asset_states[ticker]
            state.update(review_dt, tier)
            peer_tickers = sorted(retained_peer_candidates - {ticker})
            avg_abs_corr = _avg_abs_corr_to_portfolio(
                ticker=ticker,
                peer_tickers=peer_tickers,
                review_dt=review_dt,
                daily_return_state=daily_return_state,
            )
            diversifies = None if avg_abs_corr is None else avg_abs_corr < DIVERSIFICATION_CORR_THRESHOLD
            ratio = state.effective_keep_ratio(
                review_dt,
                tier,
                phase,
                diversifies=diversifies,
            )
            metrics = dict(metrics)
            metrics["phase"] = phase
            metrics["avg_abs_corr"] = round(avg_abs_corr, 3) if avg_abs_corr is not None else None
            metrics["diversifies"] = diversifies
            metrics["probation_complete"] = state.probation_completed(review_dt)
            metrics["tenure_days"] = round(state.tenure_days(review_dt), 1)
            tick_decisions[ticker] = (score, tier, ratio, metrics)

        decisions[review_dt] = tick_decisions
        review_dt = _add_days(review_dt, REVIEW_FREQ_DAYS)

    return decisions


# ── Circuit-breaker state builder ──────────────────────────────────────────────

def _build_circuit_breaker_state(
    equity_curve: list[tuple[datetime, float]],
) -> dict[datetime, str]:
    """
    Walk the equity curve and tag each timestamp with a circuit-breaker state:
      "normal"  : DD < CB_REDUCE_DD_PCT
      "reduce"  : CB_REDUCE_DD_PCT <= DD < CB_HALT_DD_PCT
      "halt"    : DD >= CB_HALT_DD_PCT

    Uses hysteresis: once in halt/reduce, only resumes when DD < CB_RESUME_DD_PCT.
    """
    cb_state: dict[datetime, str] = {}
    current_state = "normal"
    peak_equity   = 0.0

    for ts, eq in equity_curve:
        if eq > peak_equity:
            peak_equity = eq
        if peak_equity > 0:
            dd_pct = (peak_equity - eq) / peak_equity * 100.0
        else:
            dd_pct = 0.0

        if current_state == "normal":
            if   dd_pct >= CB_HALT_DD_PCT:   current_state = "halt"
            elif dd_pct >= CB_REDUCE_DD_PCT:  current_state = "reduce"
        elif current_state in ("reduce", "halt"):
            if   dd_pct >= CB_HALT_DD_PCT:   current_state = "halt"
            elif dd_pct >= CB_REDUCE_DD_PCT:  current_state = "reduce"
            elif dd_pct < CB_RESUME_DD_PCT:   current_state = "normal"

        cb_state[ts] = current_state

    return cb_state


def _get_cb_state_at(
    cb_state: dict[datetime, str],
    at_ts: datetime,
) -> str:
    """Return the most recent circuit-breaker state before at_ts."""
    relevant = [ts for ts in cb_state if ts < at_ts]
    if not relevant:
        return "normal"
    return cb_state[max(relevant)]


# ── Candidate filter ───────────────────────────────────────────────────────────

def _apply_fund_manager_filter(
    all_candidates:  list[dict],
    decisions:       dict[datetime, dict[str, tuple[int, str, float, dict]]],
    strategy_start:  datetime,
    cb_state:        dict[datetime, str],
) -> tuple[list[dict], list[datetime]]:
    """
    Filter candidates using:
      1. Phase-aware tier decisions (per asset state machine)
      2. Circuit-breaker state at time of each candidate entry

    Returns (filtered_candidates, sorted_review_dates).
    """
    sorted_reviews = sorted(decisions.keys())
    observe_cutoff  = _add_days(strategy_start, OBSERVE_ONLY_DAYS)

    # Per-ticker counters for keep-ratio thinning (resets per review window)
    ticker_counters: dict[tuple[str, datetime], int] = defaultdict(int)

    filtered: list[dict] = []
    for c in all_candidates:
        entry_ts = c["entry_ts"]

        # Observation phase: pass all candidates through unchanged
        if entry_ts < observe_cutoff:
            filtered.append(c)
            continue

        # Find the most recent review decision for this candidate
        applicable = None
        for rd in sorted_reviews:
            if rd <= entry_ts:
                applicable = rd
            else:
                break

        if applicable is None:
            filtered.append(c)
            continue

        ticker                        = c["ticker"]
        score, tier, ratio, _metrics  = decisions[applicable][ticker]

        # Apply circuit breaker
        cb = _get_cb_state_at(cb_state, entry_ts)
        if cb == "halt":
            continue   # circuit breaker: halt all new entries
        elif cb == "reduce":
            ratio = ratio * 0.5   # reduce all sizes by 50% (implemented as thin 50%)

        if ratio <= 0.0:
            continue  # tier F + eligible for removal

        # Apply keep-ratio thinning: keep 1 in every n candidates per ticker/window
        if ratio >= 1.0:
            filtered.append(c)
            continue

        key = (ticker, applicable)
        ticker_counters[key] += 1
        n = round(1.0 / ratio)
        if ticker_counters[key] % n == 0:
            filtered.append(c)

    return filtered, sorted_reviews


# ── Run engine ─────────────────────────────────────────────────────────────────

def _run(*, label: str, candidates: list[dict], macro_state: dict,
         tech_state: dict, use_dd_governor: bool = False) -> dict:
    kw = dict(**_BASE_KWARGS, **_OVERLAY_KWARGS)
    kw["extended_hours_proxy_state"] = macro_state
    kw["per_asset_technical_state"]  = tech_state
    kw["precomputed_candidates"]     = candidates
    if use_dd_governor:
        # W9+ circuit breaker thresholds
        kw["use_drawdown_governor"]    = True
        kw["drawdown_trigger_1_pct"]   = CB_REDUCE_DD_PCT
        kw["drawdown_exposure_mult_1"] = 1.5   # 3.0 * 0.5 effective
        kw["drawdown_trigger_2_pct"]   = CB_HALT_DD_PCT
        kw["drawdown_exposure_mult_2"] = 0.5   # near-halt
    result = generate_session_turtle_shared_account_report(**kw)
    s = dict(result["summary"])
    s["variant_label"] = label
    s["_trades"]       = result["trades"]
    return s


# ── Decision timeline printer ──────────────────────────────────────────────────

def _print_timeline(
    decisions:      dict[datetime, dict[str, tuple[int, str, float, dict]]],
    sorted_reviews: list[datetime],
    all_tickers:    list[str],
    strategy_start: datetime,
) -> None:
    """Print a compact monthly decision table with phase tags."""
    TIER_ICON = {"A": "A", "B": "B", "C": "c", "D": "d", "F": "X",
                 "INSUFFICIENT": "."}
    observe_cutoff     = _add_days(strategy_start, OBSERVE_ONLY_DAYS)
    full_rotation_cutoff = _add_days(strategy_start, FULL_ROTATION_DAYS)

    print("\n  FUND MANAGER DECISIONS")
    print("  Tier codes: A=full B=standard c=reduce50% d=reduce25% X=remove .=insufficient")
    print("  Phase tags: [OBS]=observe-only  [RED]=reduce-only  [ROT]=full-rotation")
    header = f"  {'Review':<12} {'Phase':<5}" + "".join(f"{tk[:7]:>8}" for tk in all_tickers)
    print(header)
    print("  " + "-" * (17 + 8 * len(all_tickers)))
    for rd in sorted_reviews:
        if rd < observe_cutoff:
            phase_tag = "OBS"
        elif rd < full_rotation_cutoff:
            phase_tag = "RED"
        else:
            phase_tag = "ROT"
        row = f"  {str(rd.date()):<12} {phase_tag:<5}"
        for tk in all_tickers:
            score, tier, ratio, _ = decisions[rd][tk]
            icon = TIER_ICON[tier]
            # Mark assets on probation / min-tenure lock with lowercase
            row += f"{icon:>8}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # ── Overlay state ────────────────────────────────────────────────────────
    print("\nLoading VIX daily closes + Crypto Fear & Greed...")
    vix_closes  = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg   = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg,
    )

    print("Building per-asset EMA200 technical state...")
    universe   = list(dict.fromkeys((t, s, ses) for t, s, ses in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe, lookback_years=5.0, warmup_days=300,
        ema_period=_EMA_PERIOD, adx_period=14,
    )
    daily_return_state = _build_daily_return_state(tech_state)

    # ── Candidates ───────────────────────────────────────────────────────────
    print("\nBuilding candidate trades...")
    all_candidates = build_session_turtle_shared_account_candidates(
        basket=_BASKET, initial_capital=1_000.0, lookback_years=_LOOKBACK_YRS,
        channel_period=_CHANNEL, base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP, directional_volume_risk_pct=_DIR_VOL_RISK,
        trend_fast_period=_TREND_FAST, trend_slow_period=_TREND_SLOW,
    )
    print(f"  {len(all_candidates)} total candidates")

    strategy_start = min(c["entry_ts"] for c in all_candidates)
    last_entry     = max(c["entry_ts"] for c in all_candidates)
    all_tickers    = sorted(set(c["ticker"] for c in all_candidates))

    print(f"  Strategy start   : {strategy_start.date()}")
    print(f"  Observe-only thru: {_add_days(strategy_start, OBSERVE_ONLY_DAYS).date()}  (W1-W4)")
    print(f"  Reduce-only thru : {_add_days(strategy_start, REDUCE_ONLY_DAYS).date()}   (W5-W8)")
    print(f"  Full rotation at : {_add_days(strategy_start, FULL_ROTATION_DAYS).date()} (W9+)")
    print(f"  Score lookback   : {SCORE_LOOKBACK_DAYS} days (13 weeks)")
    print(f"  Review frequency : {REVIEW_FREQ_DAYS} days (monthly)")
    print(f"  Probation period : {PROBATION_DAYS} days  (4 weeks, C-tier)")
    print(f"  Min live tenure  : {MIN_TENURE_DAYS} days  (8 weeks, removal gate)")
    print(f"  Circuit breakers : reduce at {CB_REDUCE_DD_PCT:.1f}% DD | "
          f"halt at {CB_HALT_DD_PCT:.1f}% DD | resume at {CB_RESUME_DD_PCT:.1f}% DD")
    print(f"  Assets           : {len(all_tickers)}  ({', '.join(all_tickers)})")

    # ── Step 1: Baseline ─────────────────────────────────────────────────────
    print("\nStep 1 — BASELINE (no moderation)...")
    baseline = _run(
        label="Baseline (no moderation)",
        candidates=all_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
    )
    print(f"  Return {baseline['total_return_pct']:.1f}%  CAGR {baseline['cagr_pct']:.1f}%  "
          f"MaxDD {baseline['max_realized_drawdown_pct']:.1f}%  "
          f"PF {baseline['profit_factor']:.2f}  WR {baseline['win_rate_pct']:.1f}%  "
          f"Trades {baseline['executed_trades']}")

    # ── Step 2: Build equity curve + circuit-breaker map ─────────────────────
    print("\nStep 2 — Building equity curve + circuit-breaker state map...")
    equity_curve = _build_equity_curve(baseline["_trades"])
    cb_state     = _build_circuit_breaker_state(equity_curve)
    cb_counts    = defaultdict(int)
    for s in cb_state.values():
        cb_counts[s] += 1
    total_exits  = len(cb_state)
    print(f"  Equity curve points: {len(equity_curve)}")
    if total_exits > 0:
        print(f"  CB state — normal: {cb_counts['normal']} exits "
              f"({cb_counts['normal']/total_exits*100:.0f}%)  "
              f"reduce: {cb_counts['reduce']} "
              f"({cb_counts['reduce']/total_exits*100:.0f}%)  "
              f"halt: {cb_counts['halt']} "
              f"({cb_counts['halt']/total_exits*100:.0f}%)")

    # ── Step 3: Build per-asset state machines ────────────────────────────────
    print("\nStep 3 — Building per-asset state machines (phase/probation/tenure)...")
    asset_start_ts = {
        tk: min(c["entry_ts"] for c in all_candidates if c["ticker"] == tk)
        for tk in all_tickers
    }
    asset_states = {tk: AssetState(tk, asset_start_ts[tk]) for tk in all_tickers}

    # ── Step 4: Build rolling decision map ───────────────────────────────────
    print("\nStep 4 — Building rolling fund-manager decisions (13-week lookback)...")
    decisions = _build_rolling_decisions(
        baseline_trades=baseline["_trades"],
        asset_states=asset_states,
        all_tickers=set(all_tickers),
        strategy_start=strategy_start,
        last_entry=last_entry,
        daily_return_state=daily_return_state,
    )
    sorted_reviews = sorted(decisions.keys())
    print(f"  {len(sorted_reviews)} monthly review checkpoints")

    # ── Step 5: Filter candidates ─────────────────────────────────────────────
    print("\nStep 5 — Filtering candidates (tier decisions + circuit breaker)...")
    fm_candidates, _ = _apply_fund_manager_filter(
        all_candidates=all_candidates,
        decisions=decisions,
        strategy_start=strategy_start,
        cb_state=cb_state,
    )
    removed = len(all_candidates) - len(fm_candidates)
    print(f"  {len(fm_candidates)} candidates kept  ({removed} filtered out  "
          f"{removed / len(all_candidates) * 100:.1f}%)")

    # ── Step 6: Moderated backtest — fund-manager filtered ────────────────────
    print("\nStep 6 — FUND MANAGER (phase gates, probation, diversification, CB prefilter)...")
    fm_result = _run(
        label="Fund Manager (phase gates, probation, diversification, CB)",
        candidates=fm_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        use_dd_governor=False,
    )
    print(f"  Return {fm_result['total_return_pct']:.1f}%  CAGR {fm_result['cagr_pct']:.1f}%  "
          f"MaxDD {fm_result['max_realized_drawdown_pct']:.1f}%  "
          f"PF {fm_result['profit_factor']:.2f}  WR {fm_result['win_rate_pct']:.1f}%  "
          f"Trades {fm_result['executed_trades']}")

    # Step 6b: legacy diagnostic to quantify the old double-circuit-breaker path.
    print("\nStep 6b — LEGACY diagnostic (prefilter CB + engine DD governor)...")
    fm_legacy_double_cb = _run(
        label="Legacy diagnostic (double CB)",
        candidates=fm_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        use_dd_governor=True,
    )
    print(f"  Return {fm_legacy_double_cb['total_return_pct']:.1f}%  CAGR {fm_legacy_double_cb['cagr_pct']:.1f}%  "
          f"MaxDD {fm_legacy_double_cb['max_realized_drawdown_pct']:.1f}%  "
          f"PF {fm_legacy_double_cb['profit_factor']:.2f}  WR {fm_legacy_double_cb['win_rate_pct']:.1f}%  "
          f"Trades {fm_legacy_double_cb['executed_trades']}")

    # ── Print decision timeline ───────────────────────────────────────────────
    _print_timeline(decisions, sorted_reviews, all_tickers, strategy_start)

    # ── Per-asset tier summary across all reviews ─────────────────────────────
    print(f"\n  ASSET TIER SUMMARY  (across {len(sorted_reviews)} monthly reviews)")
    print(f"  {'Ticker':<14} {'A':>4} {'B':>4} {'c':>4} {'d':>4} {'X':>4} {'Ins':>5}  "
          f"{'LastScore':>10}  Final")
    print("  " + "-" * 62)
    for ticker in all_tickers:
        counts   = defaultdict(int)
        last_sc  = -1
        last_tr  = "."
        for rd in sorted_reviews:
            sc, tier, ratio, _ = decisions[rd][ticker]
            display_tier = tier if tier != "INSUFFICIENT" else "."
            counts[display_tier] += 1
            last_sc, last_tr = sc, tier
        print(f"  {ticker:<14} {counts['A']:>4} {counts['B']:>4} "
              f"{counts['C']:>4} {counts['D']:>4} {counts['F']:>4} "
              f"{counts['.']:>5}  "
              f"{last_sc if last_sc >= 0 else '--':>10}  {last_tr}")

    # ── Circuit breaker events ────────────────────────────────────────────────
    print("\n  CIRCUIT BREAKER EVENTS (baseline equity proxy)")
    print(f"  Thresholds: reduce >= {CB_REDUCE_DD_PCT:.1f}%  halt >= {CB_HALT_DD_PCT:.1f}%  "
          f"resume < {CB_RESUME_DD_PCT:.1f}%")
    prev_s = "normal"
    for ts, s in sorted(cb_state.items()):
        if s != prev_s:
            eq_vals = [eq for t2, eq in equity_curve if t2 <= ts]
            if eq_vals:
                peak = max(eq_vals)
                cur  = eq_vals[-1]
                dd   = (peak - cur) / peak * 100.0
                print(f"  {str(ts.date())}  {prev_s.upper():>6} -> {s.upper():<6}  "
                      f"equity={cur:.2f}  DD={dd:.1f}%")
            prev_s = s

    # ── Comparison table ──────────────────────────────────────────────────────
    rows = [baseline, fm_result, fm_legacy_double_cb]
    cols = [
        ("Variant",     "variant_label",                "<54"),
        ("Return %",    "total_return_pct",             ">9"),
        ("CAGR %",      "cagr_pct",                     ">7"),
        ("Max DD %",    "max_realized_drawdown_pct",    ">8"),
        ("PF",          "profit_factor",                ">5"),
        ("WR %",        "win_rate_pct",                 ">6"),
        ("Trades",      "executed_trades",              ">7"),
    ]
    print("\n" + "=" * 112)
    print("  FUND MANAGER BACKTEST — RESULTS COMPARISON")
    print("=" * 112)
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(header); print(sep)
    for row in rows:
        parts = [
            f"{(f'{v:.2f}' if isinstance(v := row.get(k, '-'), float) else str(v)):{s}}"
            for _, k, s in cols
        ]
        print("  ".join(parts))

    b = baseline
    print("\n  Delta vs baseline:")
    for row in [fm_result, fm_legacy_double_cb]:
        print(f"  {row['variant_label']}")
        print(f"    Return {float(row['total_return_pct']) - float(b['total_return_pct']):+.1f}%  "
              f"CAGR {float(row['cagr_pct']) - float(b['cagr_pct']):+.1f}%  "
              f"MaxDD {float(row['max_realized_drawdown_pct']) - float(b['max_realized_drawdown_pct']):+.1f}%  "
              f"PF {float(row['profit_factor']) - float(b['profit_factor']):+.3f}  "
              f"WR {float(row['win_rate_pct']) - float(b['win_rate_pct']):+.1f}%  "
              f"Trades {int(row['executed_trades']) - int(b['executed_trades']):+d}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    for row in rows:
        safe = "".join(c if c.isalnum() or c == "_" else "_"
                       for c in row["variant_label"].lower().replace(" ", "_"))[:50]
        sub = OUTPUT_DIR / safe
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps({k: v for k, v in row.items() if k != "_trades"},
                       indent=2, default=str),
            encoding="utf-8",
        )
        trades = row.get("_trades", [])
        if trades:
            with (sub / "trades.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                w.writeheader(); w.writerows(trades)

    # Save rolling decisions for charting / audit
    decisions_out: dict[str, dict] = {}
    for rd, tick_d in decisions.items():
        decisions_out[str(rd.date())] = {
            tk: {"score": sc, "tier": ti, "keep_ratio": ratio,
                 "metrics": m}
            for tk, (sc, ti, ratio, m) in tick_d.items()
        }
    (OUTPUT_DIR / "fund_manager_decisions.json").write_text(
        json.dumps(decisions_out, indent=2, default=str),
        encoding="utf-8",
    )

    # Save circuit-breaker timeline
    cb_timeline = [
        {"ts": str(ts.date()), "state": s}
        for ts, s in sorted(cb_state.items())
        if s != "normal"
    ]
    (OUTPUT_DIR / "circuit_breaker_events.json").write_text(
        json.dumps(cb_timeline, indent=2),
        encoding="utf-8",
    )

    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  Fund manager decisions : {OUTPUT_DIR}/fund_manager_decisions.json")
    print(f"  Circuit breaker events : {OUTPUT_DIR}/circuit_breaker_events.json")

    print("\n  KEY PARAMETERS USED")
    print(f"    Ref max DD     : {REF_MAX_DD_PCT}%  (production backtest)")
    print(f"    CB reduce at   : {CB_REDUCE_DD_PCT:.2f}%  (60% of ref max DD)")
    print(f"    CB halt at     : {CB_HALT_DD_PCT:.2f}%  (80% of ref max DD)")
    print(f"    CB resume at   : {CB_RESUME_DD_PCT:.2f}%  (40% of ref max DD)")
    print(f"    Score lookback : {SCORE_LOOKBACK_DAYS}d  (13 weeks)")
    print(f"    Probation      : {PROBATION_DAYS}d  (4 weeks C-tier)")
    print(f"    Min tenure     : {MIN_TENURE_DAYS}d  (8 weeks before removal)")
    print(f"    Diversify hold : avg |corr| < {DIVERSIFICATION_CORR_THRESHOLD:.2f}")
    print(f"    Phase W1-W4    : observe only ({OBSERVE_ONLY_DAYS}d)")
    print(f"    Phase W5-W8    : reduce only ({REDUCE_ONLY_DAYS}d)")
    print(f"    Phase W9+      : full rotation")


if __name__ == "__main__":
    main()
