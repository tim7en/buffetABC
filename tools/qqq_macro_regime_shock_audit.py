"""Risk-regime and shock audit for forward QQQ returns.

This is a descriptive audit, not an investment recommendation. It uses:
- 63-trading-day smoothed macro inputs for regime classification
- standard market-condition proxies such as VIX, high-yield spreads, NFCI,
  yield-curve shape, equity trend, dollar, rates, and oil
- month-end observations to reduce overlapping daily forward-return noise

Raw externally fetched VIX / credit-spread / NFCI / 10Y-3M values are used only
to derive scores and flags; the output artifacts store derived classifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.qqq_macro_forward_return_analysis import _load_macro, _load_qqq, build_dataset, month_end_sample

DEFAULT_QQQ_PATH = ROOT / "cache" / "cache" / "cache" / "QQQ_daily.parquet"
DEFAULT_MACRO_PATH = ROOT / "cache" / "cache" / "macro_daily_1999.parquet"
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_macro_regime_shock_audit_3m_ma"


FRED_SERIES = {
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
    "nfci": "NFCI",
    "t10y3m": "T10Y3M",
}

SCORE_WEIGHTS = {
    "vix_score": 1.2,
    "hy_oas_score": 1.2,
    "nfci_score": 1.0,
    "curve_score": 0.9,
    "qqq_trend_score": 1.0,
    "qqq_momentum_score": 0.8,
    "rate_score": 0.7,
    "dxy_score": 0.6,
    "oil_score": 0.6,
}

SHOCK_COLUMNS = [
    "volatility_shock",
    "credit_spread_shock",
    "financial_conditions_shock",
    "equity_drawdown_shock",
    "oil_up_shock",
    "oil_down_shock",
    "dollar_up_shock",
    "rate_up_shock",
    "curve_inversion_shock",
]


def _fetch_fred_series(series_id: str, start_date: date, end_date: date) -> pd.Series:
    params = urllib.parse.urlencode(
        {
            "id": series_id,
            "cosd": start_date.isoformat(),
            "coed": end_date.isoformat(),
        }
    )
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")

    rows: list[tuple[pd.Timestamp, float]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        raw_date = (row.get("DATE") or row.get("observation_date") or "").strip()
        raw_value = (row.get(series_id) or "").strip()
        if not raw_date or not raw_value or raw_value == ".":
            continue
        try:
            rows.append((pd.Timestamp(raw_date), float(raw_value)))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No values returned from FRED for {series_id}")
    idx, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(idx), name=series_id).sort_index()


def _load_external_stress_series(qqq_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, str]]:
    start_date = qqq_index[0].date()
    end_date = qqq_index[-1].date()
    state: dict[str, pd.Series] = {}
    status: dict[str, str] = {}
    for label, series_id in FRED_SERIES.items():
        try:
            state[label] = _fetch_fred_series(series_id, start_date, end_date)
            status[label] = "loaded"
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            status[label] = f"missing: {type(exc).__name__}: {exc}"

    if not state:
        return pd.DataFrame(index=qqq_index), status

    raw = pd.DataFrame(state)
    aligned = raw.reindex(raw.index.union(qqq_index)).sort_index().ffill().reindex(qqq_index)
    return aligned, status


def _score_piecewise(series: pd.Series, rules: list[tuple[pd.Series, float]]) -> pd.Series:
    score = pd.Series(0.0, index=series.index)
    score[series.isna()] = np.nan
    for mask, value in rules:
        score.loc[mask & series.notna()] = value
    return score


def _add_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vix_score"] = _score_piecewise(
        out["vix_63d_ma"],
        [
            (out["vix_63d_ma"] <= 18.0, -1.0),
            ((out["vix_63d_ma"] > 18.0) & (out["vix_63d_ma"] <= 20.0), -0.5),
            ((out["vix_63d_ma"] >= 22.0) & (out["vix_63d_ma"] < 25.0), 0.35),
            ((out["vix_63d_ma"] >= 25.0) & (out["vix_63d_ma"] < 30.0), 0.75),
            (out["vix_63d_ma"] >= 30.0, 1.0),
        ],
    )
    out["hy_oas_score"] = _score_piecewise(
        out["hy_oas_63d_ma"],
        [
            (out["hy_oas_63d_ma"] <= 3.5, -1.0),
            ((out["hy_oas_63d_ma"] > 3.5) & (out["hy_oas_63d_ma"] <= 4.0), -0.5),
            ((out["hy_oas_63d_ma"] >= 4.5) & (out["hy_oas_63d_ma"] < 5.0), 0.5),
            ((out["hy_oas_63d_ma"] >= 5.0) & (out["hy_oas_63d_ma"] < 6.0), 0.75),
            (out["hy_oas_63d_ma"] >= 6.0, 1.0),
        ],
    )
    out["nfci_score"] = _score_piecewise(
        out["nfci_63d_ma"],
        [
            (out["nfci_63d_ma"] <= -0.5, -1.0),
            ((out["nfci_63d_ma"] > -0.5) & (out["nfci_63d_ma"] <= -0.25), -0.5),
            ((out["nfci_63d_ma"] >= 0.0) & (out["nfci_63d_ma"] < 0.5), 0.5),
            (out["nfci_63d_ma"] >= 0.5, 1.0),
        ],
    )
    out["curve_score"] = _score_piecewise(
        out["t10y3m_63d_ma"],
        [
            (out["t10y3m_63d_ma"] >= 1.0, -1.0),
            ((out["t10y3m_63d_ma"] >= 0.5) & (out["t10y3m_63d_ma"] < 1.0), -0.5),
            ((out["t10y3m_63d_ma"] > 0.0) & (out["t10y3m_63d_ma"] < 0.25), 0.5),
            (out["t10y3m_63d_ma"] <= 0.0, 1.0),
        ],
    )
    out["qqq_trend_score"] = _score_piecewise(
        out["qqq_vs_sma200"],
        [
            (out["qqq_vs_sma200"] >= 0.05, -1.0),
            ((out["qqq_vs_sma200"] >= 0.0) & (out["qqq_vs_sma200"] < 0.05), -0.5),
            ((out["qqq_vs_sma200"] < 0.0) & (out["qqq_vs_sma200"] > -0.05), 0.5),
            (out["qqq_vs_sma200"] <= -0.05, 1.0),
        ],
    )
    out["qqq_momentum_score"] = _score_piecewise(
        out["qqq_63d_return"],
        [
            (out["qqq_63d_return"] >= 0.05, -1.0),
            ((out["qqq_63d_return"] >= 0.0) & (out["qqq_63d_return"] < 0.05), -0.25),
            ((out["qqq_63d_return"] <= -0.05) & (out["qqq_63d_return"] > -0.10), 0.5),
            (out["qqq_63d_return"] <= -0.10, 1.0),
        ],
    )
    out["rate_score"] = _score_piecewise(
        out["us10y_3m_change_pp"],
        [
            (out["us10y_3m_change_pp"] >= 0.75, 1.0),
            ((out["us10y_3m_change_pp"] >= 0.35) & (out["us10y_3m_change_pp"] < 0.75), 0.5),
            (out["us10y_3m_change_pp"] <= -0.50, -0.5),
        ],
    )
    out["dxy_score"] = _score_piecewise(
        out["dxy_3m_return"],
        [
            (out["dxy_3m_return"] >= 0.05, 1.0),
            ((out["dxy_3m_return"] >= 0.025) & (out["dxy_3m_return"] < 0.05), 0.5),
            ((out["dxy_3m_return"] <= -0.025) & (out["dxy_3m_return"] > -0.05), -0.5),
            (out["dxy_3m_return"] <= -0.05, -1.0),
        ],
    )
    out["oil_score"] = _score_piecewise(
        out["wti_3m_return"],
        [
            (out["wti_3m_return"] >= 0.25, 1.0),
            ((out["wti_3m_return"] >= 0.15) & (out["wti_3m_return"] < 0.25), 0.5),
            (out["wti_3m_return"] <= -0.20, -0.25),
        ],
    )

    weighted = pd.Series(0.0, index=out.index)
    total_weight = pd.Series(0.0, index=out.index)
    for column, weight in SCORE_WEIGHTS.items():
        valid = out[column].notna()
        weighted.loc[valid] += out.loc[valid, column] * weight
        total_weight.loc[valid] += weight
    out["risk_score"] = weighted / total_weight.replace(0.0, np.nan)
    out["risk_score_0_100"] = (out["risk_score"] + 1.0) * 50.0
    out["risk_regime"] = np.select(
        [out["risk_score"] <= -0.25, out["risk_score"] >= 0.25],
        ["risk_on", "risk_off"],
        default="neutral",
    )
    out.loc[out["risk_score"].isna(), "risk_regime"] = "unknown"
    return out


def _add_shock_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["volatility_shock"] = (out["vix_raw"] >= 30.0) | (out["vix_raw"].diff(21) >= 8.0)
    out["credit_spread_shock"] = (out["hy_oas_raw"] >= 6.0) | (out["hy_oas_raw"].diff(63) >= 1.0)
    out["financial_conditions_shock"] = (out["nfci_raw"] >= 0.5) | (out["nfci_raw"].diff(63) >= 0.4)
    out["equity_drawdown_shock"] = (out["qqq_63d_return"] <= -0.10) | (out["qqq_vs_sma200"] <= -0.05)
    out["oil_up_shock"] = (out["wti_raw"].pct_change(63) >= 0.20) | (out["wti_raw"].pct_change(21) >= 0.10)
    out["oil_down_shock"] = out["wti_raw"].pct_change(63) <= -0.25
    out["dollar_up_shock"] = out["dxy_raw"].pct_change(63) >= 0.05
    out["rate_up_shock"] = (out["us10y_raw"].diff(63) >= 0.50) | (out["us2y_raw"].diff(63) >= 0.75)
    out["curve_inversion_shock"] = (out["t10y3m_raw"] < 0.0) | ((out["us10y_raw"] - out["us2y_raw"]) < 0.0)

    for column in SHOCK_COLUMNS:
        out[column] = out[column].fillna(False).astype(bool)
    out["shock_count"] = out[SHOCK_COLUMNS].sum(axis=1)
    out["major_shock"] = out["shock_count"] >= 2
    return out


def build_regime_dataset(
    qqq_close: pd.Series,
    macro: pd.DataFrame,
    external: pd.DataFrame,
    *,
    smoothing_days: int,
) -> pd.DataFrame:
    smoothed_macro = build_dataset(qqq_close, macro, macro_smoothing_days=smoothing_days)
    raw_macro = build_dataset(qqq_close, macro, macro_smoothing_days=0)
    out = smoothed_macro.copy()

    out["qqq_sma200"] = qqq_close.rolling(200, min_periods=200).mean()
    out["qqq_vs_sma200"] = qqq_close / out["qqq_sma200"] - 1.0
    out["qqq_63d_return"] = qqq_close.pct_change(63)
    out["dxy_raw"] = raw_macro["dxy_level"]
    out["us2y_raw"] = raw_macro["us2y_level"]
    out["us10y_raw"] = raw_macro["us10y_level"]
    out["wti_raw"] = raw_macro["wti_level"]

    for column in FRED_SERIES:
        if column in external:
            out[f"{column}_raw"] = external[column]
            out[f"{column}_63d_ma"] = external[column].rolling(smoothing_days, min_periods=smoothing_days).mean()
        else:
            out[f"{column}_raw"] = np.nan
            out[f"{column}_63d_ma"] = np.nan

    out = _add_component_scores(out)
    out = _add_shock_flags(out)
    return out


def _outcome_summary(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {}
    return {
        "n": int(len(clean)),
        "mean_cagr": float(clean.mean()),
        "median_cagr": float(clean.median()),
        "p25_cagr": float(clean.quantile(0.25)),
        "p75_cagr": float(clean.quantile(0.75)),
        "positive_rate": float((clean > 0.0).mean()),
    }


def summarize_regimes(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in [252, 504]:
        outcome = f"qqq_forward_{horizon}d_cagr"
        valid = sample[[outcome, "risk_regime", "major_shock"]].dropna()
        for regime, group in valid.groupby("risk_regime"):
            stats = _outcome_summary(group[outcome])
            if stats:
                rows.append({"horizon_days": horizon, "group_type": "risk_regime", "group": regime, **stats})
        for shock_state, group in valid.groupby("major_shock"):
            stats = _outcome_summary(group[outcome])
            if stats:
                rows.append(
                    {
                        "horizon_days": horizon,
                        "group_type": "major_shock",
                        "group": "major_shock" if bool(shock_state) else "no_major_shock",
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def summarize_shocks(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in [252, 504]:
        outcome = f"qqq_forward_{horizon}d_cagr"
        for column in SHOCK_COLUMNS:
            valid = sample[[outcome, column]].dropna()
            if valid.empty:
                continue
            on = valid.loc[valid[column].astype(bool), outcome]
            off = valid.loc[~valid[column].astype(bool), outcome]
            if len(on) < 12 or len(off) < 12:
                continue
            rows.append(
                {
                    "horizon_days": horizon,
                    "shock": column,
                    "shock_count": int(len(on)),
                    "no_shock_count": int(len(off)),
                    "shock_mean_cagr": float(on.mean()),
                    "no_shock_mean_cagr": float(off.mean()),
                    "shock_minus_no_shock_mean_cagr": float(on.mean() - off.mean()),
                    "shock_median_cagr": float(on.median()),
                    "no_shock_median_cagr": float(off.median()),
                    "shock_positive_rate": float((on > 0.0).mean()),
                    "no_shock_positive_rate": float((off > 0.0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["horizon_days", "shock_minus_no_shock_mean_cagr"])
    return out


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.1f}%"


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def current_snapshot(sample: pd.DataFrame, dataset: pd.DataFrame, external_status: dict[str, str]) -> dict[str, Any]:
    latest = dataset.dropna(subset=["qqq_adj_close"]).iloc[-1]
    latest_date = dataset.dropna(subset=["qqq_adj_close"]).index[-1]
    active_shocks = [column for column in SHOCK_COLUMNS if bool(latest.get(column, False))]
    regime_rows = {}
    active_shock_rows: dict[str, dict[str, Any]] = {}
    for horizon in [252, 504]:
        outcome = f"qqq_forward_{horizon}d_cagr"
        regime_history = sample[(sample["risk_regime"] == latest["risk_regime"]) & sample[outcome].notna()]
        shock_history = sample[(sample["major_shock"] == latest["major_shock"]) & sample[outcome].notna()]
        regime_rows[str(horizon)] = {
            "regime": _outcome_summary(regime_history[outcome]),
            "shock_state": _outcome_summary(shock_history[outcome]),
        }
        active_shock_rows[str(horizon)] = {}
        for column in active_shocks:
            valid = sample[[outcome, column]].dropna()
            if valid.empty:
                continue
            shock_values = valid.loc[valid[column].astype(bool), outcome]
            no_shock_values = valid.loc[~valid[column].astype(bool), outcome]
            shock_summary = _outcome_summary(shock_values)
            no_shock_summary = _outcome_summary(no_shock_values)
            if not shock_summary or not no_shock_summary:
                continue
            active_shock_rows[str(horizon)][column] = {
                "shock": shock_summary,
                "no_shock": no_shock_summary,
                "mean_delta": float(shock_summary["mean_cagr"] - no_shock_summary["mean_cagr"]),
            }
    return {
        "as_of": latest_date.date().isoformat(),
        "qqq_adj_close": float(latest["qqq_adj_close"]),
        "risk_regime": str(latest["risk_regime"]),
        "risk_score": None if pd.isna(latest["risk_score"]) else float(latest["risk_score"]),
        "risk_score_0_100": None if pd.isna(latest["risk_score_0_100"]) else float(latest["risk_score_0_100"]),
        "major_shock": bool(latest["major_shock"]),
        "active_shocks": active_shocks,
        "component_scores": {
            column: None if pd.isna(latest[column]) else float(latest[column])
            for column in SCORE_WEIGHTS
            if column in latest
        },
        "external_status": external_status,
        "forward_return_context": regime_rows,
        "active_shock_context": active_shock_rows,
    }


def write_report(
    out_dir: Path,
    sample: pd.DataFrame,
    regime_summary: pd.DataFrame,
    shock_summary: pd.DataFrame,
    snapshot: dict[str, Any],
    *,
    smoothing_days: int,
) -> None:
    latest_252 = snapshot["forward_return_context"].get("252", {})
    latest_504 = snapshot["forward_return_context"].get("504", {})
    active_shock_context = snapshot.get("active_shock_context", {})
    lines = [
        "# QQQ Macro Regime and Shock Audit",
        "",
        "This is a descriptive regime audit, not a forecast or investment recommendation.",
        "",
        "## Method",
        "",
        f"- Macro smoothing: `{smoothing_days}` trading-day trailing moving average.",
        "- Outcomes: QQQ adjusted-close forward CAGRs over 252 and 504 trading days.",
        "- Regime score direction: negative is risk-on, positive is risk-off.",
        "- Industry-standard proxies: VIX, high-yield OAS, NFCI, 10Y-3M curve, QQQ 200D trend, QQQ 3M momentum, 10Y rate change, DXY, and WTI.",
        "- Shock flags use unsmoothed rapid changes or threshold breaks; regime classification uses smoothed inputs.",
        "",
        "## Current Snapshot",
        "",
        f"- As of: `{snapshot['as_of']}`",
        f"- QQQ adjusted close: `{_fmt_float(snapshot['qqq_adj_close'], 2)}`",
        f"- Regime: `{snapshot['risk_regime']}`",
        f"- Risk score: `{_fmt_float(snapshot['risk_score'], 3)}` on a -1 risk-on to +1 risk-off scale",
        f"- Risk score 0-100: `{_fmt_float(snapshot['risk_score_0_100'], 1)}`",
        f"- Major shock active: `{snapshot['major_shock']}`",
        f"- Active shocks: `{', '.join(snapshot['active_shocks']) if snapshot['active_shocks'] else 'none'}`",
        "",
        "## Current Forward-Return Context",
        "",
        "| Horizon | Context | Count | Mean CAGR | Median CAGR | P25 | P75 | Positive Rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, context in [("252", latest_252), ("504", latest_504)]:
        for label, stats in context.items():
            if not stats:
                continue
            lines.append(
                "| {horizon} | {label} | {n} | {mean} | {median} | {p25} | {p75} | {positive} |".format(
                    horizon=horizon,
                    label=label,
                    n=stats["n"],
                    mean=_fmt_pct(stats["mean_cagr"]),
                    median=_fmt_pct(stats["median_cagr"]),
                    p25=_fmt_pct(stats["p25_cagr"]),
                    p75=_fmt_pct(stats["p75_cagr"]),
                    positive=_fmt_pct(stats["positive_rate"]),
                )
            )

    if any(active_shock_context.get(horizon) for horizon in ["252", "504"]):
        lines.extend(
            [
                "",
                "## Current Active Shock Context",
                "",
                "| Horizon | Active Shock | Shock Count | Shock Avg | No-Shock Avg | Difference | Shock Positive Rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon in ["252", "504"]:
            for shock, context in active_shock_context.get(horizon, {}).items():
                shock_stats = context["shock"]
                no_shock_stats = context["no_shock"]
                lines.append(
                    "| {horizon} | {shock} | {count} | {shock_avg} | {off_avg} | {diff} | {positive} |".format(
                        horizon=horizon,
                        shock=shock,
                        count=shock_stats["n"],
                        shock_avg=_fmt_pct(shock_stats["mean_cagr"]),
                        off_avg=_fmt_pct(no_shock_stats["mean_cagr"]),
                        diff=_fmt_pct(context["mean_delta"]),
                        positive=_fmt_pct(shock_stats["positive_rate"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Regime Forward Returns",
            "",
            "| Horizon | Group Type | Group | Count | Mean CAGR | Median CAGR | P25 | P75 | Positive Rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in regime_summary.iterrows():
        lines.append(
            "| {horizon} | {group_type} | {group} | {n} | {mean} | {median} | {p25} | {p75} | {positive} |".format(
                horizon=int(row["horizon_days"]),
                group_type=row["group_type"],
                group=row["group"],
                n=int(row["n"]),
                mean=_fmt_pct(row["mean_cagr"]),
                median=_fmt_pct(row["median_cagr"]),
                p25=_fmt_pct(row["p25_cagr"]),
                p75=_fmt_pct(row["p75_cagr"]),
                positive=_fmt_pct(row["positive_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Shock Forward Returns",
            "",
            "| Horizon | Shock | Shock Count | Shock Avg | No-Shock Avg | Difference | Shock Median |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not shock_summary.empty:
        for _, row in shock_summary.groupby("horizon_days", group_keys=False).head(10).iterrows():
            lines.append(
                "| {horizon} | {shock} | {count} | {shock_avg} | {off_avg} | {diff} | {median} |".format(
                    horizon=int(row["horizon_days"]),
                    shock=row["shock"],
                    count=int(row["shock_count"]),
                    shock_avg=_fmt_pct(row["shock_mean_cagr"]),
                    off_avg=_fmt_pct(row["no_shock_mean_cagr"]),
                    diff=_fmt_pct(row["shock_minus_no_shock_mean_cagr"]),
                    median=_fmt_pct(row["shock_median_cagr"]),
                )
            )

    recent_shocks = sample[sample["major_shock"]].tail(20).copy()
    lines.extend(
        [
            "",
            "## Recent Major Shock Month-Ends",
            "",
            "| Date | Regime | Shock Count | Active Shocks | 1Y CAGR | 2Y CAGR |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for day, row in recent_shocks.iterrows():
        active = [column for column in SHOCK_COLUMNS if bool(row.get(column, False))]
        lines.append(
            "| {date} | {regime} | {count} | {active} | {one} | {two} |".format(
                date=day.date(),
                regime=row["risk_regime"],
                count=int(row["shock_count"]),
                active=", ".join(active),
                one=_fmt_pct(row.get("qqq_forward_252d_cagr")),
                two=_fmt_pct(row.get("qqq_forward_504d_cagr")),
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `derived_daily_regime_dataset.csv`: derived regime features, scores, shocks, and forward returns",
            "- `month_end_regime_sample.csv`: month-end sample used for historical return tracking",
            "- `regime_forward_returns.csv`: forward returns by risk-on/neutral/risk-off and major-shock states",
            "- `shock_forward_returns.csv`: forward returns after specific shock flags",
            "- `recent_shocks.csv`: recent month-end rows with active major shock flags",
            "- `current_snapshot.json`: latest regime and historical forward-return context",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify QQQ macro risk regimes and shock forward returns.")
    parser.add_argument("--qqq-path", type=Path, default=DEFAULT_QQQ_PATH)
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoothing-days", type=int, default=63)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qqq_close = _load_qqq(args.qqq_path)
    macro = _load_macro(args.macro_path, qqq_close.index)
    external, external_status = _load_external_stress_series(qqq_close.index)
    dataset = build_regime_dataset(qqq_close, macro, external, smoothing_days=args.smoothing_days)
    sample = month_end_sample(dataset)

    regime_summary = summarize_regimes(sample)
    shock_summary = summarize_shocks(sample)
    recent_shocks = sample[sample["major_shock"]].tail(100)
    snapshot = current_snapshot(sample, dataset, external_status)

    derived_columns = [
        "qqq_adj_close",
        "qqq_vs_sma200",
        "qqq_63d_return",
        "dxy_3m_return",
        "dxy_12m_return",
        "us2y_level",
        "us10y_level",
        "us10y_3m_change_pp",
        "curve_10y2y_level",
        "wti_3m_return",
        "wti_12m_return",
        "vix_score",
        "hy_oas_score",
        "nfci_score",
        "curve_score",
        "qqq_trend_score",
        "qqq_momentum_score",
        "rate_score",
        "dxy_score",
        "oil_score",
        "risk_score",
        "risk_score_0_100",
        "risk_regime",
        *SHOCK_COLUMNS,
        "shock_count",
        "major_shock",
        "qqq_forward_252d_cagr",
        "qqq_forward_252d_end_date",
        "qqq_forward_504d_cagr",
        "qqq_forward_504d_end_date",
    ]
    dataset[derived_columns].to_csv(args.out_dir / "derived_daily_regime_dataset.csv", index_label="date")
    sample[derived_columns].to_csv(args.out_dir / "month_end_regime_sample.csv", index_label="date")
    regime_summary.to_csv(args.out_dir / "regime_forward_returns.csv", index=False)
    shock_summary.to_csv(args.out_dir / "shock_forward_returns.csv", index=False)
    recent_shocks[derived_columns].to_csv(args.out_dir / "recent_shocks.csv", index_label="date")
    (args.out_dir / "current_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.out_dir,
        sample,
        regime_summary,
        shock_summary,
        snapshot,
        smoothing_days=args.smoothing_days,
    )

    print(f"Saved QQQ macro regime/shock audit under: {args.out_dir}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
