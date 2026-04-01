"""
Macro and risk audit for the latest current-baseline strategy report.

What it does
------------
1. Reconstructs strategy performance over time from the saved trade ledger.
2. Pulls official macro drivers from FRED for the same period.
3. Builds a heuristic daily risk model with low / medium / high buckets.
4. Measures sensitivity to price expansion and the persistence of the current regime.

Default input:
  reports/strategy_health_audit/bigtech_energy_base_uncapped_equity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

from edgar.services.intraday_strategy import _ema
from tools.session_turtle_core_x2.run_selection_edge_audit import (
    BORDER,
    DARK_BG,
    MUTED,
    PANEL_BG,
    TEXT,
    _aggregate_trade_edge,
    _asset_bucket,
    _build_daily_close_state,
    _build_daily_return_state,
    _current_baseline_universe,
    _cumulative_trade_edge_rows,
    _dt,
    _ensure_tiingo_cache_symbols,
    _load_report,
    _save_json,
    _savefig,
    _sector_group,
    _sector_return_state,
    _trade_edge_rows,
    _write_csv,
)


REPORT_DIR = Path("reports/strategy_health_audit/bigtech_energy_base_uncapped_equity")
OUTPUT_DIR = Path("reports/macro_risk_audit/current_baseline")
CHART_DIR = OUTPUT_DIR / "charts"

FRED_SERIES = {
    "dff": {
        "series_id": "DFF",
        "label": "Fed Funds Effective Rate",
        "units": "Percent",
    },
    "dgs2": {
        "series_id": "DGS2",
        "label": "2Y Treasury Yield",
        "units": "Percent",
    },
    "dgs10": {
        "series_id": "DGS10",
        "label": "10Y Treasury Yield",
        "units": "Percent",
    },
    "dtwexbgs": {
        "series_id": "DTWEXBGS",
        "label": "Broad Dollar Index",
        "units": "Index Jan 2006=100",
    },
    "vixcls": {
        "series_id": "VIXCLS",
        "label": "VIX",
        "units": "Index",
    },
}

RISK_COLORS = {
    "low": "#22c55e",
    "medium": "#f59e0b",
    "high": "#ef4444",
}
CYAN = "#22d3ee"
GREEN = "#22c55e"
RED = "#ef4444"


def _build_equity_snapshot(summary: dict, trades: list[dict]) -> dict:
    initial = float(summary["initial_capital"])
    sorted_trades = sorted(trades, key=lambda trade: _dt(trade["exit_ts"]))
    dates = [_dt(summary["start_date"])]
    equities = [initial]
    for trade in sorted_trades:
        if trade.get("equity_after_exit") is None:
            continue
        dates.append(_dt(trade["exit_ts"]))
        equities.append(float(trade["equity_after_exit"]))

    peak = initial
    drawdowns_pct: list[float] = []
    for equity in equities:
        peak = max(peak, equity)
        drawdowns_pct.append((peak - equity) / peak * 100.0 if peak > 0 else 0.0)

    return {
        "dates": [dt.isoformat() for dt in dates],
        "equities": [round(value, 4) for value in equities],
        "drawdowns_pct": [round(value, 4) for value in drawdowns_pct],
    }


def _rolling_window_stats(trades: list[dict], *, window: int = 25) -> dict[str, list[float | None]]:
    sorted_trades = sorted(trades, key=lambda trade: _dt(trade["exit_ts"]))
    dates: list[str] = []
    expectancy: list[float | None] = []
    profit_factor: list[float | None] = []
    for idx, trade in enumerate(sorted_trades):
        dates.append(str(trade["exit_ts"]))
        if idx + 1 < window:
            expectancy.append(None)
            profit_factor.append(None)
            continue
        window_trades = sorted_trades[idx + 1 - window : idx + 1]
        pnls = [float(t.get("net_pnl", 0.0) or 0.0) for t in window_trades]
        expectancy.append(round(sum(pnls) / len(pnls), 4))
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = None if gross_loss <= 1e-12 else gross_profit / gross_loss
        profit_factor.append(round(pf, 4) if pf is not None and math.isfinite(pf) else None)
    return {"dates": dates, "expectancy": expectancy, "profit_factor": profit_factor}


def _fetch_fred_series(
    *,
    series_id: str,
    start_date: date,
    end_date: date,
) -> dict[date, float]:
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

    out: dict[date, float] = {}
    reader = csv.DictReader(text.splitlines())
    value_field = series_id
    for row in reader:
        raw_date = (row.get("DATE") or row.get("observation_date") or "").strip()
        raw_value = (row.get(value_field) or "").strip()
        if not raw_date or not raw_value or raw_value == ".":
            continue
        try:
            out[date.fromisoformat(raw_date)] = float(raw_value)
        except ValueError:
            continue
    if not out:
        raise RuntimeError(f"No data returned from FRED for series {series_id}")
    return dict(sorted(out.items()))


def _last_value_on_or_before(series: dict[date, float], target_day: date) -> float | None:
    chosen = None
    for day, value in series.items():
        if day <= target_day:
            chosen = value
        else:
            break
    return chosen


def _series_change(series: dict[date, float], target_day: date, lookback_days: int) -> float | None:
    current = _last_value_on_or_before(series, target_day)
    prior = _last_value_on_or_before(series, target_day - timedelta(days=lookback_days))
    if current is None or prior is None:
        return None
    return current - prior


def _build_index_from_return_state(
    return_state: dict[date, float],
    *,
    start_value: float = 100.0,
) -> dict[date, float]:
    value = float(start_value)
    out: dict[date, float] = {}
    for day, ret in sorted(return_state.items()):
        value *= 1.0 + float(ret)
        out[day] = value
    return out


def _rolling_pairwise_corr_state(
    daily_return_state: dict[str, dict[date, float]],
    *,
    window: int = 63,
) -> dict[date, float]:
    all_days = sorted({day for rows in daily_return_state.values() for day in rows})
    out: dict[date, float] = {}
    tickers = sorted(daily_return_state)
    for idx in range(window - 1, len(all_days)):
        window_days = all_days[idx + 1 - window : idx + 1]
        corrs: list[float] = []
        for left_idx, left in enumerate(tickers):
            for right in tickers[left_idx + 1 :]:
                common = [
                    day for day in window_days
                    if day in daily_return_state[left] and day in daily_return_state[right]
                ]
                if len(common) < max(20, window // 2):
                    continue
                left_vals = [daily_return_state[left][day] for day in common]
                right_vals = [daily_return_state[right][day] for day in common]
                mean_left = sum(left_vals) / len(left_vals)
                mean_right = sum(right_vals) / len(right_vals)
                cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_vals, right_vals))
                var_left = sum((a - mean_left) ** 2 for a in left_vals)
                var_right = sum((b - mean_right) ** 2 for b in right_vals)
                if var_left <= 1e-12 or var_right <= 1e-12:
                    continue
                corrs.append(cov / math.sqrt(var_left * var_right))
        if corrs:
            out[all_days[idx]] = sum(corrs) / len(corrs)
    return out


def _forward_fill_daily(
    days: list[date],
    source: dict[date, float | None],
) -> dict[date, float | None]:
    out: dict[date, float | None] = {}
    last_value: float | None = None
    for day in days:
        if day in source and source[day] is not None:
            last_value = source[day]
        out[day] = last_value
    return out


def _build_daily_strategy_state(
    summary: dict,
    trades: list[dict],
    analysis_days: list[date],
) -> dict[str, dict[date, float | None]]:
    equity_snapshot = _build_equity_snapshot(summary, trades)

    equity_by_day: dict[date, float] = {}
    dd_by_day: dict[date, float] = {}
    for dt_str, equity, dd in zip(
        equity_snapshot["dates"],
        equity_snapshot["equities"],
        equity_snapshot["drawdowns_pct"],
    ):
        day = _dt(dt_str).date()
        equity_by_day[day] = float(equity)
        dd_by_day[day] = float(dd)

    rolling = _rolling_window_stats(trades, window=25)
    pf_by_day: dict[date, float | None] = {}
    expectancy_by_day: dict[date, float | None] = {}
    for dt_str, pf, expectancy in zip(
        rolling["dates"],
        rolling["profit_factor"],
        rolling["expectancy"],
    ):
        day = _dt(dt_str).date()
        pf_by_day[day] = pf
        expectancy_by_day[day] = expectancy

    return {
        "equity": _forward_fill_daily(analysis_days, equity_by_day),
        "drawdown_pct": _forward_fill_daily(analysis_days, dd_by_day),
        "rolling_pf": _forward_fill_daily(analysis_days, pf_by_day),
        "rolling_expectancy": _forward_fill_daily(analysis_days, expectancy_by_day),
    }


def _build_market_expansion_state(
    market_return_state: dict[date, float],
    analysis_days: list[date],
) -> dict[str, dict[date, float | None]]:
    market_index = _build_index_from_return_state(market_return_state, start_value=100.0)
    ordered_days = sorted(market_index)
    ordered_values = [market_index[day] for day in ordered_days]
    ema_200 = _ema(ordered_values, 200)
    expansion_pct: dict[date, float | None] = {}
    rolling_63d_return: dict[date, float | None] = {}

    for idx, day in enumerate(ordered_days):
        ema_value = ema_200[idx]
        expansion_pct[day] = ((ordered_values[idx] / ema_value) - 1.0) * 100.0 if ema_value else None
        if idx + 1 >= 63:
            trailing = ordered_values[idx] / ordered_values[idx + 1 - 63] - 1.0
            rolling_63d_return[day] = trailing * 100.0
        else:
            rolling_63d_return[day] = None

    return {
        "market_index": _forward_fill_daily(analysis_days, market_index),
        "market_expansion_pct": _forward_fill_daily(analysis_days, expansion_pct),
        "market_63d_return_pct": _forward_fill_daily(analysis_days, rolling_63d_return),
    }


def _quantile_edges(values: list[float], buckets: int) -> list[float]:
    ordered = sorted(values)
    edges = []
    for idx in range(1, buckets):
        pos = int(round((len(ordered) - 1) * idx / buckets))
        edges.append(ordered[pos])
    return edges


def _bucket_label(value: float, edges: list[float]) -> str:
    for idx, edge in enumerate(edges, start=1):
        if value <= edge:
            return f"Q{idx}"
    return f"Q{len(edges) + 1}"


def _build_expansion_sensitivity(
    *,
    analysis_days: list[date],
    market_return_state: dict[date, float],
    expansion_state: dict[date, float | None],
    trade_edge_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    valid_days = [day for day in analysis_days if expansion_state.get(day) is not None]
    valid_values = [float(expansion_state[day]) for day in valid_days]
    edges = _quantile_edges(valid_values, 5)

    next_21d_rows = []
    for idx, day in enumerate(valid_days):
        if idx + 21 >= len(valid_days):
            continue
        future_days = valid_days[idx + 1 : idx + 22]
        future_returns = [market_return_state.get(future_day) for future_day in future_days if future_day in market_return_state]
        if len(future_returns) < 10:
            continue
        total = 1.0
        for ret in future_returns:
            total *= 1.0 + float(ret)
        expansion_value = float(expansion_state[day])
        next_21d_rows.append(
            {
                "date": day.isoformat(),
                "expansion_pct": round(expansion_value, 4),
                "expansion_bucket": _bucket_label(expansion_value, edges),
                "forward_21d_market_return_pct": round((total - 1.0) * 100.0, 4),
            }
        )

    daily_summary: list[dict] = []
    grouped_daily: dict[str, list[dict]] = defaultdict(list)
    for row in next_21d_rows:
        grouped_daily[str(row["expansion_bucket"])].append(row)
    for bucket in sorted(grouped_daily):
        rows = grouped_daily[bucket]
        daily_summary.append(
            {
                "expansion_bucket": bucket,
                "observations": len(rows),
                "avg_expansion_pct": round(sum(float(r["expansion_pct"]) for r in rows) / len(rows), 4),
                "median_expansion_pct": round(statistics.median(float(r["expansion_pct"]) for r in rows), 4),
                "avg_forward_21d_market_return_pct": round(
                    sum(float(r["forward_21d_market_return_pct"]) for r in rows) / len(rows),
                    4,
                ),
                "median_forward_21d_market_return_pct": round(
                    statistics.median(float(r["forward_21d_market_return_pct"]) for r in rows),
                    4,
                ),
            }
        )

    trade_rows: list[dict] = []
    grouped_trade: dict[str, list[dict]] = defaultdict(list)
    for row in trade_edge_rows:
        entry_day = _dt(row["entry_ts"]).date()
        expansion_value = expansion_state.get(entry_day)
        if expansion_value is None:
            continue
        bucket = _bucket_label(float(expansion_value), edges)
        tagged = dict(row)
        tagged["entry_expansion_pct"] = round(float(expansion_value), 4)
        tagged["entry_expansion_bucket"] = bucket
        trade_rows.append(tagged)
        grouped_trade[bucket].append(tagged)

    trade_summary: list[dict] = []
    for bucket in sorted(grouped_trade):
        rows = grouped_trade[bucket]
        trade_summary.append(
            {
                "expansion_bucket": bucket,
                "trades": len(rows),
                "avg_strategy_return_pct": round(
                    sum(float(r["strategy_return_pct"]) for r in rows) / len(rows),
                    4,
                ),
                "avg_edge_vs_sector_pct": round(
                    sum(float(r["edge_vs_sector_pct"]) for r in rows) / len(rows),
                    4,
                ),
                "avg_edge_vs_market_pct": round(
                    sum(float(r["edge_vs_market_pct"]) for r in rows) / len(rows),
                    4,
                ),
                "net_pnl": round(sum(float(r["net_pnl"]) for r in rows), 4),
                "positive_edge_vs_sector_rate_pct": round(
                    sum(1 for r in rows if float(r["edge_vs_sector_pct"]) > 0) / len(rows) * 100.0,
                    2,
                ),
            }
        )

    current_day = valid_days[-1]
    current_expansion = float(expansion_state[current_day])
    current_bucket = _bucket_label(current_expansion, edges)
    current_daily_bucket = next((row for row in daily_summary if row["expansion_bucket"] == current_bucket), None)

    meta = {
        "current_date": current_day.isoformat(),
        "current_expansion_pct": round(current_expansion, 4),
        "current_expansion_bucket": current_bucket,
        "current_bucket_forward_21d_market_stats": current_daily_bucket,
    }
    return daily_summary, trade_summary, meta


def _build_monthly_pnl_rows(trades: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    for trade in trades:
        dt = _dt(trade["exit_ts"])
        key = (dt.year, dt.month)
        grouped[key]["pnl"] += float(trade.get("net_pnl", 0.0) or 0.0)
        grouped[key]["trades"] += 1

    rows = []
    for (year, month), payload in sorted(grouped.items()):
        rows.append(
            {
                "year": year,
                "month": month,
                "pnl": round(payload["pnl"], 4),
                "trades": int(payload["trades"]),
            }
        )
    return rows


def _build_sector_cumulative_rows(trades: list[dict]) -> list[dict]:
    cumulative: dict[str, float] = defaultdict(float)
    rows = []
    ordered = sorted(trades, key=lambda trade: _dt(trade["exit_ts"]))
    sector_labels = sorted({_sector_group(str(trade["ticker"]))[0] for trade in ordered})
    for trade in ordered:
        sector = _sector_group(str(trade["ticker"]))[0]
        cumulative[sector] += float(trade.get("net_pnl", 0.0) or 0.0)
        row = {"exit_ts": trade["exit_ts"]}
        for label in sector_labels:
            row[f"{label}_cumulative_pnl"] = round(cumulative[label], 4)
        rows.append(row)
    return rows


def _build_macro_rows(
    *,
    analysis_days: list[date],
    macro_series: dict[str, dict[date, float]],
) -> list[dict]:
    rows = []
    for day in analysis_days:
        dff = _last_value_on_or_before(macro_series["dff"], day)
        dgs2 = _last_value_on_or_before(macro_series["dgs2"], day)
        dgs10 = _last_value_on_or_before(macro_series["dgs10"], day)
        dtwexbgs = _last_value_on_or_before(macro_series["dtwexbgs"], day)
        vix = _last_value_on_or_before(macro_series["vixcls"], day)
        rows.append(
            {
                "date": day.isoformat(),
                "dff": round(dff, 4) if dff is not None else None,
                "dgs2": round(dgs2, 4) if dgs2 is not None else None,
                "dgs10": round(dgs10, 4) if dgs10 is not None else None,
                "curve_10y_2y": round(dgs10 - dgs2, 4) if dgs10 is not None and dgs2 is not None else None,
                "dtwexbgs": round(dtwexbgs, 4) if dtwexbgs is not None else None,
                "vixcls": round(vix, 4) if vix is not None else None,
            }
        )
    return rows


def _build_risk_model_rows(
    *,
    analysis_days: list[date],
    strategy_state: dict[str, dict[date, float | None]],
    expansion_state: dict[str, dict[date, float | None]],
    pairwise_corr_state: dict[date, float],
    macro_series: dict[str, dict[date, float]],
) -> list[dict]:
    out = []
    market_index_rows = {
        day: value for day, value in expansion_state["market_index"].items() if value is not None
    }
    market_index_days = sorted(market_index_rows)
    market_index_values = [market_index_rows[day] for day in market_index_days]
    market_ema_200 = _ema(market_index_values, 200)
    market_ema_map = {
        day: market_ema_200[idx]
        for idx, day in enumerate(market_index_days)
        if market_ema_200[idx] is not None
    }

    dollar_rows = {
        day: value for day, value in macro_series["dtwexbgs"].items() if value is not None
    }
    dollar_days = sorted(dollar_rows)
    dollar_values = [dollar_rows[day] for day in dollar_days]
    dollar_ema_200 = _ema(dollar_values, 200)
    dollar_ema_map = {
        day: dollar_ema_200[idx]
        for idx, day in enumerate(dollar_days)
        if dollar_ema_200[idx] is not None
    }

    for day in analysis_days:
        drawdown = strategy_state["drawdown_pct"].get(day)
        rolling_pf = strategy_state["rolling_pf"].get(day)
        pairwise_corr = pairwise_corr_state.get(day)
        expansion_pct = expansion_state["market_expansion_pct"].get(day)
        return_63d = expansion_state["market_63d_return_pct"].get(day)

        dff = _last_value_on_or_before(macro_series["dff"], day)
        dff_63d_change = _series_change(macro_series["dff"], day, 63)
        dgs2 = _last_value_on_or_before(macro_series["dgs2"], day)
        dgs10 = _last_value_on_or_before(macro_series["dgs10"], day)
        curve = (dgs10 - dgs2) if dgs10 is not None and dgs2 is not None else None
        dollar = _last_value_on_or_before(macro_series["dtwexbgs"], day)
        dollar_63d_change = _series_change(macro_series["dtwexbgs"], day, 63)
        dollar_ema = _last_value_on_or_before(dollar_ema_map, day)
        vix = _last_value_on_or_before(macro_series["vixcls"], day)

        score_drawdown = (
            1.0 if drawdown is not None and drawdown >= 30.0 else
            0.5 if drawdown is not None and drawdown >= 20.0 else
            0.25 if drawdown is not None and drawdown >= 10.0 else
            0.0
        )
        score_pf = (
            1.0 if rolling_pf is not None and rolling_pf < 1.0 else
            0.5 if rolling_pf is not None and rolling_pf < 1.5 else
            0.0
        )
        score_corr = (
            1.0 if pairwise_corr is not None and pairwise_corr >= 0.28 else
            0.5 if pairwise_corr is not None and pairwise_corr >= 0.22 else
            0.0
        )
        score_rates = (
            1.0 if (
                dff is not None and curve is not None and dff >= 4.0 and curve <= 0.0
            ) or (
                dff_63d_change is not None and dff_63d_change >= 0.50
            ) else
            0.5 if (
                dff is not None and dff >= 3.5
            ) or (
                curve is not None and curve <= 0.5
            ) else
            0.0
        )
        score_dollar = (
            1.0 if (
                dollar is not None and dollar_ema is not None and dollar >= dollar_ema * 1.03
            ) and (
                dollar_63d_change is not None and dollar_63d_change >= 2.0
            ) else
            0.5 if (
                dollar is not None and dollar_ema is not None and dollar >= dollar_ema * 1.01
            ) or (
                dollar_63d_change is not None and dollar_63d_change >= 0.5
            ) else
            0.0
        )
        score_vix = (
            1.0 if vix is not None and vix >= 25.0 else
            0.5 if vix is not None and vix >= 18.0 else
            0.0
        )
        score_expansion = (
            1.0 if (
                expansion_pct is not None and expansion_pct >= 15.0
            ) or (
                return_63d is not None and return_63d >= 20.0
            ) else
            0.5 if (
                expansion_pct is not None and expansion_pct >= 8.0
            ) or (
                return_63d is not None and return_63d >= 10.0
            ) else
            0.0
        )

        risk_score = (
            0.20 * score_drawdown +
            0.15 * score_pf +
            0.15 * score_corr +
            0.15 * score_rates +
            0.15 * score_dollar +
            0.10 * score_vix +
            0.10 * score_expansion
        ) * 100.0

        risk_bucket = (
            "high" if risk_score >= 65.0 else
            "medium" if risk_score >= 35.0 else
            "low"
        )

        out.append(
            {
                "date": day.isoformat(),
                "drawdown_pct": round(drawdown, 4) if drawdown is not None else None,
                "rolling_pf": round(rolling_pf, 4) if rolling_pf is not None else None,
                "avg_pairwise_corr_63d": round(pairwise_corr, 4) if pairwise_corr is not None else None,
                "market_expansion_pct": round(expansion_pct, 4) if expansion_pct is not None else None,
                "market_63d_return_pct": round(return_63d, 4) if return_63d is not None else None,
                "dff": round(dff, 4) if dff is not None else None,
                "dff_63d_change": round(dff_63d_change, 4) if dff_63d_change is not None else None,
                "dgs2": round(dgs2, 4) if dgs2 is not None else None,
                "dgs10": round(dgs10, 4) if dgs10 is not None else None,
                "curve_10y_2y": round(curve, 4) if curve is not None else None,
                "dtwexbgs": round(dollar, 4) if dollar is not None else None,
                "dtwexbgs_63d_change": round(dollar_63d_change, 4) if dollar_63d_change is not None else None,
                "vixcls": round(vix, 4) if vix is not None else None,
                "score_drawdown": round(score_drawdown, 2),
                "score_pf": round(score_pf, 2),
                "score_corr": round(score_corr, 2),
                "score_rates": round(score_rates, 2),
                "score_dollar": round(score_dollar, 2),
                "score_vix": round(score_vix, 2),
                "score_expansion": round(score_expansion, 2),
                "risk_score": round(risk_score, 2),
                "risk_bucket": risk_bucket,
            }
        )
    return out


def _bucket_run_stats(risk_rows: list[dict]) -> dict:
    if not risk_rows:
        return {}
    runs: list[dict] = []
    current_bucket = str(risk_rows[0]["risk_bucket"])
    start_day = date.fromisoformat(str(risk_rows[0]["date"]))
    prev_day = start_day
    observation_count = 1

    for row in risk_rows[1:]:
        day = date.fromisoformat(str(row["date"]))
        bucket = str(row["risk_bucket"])
        if bucket == current_bucket:
            observation_count += 1
            prev_day = day
            continue
        runs.append(
            {
                "bucket": current_bucket,
                "start_date": start_day.isoformat(),
                "end_date": prev_day.isoformat(),
                "calendar_days": (prev_day - start_day).days + 1,
                "observation_days": observation_count,
            }
        )
        current_bucket = bucket
        start_day = day
        prev_day = day
        observation_count = 1

    runs.append(
        {
            "bucket": current_bucket,
            "start_date": start_day.isoformat(),
            "end_date": prev_day.isoformat(),
            "calendar_days": (prev_day - start_day).days + 1,
            "observation_days": observation_count,
        }
    )

    current_run = runs[-1]
    same_bucket_runs = [run for run in runs if run["bucket"] == current_run["bucket"]]
    sorted_lengths = sorted(int(run["calendar_days"]) for run in same_bucket_runs)
    median_length = statistics.median(sorted_lengths) if sorted_lengths else None
    p80_idx = int(round((len(sorted_lengths) - 1) * 0.8)) if sorted_lengths else None
    p80_length = sorted_lengths[p80_idx] if sorted_lengths else None

    return {
        "current_run": current_run,
        "historical_run_stats": {
            "bucket": current_run["bucket"],
            "run_count": len(same_bucket_runs),
            "median_calendar_days": median_length,
            "p80_calendar_days": p80_length,
            "max_calendar_days": max(sorted_lengths) if sorted_lengths else None,
        },
        "all_runs": runs,
    }


def _plot_equity_drawdown(equity_snapshot: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    dates = [_dt(value) for value in equity_snapshot["dates"]]
    ax1.plot(dates, equity_snapshot["equities"], linewidth=2.0, color=CYAN)
    ax1.set_title("Strategy Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(dates, [-value for value in equity_snapshot["drawdowns_pct"]], linewidth=1.6, color=RED)
    ax2.set_title("Realized Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "01_strategy_equity_drawdown.png")


def _plot_monthly_heatmap(monthly_rows: list[dict]) -> None:
    if not monthly_rows:
        return
    years = sorted({int(row["year"]) for row in monthly_rows})
    months = list(range(1, 13))
    lookup = {(int(row["year"]), int(row["month"])): float(row["pnl"]) for row in monthly_rows}
    grid = [[lookup.get((year, month), 0.0) for month in months] for year in years]
    vmax = max(abs(value) for row in grid for value in row) or 1.0

    fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.8)))
    image = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Monthly Realized PnL Heatmap")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    for row_idx, year in enumerate(years):
        for col_idx, month in enumerate(months):
            value = grid[row_idx][col_idx]
            ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7, color="#0f1117")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, CHART_DIR / "02_monthly_pnl_heatmap.png")


def _plot_strategy_vs_benchmarks(trade_edge_cumulative_rows: list[dict]) -> None:
    if not trade_edge_cumulative_rows:
        return
    dates = [_dt(row["exit_ts"]) for row in trade_edge_cumulative_rows]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, [float(row["strategy_cumulative_pnl"]) for row in trade_edge_cumulative_rows], label="Strategy", color=TEXT, linewidth=2.2)
    ax.plot(dates, [float(row["asset_benchmark_cumulative_pnl"]) for row in trade_edge_cumulative_rows], label="Same-asset benchmark", color="#60a5fa", linewidth=1.8)
    ax.plot(dates, [float(row["sector_benchmark_cumulative_pnl"]) for row in trade_edge_cumulative_rows], label="Sector benchmark", color=GREEN, linewidth=1.8)
    ax.plot(dates, [float(row["market_benchmark_cumulative_pnl"]) for row in trade_edge_cumulative_rows], label="Market benchmark", color="#f59e0b", linewidth=1.8)
    ax.set_title("Strategy PnL vs Trade-Window Benchmarks")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "03_strategy_vs_benchmarks.png")


def _plot_sector_contribution(sector_cumulative_rows: list[dict]) -> None:
    if not sector_cumulative_rows:
        return
    dates = [_dt(row["exit_ts"]) for row in sector_cumulative_rows]
    sector_keys = [key for key in sector_cumulative_rows[0].keys() if key.endswith("_cumulative_pnl")]
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in sector_keys:
        label = key.replace("_cumulative_pnl", "")
        ax.plot(dates, [float(row[key]) for row in sector_cumulative_rows], linewidth=1.8, label=label)
    ax.set_title("Cumulative Strategy PnL by Sector Group")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "04_sector_contribution.png")


def _plot_macro_regime_panel(
    *,
    daily_macro_rows: list[dict],
    strategy_state: dict[str, dict[date, float | None]],
    expansion_state: dict[str, dict[date, float | None]],
    risk_rows: list[dict],
) -> None:
    dates = [datetime.fromisoformat(str(row["date"])) for row in risk_rows]
    risk_scores = [float(row["risk_score"]) for row in risk_rows]
    buckets = [str(row["risk_bucket"]) for row in risk_rows]

    macro_lookup = {date.fromisoformat(str(row["date"])): row for row in daily_macro_rows}
    equities = [strategy_state["equity"].get(day.date()) for day in dates]
    dff = [macro_lookup[day.date()]["dff"] for day in dates]
    curve = [macro_lookup[day.date()]["curve_10y_2y"] for day in dates]
    dollar = [macro_lookup[day.date()]["dtwexbgs"] for day in dates]
    vix = [macro_lookup[day.date()]["vixcls"] for day in dates]
    expansion = [expansion_state["market_expansion_pct"].get(day.date()) for day in dates]

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(dates, equities, color=CYAN, linewidth=2.0)
    axes[0].set_title("Strategy Equity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, dff, color="#f59e0b", linewidth=1.8, label="Fed funds")
    axes[1].plot(dates, curve, color="#60a5fa", linewidth=1.6, label="10Y - 2Y")
    axes[1].axhline(0.0, color=BORDER, linewidth=0.9, linestyle="--")
    axes[1].set_title("Rates Regime")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, dollar, color="#8b5cf6", linewidth=1.8, label="Broad dollar")
    axes[2].plot(dates, vix, color=RED, linewidth=1.5, label="VIX")
    axes[2].set_title("Dollar Strength and Volatility")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(dates, expansion, color=GREEN, linewidth=1.8)
    axes[3].axhline(8.0, color=BORDER, linewidth=0.9, linestyle="--")
    axes[3].axhline(15.0, color=BORDER, linewidth=0.9, linestyle=":")
    axes[3].set_title("Equal-Weight Market Price Expansion vs 200D EMA")
    axes[3].set_ylabel("Expansion %")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(dates, risk_scores, color=TEXT, linewidth=1.8)
    axes[4].axhline(35.0, color=BORDER, linewidth=0.9, linestyle="--")
    axes[4].axhline(65.0, color=BORDER, linewidth=0.9, linestyle=":")
    axes[4].set_title("Heuristic Risk Score")
    axes[4].set_ylabel("Risk score")
    axes[4].grid(True, alpha=0.3)

    segment_start = 0
    while segment_start < len(dates):
        segment_bucket = buckets[segment_start]
        segment_end = segment_start + 1
        while segment_end < len(dates) and buckets[segment_end] == segment_bucket:
            segment_end += 1
        for ax in axes:
            ax.axvspan(
                dates[segment_start],
                dates[segment_end - 1],
                color=RISK_COLORS[segment_bucket],
                alpha=0.07,
            )
        segment_start = segment_end

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "05_macro_regime_panel.png")


def _plot_expansion_sensitivity(
    daily_summary: list[dict],
    trade_summary: list[dict],
) -> None:
    if not daily_summary or not trade_summary:
        return
    buckets = [row["expansion_bucket"] for row in daily_summary]
    x = list(range(len(buckets)))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.bar(
        [value - 0.18 for value in x],
        [float(row["avg_forward_21d_market_return_pct"]) for row in daily_summary],
        width=0.36,
        color="#60a5fa",
        label="Avg next 21d market return",
    )
    ax2.bar(
        [value + 0.18 for value in x],
        [float(row["avg_edge_vs_sector_pct"]) for row in trade_summary],
        width=0.36,
        color=GREEN,
        label="Avg trade edge vs sector",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets)
    ax1.set_title("Price Expansion Sensitivity")
    ax1.set_ylabel("Forward 21d market return %")
    ax2.set_ylabel("Trade edge vs sector %")
    ax1.grid(True, axis="y", alpha=0.3)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
    _savefig(fig, CHART_DIR / "06_expansion_sensitivity.png")


def _latest_macro_snapshot(
    *,
    macro_series: dict[str, dict[date, float]],
    target_day: date,
) -> dict:
    dff = _last_value_on_or_before(macro_series["dff"], target_day)
    dgs2 = _last_value_on_or_before(macro_series["dgs2"], target_day)
    dgs10 = _last_value_on_or_before(macro_series["dgs10"], target_day)
    dollar = _last_value_on_or_before(macro_series["dtwexbgs"], target_day)
    vix = _last_value_on_or_before(macro_series["vixcls"], target_day)
    return {
        "as_of": target_day.isoformat(),
        "fed_funds_rate": round(dff, 4) if dff is not None else None,
        "fed_funds_63d_change": round(_series_change(macro_series["dff"], target_day, 63), 4)
        if _series_change(macro_series["dff"], target_day, 63) is not None else None,
        "yield_2y": round(dgs2, 4) if dgs2 is not None else None,
        "yield_10y": round(dgs10, 4) if dgs10 is not None else None,
        "curve_10y_2y": round(dgs10 - dgs2, 4) if dgs10 is not None and dgs2 is not None else None,
        "broad_dollar_index": round(dollar, 4) if dollar is not None else None,
        "broad_dollar_63d_change": round(_series_change(macro_series["dtwexbgs"], target_day, 63), 4)
        if _series_change(macro_series["dtwexbgs"], target_day, 63) is not None else None,
        "vix": round(vix, 4) if vix is not None else None,
    }


def _print_console_summary(summary_payload: dict, *, output_dir: Path) -> None:
    current_risk = summary_payload["risk_model"]["current"]
    expansion = summary_payload["price_expansion"]["current"]
    print("\n" + "=" * 104)
    print("MACRO RISK AUDIT")
    print("=" * 104)
    print(
        "Window: "
        f"{summary_payload['analysis_window']['start_date']} -> {summary_payload['analysis_window']['end_date']}"
    )
    print(
        "Current macro: "
        f"DFF={summary_payload['latest_macro']['fed_funds_rate']}  "
        f"10Y={summary_payload['latest_macro']['yield_10y']}  "
        f"2Y={summary_payload['latest_macro']['yield_2y']}  "
        f"Curve={summary_payload['latest_macro']['curve_10y_2y']}  "
        f"Dollar={summary_payload['latest_macro']['broad_dollar_index']}  "
        f"VIX={summary_payload['latest_macro']['vix']}"
    )
    print(
        "Current risk model: "
        f"bucket={current_risk['risk_bucket']}  "
        f"score={current_risk['risk_score']}  "
        f"streak={summary_payload['risk_model']['duration']['current_run']['calendar_days']} calendar days"
    )
    print(
        "Expansion: "
        f"{expansion['current_expansion_pct']}% vs 200D EMA  "
        f"bucket={expansion['current_expansion_bucket']}"
    )
    print(
        "Continuation guide: "
        f"median same-bucket run={summary_payload['risk_model']['duration']['historical_run_stats']['median_calendar_days']} days  "
        f"p80={summary_payload['risk_model']['duration']['historical_run_stats']['p80_calendar_days']} days"
    )
    print(
        "Trade edge summary: "
        f"vs sector={summary_payload['trade_edge']['overall']['excess_vs_sector_pnl']}  "
        f"vs market={summary_payload['trade_edge']['overall']['excess_vs_market_pnl']}"
    )
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro and risk audit for the current baseline report.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lookback-years", type=float, default=5.0)
    parser.add_argument("--warmup-days", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve()
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    summary, trades = _load_report(report_dir)
    _ensure_tiingo_cache_symbols()
    universe = _current_baseline_universe()

    analysis_start = _dt(summary["start_date"]).date()
    analysis_end = _dt(summary["end_date"]).date()
    fetch_start = analysis_start - timedelta(days=400)
    fetch_end = max(analysis_end, date.today())

    print("Building local market state...")
    from edgar.services.session_turtle_portfolio import build_per_asset_technical_state

    technical_state = build_per_asset_technical_state(
        universe=list(universe),
        lookback_years=float(args.lookback_years),
        warmup_days=int(args.warmup_days),
        ema_period=200,
        adx_period=14,
    )
    daily_close_state = _build_daily_close_state(
        technical_state,
        start_date=fetch_start,
        end_date=fetch_end,
    )
    daily_return_state = _build_daily_return_state(daily_close_state)

    metadata_by_ticker = {
        ticker: {
            "asset_bucket": _asset_bucket(ticker),
            "sector_group": _sector_group(ticker)[0],
        }
        for ticker in daily_return_state
    }
    sector_return_state, _market_member_counts = _sector_return_state(daily_return_state, metadata_by_ticker)
    market_return_state = sector_return_state["Market"]
    analysis_days = sorted(day for day in market_return_state if analysis_start <= day <= analysis_end)

    print("Fetching official macro series from FRED...")
    macro_series = {
        name: _fetch_fred_series(
            series_id=payload["series_id"],
            start_date=fetch_start,
            end_date=fetch_end,
        )
        for name, payload in FRED_SERIES.items()
    }

    trade_edge_rows = _trade_edge_rows(
        trades,
        {
            ticker: {
                "asset_bucket": _asset_bucket(ticker),
                "sector_group": _sector_group(ticker)[0],
            }
            for ticker in {str(trade["ticker"]) for trade in trades}
        },
        sector_return_state,
        market_return_state,
    )
    trade_edge_cumulative_rows = _cumulative_trade_edge_rows(trade_edge_rows)

    strategy_state = _build_daily_strategy_state(summary, trades, analysis_days)
    expansion_state = _build_market_expansion_state(market_return_state, analysis_days)
    pairwise_corr_state = _rolling_pairwise_corr_state(daily_return_state, window=63)
    daily_macro_rows = _build_macro_rows(analysis_days=analysis_days, macro_series=macro_series)
    risk_rows = _build_risk_model_rows(
        analysis_days=analysis_days,
        strategy_state=strategy_state,
        expansion_state=expansion_state,
        pairwise_corr_state=pairwise_corr_state,
        macro_series=macro_series,
    )
    duration_stats = _bucket_run_stats(risk_rows)

    monthly_pnl_rows = _build_monthly_pnl_rows(trades)
    sector_cumulative_rows = _build_sector_cumulative_rows(trades)
    expansion_daily_summary, expansion_trade_summary, expansion_meta = _build_expansion_sensitivity(
        analysis_days=analysis_days,
        market_return_state=market_return_state,
        expansion_state=expansion_state["market_expansion_pct"],
        trade_edge_rows=trade_edge_rows,
    )

    overall_trade_edge = _aggregate_trade_edge(trade_edge_rows, None)[0]
    edge_by_sector = _aggregate_trade_edge(trade_edge_rows, "sector_group")
    edge_by_bucket = _aggregate_trade_edge(trade_edge_rows, "asset_bucket")

    latest_day = analysis_days[-1]
    current_risk = next(row for row in risk_rows if row["date"] == latest_day.isoformat())

    summary_payload = {
        "analysis_window": {
            "start_date": analysis_start.isoformat(),
            "end_date": analysis_end.isoformat(),
        },
        "report_source": str(report_dir),
        "latest_macro": _latest_macro_snapshot(macro_series=macro_series, target_day=latest_day),
        "trade_edge": {
            "overall": overall_trade_edge,
            "by_sector": edge_by_sector,
            "by_bucket": edge_by_bucket,
        },
        "risk_model": {
            "current": current_risk,
            "duration": duration_stats,
            "method": {
                "type": "heuristic weighted score",
                "weights": {
                    "drawdown": 0.20,
                    "rolling_pf": 0.15,
                    "pairwise_corr": 0.15,
                    "rates": 0.15,
                    "dollar": 0.15,
                    "vix": 0.10,
                    "price_expansion": 0.10,
                },
                "thresholds": {
                    "low": "< 35",
                    "medium": "35 to < 65",
                    "high": ">= 65",
                },
            },
        },
        "price_expansion": {
            "current": expansion_meta,
            "daily_buckets": expansion_daily_summary,
            "trade_entry_buckets": expansion_trade_summary,
        },
    }

    global CHART_DIR
    CHART_DIR = chart_dir

    _write_csv(output_dir / "macro_series.csv", daily_macro_rows)
    _write_csv(output_dir / "daily_risk_model.csv", risk_rows)
    _write_csv(output_dir / "monthly_pnl.csv", monthly_pnl_rows)
    _write_csv(output_dir / "sector_cumulative_pnl.csv", sector_cumulative_rows)
    _write_csv(output_dir / "trade_edge.csv", trade_edge_rows)
    _write_csv(output_dir / "trade_edge_cumulative.csv", trade_edge_cumulative_rows)
    _write_csv(output_dir / "trade_edge_by_sector.csv", edge_by_sector)
    _write_csv(output_dir / "trade_edge_by_bucket.csv", edge_by_bucket)
    _write_csv(output_dir / "expansion_daily_buckets.csv", expansion_daily_summary)
    _write_csv(output_dir / "expansion_trade_buckets.csv", expansion_trade_summary)
    _save_json(output_dir / "summary.json", summary_payload)

    equity_snapshot = _build_equity_snapshot(summary, trades)
    _plot_equity_drawdown(equity_snapshot)
    _plot_monthly_heatmap(monthly_pnl_rows)
    _plot_strategy_vs_benchmarks(trade_edge_cumulative_rows)
    _plot_sector_contribution(sector_cumulative_rows)
    _plot_macro_regime_panel(
        daily_macro_rows=daily_macro_rows,
        strategy_state=strategy_state,
        expansion_state=expansion_state,
        risk_rows=risk_rows,
    )
    _plot_expansion_sensitivity(expansion_daily_summary, expansion_trade_summary)

    _print_console_summary(summary_payload, output_dir=output_dir)


if __name__ == "__main__":
    main()
