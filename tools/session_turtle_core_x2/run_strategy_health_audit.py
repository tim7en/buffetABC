"""
Professional health audit runner for Session Turtle Core x3.

What it does
------------
1. Re-runs the current production configuration.
2. Runs an optional crypto-sleeve diversification variant that keeps lagging
   crypto symbols alive at a floor size using a bucket-scoped leadership overlay.
3. Produces an audit pack with fold stability, stress tests, concentration,
   regime attribution, crypto correlation diagnostics, and a scorecard.

Run from repo root:
  python tools/session_turtle_core_x2/run_strategy_health_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


OUTPUT_DIR = Path("reports/strategy_health_audit")
CHART_DIR = OUTPUT_DIR / "charts"

_BASKET = "core"
_LOOKBACK_YRS = 4.1
_CHANNEL = 20
_EXPOSURE_MULT = 3.0
_BASE_RISK = 0.05
_FIXED_STOP = 0.10
_DIR_VOL_RISK = 0.07
_TREND_FAST = 55
_TREND_SLOW = 200
_EMA_PERIOD = 200

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

CRYPTO_TICKERS = ("BTC-USD", "ETH-USD", "SOL-USD")
SLIPPAGE_BPS = (5, 10, 25, 50)

DARK_BG = "#0f1117"
PANEL_BG = "#171923"
BORDER = "#2b3245"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
BLUE = "#60a5fa"
CYAN = "#22d3ee"
MAGENTA = "#c084fc"

plt.rcParams.update(
    {
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": BORDER,
        "grid.linewidth": 0.6,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": BORDER,
        "font.size": 9,
    }
)


def _dt(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_profit_factor(pnls: list[float]) -> float | None:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss <= 1e-12:
        return None if gross_profit <= 1e-12 else float("inf")
    return gross_profit / gross_loss


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_trades_csv(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    return cov / math.sqrt(var_x * var_y)


def _build_daily_return_state(technical_state: dict) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, float]] = {}
    for ticker, rows in (technical_state.get("daily_ema", {}) if technical_state else {}).items():
        ordered = sorted(
            (
                datetime.fromisoformat(day).date(),
                float(payload.get("close", 0.0) or 0.0),
            )
            for day, payload in rows.items()
        )
        prev_close: float | None = None
        ticker_returns: dict[date, float] = {}
        for day, close in ordered:
            if prev_close is not None and prev_close > 0:
                ticker_returns[day] = (close / prev_close) - 1.0
            prev_close = close
        out[str(ticker)] = ticker_returns
    return out


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

    ulcer_index = math.sqrt(sum(dd * dd for dd in drawdowns_pct) / len(drawdowns_pct)) if drawdowns_pct else 0.0
    return {
        "dates": [dt.isoformat() for dt in dates],
        "equities": [round(value, 4) for value in equities],
        "drawdowns_pct": [round(value, 4) for value in drawdowns_pct],
        "ulcer_index": round(ulcer_index, 4),
    }


def _bootstrap_mean_ci(values: list[float], *, iterations: int, seed: int) -> dict | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    sampled_means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sampled_means.append(sum(sample) / n)
    sampled_means.sort()
    low_idx = max(0, int(iterations * 0.025) - 1)
    high_idx = min(iterations - 1, int(iterations * 0.975))
    return {
        "mean": round(sum(values) / len(values), 4),
        "p2_5": round(sampled_means[low_idx], 4),
        "p97_5": round(sampled_means[high_idx], 4),
    }


def _max_consecutive_losses(pnls: list[float]) -> int:
    max_losses = 0
    current = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            max_losses = max(max_losses, current)
        else:
            current = 0
    return max_losses


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
        pf = _safe_profit_factor(pnls)
        profit_factor.append(round(pf, 4) if pf is not None and math.isfinite(pf) else None)
    return {"dates": dates, "expectancy": expectancy, "profit_factor": profit_factor}


def _group_stats(trades: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get(key) or "n/a")].append(trade)

    rows = []
    for group, items in grouped.items():
        pnls = [float(item.get("net_pnl", 0.0) or 0.0) for item in items]
        wins = sum(1 for pnl in pnls if pnl > 0)
        pf = _safe_profit_factor(pnls)
        rows.append(
            {
                "group": group,
                "trades": len(items),
                "net_pnl": round(sum(pnls), 4),
                "win_rate_pct": round(wins / len(items) * 100.0, 2) if items else None,
                "profit_factor": round(pf, 4) if pf is not None and math.isfinite(pf) else None,
                "avg_trade_pnl": round(sum(pnls) / len(items), 4) if items else None,
            }
        )
    return sorted(rows, key=lambda row: (-abs(row["net_pnl"]), row["group"]))


def _fold_stats(summary: dict, trades: list[dict], *, folds: int = 4) -> list[dict]:
    sorted_trades = sorted(trades, key=lambda trade: _dt(trade["exit_ts"]))
    if not sorted_trades:
        return []
    out = []
    total = len(sorted_trades)
    for fold_idx in range(folds):
        start = fold_idx * total // folds
        end = (fold_idx + 1) * total // folds
        fold_trades = sorted_trades[start:end]
        if not fold_trades:
            continue
        pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in fold_trades]
        start_equity = (
            float(summary["initial_capital"])
            if start == 0
            else float(sorted_trades[start - 1].get("equity_after_exit", summary["initial_capital"]) or summary["initial_capital"])
        )
        end_equity = float(fold_trades[-1].get("equity_after_exit", start_equity) or start_equity)
        pf = _safe_profit_factor(pnls)
        wins = sum(1 for pnl in pnls if pnl > 0)
        out.append(
            {
                "fold": fold_idx + 1,
                "start_ts": str(fold_trades[0]["entry_ts"]),
                "end_ts": str(fold_trades[-1]["exit_ts"]),
                "trades": len(fold_trades),
                "start_equity": round(start_equity, 4),
                "end_equity": round(end_equity, 4),
                "return_pct": round(((end_equity / start_equity) - 1.0) * 100.0, 2) if start_equity > 0 else None,
                "win_rate_pct": round(wins / len(fold_trades) * 100.0, 2),
                "profit_factor": round(pf, 4) if pf is not None and math.isfinite(pf) else None,
            }
        )
    return out


def _stress_test(summary: dict, trades: list[dict], *, bps_levels: tuple[int, ...]) -> list[dict]:
    initial = float(summary["initial_capital"])
    rows = []
    for bps in bps_levels:
        adjusted_pnls = []
        for trade in trades:
            pnl = float(trade.get("net_pnl", 0.0) or 0.0)
            notional = float(trade.get("notional", 0.0) or 0.0)
            friction = 2.0 * notional * bps / 10000.0
            adjusted_pnls.append(pnl - friction)
        final_equity = initial + sum(adjusted_pnls)
        pf = _safe_profit_factor(adjusted_pnls)
        wins = sum(1 for pnl in adjusted_pnls if pnl > 0)
        rows.append(
            {
                "slippage_bps_roundtrip": bps,
                "final_equity": round(final_equity, 4),
                "total_return_pct": round(((final_equity / initial) - 1.0) * 100.0, 2) if initial > 0 else None,
                "win_rate_pct": round(wins / len(adjusted_pnls) * 100.0, 2) if adjusted_pnls else None,
                "profit_factor": round(pf, 4) if pf is not None and math.isfinite(pf) else None,
            }
        )
    return rows


def _concentration(trades: list[dict]) -> dict:
    abs_pnl_by_ticker: dict[str, float] = defaultdict(float)
    net_pnl_by_ticker: dict[str, float] = defaultdict(float)
    trade_count_by_ticker: dict[str, int] = defaultdict(int)
    for trade in trades:
        ticker = str(trade["ticker"])
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        abs_pnl_by_ticker[ticker] += abs(pnl)
        net_pnl_by_ticker[ticker] += pnl
        trade_count_by_ticker[ticker] += 1

    total_abs = sum(abs_pnl_by_ticker.values())
    ranked = sorted(abs_pnl_by_ticker.items(), key=lambda item: (-item[1], item[0]))
    rows = []
    for ticker, abs_pnl in ranked:
        share = abs_pnl / total_abs if total_abs > 0 else 0.0
        rows.append(
            {
                "ticker": ticker,
                "abs_pnl": round(abs_pnl, 4),
                "abs_pnl_share": round(share, 4),
                "net_pnl": round(net_pnl_by_ticker[ticker], 4),
                "trade_count": trade_count_by_ticker[ticker],
            }
        )
    return {
        "top_assets": rows,
        "top1_abs_pnl_share": rows[0]["abs_pnl_share"] if rows else None,
        "top3_abs_pnl_share": round(sum(row["abs_pnl_share"] for row in rows[:3]), 4) if rows else None,
        "hhi": round(sum((row["abs_pnl_share"] ** 2) for row in rows), 4) if rows else None,
    }


def _crypto_correlation_snapshot(daily_return_state: dict[str, dict[date, float]]) -> dict:
    pairwise = []
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for idx, left in enumerate(CRYPTO_TICKERS):
        for right in CRYPTO_TICKERS[idx + 1 :]:
            left_series = daily_return_state.get(left, {})
            right_series = daily_return_state.get(right, {})
            common_days = sorted(left_series.keys() & right_series.keys())
            corr = None
            if len(common_days) >= 20:
                corr = _pearson_corr(
                    [left_series[day] for day in common_days],
                    [right_series[day] for day in common_days],
                )
            pairwise.append({"pair": f"{left} vs {right}", "corr": _round_or_none(corr, 4), "obs": len(common_days)})
            if corr is not None:
                by_ticker[left].append(abs(corr))
                by_ticker[right].append(abs(corr))

    per_ticker = []
    for ticker in CRYPTO_TICKERS:
        values = by_ticker.get(ticker, [])
        per_ticker.append(
            {
                "ticker": ticker,
                "avg_abs_corr_to_crypto_peers": round(sum(values) / len(values), 4) if values else None,
            }
        )
    return {"pairwise": pairwise, "per_ticker": per_ticker}


def _crypto_sleeve_stats(trades: list[dict]) -> dict:
    crypto_trades = [trade for trade in trades if str(trade.get("asset_bucket")) == "crypto"]
    by_ticker: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "net_pnl": 0.0, "mult_sum": 0.0})
    for trade in crypto_trades:
        ticker = str(trade["ticker"])
        by_ticker[ticker]["trades"] += 1
        by_ticker[ticker]["net_pnl"] += float(trade.get("net_pnl", 0.0) or 0.0)
        by_ticker[ticker]["mult_sum"] += float(trade.get("performance_risk_mult", 1.0) or 1.0)

    rows = []
    for ticker, payload in sorted(by_ticker.items()):
        count = int(payload["trades"])
        rows.append(
            {
                "ticker": ticker,
                "trades": count,
                "net_pnl": round(payload["net_pnl"], 4),
                "avg_performance_mult": round(payload["mult_sum"] / count, 4) if count else None,
            }
        )
    return {
        "trade_count": len(crypto_trades),
        "rows": rows,
        "avg_performance_mult": round(
            sum(float(trade.get("performance_risk_mult", 1.0) or 1.0) for trade in crypto_trades) / len(crypto_trades),
            4,
        ) if crypto_trades else None,
    }


def _health_scorecard(*, summary: dict, folds: list[dict], stress_rows: list[dict], concentration: dict) -> dict:
    stress_25 = next((row for row in stress_rows if row["slippage_bps_roundtrip"] == 25), None)
    profitable_folds = sum(1 for fold in folds if (fold.get("return_pct") or 0.0) > 0)
    tests = []

    def _judge(name: str, status: str, value: str) -> None:
        tests.append({"name": name, "status": status, "value": value})

    pf = float(summary["profit_factor"])
    if pf >= 1.75:
        _judge("Profit factor", "green", f"{pf:.2f}")
    elif pf >= 1.25:
        _judge("Profit factor", "yellow", f"{pf:.2f}")
    else:
        _judge("Profit factor", "red", f"{pf:.2f}")

    max_dd = float(summary["max_realized_drawdown_pct"])
    if max_dd <= 35.0:
        _judge("Max drawdown", "green", f"{max_dd:.2f}%")
    elif max_dd <= 45.0:
        _judge("Max drawdown", "yellow", f"{max_dd:.2f}%")
    else:
        _judge("Max drawdown", "red", f"{max_dd:.2f}%")

    if profitable_folds >= 3:
        _judge("Fold stability", "green", f"{profitable_folds}/{len(folds)} profitable folds")
    elif profitable_folds == 2:
        _judge("Fold stability", "yellow", f"{profitable_folds}/{len(folds)} profitable folds")
    else:
        _judge("Fold stability", "red", f"{profitable_folds}/{len(folds)} profitable folds")

    top1_share = float(concentration.get("top1_abs_pnl_share") or 0.0)
    if top1_share <= 0.35:
        _judge("Concentration", "green", f"Top asset abs PnL share {top1_share:.1%}")
    elif top1_share <= 0.50:
        _judge("Concentration", "yellow", f"Top asset abs PnL share {top1_share:.1%}")
    else:
        _judge("Concentration", "red", f"Top asset abs PnL share {top1_share:.1%}")

    trade_count = int(summary["executed_trades"])
    if trade_count >= 175:
        _judge("Trade sample", "green", f"{trade_count} trades")
    elif trade_count >= 100:
        _judge("Trade sample", "yellow", f"{trade_count} trades")
    else:
        _judge("Trade sample", "red", f"{trade_count} trades")

    if stress_25:
        stress_return = float(stress_25["total_return_pct"] or 0.0)
        stress_pf = float(stress_25["profit_factor"] or 0.0)
        if stress_return > 0 and stress_pf >= 1.35:
            _judge("25bps stress", "green", f"Return {stress_return:.1f}% / PF {stress_pf:.2f}")
        elif stress_return > 0:
            _judge("25bps stress", "yellow", f"Return {stress_return:.1f}% / PF {stress_pf:.2f}")
        else:
            _judge("25bps stress", "red", f"Return {stress_return:.1f}% / PF {stress_pf:.2f}")

    green_count = sum(1 for test in tests if test["status"] == "green")
    red_count = sum(1 for test in tests if test["status"] == "red")
    overall = "green" if red_count == 0 and green_count >= 4 else "yellow" if red_count <= 1 else "red"
    return {"overall": overall, "tests": tests}


def _variant_payload(
    *,
    name: str,
    summary: dict,
    trades: list[dict],
    daily_return_state: dict[str, dict[date, float]],
    bootstrap_iterations: int,
) -> dict:
    pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades]
    notionals = [float(trade.get("notional", 0.0) or 0.0) for trade in trades if float(trade.get("notional", 0.0) or 0.0) > 0]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    trade_returns_bps = [
        (float(trade.get("net_pnl", 0.0) or 0.0) / float(trade.get("notional", 0.0) or 1.0)) * 10000.0
        for trade in trades
        if float(trade.get("notional", 0.0) or 0.0) > 0
    ]
    equity = _build_equity_snapshot(summary, trades)
    folds = _fold_stats(summary, trades)
    stress = _stress_test(summary, trades, bps_levels=SLIPPAGE_BPS)
    concentration = _concentration(trades)
    scorecard = _health_scorecard(summary=summary, folds=folds, stress_rows=stress, concentration=concentration)
    pf = _safe_profit_factor(pnls)
    max_dd_pct = float(summary["max_realized_drawdown_pct"])
    max_dd_dollars = float(summary["initial_capital"]) * max_dd_pct / 100.0

    return {
        "name": name,
        "summary": dict(summary),
        "health": {
            "expectancy_usd": round(sum(pnls) / len(pnls), 4) if pnls else None,
            "expectancy_bps": round(sum(trade_returns_bps) / len(trade_returns_bps), 2) if trade_returns_bps else None,
            "median_trade_pnl": round(statistics.median(pnls), 4) if pnls else None,
            "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
            "payoff_ratio": round((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)), 4) if wins and losses else None,
            "profit_factor_recomputed": round(pf, 4) if pf is not None and math.isfinite(pf) else None,
            "ulcer_index": equity["ulcer_index"],
            "mar_ratio": round(float(summary["cagr_pct"]) / max_dd_pct, 4) if max_dd_pct > 0 else None,
            "recovery_factor": round(sum(pnls) / max_dd_dollars, 4) if max_dd_dollars > 0 else None,
            "max_consecutive_losses": _max_consecutive_losses(pnls),
            "bootstrap_expectancy_ci": _bootstrap_mean_ci(pnls, iterations=bootstrap_iterations, seed=7),
            "avg_notional": round(sum(notionals) / len(notionals), 4) if notionals else None,
        },
        "equity": equity,
        "folds": folds,
        "stress_test": stress,
        "concentration": concentration,
        "regimes": {
            "by_bucket": _group_stats(trades, "asset_bucket"),
            "by_direction": _group_stats(trades, "direction"),
            "by_ext_hours_regime": _group_stats(trades, "ext_hours_proxy_regime"),
            "by_ema_regime": _group_stats(trades, "technical_ema_regime"),
        },
        "crypto": {
            "daily_correlation": _crypto_correlation_snapshot(daily_return_state),
            "executed_sleeve": _crypto_sleeve_stats(trades),
        },
        "rolling": _rolling_window_stats(trades),
        "scorecard": scorecard,
    }


def _plot_equity_comparison(variants: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    colors = [BLUE, MAGENTA]
    for idx, variant in enumerate(variants):
        eq = variant["equity"]
        dates = [_dt(value) for value in eq["dates"]]
        ax1.plot(dates, eq["equities"], linewidth=1.8, color=colors[idx % len(colors)], label=variant["name"])
        ax2.plot(dates, [-value for value in eq["drawdowns_pct"]], linewidth=1.4, color=colors[idx % len(colors)])
    ax1.set_title("Equity Comparison")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2.set_title("Realized Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "01_equity_comparison.png")


def _plot_rolling_health(variants: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = [BLUE, MAGENTA]
    for idx, variant in enumerate(variants):
        rolling = variant["rolling"]
        dates = [_dt(value) for value in rolling["dates"]]
        expectancy = [value if value is not None else float("nan") for value in rolling["expectancy"]]
        pf = [value if value is not None else float("nan") for value in rolling["profit_factor"]]
        ax1.plot(dates, expectancy, linewidth=1.5, color=colors[idx % len(colors)], label=variant["name"])
        ax2.plot(dates, pf, linewidth=1.5, color=colors[idx % len(colors)], label=variant["name"])
    ax1.axhline(0.0, color=BORDER, linewidth=0.9, linestyle="--")
    ax1.set_title("Rolling 25-Trade Expectancy")
    ax1.set_ylabel("Expectancy ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.axhline(1.0, color=BORDER, linewidth=0.9, linestyle="--")
    ax2.set_title("Rolling 25-Trade Profit Factor")
    ax2.set_ylabel("PF")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "02_rolling_health.png")


def _plot_concentration(variants: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(14, 5), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        rows = variant["concentration"]["top_assets"][:6]
        tickers = [row["ticker"] for row in rows]
        shares = [row["abs_pnl_share"] * 100.0 for row in rows]
        ax.barh(tickers, shares, color=CYAN, alpha=0.85, edgecolor=BORDER)
        ax.set_title(variant["name"])
        ax.set_xlabel("Abs PnL share %")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Top Asset Concentration")
    _savefig(fig, CHART_DIR / "03_concentration.png")


def _write_markdown_report(path: Path, variants: list[dict]) -> None:
    lines = ["# Strategy Health Audit", ""]
    for variant in variants:
        summary = variant["summary"]
        lines.extend(
            [
                f"## {variant['name']}",
                f"- Overall health: **{variant['scorecard']['overall'].upper()}**",
                f"- Return / CAGR / MaxDD: {summary['total_return_pct']:.2f}% / {summary['cagr_pct']:.2f}% / {summary['max_realized_drawdown_pct']:.2f}%",
                f"- PF / WR / Trades: {summary['profit_factor']:.2f} / {summary['win_rate_pct']:.2f}% / {summary['executed_trades']}",
                f"- Top asset abs-PnL share: {float(variant['concentration']['top1_abs_pnl_share'] or 0.0):.1%}",
                "",
                "### Scorecard",
            ]
        )
        for test in variant["scorecard"]["tests"]:
            lines.append(f"- {test['name']}: {test['status']} ({test['value']})")
        lines.extend(["", "### Crypto Sleeve", ""])
        for row in variant["crypto"]["executed_sleeve"]["rows"]:
            lines.append(
                f"- {row['ticker']}: trades={row['trades']} net_pnl={row['net_pnl']:.2f} avg_mult={row['avg_performance_mult']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_variant(
    *,
    label: str,
    candidates: list[dict],
    macro_state: dict,
    tech_state: dict,
    extra_kwargs: dict | None = None,
) -> dict:
    kwargs = dict(**_BASE_KWARGS, **_OVERLAY_KWARGS)
    kwargs["extended_hours_proxy_state"] = macro_state
    kwargs["per_asset_technical_state"] = tech_state
    kwargs["precomputed_candidates"] = candidates
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    result = generate_session_turtle_shared_account_report(**kwargs)
    return {"name": label, "summary": dict(result["summary"]), "trades": list(result["trades"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a professional health audit for Session Turtle Core x3.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Where audit outputs should be written.")
    parser.add_argument("--bootstrap-iterations", type=int, default=750, help="Bootstrap iterations for expectancy CI.")
    parser.add_argument("--skip-crypto-diversified", action="store_true", help="Only audit the baseline production setup.")
    parser.add_argument("--performance-lookback-trades", type=int, default=6, help="Closed crypto trades used for the laggard-diversification overlay.")
    parser.add_argument("--performance-decay", type=float, default=0.75, help="Exponential decay used by the laggard-diversification overlay.")
    parser.add_argument("--crypto-laggard-floor-mult", type=float, default=0.85, help="Minimum multiplier for lagging crypto symbols.")
    parser.add_argument("--crypto-leader-cap-mult", type=float, default=1.10, help="Maximum multiplier for leading crypto symbols.")
    parser.add_argument("--performance-min-history", type=int, default=3, help="Closed trades required before ranking a crypto symbol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global OUTPUT_DIR, CHART_DIR
    OUTPUT_DIR = Path(args.output_dir)
    CHART_DIR = OUTPUT_DIR / "charts"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]

    print("\nLoading VIX daily closes + Crypto Fear & Greed...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg)

    print("Building per-asset technical state...")
    universe = list(dict.fromkeys((ticker, source, session) for ticker, source, session in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe,
        lookback_years=5.0,
        warmup_days=300,
        ema_period=_EMA_PERIOD,
        adx_period=14,
    )
    daily_return_state = _build_daily_return_state(tech_state)

    print("Building candidate trades...")
    candidates = build_session_turtle_shared_account_candidates(
        basket=_BASKET,
        initial_capital=1_000.0,
        lookback_years=_LOOKBACK_YRS,
        channel_period=_CHANNEL,
        base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP,
        directional_volume_risk_pct=_DIR_VOL_RISK,
        trend_fast_period=_TREND_FAST,
        trend_slow_period=_TREND_SLOW,
    )
    print(f"  Candidates: {len(candidates)}")

    print("\nRunning baseline production variant...")
    baseline_run = _run_variant(label="Baseline production", candidates=candidates, macro_state=macro_state, tech_state=tech_state)
    run_outputs = [baseline_run]

    if not args.skip_crypto_diversified:
        print("Running crypto sleeve diversification variant...")
        run_outputs.append(
            _run_variant(
                label="Crypto laggard diversified",
                candidates=candidates,
                macro_state=macro_state,
                tech_state=tech_state,
                extra_kwargs={
                    "use_performance_leadership_overlay": True,
                    "performance_lookback_trades": args.performance_lookback_trades,
                    "performance_decay": args.performance_decay,
                    "performance_floor_mult": args.crypto_laggard_floor_mult,
                    "performance_cap_mult": args.crypto_leader_cap_mult,
                    "performance_min_history": args.performance_min_history,
                    "performance_bucket_scopes": frozenset({"crypto"}),
                },
            )
        )

    variants = [
        _variant_payload(
            name=run_output["name"],
            summary=run_output["summary"],
            trades=run_output["trades"],
            daily_return_state=daily_return_state,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        for run_output in run_outputs
    ]

    for run_output, variant in zip(run_outputs, variants):
        safe_name = variant["name"].lower().replace(" ", "_").replace("-", "_")
        variant_dir = OUTPUT_DIR / safe_name
        _save_json(variant_dir / "summary.json", variant["summary"])
        _save_json(variant_dir / "health.json", variant)
        _save_trades_csv(variant_dir / "trades.csv", run_output["trades"])

    comparison = {
        variant["name"]: {
            "total_return_pct": variant["summary"]["total_return_pct"],
            "cagr_pct": variant["summary"]["cagr_pct"],
            "max_realized_drawdown_pct": variant["summary"]["max_realized_drawdown_pct"],
            "profit_factor": variant["summary"]["profit_factor"],
            "win_rate_pct": variant["summary"]["win_rate_pct"],
            "executed_trades": variant["summary"]["executed_trades"],
            "overall_health": variant["scorecard"]["overall"],
            "top1_abs_pnl_share": variant["concentration"]["top1_abs_pnl_share"],
            "avg_crypto_performance_mult": variant["crypto"]["executed_sleeve"]["avg_performance_mult"],
        }
        for variant in variants
    }
    _save_json(OUTPUT_DIR / "variant_comparison.json", comparison)
    _save_json(OUTPUT_DIR / "health_audit.json", {"generated_at": datetime.utcnow().isoformat(), "variants": variants})

    _plot_equity_comparison(variants)
    _plot_rolling_health(variants)
    _plot_concentration(variants)
    _write_markdown_report(OUTPUT_DIR / "health_audit.md", variants)

    print("\n" + "=" * 88)
    print("STRATEGY HEALTH AUDIT")
    print("=" * 88)
    for variant in variants:
        summary = variant["summary"]
        print(
            f"{variant['name']:<28} "
            f"Return {summary['total_return_pct']:>8.2f}%  "
            f"CAGR {summary['cagr_pct']:>7.2f}%  "
            f"MaxDD {summary['max_realized_drawdown_pct']:>6.2f}%  "
            f"PF {summary['profit_factor']:>5.2f}  "
            f"WR {summary['win_rate_pct']:>5.1f}%  "
            f"Trades {summary['executed_trades']:>4}  "
            f"Health {variant['scorecard']['overall'].upper()}"
        )
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
