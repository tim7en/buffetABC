"""
Compare Donchian channel granularity for the current baseline universe.

Variants
--------
1. Session 20/10: current baseline behavior.
2. Session 10/5 : faster session-bar breakout/exit.
3. Raw 5m 20/10: intraday Donchian on raw 5-minute bars.

Run from repo root:
  python tools/session_turtle_core_x2/run_channel_granularity_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

import edgar.services.session_turtle_portfolio as stp
import edgar.services.session_turtle_trend_strategy as stts
from edgar.services.session_open_utils import SessionBar
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)
from tools.session_turtle_core_x2.run_selection_edge_audit import (
    _current_baseline_universe,
    _ensure_tiingo_cache_symbols,
    _sector_group,
)


OUTPUT_DIR = Path("reports/channel_granularity_comparison/current_baseline")
CHART_DIR = OUTPUT_DIR / "charts"

LOOKBACK_YEARS = 4.1
TREND_FAST = 55
TREND_SLOW = 200
EMA_PERIOD = 200

BASE_REPORT_KWARGS = {
    "basket": "core",
    "exposure_mult": 3.0,
    "crypto_cap_mult": 1.0,
    "gold_cap_mult": 0.8,
    "metals_cap_mult": 0.8,
    "energy_cap_mult": 0.8,
    "equity_cap_mult": None,
    "base_risk_pct": 0.05,
    "fixed_stop_pct": 0.10,
    "directional_volume_risk_pct": 0.07,
    "use_extended_hours_proxy": True,
    "extended_hours_proxy_lag_days": 1,
    "extended_hours_vix_risk_on_threshold": 15.0,
    "extended_hours_vix_risk_off_threshold": 25.0,
    "extended_hours_fg_greed_threshold": 60.0,
    "extended_hours_fg_fear_threshold": 30.0,
    "extended_hours_long_risk_on_mult": 1.0,
    "extended_hours_long_neutral_mult": 1.0,
    "extended_hours_long_risk_off_mult": 0.5,
    "extended_hours_short_risk_on_mult": 1e-9,
    "extended_hours_short_neutral_mult": 1.0,
    "extended_hours_short_risk_off_mult": 1.0,
    "use_per_asset_technical_overlay": True,
    "per_asset_ema_lag_days": 1,
    "per_asset_ema_above_long_mult": 1.0,
    "per_asset_ema_above_short_mult": 0.0,
    "per_asset_ema_below_long_mult": 0.0,
    "per_asset_ema_below_short_mult": 1.0,
    "per_asset_use_adx_gate": False,
}

DARK_BG = "#0f1117"
PANEL_BG = "#171923"
BORDER = "#2b3245"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#60a5fa"
AMBER = "#f59e0b"

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


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def _load_macro_state(root: Path) -> dict:
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    return build_extended_hours_proxy_state(daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg)


@contextmanager
def _patched_universe():
    old_core = stp.CORE_SESSION_TURTLE_UNIVERSE
    old_expanded = stp.EXPANDED_SESSION_TURTLE_UNIVERSE
    old_equity = set(stp.EQUITY_TICKERS)
    old_energy = set(stp.ENERGY_TICKERS)

    custom_universe = tuple(_current_baseline_universe())
    stp.CORE_SESSION_TURTLE_UNIVERSE = custom_universe
    stp.EXPANDED_SESSION_TURTLE_UNIVERSE = tuple(list(custom_universe) + list(stp.INDEX_SESSION_TURTLE_UNIVERSE))
    stp.EQUITY_TICKERS = set(old_equity) | {"GOOGL", "META", "NVDA"}
    stp.ENERGY_TICKERS = set(old_energy) | {"BRENT", "NATGAS"}
    try:
        yield list(dict.fromkeys(custom_universe))
    finally:
        stp.CORE_SESSION_TURTLE_UNIVERSE = old_core
        stp.EXPANDED_SESSION_TURTLE_UNIVERSE = old_expanded
        stp.EQUITY_TICKERS = old_equity
        stp.ENERGY_TICKERS = old_energy


def _identity_session_bars(bars: list[dict], session_open: str):
    del session_open
    sessions = [
        SessionBar(
            anchor=bar["timestamp"],
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar.get("volume", 0.0) or 0.0),
        )
        for bar in bars
    ]
    mapping = list(range(len(bars)))
    return sessions, mapping


@contextmanager
def _patched_raw_5m_sessions(enabled: bool):
    if not enabled:
        yield
        return
    old_aggregate = stts.aggregate_session_bars
    stts.aggregate_session_bars = _identity_session_bars
    try:
        yield
    finally:
        stts.aggregate_session_bars = old_aggregate


def _trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "avg_hold_days": 0.0,
            "median_hold_days": 0.0,
            "avg_trade_return_pct": 0.0,
        }
    hold_days = [(_dt(row["exit_ts"]) - _dt(row["entry_ts"])).total_seconds() / 86400.0 for row in trades]
    trade_returns = [
        (float(row["net_pnl"]) / float(row["notional"]) * 100.0)
        for row in trades
        if float(row.get("notional", 0.0) or 0.0) > 0
    ]
    ordered_hold = sorted(hold_days)
    mid = len(ordered_hold) // 2
    median_hold = ordered_hold[mid] if len(ordered_hold) % 2 == 1 else (ordered_hold[mid - 1] + ordered_hold[mid]) / 2
    return {
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 2),
        "median_hold_days": round(median_hold, 2),
        "avg_trade_return_pct": round(sum(trade_returns) / len(trade_returns), 4) if trade_returns else 0.0,
    }


def _sector_pnl_rows(trades: list[dict], *, variant: str) -> list[dict]:
    grouped: dict[str, float] = {}
    for trade in trades:
        sector = _sector_group(str(trade["ticker"]))[0]
        grouped[sector] = grouped.get(sector, 0.0) + float(trade.get("net_pnl", 0.0) or 0.0)
    return [
        {"variant": variant, "sector_group": sector, "net_pnl": round(pnl, 4)}
        for sector, pnl in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]


def _run_variant(
    *,
    label: str,
    channel_period: int,
    raw_5m: bool,
    macro_state: dict,
    tech_state: dict,
) -> dict:
    candidate_kwargs = {
        "basket": "core",
        "initial_capital": 1_000.0,
        "lookback_years": LOOKBACK_YEARS,
        "channel_period": channel_period,
        "base_risk_pct": 0.05,
        "fixed_stop_pct": 0.10,
        "directional_volume_risk_pct": 0.07,
        "trend_fast_period": TREND_FAST,
        "trend_slow_period": TREND_SLOW,
    }
    with _patched_raw_5m_sessions(raw_5m):
        candidates = build_session_turtle_shared_account_candidates(**candidate_kwargs)
    report_kwargs = dict(BASE_REPORT_KWARGS)
    report_kwargs["extended_hours_proxy_state"] = macro_state
    report_kwargs["per_asset_technical_state"] = tech_state
    report_kwargs["precomputed_candidates"] = candidates
    result = generate_session_turtle_shared_account_report(**report_kwargs)
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["raw_5m_channels"] = raw_5m
    summary["channel_period"] = channel_period
    summary["exit_channel_period"] = 5 if channel_period == 10 else (10 if channel_period == 20 else 20)
    summary.update(_trade_stats(result["trades"]))
    return {
        "label": label,
        "summary": summary,
        "trades": list(result["trades"]),
        "sector_rows": _sector_pnl_rows(result["trades"], variant=label),
    }


def _plot_key_metrics(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    metrics = [
        ("total_return_pct", "Total Return %"),
        ("max_realized_drawdown_pct", "Max Realized DD %"),
        ("profit_factor", "Profit Factor"),
        ("executed_trades", "Executed Trades"),
    ]
    colors = [BLUE, GREEN, AMBER]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title) in zip(axes.flat, metrics):
        vals = [float(row[field]) for row in rows]
        ax.bar(x, vals, color=colors[: len(vals)], alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, CHART_DIR / "01_key_metrics.png")


def _plot_trade_tempo(rows: list[dict]) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    avg_hold = [float(row["avg_hold_days"]) for row in rows]
    avg_trade = [float(row["avg_trade_return_pct"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.bar([value - 0.18 for value in x], avg_hold, width=0.36, color=BLUE, label="Avg hold days")
    ax2.bar([value + 0.18 for value in x], avg_trade, width=0.36, color=GREEN, label="Avg trade return %")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Trade Tempo vs Average Trade Return")
    ax1.set_ylabel("Avg hold days")
    ax2.set_ylabel("Avg trade return %")
    ax1.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    _savefig(fig, CHART_DIR / "02_trade_tempo.png")


def _plot_sector_pnl(sector_rows: list[dict]) -> None:
    variants = sorted({row["variant"] for row in sector_rows})
    sectors = sorted({row["sector_group"] for row in sector_rows})
    lookup = {(row["variant"], row["sector_group"]): float(row["net_pnl"]) for row in sector_rows}
    x = list(range(len(sectors)))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = [BLUE, GREEN, AMBER]

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, variant in enumerate(variants):
        values = [lookup.get((variant, sector), 0.0) for sector in sectors]
        ax.bar([value + offsets[idx] for value in x], values, width=width, label=variant, color=colors[idx], alpha=0.9)
    ax.axhline(0.0, color=BORDER, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=25, ha="right")
    ax.set_title("Sector PnL by Channel Variant")
    ax.set_ylabel("Net PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    _savefig(fig, CHART_DIR / "03_sector_pnl.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Donchian channel granularity variants for the current baseline.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    global OUTPUT_DIR, CHART_DIR
    OUTPUT_DIR = output_dir
    CHART_DIR = chart_dir

    _ensure_tiingo_cache_symbols()

    with _patched_universe() as universe:
        macro_state = _load_macro_state(root)
        tech_state = build_per_asset_technical_state(
            universe=list(dict.fromkeys(universe)),
            lookback_years=5.0,
            warmup_days=300,
            ema_period=EMA_PERIOD,
            adx_period=14,
        )

        runs = [
            _run_variant(
                label="Session 20/10",
                channel_period=20,
                raw_5m=False,
                macro_state=macro_state,
                tech_state=tech_state,
            ),
            _run_variant(
                label="Session 10/5",
                channel_period=10,
                raw_5m=False,
                macro_state=macro_state,
                tech_state=tech_state,
            ),
            _run_variant(
                label="Raw 5m 20/10",
                channel_period=20,
                raw_5m=True,
                macro_state=macro_state,
                tech_state=tech_state,
            ),
        ]

    comparison_rows = [run["summary"] for run in runs]
    sector_rows = [row for run in runs for row in run["sector_rows"]]

    for run in runs:
        safe = run["label"].lower().replace(" ", "_").replace("/", "_")
        subdir = output_dir / safe
        subdir.mkdir(parents=True, exist_ok=True)
        _save_json(subdir / "summary.json", run["summary"])
        _write_csv(subdir / "trades.csv", run["trades"])

    _write_csv(output_dir / "comparison.csv", comparison_rows)
    _write_csv(output_dir / "sector_pnl.csv", sector_rows)
    _save_json(output_dir / "summary.json", {"runs": comparison_rows})

    _plot_key_metrics(comparison_rows)
    _plot_trade_tempo(comparison_rows)
    _plot_sector_pnl(sector_rows)

    print("\nCHANNEL GRANULARITY COMPARISON")
    print("=" * 96)
    for row in comparison_rows:
        print(
            f"{row['variant_label']:<16}  "
            f"Ret {row['total_return_pct']:>8.2f}%  "
            f"MaxDD {row['max_realized_drawdown_pct']:>6.2f}%  "
            f"PF {row['profit_factor']:>5.2f}  "
            f"Trades {row['executed_trades']:>4}  "
            f"AvgHold {row['avg_hold_days']:>6.2f}d"
        )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
