"""
Trend-starvation counterfactual audit for a saved Session Turtle report.

Purpose
-------
Test what happens if the same assets stop making large moves over a two-year
window by compressing bar-to-bar returns while preserving the realized sign
sequence and intrabar structure.

Default run targets the current leading candidate:
  reports/channel_granularity_comparison/current_baseline/session_10_5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

import edgar.services.session_turtle_portfolio as stp
import edgar.services.session_turtle_trend_strategy as stts
from edgar.services.session_turtle_portfolio import (
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)
from tools.session_turtle_core_x2.run_path_scenario_audit import (
    CURRENT_BASELINE_CANDIDATE_KWARGS,
    CURRENT_BASELINE_REPORT_KWARGS,
    EMA_PERIOD,
    _dt,
    _ensure_tiingo_cache_symbols,
    _load_macro_state,
    _orig_load_binance_klines,
    _orig_load_tiingo_klines,
    _patched_raw_5m_sessions,
    _patched_universe,
    _save_json,
    _savefig,
    _write_csv,
    BLUE,
    GREEN,
    RED,
    TEXT,
    BORDER,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


REPORT_DIR = Path("reports/channel_granularity_comparison/current_baseline/session_10_5")
OUTPUT_DIR = Path("reports/trend_starvation_audit/session_10_5")
CHART_DIR = OUTPUT_DIR / "charts"


@dataclass(frozen=True)
class CompressionSpec:
    name: str
    label: str
    scale: float
    description: str


def _compression_specs() -> list[CompressionSpec]:
    return [
        CompressionSpec(
            name="baseline",
            label="Observed path",
            scale=1.0,
            description="Unchanged realized path.",
        ),
        CompressionSpec(
            name="half_moves",
            label="50% move amplitude",
            scale=0.50,
            description="All post-start bar moves are cut in half.",
        ),
        CompressionSpec(
            name="quarter_moves",
            label="25% move amplitude",
            scale=0.25,
            description="Large moves are heavily damped, but sign order is preserved.",
        ),
        CompressionSpec(
            name="near_flat",
            label="10% move amplitude",
            scale=0.10,
            description="Near-flat market: only 10% of realized move amplitude remains.",
        ),
    ]


class _CompressedBarLoader:
    def __init__(self, *, scale: float, start_date: date):
        self.scale = float(scale)
        self.start_date = start_date
        self._binance_cache: dict[tuple, tuple[list[dict], str]] = {}
        self._tiingo_cache: dict[tuple, tuple[list[dict], str, str]] = {}

    def _compress_bars(self, bars: list[dict]) -> list[dict]:
        if self.scale >= 0.999999:
            return [dict(row) for row in bars]

        out: list[dict] = []
        prev_close_orig: float | None = None
        prev_close_new: float | None = None

        for row in bars:
            new_row = dict(row)
            close_orig = float(row["close"])
            if prev_close_orig is None or prev_close_new is None or (row["timestamp"].date() < self.start_date):
                out.append(new_row)
                prev_close_orig = close_orig
                prev_close_new = close_orig
                continue

            if prev_close_orig <= 0 or prev_close_new <= 0:
                out.append(new_row)
                prev_close_orig = close_orig
                prev_close_new = close_orig
                continue

            def _scaled_price(original_price: float) -> float:
                rel = (float(original_price) / prev_close_orig) - 1.0
                return prev_close_new * (1.0 + (self.scale * rel))

            open_new = _scaled_price(float(row["open"]))
            high_new = _scaled_price(float(row["high"]))
            low_new = _scaled_price(float(row["low"]))
            close_new = _scaled_price(close_orig)
            high_new = max(high_new, open_new, close_new)
            low_new = min(low_new, open_new, close_new)

            new_row["open"] = open_new
            new_row["high"] = high_new
            new_row["low"] = low_new
            new_row["close"] = close_new
            out.append(new_row)

            prev_close_orig = close_orig
            prev_close_new = close_new

        return out

    def load_binance(self, *, ticker: str, interval: str, lookback_years: float, warmup_days: int, market_data_symbol: str | None = None):
        key = (ticker, interval, lookback_years, warmup_days, market_data_symbol)
        if key not in self._binance_cache:
            bars, symbol = _orig_load_binance_klines(
                ticker=ticker,
                interval=interval,
                lookback_years=lookback_years,
                warmup_days=warmup_days,
                market_data_symbol=market_data_symbol,
            )
            self._binance_cache[key] = (self._compress_bars(bars), symbol)
        bars, symbol = self._binance_cache[key]
        return [dict(row) for row in bars], symbol

    def load_tiingo(self, *, ticker: str, interval: str, lookback_years: float, warmup_days: int, market_data_symbol: str | None = None):
        key = (ticker, interval, lookback_years, warmup_days, market_data_symbol)
        if key not in self._tiingo_cache:
            bars, symbol, path = _orig_load_tiingo_klines(
                ticker=ticker,
                interval=interval,
                lookback_years=lookback_years,
                warmup_days=warmup_days,
                market_data_symbol=market_data_symbol,
            )
            self._tiingo_cache[key] = (self._compress_bars(bars), symbol, path)
        bars, symbol, path = self._tiingo_cache[key]
        return [dict(row) for row in bars], symbol, path


@contextmanager
def _patched_compression_loaders(*, scale: float, start_date: date):
    loader = _CompressedBarLoader(scale=scale, start_date=start_date)
    old_stts_binance = stts.load_local_binance_klines
    old_stts_tiingo = stts.load_local_tiingo_klines
    old_stp_binance = stp.load_local_binance_klines
    old_stp_tiingo = stp.load_local_tiingo_klines
    stts.load_local_binance_klines = loader.load_binance
    stts.load_local_tiingo_klines = loader.load_tiingo
    stp.load_local_binance_klines = loader.load_binance
    stp.load_local_tiingo_klines = loader.load_tiingo
    try:
        yield
    finally:
        stts.load_local_binance_klines = old_stts_binance
        stts.load_local_tiingo_klines = old_stts_tiingo
        stp.load_local_binance_klines = old_stp_binance
        stp.load_local_tiingo_klines = old_stp_tiingo


def _trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "avg_hold_days": 0.0,
            "avg_trade_return_pct": 0.0,
        }
    hold_days = [(_dt(row["exit_ts"]) - _dt(row["entry_ts"])).total_seconds() / 86400.0 for row in trades]
    trade_returns = [
        (float(row["net_pnl"]) / float(row["notional"]) * 100.0)
        for row in trades
        if float(row.get("notional", 0.0) or 0.0) > 0
    ]
    return {
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 2),
        "avg_trade_return_pct": round(sum(trade_returns) / len(trade_returns), 4) if trade_returns else 0.0,
    }


def _window_sector_rows(trades: list[dict], *, start_date: date, variant: str) -> list[dict]:
    grouped: dict[str, float] = {}
    total = 0.0
    total_trades = 0
    for trade in trades:
        if _dt(trade["entry_ts"]).date() < start_date:
            continue
        sector = trade["asset_bucket"] if trade.get("asset_bucket") in {"crypto", "gold", "metals", "energy"} else "equity"
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        grouped[sector] = grouped.get(sector, 0.0) + pnl
        total += pnl
        total_trades += 1
    rows = [
        {"variant": variant, "bucket": bucket, "window_pnl": round(pnl, 4), "window_trades": ""}
        for bucket, pnl in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]
    rows.append({"variant": variant, "bucket": "overall", "window_pnl": round(total, 4), "window_trades": total_trades})
    return rows


def _equity_curve_rows(summary: dict, trades: list[dict]) -> list[dict]:
    capital = float(summary["initial_capital"])
    rows = [{"date": str(summary["start_date"]), "equity": round(capital, 4)}]
    for trade in sorted(trades, key=lambda row: _dt(row["exit_ts"])):
        capital += float(trade.get("net_pnl", 0.0) or 0.0)
        rows.append({"date": str(trade["exit_ts"]), "equity": round(capital, 4)})
    return rows


def _run_variant(
    *,
    report_summary: dict,
    raw_5m_channels: bool,
    spec: CompressionSpec,
    start_date: date,
    history_years: float,
    root: Path,
) -> dict:
    macro_state = _load_macro_state(root)
    channel_period = int(report_summary.get("channel_period", 20) or 20)
    with _patched_universe() as universe:
        with _patched_compression_loaders(scale=spec.scale, start_date=start_date):
            with _patched_raw_5m_sessions(raw_5m_channels):
                tech_state = build_per_asset_technical_state(
                    universe=list(dict.fromkeys(universe)),
                    lookback_years=history_years,
                    warmup_days=300,
                    ema_period=EMA_PERIOD,
                    adx_period=14,
                )
                candidate_kwargs = dict(CURRENT_BASELINE_CANDIDATE_KWARGS)
                candidate_kwargs["lookback_years"] = history_years
                candidate_kwargs["channel_period"] = channel_period
                candidates = build_session_turtle_shared_account_candidates(**candidate_kwargs)
                report_kwargs = dict(CURRENT_BASELINE_REPORT_KWARGS)
                report_kwargs["extended_hours_proxy_state"] = macro_state
                report_kwargs["per_asset_technical_state"] = tech_state
                report_kwargs["precomputed_candidates"] = candidates
                result = generate_session_turtle_shared_account_report(**report_kwargs)

    summary = dict(result["summary"])
    summary["variant_label"] = spec.label
    summary["compression_scale"] = spec.scale
    summary["channel_period"] = channel_period
    summary["exit_channel_period"] = 5 if channel_period == 10 else (10 if channel_period == 20 else 20)
    summary["raw_5m_channels"] = raw_5m_channels
    summary.update(_trade_stats(result["trades"]))
    return {
        "label": spec.label,
        "description": spec.description,
        "summary": summary,
        "trades": list(result["trades"]),
        "equity_curve": _equity_curve_rows(summary, result["trades"]),
        "window_rows": _window_sector_rows(result["trades"], start_date=start_date, variant=spec.label),
    }


def _plot_equity_curves(runs: list[dict]) -> None:
    colors = [TEXT, BLUE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, run in enumerate(runs):
        dates = [_dt(row["date"]) for row in run["equity_curve"]]
        values = [float(row["equity"]) for row in run["equity_curve"]]
        ax.plot(dates, values, linewidth=2.0, label=run["label"], color=colors[idx])
    ax.set_title("Trend Starvation Replay: Equity Curves")
    ax.set_ylabel("Realized equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "01_equity_curves.png")


def _plot_key_metrics(rows: list[dict]) -> None:
    labels = [row["variant_label"] for row in rows]
    x = list(range(len(labels)))
    metrics = [
        ("total_return_pct", "Total Return %"),
        ("max_realized_drawdown_pct", "Max Realized DD %"),
        ("profit_factor", "Profit Factor"),
        ("executed_trades", "Executed Trades"),
    ]
    colors = [TEXT, BLUE, GREEN, RED]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title) in zip(axes.flat, metrics):
        vals = [float(row[field]) for row in rows]
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, CHART_DIR / "02_key_metrics.png")


def _plot_window_pnl(window_rows: list[dict]) -> None:
    rows = [row for row in window_rows if row["bucket"] != "overall"]
    variants = sorted({row["variant"] for row in rows})
    buckets = sorted({row["bucket"] for row in rows})
    lookup = {(row["variant"], row["bucket"]): float(row["window_pnl"]) for row in rows}
    x = list(range(len(buckets)))
    width = 0.18
    offsets = [-0.27, -0.09, 0.09, 0.27]
    colors = [TEXT, BLUE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, variant in enumerate(variants):
        vals = [lookup.get((variant, bucket), 0.0) for bucket in buckets]
        ax.bar([value + offsets[idx] for value in x], vals, width=width, label=variant, color=colors[idx], alpha=0.9)
    ax.axhline(0.0, color=BORDER, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_title("Two-Year Window PnL by Sleeve Under Trend Compression")
    ax.set_ylabel("PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    _savefig(fig, CHART_DIR / "03_window_pnl.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trend-starvation audit for a saved Session Turtle report.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-years", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve()
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    global OUTPUT_DIR, CHART_DIR
    OUTPUT_DIR = output_dir
    CHART_DIR = chart_dir

    _ensure_tiingo_cache_symbols()

    report_summary = json.loads((report_dir / "summary.json").read_text())
    raw_5m_channels = bool(report_summary.get("raw_5m_channels", False))
    analysis_end = _dt(report_summary["end_date"]).date()
    start_date = analysis_end - timedelta(days=int(365.25 * float(args.window_years)))
    history_years = max(float(args.window_years) + 1.0, 3.0)

    print(f"Trend starvation window: {start_date.isoformat()} -> {analysis_end.isoformat()}")
    print(f"Replay history: {history_years:.2f} years")

    runs = []
    for spec in _compression_specs():
        print(f"Running variant: {spec.label} (scale={spec.scale:.2f})")
        runs.append(
            _run_variant(
                report_summary=report_summary,
                raw_5m_channels=raw_5m_channels,
                spec=spec,
                start_date=start_date,
                history_years=history_years,
                root=root,
            )
        )

    comparison_rows = [run["summary"] for run in runs]
    window_rows = [row for run in runs for row in run["window_rows"]]

    for run in runs:
        safe = run["label"].lower().replace("%", "pct").replace(" ", "_")
        subdir = output_dir / safe
        subdir.mkdir(parents=True, exist_ok=True)
        _save_json(subdir / "summary.json", run["summary"])
        _write_csv(subdir / "trades.csv", run["trades"])

    _write_csv(output_dir / "comparison.csv", comparison_rows)
    _write_csv(output_dir / "window_pnl.csv", window_rows)
    _save_json(
        output_dir / "summary.json",
        {
            "report_source": str(report_dir),
            "window_start": start_date.isoformat(),
            "window_end": analysis_end.isoformat(),
            "runs": comparison_rows,
        },
    )

    _plot_equity_curves(runs)
    _plot_key_metrics(comparison_rows)
    _plot_window_pnl(window_rows)

    print("\nTREND STARVATION AUDIT")
    print("=" * 96)
    for row in comparison_rows:
        print(
            f"{row['variant_label']:<20}  "
            f"Ret {row['total_return_pct']:>8.2f}%  "
            f"MaxDD {row['max_realized_drawdown_pct']:>6.2f}%  "
            f"PF {row['profit_factor']:>5.2f}  "
            f"Trades {row['executed_trades']:>4}  "
            f"AvgHold {row['avg_hold_days']:>6.2f}d"
        )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
