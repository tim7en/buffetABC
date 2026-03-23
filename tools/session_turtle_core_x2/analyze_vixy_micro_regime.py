"""Analyze a VIXY-style 5-minute volatility proxy against the core x2 baseline.

This script treats the proxy as an intraday regime overlay for the
New York equity session only. It writes simple CSV/JSON outputs under
reports/vixy_micro_regime_core_x2/ so the study can be reviewed or rerun.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRADE_CSV = (
    "reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_trades.csv"
)
DEFAULT_SPY_PARQUET = "cache/cache/tiingo/SPY_5m.parquet"
DEFAULT_PROXY_CANDIDATES = [
    "cache/cache/tiingo/VIXY_5m.parquet",
    "cache/cache/tiingo/VIX_5m.parquet",
]
DEFAULT_REPORT_DIR = "reports/vixy_micro_regime_core_x2"


@dataclass
class ProxyBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a VIXY-style 5-minute volatility proxy against the core x2 baseline."
    )
    parser.add_argument(
        "--proxy-parquet",
        default=None,
        help=(
            "Path to the 5-minute proxy parquet. If omitted, the script tries "
            "cache/cache/tiingo/VIXY_5m.parquet then cache/cache/tiingo/VIX_5m.parquet."
        ),
    )
    parser.add_argument(
        "--trade-csv",
        default=DEFAULT_TRADE_CSV,
        help="Baseline shared-account trade CSV to analyze.",
    )
    parser.add_argument(
        "--spy-parquet",
        default=DEFAULT_SPY_PARQUET,
        help="Reference SPY parquet used to estimate trading-day coverage.",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Output directory for CSV/JSON report artifacts.",
    )
    parser.add_argument(
        "--proxy-label",
        default="VIXY",
        help="Human-readable label for the volatility proxy used in the report outputs.",
    )
    parser.add_argument(
        "--fresh-max-minutes",
        type=int,
        default=60,
        help="Maximum age of the last proxy bar for a same-session signal to be considered fresh.",
    )
    parser.add_argument(
        "--short-ma-bars",
        type=int,
        default=78,
        help="Short moving-average length in 5-minute bars. Default is 1 US session.",
    )
    parser.add_argument(
        "--long-ma-bars",
        type=int,
        default=390,
        help="Long moving-average length in 5-minute bars. Default is 5 US sessions.",
    )
    parser.add_argument(
        "--shock-lookback-bars",
        type=int,
        default=12,
        help="Lookback for the intraday shock return. Default is 1 hour.",
    )
    parser.add_argument(
        "--shock-up-threshold",
        type=float,
        default=0.05,
        help="Shock-up threshold for the 1-hour return.",
    )
    parser.add_argument(
        "--relief-down-threshold",
        type=float,
        default=-0.03,
        help="Relief-down threshold for the 1-hour return.",
    )
    return parser.parse_args()


def _resolve_input(root: Path, value: str | None, defaults: list[str] | None = None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    for candidate in defaults or []:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Could not resolve a proxy parquet path.")


def _read_parquet_columns(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read local parquet caches") from exc

    table = pq.read_table(path, columns=columns)
    return {name: table[name].to_pylist() for name in table.column_names}


def _normalize_timestamp(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if getattr(value, "tzinfo", None) is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _load_proxy_bars(path: Path) -> list[ProxyBar]:
    cols = _read_parquet_columns(path, ["time", "o", "h", "l", "c", "v"])
    bars: list[ProxyBar] = []
    for idx in range(len(cols["time"])):
        bars.append(
            ProxyBar(
                timestamp=_normalize_timestamp(cols["time"][idx]),
                open=float(cols["o"][idx]),
                high=float(cols["h"][idx]),
                low=float(cols["l"][idx]),
                close=float(cols["c"][idx]),
                volume=float(cols["v"][idx] or 0.0),
            )
        )
    return bars


def _load_unique_days(path: Path) -> set[str]:
    cols = _read_parquet_columns(path, ["time"])
    days: set[str] = set()
    for value in cols["time"]:
        ts = _normalize_timestamp(value)
        days.add(ts.date().isoformat())
    return days


def _simple_moving_average(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= window:
            total -= values[idx - window]
        if idx >= window - 1:
            out[idx] = total / window
    return out


def _rolling_return(values: list[float], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for idx in range(lookback, len(values)):
        prior = values[idx - lookback]
        if prior:
            out[idx] = values[idx] / prior - 1.0
    return out


def _load_trades(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["entry_ts"] = datetime.fromisoformat(row["entry_ts"])
        row["net_pnl"] = float(row["net_pnl"])
        row["notional"] = float(row["notional"])
    return rows


def _regime_for_bar(
    close: float,
    sma_short: float | None,
    sma_long: float | None,
) -> str:
    if sma_short is None or sma_long is None:
        return "insufficient_history"
    if close <= sma_short and sma_short <= sma_long:
        return "risk_on_micro"
    if close > sma_short and sma_short > sma_long:
        return "risk_off_micro"
    return "neutral_micro"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _contiguous_day_blocks(days: set[str]) -> list[dict[str, Any]]:
    if not days:
        return []
    day_values = sorted(datetime.fromisoformat(day).date() for day in days)
    blocks: list[dict[str, Any]] = []
    block_start = day_values[0]
    block_end = day_values[0]
    for day in day_values[1:]:
        if day - block_end > timedelta(days=4):
            blocks.append(
                {
                    "start_date": block_start.isoformat(),
                    "end_date": block_end.isoformat(),
                    "calendar_days": (block_end - block_start).days + 1,
                }
            )
            block_start = day
        block_end = day
    blocks.append(
        {
            "start_date": block_start.isoformat(),
            "end_date": block_end.isoformat(),
            "calendar_days": (block_end - block_start).days + 1,
        }
    )
    return blocks


def main() -> None:
    args = _parse_args()
    root = _project_root()
    proxy_path = _resolve_input(root, args.proxy_parquet, DEFAULT_PROXY_CANDIDATES)
    trade_path = _resolve_input(root, args.trade_csv)
    spy_path = _resolve_input(root, args.spy_parquet)
    report_dir = _resolve_input(root, args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    proxy_bars = _load_proxy_bars(proxy_path)
    if not proxy_bars:
        raise RuntimeError(f"No bars found in {proxy_path}")

    proxy_times = [bar.timestamp for bar in proxy_bars]
    proxy_close = [bar.close for bar in proxy_bars]
    proxy_days = {bar.timestamp.date().isoformat() for bar in proxy_bars}
    spy_days = _load_unique_days(spy_path)
    coverage_blocks = _contiguous_day_blocks(proxy_days)

    window_start = proxy_bars[0].timestamp.date().isoformat()
    window_end = proxy_bars[-1].timestamp.date().isoformat()
    relevant_spy_days = {
        day for day in spy_days if window_start <= day <= window_end
    }

    sma_short = _simple_moving_average(proxy_close, args.short_ma_bars)
    sma_long = _simple_moving_average(proxy_close, args.long_ma_bars)
    ret_1h = _rolling_return(proxy_close, args.shock_lookback_bars)

    trades = _load_trades(trade_path)
    enriched_rows: list[dict[str, Any]] = []
    fresh_rows: list[dict[str, Any]] = []

    for row in trades:
        entry_ts: datetime = row["entry_ts"]
        bar_idx = bisect.bisect_right(proxy_times, entry_ts) - 1
        if bar_idx < 0:
            enriched = {
                **row,
                "signal_age_min": None,
                "proxy_bar_ts": None,
                "proxy_close": None,
                "proxy_sma_short": None,
                "proxy_sma_long": None,
                "proxy_ret_1h": None,
                "proxy_regime": "missing_before_start",
                "fresh_same_session_signal": False,
                "shock_up": False,
                "relief_down": False,
            }
        else:
            matched_ts = proxy_times[bar_idx]
            age_min = (entry_ts - matched_ts).total_seconds() / 60.0
            regime = _regime_for_bar(proxy_close[bar_idx], sma_short[bar_idx], sma_long[bar_idx])
            shock_flag = ret_1h[bar_idx] is not None and ret_1h[bar_idx] >= args.shock_up_threshold
            relief_flag = (
                ret_1h[bar_idx] is not None and ret_1h[bar_idx] <= args.relief_down_threshold
            )
            fresh_signal = (
                row["session_open"] == "new_york_equity_open"
                and age_min <= args.fresh_max_minutes
                and regime != "insufficient_history"
            )
            enriched = {
                **row,
                "signal_age_min": round(age_min, 1),
                "proxy_bar_ts": matched_ts.isoformat(),
                "proxy_close": round(proxy_close[bar_idx], 4),
                "proxy_sma_short": _round_or_none(sma_short[bar_idx]),
                "proxy_sma_long": _round_or_none(sma_long[bar_idx]),
                "proxy_ret_1h": _round_or_none(ret_1h[bar_idx]),
                "proxy_regime": regime,
                "fresh_same_session_signal": fresh_signal,
                "shock_up": bool(shock_flag),
                "relief_down": bool(relief_flag),
            }
        enriched_rows.append(enriched)
        if enriched["fresh_same_session_signal"]:
            fresh_rows.append(enriched)

    entry_fieldnames = [
        "ticker",
        "source",
        "session_open",
        "asset_bucket",
        "direction",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "shares",
        "notional",
        "entry_exposure_mult",
        "scale",
        "entry_rel_volume",
        "risk_model",
        "net_pnl",
        "equity_after_exit",
        "signal_age_min",
        "proxy_bar_ts",
        "proxy_close",
        "proxy_sma_short",
        "proxy_sma_long",
        "proxy_ret_1h",
        "proxy_regime",
        "fresh_same_session_signal",
        "shock_up",
        "relief_down",
    ]
    _write_csv(report_dir / "trade_entries_with_vixy_regime.csv", enriched_rows, entry_fieldnames)

    direction_summary: list[dict[str, Any]] = []
    for direction in ["long", "short"]:
        direction_rows = [row for row in fresh_rows if row["direction"] == direction]
        for regime in ["risk_on_micro", "neutral_micro", "risk_off_micro"]:
            regime_rows = [row for row in direction_rows if row["proxy_regime"] == regime]
            if not regime_rows:
                continue
            wins = sum(1 for row in regime_rows if row["net_pnl"] > 0)
            direction_summary.append(
                {
                    "direction": direction,
                    "regime": regime,
                    "trades": len(regime_rows),
                    "wins": wins,
                    "win_rate_pct": round(100.0 * wins / len(regime_rows), 2),
                    "net_pnl": round(sum(row["net_pnl"] for row in regime_rows), 4),
                    "avg_pnl": round(
                        sum(row["net_pnl"] for row in regime_rows) / len(regime_rows),
                        4,
                    ),
                    "avg_notional": round(
                        sum(row["notional"] for row in regime_rows) / len(regime_rows),
                        4,
                    ),
                }
            )

    _write_csv(
        report_dir / "regime_direction_summary.csv",
        direction_summary,
        [
            "direction",
            "regime",
            "trades",
            "wins",
            "win_rate_pct",
            "net_pnl",
            "avg_pnl",
            "avg_notional",
        ],
    )

    overlay_rules = {
        "baseline_no_overlay": lambda row: 1.0,
        "half_long_risk_off": (
            lambda row: 0.5
            if row["direction"] == "long" and row["proxy_regime"] == "risk_off_micro"
            else 1.0
        ),
        "skip_long_risk_off": (
            lambda row: 0.0
            if row["direction"] == "long" and row["proxy_regime"] == "risk_off_micro"
            else 1.0
        ),
        "half_short_risk_on": (
            lambda row: 0.5
            if row["direction"] == "short" and row["proxy_regime"] == "risk_on_micro"
            else 1.0
        ),
        "skip_short_risk_on": (
            lambda row: 0.0
            if row["direction"] == "short" and row["proxy_regime"] == "risk_on_micro"
            else 1.0
        ),
        "half_both_mismatched": (
            lambda row: 0.5
            if (
                (row["direction"] == "long" and row["proxy_regime"] == "risk_off_micro")
                or (row["direction"] == "short" and row["proxy_regime"] == "risk_on_micro")
            )
            else 1.0
        ),
        "skip_both_mismatched": (
            lambda row: 0.0
            if (
                (row["direction"] == "long" and row["proxy_regime"] == "risk_off_micro")
                or (row["direction"] == "short" and row["proxy_regime"] == "risk_on_micro")
            )
            else 1.0
        ),
        "half_long_risk_off_plus_skip_short_risk_on": (
            lambda row: 0.5
            if row["direction"] == "long" and row["proxy_regime"] == "risk_off_micro"
            else (
                0.0
                if row["direction"] == "short" and row["proxy_regime"] == "risk_on_micro"
                else 1.0
            )
        ),
    }

    base_pnl = sum(row["net_pnl"] for row in enriched_rows)
    overlay_summary: list[dict[str, Any]] = []
    for name, rule in overlay_rules.items():
        scaled_total = 0.0
        affected_count = 0
        fresh_affected_count = 0
        for row in enriched_rows:
            scale = 1.0
            if row["fresh_same_session_signal"]:
                scale = rule(row)
                if scale != 1.0:
                    fresh_affected_count += 1
            else:
                scale = 1.0
            if scale != 1.0:
                affected_count += 1
            scaled_total += row["net_pnl"] * scale
        overlay_summary.append(
            {
                "overlay": name,
                "trade_level_proxy_pnl": round(scaled_total, 4),
                "delta_vs_baseline": round(scaled_total - base_pnl, 4),
                "affected_trades": affected_count,
                "fresh_signal_affected_trades": fresh_affected_count,
            }
        )

    _write_csv(
        report_dir / "overlay_proxy_summary.csv",
        overlay_summary,
        [
            "overlay",
            "trade_level_proxy_pnl",
            "delta_vs_baseline",
            "affected_trades",
            "fresh_signal_affected_trades",
        ],
    )

    _write_csv(
        report_dir / "coverage_blocks.csv",
        coverage_blocks,
        ["start_date", "end_date", "calendar_days"],
    )

    coverage_summary = {
        "proxy_file": str(proxy_path.relative_to(root)),
        "proxy_label": args.proxy_label,
        "proxy_bar_count": len(proxy_bars),
        "proxy_start_utc": proxy_bars[0].timestamp.isoformat(),
        "proxy_end_utc": proxy_bars[-1].timestamp.isoformat(),
        "proxy_unique_days": len(proxy_days),
        "spy_unique_days_in_proxy_window": len(relevant_spy_days),
        "proxy_day_coverage_pct_vs_spy": round(
            100.0 * len(proxy_days & relevant_spy_days) / max(len(relevant_spy_days), 1),
            2,
        ),
        "coverage_blocks": coverage_blocks,
        "fresh_same_session_trade_matches": len(fresh_rows),
        "baseline_trade_count": len(trades),
        "fresh_same_session_trade_coverage_pct": round(
            100.0 * len(fresh_rows) / max(len(trades), 1),
            2,
        ),
        "detected_proxy_label": args.proxy_label,
        "fresh_signal_rule": (
            f"session_open == new_york_equity_open and last proxy bar <= {args.fresh_max_minutes} minutes old"
        ),
        "short_ma_bars": args.short_ma_bars,
        "long_ma_bars": args.long_ma_bars,
        "shock_lookback_bars": args.shock_lookback_bars,
        "shock_up_threshold": args.shock_up_threshold,
        "relief_down_threshold": args.relief_down_threshold,
    }

    with (report_dir / "coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage_summary, handle, indent=2)

    print(json.dumps(coverage_summary, indent=2))


if __name__ == "__main__":
    main()
