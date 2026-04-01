"""
Selection-edge audit for the current shared-account baseline.

Purpose
-------
Assess whether the latest baseline result is mostly explained by:
1. broad market/sector drift,
2. a concentrated set of highly correlated winners, or
3. actual trade-level edge over the same assets and sectors.

Default input is the latest saved baseline report:
  reports/strategy_health_audit/bigtech_energy_base_uncapped_equity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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
    build_per_asset_technical_state,
)


REPORT_DIR = Path("reports/strategy_health_audit/bigtech_energy_base_uncapped_equity")
OUTPUT_DIR = Path("reports/selection_edge_audit/current_baseline")
CHART_DIR = OUTPUT_DIR / "charts"

PRIMARY_TIINGO_DIR = Path("cache/cache/tiingo")
FALLBACK_TIINGO_DIR = Path("cache/cache/cache")

ADDED_EQUITIES = ("GOOGL", "META", "NVDA")
ADDED_ENERGY = ("BRENT", "NATGAS")
ADDED_TIINGO_SYMBOLS = ADDED_EQUITIES + ADDED_ENERGY

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
GOLD_TICKERS = {"PAXG-USD", "GLD"}
METALS_TICKERS = {"COPPER", "PPLT", "SLV"}
ENERGY_TICKERS = {"BRENT", "NATGAS"}
EQUITY_TICKERS = {
    "AMZN",
    "COIN",
    "CRCL",
    "GOOGL",
    "HOOD",
    "INTC",
    "META",
    "MSTR",
    "NVDA",
    "PLTR",
    "TSLA",
}

EQUITY_SECTOR_MAP = {
    "AMZN": ("Consumer Discretionary", "Broadline Retail"),
    "COIN": ("Financials", "Capital Markets"),
    "CRCL": ("Financials", "Transaction & Payment Processing"),
    "GOOGL": ("Communication Services", "Interactive Media & Services"),
    "HOOD": ("Financials", "Brokerage & Capital Markets"),
    "INTC": ("Information Technology", "Semiconductors"),
    "META": ("Communication Services", "Interactive Media & Services"),
    "MSTR": ("Information Technology", "Application Software"),
    "NVDA": ("Information Technology", "Semiconductors"),
    "PLTR": ("Information Technology", "Application Software"),
    "TSLA": ("Consumer Discretionary", "Automobiles"),
}

SECTOR_COLORS = {
    "Crypto": "#f59e0b",
    "Gold": "#eab308",
    "Metals": "#94a3b8",
    "Energy": "#ef4444",
    "Consumer Discretionary": "#22c55e",
    "Financials": "#06b6d4",
    "Communication Services": "#8b5cf6",
    "Information Technology": "#3b82f6",
    "Market": "#e2e8f0",
}

DARK_BG = "#0f1117"
PANEL_BG = "#171923"
BORDER = "#2b3245"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"

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


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _ensure_tiingo_cache_symbols() -> None:
    PRIMARY_TIINGO_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in ADDED_TIINGO_SYMBOLS:
        dst = PRIMARY_TIINGO_DIR / f"{symbol}_5m.parquet"
        if dst.exists():
            continue
        src = FALLBACK_TIINGO_DIR / f"{symbol}_5m.parquet"
        if not src.exists():
            raise FileNotFoundError(
                f"Missing local Tiingo parquet for {symbol}. Checked {src} and {dst}."
            )
        shutil.copy2(src, dst)


def _current_baseline_universe() -> tuple[tuple[str, str, str], ...]:
    base = list(CORE_SESSION_TURTLE_UNIVERSE)
    extra = [(ticker, "tiingo", "new_york_equity_open") for ticker in ADDED_TIINGO_SYMBOLS]
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for row in [*base, *extra]:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return tuple(out)


def _asset_bucket(ticker: str) -> str:
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in GOLD_TICKERS:
        return "gold"
    if ticker in METALS_TICKERS:
        return "metals"
    if ticker in ENERGY_TICKERS:
        return "energy"
    if ticker in EQUITY_TICKERS:
        return "equity"
    return "other"


def _sector_group(ticker: str) -> tuple[str, str | None]:
    bucket = _asset_bucket(ticker)
    if bucket == "crypto":
        return "Crypto", None
    if bucket == "gold":
        return "Gold", None
    if bucket == "metals":
        return "Metals", None
    if bucket == "energy":
        return "Energy", None
    return EQUITY_SECTOR_MAP.get(ticker, ("Other Equity", None))


def _load_report(report_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    with (report_dir / "trades.csv").open(encoding="utf-8", newline="") as fh:
        trades = list(csv.DictReader(fh))
    return summary, trades


def _build_daily_close_state(
    technical_state: dict,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, float]] = {}
    for ticker, rows in (technical_state.get("daily_ema", {}) if technical_state else {}).items():
        mapped: dict[date, float] = {}
        for day_str, payload in rows.items():
            day = _date(day_str)
            if start_date <= day <= end_date:
                mapped[day] = float(payload.get("close", 0.0) or 0.0)
        if mapped:
            out[str(ticker)] = dict(sorted(mapped.items()))
    return out


def _build_daily_return_state(
    daily_close_state: dict[str, dict[date, float]]
) -> dict[str, dict[date, float]]:
    out: dict[str, dict[date, float]] = {}
    for ticker, rows in daily_close_state.items():
        prev_close: float | None = None
        returns: dict[date, float] = {}
        for day, close in sorted(rows.items()):
            if prev_close is not None and prev_close > 0:
                returns[day] = (close / prev_close) - 1.0
            prev_close = close
        if returns:
            out[ticker] = returns
    return out


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


def _compound_returns(values: list[float]) -> float:
    total = 1.0
    for value in values:
        total *= 1.0 + value
    return total - 1.0


def _build_asset_metadata(universe: tuple[tuple[str, str, str], ...]) -> list[dict]:
    source_by_ticker: dict[str, str] = {}
    session_count: dict[str, int] = defaultdict(int)
    for ticker, source, _session in universe:
        source_by_ticker[ticker] = source
        session_count[ticker] += 1

    rows = []
    for ticker in sorted(source_by_ticker):
        sector, sub_sector = _sector_group(ticker)
        rows.append(
            {
                "ticker": ticker,
                "source": source_by_ticker[ticker],
                "session_count": session_count[ticker],
                "asset_bucket": _asset_bucket(ticker),
                "sector_group": sector,
                "sub_sector": sub_sector or "",
            }
        )
    return rows


def _pairwise_asset_correlations(
    daily_return_state: dict[str, dict[date, float]],
    metadata_by_ticker: dict[str, dict],
) -> list[dict]:
    tickers = sorted(daily_return_state)
    rows = []
    for idx, left in enumerate(tickers):
        for right in tickers[idx + 1 :]:
            left_series = daily_return_state[left]
            right_series = daily_return_state[right]
            common_days = sorted(left_series.keys() & right_series.keys())
            corr = None
            if len(common_days) >= 20:
                corr = _pearson_corr(
                    [left_series[day] for day in common_days],
                    [right_series[day] for day in common_days],
                )
            rows.append(
                {
                    "left_ticker": left,
                    "right_ticker": right,
                    "left_sector": metadata_by_ticker[left]["sector_group"],
                    "right_sector": metadata_by_ticker[right]["sector_group"],
                    "left_bucket": metadata_by_ticker[left]["asset_bucket"],
                    "right_bucket": metadata_by_ticker[right]["asset_bucket"],
                    "same_sector": metadata_by_ticker[left]["sector_group"] == metadata_by_ticker[right]["sector_group"],
                    "same_bucket": metadata_by_ticker[left]["asset_bucket"] == metadata_by_ticker[right]["asset_bucket"],
                    "corr": round(corr, 4) if corr is not None else None,
                    "abs_corr": round(abs(corr), 4) if corr is not None else None,
                    "obs": len(common_days),
                }
            )
    return rows


def _correlation_matrix(labels: list[str], pairs: list[dict], left_key: str, right_key: str) -> list[list[str | float]]:
    lookup: dict[tuple[str, str], float | None] = {}
    for row in pairs:
        value = row["corr"]
        lookup[(row[left_key], row[right_key])] = value
        lookup[(row[right_key], row[left_key])] = value

    table: list[list[str | float]] = [["label", *labels]]
    for left in labels:
        row: list[str | float] = [left]
        for right in labels:
            if left == right:
                row.append(1.0)
            else:
                row.append(lookup.get((left, right)))
        table.append(row)
    return table


def _sector_return_state(
    daily_return_state: dict[str, dict[date, float]],
    metadata_by_ticker: dict[str, dict],
) -> tuple[dict[str, dict[date, float]], dict[date, int]]:
    sector_daily: dict[str, dict[date, list[float]]] = defaultdict(lambda: defaultdict(list))
    market_daily: dict[date, list[float]] = defaultdict(list)
    market_count: dict[date, int] = defaultdict(int)

    for ticker, rows in daily_return_state.items():
        sector = metadata_by_ticker[ticker]["sector_group"]
        for day, ret in rows.items():
            sector_daily[sector][day].append(ret)
            market_daily[day].append(ret)
            market_count[day] += 1

    out: dict[str, dict[date, float]] = {}
    for sector, rows in sector_daily.items():
        out[sector] = {
            day: sum(values) / len(values)
            for day, values in sorted(rows.items())
            if values
        }
    out["Market"] = {
        day: sum(values) / len(values)
        for day, values in sorted(market_daily.items())
        if values
    }
    return out, dict(sorted(market_count.items()))


def _sector_member_count_state(
    daily_return_state: dict[str, dict[date, float]],
    metadata_by_ticker: dict[str, dict],
) -> dict[str, dict[date, int]]:
    out: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    for ticker, rows in daily_return_state.items():
        sector = metadata_by_ticker[ticker]["sector_group"]
        for day in rows:
            out[sector][day] += 1
    return {sector: dict(sorted(rows.items())) for sector, rows in sorted(out.items())}


def _performance_summary(return_state: dict[str, dict[date, float]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label, rows in sorted(return_state.items()):
        ordered = sorted(rows.items())
        if not ordered:
            continue
        dates = [day for day, _ in ordered]
        returns = [value for _, value in ordered]
        total_return = _compound_returns(returns)
        year_counts = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
        cagr = (1.0 + total_return) ** (1.0 / year_counts) - 1.0 if total_return > -1.0 else None
        avg_daily = sum(returns) / len(returns)
        daily_vol = math.sqrt(
            sum((value - avg_daily) ** 2 for value in returns) / max(len(returns) - 1, 1)
        )
        out[label] = {
            "first_day": dates[0].isoformat(),
            "last_day": dates[-1].isoformat(),
            "observations": len(returns),
            "total_return_pct": round(total_return * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2) if cagr is not None else None,
            "annualized_vol_pct": round(daily_vol * math.sqrt(252.0) * 100.0, 2),
        }
    return out


def _yearly_return_rows(return_state: dict[str, dict[date, float]]) -> list[dict]:
    rows = []
    for label, series in sorted(return_state.items()):
        grouped: dict[int, list[float]] = defaultdict(list)
        for day, ret in series.items():
            grouped[day.year].append(ret)
        for year, returns in sorted(grouped.items()):
            rows.append(
                {
                    "label": label,
                    "year": year,
                    "return_pct": round(_compound_returns(returns) * 100.0, 2),
                    "observations": len(returns),
                }
            )
    return rows


def _daily_path_rows(
    return_state: dict[str, dict[date, float]],
    sector_member_counts: dict[str, dict[date, int]],
    market_member_count: dict[date, int],
) -> list[dict]:
    labels = sorted(return_state)
    cumulative: dict[str, float] = {label: 1.0 for label in labels}
    all_days = sorted({day for rows in return_state.values() for day in rows})
    out = []
    for day in all_days:
        row = {"date": day.isoformat()}
        for label in labels:
            ret = return_state[label].get(day)
            if ret is not None:
                cumulative[label] *= 1.0 + ret
            row[f"{label}_daily_return_pct"] = round(ret * 100.0, 4) if ret is not None else None
            row[f"{label}_cumulative_return_pct"] = round((cumulative[label] - 1.0) * 100.0, 4)
            if label == "Market":
                row[f"{label}_member_count"] = market_member_count.get(day, 0)
            else:
                row[f"{label}_member_count"] = sector_member_counts.get(label, {}).get(day, 0)
        out.append(row)
    return out


def _rolling_market_stats(
    daily_return_state: dict[str, dict[date, float]],
    metadata_by_ticker: dict[str, dict],
    *,
    window: int = 63,
) -> list[dict]:
    all_days = sorted({day for rows in daily_return_state.values() for day in rows})
    out = []
    for idx in range(window - 1, len(all_days)):
        window_days = all_days[idx + 1 - window : idx + 1]
        pair_corrs: list[float] = []
        within_sector_corrs: list[float] = []
        cross_sector_corrs: list[float] = []
        dispersion_values: list[float] = []

        tickers = sorted(daily_return_state)
        for left_idx, left in enumerate(tickers):
            for right in tickers[left_idx + 1 :]:
                common = [
                    day for day in window_days
                    if day in daily_return_state[left] and day in daily_return_state[right]
                ]
                if len(common) < max(20, window // 2):
                    continue
                corr = _pearson_corr(
                    [daily_return_state[left][day] for day in common],
                    [daily_return_state[right][day] for day in common],
                )
                if corr is None:
                    continue
                pair_corrs.append(corr)
                if metadata_by_ticker[left]["sector_group"] == metadata_by_ticker[right]["sector_group"]:
                    within_sector_corrs.append(corr)
                else:
                    cross_sector_corrs.append(corr)

        for day in window_days:
            observed = [rows[day] for rows in daily_return_state.values() if day in rows]
            if len(observed) >= 2:
                avg = sum(observed) / len(observed)
                variance = sum((value - avg) ** 2 for value in observed) / (len(observed) - 1)
                dispersion_values.append(math.sqrt(max(variance, 0.0)))

        out.append(
            {
                "date": all_days[idx].isoformat(),
                "window_days": window,
                "avg_pairwise_corr": round(sum(pair_corrs) / len(pair_corrs), 4) if pair_corrs else None,
                "avg_within_sector_corr": round(sum(within_sector_corrs) / len(within_sector_corrs), 4)
                if within_sector_corrs else None,
                "avg_cross_sector_corr": round(sum(cross_sector_corrs) / len(cross_sector_corrs), 4)
                if cross_sector_corrs else None,
                "avg_cross_sectional_dispersion": round(
                    sum(dispersion_values) / len(dispersion_values), 6
                ) if dispersion_values else None,
            }
        )
    return out


def _trade_period_return(
    return_series: dict[date, float],
    *,
    entry_day: date,
    exit_day: date,
) -> float | None:
    relevant = [ret for day, ret in sorted(return_series.items()) if entry_day < day <= exit_day]
    if not relevant:
        return 0.0
    return _compound_returns(relevant)


def _trade_edge_rows(
    trades: list[dict],
    metadata_by_ticker: dict[str, dict],
    sector_return_state: dict[str, dict[date, float]],
    market_return_state: dict[str, dict[date, float]],
) -> list[dict]:
    out = []
    for trade in trades:
        ticker = str(trade["ticker"])
        sector = metadata_by_ticker[ticker]["sector_group"]
        direction = str(trade["direction"])
        entry_ts = _dt(trade["entry_ts"])
        exit_ts = _dt(trade["exit_ts"])
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        notional = float(trade.get("notional", 0.0) or 0.0)
        net_pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        strategy_return = (net_pnl / notional) if notional > 0 else 0.0

        raw_asset_move = (exit_price / entry_price) - 1.0 if entry_price > 0 else 0.0
        direction_sign = 1.0 if direction == "long" else -1.0
        asset_benchmark_return = raw_asset_move * direction_sign

        sector_return = _trade_period_return(
            sector_return_state.get(sector, {}),
            entry_day=entry_ts.date(),
            exit_day=exit_ts.date(),
        )
        market_return = _trade_period_return(
            market_return_state,
            entry_day=entry_ts.date(),
            exit_day=exit_ts.date(),
        )
        sector_benchmark_return = (sector_return or 0.0) * direction_sign
        market_benchmark_return = (market_return or 0.0) * direction_sign

        out.append(
            {
                "ticker": ticker,
                "asset_bucket": metadata_by_ticker[ticker]["asset_bucket"],
                "sector_group": sector,
                "direction": direction,
                "entry_ts": entry_ts.isoformat(),
                "exit_ts": exit_ts.isoformat(),
                "holding_days": (exit_ts.date() - entry_ts.date()).days,
                "notional": round(notional, 4),
                "net_pnl": round(net_pnl, 4),
                "strategy_return_pct": round(strategy_return * 100.0, 4),
                "asset_benchmark_return_pct": round(asset_benchmark_return * 100.0, 4),
                "sector_benchmark_return_pct": round(sector_benchmark_return * 100.0, 4),
                "market_benchmark_return_pct": round(market_benchmark_return * 100.0, 4),
                "edge_vs_asset_pct": round((strategy_return - asset_benchmark_return) * 100.0, 4),
                "edge_vs_sector_pct": round((strategy_return - sector_benchmark_return) * 100.0, 4),
                "edge_vs_market_pct": round((strategy_return - market_benchmark_return) * 100.0, 4),
                "synthetic_asset_pnl": round(asset_benchmark_return * notional, 4),
                "synthetic_sector_pnl": round(sector_benchmark_return * notional, 4),
                "synthetic_market_pnl": round(market_benchmark_return * notional, 4),
            }
        )
    return out


def _aggregate_trade_edge(rows: list[dict], group_key: str | None = None) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    if group_key is None:
        grouped["overall"] = rows
    else:
        for row in rows:
            grouped[str(row[group_key])].append(row)

    out = []
    for group, items in sorted(grouped.items()):
        strategy_returns = [float(item["strategy_return_pct"]) for item in items]
        edge_asset = [float(item["edge_vs_asset_pct"]) for item in items]
        edge_sector = [float(item["edge_vs_sector_pct"]) for item in items]
        edge_market = [float(item["edge_vs_market_pct"]) for item in items]
        net_pnl = sum(float(item["net_pnl"]) for item in items)
        synthetic_asset = sum(float(item["synthetic_asset_pnl"]) for item in items)
        synthetic_sector = sum(float(item["synthetic_sector_pnl"]) for item in items)
        synthetic_market = sum(float(item["synthetic_market_pnl"]) for item in items)
        out.append(
            {
                "group": group,
                "trades": len(items),
                "avg_strategy_return_pct": round(sum(strategy_returns) / len(strategy_returns), 4),
                "avg_edge_vs_asset_pct": round(sum(edge_asset) / len(edge_asset), 4),
                "avg_edge_vs_sector_pct": round(sum(edge_sector) / len(edge_sector), 4),
                "avg_edge_vs_market_pct": round(sum(edge_market) / len(edge_market), 4),
                "positive_edge_vs_asset_rate_pct": round(
                    sum(1 for value in edge_asset if value > 0) / len(edge_asset) * 100.0,
                    2,
                ),
                "positive_edge_vs_sector_rate_pct": round(
                    sum(1 for value in edge_sector if value > 0) / len(edge_sector) * 100.0,
                    2,
                ),
                "net_pnl": round(net_pnl, 4),
                "synthetic_asset_pnl": round(synthetic_asset, 4),
                "synthetic_sector_pnl": round(synthetic_sector, 4),
                "synthetic_market_pnl": round(synthetic_market, 4),
                "excess_vs_asset_pnl": round(net_pnl - synthetic_asset, 4),
                "excess_vs_sector_pnl": round(net_pnl - synthetic_sector, 4),
                "excess_vs_market_pnl": round(net_pnl - synthetic_market, 4),
            }
        )
    return out


def _cumulative_trade_edge_rows(trade_edge_rows: list[dict]) -> list[dict]:
    cumulative_strategy = 0.0
    cumulative_asset = 0.0
    cumulative_sector = 0.0
    cumulative_market = 0.0
    out = []
    for row in sorted(trade_edge_rows, key=lambda item: item["exit_ts"]):
        cumulative_strategy += float(row["net_pnl"])
        cumulative_asset += float(row["synthetic_asset_pnl"])
        cumulative_sector += float(row["synthetic_sector_pnl"])
        cumulative_market += float(row["synthetic_market_pnl"])
        out.append(
            {
                "exit_ts": row["exit_ts"],
                "strategy_cumulative_pnl": round(cumulative_strategy, 4),
                "asset_benchmark_cumulative_pnl": round(cumulative_asset, 4),
                "sector_benchmark_cumulative_pnl": round(cumulative_sector, 4),
                "market_benchmark_cumulative_pnl": round(cumulative_market, 4),
                "edge_vs_asset_cumulative_pnl": round(cumulative_strategy - cumulative_asset, 4),
                "edge_vs_sector_cumulative_pnl": round(cumulative_strategy - cumulative_sector, 4),
                "edge_vs_market_cumulative_pnl": round(cumulative_strategy - cumulative_market, 4),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(path: Path, matrix: list[list[str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(matrix)


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def _plot_correlation_heatmap(matrix: list[list[str | float]], path: Path, title: str) -> None:
    labels = matrix[0][1:]
    values = [[0.0 if cell is None else float(cell) for cell in row[1:]] for row in matrix[1:]]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), max(6, len(labels) * 0.5)))
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = values[i][j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=TEXT if abs(value) < 0.65 else "#0f1117",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, path)


def _plot_sector_performance(daily_path_rows: list[dict], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = [datetime.fromisoformat(row["date"]) for row in daily_path_rows]
    for label in labels:
        values = [row[f"{label}_cumulative_return_pct"] for row in daily_path_rows]
        ax.plot(
            dates,
            values,
            linewidth=2.0 if label == "Market" else 1.7,
            label=label,
            color=SECTOR_COLORS.get(label),
        )
    ax.set_title("Passive Sector/Market Performance Over Time")
    ax.set_ylabel("Cumulative return %")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    _savefig(fig, path)


def _plot_rolling_market_stats(rows: list[dict], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    dates = [datetime.fromisoformat(row["date"]) for row in rows]
    avg_corr = [row["avg_pairwise_corr"] if row["avg_pairwise_corr"] is not None else float("nan") for row in rows]
    cross_corr = [
        row["avg_cross_sector_corr"] if row["avg_cross_sector_corr"] is not None else float("nan")
        for row in rows
    ]
    dispersion = [
        row["avg_cross_sectional_dispersion"] if row["avg_cross_sectional_dispersion"] is not None else float("nan")
        for row in rows
    ]

    ax1.plot(dates, avg_corr, linewidth=1.8, color="#60a5fa", label="All pairs")
    ax1.plot(dates, cross_corr, linewidth=1.6, color="#c084fc", label="Cross-sector")
    ax1.set_title("Rolling 63-Day Average Correlation")
    ax1.set_ylabel("Correlation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(dates, dispersion, linewidth=1.8, color="#22c55e")
    ax2.set_title("Rolling 63-Day Cross-Sectional Dispersion")
    ax2.set_ylabel("Std. dev. of daily returns")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _savefig(fig, path)


def _plot_cumulative_trade_edge(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = [datetime.fromisoformat(row["exit_ts"]) for row in rows]
    ax.plot(dates, [row["strategy_cumulative_pnl"] for row in rows], linewidth=2.2, label="Strategy", color="#e2e8f0")
    ax.plot(
        dates,
        [row["asset_benchmark_cumulative_pnl"] for row in rows],
        linewidth=1.7,
        label="Underlying move benchmark",
        color="#60a5fa",
    )
    ax.plot(
        dates,
        [row["sector_benchmark_cumulative_pnl"] for row in rows],
        linewidth=1.7,
        label="Sector benchmark",
        color="#22c55e",
    )
    ax.plot(
        dates,
        [row["market_benchmark_cumulative_pnl"] for row in rows],
        linewidth=1.7,
        label="Market benchmark",
        color="#f59e0b",
    )
    ax.set_title("Cumulative Realized PnL vs Trade-Window Benchmarks")
    ax.set_ylabel("PnL ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    _savefig(fig, path)


def _print_console_summary(summary_payload: dict, *, output_dir: Path) -> None:
    print("\n" + "=" * 96)
    print("SELECTION-EDGE AUDIT")
    print("=" * 96)
    print(
        "Analysis window: "
        f"{summary_payload['analysis_window']['start_date']} -> {summary_payload['analysis_window']['end_date']}"
    )
    print(
        "Universe: "
        f"{summary_payload['universe']['unique_tickers']} tickers / "
        f"{summary_payload['universe']['session_entries']} session entries"
    )
    print(
        "Correlation summary: "
        f"within-sector={summary_payload['correlation']['avg_within_sector_corr']}  "
        f"cross-sector={summary_payload['correlation']['avg_cross_sector_corr']}"
    )
    print(
        "Passive market: "
        f"{summary_payload['market_performance']['Market']['total_return_pct']}% total / "
        f"{summary_payload['market_performance']['Market']['cagr_pct']}% CAGR"
    )
    edge = summary_payload["trade_edge"]["overall"]
    print(
        "Trade edge: "
        f"strategy pnl={edge['net_pnl']:.2f}  "
        f"vs asset benchmark={edge['synthetic_asset_pnl']:.2f}  "
        f"vs sector benchmark={edge['synthetic_sector_pnl']:.2f}  "
        f"vs market benchmark={edge['synthetic_market_pnl']:.2f}"
    )
    print(
        "Excess edge: "
        f"asset={edge['excess_vs_asset_pnl']:.2f}  "
        f"sector={edge['excess_vs_sector_pnl']:.2f}  "
        f"market={edge['excess_vs_market_pnl']:.2f}"
    )
    print(f"Outputs saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess selection edge for the latest baseline report.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lookback-years", type=float, default=5.0)
    parser.add_argument("--warmup-days", type=int, default=300)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve()

    summary, trades = _load_report(report_dir)

    _ensure_tiingo_cache_symbols()
    universe = _current_baseline_universe()
    metadata_rows = _build_asset_metadata(universe)
    metadata_by_ticker = {row["ticker"]: row for row in metadata_rows}

    start_date = _dt(summary["start_date"]).date()
    end_date = _dt(summary["end_date"]).date()

    print("Building daily market state for current baseline universe...")
    technical_state = build_per_asset_technical_state(
        universe=list(universe),
        lookback_years=float(args.lookback_years),
        warmup_days=int(args.warmup_days),
        ema_period=200,
        adx_period=14,
    )

    daily_close_state = _build_daily_close_state(
        technical_state,
        start_date=start_date,
        end_date=end_date,
    )
    daily_return_state = _build_daily_return_state(daily_close_state)

    asset_corr_rows = _pairwise_asset_correlations(daily_return_state, metadata_by_ticker)
    asset_labels = sorted(daily_return_state)
    asset_corr_matrix = _correlation_matrix(
        labels=asset_labels,
        pairs=asset_corr_rows,
        left_key="left_ticker",
        right_key="right_ticker",
    )

    sector_return_state, market_member_count = _sector_return_state(daily_return_state, metadata_by_ticker)
    sector_member_counts = _sector_member_count_state(daily_return_state, metadata_by_ticker)
    sector_corr_rows = _pairwise_asset_correlations(sector_return_state, {
        label: {"sector_group": label, "asset_bucket": label}
        for label in sector_return_state
    })
    sector_labels = sorted(sector_return_state)
    sector_corr_matrix = _correlation_matrix(
        labels=sector_labels,
        pairs=sector_corr_rows,
        left_key="left_ticker",
        right_key="right_ticker",
    )

    daily_path_rows = _daily_path_rows(
        sector_return_state,
        sector_member_counts,
        market_member_count,
    )
    yearly_return_rows = _yearly_return_rows(sector_return_state)
    rolling_stats_rows = _rolling_market_stats(daily_return_state, metadata_by_ticker, window=63)
    trade_edge_rows = _trade_edge_rows(
        trades,
        metadata_by_ticker,
        sector_return_state,
        sector_return_state["Market"],
    )
    trade_edge_cumulative_rows = _cumulative_trade_edge_rows(trade_edge_rows)

    overall_edge = _aggregate_trade_edge(trade_edge_rows, None)[0]
    edge_by_sector = _aggregate_trade_edge(trade_edge_rows, "sector_group")
    edge_by_bucket = _aggregate_trade_edge(trade_edge_rows, "asset_bucket")
    edge_by_ticker = _aggregate_trade_edge(trade_edge_rows, "ticker")

    valid_asset_corrs = [row["corr"] for row in asset_corr_rows if row["corr"] is not None]
    within_sector_corrs = [row["corr"] for row in asset_corr_rows if row["corr"] is not None and row["same_sector"]]
    cross_sector_corrs = [row["corr"] for row in asset_corr_rows if row["corr"] is not None and not row["same_sector"]]

    top_corr_pairs = sorted(
        [row for row in asset_corr_rows if row["corr"] is not None],
        key=lambda row: row["corr"],
        reverse=True,
    )[:10]
    least_corr_pairs = sorted(
        [row for row in asset_corr_rows if row["corr"] is not None],
        key=lambda row: row["corr"],
    )[:10]

    summary_payload = {
        "analysis_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "report_source": str(report_dir),
        "universe": {
            "session_entries": len(universe),
            "unique_tickers": len(metadata_rows),
            "asset_buckets": {
                bucket: sum(1 for row in metadata_rows if row["asset_bucket"] == bucket)
                for bucket in sorted({row["asset_bucket"] for row in metadata_rows})
            },
            "sector_groups": {
                sector: sum(1 for row in metadata_rows if row["sector_group"] == sector)
                for sector in sorted({row["sector_group"] for row in metadata_rows})
            },
        },
        "correlation": {
            "avg_pairwise_corr": round(sum(valid_asset_corrs) / len(valid_asset_corrs), 4) if valid_asset_corrs else None,
            "avg_within_sector_corr": round(sum(within_sector_corrs) / len(within_sector_corrs), 4)
            if within_sector_corrs else None,
            "avg_cross_sector_corr": round(sum(cross_sector_corrs) / len(cross_sector_corrs), 4)
            if cross_sector_corrs else None,
            "top_positive_pairs": top_corr_pairs,
            "lowest_pairs": least_corr_pairs,
        },
        "market_performance": _performance_summary(sector_return_state),
        "trade_edge": {
            "overall": overall_edge,
            "by_sector": edge_by_sector,
            "by_bucket": edge_by_bucket,
            "by_ticker": edge_by_ticker,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    _save_json(output_dir / "summary.json", summary_payload)
    _write_csv(output_dir / "asset_metadata.csv", metadata_rows)
    _write_csv(output_dir / "asset_pairwise_correlations.csv", asset_corr_rows)
    _write_matrix_csv(output_dir / "asset_correlation_matrix.csv", asset_corr_matrix)
    _write_csv(output_dir / "sector_pairwise_correlations.csv", sector_corr_rows)
    _write_matrix_csv(output_dir / "sector_correlation_matrix.csv", sector_corr_matrix)
    _write_csv(output_dir / "sector_daily_paths.csv", daily_path_rows)
    _write_csv(output_dir / "sector_yearly_returns.csv", yearly_return_rows)
    _write_csv(output_dir / "rolling_market_stats.csv", rolling_stats_rows)
    _write_csv(output_dir / "trade_edge.csv", trade_edge_rows)
    _write_csv(output_dir / "trade_edge_cumulative.csv", trade_edge_cumulative_rows)
    _write_csv(output_dir / "trade_edge_by_sector.csv", edge_by_sector)
    _write_csv(output_dir / "trade_edge_by_bucket.csv", edge_by_bucket)
    _write_csv(output_dir / "trade_edge_by_ticker.csv", edge_by_ticker)

    _plot_correlation_heatmap(
        asset_corr_matrix,
        chart_dir / "01_asset_correlation_heatmap.png",
        "Asset Return Correlation",
    )
    _plot_correlation_heatmap(
        sector_corr_matrix,
        chart_dir / "02_sector_correlation_heatmap.png",
        "Sector / Sleeve Return Correlation",
    )
    _plot_sector_performance(
        daily_path_rows,
        labels=[label for label in sector_labels if label != "Market"] + ["Market"],
        path=chart_dir / "03_sector_performance.png",
    )
    _plot_rolling_market_stats(
        rolling_stats_rows,
        chart_dir / "04_rolling_market_stats.png",
    )
    _plot_cumulative_trade_edge(
        trade_edge_cumulative_rows,
        chart_dir / "05_cumulative_trade_edge.png",
    )

    _print_console_summary(summary_payload, output_dir=output_dir)


if __name__ == "__main__":
    main()
