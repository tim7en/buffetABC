"""
Counterfactual path-shock audit for the current baseline.

Purpose
-------
Apply explicit cross-asset scenario shocks to the underlying cached price bars,
rerun the actual Session Turtle shared-account engine, and measure:
1. overall performance changes,
2. sector/sleeve PnL changes after the shock date,
3. whether the Donchian engine flips into the expected direction, and
4. how long that directional capture takes.

Default scenario pack
---------------------
Hormuz normalization / cross-asset rotation:
- Energy reprices down.
- Technology and Financials reprice lower.
- Gold and Metals reprice higher.
- Crypto is left neutral to isolate the thesis.

Run from repo root:
  python tools/session_turtle_core_x2/run_path_scenario_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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

import edgar.services.session_turtle_portfolio as stp
import edgar.services.session_turtle_trend_strategy as stts
from edgar.services.binance_data import load_local_binance_klines as _orig_load_binance_klines
from edgar.services.local_tiingo_data import load_local_tiingo_klines as _orig_load_tiingo_klines
from edgar.services.session_open_utils import SessionBar
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)
from tools.session_turtle_core_x2.run_selection_edge_audit import (
    _current_baseline_universe,
    _dt,
    _ensure_tiingo_cache_symbols,
    _sector_group,
)


REPORT_DIR = Path("reports/strategy_health_audit/bigtech_energy_base_uncapped_equity")
OUTPUT_DIR = Path("reports/scenario_shock_audit/hormuz_normalization_rotation")
CHART_DIR = OUTPUT_DIR / "charts"

LOOKBACK_YEARS = 4.1
CHANNEL_PERIOD = 20
TREND_FAST = 55
TREND_SLOW = 200
EMA_PERIOD = 200

CURRENT_BASELINE_REPORT_KWARGS = {
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

CURRENT_BASELINE_CANDIDATE_KWARGS = {
    "basket": "core",
    "initial_capital": 1_000.0,
    "lookback_years": LOOKBACK_YEARS,
    "channel_period": CHANNEL_PERIOD,
    "base_risk_pct": 0.05,
    "fixed_stop_pct": 0.10,
    "directional_volume_risk_pct": 0.07,
    "trend_fast_period": TREND_FAST,
    "trend_slow_period": TREND_SLOW,
}

ADDED_EQUITIES = {"GOOGL", "META", "NVDA"}
ADDED_ENERGY = {"BRENT", "NATGAS"}

DARK_BG = "#0f1117"
PANEL_BG = "#171923"
BORDER = "#2b3245"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
BLUE = "#60a5fa"
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


@dataclass(frozen=True)
class ShockLeg:
    end_pct: float
    gap_pct: float

    @property
    def expected_direction(self) -> str:
        return "long" if self.end_pct > 0 else "short"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    label: str
    description: str
    shock_start: date
    shock_end: date
    sector_legs: dict[str, ShockLeg]


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


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


def _scenario_pack(*, shock_start: date, shock_end: date) -> list[ScenarioSpec]:
    targets = {
        "Energy": -0.18,
        "Financials": -0.12,
        "Information Technology": -0.12,
        "Communication Services": -0.10,
        "Consumer Discretionary": -0.08,
        "Gold": 0.12,
        "Metals": 0.10,
    }
    gap_share = {
        "slow": 0.20,
        "base": 0.40,
        "fast": 0.65,
    }
    descriptions = {
        "slow": "Slow normalization: small initial gap, most of the move arrives as drift.",
        "base": "Base normalization: moderate repricing followed by persistent cross-asset rotation.",
        "fast": "Fast normalization: front-loaded repricing with the bulk of the move hitting quickly.",
    }
    scenarios = []
    for name in ("slow", "base", "fast"):
        sector_legs = {
            sector: ShockLeg(end_pct=target, gap_pct=target * gap_share[name])
            for sector, target in targets.items()
        }
        scenarios.append(
            ScenarioSpec(
                name=name,
                label=f"Hormuz normalization ({name})",
                description=descriptions[name],
                shock_start=shock_start,
                shock_end=shock_end,
                sector_legs=sector_legs,
            )
        )
    return scenarios


class _ShockedBarLoader:
    def __init__(self, scenario: ScenarioSpec | None):
        self.scenario = scenario
        self._binance_cache: dict[tuple, tuple[list[dict], str]] = {}
        self._tiingo_cache: dict[tuple, tuple[list[dict], str, str]] = {}

    def _leg_for_ticker(self, ticker: str) -> ShockLeg | None:
        if self.scenario is None:
            return None
        sector = _sector_group(str(ticker))[0]
        return self.scenario.sector_legs.get(sector)

    def _multiplier_for_timestamp(self, ts: datetime, leg: ShockLeg | None) -> float:
        if self.scenario is None or leg is None:
            return 1.0
        if ts.date() < self.scenario.shock_start:
            return 1.0
        if self.scenario.shock_end <= self.scenario.shock_start:
            return 1.0 + leg.end_pct

        start_mult = 1.0 + float(leg.gap_pct)
        end_mult = 1.0 + float(leg.end_pct)
        if start_mult <= 0 or end_mult <= 0:
            raise ValueError("Scenario multipliers must remain positive.")

        start_dt = datetime.combine(self.scenario.shock_start, datetime.min.time())
        end_dt = datetime.combine(self.scenario.shock_end, datetime.max.time())
        if ts >= end_dt:
            return end_mult
        span = max((end_dt - start_dt).total_seconds(), 1.0)
        progress = min(max((ts - start_dt).total_seconds() / span, 0.0), 1.0)
        return start_mult * ((end_mult / start_mult) ** progress)

    def _transform_bars(self, ticker: str, bars: list[dict]) -> list[dict]:
        leg = self._leg_for_ticker(ticker)
        if leg is None:
            return [dict(row) for row in bars]

        out: list[dict] = []
        for row in bars:
            ts = row.get("timestamp") or row.get("ts")
            mult = self._multiplier_for_timestamp(ts, leg)
            new_row = dict(row)
            new_row["open"] = float(row["open"]) * mult
            new_row["high"] = float(row["high"]) * mult
            new_row["low"] = float(row["low"]) * mult
            new_row["close"] = float(row["close"]) * mult
            new_row["high"] = max(new_row["high"], new_row["open"], new_row["close"])
            new_row["low"] = min(new_row["low"], new_row["open"], new_row["close"])
            out.append(new_row)
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
            self._binance_cache[key] = (self._transform_bars(ticker, bars), symbol)
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
            self._tiingo_cache[key] = (self._transform_bars(ticker, bars), symbol, path)
        bars, symbol, path = self._tiingo_cache[key]
        return [dict(row) for row in bars], symbol, path


@contextmanager
def _patched_universe():
    old_core = stp.CORE_SESSION_TURTLE_UNIVERSE
    old_expanded = stp.EXPANDED_SESSION_TURTLE_UNIVERSE
    old_equity = set(stp.EQUITY_TICKERS)
    old_energy = set(stp.ENERGY_TICKERS)

    custom_universe = tuple(_current_baseline_universe())
    stp.CORE_SESSION_TURTLE_UNIVERSE = custom_universe
    stp.EXPANDED_SESSION_TURTLE_UNIVERSE = tuple(list(custom_universe) + list(stp.INDEX_SESSION_TURTLE_UNIVERSE))
    stp.EQUITY_TICKERS = set(old_equity) | ADDED_EQUITIES
    stp.ENERGY_TICKERS = set(old_energy) | ADDED_ENERGY
    try:
        yield list(dict.fromkeys(custom_universe))
    finally:
        stp.CORE_SESSION_TURTLE_UNIVERSE = old_core
        stp.EXPANDED_SESSION_TURTLE_UNIVERSE = old_expanded
        stp.EQUITY_TICKERS = old_equity
        stp.ENERGY_TICKERS = old_energy


@contextmanager
def _patched_loaders(scenario: ScenarioSpec | None):
    loader = _ShockedBarLoader(scenario)
    old_stts_binance = stts.load_local_binance_klines
    old_stts_tiingo = stts.load_local_tiingo_klines
    old_stp_binance = stp.load_local_binance_klines
    old_stp_tiingo = stp.load_local_tiingo_klines
    stts.load_local_binance_klines = loader.load_binance
    stts.load_local_tiingo_klines = loader.load_tiingo
    stp.load_local_binance_klines = loader.load_binance
    stp.load_local_tiingo_klines = loader.load_tiingo
    try:
        yield loader
    finally:
        stts.load_local_binance_klines = old_stts_binance
        stts.load_local_tiingo_klines = old_stts_tiingo
        stp.load_local_binance_klines = old_stp_binance
        stp.load_local_tiingo_klines = old_stp_tiingo


def _load_macro_state(root: Path) -> dict:
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    return build_extended_hours_proxy_state(daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg)


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


def _equity_curve_from_trades(*, trades: list[dict], initial_capital: float, start_date: datetime) -> list[dict]:
    capital = float(initial_capital)
    rows = [{"date": start_date.isoformat(), "equity": round(capital, 4)}]
    for trade in sorted(trades, key=lambda row: _dt(row["exit_ts"])):
        capital += float(trade.get("net_pnl", 0.0) or 0.0)
        rows.append({"date": str(trade["exit_ts"]), "equity": round(capital, 4)})
    return rows


def _post_shock_sector_pnl_rows(trades: list[dict], *, shock_start: date) -> list[dict]:
    grouped: dict[str, float] = {}
    counts: dict[str, int] = {}
    total = 0.0
    total_trades = 0
    for trade in trades:
        exit_day = _dt(trade["exit_ts"]).date()
        if exit_day < shock_start:
            continue
        sector = _sector_group(str(trade["ticker"]))[0]
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        grouped[sector] = grouped.get(sector, 0.0) + pnl
        counts[sector] = counts.get(sector, 0) + 1
        total += pnl
        total_trades += 1

    rows = [
        {
            "sector_group": sector,
            "post_shock_trades": counts.get(sector, 0),
            "post_shock_pnl": round(pnl, 4),
        }
        for sector, pnl in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]
    rows.append(
        {
            "sector_group": "Overall",
            "post_shock_trades": total_trades,
            "post_shock_pnl": round(total, 4),
        }
    )
    return rows


def _first_matching_entry(rows: list[dict], *, direction: str, shock_start: date, field: str) -> str | None:
    matches = []
    for row in rows:
        entry_ts = _dt(row[field])
        if entry_ts.date() >= shock_start and str(row["direction"]) == direction:
            matches.append(entry_ts)
    if not matches:
        return None
    return min(matches).isoformat()


def _days_between(start_day: date, maybe_ts: str | None) -> float | None:
    if maybe_ts is None:
        return None
    return round((_dt(maybe_ts).date() - start_day).days, 2)


def _capture_rows(
    *,
    scenario: ScenarioSpec,
    candidates: list[dict],
    trades: list[dict],
    universe: list[tuple[str, str, str]],
) -> list[dict]:
    tickers = sorted({ticker for ticker, _, _ in universe})
    rows: list[dict] = []
    for ticker in tickers:
        leg = scenario.sector_legs.get(_sector_group(ticker)[0])
        if leg is None:
            continue
        expected_direction = leg.expected_direction
        ticker_candidates = [row for row in candidates if str(row["ticker"]) == ticker]
        ticker_trades = [row for row in trades if str(row["ticker"]) == ticker]
        first_candidate = _first_matching_entry(
            ticker_candidates,
            direction=expected_direction,
            shock_start=scenario.shock_start,
            field="entry_ts",
        )
        first_trade = _first_matching_entry(
            ticker_trades,
            direction=expected_direction,
            shock_start=scenario.shock_start,
            field="entry_ts",
        )
        correct_pnl = sum(
            float(row.get("net_pnl", row.get("pnl", 0.0)) or 0.0)
            for row in ticker_trades
            if _dt(row["entry_ts"]).date() >= scenario.shock_start
            and str(row["direction"]) == expected_direction
        )
        wrong_pnl = sum(
            float(row.get("net_pnl", row.get("pnl", 0.0)) or 0.0)
            for row in ticker_trades
            if _dt(row["entry_ts"]).date() >= scenario.shock_start
            and str(row["direction"]) != expected_direction
        )
        rows.append(
            {
                "ticker": ticker,
                "sector_group": _sector_group(ticker)[0],
                "expected_direction": expected_direction,
                "shock_end_pct": round(leg.end_pct * 100.0, 2),
                "first_correct_candidate_ts": first_candidate,
                "candidate_lag_days": _days_between(scenario.shock_start, first_candidate),
                "first_correct_trade_ts": first_trade,
                "trade_lag_days": _days_between(scenario.shock_start, first_trade),
                "correct_direction_trades": sum(
                    1
                    for row in ticker_trades
                    if _dt(row["entry_ts"]).date() >= scenario.shock_start
                    and str(row["direction"]) == expected_direction
                ),
                "wrong_direction_trades": sum(
                    1
                    for row in ticker_trades
                    if _dt(row["entry_ts"]).date() >= scenario.shock_start
                    and str(row["direction"]) != expected_direction
                ),
                "correct_direction_pnl": round(correct_pnl, 4),
                "wrong_direction_pnl": round(wrong_pnl, 4),
            }
        )
    return rows


def _scenario_comparison_row(
    *,
    scenario_name: str,
    label: str,
    description: str,
    shock_start: date,
    summary: dict,
    post_shock_rows: list[dict],
) -> dict:
    overall_post = next(row for row in post_shock_rows if row["sector_group"] == "Overall")
    return {
        "scenario_name": scenario_name,
        "label": label,
        "description": description,
        "shock_start": shock_start.isoformat(),
        "final_equity": round(float(summary["final_equity"]), 4),
        "total_return_pct": round(float(summary["total_return_pct"]), 2),
        "cagr_pct": round(float(summary["cagr_pct"]), 2),
        "max_realized_drawdown_pct": round(float(summary["max_realized_drawdown_pct"]), 2),
        "profit_factor": round(float(summary["profit_factor"]), 4),
        "win_rate_pct": round(float(summary["win_rate_pct"]), 2),
        "executed_trades": int(summary["executed_trades"]),
        "post_shock_trades": int(overall_post["post_shock_trades"]),
        "post_shock_pnl": round(float(overall_post["post_shock_pnl"]), 4),
    }


def _run_variant(
    *,
    scenario: ScenarioSpec | None,
    label: str,
    description: str,
    root: Path,
    report_summary: dict,
    raw_5m_channels: bool,
) -> dict:
    macro_state = _load_macro_state(root)
    with _patched_universe() as universe:
        with _patched_loaders(scenario):
            with _patched_raw_5m_sessions(raw_5m_channels):
                tech_state = build_per_asset_technical_state(
                    universe=list(dict.fromkeys(universe)),
                    lookback_years=5.0,
                    warmup_days=300,
                    ema_period=EMA_PERIOD,
                    adx_period=14,
                )
                candidate_kwargs = dict(CURRENT_BASELINE_CANDIDATE_KWARGS)
                candidate_kwargs["channel_period"] = int(report_summary.get("channel_period", CHANNEL_PERIOD) or CHANNEL_PERIOD)
                candidates = build_session_turtle_shared_account_candidates(**candidate_kwargs)
                report_kwargs = dict(CURRENT_BASELINE_REPORT_KWARGS)
                report_kwargs["extended_hours_proxy_state"] = macro_state
                report_kwargs["per_asset_technical_state"] = tech_state
                report_kwargs["precomputed_candidates"] = candidates
                result = generate_session_turtle_shared_account_report(**report_kwargs)

    return {
        "name": "baseline" if scenario is None else scenario.name,
        "label": label,
        "description": description,
        "shock_start": None if scenario is None else scenario.shock_start.isoformat(),
        "shock_end": None if scenario is None else scenario.shock_end.isoformat(),
        "summary": dict(result["summary"]),
        "trades": list(result["trades"]),
        "candidates": list(candidates),
        "universe": list(universe),
    }


def _representative_path_rows(
    *,
    scenario: ScenarioSpec,
    tickers: list[str],
) -> list[dict]:
    loader = _ShockedBarLoader(scenario)
    rows: list[dict] = []
    for ticker in tickers:
        source = "binance" if ticker.endswith("-USD") and ticker not in {"GLD", "PPLT", "SLV", "BRENT", "NATGAS", "AMZN", "COIN", "CRCL", "GOOGL", "HOOD", "INTC", "META", "MSTR", "NVDA", "PLTR", "TSLA"} else "tiingo"
        if source == "binance":
            baseline_bars, _ = _orig_load_binance_klines(
                ticker=ticker,
                interval="15m",
                lookback_years=LOOKBACK_YEARS,
                warmup_days=120,
                market_data_symbol=None,
            )
            scenario_bars, _ = loader.load_binance(
                ticker=ticker,
                interval="15m",
                lookback_years=LOOKBACK_YEARS,
                warmup_days=120,
                market_data_symbol=None,
            )
        else:
            baseline_bars, _, _ = _orig_load_tiingo_klines(
                ticker=ticker,
                interval="5m",
                lookback_years=LOOKBACK_YEARS,
                warmup_days=120,
                market_data_symbol=None,
            )
            scenario_bars, _, _ = loader.load_tiingo(
                ticker=ticker,
                interval="5m",
                lookback_years=LOOKBACK_YEARS,
                warmup_days=120,
                market_data_symbol=None,
            )

        daily_baseline: dict[str, float] = {}
        daily_scenario: dict[str, float] = {}
        for bar in baseline_bars:
            ts = bar.get("timestamp") or bar.get("ts")
            day = ts.date().isoformat()
            if day >= scenario.shock_start.isoformat():
                daily_baseline[day] = float(bar["close"])
        for bar in scenario_bars:
            ts = bar.get("timestamp") or bar.get("ts")
            day = ts.date().isoformat()
            if day >= scenario.shock_start.isoformat():
                daily_scenario[day] = float(bar["close"])

        common_days = sorted(set(daily_baseline) & set(daily_scenario))
        if not common_days:
            continue
        base0 = daily_baseline[common_days[0]]
        scen0 = daily_scenario[common_days[0]]
        for day in common_days:
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "baseline_index": round(daily_baseline[day] / base0 * 100.0, 4) if base0 > 0 else None,
                    "scenario_index": round(daily_scenario[day] / scen0 * 100.0, 4) if scen0 > 0 else None,
                }
            )
    return rows


def _plot_equity_comparison(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = {
        "baseline": TEXT,
        "slow": BLUE,
        "base": GREEN,
        "fast": RED,
    }
    for run in runs:
        summary = run["summary"]
        curve = _equity_curve_from_trades(
            trades=run["trades"],
            initial_capital=float(summary["initial_capital"]),
            start_date=_dt(summary["start_date"]),
        )
        dates = [_dt(row["date"]) for row in curve]
        equities = [float(row["equity"]) for row in curve]
        ax.plot(dates, equities, linewidth=2.0, label=run["label"], color=palette.get(run["name"], CYAN))
    ax.set_title("Counterfactual Replay: Equity Curves")
    ax.set_ylabel("Realized Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "01_equity_comparison.png")


def _plot_price_paths(path_rows: list[dict], *, label: str) -> None:
    if not path_rows:
        return
    tickers = sorted({str(row["ticker"]) for row in path_rows})
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, max(8, len(tickers) * 2.4)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        rows = [row for row in path_rows if str(row["ticker"]) == ticker]
        dates = [date.fromisoformat(str(row["date"])) for row in rows]
        ax.plot(dates, [float(row["baseline_index"]) for row in rows], label="Baseline path", color=TEXT, linewidth=1.8)
        ax.plot(dates, [float(row["scenario_index"]) for row in rows], label=label, color=GREEN, linewidth=1.8)
        ax.set_title(ticker)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _savefig(fig, CHART_DIR / "02_representative_price_paths.png")


def _plot_sector_pnl_delta(rows: list[dict], *, label: str) -> None:
    sector_rows = [row for row in rows if row["sector_group"] != "Overall"]
    if not sector_rows:
        return
    sectors = [str(row["sector_group"]) for row in sector_rows]
    baseline = [float(row["baseline_post_shock_pnl"]) for row in sector_rows]
    scenario = [float(row["scenario_post_shock_pnl"]) for row in sector_rows]
    x = list(range(len(sectors)))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar([value - 0.18 for value in x], baseline, width=0.36, color=TEXT, label="Baseline")
    ax.bar([value + 0.18 for value in x], scenario, width=0.36, color=GREEN, label=label)
    ax.axhline(0.0, color=BORDER, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=25, ha="right")
    ax.set_title("Post-Shock PnL by Sector")
    ax.set_ylabel("PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    _savefig(fig, CHART_DIR / "03_sector_pnl_delta.png")


def _plot_capture_lag(rows: list[dict], *, label: str) -> None:
    filtered = [row for row in rows if row["trade_lag_days"] is not None]
    if not filtered:
        return
    ordered = sorted(filtered, key=lambda row: (float(row["trade_lag_days"]), str(row["ticker"])))
    tickers = [str(row["ticker"]) for row in ordered]
    values = [float(row["trade_lag_days"]) for row in ordered]
    colors = [GREEN if str(row["expected_direction"]) == "long" else RED for row in ordered]

    fig, ax = plt.subplots(figsize=(12, max(5, len(tickers) * 0.45)))
    ax.barh(tickers, values, color=colors, alpha=0.9)
    ax.set_title(f"Days To First Correct-Direction Executed Trade ({label})")
    ax.set_xlabel("Calendar days after shock start")
    ax.grid(True, axis="x", alpha=0.3)
    _savefig(fig, CHART_DIR / "04_capture_lag.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run counterfactual cross-asset path shocks against the current baseline.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--shock-start", type=str, default=None, help="Shock start date in YYYY-MM-DD. Defaults to 180 calendar days before report end.")
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
    shock_start = _date(args.shock_start) if args.shock_start else (analysis_end - timedelta(days=180))
    shock_end = analysis_end

    scenarios = _scenario_pack(shock_start=shock_start, shock_end=shock_end)

    print("Running baseline replay...")
    baseline_run = _run_variant(
        scenario=None,
        label="Current baseline replay",
        description="Unshocked replay using the current baseline configuration and the current local caches.",
        root=root,
        report_summary=report_summary,
        raw_5m_channels=raw_5m_channels,
    )

    runs = [baseline_run]
    for scenario in scenarios:
        print(f"Running scenario: {scenario.label}...")
        runs.append(
            _run_variant(
                scenario=scenario,
                label=scenario.label,
                description=scenario.description,
                root=root,
                report_summary=report_summary,
                raw_5m_channels=raw_5m_channels,
            )
        )

    comparison_rows: list[dict] = []
    baseline_post_shock = _post_shock_sector_pnl_rows(baseline_run["trades"], shock_start=shock_start)
    baseline_post_map = {row["sector_group"]: row for row in baseline_post_shock}

    scenario_payloads: list[dict] = []
    base_scenario_capture: list[dict] = []
    base_scenario_sector_compare: list[dict] = []
    for run in runs:
        post_shock_rows = _post_shock_sector_pnl_rows(run["trades"], shock_start=shock_start)
        comparison_rows.append(
            _scenario_comparison_row(
                scenario_name=run["name"],
                label=run["label"],
                description=run["description"],
                shock_start=shock_start,
                summary=run["summary"],
                post_shock_rows=post_shock_rows,
            )
        )
        payload = {
            "name": run["name"],
            "label": run["label"],
            "description": run["description"],
            "summary": run["summary"],
            "post_shock_sector_pnl": post_shock_rows,
        }
        if run["name"] != "baseline":
            scenario_spec = next(item for item in scenarios if item.name == run["name"])
            capture_rows = _capture_rows(
                scenario=scenario_spec,
                candidates=run["candidates"],
                trades=run["trades"],
                universe=run["universe"],
            )
            payload["capture"] = capture_rows
            if run["name"] == "base":
                base_scenario_capture = capture_rows
                scenario_map = {row["sector_group"]: row for row in post_shock_rows}
                sectors = sorted(set(baseline_post_map) | set(scenario_map))
                base_scenario_sector_compare = [
                    {
                        "sector_group": sector,
                        "baseline_post_shock_pnl": round(float(baseline_post_map.get(sector, {}).get("post_shock_pnl", 0.0) or 0.0), 4),
                        "scenario_post_shock_pnl": round(float(scenario_map.get(sector, {}).get("post_shock_pnl", 0.0) or 0.0), 4),
                        "delta_post_shock_pnl": round(
                            float(scenario_map.get(sector, {}).get("post_shock_pnl", 0.0) or 0.0)
                            - float(baseline_post_map.get(sector, {}).get("post_shock_pnl", 0.0) or 0.0),
                            4,
                        ),
                    }
                    for sector in sectors
                ]
        scenario_payloads.append(payload)

    for run in runs:
        subdir = output_dir / run["name"]
        subdir.mkdir(parents=True, exist_ok=True)
        _save_json(subdir / "summary.json", run["summary"])
        _write_csv(subdir / "trades.csv", run["trades"])

    representative_rows = _representative_path_rows(
        scenario=next(item for item in scenarios if item.name == "base"),
        tickers=["BRENT", "NVDA", "HOOD", "GLD", "COPPER"],
    )

    _write_csv(output_dir / "scenario_comparison.csv", comparison_rows)
    _write_csv(output_dir / "baseline_post_shock_sector_pnl.csv", baseline_post_shock)
    _write_csv(output_dir / "base_capture_lag.csv", base_scenario_capture)
    _write_csv(output_dir / "base_sector_pnl_comparison.csv", base_scenario_sector_compare)
    _write_csv(output_dir / "representative_price_paths.csv", representative_rows)
    _save_json(
        output_dir / "summary.json",
        {
            "report_source": str(report_dir.resolve()),
            "shock_start": shock_start.isoformat(),
            "shock_end": shock_end.isoformat(),
            "method": {
                "type": "counterfactual path shock replay",
                "notes": [
                    "Price bars are shocked from the chosen start date onward.",
                    "The actual Session Turtle per-asset Donchian engine is rerun on shocked bars.",
                    "The per-asset EMA overlay is also rebuilt on shocked bars.",
                    "Extended-hours VIX / Crypto Fear & Greed overlay is left unchanged to isolate asset-path effects.",
                ],
            },
            "runs": scenario_payloads,
        },
    )

    _plot_equity_comparison(runs)
    _plot_price_paths(representative_rows, label="Hormuz normalization (base)")
    _plot_sector_pnl_delta(base_scenario_sector_compare, label="Hormuz normalization (base)")
    _plot_capture_lag(base_scenario_capture, label="Hormuz normalization (base)")

    print("\nSCENARIO SHOCK AUDIT")
    print("=" * 96)
    print(f"Shock window: {shock_start.isoformat()} -> {shock_end.isoformat()}")
    for row in comparison_rows:
        print(
            f"{row['label']:<32}  "
            f"Ret {row['total_return_pct']:>8.2f}%  "
            f"MaxDD {row['max_realized_drawdown_pct']:>6.2f}%  "
            f"PF {row['profit_factor']:>5.2f}  "
            f"Post-shock PnL {row['post_shock_pnl']:>10.2f}"
        )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
