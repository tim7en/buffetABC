"""
Session Turtle Portfolio — Pruned & Grouped (21 symbols, per-group channel periods).

This module extends the x3-migrated portfolio allocator with:
  - Pruned universe: 21 symbols (removed AAPL, AMZN, NVDA, SPY, QQQ, EWJ)
  - Per-group Donchian channel periods:
      Group A (10/5): Commodities + High-Beta Equities
      Group B (20/10): Crypto + Mega-Cap Equities
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest


log = logging.getLogger("session_turtle_portfolio_pruned")


# ── Per-group channel period configuration ─────────────────────────────────────
GROUP_A_CHANNEL = 10   # entry channel
GROUP_A_EXIT    = 5    # exit channel
GROUP_B_CHANNEL = 20   # entry channel
GROUP_B_EXIT    = 10   # exit channel


# Group A: Fast Breakout — Commodities + High-Beta Equities (10/5)
GROUP_A_TICKERS = {
    # Commodities (engine tickers)
    "BRENT", "NATGAS-USD", "COPPER-USD", "XPD-USD", "PPLT", "SLV",
    # High-Beta Equities
    "COIN", "CRCL", "HOOD", "INTC", "MSTR", "PLTR", "TSLA",
}

# Group B: Slow Breakout — Crypto + Mega-Cap Equities (20/10)
GROUP_B_TICKERS = {
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Gold (crypto-adjacent, 24/7)
    "PAXG-USD",
    # Mega-Cap Equities
    "GOOGL", "META", "TSM", "EWY",
}


def _channel_periods_for_ticker(ticker: str) -> tuple[int, int]:
    """Return (channel_period, exit_channel_period) for the given ticker."""
    if ticker in GROUP_A_TICKERS:
        return GROUP_A_CHANNEL, GROUP_A_EXIT
    if ticker in GROUP_B_TICKERS:
        return GROUP_B_CHANNEL, GROUP_B_EXIT
    # Fallback: Group A parameters (faster breakout)
    return GROUP_A_CHANNEL, GROUP_A_EXIT


PRUNED_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    # ── Group B: Crypto (20/10) — dual session ────────────────────────────
    ("BTC-USD", "binance", "hong_kong_open"),
    ("BTC-USD", "binance", "new_york_equity_open"),
    ("ETH-USD", "binance", "hong_kong_open"),
    ("ETH-USD", "binance", "new_york_equity_open"),
    ("SOL-USD", "binance", "hong_kong_open"),
    ("SOL-USD", "binance", "new_york_equity_open"),
    # ── Group B: Gold (20/10) — dual session ──────────────────────────────
    ("PAXG-USD", "binance", "hong_kong_open"),
    ("PAXG-USD", "binance", "new_york_equity_open"),
    # ── Group A: Metals (10/5) — NY only ──────────────────────────────────
    ("COPPER-USD", "tiingo", "new_york_equity_open"),
    ("SLV", "tiingo", "new_york_equity_open"),
    ("XPD-USD", "tiingo", "new_york_equity_open"),
    ("PPLT", "tiingo", "new_york_equity_open"),
    # ── Group A: Energy (10/5) — NY only ──────────────────────────────────
    ("BRENT", "tiingo", "new_york_equity_open"),
    ("NATGAS-USD", "tiingo", "new_york_equity_open"),
    # ── Group A: High-Beta Equities (10/5) — NY only ─────────────────────
    ("COIN", "tiingo", "new_york_equity_open"),
    ("CRCL", "tiingo", "new_york_equity_open"),
    ("HOOD", "tiingo", "new_york_equity_open"),
    ("INTC", "tiingo", "new_york_equity_open"),
    ("MSTR", "tiingo", "new_york_equity_open"),
    ("PLTR", "tiingo", "new_york_equity_open"),
    ("TSLA", "tiingo", "new_york_equity_open"),
    # ── Group B: Mega-Cap Equities (20/10) — NY only ─────────────────────
    ("EWY", "tiingo", "new_york_equity_open"),
    ("GOOGL", "tiingo", "new_york_equity_open"),
    ("META", "tiingo", "new_york_equity_open"),
    ("TSM", "tiingo", "new_york_equity_open"),
)

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
GOLD_TICKERS = {"PAXG-USD"}
METAL_TICKERS = {"COPPER-USD", "SLV", "XPD-USD", "PPLT"}
ENERGY_TICKERS = {"BRENT", "NATGAS-USD"}
EQUITY_TICKERS = {"COIN", "CRCL", "GOOGL", "HOOD", "INTC", "META", "MSTR", "PLTR", "TSLA", "TSM"}
ETF_TICKERS = {"EWY"}
VIX_GOVERNED_BUCKETS = {"equity", "etf"}
RANKED_FILL_BUCKETS = {"crypto", "equity", "etf"}
SOFT_CROWDING_BUCKETS = {"crypto", "equity", "etf"}

_BINANCE_FUTURES_EXECUTION_MAP: dict[str, tuple[str, str | None]] = {
    "BTC-USD": ("BTC", "BTCUSDT"),
    "BRENT": ("BZ", "BZUSDT"),
    "COIN": ("COIN", "COINUSDT"),
    "COPPER-USD": ("COPPER", "COPPERUSDT"),
    "CRCL": ("CRCL", "CRCLUSDT"),
    "ETH-USD": ("ETH", "ETHUSDT"),
    "EWY": ("EWY", "EWYUSDT"),
    "GOOGL": ("GOOGL", "GOOGLUSDT"),
    "HOOD": ("HOOD", "HOODUSDT"),
    "INTC": ("INTC", "INTCUSDT"),
    "META": ("META", "METAUSDT"),
    "MSTR": ("MSTR", "MSTRUSDT"),
    "NATGAS-USD": ("NATGAS", "NATGASUSDT"),
    "PAXG-USD": ("PAXG", "PAXGUSDT"),
    "PLTR": ("PLTR", "PLTRUSDT"),
    "SOL-USD": ("SOL", "SOLUSDT"),
    "TSLA": ("TSLA", "TSLAUSDT"),
    "TSM": ("TSM", "TSMUSDT"),
    "SLV": ("XAG", "XAGUSDT"),
    "XPD-USD": ("XPD", "XPDUSDT"),
    "PPLT": ("XPT", "XPTUSDT"),
}


def _resolve_universe(basket: str) -> tuple[tuple[str, str, str], ...]:
    key = (basket or "pruned").strip().lower()
    if key in {"pruned", "core", "grouped"}:
        return PRUNED_SESSION_TURTLE_UNIVERSE
    raise ValueError("basket must be one of {'pruned', 'core', 'grouped'}")


def _asset_bucket(ticker: str) -> str:
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in GOLD_TICKERS:
        return "gold"
    if ticker in ETF_TICKERS:
        return "etf"
    if ticker in EQUITY_TICKERS:
        return "equity"
    if ticker in METAL_TICKERS:
        return "metals"
    if ticker in ENERGY_TICKERS:
        return "energy"
    return "other"


def _execution_metadata(ticker: str, execution_mode: str) -> dict[str, object]:
    if execution_mode == "research":
        return {
            "execution_ticker": ticker,
            "execution_symbol": None,
            "tradeable": True,
        }
    execution_ticker, execution_symbol = _BINANCE_FUTURES_EXECUTION_MAP.get(ticker, (ticker, None))
    return {
        "execution_ticker": execution_ticker,
        "execution_symbol": execution_symbol,
        "tradeable": bool(execution_symbol),
    }


def get_binance_futures_execution_universe(basket: str = "pruned") -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for ticker, source, session_open in _resolve_universe(basket):
        meta = _execution_metadata(ticker=ticker, execution_mode="binance_futures")
        symbol = meta.get("execution_symbol")
        if symbol is None:
            continue
        symbol_text = str(symbol)
        if symbol_text in seen:
            continue
        seen.add(symbol_text)
        items.append(
            {
                "ticker": ticker,
                "source": source,
                "asset_bucket": _asset_bucket(ticker),
                "session_open": session_open,
                "execution_ticker": str(meta["execution_ticker"]),
                "execution_symbol": symbol_text,
            }
        )
    items.sort(key=lambda item: item["execution_symbol"])
    return items


def grouped_channel_summary() -> dict[str, dict[str, object]]:
    return {
        "A_COMMODITY": {
            "channel": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
            "symbols": ["BRENT", "NATGAS-USD", "COPPER-USD", "XPD-USD", "PPLT", "SLV"],
        },
        "A_HIGH_BETA": {
            "channel": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
            "symbols": ["COIN", "CRCL", "HOOD", "INTC", "MSTR", "PLTR", "TSLA"],
        },
        "B_CRYPTO": {
            "channel": f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"],
        },
        "B_MEGA_ETF": {
            "channel": f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
            "symbols": ["GOOGL", "META", "TSM", "EWY"],
        },
    }


def removed_ticker_summary() -> list[str]:
    return ["AAPL", "AMZN", "NVDA", "SPY", "QQQ", "EWJ"]


def build_pruned_grouped_candidates(
    *,
    basket: str = "pruned",
    initial_capital: float = 1_000.0,
    lookback_years: float = 4.1,
    base_risk_pct: float = 0.05,
    fixed_stop_pct: float = 0.10,
    directional_volume_risk_pct: float = 0.07,
    use_breakout_conviction_boost: bool = False,
    conviction_rel_volume_ratio_period: int = 5,
    conviction_max_mult: float = 1.25,
    conviction_rel_volume_weight: float = 0.10,
    conviction_rel_volume_ratio_weight: float = 0.15,
    conviction_breakout_weight: float = 0.10,
    conviction_close_location_weight: float = 0.05,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    use_extended_hours_protective_exits: bool = False,
    extended_hours_core_session_minutes: int = 390,
    close_positions_before_weekend: bool = False,
) -> list[dict]:
    universe = _resolve_universe(basket)
    candidates: list[dict] = []

    for combo_idx, (ticker, source, session_open) in enumerate(universe):
        use_extended_hours_mode = (
            use_extended_hours_protective_exits
            and source == "tiingo"
            and session_open == "new_york_equity_open"
        )
        channel_period, exit_channel_period = _channel_periods_for_ticker(ticker)
        payload = run_session_turtle_trend_backtest(
            ticker=ticker,
            initial_capital=initial_capital,
            interval="5m",
            lookback_years=lookback_years,
            market_data_source=source,
            session_open=session_open,
            channel_period=channel_period,
            exit_channel_period=exit_channel_period,
            base_risk_pct=base_risk_pct,
            max_position_pct=0.90,
            fixed_stop_pct=fixed_stop_pct,
            entry_window_minutes=extended_hours_core_session_minutes if use_extended_hours_mode else 480,
            core_session_minutes=extended_hours_core_session_minutes if use_extended_hours_mode else None,
            use_4h_trend_filter=True,
            trend_fast_period=trend_fast_period,
            trend_slow_period=trend_slow_period,
            use_extended_hours_protective_exits_only=use_extended_hours_mode,
            close_positions_before_weekend=close_positions_before_weekend,
            use_directional_volume_risk_boost=True,
            directional_volume_min_rel_volume=1.25,
            directional_volume_close_location_threshold=0.65,
            directional_volume_risk_pct=directional_volume_risk_pct,
            use_breakout_conviction_boost=use_breakout_conviction_boost,
            conviction_rel_volume_ratio_period=conviction_rel_volume_ratio_period,
            conviction_max_mult=conviction_max_mult,
            conviction_rel_volume_weight=conviction_rel_volume_weight,
            conviction_rel_volume_ratio_weight=conviction_rel_volume_ratio_weight,
            conviction_breakout_weight=conviction_breakout_weight,
            conviction_close_location_weight=conviction_close_location_weight,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        asset_bucket = _asset_bucket(ticker)
        for trade_idx, trade in enumerate(payload["trades"]):
            entry_price = float(trade["entry_price"])
            stop_loss = trade.get("stop_loss")
            if stop_loss is None and fixed_stop_pct is not None:
                if str(trade["direction"]) == "long":
                    stop_loss = entry_price * (1.0 - fixed_stop_pct)
                else:
                    stop_loss = entry_price * (1.0 + fixed_stop_pct)
            candidates.append(
                {
                    "combo_idx": combo_idx,
                    "trade_idx": trade_idx,
                    "ticker": ticker,
                    "source": source,
                    "session_open": session_open,
                    "direction": trade["direction"],
                    "entry_ts": datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts": datetime.fromisoformat(trade["exit_date"]),
                    "entry_price": entry_price,
                    "exit_price": float(trade["exit_price"]),
                    "stop_loss": float(stop_loss) if stop_loss is not None else None,
                    "risk_pct": float(trade.get("risk_pct", 0.0) or 0.0),
                    "shares": float(trade["shares"]),
                    "position_size": float(trade["position_size"]),
                    "pnl": float(trade["pnl"]),
                    "risk_model": str(trade["risk_model"]),
                    "entry_rel_volume": float(trade["entry_rel_volume"]),
                    "rel_volume_ratio": float(trade.get("rel_volume_ratio", 1.0) or 1.0),
                    "conviction_mult": float(trade.get("conviction_mult", 1.0) or 1.0),
                    "breakout_penetration": float(trade.get("breakout_penetration", 0.0) or 0.0),
                    "directional_close_score": float(trade.get("directional_close_score", 0.0) or 0.0),
                    "asset_bucket": asset_bucket,
                    "channel_period": channel_period,
                    "exit_channel_period": exit_channel_period,
                }
            )
    return candidates
