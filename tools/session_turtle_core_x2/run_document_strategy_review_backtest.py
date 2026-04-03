"""Run a document-style Session Turtle X3 backtest with the current repo engine.

This script approximates the 2026-04-03 strategy review the user supplied.

Important differences versus the pasted live-strategy review:
  - The current repo allocator does not implement the old ranked-fill / priority_score path.
  - The local cache is missing XPD-USD, so the run excludes it.
  - Some requested review tickers are mapped to local cache proxies:
      XAG-USD -> SLV
      XPT-USD -> PPLT
      BZ-USD  -> BRENT

Output:
  reports/session_turtle_x3_document_review_20260403/
    - summary.json
    - trades.csv

Run:
  /Users/timursabitov/Development/Python_Projects/buffetABC/.venv/bin/python \
      tools/session_turtle_core_x2/run_document_strategy_review_backtest.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

import edgar.services.session_turtle_portfolio as stp
from edgar.services.local_tiingo_data import available_tiingo_symbols
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)


OUTPUT_DIR = Path("reports/session_turtle_x3_document_review_20260403")


@dataclass(frozen=True)
class RequestedSymbol:
    requested_ticker: str
    engine_ticker: str | None
    source: str
    session_opens: tuple[str, ...]
    asset_bucket: str
    proxy_note: str | None = None


REQUESTED_SYMBOLS: tuple[RequestedSymbol, ...] = (
    RequestedSymbol("BTC-USD", "BTC-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("ETH-USD", "ETH-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("SOL-USD", "SOL-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("PAXG-USD", "PAXG-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "gold"),
    RequestedSymbol("COPPER-USD", "COPPER-USD", "tiingo", ("new_york_equity_open",), "metals"),
    RequestedSymbol("XAG-USD", "SLV", "tiingo", ("new_york_equity_open",), "metals", "proxy=SLV"),
    RequestedSymbol("XPD-USD", None, "tiingo", ("new_york_equity_open",), "metals", "missing local cache"),
    RequestedSymbol("XPT-USD", "PPLT", "tiingo", ("new_york_equity_open",), "metals", "proxy=PPLT"),
    RequestedSymbol("BZ-USD", "BRENT", "tiingo", ("new_york_equity_open",), "energy", "proxy=BRENT"),
    RequestedSymbol("NATGAS-USD", "NATGAS-USD", "tiingo", ("new_york_equity_open",), "energy"),
    RequestedSymbol("AMZN", "AMZN", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("COIN", "COIN", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("CRCL", "CRCL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("GOOGL", "GOOGL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("HOOD", "HOOD", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("INTC", "INTC", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("META", "META", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("MSTR", "MSTR", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("NVDA", "NVDA", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("PLTR", "PLTR", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSLA", "TSLA", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("EWJ", "EWJ", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("EWY", "EWY", "tiingo", ("new_york_equity_open",), "equity"),
)


def _load_macro_state(root: Path) -> dict:
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text(encoding="utf-8"))
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text(encoding="utf-8"))
    return build_extended_hours_proxy_state(daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg)


def _requested_combo_count(symbols: list[RequestedSymbol]) -> int:
    return sum(len(symbol.session_opens) for symbol in symbols)


def _resolve_runnable_symbols() -> tuple[list[RequestedSymbol], list[RequestedSymbol]]:
    tiingo_symbols = available_tiingo_symbols()
    runnable: list[RequestedSymbol] = []
    missing: list[RequestedSymbol] = []
    for symbol in REQUESTED_SYMBOLS:
        if symbol.engine_ticker is None:
            missing.append(symbol)
            continue
        if symbol.source == "tiingo":
            resolved = symbol.engine_ticker[:-4] if symbol.engine_ticker.endswith("-USD") else symbol.engine_ticker
            if resolved not in tiingo_symbols:
                missing.append(symbol)
                continue
        runnable.append(symbol)
    return runnable, missing


def _build_custom_universe(runnable: list[RequestedSymbol]) -> list[tuple[str, str, str]]:
    universe: list[tuple[str, str, str]] = []
    for symbol in runnable:
        assert symbol.engine_ticker is not None
        for session_open in symbol.session_opens:
            universe.append((symbol.engine_ticker, symbol.source, session_open))
    return universe


@contextmanager
def _patched_document_universe(runnable: list[RequestedSymbol]):
    old_core = stp.CORE_SESSION_TURTLE_UNIVERSE
    old_expanded = stp.EXPANDED_SESSION_TURTLE_UNIVERSE
    old_crypto = set(stp.CRYPTO_TICKERS)
    old_gold = set(stp.GOLD_TICKERS)
    old_equity = set(stp.EQUITY_TICKERS)
    old_metals = set(stp.METAL_TICKERS)
    old_energy = set(stp.ENERGY_TICKERS)

    custom_universe = tuple(_build_custom_universe(runnable))
    bucket_map: dict[str, set[str]] = {
        "crypto": set(),
        "gold": set(),
        "equity": set(),
        "metals": set(),
        "energy": set(),
    }
    for symbol in runnable:
        assert symbol.engine_ticker is not None
        bucket_map[symbol.asset_bucket].add(symbol.engine_ticker)

    stp.CORE_SESSION_TURTLE_UNIVERSE = custom_universe
    stp.EXPANDED_SESSION_TURTLE_UNIVERSE = tuple(list(custom_universe) + list(stp.INDEX_SESSION_TURTLE_UNIVERSE))
    stp.CRYPTO_TICKERS = bucket_map["crypto"]
    stp.GOLD_TICKERS = bucket_map["gold"]
    stp.EQUITY_TICKERS = bucket_map["equity"]
    stp.METAL_TICKERS = bucket_map["metals"]
    stp.ENERGY_TICKERS = bucket_map["energy"]
    try:
        yield list(dict.fromkeys(custom_universe))
    finally:
        stp.CORE_SESSION_TURTLE_UNIVERSE = old_core
        stp.EXPANDED_SESSION_TURTLE_UNIVERSE = old_expanded
        stp.CRYPTO_TICKERS = old_crypto
        stp.GOLD_TICKERS = old_gold
        stp.EQUITY_TICKERS = old_equity
        stp.METAL_TICKERS = old_metals
        stp.ENERGY_TICKERS = old_energy


@contextmanager
def _patched_document_macro_scope():
    old_lookup = stp._lookup_extended_hours_signal

    def _document_lookup_extended_hours_signal(*, asset_bucket: str, **kwargs):
        if asset_bucket not in {"crypto", "equity"}:
            return None, 1.0, None, None
        return old_lookup(asset_bucket=asset_bucket, **kwargs)

    stp._lookup_extended_hours_signal = _document_lookup_extended_hours_signal
    try:
        yield
    finally:
        stp._lookup_extended_hours_signal = old_lookup


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runnable, missing = _resolve_runnable_symbols()
    runnable_universe = _build_custom_universe(runnable)

    print("Document strategy review backtest")
    print(f"  requested symbols : {len(REQUESTED_SYMBOLS)}")
    print(f"  runnable symbols  : {len(runnable)}")
    print(f"  missing symbols   : {len(missing)}")
    if missing:
        print("  missing tickers   : " + ", ".join(symbol.requested_ticker for symbol in missing))

    macro_state = _load_macro_state(root)

    with _patched_document_universe(runnable), _patched_document_macro_scope():
        tech_state = build_per_asset_technical_state(
            universe=list(dict.fromkeys(runnable_universe)),
            lookback_years=5.0,
            warmup_days=300,
            ema_period=200,
            adx_period=14,
        )

        candidates = build_session_turtle_shared_account_candidates(
            basket="core",
            initial_capital=1_000.0,
            lookback_years=4.1,
            channel_period=10,
            base_risk_pct=0.05,
            fixed_stop_pct=0.10,
            directional_volume_risk_pct=0.07,
            use_breakout_conviction_boost=True,
            conviction_max_mult=1.25,
            trend_fast_period=55,
            trend_slow_period=200,
        )

        result = generate_session_turtle_shared_account_report(
            basket="core",
            exposure_mult=3.0,
            use_drawdown_governor=True,
            drawdown_trigger_1_pct=15.0,
            drawdown_exposure_mult_1=1.5,
            drawdown_trigger_2_pct=25.0,
            drawdown_exposure_mult_2=0.5,
            crypto_cap_mult=1.0,
            gold_cap_mult=1.0,
            metals_cap_mult=1.0,
            energy_cap_mult=1.0,
            equity_cap_mult=None,
            base_risk_pct=0.05,
            fixed_stop_pct=0.10,
            directional_volume_risk_pct=0.07,
            lookback_years=4.1,
            channel_period=10,
            use_breakout_conviction_boost=True,
            conviction_max_mult=1.25,
            trend_fast_period=55,
            trend_slow_period=200,
            precomputed_candidates=candidates,
            use_extended_hours_proxy=True,
            extended_hours_proxy_state=macro_state,
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
            per_asset_technical_state=tech_state,
            per_asset_ema_lag_days=1,
            per_asset_ema_above_long_mult=1.0,
            per_asset_ema_above_short_mult=0.25,
            per_asset_ema_below_long_mult=0.25,
            per_asset_ema_below_short_mult=1.0,
            per_asset_use_adx_gate=False,
        )

    summary = dict(result["summary"])
    trades = list(result["trades"])

    proxy_map = {
        symbol.requested_ticker: symbol.engine_ticker
        for symbol in runnable
        if symbol.engine_ticker is not None and symbol.requested_ticker != symbol.engine_ticker
    }
    summary.update(
        {
            "variant_label": "document_review_approximation",
            "assumption": "Current engine approximation of pasted live-strategy review; ranked-fill logic is not reproduced.",
            "requested_symbol_count": len(REQUESTED_SYMBOLS),
            "requested_session_combo_count": _requested_combo_count(list(REQUESTED_SYMBOLS)),
            "runnable_symbol_count": len(runnable),
            "runnable_session_combo_count": _requested_combo_count(runnable),
            "missing_requested_symbols": [symbol.requested_ticker for symbol in missing],
            "proxy_symbol_map": proxy_map,
            "requested_symbols": [asdict(symbol) for symbol in REQUESTED_SYMBOLS],
        }
    )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if trades:
        trades_path = OUTPUT_DIR / "trades.csv"
        with trades_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
            writer.writeheader()
            writer.writerows(trades)

    print()
    print("Result")
    print(f"  start            : {summary['start_date']}")
    print(f"  end              : {summary['end_date']}")
    print(f"  candidate trades : {summary['candidate_trades']}")
    print(f"  executed trades  : {summary['executed_trades']}")
    print(f"  total return pct : {summary['total_return_pct']:.2f}")
    print(f"  CAGR pct         : {summary['cagr_pct']:.2f}")
    print(f"  max DD pct       : {summary['max_realized_drawdown_pct']:.2f}")
    print(f"  profit factor    : {summary['profit_factor']:.2f}")
    print(f"  final equity     : {summary['final_equity']:.4f}")
    print(f"  summary path     : {summary_path}")


if __name__ == "__main__":
    main()