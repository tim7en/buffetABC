"""Run Session Turtle backtest — Pruned & Grouped universe (21 symbols).
=======================================================================
Group A (10/5 Donchian): Commodities + High-Beta Equities  → 13 combos
Group B (20/10 Donchian): Crypto + Mega-Cap Equities       → 12 combos (crypto dual-session)

Removed (PF < 1.0): AAPL, AMZN, NVDA, SPY, QQQ, EWJ

Usage:
    python tools/run_pruned_grouped_backtest.py
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")
import django
django.setup()

import edgar.services.session_turtle_portfolio as stp
from edgar.services.session_turtle_portfolio import (
    build_per_asset_technical_state,
    generate_session_turtle_shared_account_report,
    _asset_bucket,
)
from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest
from edgar.services.local_tiingo_data import available_tiingo_symbols
from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
    RequestedSymbol,
    _load_macro_state,
    _patched_document_macro_scope,
    _patched_document_universe,
)

# ── output dir ────────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "reports" / "pruned_grouped_backtest_20260404"

# ── portfolio parameters (x3 production) ──────────────────────────────────────
X3_PARAMS = dict(
    basket="core",
    exposure_mult=3.0,
    use_drawdown_governor=True,
    drawdown_trigger_1_pct=15.0,
    drawdown_exposure_mult_1=1.5,
    drawdown_trigger_2_pct=25.0,
    drawdown_exposure_mult_2=0.5,
    base_risk_pct=0.05,
    fixed_stop_pct=0.10,
    directional_volume_risk_pct=0.07,
    lookback_years=4.1,
    channel_period=10,
    use_breakout_conviction_boost=True,
    conviction_max_mult=1.25,
    trend_fast_period=55,
    trend_slow_period=200,
    crypto_cap_mult=1.0,
    gold_cap_mult=1.0,
    metals_cap_mult=1.0,
    energy_cap_mult=1.0,
    equity_cap_mult=None,
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
    per_asset_ema_above_short_mult=0.25,
    per_asset_ema_below_long_mult=0.25,
    per_asset_ema_below_short_mult=1.0,
    per_asset_use_adx_gate=False,
)


# ══════════════════════════════════════════════════════════════════════════════
#  ASSET DEFINITIONS — Pruned Universe (21 symbols, 2 groups)
# ══════════════════════════════════════════════════════════════════════════════

# ── Group A: Fast Breakout (10/5 Donchian) ────────────────────────────────────
GROUP_A_CHANNEL = 10
GROUP_A_EXIT = 5

COMMODITY_SYMBOLS = (
    RequestedSymbol("BZ-USD",     "BRENT",      "tiingo", ("new_york_equity_open",), "energy",  "proxy=BRENT"),
    RequestedSymbol("NATGAS-USD", "NATGAS-USD",  "tiingo", ("new_york_equity_open",), "energy"),
    RequestedSymbol("COPPER-USD", "COPPER-USD",  "tiingo", ("new_york_equity_open",), "metals"),
    RequestedSymbol("XPD-USD",    "XPD-USD",     "tiingo", ("new_york_equity_open",), "metals"),
    RequestedSymbol("XPT-USD",    "PPLT",        "tiingo", ("new_york_equity_open",), "metals", "proxy=PPLT"),
    RequestedSymbol("XAG-USD",    "SLV",         "tiingo", ("new_york_equity_open",), "metals", "proxy=SLV"),
)

HIGH_BETA_SYMBOLS = (
    RequestedSymbol("COIN",  "COIN",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("CRCL",  "CRCL",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("HOOD",  "HOOD",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("INTC",  "INTC",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("MSTR",  "MSTR",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("PLTR",  "PLTR",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSLA",  "TSLA",  "tiingo", ("new_york_equity_open",), "equity"),
)

GROUP_A = [
    ("COMMODITY", COMMODITY_SYMBOLS, GROUP_A_CHANNEL, GROUP_A_EXIT),
    ("HIGH_BETA", HIGH_BETA_SYMBOLS, GROUP_A_CHANNEL, GROUP_A_EXIT),
]

# ── Group B: Slow Breakout (20/10 Donchian) ───────────────────────────────────
GROUP_B_CHANNEL = 20
GROUP_B_EXIT = 10

CRYPTO_SYMBOLS = (
    RequestedSymbol("BTC-USD",  "BTC-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("ETH-USD",  "ETH-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("SOL-USD",  "SOL-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("PAXG-USD", "PAXG-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "gold"),
)

MEGA_ETF_SYMBOLS = (
    RequestedSymbol("GOOGL", "GOOGL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("META",  "META",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSM",   "TSM",   "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("EWY",   "EWY",   "tiingo", ("new_york_equity_open",), "equity"),
)

GROUP_B = [
    ("CRYPTO",   CRYPTO_SYMBOLS,   GROUP_B_CHANNEL, GROUP_B_EXIT),
    ("MEGA_ETF", MEGA_ETF_SYMBOLS, GROUP_B_CHANNEL, GROUP_B_EXIT),
]

ALL_GROUPS = GROUP_A + GROUP_B
REMOVED_TICKERS = ["AAPL", "AMZN", "NVDA", "SPY", "QQQ", "EWJ"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_runnable(symbols: tuple[RequestedSymbol, ...]) -> list[RequestedSymbol]:
    tiingo_avail = available_tiingo_symbols()
    runnable: list[RequestedSymbol] = []
    for sym in symbols:
        if sym.engine_ticker is None:
            continue
        if sym.source == "tiingo":
            resolved = sym.engine_ticker[:-4] if sym.engine_ticker.endswith("-USD") else sym.engine_ticker
            if resolved not in tiingo_avail:
                print(f"  SKIP {sym.requested_ticker}: not in local cache")
                continue
        runnable.append(sym)
    return runnable


def _build_candidates(
    symbols: list[RequestedSymbol],
    channel_period: int,
    exit_channel_period: int,
    label: str,
    combo_idx_offset: int = 0,
) -> list[dict]:
    print(f"    [{label} {channel_period}/{exit_channel_period}] "
          f"{len(symbols)} symbols", flush=True)

    candidates: list[dict] = []
    combo_idx = combo_idx_offset

    for sym in symbols:
        assert sym.engine_ticker is not None
        for session_open in sym.session_opens:
            try:
                payload = run_session_turtle_trend_backtest(
                    ticker=sym.engine_ticker,
                    initial_capital=1_000.0,
                    interval="5m",
                    lookback_years=4.1,
                    market_data_source=sym.source,
                    session_open=session_open,
                    channel_period=channel_period,
                    exit_channel_period=exit_channel_period,
                    base_risk_pct=0.05,
                    fixed_stop_pct=0.10,
                    max_position_pct=0.90,
                    directional_volume_risk_pct=0.07,
                    entry_window_minutes=480,
                    use_4h_trend_filter=True,
                    trend_fast_period=55,
                    trend_slow_period=200,
                    use_directional_volume_risk_boost=True,
                    directional_volume_min_rel_volume=1.25,
                    directional_volume_close_location_threshold=0.65,
                    use_breakout_conviction_boost=True,
                    conviction_max_mult=1.25,
                    enable_pyramiding=False,
                    use_break_even_stop=False,
                    use_chandelier_exit=False,
                )
            except Exception as exc:
                print(f"      WARN: {sym.engine_ticker}/{session_open} failed: {exc}")
                combo_idx += 1
                continue

            for trade_idx, trade in enumerate(payload["trades"]):
                entry_price = float(trade["entry_price"])
                stop_loss = trade.get("stop_loss")
                if stop_loss is None:
                    if str(trade["direction"]) == "long":
                        stop_loss = entry_price * (1.0 - 0.10)
                    else:
                        stop_loss = entry_price * (1.0 + 0.10)
                candidates.append({
                    "combo_idx":               combo_idx,
                    "trade_idx":               trade_idx,
                    "ticker":                  sym.engine_ticker,
                    "source":                  sym.source,
                    "session_open":            session_open,
                    "direction":               trade["direction"],
                    "entry_ts":                _dt.datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts":                 _dt.datetime.fromisoformat(trade["exit_date"]),
                    "entry_price":             entry_price,
                    "exit_price":              float(trade["exit_price"]),
                    "stop_loss":               float(stop_loss),
                    "risk_pct":                float(trade.get("risk_pct", 0.0) or 0.0),
                    "shares":                  float(trade["shares"]),
                    "position_size":           float(trade["position_size"]),
                    "pnl":                     float(trade["pnl"]),
                    "risk_model":              str(trade["risk_model"]),
                    "entry_rel_volume":        float(trade["entry_rel_volume"]),
                    "rel_volume_ratio":        float(trade.get("rel_volume_ratio", 1.0) or 1.0),
                    "conviction_mult":         float(trade.get("conviction_mult", 1.0) or 1.0),
                    "breakout_penetration":    float(trade.get("breakout_penetration", 0.0) or 0.0),
                    "directional_close_score": float(trade.get("directional_close_score", 0.0) or 0.0),
                    "asset_bucket":            _asset_bucket(sym.engine_ticker),
                })
            combo_idx += 1

    print(f"    [{label}] -> {len(candidates)} raw candidates", flush=True)
    return candidates


def _run_portfolio(all_runnable, candidates, macro_state, label):
    with _patched_document_universe(all_runnable) as runnable_universe, \
         _patched_document_macro_scope():
        tech_state = build_per_asset_technical_state(
            universe=list(dict.fromkeys(runnable_universe)),
            lookback_years=5.0,
            warmup_days=300,
            ema_period=200,
            adx_period=14,
        )
        result = generate_session_turtle_shared_account_report(
            precomputed_candidates=candidates,
            extended_hours_proxy_state=macro_state,
            per_asset_technical_state=tech_state,
            **X3_PARAMS,
        )
    return result


def _per_asset_stats(trades):
    s = defaultdict(lambda: dict(
        n=0, wins=0, gross_win=0.0, gross_loss=0.0,
        total_pnl=0.0, bucket="?",
        long_n=0, short_n=0, long_pnl=0.0, short_pnl=0.0,
    ))
    for t in trades:
        k = t["ticker"]
        pnl = float(t["net_pnl"])
        s[k]["n"] += 1
        s[k]["total_pnl"] += pnl
        s[k]["bucket"] = t["asset_bucket"]
        if t["direction"] == "long":
            s[k]["long_n"] += 1
            s[k]["long_pnl"] += pnl
        else:
            s[k]["short_n"] += 1
            s[k]["short_pnl"] += pnl
        if pnl > 0:
            s[k]["wins"] += 1
            s[k]["gross_win"] += pnl
        else:
            s[k]["gross_loss"] += abs(pnl)
    for ticker, st in s.items():
        st["win_rate_pct"] = round(st["wins"] / st["n"] * 100, 1) if st["n"] else None
        st["profit_factor"] = (round(st["gross_win"] / st["gross_loss"], 2)
                               if st["gross_loss"] > 0
                               else (float("inf") if st["gross_win"] > 0 else None))
        st["total_pnl"] = round(st["total_pnl"], 2)
        st["long_pnl"] = round(st["long_pnl"], 2)
        st["short_pnl"] = round(st["short_pnl"], 2)
    return dict(s)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    macro_state = _load_macro_state(ROOT)

    print("=" * 100)
    print("  PRUNED & GROUPED BACKTEST — 21 Symbols, Per-Group Channel Periods")
    print("=" * 100)
    print(f"\n  Removed: {', '.join(REMOVED_TICKERS)}")

    # ── Resolve symbols and build candidates per group ────────────────────
    all_runnable: list[RequestedSymbol] = []
    all_candidates: list[dict] = []
    group_map: dict[str, str] = {}
    combo_offset = 0

    for group_name, symbols, ch_period, ch_exit in ALL_GROUPS:
        runnable = _resolve_runnable(symbols)
        all_runnable.extend(runnable)
        for sym in runnable:
            group_map[sym.engine_ticker] = group_name
        tickers_str = ", ".join(s.engine_ticker for s in runnable)
        print(f"  {group_name:<12} : {len(runnable)} symbols  "
              f"channel={ch_period}/{ch_exit}  [{tickers_str}]")

        candidates = _build_candidates(
            symbols=runnable,
            channel_period=ch_period,
            exit_channel_period=ch_exit,
            label=group_name,
            combo_idx_offset=combo_offset,
        )
        all_candidates.extend(candidates)
        combo_offset += len(runnable) * 3  # generous offset for session combos

    print(f"\n  Total: {len(all_runnable)} symbols, {len(all_candidates)} raw candidates")

    # ── Run portfolio ─────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  RUNNING PORTFOLIO SIMULATION")
    print(f"{'=' * 100}\n")

    result = _run_portfolio(all_runnable, all_candidates, macro_state,
                            "Pruned & Grouped x3")
    summary = result["summary"]
    trades = result["trades"]

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  RESULTS")
    print(f"{'=' * 100}")
    print(f"""
  Strategy:         {summary['label']}
  Period:           {summary.get('start_date', '?')[:10]} → {summary.get('end_date', '?')[:10]}
  Initial Capital:  ${summary['initial_capital']:,.2f}
  Final Equity:     ${summary['final_equity']:,.2f}
  Total Return:     {summary['total_return_pct']:+,.2f}%
  CAGR:             {summary['cagr_pct']:.2f}%
  Max Drawdown:     {summary['max_realized_drawdown_pct']:.2f}%
  Profit Factor:    {summary['profit_factor']:.2f}
  Win Rate:         {summary['win_rate_pct']:.1f}%
  Executed Trades:  {summary['executed_trades']}
  Candidate Trades: {summary['candidate_trades']}
  Long Trades:      {summary['long_trades']}
  Short Trades:     {summary['short_trades']}
  Skipped (same):   {summary['skipped_same_ticker']}
  Skipped (cap):    {summary['skipped_no_capacity']}
""")

    # ── Per-group breakdown ───────────────────────────────────────────────
    group_stats: dict[str, dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in trades:
        grp = group_map.get(t["ticker"], "?")
        group_stats[grp]["trades"] += 1
        group_stats[grp]["pnl"] += float(t["net_pnl"])

    print("  --- Per-Group P&L ---")
    print(f"  {'Group':<14} {'Channel':>8} {'Trades':>8} {'P&L':>14}")
    print(f"  {'-' * 50}")
    ch_map = {
        "COMMODITY": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
        "HIGH_BETA": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
        "CRYPTO":    f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
        "MEGA_ETF":  f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
    }
    for grp in ["CRYPTO", "COMMODITY", "MEGA_ETF", "HIGH_BETA"]:
        gs = group_stats[grp]
        ch = ch_map.get(grp, "?")
        print(f"  {grp:<14} {ch:>8} {gs['trades']:>8} ${gs['pnl']:>13,.2f}")

    # ── Per-bucket breakdown ──────────────────────────────────────────────
    print(f"\n  --- Per-Bucket P&L ---")
    for bucket in ["crypto", "gold", "equity", "metals", "energy"]:
        key = f"{bucket}_pnl"
        tkey = f"{bucket}_trades"
        if key in summary:
            print(f"  {bucket:<12} {summary[tkey]:>4} trades  ${summary[key]:>13,.2f}")

    # ── Per-asset stats ───────────────────────────────────────────────────
    asset_stats = _per_asset_stats(trades)

    print(f"\n  --- Per-Asset Performance (sorted by P&L) ---")
    print(f"  {'Ticker':<14} {'Group':<12} {'Trades':>6} {'WR%':>7} {'PF':>8} "
          f"{'TotPnL':>12} {'L.PnL':>10} {'S.PnL':>10}")
    print(f"  {'-' * 85}")
    for ticker, st in sorted(asset_stats.items(), key=lambda x: -x[1]["total_pnl"]):
        grp = group_map.get(ticker, "?")
        pf_str = f"{st['profit_factor']:.2f}" if st["profit_factor"] != float("inf") else "inf"
        print(f"  {ticker:<14} {grp:<12} {st['n']:>6} {st['win_rate_pct']:>6.1f}% "
              f"{pf_str:>8} {st['total_pnl']:>+12,.2f} "
              f"{st['long_pnl']:>+10,.2f} {st['short_pnl']:>+10,.2f}")

    # ── Yearly returns ────────────────────────────────────────────────────
    yearly = result.get("yearly_returns", [])
    if yearly:
        print(f"\n  --- Yearly Returns ---")
        print(f"  {'Year':>6} {'Return%':>10} {'P&L':>14} {'Trades':>8} {'WR%':>8}")
        print(f"  {'-' * 50}")
        for yr in yearly:
            print(f"  {yr['year']:>6} {yr['return_pct']:>+9.1f}% "
                  f"${yr['pnl']:>13,.2f} {yr['trades']:>8} {yr['win_rate_pct']:>7.1f}%")

    # ── Save summary ──────────────────────────────────────────────────────
    save_payload = {
        "groups": {
            "A_COMMODITY":  {"channel": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
                             "symbols": [s.engine_ticker for s in COMMODITY_SYMBOLS]},
            "A_HIGH_BETA":  {"channel": f"{GROUP_A_CHANNEL}/{GROUP_A_EXIT}",
                             "symbols": [s.engine_ticker for s in HIGH_BETA_SYMBOLS]},
            "B_CRYPTO":     {"channel": f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
                             "symbols": [s.engine_ticker for s in CRYPTO_SYMBOLS]},
            "B_MEGA_ETF":   {"channel": f"{GROUP_B_CHANNEL}/{GROUP_B_EXIT}",
                             "symbols": [s.engine_ticker for s in MEGA_ETF_SYMBOLS]},
        },
        "removed": REMOVED_TICKERS,
        "summary": summary,
        "per_asset": asset_stats,
    }
    out_file = OUTPUT_DIR / "summary.json"
    out_file.write_text(json.dumps(save_payload, indent=2, default=str), encoding="utf-8")

    # Save trades CSV
    import csv
    trades_file = OUTPUT_DIR / "trades.csv"
    if trades:
        fieldnames = list(trades[0].keys())
        with open(trades_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

    print(f"\n  Saved: {out_file}")
    print(f"  Saved: {trades_file}")
    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
