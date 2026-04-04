"""Asset-Class Grouped Donchian Channel Optimization
=====================================================
Separates assets into 4 natural groups by trading-hours and asset class,
then tests a matrix of Donchian entry/exit channel periods per group to
find the optimal breakout timeline for each.

Groups:
  1. CRYPTO      — 24/7 trading  (BTC, ETH, SOL, PAXG)
  2. COMMODITIES — near-24h      (BRENT, NATGAS, COPPER, XPD, XPT/PPLT, XAG/SLV)
  3. MEGA_ETF    — US mega-caps + broad ETFs  (AAPL, AMZN, GOOGL, NVDA, META, MSFT→TSM, QQQ, SPY, EWJ, EWY)
  4. HIGH_BETA   — US high-beta / mid-cap  (COIN, CRCL, HOOD, INTC, MSTR, PLTR, TSLA)

For each group we test channel periods: 5/3, 10/5, 15/7, 20/10, 30/15
Then we build combined portfolios with each group at its best period.

Usage:
    python tools/group_channel_optimization.py
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")
import django
django.setup()

import edgar.services.session_turtle_portfolio as stp
from edgar.services.session_turtle_portfolio import (
    build_per_asset_technical_state,
    generate_session_turtle_shared_account_report,
)
from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest
from edgar.services.session_turtle_portfolio import _asset_bucket
from edgar.services.local_tiingo_data import available_tiingo_symbols
from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
    RequestedSymbol,
    _load_macro_state,
    _patched_document_macro_scope,
    _patched_document_universe,
)

# ── output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "reports" / "group_channel_optimization_20260404"

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

# ── asset group definitions ───────────────────────────────────────────────────
CRYPTO_SYMBOLS = (
    RequestedSymbol("BTC-USD",  "BTC-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("ETH-USD",  "ETH-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("SOL-USD",  "SOL-USD",  "binance", ("hong_kong_open", "new_york_equity_open"), "crypto"),
    RequestedSymbol("PAXG-USD", "PAXG-USD", "binance", ("hong_kong_open", "new_york_equity_open"), "gold"),
)

COMMODITY_SYMBOLS = (
    RequestedSymbol("BZ-USD",     "BRENT",      "tiingo", ("new_york_equity_open",), "energy",  "proxy=BRENT"),
    RequestedSymbol("NATGAS-USD", "NATGAS-USD",  "tiingo", ("new_york_equity_open",), "energy"),
    RequestedSymbol("COPPER-USD", "COPPER-USD",  "tiingo", ("new_york_equity_open",), "metals"),
    RequestedSymbol("XPD-USD",    "XPD-USD",     "tiingo", ("new_york_equity_open",), "metals"),
    RequestedSymbol("XPT-USD",    "PPLT",        "tiingo", ("new_york_equity_open",), "metals", "proxy=PPLT"),
    RequestedSymbol("XAG-USD",    "SLV",         "tiingo", ("new_york_equity_open",), "metals", "proxy=SLV"),
)

MEGA_ETF_SYMBOLS = (
    RequestedSymbol("AAPL",  "AAPL",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("AMZN",  "AMZN",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("GOOGL", "GOOGL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("META",  "META",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("NVDA",  "NVDA",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSM",   "TSM",   "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("QQQ",   "QQQ",   "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("SPY",   "SPY",   "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("EWJ",   "EWJ",   "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("EWY",   "EWY",   "tiingo", ("new_york_equity_open",), "equity"),
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

GROUPS = {
    "CRYPTO":    CRYPTO_SYMBOLS,
    "COMMODITY": COMMODITY_SYMBOLS,
    "MEGA_ETF":  MEGA_ETF_SYMBOLS,
    "HIGH_BETA": HIGH_BETA_SYMBOLS,
}

# Channel periods to test: (entry_period, exit_period)
# Engine only allows channel_period in {10, 20, 55}
# exit_channel_period defaults to entry/2 when not specified
CHANNEL_PERIODS = [
    (10, 5),
    (20, 10),
    (55, 27),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _ann_vol(engine_ticker: str, source: str) -> float | None:
    try:
        if source == "binance":
            bmap = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
                    "SOL-USD": "SOLUSDT", "PAXG-USD": "PAXGUSDT"}
            bt = bmap.get(engine_ticker)
            if not bt:
                return None
            gz_files = list((ROOT / "cache" / "binance_asia_orb").glob(f"{bt}_*.csv.gz"))
            if not gz_files:
                return None
            df = pd.concat([pd.read_csv(f, compression="gzip") for f in gz_files])
            time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), "close")
            df["_dt"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
            close = df.set_index("_dt").sort_index()[close_col].astype(float)
        else:
            stem = engine_ticker.replace("-USD", "")
            pq = ROOT / "cache" / "cache" / "tiingo" / f"{stem}_5m.parquet"
            if not pq.exists():
                return None
            df = pd.read_parquet(pq)
            time_col = next((c for c in df.columns if c.lower() in ("time", "ts")), None)
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), None)
            if not time_col or not close_col:
                return None
            df["_dt"] = pd.to_datetime(df[time_col], utc=True)
            close = df.set_index("_dt").sort_index()[close_col].astype(float)

        daily = close.resample("1D").last().dropna()
        ret = np.log(daily / daily.shift(1)).dropna()
        return round(float(ret.std()) * math.sqrt(252) * 100, 1)
    except Exception:
        return None


def _per_asset_stats(trades: list[dict], group_map: dict[str, str]) -> dict[str, dict]:
    s: dict[str, dict] = defaultdict(lambda: dict(
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
        st["group"] = group_map.get(ticker, "?")
        st["win_rate_pct"] = round(st["wins"] / st["n"] * 100, 1) if st["n"] > 0 else None
        st["profit_factor"] = (round(st["gross_win"] / st["gross_loss"], 2)
                               if st["gross_loss"] > 0
                               else (float("inf") if st["gross_win"] > 0 else None))
        st["total_pnl"] = round(st["total_pnl"], 2)
        st["long_pnl"] = round(st["long_pnl"], 2)
        st["short_pnl"] = round(st["short_pnl"], 2)
    return dict(s)


def _resolve_runnable(symbols: tuple[RequestedSymbol, ...]) -> list[RequestedSymbol]:
    tiingo_avail = available_tiingo_symbols()
    runnable = []
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


# ── candidate generation ─────────────────────────────────────────────────────
def _build_candidates(
    symbols: list[RequestedSymbol],
    channel_period: int,
    exit_channel_period: int,
    group_label: str,
    combo_idx_offset: int = 0,
) -> list[dict]:
    import datetime as _dt_mod

    print(f"    [{group_label}] {len(symbols)} symbols  "
          f"channel={channel_period}/{exit_channel_period}", flush=True)

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
                    "entry_ts":                _dt_mod.datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts":                 _dt_mod.datetime.fromisoformat(trade["exit_date"]),
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

    print(f"    [{group_label}] -> {len(candidates)} raw candidates", flush=True)
    return candidates


# ── portfolio simulation ─────────────────────────────────────────────────────
def _run_portfolio(
    all_runnable: list[RequestedSymbol],
    candidates: list[dict],
    macro_state: dict,
    label: str,
) -> dict:
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
    return {
        "label": label,
        "summary": {k: v for k, v in result["summary"].items()
                    if not isinstance(v, (list, dict))},
        "trades": result["trades"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    macro_state = _load_macro_state(ROOT)

    # ── resolve all symbols ───────────────────────────────────────────────────
    all_symbols = []
    group_runnable: dict[str, list[RequestedSymbol]] = {}
    group_map: dict[str, str] = {}  # ticker -> group name

    print("=" * 80)
    print("  ASSET-CLASS GROUPED DONCHIAN CHANNEL OPTIMIZATION")
    print("=" * 80)

    for group_name, symbols in GROUPS.items():
        runnable = _resolve_runnable(symbols)
        group_runnable[group_name] = runnable
        all_symbols.extend(runnable)
        for sym in runnable:
            group_map[sym.engine_ticker] = group_name
        print(f"  {group_name:<12} : {len(runnable)} symbols  "
              f"[{', '.join(s.engine_ticker for s in runnable)}]")

    print(f"\n  Total symbols: {len(all_symbols)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: Per-Group Channel Period Grid Search (isolated per-group)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  PHASE 1: Per-Group Channel Period Scan (isolated single-group portfolios)")
    print(f"{'=' * 80}\n")

    # Cache all candidate sets: group_candidates[group_name][(entry,exit)] = candidates
    group_candidates: dict[str, dict[tuple[int, int], list[dict]]] = {}
    combo_offset = 0

    for group_name, symbols in group_runnable.items():
        group_candidates[group_name] = {}
        for entry_p, exit_p in CHANNEL_PERIODS:
            label = f"{group_name} {entry_p}/{exit_p}"
            cands = _build_candidates(
                symbols, entry_p, exit_p, label,
                combo_idx_offset=combo_offset,
            )
            group_candidates[group_name][(entry_p, exit_p)] = cands
            combo_offset = max((c["combo_idx"] for c in cands), default=combo_offset) + 1

    # ── run isolated portfolio for each group x period ────────────────────────
    print(f"\n  Running isolated portfolio simulations ...\n")

    group_results: dict[str, list[dict]] = {}  # group_name -> list of result dicts
    best_period: dict[str, tuple[int, int]] = {}  # group_name -> best (entry, exit)

    for group_name, symbols in group_runnable.items():
        group_results[group_name] = []
        best_pf = -1.0
        best_p = CHANNEL_PERIODS[1]  # default 10/5

        for entry_p, exit_p in CHANNEL_PERIODS:
            cands = group_candidates[group_name][(entry_p, exit_p)]
            if not cands:
                continue
            label = f"{group_name} {entry_p}/{exit_p}"
            result = _run_portfolio(symbols, cands, macro_state, label)
            s = result["summary"]
            group_results[group_name].append({
                "period": f"{entry_p}/{exit_p}",
                "entry": entry_p,
                "exit": exit_p,
                "trades": s.get("executed_trades", 0),
                "return_pct": s.get("total_return_pct", 0),
                "cagr_pct": s.get("cagr_pct", 0),
                "max_dd_pct": s.get("max_realized_drawdown_pct", 0),
                "profit_factor": s.get("profit_factor", 0),
                "win_rate_pct": s.get("win_rate_pct", 0),
                "final_equity": s.get("final_equity", 0),
            })

            pf = s.get("profit_factor", 0)
            if pf > best_pf:
                best_pf = pf
                best_p = (entry_p, exit_p)

        best_period[group_name] = best_p

    # ── print Phase 1 results ─────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  PHASE 1 RESULTS: Isolated Per-Group Channel Period Scan")
    print(f"{'=' * 100}")

    for group_name in GROUPS:
        results = group_results[group_name]
        bp = best_period[group_name]
        print(f"\n  --- {group_name} ({len(group_runnable[group_name])} symbols) ---")
        print(f"  {'Period':<8}  {'Trades':>6}  {'Return%':>10}  {'CAGR%':>8}  "
              f"{'MaxDD%':>7}  {'PF':>6}  {'WR%':>6}  {'Final$':>12}  {'Best':>5}")
        print(f"  {'-' * 85}")
        for r in results:
            is_best = "*" if (r["entry"], r["exit"]) == bp else ""
            print(f"  {r['period']:<8}  {r['trades']:>6}  {r['return_pct']:>+9.2f}%  "
                  f"{r['cagr_pct']:>7.2f}%  {r['max_dd_pct']:>6.2f}%  "
                  f"{r['profit_factor']:>6.2f}  {r['win_rate_pct']:>5.1f}%  "
                  f"${r['final_equity']:>10,.2f}  {is_best:>5}")

    print(f"\n  Best periods by group (highest PF):")
    for group_name, bp in best_period.items():
        print(f"    {group_name:<12} : {bp[0]}/{bp[1]}")

    # ── also determine best by CAGR and by risk-adjusted (PF * sqrt(trades)) ─
    best_by_cagr: dict[str, tuple[int, int]] = {}
    best_by_risk_adj: dict[str, tuple[int, int]] = {}
    for group_name in GROUPS:
        results = group_results[group_name]
        if not results:
            continue
        best_by_cagr[group_name] = max(
            ((r["entry"], r["exit"]) for r in results),
            key=lambda p: next(r["cagr_pct"] for r in results if (r["entry"], r["exit"]) == p),
        )
        best_by_risk_adj[group_name] = max(
            ((r["entry"], r["exit"]) for r in results),
            key=lambda p: next(
                r["profit_factor"] * math.sqrt(max(r["trades"], 1))
                for r in results if (r["entry"], r["exit"]) == p
            ),
        )

    print(f"\n  Best periods by CAGR:")
    for gn, bp in best_by_cagr.items():
        print(f"    {gn:<12} : {bp[0]}/{bp[1]}")
    print(f"\n  Best periods by risk-adjusted (PF * sqrt(N)):")
    for gn, bp in best_by_risk_adj.items():
        print(f"    {gn:<12} : {bp[0]}/{bp[1]}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: Combined Portfolio — Test Key Configurations
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print("  PHASE 2: Combined Portfolio with Grouped Channel Periods")
    print(f"{'=' * 80}\n")

    # Build combined configs to test
    configs = [
        ("BASELINE (all 10/5)", {g: (10, 5) for g in GROUPS}),
        ("ALL 20/10", {g: (20, 10) for g in GROUPS}),
        ("ALL 55/27", {g: (55, 27) for g in GROUPS}),
        ("BEST_PF per group", {g: best_period[g] for g in GROUPS}),
        ("BEST_CAGR per group", {g: best_by_cagr[g] for g in GROUPS}),
        ("BEST_RISK_ADJ per group", {g: best_by_risk_adj[g] for g in GROUPS}),
        ("24h 20/10 + US 10/5", {"CRYPTO": (20, 10), "COMMODITY": (20, 10),
                                  "MEGA_ETF": (10, 5), "HIGH_BETA": (10, 5)}),
        ("24h 55/27 + US 10/5", {"CRYPTO": (55, 27), "COMMODITY": (55, 27),
                                  "MEGA_ETF": (10, 5), "HIGH_BETA": (10, 5)}),
        ("24h 20/10 + Mega 20/10 + HiBeta 10/5",
         {"CRYPTO": (20, 10), "COMMODITY": (20, 10),
          "MEGA_ETF": (20, 10), "HIGH_BETA": (10, 5)}),
        ("CRYPTO 20/10 + COMMODITY 10/5 + US 10/5",
         {"CRYPTO": (20, 10), "COMMODITY": (10, 5),
          "MEGA_ETF": (10, 5), "HIGH_BETA": (10, 5)}),
        ("CRYPTO 55/27 + rest 10/5",
         {"CRYPTO": (55, 27), "COMMODITY": (10, 5),
          "MEGA_ETF": (10, 5), "HIGH_BETA": (10, 5)}),
        ("24h 20/10 + Mega 20/10 + HiBeta 20/10",
         {"CRYPTO": (20, 10), "COMMODITY": (20, 10),
          "MEGA_ETF": (20, 10), "HIGH_BETA": (20, 10)}),
    ]

    combined_results = []

    for config_label, config_periods in configs:
        print(f"  Config: {config_label}")
        for g, (e, x) in config_periods.items():
            print(f"    {g:<12} : {e}/{x}")

        # Merge candidate lists
        merged_candidates = []
        for group_name in GROUPS:
            ep = config_periods[group_name]
            cands = group_candidates[group_name].get(ep)
            if cands:
                merged_candidates.extend(cands)

        if not merged_candidates:
            print(f"    SKIP: no candidates\n")
            continue

        result = _run_portfolio(all_symbols, merged_candidates, macro_state, config_label)
        s = result["summary"]
        asset_stats = _per_asset_stats(result["trades"], group_map)
        combined_results.append({
            "label": config_label,
            "config": {g: f"{e}/{x}" for g, (e, x) in config_periods.items()},
            "trades": s.get("executed_trades", 0),
            "return_pct": s.get("total_return_pct", 0),
            "cagr_pct": s.get("cagr_pct", 0),
            "max_dd_pct": s.get("max_realized_drawdown_pct", 0),
            "profit_factor": s.get("profit_factor", 0),
            "win_rate_pct": s.get("win_rate_pct", 0),
            "final_equity": s.get("final_equity", 0),
            "per_asset": asset_stats,
            "per_group_pnl": {},
        })

        # Compute per-group aggregates
        for gn in GROUPS:
            grp_pnl = sum(st["total_pnl"] for t, st in asset_stats.items() if st["group"] == gn)
            grp_trades = sum(st["n"] for t, st in asset_stats.items() if st["group"] == gn)
            grp_wins = sum(st["wins"] for t, st in asset_stats.items() if st["group"] == gn)
            combined_results[-1]["per_group_pnl"][gn] = {
                "pnl": round(grp_pnl, 2),
                "trades": grp_trades,
                "win_rate": round(grp_wins / grp_trades * 100, 1) if grp_trades > 0 else 0,
            }

        print(f"    -> Trades: {s.get('executed_trades',0)}  "
              f"Return: {s.get('total_return_pct',0):+.2f}%  "
              f"CAGR: {s.get('cagr_pct',0):.2f}%  "
              f"MaxDD: {s.get('max_realized_drawdown_pct',0):.2f}%  "
              f"PF: {s.get('profit_factor',0):.2f}\n")

    # ── print Phase 2 comparison table ────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  PHASE 2 RESULTS: Combined Portfolio Comparison")
    print(f"{'=' * 120}")

    print(f"\n  {'Config':<45}  {'Trades':>6}  {'Return%':>10}  {'CAGR%':>8}  "
          f"{'MaxDD%':>7}  {'PF':>6}  {'WR%':>6}  {'Final$':>12}")
    print(f"  {'-' * 110}")
    for r in combined_results:
        print(f"  {r['label']:<45}  {r['trades']:>6}  {r['return_pct']:>+9.2f}%  "
              f"{r['cagr_pct']:>7.2f}%  {r['max_dd_pct']:>6.2f}%  "
              f"{r['profit_factor']:>6.2f}  {r['win_rate_pct']:>5.1f}%  "
              f"${r['final_equity']:>10,.2f}")

    # ── per-group P&L breakdown for each config ───────────────────────────────
    print(f"\n  Per-Group P&L Breakdown:")
    print(f"  {'Config':<45}  ", end="")
    for gn in GROUPS:
        print(f"  {gn:>12}", end="")
    print()
    print(f"  {'-' * 105}")
    for r in combined_results:
        print(f"  {r['label']:<45}  ", end="")
        for gn in GROUPS:
            pnl = r["per_group_pnl"].get(gn, {}).get("pnl", 0)
            print(f"  ${pnl:>10,.2f}", end="")
        print()

    # ── per-asset detail for best config ──────────────────────────────────────
    if combined_results:
        best = max(combined_results, key=lambda r: r["profit_factor"])
        print(f"\n\n  BEST CONFIG BY PF: {best['label']}")
        print(f"  {'Ticker':<14} {'Group':<10} {'Trades':>6}  {'WR%':>6}  "
              f"{'PF':>6}  {'TotPnL':>10}  {'L.PnL':>9}  {'S.PnL':>9}")
        print(f"  {'-' * 85}")
        for ticker, st in sorted(best["per_asset"].items(), key=lambda x: -x[1]["total_pnl"]):
            wr_s = f"{st['win_rate_pct']:.1f}%" if st["win_rate_pct"] else "-"
            pf_s = (f"{st['profit_factor']:.2f}"
                    if st["profit_factor"] and st["profit_factor"] != float("inf")
                    else "inf")
            print(f"  {ticker:<14} {st['group']:<10} {st['n']:>6}  {wr_s:>6}  "
                  f"{pf_s:>6}  {st['total_pnl']:>+10.2f}  "
                  f"{st['long_pnl']:>+9.2f}  {st['short_pnl']:>+9.2f}")

    # ── save results ──────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "groups": {
            gn: {
                "tickers": [s.engine_ticker for s in syms],
                "description": {
                    "CRYPTO": "24/7 crypto assets",
                    "COMMODITY": "Near-24h commodities and precious metals",
                    "MEGA_ETF": "US mega-caps and broad ETFs",
                    "HIGH_BETA": "US high-beta / mid-cap equities",
                }.get(gn, ""),
            }
            for gn, syms in group_runnable.items()
        },
        "channel_periods_tested": [f"{e}/{x}" for e, x in CHANNEL_PERIODS],
        "phase1_isolated": {
            gn: {
                "results": group_results[gn],
                "best_by_pf": f"{best_period[gn][0]}/{best_period[gn][1]}",
                "best_by_cagr": f"{best_by_cagr.get(gn, (10,5))[0]}/{best_by_cagr.get(gn, (10,5))[1]}",
                "best_by_risk_adj": f"{best_by_risk_adj.get(gn, (10,5))[0]}/{best_by_risk_adj.get(gn, (10,5))[1]}",
            }
            for gn in GROUPS
        },
        "phase2_combined": [
            {k: v for k, v in r.items() if k != "per_asset"}
            for r in combined_results
        ],
        "phase2_best_config": {
            "label": best["label"] if combined_results else None,
            "per_asset": best.get("per_asset") if combined_results else None,
        },
    }

    json_path = OUTPUT_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    print(f"\n{'=' * 80}")
    print("  DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
