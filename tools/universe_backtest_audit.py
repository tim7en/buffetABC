"""Universe Backtest Audit — Session Turtle X3 (live config)

Runs the live x3 DD-15/25 strategy against multiple universe configurations.
QQQ, SPY, TSM, AAPL are treated as SOON-TO-BE-ADDED assets and included in the
baseline FULL universe (27 symbols).  Variants slice from that full universe.

  1. FULL_27        — 27 symbols (23 live + QQQ, SPY, TSM, AAPL)   ← new baseline
  2. ORIGINAL_23    — 23 symbols (current live universe, reference)
  3. HIGHBETA_ONLY  — crypto, crypto-proxy equities, precious metals (no ETFs/energy)
  4. EQUITIES_ONLY  — all equities from full 27
  5. NO_BROAD_ETF   — full 27 minus SPY/QQQ (keep TSM, AAPL, EWJ, EWY)
  6. NO_DRAG        — full 27 minus laggards: ETH, EWJ, GOOGL, AMZN, NATGAS
  7. HIGHBETA_PLUS  — high-beta core + TSM + AAPL (selective tech add)

For each universe: full summary stats + per-asset PnL breakdown + volatility tiers.
Saves:
  reports/universe_backtest_audit_20260403/summary.json
  reports/universe_backtest_audit_20260403/report.md
"""
from __future__ import annotations

import csv
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
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)
from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
    RequestedSymbol,
    _load_macro_state,
    _patched_document_macro_scope,
    _patched_document_universe,
    _resolve_runnable_symbols,
)

# ── live x3 parameters (unchanged from production) ────────────────────────────
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

# ── extra symbols (recently added / tested) ───────────────────────────────────
EXTRA_SYMBOLS = (
    RequestedSymbol("QQQ",  "QQQ",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("SPY",  "SPY",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSM",  "TSM",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("AAPL", "AAPL", "tiingo", ("new_york_equity_open",), "equity"),
)

# ── volatility tier helpers ────────────────────────────────────────────────────
def _vol_tier(v: float | None) -> str:
    if v is None: return "unknown"
    if v >= 80:   return "extreme"
    if v >= 50:   return "very_high"
    if v >= 30:   return "high"
    if v >= 15:   return "medium"
    return "low"


def _ann_vol(engine_ticker: str, source: str) -> float | None:
    try:
        if source == "binance":
            bmap = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
                    "SOL-USD": "SOLUSDT", "PAXG-USD": "PAXGUSDT"}
            bt = bmap.get(engine_ticker)
            if not bt: return None
            gz_files = list((ROOT / "cache" / "binance_asia_orb").glob(f"{bt}_*.csv.gz"))
            if not gz_files: return None
            df = pd.concat([pd.read_csv(f, compression="gzip") for f in gz_files])
            time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
            close_col = next((c for c in df.columns if c.lower() in ("close","c")), "close")
            df["_dt"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
            close = df.set_index("_dt").sort_index()[close_col].astype(float)
        else:
            stem = engine_ticker.replace("-USD", "")
            pq = ROOT / "cache" / "cache" / "tiingo" / f"{stem}_5m.parquet"
            if not pq.exists(): return None
            df = pd.read_parquet(pq)
            time_col = next((c for c in df.columns if c.lower() in ("time","ts")), None)
            close_col = next((c for c in df.columns if c.lower() in ("close","c")), None)
            if not time_col or not close_col: return None
            df["_dt"] = pd.to_datetime(df[time_col], utc=True)
            close = df.set_index("_dt").sort_index()[close_col].astype(float)

        daily = close.resample("1D").last().dropna()
        ret   = np.log(daily / daily.shift(1)).dropna()
        return round(float(ret.std()) * math.sqrt(252) * 100, 1)
    except Exception:
        return None


# ── per-asset stats from trade list ───────────────────────────────────────────
def _per_asset_stats(trades: list[dict]) -> dict[str, dict]:
    s: dict[str, dict] = defaultdict(lambda: dict(
        n=0, wins=0, losses=0, gross_win=0.0, gross_loss=0.0,
        total_pnl=0.0, total_notional=0.0, bucket="?",
        long_n=0, short_n=0, long_pnl=0.0, short_pnl=0.0,
    ))
    for t in trades:
        k = t["ticker"]
        pnl = float(t["net_pnl"])
        notional = abs(float(t["notional"]))
        s[k]["n"] += 1
        s[k]["total_pnl"] += pnl
        s[k]["total_notional"] += notional
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
            s[k]["losses"] += 1
            s[k]["gross_loss"] += abs(pnl)
    return dict(s)


# ── run one universe at x3 live config ────────────────────────────────────────
def _run_one(runnable: list[RequestedSymbol], macro_state: dict, label: str) -> dict:
    print(f"  [{label}] running {len(runnable)} symbols …", flush=True)
    with _patched_document_universe(runnable) as runnable_universe, \
         _patched_document_macro_scope():

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
            precomputed_candidates=candidates,
            extended_hours_proxy_state=macro_state,
            per_asset_technical_state=tech_state,
            **X3_PARAMS,
        )

    trades = result["trades"]
    summary = result["summary"]
    asset_stats = _per_asset_stats(trades)

    # enrich per-asset with vol
    for ticker, st in asset_stats.items():
        src = next((r.source for r in runnable if r.engine_ticker == ticker), "tiingo")
        vol = _ann_vol(ticker, src)
        st["ann_vol_pct"] = vol
        st["vol_tier"]    = _vol_tier(vol)
        st["win_rate_pct"] = round(st["wins"] / st["n"] * 100, 1) if st["n"] > 0 else None
        st["profit_factor"] = round(st["gross_win"] / st["gross_loss"], 2) \
            if st["gross_loss"] > 0 else (float("inf") if st["gross_win"] > 0 else None)
        st["avg_notional"] = round(st["total_notional"] / st["n"], 2) if st["n"] > 0 else None
        st["total_pnl"]   = round(st["total_pnl"], 2)
        st["long_pnl"]    = round(st["long_pnl"], 2)
        st["short_pnl"]   = round(st["short_pnl"], 2)

    # bucket-level roll-up
    bucket_stats: dict[str, dict] = defaultdict(lambda: dict(
        trades=0, total_pnl=0.0, gross_win=0.0, gross_loss=0.0, wins=0,
    ))
    for st in asset_stats.values():
        b = st["bucket"]
        bucket_stats[b]["trades"]     += st["n"]
        bucket_stats[b]["total_pnl"]  += st["total_pnl"]
        bucket_stats[b]["gross_win"]  += st["gross_win"]
        bucket_stats[b]["gross_loss"] += st["gross_loss"]
        bucket_stats[b]["wins"]       += st["wins"]
    for b, bs in bucket_stats.items():
        bs["win_rate_pct"]  = round(bs["wins"] / bs["trades"] * 100, 1) if bs["trades"] else None
        bs["profit_factor"] = round(bs["gross_win"] / bs["gross_loss"], 2) \
            if bs["gross_loss"] > 0 else (float("inf") if bs["gross_win"] > 0 else None)
        bs["total_pnl"] = round(bs["total_pnl"], 2)

    # vol-tier-level roll-up
    tier_stats: dict[str, dict] = defaultdict(lambda: dict(
        assets=0, trades=0, total_pnl=0.0, gross_win=0.0, gross_loss=0.0, wins=0,
    ))
    for st in asset_stats.values():
        t = st["vol_tier"]
        tier_stats[t]["assets"]     += 1
        tier_stats[t]["trades"]     += st["n"]
        tier_stats[t]["total_pnl"]  += st["total_pnl"]
        tier_stats[t]["gross_win"]  += st["gross_win"]
        tier_stats[t]["gross_loss"] += st["gross_loss"]
        tier_stats[t]["wins"]       += st["wins"]
    for t, ts in tier_stats.items():
        ts["win_rate_pct"]  = round(ts["wins"] / ts["trades"] * 100, 1) if ts["trades"] else None
        ts["profit_factor"] = round(ts["gross_win"] / ts["gross_loss"], 2) \
            if ts["gross_loss"] > 0 else (float("inf") if ts["gross_win"] > 0 else None)
        ts["total_pnl"] = round(ts["total_pnl"], 2)

    return {
        "label":        label,
        "symbol_count": len(runnable),
        "tickers":      [r.requested_ticker for r in runnable],
        "summary":      {
            k: v for k, v in summary.items()
            if not isinstance(v, (list, dict))
        },
        "per_asset":    asset_stats,
        "by_bucket":    dict(bucket_stats),
        "by_vol_tier":  dict(tier_stats),
    }


# ── build universe subsets ────────────────────────────────────────────────────
def _build_universes(
    orig: list[RequestedSymbol],
    extra: list[RequestedSymbol],
) -> list[tuple[str, list[RequestedSymbol]]]:

    # deduplicated full 27-symbol universe (orig 23 + 4 new)
    seen = {r.engine_ticker for r in orig}
    new_only = [r for r in extra if r.engine_ticker not in seen]
    full27 = orig + new_only          # ← new baseline (27 symbols)

    # helper sets (by engine_ticker)
    high_beta_tickers = {
        "BTC-USD","ETH-USD","SOL-USD","PAXG-USD",   # crypto + crypto-gold
        "MSTR","COIN","CRCL",                         # crypto-proxy
        "SLV","PPLT","XPD-USD","COPPER-USD",          # metals
    }
    equity_tickers_full = {r.engine_ticker for r in full27 if r.asset_bucket == "equity"}
    broad_etf_tickers   = {"QQQ","SPY"}
    drag_tickers        = {"ETH-USD","EWJ","GOOGL","AMZN","NATGAS-USD"}
    highbeta_plus_extra = high_beta_tickers | {"TSM","AAPL"}   # selective tech add

    def _drop(pool: list[RequestedSymbol], exclude: set[str]) -> list[RequestedSymbol]:
        return [r for r in pool if r.engine_ticker not in exclude]

    def _pick(pool: list[RequestedSymbol], include: set[str]) -> list[RequestedSymbol]:
        return [r for r in pool if r.engine_ticker in include]

    return [
        ("FULL_27",        full27),
        ("ORIGINAL_23",    orig),
        ("HIGHBETA_ONLY",  _pick(full27, high_beta_tickers)),
        ("EQUITIES_ONLY",  _pick(full27, equity_tickers_full)),
        ("NO_BROAD_ETF",   _drop(full27, broad_etf_tickers)),
        ("NO_DRAG",        _drop(full27, drag_tickers)),
        ("HIGHBETA_PLUS",  _pick(full27, highbeta_plus_extra)),
    ]


# ── console pretty-print ──────────────────────────────────────────────────────
def _print_universe_result(r: dict) -> None:
    s = r["summary"]
    print(f"\n  {'─'*80}")
    print(f"  Universe : {r['label']}  ({r['symbol_count']} symbols)")
    print(f"  Tickers  : {', '.join(r['tickers'])}")
    print(f"  {'─'*80}")
    print(f"  {'Return':>10}  {'CAGR':>8}  {'MaxDD':>7}  {'PF':>6}  {'WR%':>6}  "
          f"{'Trades':>7}  {'FinalEq':>12}")
    fe   = s.get("final_equity",             0)
    ret  = s.get("total_return_pct",         0)
    cagr = s.get("cagr_pct",                 0)
    dd   = s.get("max_realized_drawdown_pct",0)
    pf   = s.get("profit_factor",            0)
    wr   = s.get("win_rate_pct",             0)
    tr   = s.get("executed_trades",          0)
    print(f"  {ret:>+9.2f}%  {cagr:>7.2f}%  {dd:>6.2f}%  {pf:>6.2f}  {wr:>5.1f}%  "
          f"{tr:>7}  ${fe:>10,.2f}")

    # bucket breakdown
    print(f"\n  {'Bucket':<10}  {'Trades':>6}  {'PnL':>10}  {'WR%':>6}  {'PF':>6}")
    print(f"  {'─'*50}")
    for b, bs in sorted(r["by_bucket"].items(), key=lambda x: -x[1]["total_pnl"]):
        wr_s = f"{bs['win_rate_pct']:.0f}%" if bs["win_rate_pct"] else " — "
        pf_s = f"{bs['profit_factor']:.2f}" if bs["profit_factor"] and bs["profit_factor"] != float("inf") else "inf"
        print(f"  {b:<10}  {bs['trades']:>6}  {bs['total_pnl']:>+10.2f}  {wr_s:>6}  {pf_s:>6}")

    # vol tier breakdown
    tier_order = ["extreme","very_high","high","medium","low","unknown"]
    print(f"\n  {'VolTier':<12}  {'Assets':>6}  {'Trades':>6}  {'PnL':>10}  {'WR%':>6}  {'PF':>6}")
    print(f"  {'─'*60}")
    for t in tier_order:
        if t not in r["by_vol_tier"]: continue
        ts = r["by_vol_tier"][t]
        wr_s = f"{ts['win_rate_pct']:.0f}%" if ts["win_rate_pct"] else " — "
        pf_s = f"{ts['profit_factor']:.2f}" if ts["profit_factor"] and ts["profit_factor"] != float("inf") else "inf"
        print(f"  {t:<12}  {ts['assets']:>6}  {ts['trades']:>6}  {ts['total_pnl']:>+10.2f}  {wr_s:>6}  {pf_s:>6}")

    # per-asset detail
    print(f"\n  {'Ticker':<13}  {'Vol%':>5}  {'Tier':<10}  {'N':>5}  {'WR%':>6}  "
          f"{'PF':>6}  {'TotPnL':>10}  {'L.PnL':>9}  {'S.PnL':>9}")
    print(f"  {'─'*90}")
    for ticker, st in sorted(r["per_asset"].items(), key=lambda x: -x[1]["total_pnl"]):
        vol_s = f"{st['ann_vol_pct']:.0f}%" if st["ann_vol_pct"] else " n/a"
        wr_s  = f"{st['win_rate_pct']:.0f}%" if st["win_rate_pct"] else " — "
        pf_s  = f"{st['profit_factor']:.2f}" if st["profit_factor"] and st["profit_factor"] != float("inf") else "inf"
        print(f"  {ticker:<13}  {vol_s:>5}  {st['vol_tier']:<10}  {st['n']:>5}  {wr_s:>6}  "
              f"{pf_s:>6}  {st['total_pnl']:>+10.2f}  {st['long_pnl']:>+9.2f}  {st['short_pnl']:>+9.2f}")


# ── markdown report ───────────────────────────────────────────────────────────
def _build_markdown(all_results: list[dict]) -> str:
    lines = [
        "# Universe Backtest Audit — Session Turtle X3",
        "",
        f"**Strategy:** x3 exposure, DD Governor 15%/25%, live production config  ",
        f"**New assets (soon-to-add):** QQQ, SPY, TSM, AAPL included in FULL_27 baseline  ",
        f"**Run date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Universes tested:** {len(all_results)}",
        "",
        "## Universe Comparison",
        "",
        "| Universe | Symbols | Trades | Final $ | Return% | CAGR% | MaxDD% | PF | WR% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_results:
        s = r["summary"]
        lines.append(
            f"| {r['label']} | {r['symbol_count']} "
            f"| {s.get('executed_trades',0)} "
            f"| ${s.get('final_equity',0):,.2f} "
            f"| {s.get('total_return_pct',0):+.2f}% "
            f"| {s.get('cagr_pct',0):.2f}% "
            f"| {s.get('max_realized_drawdown_pct',0):.2f}% "
            f"| {s.get('profit_factor',0):.2f} "
            f"| {s.get('win_rate_pct',0):.1f}% |"
        )

    for r in all_results:
        s = r["summary"]
        lines += [
            "",
            f"---",
            f"## {r['label']}  ({r['symbol_count']} symbols)",
            "",
            f"**Tickers:** {', '.join(r['tickers'])}",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total Return | {s.get('total_return_pct',0):+.2f}% |",
            f"| CAGR | {s.get('cagr_pct',0):.2f}% |",
            f"| Max Drawdown | {s.get('max_realized_drawdown_pct',0):.2f}% |",
            f"| Profit Factor | {s.get('profit_factor',0):.2f} |",
            f"| Win Rate | {s.get('win_rate_pct',0):.1f}% |",
            f"| Executed Trades | {s.get('executed_trades',0)} |",
            f"| Long Trades | {s.get('long_trades',0)} |",
            f"| Short Trades | {s.get('short_trades',0)} |",
            f"| Final Equity | ${s.get('final_equity',0):,.2f} |",
            f"| Entries @ Base Exposure | {s.get('entries_at_base_exposure',0)} |",
            f"| Entries @ DD Exposure 1 | {s.get('entries_at_drawdown_exposure_1',0)} |",
            f"| Entries @ DD Exposure 2 | {s.get('entries_at_drawdown_exposure_2',0)} |",
            "",
            "### Bucket Breakdown",
            "",
            "| Bucket | Trades | Total PnL | WR% | PF |",
            "|---|---:|---:|---:|---:|",
        ]
        for b, bs in sorted(r["by_bucket"].items(), key=lambda x: -x[1]["total_pnl"]):
            pf_s = f"{bs['profit_factor']:.2f}" if bs["profit_factor"] and bs["profit_factor"] != float("inf") else "inf"
            wr_s = f"{bs['win_rate_pct']:.1f}%" if bs["win_rate_pct"] else "—"
            lines.append(f"| {b} | {bs['trades']} | {bs['total_pnl']:+,.2f} | {wr_s} | {pf_s} |")

        lines += [
            "",
            "### Volatility Tier Breakdown",
            "",
            "| Tier | Assets | Trades | Total PnL | WR% | PF |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        tier_order = ["extreme","very_high","high","medium","low","unknown"]
        for t in tier_order:
            if t not in r["by_vol_tier"]: continue
            ts = r["by_vol_tier"][t]
            pf_s = f"{ts['profit_factor']:.2f}" if ts["profit_factor"] and ts["profit_factor"] != float("inf") else "inf"
            wr_s = f"{ts['win_rate_pct']:.1f}%" if ts["win_rate_pct"] else "—"
            lines.append(f"| {t} | {ts['assets']} | {ts['trades']} | {ts['total_pnl']:+,.2f} | {wr_s} | {pf_s} |")

        lines += [
            "",
            "### Per-Asset Performance",
            "",
            "| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for ticker, st in sorted(r["per_asset"].items(), key=lambda x: -x[1]["total_pnl"]):
            vol_s = f"{st['ann_vol_pct']:.0f}%" if st["ann_vol_pct"] else "n/a"
            wr_s  = f"{st['win_rate_pct']:.1f}%" if st["win_rate_pct"] is not None else "—"
            pf_s  = f"{st['profit_factor']:.2f}" if st["profit_factor"] and st["profit_factor"] != float("inf") else "inf"
            lines.append(
                f"| {ticker} | {vol_s} | {st['vol_tier']} "
                f"| {st['n']} | {wr_s} | {pf_s} "
                f"| {st['total_pnl']:+,.2f} | {st['long_pnl']:+,.2f} | {st['short_pnl']:+,.2f} |"
            )

    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "universe_backtest_audit_20260403"
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_state = _load_macro_state(ROOT)

    # resolve base universes
    orig_runnable, missing = _resolve_runnable_symbols()
    print(f"Original live universe : {len(orig_runnable)} symbols, {len(missing)} missing")

    from edgar.services.local_tiingo_data import available_tiingo_symbols
    tiingo_av = available_tiingo_symbols()
    extra_runnable = [s for s in EXTRA_SYMBOLS if s.engine_ticker in tiingo_av]

    print(f"New symbols (soon-to-add): {[s.requested_ticker for s in extra_runnable]}")
    print(f"Full 27-symbol universe : {len(orig_runnable) + len(extra_runnable)} symbols\n")

    universes = _build_universes(orig_runnable, extra_runnable)
    print(f"Running {len(universes)} universe configurations at x3 live params ...\n")

    all_results = []
    for label, syms in universes:
        if not syms:
            print(f"  [{label}] SKIPPED — no symbols resolved")
            continue
        result = _run_one(syms, macro_state, label)
        _print_universe_result(result)
        all_results.append(result)

    # ── summary comparison table ───────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  UNIVERSE COMPARISON SUMMARY")
    print(f"{'='*100}")
    hdr = (f"  {'Universe':<20}  {'Syms':>4}  {'Trades':>6}  {'Return%':>10}  "
           f"{'CAGR%':>8}  {'MaxDD%':>7}  {'PF':>6}  {'WR%':>6}  {'Final$':>12}")
    print(hdr)
    print("  " + "-" * 90)

    orig_cagr = None
    for r in all_results:
        s = r["summary"]
        cagr = s.get("cagr_pct", 0)
        if r["label"] == "ORIGINAL":
            orig_cagr = cagr
        delta = f"  (Δ{cagr - orig_cagr:+.1f}%)" if orig_cagr is not None and r["label"] != "ORIGINAL" else ""
        print(f"  {r['label']:<20}  {r['symbol_count']:>4}  "
              f"{s.get('executed_trades',0):>6}  "
              f"{s.get('total_return_pct',0):>+9.2f}%  "
              f"{cagr:>7.2f}%  "
              f"{s.get('max_realized_drawdown_pct',0):>6.2f}%  "
              f"{s.get('profit_factor',0):>6.2f}  "
              f"{s.get('win_rate_pct',0):>5.1f}%  "
              f"${s.get('final_equity',0):>10,.2f}{delta}")

    # ── save ───────────────────────────────────────────────────────────────
    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": run_ts, "universes": all_results}, f, indent=2, default=str)
    print(f"\n  JSON saved  -> {json_path}")

    md_path = out_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(all_results))
    print(f"  Markdown saved -> {md_path}")

    # save per-universe trades CSV
    for r in all_results:
        lbl = r["label"].lower()
        asset_path = out_dir / f"per_asset_{lbl}.json"
        with open(asset_path, "w", encoding="utf-8") as f:
            json.dump(r["per_asset"], f, indent=2, default=str)
    print(f"  Per-asset JSONs saved -> {out_dir}/per_asset_*.json")


if __name__ == "__main__":
    main()
