"""Donchian Group Backtest — Session Turtle X3

Tests the dual-period Donchian configuration:

  Group A  (channel_period=10, exit_channel_period=5 ) — high-vol trending assets
  Group B  (channel_period=20, exit_channel_period=10) — large/mega-caps + ETFs (lower vol)

Three runs are compared:

  BASELINE     — all assets at period=10/5            (current production config)
  DUAL_PERIOD  — Group A at 10/5,  Group B at 20/10   (proposed new config)
  GROUP_B_ONLY — Group B assets only at 20/10         (isolated view)

Group B: NVDA, COPPER-USD, AMZN, GOOGL, EWJ, EWY, QQQ, SPY, AAPL, TSM
Group A: All remaining assets (BTC, ETH, SOL, PAXG, SLV, XPD, PPLT, BRENT,
         NATGAS, COIN, CRCL, HOOD, INTC, META, MSTR, PLTR, TSLA)

New additions vs prior 23-asset universe: QQQ, SPY, AAPL, TSM

Saves:
  reports/donchian_group_backtest_20260403/summary.json
  reports/donchian_group_backtest_20260403/report.md
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
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)
from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
    RequestedSymbol,
    _load_macro_state,
    _patched_document_macro_scope,
    _patched_document_universe,
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

# ── group definitions ─────────────────────────────────────────────────────────
# Group B = large/mega-caps + ETFs (lower vol, less trend-following friendly)
GROUP_B_TICKERS = {"NVDA", "COPPER-USD", "AMZN", "GOOGL", "EWJ", "EWY",
                   "QQQ", "SPY", "AAPL", "TSM"}

# Channel config per group
GROUP_A_ENTRY = 10
GROUP_A_EXIT  = 5
GROUP_B_ENTRY = 20
GROUP_B_EXIT  = 10  # default for channel_period=20; None also works (engine derives it)


# ── vol helpers ───────────────────────────────────────────────────────────────
def _vol_tier(v: float | None) -> str:
    if v is None:   return "unknown"
    if v >= 80:     return "extreme"
    if v >= 50:     return "very_high"
    if v >= 30:     return "high"
    if v >= 15:     return "medium"
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


# ── per-asset stats ───────────────────────────────────────────────────────────
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


# ── build candidates for a subset of symbols at a given channel period ────────
def _build_candidates_for_group(
    runnable: list[RequestedSymbol],
    macro_state: dict,
    channel_period: int,
    exit_channel_period: int,
    group_label: str,
    combo_idx_offset: int = 0,
) -> list[dict]:
    """Run individual per-ticker backtests and collect raw candidates."""
    from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest
    from edgar.services.session_turtle_portfolio import _asset_bucket

    print(f"    [{group_label}] {len(runnable)} symbols  "
          f"channel={channel_period}/{exit_channel_period} …", flush=True)

    candidates: list[dict] = []
    combo_idx = combo_idx_offset

    for sym in runnable:
        assert sym.engine_ticker is not None
        for session_open in sym.session_opens:
            print(f"      {sym.engine_ticker:<14} ({session_open})", flush=True)
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
                print(f"        WARN: {sym.engine_ticker}/{session_open} failed — {exc}")
                combo_idx += 1
                continue

            for trade_idx, trade in enumerate(payload["trades"]):
                import datetime as _dt_mod
                entry_price = float(trade["entry_price"])
                stop_loss = trade.get("stop_loss")
                if stop_loss is None:
                    if str(trade["direction"]) == "long":
                        stop_loss = entry_price * (1.0 - 0.10)
                    else:
                        stop_loss = entry_price * (1.0 + 0.10)
                candidates.append({
                    "combo_idx":              combo_idx,
                    "trade_idx":              trade_idx,
                    "ticker":                 sym.engine_ticker,
                    "source":                 sym.source,
                    "session_open":           session_open,
                    "direction":              trade["direction"],
                    "entry_ts":               _dt_mod.datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts":                _dt_mod.datetime.fromisoformat(trade["exit_date"]),
                    "entry_price":            entry_price,
                    "exit_price":             float(trade["exit_price"]),
                    "stop_loss":              float(stop_loss),
                    "risk_pct":               float(trade.get("risk_pct", 0.0) or 0.0),
                    "shares":                 float(trade["shares"]),
                    "position_size":          float(trade["position_size"]),
                    "pnl":                    float(trade["pnl"]),
                    "risk_model":             str(trade["risk_model"]),
                    "entry_rel_volume":       float(trade["entry_rel_volume"]),
                    "rel_volume_ratio":       float(trade.get("rel_volume_ratio", 1.0) or 1.0),
                    "conviction_mult":        float(trade.get("conviction_mult", 1.0) or 1.0),
                    "breakout_penetration":   float(trade.get("breakout_penetration", 0.0) or 0.0),
                    "directional_close_score":float(trade.get("directional_close_score", 0.0) or 0.0),
                    "asset_bucket":           _asset_bucket(sym.engine_ticker),
                })
            combo_idx += 1

    print(f"    [{group_label}] → {len(candidates)} raw candidates", flush=True)
    return candidates


# ── run full portfolio simulation on a candidate list ─────────────────────────
def _run_portfolio(
    all_runnable: list[RequestedSymbol],
    candidates: list[dict],
    macro_state: dict,
    label: str,
) -> dict:
    print(f"  [{label}] simulating portfolio …", flush=True)

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

    trades = result["trades"]
    summary = result["summary"]
    asset_stats = _per_asset_stats(trades)

    for ticker, st in asset_stats.items():
        src = next((r.source for r in all_runnable if r.engine_ticker == ticker), "tiingo")
        vol = _ann_vol(ticker, src)
        st["ann_vol_pct"]   = vol
        st["vol_tier"]      = _vol_tier(vol)
        st["donchian_group"] = "B" if ticker in GROUP_B_TICKERS else "A"
        st["win_rate_pct"]  = round(st["wins"] / st["n"] * 100, 1) if st["n"] > 0 else None
        st["profit_factor"] = round(st["gross_win"] / st["gross_loss"], 2) \
            if st["gross_loss"] > 0 else (float("inf") if st["gross_win"] > 0 else None)
        st["total_pnl"]     = round(st["total_pnl"], 2)
        st["long_pnl"]      = round(st["long_pnl"], 2)
        st["short_pnl"]     = round(st["short_pnl"], 2)

    return {
        "label":        label,
        "symbol_count": len(all_runnable),
        "summary":      {k: v for k, v in summary.items() if not isinstance(v, (list, dict))},
        "per_asset":    asset_stats,
    }


# ── console print ─────────────────────────────────────────────────────────────
def _print_result(r: dict, group_a_period: str, group_b_period: str) -> None:
    s = r["summary"]
    fe   = s.get("final_equity",             0)
    ret  = s.get("total_return_pct",         0)
    cagr = s.get("cagr_pct",                 0)
    dd   = s.get("max_realized_drawdown_pct",0)
    pf   = s.get("profit_factor",            0)
    wr   = s.get("win_rate_pct",             0)
    tr   = s.get("executed_trades",          0)

    print(f"\n  {'─'*80}")
    print(f"  Run      : {r['label']}  ({r['symbol_count']} symbols)")
    print(f"  Groups   : A={group_a_period}  B={group_b_period}")
    print(f"  {'─'*80}")
    print(f"  {'Return':>10}  {'CAGR':>8}  {'MaxDD':>7}  {'PF':>6}  {'WR%':>6}  "
          f"{'Trades':>7}  {'FinalEq':>12}")
    print(f"  {ret:>+9.2f}%  {cagr:>7.2f}%  {dd:>6.2f}%  {pf:>6.2f}  {wr:>5.1f}%  "
          f"{tr:>7}  ${fe:>10,.2f}")

    print(f"\n  {'Ticker':<13}  {'Grp':>3}  {'Vol%':>5}  {'N':>5}  {'WR%':>6}  "
          f"{'PF':>6}  {'TotPnL':>10}  {'L.PnL':>9}  {'S.PnL':>9}")
    print(f"  {'─'*90}")
    for ticker, st in sorted(r["per_asset"].items(), key=lambda x: -x[1]["total_pnl"]):
        vol_s = f"{st['ann_vol_pct']:.0f}%" if st["ann_vol_pct"] else " n/a"
        wr_s  = f"{st['win_rate_pct']:.0f}%" if st["win_rate_pct"] else " — "
        pf_s  = f"{st['profit_factor']:.2f}" if st["profit_factor"] and st["profit_factor"] != float("inf") else "inf"
        grp   = st.get("donchian_group", "A")
        print(f"  {ticker:<13}  {grp:>3}  {vol_s:>5}  {st['n']:>5}  {wr_s:>6}  "
              f"{pf_s:>6}  {st['total_pnl']:>+10.2f}  {st['long_pnl']:>+9.2f}  {st['short_pnl']:>+9.2f}")


# ── markdown ──────────────────────────────────────────────────────────────────
def _build_markdown(all_results: list[dict], meta: dict) -> str:
    lines = [
        "# Donchian Group Backtest — Session Turtle X3",
        "",
        f"**Group A:** period={meta['group_a_entry']}/{meta['group_a_exit']} "
        f"— {', '.join(sorted(meta['group_a_tickers']))}",
        f"**Group B:** period={meta['group_b_entry']}/{meta['group_b_exit']} "
        f"— {', '.join(sorted(meta['group_b_tickers']))}",
        f"**Run date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Comparison Summary",
        "",
        "| Run | Syms | Trades | Return% | CAGR% | MaxDD% | PF | WR% | Final$ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_results:
        s = r["summary"]
        lines.append(
            f"| {r['label']} | {r['symbol_count']} "
            f"| {s.get('executed_trades',0)} "
            f"| {s.get('total_return_pct',0):+.2f}% "
            f"| {s.get('cagr_pct',0):.2f}% "
            f"| {s.get('max_realized_drawdown_pct',0):.2f}% "
            f"| {s.get('profit_factor',0):.2f} "
            f"| {s.get('win_rate_pct',0):.1f}% "
            f"| ${s.get('final_equity',0):,.2f} |"
        )

    for r in all_results:
        s = r["summary"]
        lines += [
            "", "---",
            f"## {r['label']}  ({r['symbol_count']} symbols)",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total Return | {s.get('total_return_pct',0):+.2f}% |",
            f"| CAGR | {s.get('cagr_pct',0):.2f}% |",
            f"| Max Drawdown | {s.get('max_realized_drawdown_pct',0):.2f}% |",
            f"| Profit Factor | {s.get('profit_factor',0):.2f} |",
            f"| Win Rate | {s.get('win_rate_pct',0):.1f}% |",
            f"| Executed Trades | {s.get('executed_trades',0)} |",
            f"| Final Equity | ${s.get('final_equity',0):,.2f} |",
            "",
            "### Per-Asset Performance",
            "",
            "| Ticker | Group | Vol% | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ticker, st in sorted(r["per_asset"].items(), key=lambda x: -x[1]["total_pnl"]):
            vol_s = f"{st['ann_vol_pct']:.0f}%" if st["ann_vol_pct"] else "n/a"
            wr_s  = f"{st['win_rate_pct']:.1f}%" if st["win_rate_pct"] is not None else "—"
            pf_s  = f"{st['profit_factor']:.2f}" if st["profit_factor"] and st["profit_factor"] != float("inf") else "inf"
            lines.append(
                f"| {ticker} | {st.get('donchian_group','A')} | {vol_s} "
                f"| {st['n']} | {wr_s} | {pf_s} "
                f"| {st['total_pnl']:+,.2f} | {st['long_pnl']:+,.2f} | {st['short_pnl']:+,.2f} |"
            )

    return "\n".join(lines) + "\n"


# ── extended universe (original 23 + QQQ, SPY, AAPL, TSM) ───────────────────
EXTRA_SYMBOLS: tuple[RequestedSymbol, ...] = (
    RequestedSymbol("QQQ",  "QQQ",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("SPY",  "SPY",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("AAPL", "AAPL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSM",  "TSM",  "tiingo", ("new_york_equity_open",), "equity"),
)


def _resolve_extended_universe() -> tuple[list[RequestedSymbol], list[RequestedSymbol]]:
    """Returns (runnable, missing) for original 23 + 4 new symbols."""
    from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
        REQUESTED_SYMBOLS as BASE_SYMBOLS,
    )
    from edgar.services.local_tiingo_data import available_tiingo_symbols
    tiingo_avail = available_tiingo_symbols()
    all_symbols = list(BASE_SYMBOLS) + list(EXTRA_SYMBOLS)
    runnable, missing = [], []
    for sym in all_symbols:
        if sym.engine_ticker is None:
            missing.append(sym); continue
        if sym.source == "tiingo":
            resolved = sym.engine_ticker[:-4] if sym.engine_ticker.endswith("-USD") else sym.engine_ticker
            if resolved not in tiingo_avail:
                missing.append(sym); continue
        runnable.append(sym)
    return runnable, missing


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "donchian_group_backtest_20260403"
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_state = _load_macro_state(ROOT)
    orig_runnable, missing = _resolve_extended_universe()
    print(f"Live universe : {len(orig_runnable)} symbols  |  {len(missing)} missing")

    # split into groups
    group_a = [r for r in orig_runnable if r.engine_ticker not in GROUP_B_TICKERS]
    group_b = [r for r in orig_runnable if r.engine_ticker in GROUP_B_TICKERS]
    print(f"Group A ({GROUP_A_ENTRY}/{GROUP_A_EXIT}) : {[r.engine_ticker for r in group_a]}")
    print(f"Group B ({GROUP_B_ENTRY}/{GROUP_B_EXIT}) : {[r.engine_ticker for r in group_b]}")

    n_all = len(orig_runnable)

    # ── BASELINE: all assets at period=10/5 ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Building BASELINE candidates (all {n_all} @ 10/5) …")
    candidates_baseline = _build_candidates_for_group(
        orig_runnable, macro_state,
        channel_period=GROUP_A_ENTRY,
        exit_channel_period=GROUP_A_EXIT,
        group_label="BASELINE",
        combo_idx_offset=0,
    )

    # ── DUAL_PERIOD: Group A @ 10/5 + Group B @ 20/10 ────────────────────────
    print(f"\n{'='*60}")
    print("Building DUAL_PERIOD candidates …")
    # Reuse Group A candidates from BASELINE (filter by ticker)
    group_a_tickers = {r.engine_ticker for r in group_a}
    candidates_a = [c for c in candidates_baseline if c["ticker"] in group_a_tickers]
    print(f"  [Group A reused] {len(candidates_a)} candidates from BASELINE")

    # Group B at 20/10 — fresh run; offset combo_idx past group A range
    max_combo_a = max((c["combo_idx"] for c in candidates_a), default=-1)
    candidates_b_new = _build_candidates_for_group(
        group_b, macro_state,
        channel_period=GROUP_B_ENTRY,
        exit_channel_period=GROUP_B_EXIT,
        group_label="Group B (20/10)",
        combo_idx_offset=max_combo_a + 1,
    )
    candidates_dual = candidates_a + candidates_b_new

    # ── GROUP_B_ONLY: only group_b assets at 20/10 ───────────────────────────
    print(f"\n{'='*60}")
    print(f"Building GROUP_B_ONLY candidates ({len(group_b)} assets @ 20/10) …")
    candidates_b_only = candidates_b_new  # already computed

    # ── run portfolio simulations ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running portfolio simulations …\n")

    result_baseline = _run_portfolio(
        orig_runnable, candidates_baseline, macro_state, "BASELINE (10/5 all)")
    result_dual = _run_portfolio(
        orig_runnable, candidates_dual, macro_state, "DUAL_PERIOD (A:10/5 B:20/10)")
    result_b_only = _run_portfolio(
        group_b, candidates_b_only, macro_state, "GROUP_B_ONLY (20/10)")

    all_results = [result_baseline, result_dual, result_b_only]

    # ── comparison table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("  DONCHIAN GROUP COMPARISON")
    print(f"{'='*100}")
    hdr = (f"  {'Run':<28}  {'Trades':>6}  {'Return%':>10}  "
           f"{'CAGR%':>8}  {'MaxDD%':>7}  {'PF':>6}  {'WR%':>6}  {'Final$':>12}")
    print(hdr)
    print("  " + "-"*88)
    baseline_cagr = None
    for r in all_results:
        s = r["summary"]
        cagr = s.get("cagr_pct", 0)
        if r["label"].startswith("BASELINE"):
            baseline_cagr = cagr
        delta = (f"  Δ{cagr - baseline_cagr:+.1f}% vs baseline"
                 if baseline_cagr is not None and not r["label"].startswith("BASELINE") else "")
        print(f"  {r['label']:<28}  "
              f"{s.get('executed_trades',0):>6}  "
              f"{s.get('total_return_pct',0):>+9.2f}%  "
              f"{cagr:>7.2f}%  "
              f"{s.get('max_realized_drawdown_pct',0):>6.2f}%  "
              f"{s.get('profit_factor',0):>6.2f}  "
              f"{s.get('win_rate_pct',0):>5.1f}%  "
              f"${s.get('final_equity',0):>10,.2f}{delta}")

    for r in all_results:
        _print_result(r,
                      group_a_period=f"{GROUP_A_ENTRY}/{GROUP_A_EXIT}",
                      group_b_period=f"{GROUP_B_ENTRY}/{GROUP_B_EXIT}")

    # ── save ──────────────────────────────────────────────────────────────────
    meta = {
        "group_a_tickers": [r.engine_ticker for r in group_a],
        "group_b_tickers": [r.engine_ticker for r in group_b],
        "group_a_entry":   GROUP_A_ENTRY,
        "group_a_exit":    GROUP_A_EXIT,
        "group_b_entry":   GROUP_B_ENTRY,
        "group_b_exit":    GROUP_B_EXIT,
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": run_ts, "meta": meta, "results": all_results},
                  f, indent=2, default=str)
    print(f"\n  JSON   -> {json_path}")

    md_path = out_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(all_results, meta))
    print(f"  MD     -> {md_path}")


if __name__ == "__main__":
    main()
