"""Pruned Universe Backtest + Correlation / Stress Analysis
============================================================
Removes unprofitable assets (PF < 1.0 in BEST_PF optimisation),
re-runs the backtest with optimal per-group channel periods,
then analyses pairwise correlation and stress-period behaviour.

Kept universe (21 symbols):
  CRYPTO    (4): BTC-USD, ETH-USD, SOL-USD, PAXG-USD          channel 20/10
  COMMODITY (6): BRENT, NATGAS-USD, COPPER-USD, XPD-USD, PPLT, SLV   10/5
  MEGA_ETF  (4): GOOGL, META, TSM, EWY                        channel 20/10
  HIGH_BETA (7): COIN, CRCL, HOOD, INTC, MSTR, PLTR, TSLA    channel 10/5

Removed (PF < 1.0): AAPL, AMZN, NVDA, SPY, QQQ, EWJ

Usage:
    python tools/pruned_correlation_backtest.py
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
OUTPUT_DIR = ROOT / "reports" / "pruned_correlation_backtest_20260404"

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
#  ASSET DEFINITIONS — Pruned Universe
# ══════════════════════════════════════════════════════════════════════════════
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

# Pruned: removed AAPL, AMZN, NVDA, SPY, QQQ, EWJ
MEGA_ETF_SYMBOLS = (
    RequestedSymbol("GOOGL", "GOOGL", "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("META",  "META",  "tiingo", ("new_york_equity_open",), "equity"),
    RequestedSymbol("TSM",   "TSM",   "tiingo", ("new_york_equity_open",), "equity"),
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
    "CRYPTO":    (CRYPTO_SYMBOLS,    (20, 10)),
    "COMMODITY": (COMMODITY_SYMBOLS, (10, 5)),
    "MEGA_ETF":  (MEGA_ETF_SYMBOLS,  (20, 10)),
    "HIGH_BETA": (HIGH_BETA_SYMBOLS, (10, 5)),
}

REMOVED = ["AAPL", "AMZN", "NVDA", "SPY", "QQQ", "EWJ"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _resolve_runnable(symbols):
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


def _build_candidates(symbols, channel_period, exit_channel_period, label, combo_idx_offset=0):
    import datetime as _dt

    print(f"    [{label}] {len(symbols)} symbols  channel={channel_period}/{exit_channel_period}",
          flush=True)

    candidates = []
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


def _per_asset_stats(trades, group_map):
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
        st["group"] = group_map.get(ticker, "?")
        st["win_rate_pct"] = round(st["wins"] / st["n"] * 100, 1) if st["n"] else None
        st["profit_factor"] = (round(st["gross_win"] / st["gross_loss"], 2)
                               if st["gross_loss"] > 0
                               else (float("inf") if st["gross_win"] > 0 else None))
        st["total_pnl"] = round(st["total_pnl"], 2)
        st["long_pnl"] = round(st["long_pnl"], 2)
        st["short_pnl"] = round(st["short_pnl"], 2)
    return dict(s)


# ══════════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _load_daily_returns() -> pd.DataFrame:
    """Load daily close-to-close returns for each asset in the pruned universe."""
    all_tickers = {}

    # Tiingo assets
    tiingo_map = {
        "BRENT": "BRENT", "NATGAS-USD": "NATGAS-USD", "COPPER-USD": "COPPER-USD",
        "XPD-USD": "XPD-USD", "PPLT": "PPLT", "SLV": "SLV",
        "GOOGL": "GOOGL", "META": "META", "TSM": "TSM", "EWY": "EWY",
        "COIN": "COIN", "CRCL": "CRCL", "HOOD": "HOOD", "INTC": "INTC",
        "MSTR": "MSTR", "PLTR": "PLTR", "TSLA": "TSLA",
    }
    for ticker, stem in tiingo_map.items():
        pq = ROOT / "cache" / "cache" / "tiingo" / f"{stem}_5m.parquet"
        if not pq.exists():
            pq_alt = ROOT / "cache" / "cache" / "tiingo" / f"{stem.replace('-USD', '')}_5m.parquet"
            if pq_alt.exists():
                pq = pq_alt
            else:
                continue
        try:
            df = pd.read_parquet(pq)
            time_col = next((c for c in df.columns if c.lower() in ("time", "ts")), None)
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), None)
            if not time_col or not close_col:
                continue
            df["_dt"] = pd.to_datetime(df[time_col], utc=True)
            daily = df.set_index("_dt")[close_col].astype(float).resample("1D").last().dropna()
            ret = daily.pct_change().dropna()
            ret.name = ticker
            all_tickers[ticker] = ret
        except Exception:
            continue

    # Binance crypto
    binance_map = {
        "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
        "SOL-USD": "SOLUSDT", "PAXG-USD": "PAXGUSDT",
    }
    for ticker, bt in binance_map.items():
        gz_files = list((ROOT / "cache" / "binance_asia_orb").glob(f"{bt}_*.csv.gz"))
        if not gz_files:
            continue
        try:
            df = pd.concat([pd.read_csv(f, compression="gzip") for f in gz_files])
            time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), "close")
            df["_dt"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
            daily = df.set_index("_dt").sort_index()[close_col].astype(float).resample("1D").last().dropna()
            ret = daily.pct_change().dropna()
            ret.name = ticker
            all_tickers[ticker] = ret
        except Exception:
            continue

    returns_df = pd.DataFrame(all_tickers)
    returns_df.index = pd.to_datetime(returns_df.index)
    # Use overlapping dates only
    returns_df = returns_df.dropna(how="all")
    return returns_df


def correlation_analysis(trades_df: pd.DataFrame, group_map: dict):
    """Full + stress-period correlation analysis."""
    print("\n" + "=" * 100)
    print("  CORRELATION & STRESS ANALYSIS")
    print("=" * 100)

    returns_df = _load_daily_returns()
    n_assets = returns_df.shape[1]
    print(f"\n  Loaded daily returns for {n_assets} assets  "
          f"({returns_df.index.min().date()} → {returns_df.index.max().date()})")

    # ── 1. Full-period correlation matrix ─────────────────────────────────────
    full_corr = returns_df.corr()
    print(f"\n  --- Full-Period Correlation Matrix ---")

    # Group the matrix by asset group for display
    group_order = ["CRYPTO", "COMMODITY", "MEGA_ETF", "HIGH_BETA"]
    ticker_order = []
    for g in group_order:
        for t in full_corr.columns:
            if group_map.get(t, "?") == g and t not in ticker_order:
                ticker_order.append(t)
    # Any remaining
    for t in full_corr.columns:
        if t not in ticker_order:
            ticker_order.append(t)

    full_corr = full_corr.loc[ticker_order, ticker_order]
    returns_df = returns_df[ticker_order]

    # Print condensed version: cross-group average correlations
    print(f"\n  Cross-Group Average Correlations (full period):")
    print(f"  {'':>12}", end="")
    for g2 in group_order:
        print(f"  {g2:>12}", end="")
    print()
    for g1 in group_order:
        g1_tickers = [t for t in ticker_order if group_map.get(t) == g1]
        print(f"  {g1:>12}", end="")
        for g2 in group_order:
            g2_tickers = [t for t in ticker_order if group_map.get(t) == g2]
            if g1 == g2:
                # Intra-group: average off-diagonal
                sub = full_corr.loc[g1_tickers, g2_tickers]
                mask = np.ones_like(sub.values, dtype=bool)
                np.fill_diagonal(mask, False)
                avg = sub.values[mask].mean() if mask.sum() > 0 else 1.0
            else:
                avg = full_corr.loc[g1_tickers, g2_tickers].values.mean()
            print(f"  {avg:>12.3f}", end="")
        print()

    # ── 2. Identify stress periods ────────────────────────────────────────────
    # Build portfolio equity curve from trade-level data
    trades_sorted = trades_df.sort_values("exit_ts").reset_index(drop=True)
    initial_cap = 1_000.0
    equity = [initial_cap]
    for pnl in trades_sorted["net_pnl"]:
        equity.append(equity[-1] + float(pnl))
    equity_s = pd.Series(equity)
    running_max = equity_s.cummax()
    dd_pct = (equity_s - running_max) / running_max

    # Map equity index to dates via exit timestamps
    eq_dates = [trades_sorted["entry_ts"].iloc[0]] + list(trades_sorted["exit_ts"])
    eq_dates = pd.to_datetime(eq_dates)

    # Find periods where portfolio was in >10% drawdown
    stress_mask = dd_pct < -0.10
    stress_indices = stress_mask[stress_mask].index.tolist()
    stress_dates = set()
    for idx in stress_indices:
        if idx < len(eq_dates):
            d = eq_dates[idx]
            # Expand to +-5 business days around stress event
            for delta in range(-5, 6):
                stress_dates.add(d + pd.Timedelta(days=delta))

    # Also identify the worst 10% of portfolio daily returns as stress
    # Approximate daily portfolio returns by summing trade returns per day
    trades_sorted["exit_date"] = pd.to_datetime(trades_sorted["exit_ts"]).dt.date
    daily_pnl = trades_sorted.groupby("exit_date")["net_pnl"].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)

    # Bottom 10% of daily P&L days
    if len(daily_pnl) > 10:
        threshold = daily_pnl.quantile(0.10)
        stress_pnl_dates = set(daily_pnl[daily_pnl <= threshold].index)
        stress_dates.update(stress_pnl_dates)

    print(f"\n  Stress periods identified: {len(stress_dates)} days "
          f"(portfolio DD > 10% + worst-10% P&L days)")

    # ── 3. Stress-period correlations ─────────────────────────────────────────
    stress_dates_norm = set(pd.to_datetime(list(stress_dates)).normalize())
    ret_dates_norm = returns_df.index.normalize()
    stress_mask_ret = ret_dates_norm.isin(stress_dates_norm)
    stress_returns = returns_df.loc[stress_mask_ret]
    normal_returns = returns_df.loc[~stress_mask_ret]

    if len(stress_returns) > 5:
        stress_corr = stress_returns.corr()

        print(f"\n  Stress-Period Cross-Group Average Correlations ({len(stress_returns)} days):")
        print(f"  {'':>12}", end="")
        for g2 in group_order:
            print(f"  {g2:>12}", end="")
        print()
        for g1 in group_order:
            g1_tickers = [t for t in ticker_order if group_map.get(t) == g1]
            print(f"  {g1:>12}", end="")
            for g2 in group_order:
                g2_tickers = [t for t in ticker_order if group_map.get(t) == g2]
                if g1 == g2:
                    sub = stress_corr.loc[g1_tickers, g2_tickers]
                    mask = np.ones_like(sub.values, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg = sub.values[mask].mean() if mask.sum() > 0 else 1.0
                else:
                    avg = stress_corr.loc[g1_tickers, g2_tickers].values.mean()
                print(f"  {avg:>12.3f}", end="")
            print()

        # ── 4. Correlation change: stress vs normal ───────────────────────────
        normal_corr = normal_returns.corr()
        print(f"\n  Correlation SHIFT (Stress minus Normal):")
        print(f"  {'':>12}", end="")
        for g2 in group_order:
            print(f"  {g2:>12}", end="")
        print()
        for g1 in group_order:
            g1_tickers = [t for t in ticker_order if group_map.get(t) == g1]
            print(f"  {g1:>12}", end="")
            for g2 in group_order:
                g2_tickers = [t for t in ticker_order if group_map.get(t) == g2]
                if g1 == g2:
                    sub_s = stress_corr.loc[g1_tickers, g2_tickers]
                    sub_n = normal_corr.loc[g1_tickers, g2_tickers]
                    mask = np.ones_like(sub_s.values, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_s = sub_s.values[mask].mean() if mask.sum() > 0 else 0
                    avg_n = sub_n.values[mask].mean() if mask.sum() > 0 else 0
                    delta = avg_s - avg_n
                else:
                    avg_s = stress_corr.loc[g1_tickers, g2_tickers].values.mean()
                    avg_n = normal_corr.loc[g1_tickers, g2_tickers].values.mean()
                    delta = avg_s - avg_n
                sign = "+" if delta >= 0 else ""
                print(f"  {sign}{delta:>11.3f}", end="")
            print()
    else:
        print(f"\n  Insufficient stress-period data for correlation analysis.")

    # ── 5. Per-asset stress behaviour ─────────────────────────────────────────
    print(f"\n  Per-Asset Behaviour: Stress vs Normal")
    print(f"  {'Ticker':<14} {'Group':<12} {'Normal μ%':>10} {'Stress μ%':>10} "
          f"{'Normal σ%':>10} {'Stress σ%':>10} {'Stress Beta':>12}")
    print(f"  {'-' * 80}")

    # Portfolio daily return as benchmark
    portfolio_daily = returns_df.mean(axis=1)
    for ticker in ticker_order:
        if ticker not in returns_df.columns:
            continue
        grp = group_map.get(ticker, "?")
        normal_ret = normal_returns[ticker].dropna()
        stress_ret = stress_returns[ticker].dropna() if ticker in stress_returns.columns else pd.Series(dtype=float)

        norm_mean = normal_ret.mean() * 100 if len(normal_ret) > 0 else 0
        stress_mean = stress_ret.mean() * 100 if len(stress_ret) > 0 else 0
        norm_std = normal_ret.std() * 100 if len(normal_ret) > 0 else 0
        stress_std = stress_ret.std() * 100 if len(stress_ret) > 0 else 0

        # Beta to portfolio during stress
        if len(stress_ret) > 5:
            port_stress = portfolio_daily.loc[stress_ret.index].dropna()
            common = stress_ret.index.intersection(port_stress.index)
            if len(common) > 5:
                cov_matrix = np.cov(stress_ret.loc[common].values, port_stress.loc[common].values)
                stress_beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
            else:
                stress_beta = float("nan")
        else:
            stress_beta = float("nan")

        print(f"  {ticker:<14} {grp:<12} {norm_mean:>+9.3f}% {stress_mean:>+9.3f}% "
              f"{norm_std:>9.3f}% {stress_std:>9.3f}% {stress_beta:>11.2f}")

    # ── 6. Diversification metrics ────────────────────────────────────────────
    print(f"\n  --- Diversification Summary ---")
    # Average pairwise correlation
    upper_tri = np.triu_indices(n_assets, k=1)
    avg_corr = full_corr.values[upper_tri].mean()
    # Effective number of independent bets (Bouchaud)
    eigenvalues = np.linalg.eigvalsh(full_corr.fillna(0).values)
    eigenvalues = eigenvalues[eigenvalues > 0]
    eff_n = np.exp(-np.sum((eigenvalues / eigenvalues.sum()) *
                            np.log(eigenvalues / eigenvalues.sum())))
    # Max individual correlation
    np.fill_diagonal(full_corr.values, 0)
    max_corr_pair = np.unravel_index(np.abs(full_corr.values).argmax(), full_corr.shape)
    max_corr_val = full_corr.values[max_corr_pair]
    max_corr_t1 = full_corr.index[max_corr_pair[0]]
    max_corr_t2 = full_corr.columns[max_corr_pair[1]]

    print(f"  Average pairwise correlation:     {avg_corr:.3f}")
    print(f"  Effective independent bets:       {eff_n:.1f} / {n_assets}")
    print(f"  Most correlated pair:             {max_corr_t1} ↔ {max_corr_t2} = {max_corr_val:.3f}")

    # Pairs with |corr| > 0.5
    high_corr_pairs = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            c = full_corr.values[i, j]
            if abs(c) > 0.5:
                high_corr_pairs.append((full_corr.index[i], full_corr.columns[j], c))
    if high_corr_pairs:
        print(f"\n  High-Correlation Pairs (|ρ| > 0.5):")
        for t1, t2, c in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"    {t1:<12} ↔ {t2:<12} = {c:+.3f}")
    else:
        print(f"\n  No pairs with |ρ| > 0.5")

    return full_corr, stress_corr if len(stress_returns) > 5 else None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    macro_state = _load_macro_state(ROOT)

    # ── resolve symbols ───────────────────────────────────────────────────────
    all_symbols = []
    group_map = {}

    print("=" * 100)
    print("  PRUNED UNIVERSE BACKTEST + CORRELATION ANALYSIS")
    print("=" * 100)
    print(f"\n  Removed: {', '.join(REMOVED)}")

    for group_name, (symbols, (ep, xp)) in GROUPS.items():
        runnable = _resolve_runnable(symbols)
        all_symbols.extend(runnable)
        for sym in runnable:
            group_map[sym.engine_ticker] = group_name
        print(f"  {group_name:<12} : {len(runnable)} symbols  "
              f"channel={ep}/{xp}  "
              f"[{', '.join(s.engine_ticker for s in runnable)}]")

    kept_count = len(all_symbols)
    print(f"\n  Total kept: {kept_count}  (removed {len(REMOVED)})")

    # ── build candidates per group at optimal channel periods ─────────────────
    print(f"\n{'=' * 100}")
    print("  PHASE 1: Building Candidates (Optimal Channel Per Group)")
    print(f"{'=' * 100}\n")

    all_candidates = []
    combo_offset = 0

    for group_name, (symbols, (ep, xp)) in GROUPS.items():
        runnable = _resolve_runnable(symbols)
        cands = _build_candidates(runnable, ep, xp, group_name, combo_offset)
        all_candidates.extend(cands)
        combo_offset = max((c["combo_idx"] for c in cands), default=combo_offset) + 1

    print(f"\n  Total candidates: {len(all_candidates)}")

    # ── run portfolio ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  PHASE 2: Portfolio Simulation (Pruned Universe)")
    print(f"{'=' * 100}\n")

    result = _run_portfolio(all_symbols, all_candidates, macro_state, "PRUNED")
    s = result["summary"]
    trades = result["trades"]
    asset_stats = _per_asset_stats(trades, group_map)

    print(f"  {'Metric':<35} {'Value':>15}")
    print(f"  {'-' * 52}")
    print(f"  {'Executed Trades':<35} {s.get('executed_trades', 0):>15}")
    print(f"  {'Total Return':<35} {s.get('total_return_pct', 0):>+14.2f}%")
    print(f"  {'CAGR':<35} {s.get('cagr_pct', 0):>14.2f}%")
    print(f"  {'Max Drawdown':<35} {s.get('max_realized_drawdown_pct', 0):>14.2f}%")
    print(f"  {'Profit Factor':<35} {s.get('profit_factor', 0):>15.2f}")
    print(f"  {'Win Rate':<35} {s.get('win_rate_pct', 0):>14.1f}%")
    print(f"  {'Final Equity':<35} {'${:,.2f}'.format(s.get('final_equity', 0)):>15}")

    # Per-group breakdown
    print(f"\n  --- Per-Group Summary ---")
    print(f"  {'Group':<12} {'Trades':>7} {'P&L':>12} {'WR%':>7} {'Avg PF':>8}")
    print(f"  {'-' * 48}")
    for gn in ["CRYPTO", "COMMODITY", "MEGA_ETF", "HIGH_BETA"]:
        g_assets = {t: st for t, st in asset_stats.items() if st["group"] == gn}
        g_trades = sum(st["n"] for st in g_assets.values())
        g_pnl = sum(st["total_pnl"] for st in g_assets.values())
        g_wins = sum(st["wins"] for st in g_assets.values())
        g_wr = round(g_wins / g_trades * 100, 1) if g_trades > 0 else 0
        g_gross_win = sum(st["gross_win"] for st in g_assets.values())
        g_gross_loss = sum(st["gross_loss"] for st in g_assets.values())
        g_pf = round(g_gross_win / g_gross_loss, 2) if g_gross_loss > 0 else float("inf")
        print(f"  {gn:<12} {g_trades:>7} {g_pnl:>+11.2f} {g_wr:>6.1f}% {g_pf:>8.2f}")

    # Per-asset detail
    print(f"\n  --- Per-Asset Detail ---")
    print(f"  {'Ticker':<14} {'Group':<12} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Total P&L':>11} {'Long P&L':>10} {'Short P&L':>10}")
    print(f"  {'-' * 82}")
    for ticker in sorted(asset_stats, key=lambda t: asset_stats[t]["total_pnl"], reverse=True):
        st = asset_stats[ticker]
        pf_str = f"{st['profit_factor']:.2f}" if st["profit_factor"] != float("inf") else "  inf"
        print(f"  {ticker:<14} {st['group']:<12} {st['n']:>7} {st['win_rate_pct']:>5.1f}% "
              f"{pf_str:>7} {st['total_pnl']:>+10.2f} {st['long_pnl']:>+10.2f} "
              f"{st['short_pnl']:>+10.2f}")

    # ── Comparison with full universe ─────────────────────────────────────────
    print(f"\n  --- Comparison: Pruned vs Full 27-symbol BEST_PF ---")
    full_ref = {
        "trades": 459, "return_pct": 2556.08, "cagr_pct": 120.69,
        "max_dd_pct": 21.52, "profit_factor": 2.60, "final_equity": 26560.82,
    }
    print(f"  {'Metric':<20} {'Full (27)':>15} {'Pruned (21)':>15} {'Delta':>12}")
    print(f"  {'-' * 65}")
    for metric, full_val in full_ref.items():
        pruned_val = s.get(metric, s.get(
            {"trades": "executed_trades", "return_pct": "total_return_pct",
             "max_dd_pct": "max_realized_drawdown_pct"}.get(metric, metric), 0))
        if metric == "trades":
            pruned_val = s.get("executed_trades", 0)
        elif metric == "return_pct":
            pruned_val = s.get("total_return_pct", 0)
        elif metric == "max_dd_pct":
            pruned_val = s.get("max_realized_drawdown_pct", 0)
        else:
            pruned_val = s.get(metric, 0)
        delta = pruned_val - full_val
        fmt = ".2f" if metric != "trades" else ".0f"
        if metric == "final_equity":
            print(f"  {metric:<20} {'${:,.2f}'.format(full_val):>15} "
                  f"{'${:,.2f}'.format(pruned_val):>15} {'${:+,.2f}'.format(delta):>12}")
        elif metric == "trades":
            print(f"  {metric:<20} {full_val:>15.0f} {pruned_val:>15.0f} {delta:>+12.0f}")
        else:
            print(f"  {metric:<20} {full_val:>14{fmt}}% {pruned_val:>14{fmt}}% {delta:>+11{fmt}}%")

    # ── Correlation analysis ──────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_ts"])
    trades_df["exit_ts"] = pd.to_datetime(trades_df["exit_ts"])

    full_corr, stress_corr = correlation_analysis(trades_df, group_map)

    # ── Save results ──────────────────────────────────────────────────────────
    # CSV trades
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    # Summary JSON
    summary_out = {
        "pruned_from": 27,
        "kept": kept_count,
        "removed": REMOVED,
        "groups": {gn: {"symbols": [sym.engine_ticker for sym in syms],
                        "channel": f"{ep}/{xp}"}
                   for gn, (syms, (ep, xp)) in GROUPS.items()},
        "summary": {k: v for k, v in s.items() if not isinstance(v, (list, dict))},
        "per_asset": {t: {k: v for k, v in st.items()
                         if not isinstance(v, (list, dict)) and v != float("inf")}
                     for t, st in asset_stats.items()},
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, default=str)

    # Correlation matrices
    full_corr.to_csv(OUTPUT_DIR / "correlation_full.csv")
    if stress_corr is not None:
        stress_corr.to_csv(OUTPUT_DIR / "correlation_stress.csv")

    print(f"\n  Saved: {OUTPUT_DIR}")
    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
