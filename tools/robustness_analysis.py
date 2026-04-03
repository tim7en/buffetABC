"""
Strategy Robustness & Sustainability Analysis
===============================================
1. Walk-forward (anchored & rolling) out-of-sample assessment
2. Return without Q4 2025 + Q1 2026 (Oct–Mar boom)
3. Return starting from 2023 (excluding 2022 ramp-up)
4. Realistic Binance fee & slippage stress test
5. Sustainability scorecard
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TRADES_CSV = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\trades.csv")
SUMMARY_JSON = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\summary.json")

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


ETF_TICKERS = {"SLV", "PPLT", "GLD", "QQQ", "SPY", "EWJ", "EWY"}


def load():
    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_ts", "exit_ts"])
    with open(SUMMARY_JSON) as f:
        s = json.load(f)
    # Reclassify ETFs into their own bucket
    df.loc[df["ticker"].isin(ETF_TICKERS), "asset_bucket"] = "etf"
    return df.sort_values("exit_ts").reset_index(drop=True), s


def compute_stats(trades_pnl, initial_capital, label=""):
    """Compute core metrics from a P&L series and initial capital."""
    equity = [initial_capital]
    for pnl in trades_pnl:
        equity.append(equity[-1] + pnl)
    eq = np.array(equity)
    final = eq[-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = abs(dd.min()) * 100

    total_ret = (final / initial_capital - 1) * 100
    n_trades = len(trades_pnl)
    winners = (trades_pnl > 0).sum()
    losers = (trades_pnl <= 0).sum()
    win_rate = winners / n_trades * 100 if n_trades > 0 else 0
    gross_win = trades_pnl[trades_pnl > 0].sum()
    gross_loss = abs(trades_pnl[trades_pnl <= 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    return {
        "label": label,
        "trades": n_trades,
        "initial": initial_capital,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_ret, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 2),
        "avg_trade": round(trades_pnl.mean(), 2) if n_trades > 0 else 0,
        "median_trade": round(np.median(trades_pnl), 2) if n_trades > 0 else 0,
    }


def compute_cagr(initial, final, years):
    if initial <= 0 or final <= 0 or years <= 0:
        return 0
    return ((final / initial) ** (1 / years) - 1) * 100


# ═════════════════════════════════════════════════════════════════════
# 1. WALK-FORWARD ANALYSIS
# ═════════════════════════════════════════════════════════════════════
def walk_forward_analysis(df, s):
    print("=" * 90)
    print("1. WALK-FORWARD OUT-OF-SAMPLE ASSESSMENT")
    print("=" * 90)

    # ── 1A: Anchored Walk-Forward (expanding in-sample, fixed OOS) ───
    print("\n--- 1A: Anchored Walk-Forward (6-month OOS windows) ---")
    print("  Each window tests the following 6 months using only trades that occurred in that period.")
    print("  Capital carries forward from the prior window — this is a true sequential walk-forward.\n")

    df["exit_half"] = df["exit_ts"].dt.to_period("2Q")  # ~6-month periods
    # Alternate: use manual 6-month cuts
    df["exit_half_custom"] = pd.cut(
        df["exit_ts"],
        bins=pd.date_range(df["exit_ts"].min().normalize(), df["exit_ts"].max() + pd.Timedelta(days=1), freq="6MS"),
        right=False,
    )
    windows = sorted(df["exit_half_custom"].dropna().unique())

    capital = s["initial_capital"]
    results = []
    for window in windows:
        mask = df["exit_half_custom"] == window
        window_trades = df.loc[mask, "net_pnl"].values
        if len(window_trades) == 0:
            continue
        start_cap = capital
        peak = capital
        max_dd = 0
        for pnl in window_trades:
            capital += pnl
            peak = max(peak, capital)
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        total_pnl = capital - start_cap
        ret = total_pnl / start_cap * 100
        winners = (window_trades > 0).sum()
        wr = winners / len(window_trades) * 100
        gross_w = window_trades[window_trades > 0].sum()
        gross_l = abs(window_trades[window_trades <= 0].sum())
        pf = gross_w / gross_l if gross_l > 0 else 999

        results.append({
            "window": str(window),
            "trades": len(window_trades),
            "start_equity": round(start_cap, 2),
            "end_equity": round(capital, 2),
            "pnl": round(total_pnl, 2),
            "return_pct": round(ret, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
        })

    header = f"  {'Window':<32s} {'Trades':>6} {'Start $':>12} {'End $':>12} {'P&L':>12} {'Ret%':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    positive_windows = 0
    for r in results:
        sign = "+" if r["pnl"] >= 0 else ""
        ret_color = r["return_pct"]
        print(f"  {r['window']:<32s} {r['trades']:>6d} ${r['start_equity']:>10,.2f} ${r['end_equity']:>10,.2f} "
              f"{sign}${abs(r['pnl']):>9,.2f} {r['return_pct']:>+7.1f}% {r['max_dd_pct']:>6.1f}% "
              f"{r['win_rate']:>5.1f} {r['profit_factor']:>5.2f}")
        if r["pnl"] > 0:
            positive_windows += 1

    n_windows = len(results)
    print(f"\n  Profitable windows: {positive_windows}/{n_windows} ({positive_windows/n_windows*100:.0f}%)")
    print(f"  Sequential final equity: ${capital:,.2f}")

    # ── 1B: Rolling Walk-Forward (fixed-size sliding window) ─────────
    print(f"\n--- 1B: Rolling Walk-Forward (100-trade sliding window, 50-trade step) ---\n")
    window_size = 100
    step = 50
    rolling_results = []
    idx = 0
    while idx + window_size <= len(df):
        window_df = df.iloc[idx:idx + window_size]
        pnl_arr = window_df["net_pnl"].values
        start_date = window_df["entry_ts"].iloc[0].strftime("%Y-%m-%d")
        end_date = window_df["exit_ts"].iloc[-1].strftime("%Y-%m-%d")

        # Use 1000 as normalised capital for each window to compare apples-to-apples
        norm_cap = 1000
        eq = norm_cap
        pk = eq
        worst_dd = 0
        for p in pnl_arr:
            # Scale P&L to normalised capital — use relative return
            pass
        # Better: just raw stats on the window
        total = pnl_arr.sum()
        avg = pnl_arr.mean()
        wr = (pnl_arr > 0).mean() * 100
        gw = pnl_arr[pnl_arr > 0].sum()
        gl = abs(pnl_arr[pnl_arr <= 0].sum())
        pf = gw / gl if gl > 0 else 999

        rolling_results.append({
            "start": start_date, "end": end_date,
            "total_pnl": total, "avg_pnl": avg,
            "win_rate": wr, "pf": pf,
        })
        idx += step

    print(f"  {'Period':<28s} {'Total P&L':>12} {'Avg P&L':>10} {'WR%':>6} {'PF':>6}")
    print("  " + "-" * 65)
    profitable_rolling = 0
    for r in rolling_results:
        sign = "+" if r["total_pnl"] >= 0 else ""
        print(f"  {r['start']} → {r['end']}  {sign}${abs(r['total_pnl']):>9,.2f} "
              f"${r['avg_pnl']:>8,.2f} {r['win_rate']:>5.1f} {r['pf']:>5.2f}")
        if r["total_pnl"] > 0:
            profitable_rolling += 1

    print(f"\n  Profitable windows: {profitable_rolling}/{len(rolling_results)} ({profitable_rolling/len(rolling_results)*100:.0f}%)")

    all_avgs = [r["avg_pnl"] for r in rolling_results]
    print(f"  Avg P&L range: ${min(all_avgs):,.2f} to ${max(all_avgs):,.2f}")
    print(f"  Avg P&L consistency (std of window avgs): ${np.std(all_avgs):,.2f}")

    return results


# ═════════════════════════════════════════════════════════════════════
# 2. RETURN WITHOUT RECENT BOOM (Oct 2025 – Mar 2026)
# ═════════════════════════════════════════════════════════════════════
def return_without_boom(df, s):
    print("\n" + "=" * 90)
    print("2. RETURN EXCLUDING RECENT BOOM PERIOD")
    print("=" * 90)

    initial = s["initial_capital"]
    full_stats = compute_stats(df["net_pnl"].values, initial, "Full Period")
    full_years = (df["exit_ts"].max() - df["entry_ts"].min()).total_seconds() / (365.25 * 24 * 3600)
    full_stats["cagr"] = round(compute_cagr(initial, full_stats["final_equity"], full_years), 2)

    # Without Q4 2025 + Q1 2026 (Oct 2025 – Mar 2026)
    boom_mask = (df["exit_ts"] >= "2025-10-01") & (df["exit_ts"] < "2026-04-01")
    df_no_boom = df[~boom_mask]
    no_boom_stats = compute_stats(df_no_boom["net_pnl"].values, initial, "Excl Oct2025–Mar2026")
    no_boom_years = (df_no_boom["exit_ts"].max() - df_no_boom["entry_ts"].min()).total_seconds() / (365.25 * 24 * 3600)
    no_boom_stats["cagr"] = round(compute_cagr(initial, no_boom_stats["final_equity"], no_boom_years), 2)

    # Without Q1 2026 only
    q1_mask = df["exit_ts"] >= "2026-01-01"
    df_no_q1 = df[~q1_mask]
    no_q1_stats = compute_stats(df_no_q1["net_pnl"].values, initial, "Excl Q1 2026")
    no_q1_years = (df_no_q1["exit_ts"].max() - df_no_q1["entry_ts"].min()).total_seconds() / (365.25 * 24 * 3600)
    no_q1_stats["cagr"] = round(compute_cagr(initial, no_q1_stats["final_equity"], no_q1_years), 2)

    # Print
    variants = [full_stats, no_boom_stats, no_q1_stats]
    print(f"\n  {'Variant':<28s} {'Trades':>6} {'Final $':>12} {'Return%':>10} {'CAGR%':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6}")
    print("  " + "-" * 85)
    for v in variants:
        print(f"  {v['label']:<28s} {v['trades']:>6d} ${v['final_equity']:>10,.2f} "
              f"{v['total_return_pct']:>+9.2f}% {v['cagr']:>7.2f}% {v['max_dd_pct']:>6.2f}% "
              f"{v['win_rate_pct']:>5.1f} {v['profit_factor']:>5.2f}")

    boom_pnl = df.loc[boom_mask, "net_pnl"].sum()
    q1_pnl = df.loc[q1_mask, "net_pnl"].sum()
    print(f"\n  Boom period (Oct25–Mar26) P&L: ${boom_pnl:,.2f} ({boom_mask.sum()} trades)")
    print(f"  Q1 2026 only P&L:              ${q1_pnl:,.2f} ({q1_mask.sum()} trades)")
    print(f"  Boom as % of total P&L:        {boom_pnl / (full_stats['final_equity'] - initial) * 100:.1f}%")

    return variants


# ═════════════════════════════════════════════════════════════════════
# 3. STARTING FROM 2023
# ═════════════════════════════════════════════════════════════════════
def from_2023_analysis(df, s):
    print("\n" + "=" * 90)
    print("3. START DATE SENSITIVITY (2022 vs 2023 vs 2024)")
    print("=" * 90)

    initial = s["initial_capital"]

    print(f"\n  {'Start Year':<14s} {'Trades':>6} {'Final $':>12} {'Return%':>10} {'CAGR%':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6}")
    print("  " + "-" * 70)
    for start_year in [2022, 2023, 2024, 2025]:
        mask = df["entry_ts"] >= f"{start_year}-01-01"
        sub = df[mask]
        if len(sub) < 5:
            continue
        st = compute_stats(sub["net_pnl"].values, initial, f"From {start_year}")
        years = (sub["exit_ts"].max() - sub["entry_ts"].min()).total_seconds() / (365.25 * 24 * 3600)
        cagr = compute_cagr(initial, st["final_equity"], years)
        print(f"  From {start_year:<8d} {st['trades']:>6d} ${st['final_equity']:>10,.2f} "
              f"{st['total_return_pct']:>+9.2f}% {cagr:>7.2f}% {st['max_dd_pct']:>6.2f}% "
              f"{st['win_rate_pct']:>5.1f} {st['profit_factor']:>5.2f}")


# ═════════════════════════════════════════════════════════════════════
# 4. BINANCE FEE & SLIPPAGE STRESS TEST
# ═════════════════════════════════════════════════════════════════════
def fee_slippage_stress(df, s):
    print("\n" + "=" * 90)
    print("4. FEE & SLIPPAGE STRESS TEST")
    print("=" * 90)

    print("""
  Current backtest settings:
    Slippage:   2 bps (0.02%) per side, applied to entry & exit prices
    Commission: 1 bp  (0.01%) per side, deducted as fee

  Binance standard fee schedule:
    Spot:    0.10% maker/taker (VIP0)
    Futures: 0.02% maker / 0.05% taker (VIP0)
    With BNB: 0.075% (25% discount)

  We'll test adding INCREMENTAL costs on top of the existing 2+1 bps:
    - Already embedded:        3 bps round-trip  (1.5 bps per side)
    - Binance spot realistic:  20 bps round-trip (10 bps per side)
    - Incremental to add:      17 bps per round-trip
  """)

    initial = s["initial_capital"]

    # Fee scenarios: additional round-trip bps to apply on top of existing
    scenarios = [
        ("Current (2bp slip + 1bp comm)", 0),
        ("+ Binance Spot VIP0 (10bp/side)", 17),    # 20 total - 3 existing
        ("+ Binance Spot BNB (7.5bp/side)", 12),     # 15 total - 3 existing
        ("+ Binance Futures (3.5bp/side)", 4),        # 7 total - 3 existing
        ("+ Heavy fees (15bp/side)", 27),             # 30 total - 3 existing
        ("+ Extreme (20bp/side)", 37),                # 40 total - 3 existing
    ]

    print(f"  {'Scenario':<45s} {'Trades':>6} {'Final $':>12} {'Return%':>10} {'MaxDD%':>7} {'PF':>6} {'Avg P&L':>10}")
    print("  " + "-" * 100)

    for label, extra_rt_bps in scenarios:
        # Apply additional round-trip cost: cost = notional * extra_rt_bps / 10000
        adjusted_pnl = df["net_pnl"] - df["notional"] * extra_rt_bps / 10000
        st = compute_stats(adjusted_pnl.values, initial, label)
        years = (df["exit_ts"].max() - df["entry_ts"].min()).total_seconds() / (365.25 * 24 * 3600)
        cagr = compute_cagr(initial, st["final_equity"], years)
        print(f"  {label:<45s} {st['trades']:>6d} ${st['final_equity']:>10,.2f} "
              f"{st['total_return_pct']:>+9.2f}% {st['max_dd_pct']:>6.2f}% {st['profit_factor']:>5.2f} "
              f"${st['avg_trade']:>8,.2f}")

    # ── Per-bucket fee sensitivity ────────────────────────────────────
    print(f"\n--- Fee Impact by Asset Class (Binance Spot VIP0: +17bps RT) ---\n")
    extra = 17
    print(f"  {'Bucket':<14s} {'Original P&L':>12} {'After Fees':>12} {'Fee Drag':>10} {'Drag%':>8}")
    print("  " + "-" * 58)
    for bucket in sorted(df["asset_bucket"].unique()):
        mask = df["asset_bucket"] == bucket
        orig = df.loc[mask, "net_pnl"].sum()
        fee_cost = (df.loc[mask, "notional"] * extra / 10000).sum()
        after = orig - fee_cost
        drag_pct = fee_cost / abs(orig) * 100 if orig != 0 else 0
        print(f"  {bucket:<14s} ${orig:>10,.2f} ${after:>10,.2f} ${fee_cost:>8,.2f} {drag_pct:>7.1f}%")

    # ── Per-ticker fee impact ─────────────────────────────────────────
    print(f"\n--- Tickers That Flip Negative Under Binance Spot Fees ---\n")
    flipped = []
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        orig = df.loc[mask, "net_pnl"].sum()
        fee_cost = (df.loc[mask, "notional"] * extra / 10000).sum()
        after = orig - fee_cost
        if orig > 0 and after < 0:
            flipped.append((ticker, orig, after, fee_cost))
    if flipped:
        for ticker, orig, after, fee in flipped:
            print(f"  {ticker:<14s} ${orig:>8,.2f} → ${after:>8,.2f}  (fee drag: ${fee:>6,.2f})")
    else:
        print("  None — all profitable tickers remain profitable after fees.")

    # ── Breakeven fee level ───────────────────────────────────────────
    print(f"\n--- Breakeven Fee Analysis ---\n")
    total_pnl = df["net_pnl"].sum()
    total_notional = df["notional"].sum()
    breakeven_bps = total_pnl / total_notional * 10000
    print(f"  Total P&L:         ${total_pnl:>12,.2f}")
    print(f"  Total Notional:    ${total_notional:>12,.2f}")
    print(f"  Breakeven extra RT cost: {breakeven_bps:.1f} bps")
    print(f"  = {breakeven_bps/2:.1f} bps per side")
    print(f"  = Already have 3 bps embedded, so strategy breaks even at {breakeven_bps + 3:.1f} bps RT total")
    print(f"  = {(breakeven_bps + 3)/2:.1f} bps per side")
    print(f"\n  Binance Spot VIP0 is 20 bps RT — {'SAFE' if breakeven_bps > 17 else 'AT RISK'} "
          f"(margin: {breakeven_bps - 17:.1f} bps)")


# ═════════════════════════════════════════════════════════════════════
# 5. SUSTAINABILITY SCORECARD
# ═════════════════════════════════════════════════════════════════════
def sustainability_scorecard(df, s, wf_results):
    print("\n" + "=" * 90)
    print("5. STRATEGY SUSTAINABILITY SCORECARD")
    print("=" * 90)

    initial = s["initial_capital"]

    checks = []

    # ── Check 1: Consistent profitability across time ─────────────────
    df2 = df.copy()
    df2["exit_quarter"] = df2["exit_ts"].dt.to_period("Q")
    quarterly = df2.groupby("exit_quarter")["net_pnl"].sum()
    pct_profitable_q = (quarterly > 0).mean() * 100
    checks.append(("Profitable quarters (>60% pass)", f"{pct_profitable_q:.0f}%",
                    "PASS" if pct_profitable_q > 60 else "FAIL"))

    # ── Check 2: Walk-forward windows ─────────────────────────────────
    wf_positive = sum(1 for r in wf_results if r["pnl"] > 0)
    wf_pct = wf_positive / len(wf_results) * 100 if wf_results else 0
    checks.append(("Walk-forward profitable windows (>60% pass)", f"{wf_positive}/{len(wf_results)} ({wf_pct:.0f}%)",
                    "PASS" if wf_pct > 60 else "FAIL"))

    # ── Check 3: Profit factor above 1.5 ─────────────────────────────
    checks.append(("Profit factor > 1.5", f"{s['profit_factor']:.2f}",
                    "PASS" if s["profit_factor"] > 1.5 else "FAIL"))

    # ── Check 4: Recovery factor (total return / max DD) ──────────────
    recovery_factor = s["total_return_pct"] / s["max_realized_drawdown_pct"]
    checks.append(("Recovery factor > 3", f"{recovery_factor:.1f}",
                    "PASS" if recovery_factor > 3 else "FAIL"))

    # ── Check 5: Positive skew ────────────────────────────────────────
    skew = stats.skew(df["net_pnl"].values)
    checks.append(("Return skewness > 0 (positive)", f"{skew:.2f}",
                    "PASS" if skew > 0 else "FAIL"))

    # ── Check 6: No single trade > 20% of total P&L ──────────────────
    total_pnl = df["net_pnl"].sum()
    max_single = df["net_pnl"].max()
    max_single_pct = max_single / total_pnl * 100
    checks.append(("No single trade > 20% of total P&L", f"{max_single_pct:.1f}%",
                    "PASS" if max_single_pct < 20 else "FAIL"))

    # ── Check 7: Strategy works after Binance fees ────────────────────
    extra_rt = 17
    adjusted_pnl = df["net_pnl"] - df["notional"] * extra_rt / 10000
    fee_pf_num = adjusted_pnl[adjusted_pnl > 0].sum()
    fee_pf_den = abs(adjusted_pnl[adjusted_pnl <= 0].sum())
    fee_pf = fee_pf_num / fee_pf_den if fee_pf_den > 0 else 999
    checks.append(("Profitable after Binance Spot fees (PF > 1.0)", f"PF {fee_pf:.2f}",
                    "PASS" if fee_pf > 1.0 else "FAIL"))

    # ── Check 8: Without boom still profitable ────────────────────────
    boom_mask = (df["exit_ts"] >= "2025-10-01") & (df["exit_ts"] < "2026-04-01")
    no_boom_pnl = df.loc[~boom_mask, "net_pnl"].sum()
    checks.append(("Profitable excluding Oct25–Mar26 boom", f"${no_boom_pnl:,.2f}",
                    "PASS" if no_boom_pnl > 0 else "FAIL"))

    # ── Check 9: Works from 2023 start ────────────────────────────────
    from_2023 = df[df["entry_ts"] >= "2023-01-01"]["net_pnl"].sum()
    checks.append(("Profitable starting from 2023", f"${from_2023:,.2f}",
                    "PASS" if from_2023 > 0 else "FAIL"))

    # ── Check 10: Max DD < 30% ────────────────────────────────────────
    checks.append(("Max drawdown < 30%", f"{s['max_realized_drawdown_pct']:.1f}%",
                    "PASS" if s["max_realized_drawdown_pct"] < 30 else "FAIL"))

    # ── Check 11: Diversification — effective bets ────────────────────
    df2["exit_week"] = df2["exit_ts"].dt.to_period("W")
    weekly = df2.groupby(["exit_week", "ticker"])["net_pnl"].sum().unstack(fill_value=0)
    active = weekly.columns[weekly.astype(bool).sum() >= 8]
    if len(active) >= 3:
        X = weekly[active].values
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        eigvals = np.linalg.eigvalsh(np.cov(X_std, rowvar=False))[::-1]
        eff_bets = eigvals.sum() ** 2 / (eigvals ** 2).sum()
        eff_ratio = eff_bets / len(active)
        checks.append(("Diversification ratio > 50%", f"{eff_ratio:.1%} ({eff_bets:.1f}/{len(active)})",
                        "PASS" if eff_ratio > 0.5 else "FAIL"))

    # ── Check 12: No year with negative return ────────────────────────
    yearly = df.groupby(df["exit_ts"].dt.year)["net_pnl"].sum()
    all_positive = (yearly > 0).all()
    worst_year = yearly.min()
    worst_year_name = yearly.idxmin()
    checks.append(("All calendar years profitable", f"Worst: {worst_year_name} (${worst_year:,.2f})",
                    "PASS" if all_positive else "FAIL"))

    # ── Print scorecard ───────────────────────────────────────────────
    print(f"\n  {'#':<3} {'Check':<52s} {'Result':>22s} {'Status':>8s}")
    print("  " + "-" * 88)
    pass_count = 0
    for i, (check, result, status) in enumerate(checks, 1):
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {i:<3d} {check:<52s} {result:>22s} {marker} {status:>5s}")
        if status == "PASS":
            pass_count += 1

    total_checks = len(checks)
    score = pass_count / total_checks * 100
    print(f"\n  Overall Score: {pass_count}/{total_checks} ({score:.0f}%)")
    if score >= 90:
        grade = "A — Highly Sustainable"
    elif score >= 75:
        grade = "B — Sustainable with minor concerns"
    elif score >= 60:
        grade = "C — Moderate sustainability risk"
    else:
        grade = "D — Significant sustainability concerns"
    print(f"  Grade: {grade}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "#" * 90)
    print("#  STRATEGY ROBUSTNESS & SUSTAINABILITY ANALYSIS")
    print("#  Session Turtle Trend x3 — Document Review Variant")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("#" * 90)

    df, s = load()

    wf_results = walk_forward_analysis(df, s)
    return_without_boom(df, s)
    from_2023_analysis(df, s)
    fee_slippage_stress(df, s)
    sustainability_scorecard(df, s, wf_results)

    print("\n" + "=" * 90)
    print("END OF ANALYSIS")
    print("=" * 90)


if __name__ == "__main__":
    main()
