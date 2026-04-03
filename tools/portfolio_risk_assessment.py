"""
Comprehensive Portfolio Risk Assessment
========================================
Analyses the Document Review x3 backtest trades for:
  1. Portfolio risk metrics (VaR, CVaR, drawdown, Sortino, Calmar, tail risk)
  2. Beta assessment (per-asset and per-bucket vs portfolio)
  3. Multicollinearity assessment (VIF, correlation clustering)
  4. Exposure assessment (notional, leverage, concentration, time-in-market)
  5. Stress testing (historical shocks, Monte Carlo, regime analysis)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TRADES_PATH = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\trades.csv")
SUMMARY_PATH = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\summary.json")

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)


def load_data():
    df = pd.read_csv(TRADES_PATH, parse_dates=["entry_ts", "exit_ts"])
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    return df, summary


# ─────────────────────────────────────────────────────────────────────
# 1. PORTFOLIO RISK METRICS
# ─────────────────────────────────────────────────────────────────────
def portfolio_risk_metrics(df: pd.DataFrame, summary: dict):
    print("=" * 80)
    print("1. PORTFOLIO RISK ASSESSMENT")
    print("=" * 80)

    initial_capital = summary["initial_capital"]

    # Build equity curve from trade P&L
    df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
    equity = [initial_capital]
    for pnl in df_sorted["net_pnl"]:
        equity.append(equity[-1] + pnl)
    equity_series = pd.Series(equity)
    dates = [df_sorted["entry_ts"].min()] + list(df_sorted["exit_ts"])

    # Trade-level return series
    trade_returns = df_sorted["net_pnl"] / df_sorted["notional"]
    trade_returns_on_equity = df_sorted["net_pnl"] / equity_series.iloc[:-1].values

    # ── Drawdown analysis ─────────────────────────────────────────────
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_dd = drawdowns.min()
    max_dd_idx = drawdowns.idxmin()

    # Find drawdown durations
    dd_start = None
    dd_periods = []
    for i in range(len(equity_series)):
        if drawdowns.iloc[i] < 0 and dd_start is None:
            dd_start = i
        elif drawdowns.iloc[i] >= 0 and dd_start is not None:
            dd_periods.append((dd_start, i, i - dd_start))
            dd_start = None
    if dd_start is not None:
        dd_periods.append((dd_start, len(equity_series) - 1, len(equity_series) - 1 - dd_start))

    longest_dd = max(dd_periods, key=lambda x: x[2]) if dd_periods else (0, 0, 0)

    # Peak-to-trough timing
    peak_idx = equity_series[:max_dd_idx + 1].idxmax()
    recovery_candidates = [i for i in range(max_dd_idx, len(equity_series))
                           if equity_series.iloc[i] >= equity_series.iloc[peak_idx]]
    recovery_idx = recovery_candidates[0] if recovery_candidates else None

    print(f"\n{'Metric':<45} {'Value':>15}")
    print("-" * 62)
    print(f"{'Initial Capital':<45} {'${:,.2f}'.format(initial_capital):>15}")
    print(f"{'Final Equity':<45} {'${:,.2f}'.format(summary['final_equity']):>15}")
    print(f"{'Total Return':<45} {'{:.2f}%'.format(summary['total_return_pct']):>15}")
    print(f"{'CAGR':<45} {'{:.2f}%'.format(summary['cagr_pct']):>15}")

    # ── VaR / CVaR at trade level ─────────────────────────────────────
    confidence_levels = [0.95, 0.99]
    print(f"\n--- Value at Risk (Trade-Level) ---")
    for cl in confidence_levels:
        var = np.percentile(trade_returns_on_equity, (1 - cl) * 100)
        cvar = trade_returns_on_equity[trade_returns_on_equity <= var].mean()
        print(f"  VaR  ({cl:.0%} confidence):{var:>18.4%} of equity per trade")
        print(f"  CVaR ({cl:.0%} confidence):{cvar:>18.4%} of equity per trade")

    # VaR in dollar terms at final equity
    final_eq = summary["final_equity"]
    for cl in confidence_levels:
        var = np.percentile(trade_returns_on_equity, (1 - cl) * 100)
        cvar = trade_returns_on_equity[trade_returns_on_equity <= var].mean()
        print(f"  VaR  ({cl:.0%}, ${final_eq:,.0f} equity): ${var * final_eq:>12,.2f}")
        print(f"  CVaR ({cl:.0%}, ${final_eq:,.0f} equity): ${cvar * final_eq:>12,.2f}")

    # ── Drawdown metrics ──────────────────────────────────────────────
    print(f"\n--- Drawdown Analysis ---")
    print(f"  Max Realized Drawdown:{max_dd:>18.2%}")
    print(f"  Peak-to-Trough trades:{max_dd_idx - peak_idx:>15d} trades")
    if recovery_idx:
        print(f"  Trough-to-Recovery:{recovery_idx - max_dd_idx:>18d} trades")
    else:
        print(f"  Trough-to-Recovery:{'NOT RECOVERED':>18s}")
    print(f"  Longest DD (trades):{longest_dd[2]:>18d} trades")

    # Top 5 drawdowns
    dd_sorted = drawdowns.sort_values()
    print(f"\n  Top 5 Drawdown Points:")
    seen_regimes = set()
    count = 0
    for idx in dd_sorted.index[:20]:
        regime_key = idx // 20
        if regime_key in seen_regimes:
            continue
        seen_regimes.add(regime_key)
        ts = dates[idx].strftime("%Y-%m-%d") if idx < len(dates) else "N/A"
        print(f"    {ts}: {drawdowns.iloc[idx]:.2%}")
        count += 1
        if count >= 5:
            break

    # ── Risk-adjusted ratios ──────────────────────────────────────────
    # Approximate using 252 trading days / ~130 trades per year average
    n_trades = len(df_sorted)
    start = df_sorted["entry_ts"].min()
    end = df_sorted["exit_ts"].max()
    years = (end - start).total_seconds() / (365.25 * 24 * 3600)
    trades_per_year = n_trades / years if years > 0 else n_trades

    mean_return = trade_returns_on_equity.mean()
    std_return = trade_returns_on_equity.std()
    downside_returns = trade_returns_on_equity[trade_returns_on_equity < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-9

    # Annualize
    ann_return = mean_return * trades_per_year
    ann_std = std_return * np.sqrt(trades_per_year)
    ann_downside_std = downside_std * np.sqrt(trades_per_year)

    sharpe = ann_return / ann_std if ann_std > 0 else 0
    sortino = ann_return / ann_downside_std if ann_downside_std > 0 else 0
    calmar = summary["cagr_pct"] / summary["max_realized_drawdown_pct"] if summary["max_realized_drawdown_pct"] > 0 else 0

    print(f"\n--- Risk-Adjusted Ratios ---")
    print(f"  Sharpe Ratio (annualized):{sharpe:>14.2f}")
    print(f"  Sortino Ratio (annualized):{sortino:>13.2f}")
    print(f"  Calmar Ratio:{calmar:>27.2f}")
    print(f"  Profit Factor:{summary['profit_factor']:>26.2f}")
    print(f"  Win Rate:{summary['win_rate_pct']:>28.2f}%")

    # ── Tail risk metrics ─────────────────────────────────────────────
    skewness = stats.skew(trade_returns_on_equity)
    kurtosis_excess = stats.kurtosis(trade_returns_on_equity)
    print(f"\n--- Tail Risk ---")
    print(f"  Return Skewness:{skewness:>24.4f}")
    print(f"  Excess Kurtosis:{kurtosis_excess:>24.4f}")
    print(f"  {'(positive skew = right-tailed = beneficial)' if skewness > 0 else '(negative skew = left-tailed = harmful)'}")
    print(f"  {'(leptokurtic = fat tails)' if kurtosis_excess > 0 else '(platykurtic = thin tails)'}")

    # Worst / best trades
    worst5 = df_sorted.nsmallest(5, "net_pnl")[["ticker", "direction", "entry_ts", "net_pnl", "notional"]]
    best5 = df_sorted.nlargest(5, "net_pnl")[["ticker", "direction", "entry_ts", "net_pnl", "notional"]]
    print(f"\n  5 Worst Trades:")
    for _, row in worst5.iterrows():
        ret_pct = row["net_pnl"] / row["notional"] * 100
        print(f"    {row['ticker']:12s} {row['direction']:5s} {row['entry_ts'].strftime('%Y-%m-%d')} "
              f"P&L: ${row['net_pnl']:>10,.2f}  ({ret_pct:>+6.1f}%)")
    print(f"\n  5 Best Trades:")
    for _, row in best5.iterrows():
        ret_pct = row["net_pnl"] / row["notional"] * 100
        print(f"    {row['ticker']:12s} {row['direction']:5s} {row['entry_ts'].strftime('%Y-%m-%d')} "
              f"P&L: ${row['net_pnl']:>10,.2f}  ({ret_pct:>+6.1f}%)")

    # ── Consecutive loss analysis ─────────────────────────────────────
    streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    win_streak = 0
    for pnl in df_sorted["net_pnl"]:
        if pnl <= 0:
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
            win_streak = 0
        else:
            win_streak += 1
            max_win_streak = max(max_win_streak, win_streak)
            streak = 0

    print(f"\n--- Streak Analysis ---")
    print(f"  Max Consecutive Losses:{max_loss_streak:>17d}")
    print(f"  Max Consecutive Wins:{max_win_streak:>19d}")
    print(f"  Avg Trade P&L:{'${:,.2f}'.format(df_sorted['net_pnl'].mean()):>25}")
    print(f"  Median Trade P&L:{'${:,.2f}'.format(df_sorted['net_pnl'].median()):>22}")
    print(f"  Std Dev Trade P&L:{'${:,.2f}'.format(df_sorted['net_pnl'].std()):>21}")

    return equity_series, trade_returns_on_equity, dates, trades_per_year


# ─────────────────────────────────────────────────────────────────────
# 2. BETA ASSESSMENT
# ─────────────────────────────────────────────────────────────────────
def beta_assessment(df: pd.DataFrame, summary: dict):
    print("\n" + "=" * 80)
    print("2. BETA ASSESSMENT")
    print("=" * 80)

    # Build daily P&L by ticker
    df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
    initial_capital = summary["initial_capital"]

    # Build equity curve for the portfolio
    equity = initial_capital
    portfolio_returns_by_trade = []
    for _, row in df_sorted.iterrows():
        ret = row["net_pnl"] / equity
        portfolio_returns_by_trade.append(ret)
        equity += row["net_pnl"]

    # Compute per-ticker contribution beta (vs total portfolio return series)
    # Group trades by ticker and compute return correlation / beta vs portfolio
    portfolio_ret = np.array(portfolio_returns_by_trade)

    # ── Per-ticker beta ───────────────────────────────────────────────
    print(f"\n--- Per-Ticker Beta (vs Portfolio Equity Return Series) ---")
    print(f"  Beta measures each asset's P&L sensitivity to portfolio moves.\n")

    ticker_stats = []
    for ticker in df_sorted["ticker"].unique():
        mask = df_sorted["ticker"] == ticker
        ticker_returns = np.where(mask, portfolio_ret, 0.0)
        n_trades = mask.sum()
        total_pnl = df_sorted.loc[mask, "net_pnl"].sum()
        avg_pnl = df_sorted.loc[mask, "net_pnl"].mean()

        if n_trades < 3:
            ticker_stats.append({"ticker": ticker, "n_trades": n_trades,
                                 "total_pnl": total_pnl, "avg_pnl": avg_pnl,
                                 "beta": np.nan, "corr": np.nan, "pnl_std": 0})
            continue

        # Compute covariance-based beta against the portfolio
        ticker_pnl = df_sorted.loc[mask, "net_pnl"].values
        ticker_indices = np.where(mask)[0]
        port_at_ticker = portfolio_ret[ticker_indices]

        if np.std(port_at_ticker) > 1e-9 and np.std(ticker_pnl) > 1e-9:
            beta = np.cov(ticker_pnl, port_at_ticker)[0, 1] / np.var(port_at_ticker) if np.var(port_at_ticker) > 0 else 0
            corr = np.corrcoef(ticker_pnl, port_at_ticker)[0, 1]
        else:
            beta = 0
            corr = 0

        ticker_stats.append({"ticker": ticker, "n_trades": n_trades,
                             "total_pnl": total_pnl, "avg_pnl": avg_pnl,
                             "beta": beta, "corr": corr,
                             "pnl_std": df_sorted.loc[mask, "net_pnl"].std()})

    ticker_df = pd.DataFrame(ticker_stats).sort_values("total_pnl", ascending=False)
    print(f"  {'Ticker':<14} {'Trades':>6} {'Total P&L':>12} {'Avg P&L':>10} {'P&L StdDev':>12} {'Beta':>8} {'Corr':>8}")
    print("  " + "-" * 72)
    for _, row in ticker_df.iterrows():
        beta_str = f"{row['beta']:>8.3f}" if not np.isnan(row["beta"]) else "   N/A  "
        corr_str = f"{row['corr']:>8.3f}" if not np.isnan(row["corr"]) else "   N/A  "
        print(f"  {row['ticker']:<14} {row['n_trades']:>6d} ${row['total_pnl']:>10,.2f} "
              f"${row['avg_pnl']:>8,.2f} ${row['pnl_std']:>10,.2f} {beta_str} {corr_str}")

    # ── Per-bucket beta ───────────────────────────────────────────────
    print(f"\n--- Per-Bucket (Asset Class) Beta ---\n")
    bucket_stats = []
    for bucket in df_sorted["asset_bucket"].unique():
        mask = df_sorted["asset_bucket"] == bucket
        n_trades = mask.sum()
        total_pnl = df_sorted.loc[mask, "net_pnl"].sum()

        bucket_pnl = df_sorted.loc[mask, "net_pnl"].values
        bucket_indices = np.where(mask)[0]
        port_at_bucket = portfolio_ret[bucket_indices]

        if n_trades >= 3 and np.var(port_at_bucket) > 1e-9:
            beta = np.cov(bucket_pnl, port_at_bucket)[0, 1] / np.var(port_at_bucket)
            corr = np.corrcoef(bucket_pnl, port_at_bucket)[0, 1] if np.std(bucket_pnl) > 1e-9 else 0
        else:
            beta = np.nan
            corr = np.nan

        bucket_stats.append({"bucket": bucket, "n_trades": n_trades,
                             "total_pnl": total_pnl, "beta": beta, "corr": corr,
                             "pnl_share": total_pnl})

    bucket_df = pd.DataFrame(bucket_stats).sort_values("total_pnl", ascending=False)
    total_pnl_all = bucket_df["total_pnl"].sum()
    print(f"  {'Bucket':<14} {'Trades':>6} {'Total P&L':>12} {'P&L Share':>10} {'Beta':>8} {'Corr':>8}")
    print("  " + "-" * 60)
    for _, row in bucket_df.iterrows():
        share = row["total_pnl"] / total_pnl_all * 100 if total_pnl_all != 0 else 0
        beta_str = f"{row['beta']:>8.2f}" if not np.isnan(row["beta"]) else "   N/A  "
        corr_str = f"{row['corr']:>8.3f}" if not np.isnan(row["corr"]) else "   N/A  "
        print(f"  {row['bucket']:<14} {row['n_trades']:>6d} ${row['total_pnl']:>10,.2f} {share:>9.1f}% {beta_str} {corr_str}")

    # ── Long / Short beta ─────────────────────────────────────────────
    print(f"\n--- Directional Beta ---\n")
    for direction in ["long", "short"]:
        mask = df_sorted["direction"] == direction
        n = mask.sum()
        total = df_sorted.loc[mask, "net_pnl"].sum()
        avg = df_sorted.loc[mask, "net_pnl"].mean()
        win_rate = (df_sorted.loc[mask, "net_pnl"] > 0).mean() * 100
        winners = df_sorted.loc[mask & (df_sorted["net_pnl"] > 0), "net_pnl"].sum()
        losers = abs(df_sorted.loc[mask & (df_sorted["net_pnl"] <= 0), "net_pnl"].sum())
        pf = winners / losers if losers > 0 else 999
        print(f"  {direction.upper():>5s}: {n:>4d} trades | P&L: ${total:>10,.2f} | "
              f"Avg: ${avg:>8,.2f} | WR: {win_rate:.1f}% | PF: {pf:.2f}")

    return ticker_df


# ─────────────────────────────────────────────────────────────────────
# 3. MULTICOLLINEARITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────
def multicollinearity_assessment(df: pd.DataFrame, summary: dict):
    print("\n" + "=" * 80)
    print("3. MULTICOLLINEARITY ASSESSMENT")
    print("=" * 80)

    # Build a weekly P&L matrix per ticker
    df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
    df_sorted["exit_week"] = df_sorted["exit_ts"].dt.to_period("W")

    # Create P&L pivot: weeks x tickers
    weekly_pnl = df_sorted.groupby(["exit_week", "ticker"])["net_pnl"].sum().unstack(fill_value=0)

    # Only keep tickers with enough activity
    min_weeks = 10
    active_tickers = weekly_pnl.columns[weekly_pnl.astype(bool).sum() >= min_weeks]
    weekly_active = weekly_pnl[active_tickers]

    print(f"\n  Weekly P&L matrix: {weekly_active.shape[0]} weeks x {len(active_tickers)} active tickers")
    print(f"  (tickers with >= {min_weeks} active weeks)\n")

    if len(active_tickers) < 3:
        print("  Insufficient active tickers for multicollinearity analysis.")
        return

    # ── Correlation matrix ────────────────────────────────────────────
    corr_matrix = weekly_active.corr()
    print("--- Pairwise Correlation Matrix (Weekly P&L) ---\n")

    # Print top correlated pairs
    pairs = []
    tickers = list(corr_matrix.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append((tickers[i], tickers[j], corr_matrix.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"  {'Pair':<30s} {'Correlation':>12s} {'Risk Level':>12s}")
    print("  " + "-" * 56)
    for t1, t2, corr in pairs[:15]:
        if abs(corr) >= 0.7:
            level = "HIGH"
        elif abs(corr) >= 0.4:
            level = "MODERATE"
        else:
            level = "LOW"
        print(f"  {t1:>12s} / {t2:<14s} {corr:>12.4f} {level:>12s}")

    # ── VIF (Variance Inflation Factor) ───────────────────────────────
    print(f"\n--- Variance Inflation Factor (VIF) ---")
    print(f"  VIF > 5 = concerning, VIF > 10 = serious multicollinearity\n")

    # Use only tickers with sufficient variance
    vif_tickers = [t for t in active_tickers if weekly_active[t].std() > 1e-9]
    if len(vif_tickers) >= 3:
        X = weekly_active[vif_tickers].values
        X_centered = X - X.mean(axis=0)

        vif_results = []
        for i, ticker in enumerate(vif_tickers):
            y = X_centered[:, i]
            X_others = np.delete(X_centered, i, axis=1)
            if X_others.shape[1] == 0:
                vif_results.append((ticker, 1.0))
                continue
            # OLS: R^2 of regressing ticker on all others
            try:
                beta = np.linalg.lstsq(X_others, y, rcond=None)[0]
                y_hat = X_others @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r_squared) if r_squared < 1 else 999
            except Exception:
                vif = np.nan
            vif_results.append((ticker, vif))

        vif_results.sort(key=lambda x: x[1], reverse=True)
        print(f"  {'Ticker':<14s} {'VIF':>8s} {'Assessment':>14s}")
        print("  " + "-" * 38)
        for ticker, vif in vif_results:
            if np.isnan(vif):
                assessment = "N/A"
            elif vif >= 10:
                assessment = "SERIOUS"
            elif vif >= 5:
                assessment = "CONCERNING"
            else:
                assessment = "OK"
            print(f"  {ticker:<14s} {vif:>8.2f} {assessment:>14s}")
    else:
        print("  Insufficient tickers with variance for VIF computation.")

    # ── Bucket-level correlation ──────────────────────────────────────
    print(f"\n--- Bucket-Level Correlation (Weekly P&L) ---\n")
    weekly_bucket = df_sorted.groupby(["exit_week", "asset_bucket"])["net_pnl"].sum().unstack(fill_value=0)
    bucket_corr = weekly_bucket.corr()
    print(bucket_corr.round(4).to_string())

    # ── Eigenvalue analysis (PCA-based concentration) ─────────────────
    print(f"\n--- Principal Component Analysis (P&L Concentration) ---")
    if len(active_tickers) >= 3:
        X_std = (weekly_active - weekly_active.mean()) / (weekly_active.std() + 1e-9)
        cov_mat = np.cov(X_std.values, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_mat)[::-1]
        explained = eigenvalues / eigenvalues.sum() * 100
        cumulative = np.cumsum(explained)
        print(f"\n  {'PC':<5s} {'Eigenvalue':>12s} {'% Explained':>12s} {'Cumulative %':>14s}")
        print("  " + "-" * 45)
        for i in range(min(8, len(eigenvalues))):
            print(f"  PC{i + 1:<3d} {eigenvalues[i]:>12.4f} {explained[i]:>11.2f}% {cumulative[i]:>12.2f}%")
        # Effective number of independent bets
        eff_bets = eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()
        print(f"\n  Effective independent bets: {eff_bets:.1f} / {len(eigenvalues)} tickers")
        print(f"  Diversification ratio: {eff_bets / len(eigenvalues):.2%}")


# ─────────────────────────────────────────────────────────────────────
# 4. EXPOSURE ASSESSMENT
# ─────────────────────────────────────────────────────────────────────
def exposure_assessment(df: pd.DataFrame, summary: dict):
    print("\n" + "=" * 80)
    print("4. EXPOSURE ASSESSMENT")
    print("=" * 80)

    df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
    initial_capital = summary["initial_capital"]

    # ── Notional exposure by bucket ───────────────────────────────────
    print(f"\n--- Notional Exposure by Asset Class ---\n")
    bucket_exposure = df_sorted.groupby("asset_bucket").agg(
        trades=("net_pnl", "count"),
        total_notional=("notional", "sum"),
        avg_notional=("notional", "mean"),
        max_notional=("notional", "max"),
        total_pnl=("net_pnl", "sum"),
    )
    total_notional = bucket_exposure["total_notional"].sum()
    bucket_exposure["notional_share"] = bucket_exposure["total_notional"] / total_notional * 100
    print(bucket_exposure.round(2).to_string())

    # ── Leverage analysis ─────────────────────────────────────────────
    print(f"\n--- Leverage / Exposure Timeline ---\n")

    # Reconstruct concurrent positions at each point
    events = []
    equity_at_entry = {}
    eq = initial_capital
    eq_by_trade = {}
    for idx, row in df_sorted.iterrows():
        eq_by_trade[idx] = eq
        eq += row["net_pnl"]

    for idx, row in df.iterrows():
        events.append(("open", row["entry_ts"], row["notional"], row["asset_bucket"], row["ticker"], idx))
        events.append(("close", row["exit_ts"], row["notional"], row["asset_bucket"], row["ticker"], idx))
    events.sort(key=lambda x: (x[1], 0 if x[0] == "close" else 1))

    max_concurrent = 0
    max_notional_open = 0
    max_leverage = 0
    current_open = {}
    leverage_snapshots = []

    # Simple equity tracking for leverage
    eq_track = initial_capital
    closed_pnl = {}
    for event_type, ts, notional, bucket, ticker, idx in events:
        if event_type == "open":
            current_open[idx] = {"notional": notional, "bucket": bucket, "ticker": ticker}
        else:
            if idx in current_open:
                del current_open[idx]
            # Update equity on close
            pnl = df.loc[idx, "net_pnl"]
            eq_track += pnl

        total_open_notional = sum(v["notional"] for v in current_open.values())
        n_concurrent = len(current_open)
        leverage = total_open_notional / eq_track if eq_track > 0 else 0

        max_concurrent = max(max_concurrent, n_concurrent)
        max_notional_open = max(max_notional_open, total_open_notional)
        max_leverage = max(max_leverage, leverage)

        if n_concurrent > 0:
            leverage_snapshots.append({
                "ts": ts, "concurrent": n_concurrent,
                "notional": total_open_notional, "leverage": leverage,
                "equity": eq_track
            })

    print(f"  {'Metric':<40s} {'Value':>15s}")
    print("  " + "-" * 57)
    print(f"  {'Max Concurrent Positions':<40s} {max_concurrent:>15d}")
    print(f"  {'Max Open Notional':<40s} {'${:,.2f}'.format(max_notional_open):>15s}")
    print(f"  {'Max Leverage (notional/equity)':<40s} {max_leverage:>14.2f}x")

    if leverage_snapshots:
        lev_arr = [s["leverage"] for s in leverage_snapshots]
        conc_arr = [s["concurrent"] for s in leverage_snapshots]
        print(f"  {'Avg Leverage When Positions Open':<40s} {np.mean(lev_arr):>14.2f}x")
        print(f"  {'Median Concurrent Positions':<40s} {np.median(conc_arr):>15.1f}")
        print(f"  {'P95 Leverage':<40s} {np.percentile(lev_arr, 95):>14.2f}x")
        print(f"  {'P99 Leverage':<40s} {np.percentile(lev_arr, 99):>14.2f}x")

    # ── Concentration risk (HHI) ──────────────────────────────────────
    print(f"\n--- Concentration Risk (Herfindahl-Hirschman Index) ---\n")
    ticker_notional = df_sorted.groupby("ticker")["notional"].sum()
    ticker_shares = ticker_notional / ticker_notional.sum()
    hhi = (ticker_shares ** 2).sum()
    eff_n = 1 / hhi if hhi > 0 else len(ticker_shares)
    print(f"  Ticker-level HHI: {hhi:.4f} (1/{eff_n:.1f} effective tickers)")
    print(f"  {'Well-diversified' if hhi < 0.10 else 'Moderate concentration' if hhi < 0.18 else 'High concentration'}")

    bucket_notional = df_sorted.groupby("asset_bucket")["notional"].sum()
    bucket_shares = bucket_notional / bucket_notional.sum()
    hhi_bucket = (bucket_shares ** 2).sum()
    eff_n_bucket = 1 / hhi_bucket if hhi_bucket > 0 else len(bucket_shares)
    print(f"\n  Bucket-level HHI: {hhi_bucket:.4f} (1/{eff_n_bucket:.1f} effective buckets)")
    print(f"  {'Well-diversified' if hhi_bucket < 0.25 else 'Moderate concentration' if hhi_bucket < 0.40 else 'High concentration'}")

    # ── Top ticker exposure shares ────────────────────────────────────
    print(f"\n  Top 10 Tickers by Notional Exposure:")
    print(f"  {'Ticker':<14s} {'Notional':>12s} {'Share':>8s} {'Trades':>8s}")
    print("  " + "-" * 44)
    top = ticker_shares.sort_values(ascending=False).head(10)
    trade_counts = df_sorted.groupby("ticker")["net_pnl"].count()
    for ticker, share in top.items():
        notional = ticker_notional[ticker]
        n = trade_counts.get(ticker, 0)
        print(f"  {ticker:<14s} ${notional:>10,.0f} {share:>7.1%} {n:>8d}")

    # ── Time exposure (holding period analysis) ───────────────────────
    print(f"\n--- Holding Period Analysis ---\n")
    df_sorted["holding_hours"] = (df_sorted["exit_ts"] - df_sorted["entry_ts"]).dt.total_seconds() / 3600
    print(f"  {'Metric':<35s} {'Value':>15s}")
    print("  " + "-" * 52)
    print(f"  {'Mean Holding Period':<35s} {df_sorted['holding_hours'].mean():>12.1f} hrs")
    print(f"  {'Median Holding Period':<35s} {df_sorted['holding_hours'].median():>12.1f} hrs")
    print(f"  {'Max Holding Period':<35s} {df_sorted['holding_hours'].max():>12.1f} hrs")
    print(f"  {'Min Holding Period':<35s} {df_sorted['holding_hours'].min():>12.1f} hrs")
    print(f"  {'P25 Holding Period':<35s} {df_sorted['holding_hours'].quantile(0.25):>12.1f} hrs")
    print(f"  {'P75 Holding Period':<35s} {df_sorted['holding_hours'].quantile(0.75):>12.1f} hrs")

    # Holding period by direction
    for d in ["long", "short"]:
        m = df_sorted["direction"] == d
        print(f"  {d.upper() + ' Avg Holding':<35s} {df_sorted.loc[m, 'holding_hours'].mean():>12.1f} hrs")

    # ── Exposure multiplier distribution ──────────────────────────────
    print(f"\n--- Exposure Multiplier Distribution ---\n")
    exp_counts = df_sorted["entry_exposure_mult"].value_counts().sort_index()
    for mult, count in exp_counts.items():
        pnl = df_sorted.loc[df_sorted["entry_exposure_mult"] == mult, "net_pnl"].sum()
        print(f"  Exposure {mult:>4.1f}x: {count:>4d} trades | P&L: ${pnl:>10,.2f}")

    # ── EMA overlay regime breakdown ──────────────────────────────────
    print(f"\n--- Per-Asset EMA Regime Breakdown ---\n")
    if "technical_ema_regime" in df_sorted.columns:
        for regime in df_sorted["technical_ema_regime"].unique():
            m = df_sorted["technical_ema_regime"] == regime
            n = m.sum()
            pnl = df_sorted.loc[m, "net_pnl"].sum()
            wr = (df_sorted.loc[m, "net_pnl"] > 0).mean() * 100
            print(f"  {str(regime):>12s}: {n:>4d} trades | P&L: ${pnl:>10,.2f} | WR: {wr:.1f}%")


# ─────────────────────────────────────────────────────────────────────
# 5. STRESS TESTING
# ─────────────────────────────────────────────────────────────────────
def stress_testing(df: pd.DataFrame, summary: dict, equity_series, trade_returns, trades_per_year):
    print("\n" + "=" * 80)
    print("5. STRESS TESTING")
    print("=" * 80)

    df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
    initial_capital = summary["initial_capital"]

    # ── Historical Rolling Windows ────────────────────────────────────
    print(f"\n--- Rolling Window Analysis ---\n")
    window_sizes = [10, 20, 50, 100]
    for w in window_sizes:
        if len(df_sorted) < w:
            continue
        rolling_pnl = df_sorted["net_pnl"].rolling(w).sum()
        worst = rolling_pnl.min()
        best = rolling_pnl.max()
        worst_idx = rolling_pnl.idxmin()
        best_idx = rolling_pnl.idxmax()
        worst_start = df_sorted.loc[max(0, worst_idx - w + 1), "entry_ts"].strftime("%Y-%m-%d") if not np.isnan(worst) else "N/A"
        print(f"  {w:>3d}-trade window: Worst ${worst:>10,.2f} (from {worst_start}) | Best ${best:>10,.2f}")

    # ── Regime-Based Stress Test (by year) ────────────────────────────
    print(f"\n--- Yearly Performance Regime ---\n")
    df_sorted["exit_year"] = df_sorted["exit_ts"].dt.year
    yearly = df_sorted.groupby("exit_year").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        max_loss=("net_pnl", "min"),
        max_win=("net_pnl", "max"),
    )
    print(f"  {'Year':>6s} {'Trades':>6s} {'Total P&L':>12s} {'Avg P&L':>10s} {'WR%':>6s} {'Max Loss':>10s} {'Max Win':>10}")
    print("  " + "-" * 64)
    for year, row in yearly.iterrows():
        print(f"  {year:>6d} {row['trades']:>6.0f} ${row['total_pnl']:>10,.2f} ${row['avg_pnl']:>8,.2f} "
              f"{row['win_rate']:>5.1f}% ${row['max_loss']:>8,.2f} ${row['max_win']:>8,.2f}")

    # ── Quarterly breakdown ───────────────────────────────────────────
    print(f"\n--- Quarterly Performance ---\n")
    df_sorted["exit_quarter"] = df_sorted["exit_ts"].dt.to_period("Q")
    quarterly = df_sorted.groupby("exit_quarter").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    )
    worst_q = quarterly["total_pnl"].idxmin()
    best_q = quarterly["total_pnl"].idxmax()
    print(f"  Worst Quarter: {worst_q} → P&L: ${quarterly.loc[worst_q, 'total_pnl']:,.2f} "
          f"({quarterly.loc[worst_q, 'trades']:.0f} trades, {quarterly.loc[worst_q, 'win_rate']:.1f}% WR)")
    print(f"  Best Quarter:  {best_q} → P&L: ${quarterly.loc[best_q, 'total_pnl']:,.2f} "
          f"({quarterly.loc[best_q, 'trades']:.0f} trades, {quarterly.loc[best_q, 'win_rate']:.1f}% WR)")
    print()
    print(f"  {'Quarter':>8s} {'Trades':>6s} {'P&L':>12s} {'WR%':>6s}")
    print("  " + "-" * 36)
    for q, row in quarterly.iterrows():
        print(f"  {str(q):>8s} {row['trades']:>6.0f} ${row['total_pnl']:>10,.2f} {row['win_rate']:>5.1f}%")

    # ── Monte Carlo Stress Test ───────────────────────────────────────
    print(f"\n--- Monte Carlo Simulation (10,000 paths) ---\n")
    np.random.seed(42)
    n_simulations = 10_000
    n_trades_forward = int(trades_per_year * 1)  # 1-year forward simulation
    trade_pnl = df_sorted["net_pnl"].values

    # Bootstrap from actual trade P&L distribution
    final_equities = []
    max_drawdowns_mc = []
    for _ in range(n_simulations):
        sampled = np.random.choice(trade_pnl, size=n_trades_forward, replace=True)
        eq = summary["final_equity"]
        peak = eq
        max_dd = 0
        for pnl in sampled:
            eq += pnl
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        final_equities.append(eq)
        max_drawdowns_mc.append(max_dd)

    final_equities = np.array(final_equities)
    max_drawdowns_mc = np.array(max_drawdowns_mc)

    print(f"  Starting equity: ${summary['final_equity']:,.2f}")
    print(f"  Forward trades: {n_trades_forward} (~1 year)")
    print(f"\n  {'Percentile':<15s} {'Final Equity':>14s} {'Return':>10s} {'Max DD':>10s}")
    print("  " + "-" * 51)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        eq_p = np.percentile(final_equities, p)
        ret_p = (eq_p / summary["final_equity"] - 1) * 100
        dd_p = np.percentile(max_drawdowns_mc, 100 - p) * 100  # inverse for DD
        print(f"  {'P' + str(p):<15s} ${eq_p:>12,.0f} {ret_p:>+9.1f}% {dd_p:>9.1f}%")

    ruin_pct = (final_equities < summary["final_equity"] * 0.5).mean() * 100
    breakeven_pct = (final_equities < summary["final_equity"]).mean() * 100
    print(f"\n  Probability of losing >50%: {ruin_pct:.2f}%")
    print(f"  Probability of negative return: {breakeven_pct:.2f}%")
    print(f"  Median expected equity: ${np.median(final_equities):,.0f}")

    # ── Correlation-Aware Stress (Block Bootstrap) ────────────────────
    print(f"\n--- Block Bootstrap Stress Test (preserves trade clustering) ---\n")
    block_size = 20  # trade blocks to preserve temporal structure
    n_blocks = n_trades_forward // block_size
    block_finals = []
    block_max_dds = []

    for _ in range(n_simulations):
        eq = summary["final_equity"]
        peak = eq
        max_dd = 0
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, len(trade_pnl) - block_size))
            block = trade_pnl[start:start + block_size]
            for pnl in block:
                eq += pnl
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        block_finals.append(eq)
        block_max_dds.append(max_dd)

    block_finals = np.array(block_finals)
    block_max_dds = np.array(block_max_dds)

    print(f"  Block size: {block_size} trades (preserves sequential correlation)")
    print(f"\n  {'Percentile':<15s} {'Final Equity':>14s} {'Return':>10s} {'Max DD':>10s}")
    print("  " + "-" * 51)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        eq_p = np.percentile(block_finals, p)
        ret_p = (eq_p / summary["final_equity"] - 1) * 100
        dd_p = np.percentile(block_max_dds, 100 - p) * 100
        print(f"  {'P' + str(p):<15s} ${eq_p:>12,.0f} {ret_p:>+9.1f}% {dd_p:>9.1f}%")

    print(f"\n  Block P(loss >50%): {(block_finals < summary['final_equity'] * 0.5).mean() * 100:.2f}%")
    print(f"  Block P(negative return): {(block_finals < summary['final_equity']).mean() * 100:.2f}%")

    # ── Scenario: What if worst month repeated ────────────────────────
    print(f"\n--- Scenario Analysis ---\n")
    df_sorted["exit_month"] = df_sorted["exit_ts"].dt.to_period("M")
    monthly_pnl = df_sorted.groupby("exit_month")["net_pnl"].sum()
    worst_month = monthly_pnl.min()
    worst_month_period = monthly_pnl.idxmin()
    best_month = monthly_pnl.max()
    best_month_period = monthly_pnl.idxmax()

    curr_eq = summary["final_equity"]
    print(f"  Worst Month: {worst_month_period} → ${worst_month:,.2f}")
    print(f"  Best Month:  {best_month_period} → ${best_month:,.2f}")
    print(f"\n  Scenario: 3x Worst Month Consecutively")
    shock_eq = curr_eq + 3 * worst_month
    shock_dd = (curr_eq - shock_eq) / curr_eq * 100 if shock_eq < curr_eq else 0
    print(f"    Equity after: ${shock_eq:,.2f} (drawdown: {shock_dd:.1f}%)")

    print(f"\n  Scenario: All positions hit stops simultaneously")
    avg_notional = df_sorted["notional"].mean()
    stop_pct = 0.10  # fixed 10% stop
    max_conc = df_sorted.groupby(
        df_sorted["entry_ts"].dt.date
    ).size().max()
    stop_loss_scenario = max_conc * avg_notional * stop_pct
    stop_dd = stop_loss_scenario / curr_eq * 100
    print(f"    Peak concurrent: ~{max_conc} positions")
    print(f"    Avg notional: ${avg_notional:,.2f}")
    print(f"    Total stop loss: ${stop_loss_scenario:,.2f} ({stop_dd:.1f}% of equity)")

    # ── Bucket regime stress ──────────────────────────────────────────
    print(f"\n--- Bucket-Level Stress: Worst Quarter per Bucket ---\n")
    monthly_bucket = df_sorted.groupby(["exit_quarter", "asset_bucket"])["net_pnl"].sum().unstack(fill_value=0)
    print(f"  {'Bucket':<14s} {'Worst Quarter':>14s} {'P&L':>12s}")
    print("  " + "-" * 42)
    for bucket in monthly_bucket.columns:
        worst_val = monthly_bucket[bucket].min()
        worst_per = monthly_bucket[bucket].idxmin()
        print(f"  {bucket:<14s} {str(worst_per):>14s} ${worst_val:>10,.2f}")

    # ── Extended hours proxy regime stress ─────────────────────────────
    print(f"\n--- VIX/FG Proxy Regime Performance ---\n")
    if "ext_hours_proxy_regime" in df_sorted.columns:
        regime_stats = df_sorted.groupby("ext_hours_proxy_regime").agg(
            trades=("net_pnl", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        )
        print(f"  {'Regime':<16s} {'Trades':>6s} {'Total P&L':>12s} {'Avg P&L':>10s} {'WR%':>6s}")
        print("  " + "-" * 52)
        for regime, row in regime_stats.iterrows():
            print(f"  {str(regime):<16s} {row['trades']:>6.0f} ${row['total_pnl']:>10,.2f} ${row['avg_pnl']:>8,.2f} {row['win_rate']:>5.1f}%")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 80)
    print("#  COMPREHENSIVE PORTFOLIO RISK ASSESSMENT")
    print("#  Strategy: Session Turtle Trend Document Review x3")
    print(f"#  Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("#" * 80)

    df, summary = load_data()

    equity_series, trade_returns, dates, trades_per_year = portfolio_risk_metrics(df, summary)
    ticker_df = beta_assessment(df, summary)
    multicollinearity_assessment(df, summary)
    exposure_assessment(df, summary)
    stress_testing(df, summary, equity_series, trade_returns, trades_per_year)

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EXECUTIVE RISK SUMMARY")
    print("=" * 80)
    print(f"""
  Strategy: Session Turtle Trend x3 Document Review
  Period: {summary['start_date'][:10]} to {summary['end_date'][:10]}
  Trades: {summary['executed_trades']} ({summary['long_trades']} long / {summary['short_trades']} short)

  RETURN: {summary['total_return_pct']:.2f}% total | {summary['cagr_pct']:.2f}% CAGR
  RISK:   {summary['max_realized_drawdown_pct']:.2f}% max drawdown | PF {summary['profit_factor']:.2f}
  """)

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
