"""
Session Turtle Trend x3 — Investor Research Report Generator
=============================================================
Generates a self-contained HTML report with embedded plots suitable
for distribution to investors.  Open the output in a browser and
print to PDF for a polished document.

Usage:
    python tools/investor_report.py
"""

import base64
import io
import json
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────
TRADES_CSV = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\trades.csv")
SUMMARY_JSON = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\summary.json")
OUTPUT_HTML = Path(r"d:\buffetABC\reports\session_turtle_x3_document_review_20260403\investor_report.html")

# ── style ────────────────────────────────────────────────────────────
BRAND = "#1a1a2e"
ACCENT = "#16213e"
GREEN = "#27ae60"
RED = "#e74c3c"
BLUE = "#2980b9"
GOLD = "#f39c12"
PURPLE = "#8e44ad"
GREY = "#7f8c8d"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def fig_to_base64(fig, dpi=150, tight=True):
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    else:
        fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_ts", "exit_ts"])
    with open(SUMMARY_JSON) as f:
        s = json.load(f)
    df = df.sort_values("exit_ts").reset_index(drop=True)
    df["holding_hours"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 3600
    df["return_pct"] = df["net_pnl"] / df["notional"] * 100
    df["exit_month"] = df["exit_ts"].dt.to_period("M")
    df["exit_quarter"] = df["exit_ts"].dt.to_period("Q")
    df["exit_year"] = df["exit_ts"].dt.year
    return df, s


# ─────────────────────────────────────────────────────────────────────
# PLOT GENERATORS
# ─────────────────────────────────────────────────────────────────────

def plot_equity_curve(df, s):
    """Equity curve with drawdown shading."""
    equity = [s["initial_capital"]]
    for pnl in df["net_pnl"]:
        equity.append(equity[-1] + pnl)
    eq = np.array(equity)
    dates = [df["entry_ts"].min()] + list(df["exit_ts"])
    dates = pd.to_datetime(dates)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.08})
    ax1.fill_between(dates, equity, alpha=0.15, color=BLUE)
    ax1.plot(dates, equity, color=BLUE, linewidth=1.8, label="Portfolio Equity")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Portfolio Equity Curve", fontweight="bold", fontsize=14)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_yscale("log")

    ax2.fill_between(dates, dd, 0, color=RED, alpha=0.35)
    ax2.plot(dates, dd, color=RED, linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_ylim(min(dd) * 1.15, 2)
    ax2.axhline(0, color="black", linewidth=0.5)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=30)
    return fig_to_base64(fig), eq, dates


def plot_monthly_returns_heatmap(df):
    """Calendar heatmap of monthly returns."""
    monthly = df.groupby("exit_month")["net_pnl"].sum()
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["period", "pnl"]
    monthly_df["year"] = monthly_df["period"].apply(lambda p: p.year)
    monthly_df["month"] = monthly_df["period"].apply(lambda p: p.month)
    pivot = monthly_df.pivot(index="year", columns="month", values="pnl").fillna(0)
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
    # Remap column names properly
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly_df2 = monthly_df.copy()
    monthly_df2["month_name"] = monthly_df2["month"].map(month_names)
    pivot2 = monthly_df2.pivot(index="year", columns="month", values="pnl").fillna(0)
    pivot2.columns = [month_names[c] for c in pivot2.columns]

    fig, ax = plt.subplots(figsize=(14, 4))
    vmax = max(abs(pivot2.values.min()), abs(pivot2.values.max()))
    sns.heatmap(pivot2, annot=True, fmt=",.0f", center=0,
                cmap=sns.diverging_palette(10, 133, as_cmap=True),
                linewidths=0.8, linecolor="white", ax=ax,
                vmin=-vmax, vmax=vmax,
                annot_kws={"fontsize": 9},
                cbar_kws={"label": "P&L ($)", "shrink": 0.8})
    ax.set_title("Monthly P&L Heatmap ($)", fontweight="bold", fontsize=14)
    ax.set_ylabel("")
    ax.set_xlabel("")
    return fig_to_base64(fig)


def plot_return_distribution(df):
    """Trade return distribution with VaR markers."""
    rets = df["return_pct"].values
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.linspace(np.percentile(rets, 0.5), np.percentile(rets, 99.5), 80)
    ax.hist(rets, bins=bins, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.5)

    var95 = np.percentile(rets, 5)
    var99 = np.percentile(rets, 1)
    ax.axvline(var95, color=GOLD, linewidth=2, linestyle="--", label=f"VaR 95%: {var95:.1f}%")
    ax.axvline(var99, color=RED, linewidth=2, linestyle="--", label=f"VaR 99%: {var99:.1f}%")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(np.mean(rets), color=GREEN, linewidth=2, linestyle="-", label=f"Mean: {np.mean(rets):.1f}%")

    ax.set_title("Trade Return Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Return per Trade (%)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.9)
    return fig_to_base64(fig)


def plot_win_loss_analysis(df):
    """Win/loss breakdown by direction and bucket."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1 — Win rate by direction
    for i, grp in enumerate(["direction", "asset_bucket"]):
        ax = axes[i]
        grouped = df.groupby(grp)["net_pnl"].apply(lambda x: (x > 0).mean() * 100)
        colors = [GREEN if v >= 40 else GOLD if v >= 30 else RED for v in grouped.values]
        bars = ax.barh(grouped.index, grouped.values, color=colors, edgecolor="white")
        ax.set_xlabel("Win Rate (%)")
        ax.set_title(f"Win Rate by {grp.replace('_', ' ').title()}", fontweight="bold")
        ax.axvline(50, color="grey", linewidth=0.8, linestyle=":")
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)

    # 3 — Profit factor by bucket
    ax = axes[2]
    buckets = df["asset_bucket"].unique()
    pfs = []
    for b in buckets:
        m = df["asset_bucket"] == b
        w = df.loc[m & (df["net_pnl"] > 0), "net_pnl"].sum()
        l = abs(df.loc[m & (df["net_pnl"] <= 0), "net_pnl"].sum())
        pfs.append(w / l if l > 0 else 0)
    colors = [GREEN if v >= 2 else GOLD if v >= 1 else RED for v in pfs]
    bars = ax.barh(buckets, pfs, color=colors, edgecolor="white")
    ax.set_xlabel("Profit Factor")
    ax.set_title("Profit Factor by Asset Class", fontweight="bold")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    for bar, val in zip(bars, pfs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_exposure_breakdown(df, s):
    """Notional exposure by bucket — pie + bar."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bucket_notional = df.groupby("asset_bucket")["notional"].sum().sort_values(ascending=False)
    colors_map = {"crypto": "#f7931a", "equity": BLUE, "gold": GOLD,
                  "metals": "#95a5a6", "energy": "#e67e22"}
    colors = [colors_map.get(b, GREY) for b in bucket_notional.index]

    axes[0].pie(bucket_notional, labels=bucket_notional.index, autopct="%1.1f%%",
                colors=colors, startangle=140, textprops={"fontsize": 10},
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Notional Exposure by Asset Class", fontweight="bold")

    # P&L contribution
    bucket_pnl = df.groupby("asset_bucket")["net_pnl"].sum().reindex(bucket_notional.index)
    bars = axes[1].bar(bucket_pnl.index, bucket_pnl.values, color=colors, edgecolor="white")
    axes[1].set_title("P&L Contribution by Asset Class", fontweight="bold")
    axes[1].set_ylabel("Total P&L ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, val in zip(bars, bucket_pnl.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_correlation_matrix(df):
    """Weekly P&L correlation heatmap."""
    df2 = df.copy()
    df2["exit_week"] = df2["exit_ts"].dt.to_period("W")
    weekly = df2.groupby(["exit_week", "ticker"])["net_pnl"].sum().unstack(fill_value=0)
    active = weekly.columns[weekly.astype(bool).sum() >= 8]
    corr = weekly[active].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"fontsize": 8},
                cbar_kws={"shrink": 0.75, "label": "Correlation"})
    ax.set_title("Ticker Pairwise Correlation (Weekly P&L)", fontweight="bold", fontsize=14)
    return fig_to_base64(fig)


def plot_bucket_correlation(df):
    """Bucket-level weekly correlation matrix."""
    df2 = df.copy()
    df2["exit_week"] = df2["exit_ts"].dt.to_period("W")
    weekly_bucket = df2.groupby(["exit_week", "asset_bucket"])["net_pnl"].sum().unstack(fill_value=0)
    corr = weekly_bucket.corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".3f", center=0, square=True,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"fontsize": 11, "fontweight": "bold"},
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax.set_title("Asset Class Correlation (Weekly P&L)", fontweight="bold", fontsize=13)
    return fig_to_base64(fig)


def plot_rolling_sharpe(df, s):
    """Rolling 50-trade Sharpe ratio."""
    equity = [s["initial_capital"]]
    for pnl in df["net_pnl"]:
        equity.append(equity[-1] + pnl)
    eq = np.array(equity[:-1])
    ret_on_eq = df["net_pnl"].values / eq

    window = 50
    if len(ret_on_eq) < window:
        return None

    n_trades = len(df)
    start = df["entry_ts"].min()
    end = df["exit_ts"].max()
    years = (end - start).total_seconds() / (365.25 * 24 * 3600)
    tpy = n_trades / years

    rolling_mean = pd.Series(ret_on_eq).rolling(window).mean()
    rolling_std = pd.Series(ret_on_eq).rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(tpy)
    dates = df["exit_ts"]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(dates, rolling_sharpe, color=BLUE, linewidth=1.4)
    ax.fill_between(dates, rolling_sharpe, alpha=0.15, color=BLUE)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.0, color=GOLD, linewidth=1, linestyle="--", alpha=0.7, label="Sharpe = 1.0")
    ax.axhline(2.0, color=GREEN, linewidth=1, linestyle="--", alpha=0.7, label="Sharpe = 2.0")
    ax.set_title(f"Rolling {window}-Trade Annualised Sharpe Ratio", fontweight="bold", fontsize=14)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return fig_to_base64(fig)


def plot_monte_carlo(df, s):
    """Monte Carlo forward simulation fan chart."""
    np.random.seed(42)
    trade_pnl = df["net_pnl"].values
    n_years = len(df)
    start_ts = df["entry_ts"].min()
    end_ts = df["exit_ts"].max()
    years = (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600)
    trades_per_year = int(n_years / years) if years > 0 else n_years
    n_forward = trades_per_year  # 1 year forward

    n_sims = 5000
    paths = np.zeros((n_sims, n_forward + 1))
    start_eq = s["final_equity"]
    paths[:, 0] = start_eq

    for sim in range(n_sims):
        sampled = np.random.choice(trade_pnl, size=n_forward, replace=True)
        paths[sim, 1:] = start_eq + np.cumsum(sampled)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_forward + 1)

    # Percentile bands
    for lo, hi, alpha, label in [
        (1, 99, 0.08, "P1–P99"),
        (5, 95, 0.12, "P5–P95"),
        (10, 90, 0.15, "P10–P90"),
        (25, 75, 0.20, "P25–P75"),
    ]:
        p_lo = np.percentile(paths, lo, axis=0)
        p_hi = np.percentile(paths, hi, axis=0)
        ax.fill_between(x, p_lo, p_hi, alpha=alpha, color=BLUE, label=label)

    ax.plot(x, np.median(paths, axis=0), color=BLUE, linewidth=2.5, label="Median")
    ax.plot(x, np.percentile(paths, 5, axis=0), color=RED, linewidth=1, linestyle="--", label="P5 (worst)")
    ax.plot(x, np.percentile(paths, 95, axis=0), color=GREEN, linewidth=1, linestyle="--", label="P95 (best)")
    ax.axhline(start_eq, color="grey", linewidth=0.8, linestyle=":")

    ax.set_title("Monte Carlo Forward Simulation (1-Year, 5,000 Paths)", fontweight="bold", fontsize=14)
    ax.set_xlabel(f"Trade Number (≈{trades_per_year} trades/year)")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    return fig_to_base64(fig), paths


def plot_mc_distribution(paths, s):
    """Terminal wealth distribution from Monte Carlo."""
    finals = paths[:, -1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(finals, bins=80, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.5)

    p5 = np.percentile(finals, 5)
    median = np.median(finals)
    p95 = np.percentile(finals, 95)
    ax.axvline(s["final_equity"], color="grey", linewidth=2, linestyle=":", label=f"Current: ${s['final_equity']:,.0f}")
    ax.axvline(p5, color=RED, linewidth=2, linestyle="--", label=f"P5: ${p5:,.0f}")
    ax.axvline(median, color=BLUE, linewidth=2, label=f"Median: ${median:,.0f}")
    ax.axvline(p95, color=GREEN, linewidth=2, linestyle="--", label=f"P95: ${p95:,.0f}")

    ax.set_title("Simulated 1-Year Terminal Equity Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Terminal Equity ($)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper right", framealpha=0.9)
    return fig_to_base64(fig)


def plot_max_drawdown_distribution(df, s):
    """Monte Carlo max-drawdown distribution."""
    np.random.seed(123)
    trade_pnl = df["net_pnl"].values
    n = len(df)
    start_ts = df["entry_ts"].min()
    end_ts = df["exit_ts"].max()
    years = (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600)
    tpy = int(n / years) if years > 0 else n

    n_sims = 10_000
    max_dds = []
    start_eq = s["final_equity"]
    for _ in range(n_sims):
        sampled = np.random.choice(trade_pnl, size=tpy, replace=True)
        eq = start_eq
        peak = eq
        worst_dd = 0
        for pnl in sampled:
            eq += pnl
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            worst_dd = max(worst_dd, dd)
        max_dds.append(worst_dd * 100)

    max_dds = np.array(max_dds)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(max_dds, bins=80, color=RED, alpha=0.5, edgecolor="white", linewidth=0.5)

    p50 = np.median(max_dds)
    p95 = np.percentile(max_dds, 95)
    p99 = np.percentile(max_dds, 99)
    ax.axvline(p50, color=GOLD, linewidth=2, linestyle="--", label=f"Median DD: {p50:.1f}%")
    ax.axvline(p95, color=RED, linewidth=2, linestyle="--", label=f"P95 DD: {p95:.1f}%")
    ax.axvline(p99, color="#8b0000", linewidth=2, linestyle=":", label=f"P99 DD: {p99:.1f}%")
    ax.axvline(s["max_realized_drawdown_pct"], color="black", linewidth=2, label=f"Historical: {s['max_realized_drawdown_pct']:.1f}%")

    ax.set_title("Simulated Maximum Drawdown Distribution (1-Year)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.9)
    return fig_to_base64(fig)


def plot_regime_analysis(df):
    """VIX/FG proxy regime and EMA overlay performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # VIX/FG regime
    reg = df.groupby("ext_hours_proxy_regime").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
    ).reindex(["risk_on_micro", "neutral_micro", "risk_off_micro"])
    colors = [GREEN, GREY, RED]
    bars = axes[0].bar(reg.index, reg["total_pnl"], color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("P&L by VIX/Fear-Greed Regime", fontweight="bold")
    axes[0].set_ylabel("Total P&L ($)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, val, n in zip(bars, reg["total_pnl"], reg["trades"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val:,.0f}\n({n:.0f} trades)", ha="center", va="bottom", fontsize=9)

    # EMA regime
    ema = df.groupby("technical_ema_regime").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
    )
    ema = ema[ema.index.notna() & (ema.index != "")]
    colors2 = [GREEN if idx == "above_ema" else RED for idx in ema.index]
    bars2 = axes[1].bar(ema.index, ema["total_pnl"], color=colors2, edgecolor="white", width=0.5)
    axes[1].set_title("P&L by EMA-200 Regime", fontweight="bold")
    axes[1].set_ylabel("Total P&L ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, val, n in zip(bars2, ema["total_pnl"], ema["trades"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val:,.0f}\n({n:.0f} trades)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_top_tickers(df):
    """Top and bottom tickers by P&L."""
    ticker_pnl = df.groupby("ticker")["net_pnl"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [GREEN if v >= 0 else RED for v in ticker_pnl.values]
    ax.barh(ticker_pnl.index, ticker_pnl.values, color=colors, edgecolor="white")
    ax.set_title("Total P&L by Ticker", fontweight="bold", fontsize=14)
    ax.set_xlabel("Total P&L ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (ticker, val) in enumerate(zip(ticker_pnl.index, ticker_pnl.values)):
        ha = "left" if val >= 0 else "right"
        offset = 80 if val >= 0 else -80
        ax.text(val + offset, i, f"${val:,.0f}", va="center", ha=ha, fontsize=8)
    return fig_to_base64(fig)


def plot_quarterly_performance(df):
    """Quarterly P&L bar chart."""
    q = df.groupby("exit_quarter")["net_pnl"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [GREEN if v >= 0 else RED for v in q.values]
    x_labels = [str(p) for p in q.index]
    bars = ax.bar(x_labels, q.values, color=colors, edgecolor="white", width=0.7)
    ax.set_title("Quarterly P&L Performance", fontweight="bold", fontsize=14)
    ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.axhline(0, color="black", linewidth=0.5)
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, q.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (150 if val >= 0 else -250),
                f"${val:,.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_holding_period(df):
    """Holding period distribution, colored by outcome."""
    fig, ax = plt.subplots(figsize=(12, 5))
    winners = df.loc[df["net_pnl"] > 0, "holding_hours"]
    losers = df.loc[df["net_pnl"] <= 0, "holding_hours"]
    bins = np.linspace(0, min(df["holding_hours"].quantile(0.99), 1500), 60)
    ax.hist(winners, bins=bins, alpha=0.6, color=GREEN, label=f"Winners ({len(winners)})", edgecolor="white")
    ax.hist(losers, bins=bins, alpha=0.5, color=RED, label=f"Losers ({len(losers)})", edgecolor="white")
    ax.axvline(df["holding_hours"].median(), color="black", linewidth=1.5, linestyle="--",
               label=f"Median: {df['holding_hours'].median():.0f}h")
    ax.set_title("Holding Period Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.9)
    return fig_to_base64(fig)


def plot_leverage_timeline(df, s):
    """Concurrent positions and leverage over time."""
    events = []
    for _, row in df.iterrows():
        events.append((row["entry_ts"], "open", row["notional"], row.name))
        events.append((row["exit_ts"], "close", row["notional"], row.name))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "close" else 1))

    eq = s["initial_capital"]
    current = {}
    ts_list, lev_list, conc_list = [], [], []
    closed_pnl = {}

    for ts, action, notional, idx in events:
        if action == "open":
            current[idx] = notional
        else:
            if idx in current:
                del current[idx]
            eq += df.loc[idx, "net_pnl"]
        total_notional = sum(current.values())
        leverage = total_notional / eq if eq > 0 else 0
        ts_list.append(ts)
        lev_list.append(leverage)
        conc_list.append(len(current))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                     gridspec_kw={"hspace": 0.08})
    ax1.fill_between(ts_list, conc_list, alpha=0.3, color=BLUE, step="post")
    ax1.step(ts_list, conc_list, color=BLUE, linewidth=1, where="post")
    ax1.set_ylabel("Concurrent Positions")
    ax1.set_title("Portfolio Leverage & Position Count Over Time", fontweight="bold", fontsize=14)

    ax2.fill_between(ts_list, lev_list, alpha=0.3, color=PURPLE, step="post")
    ax2.step(ts_list, lev_list, color=PURPLE, linewidth=1, where="post")
    ax2.axhline(s["exposure_mult"], color=RED, linewidth=1, linestyle="--",
                label=f"Max exposure: {s['exposure_mult']}x", alpha=0.7)
    ax2.set_ylabel("Leverage (x)")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────
# COMPUTE TABLES
# ─────────────────────────────────────────────────────────────────────

def compute_risk_metrics(df, s):
    equity = [s["initial_capital"]]
    for pnl in df["net_pnl"]:
        equity.append(equity[-1] + pnl)
    eq = np.array(equity[:-1])
    ret_on_eq = df["net_pnl"].values / eq

    n = len(df)
    start = df["entry_ts"].min()
    end = df["exit_ts"].max()
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    tpy = n / years

    mean_r = ret_on_eq.mean()
    std_r = ret_on_eq.std()
    down_r = ret_on_eq[ret_on_eq < 0]
    down_std = down_r.std() if len(down_r) > 0 else 1e-9
    ann_ret = mean_r * tpy
    ann_std = std_r * np.sqrt(tpy)
    ann_down = down_std * np.sqrt(tpy)

    var95 = np.percentile(ret_on_eq, 5)
    cvar95 = ret_on_eq[ret_on_eq <= var95].mean()
    var99 = np.percentile(ret_on_eq, 1)
    cvar99 = ret_on_eq[ret_on_eq <= var99].mean()

    return {
        "total_return": f"{s['total_return_pct']:,.2f}%",
        "cagr": f"{s['cagr_pct']:.2f}%",
        "max_dd": f"{s['max_realized_drawdown_pct']:.2f}%",
        "sharpe": f"{ann_ret / ann_std:.2f}" if ann_std > 0 else "N/A",
        "sortino": f"{ann_ret / ann_down:.2f}" if ann_down > 0 else "N/A",
        "calmar": f"{s['cagr_pct'] / s['max_realized_drawdown_pct']:.2f}" if s['max_realized_drawdown_pct'] > 0 else "N/A",
        "profit_factor": f"{s['profit_factor']:.2f}",
        "win_rate": f"{s['win_rate_pct']:.2f}%",
        "var95": f"{var95:.2%}",
        "cvar95": f"{cvar95:.2%}",
        "var99": f"{var99:.2%}",
        "cvar99": f"{cvar99:.2%}",
        "skewness": f"{stats.skew(ret_on_eq):.2f}",
        "kurtosis": f"{stats.kurtosis(ret_on_eq):.2f}",
        "trades": str(s["executed_trades"]),
        "long": str(s["long_trades"]),
        "short": str(s["short_trades"]),
        "avg_trade": f"${df['net_pnl'].mean():,.2f}",
        "median_trade": f"${df['net_pnl'].median():,.2f}",
        "years": f"{years:.2f}",
        "initial_capital": f"${s['initial_capital']:,.2f}",
        "final_equity": f"${s['final_equity']:,.2f}",
    }


# ─────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session Turtle Trend x3 — Investor Research Report</title>
<style>
  @page { size: A4 landscape; margin: 12mm; }
  @media print {
    .page-break { page-break-before: always; }
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: #1a1a2e; background: #fff; line-height: 1.55;
    max-width: 1300px; margin: auto; padding: 28px 36px;
  }
  h1 { font-size: 28px; margin-bottom: 4px; }
  h2 {
    font-size: 20px; color: #16213e; margin: 32px 0 14px;
    border-bottom: 2px solid #2980b9; padding-bottom: 6px;
  }
  h3 { font-size: 15px; color: #34495e; margin: 20px 0 8px; }
  p, li { font-size: 13px; }
  .subtitle { font-size: 14px; color: #555; margin-bottom: 18px; }
  .header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff; padding: 32px 36px; border-radius: 10px; margin-bottom: 28px;
  }
  .header-bar h1 { color: #fff; font-size: 30px; }
  .header-bar .subtitle { color: #b0c4de; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px; margin: 18px 0 26px;
  }
  .kpi {
    background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px 18px; text-align: center;
  }
  .kpi .value { font-size: 26px; font-weight: 700; color: #16213e; }
  .kpi .label { font-size: 11px; color: #777; text-transform: uppercase; letter-spacing: 0.5px; }
  .plot { text-align: center; margin: 18px 0; }
  .plot img { max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }
  table {
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12px;
  }
  th {
    background: #16213e; color: #fff; padding: 10px 12px; text-align: left;
    font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;
  }
  td { padding: 8px 12px; border-bottom: 1px solid #e8e8e8; }
  tr:nth-child(even) { background: #f8f9fa; }
  .highlight-green { color: #27ae60; font-weight: 600; }
  .highlight-red { color: #e74c3c; font-weight: 600; }
  .disclaimer {
    margin-top: 40px; padding: 20px; background: #fef9e7; border: 1px solid #f0e68c;
    border-radius: 8px; font-size: 11px; color: #666;
  }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  .risk-table td:first-child { font-weight: 600; width: 45%; }
  .section-note { font-size: 12px; color: #888; font-style: italic; margin-bottom: 12px; }
  .footer { text-align: center; font-size: 11px; color: #aaa; margin-top: 36px; padding-top: 16px; border-top: 1px solid #eee; }
</style>
</head>
<body>

<div class="header-bar">
  <h1>Session Turtle Trend x3</h1>
  <p class="subtitle">Quantitative Strategy Research Report &mdash; Document Review Variant<br>
  Report generated {{report_date}} &bull; Data period: {{start_date}} to {{end_date}}</p>
</div>

<!-- ═══════ EXECUTIVE SUMMARY ═══════ -->
<h2>1. Executive Summary</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="value">{{m.total_return}}</div><div class="label">Total Return</div></div>
  <div class="kpi"><div class="value">{{m.cagr}}</div><div class="label">CAGR</div></div>
  <div class="kpi"><div class="value">{{m.max_dd}}</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value">{{m.sharpe}}</div><div class="label">Sharpe Ratio</div></div>
  <div class="kpi"><div class="value">{{m.sortino}}</div><div class="label">Sortino Ratio</div></div>
  <div class="kpi"><div class="value">{{m.calmar}}</div><div class="label">Calmar Ratio</div></div>
  <div class="kpi"><div class="value">{{m.profit_factor}}</div><div class="label">Profit Factor</div></div>
  <div class="kpi"><div class="value">{{m.win_rate}}</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">{{m.initial_capital}}</div><div class="label">Initial Capital</div></div>
  <div class="kpi"><div class="value">{{m.final_equity}}</div><div class="label">Final Equity</div></div>
  <div class="kpi"><div class="value">{{m.trades}}</div><div class="label">Executed Trades</div></div>
  <div class="kpi"><div class="value">{{m.years}}</div><div class="label">Backtest Years</div></div>
</div>

<p>The strategy employs a <strong>Donchian channel breakout</strong> system on 5-minute session
bars across a diversified multi-asset universe (crypto, equities, gold, metals, energy).
Signals are filtered by a 4-hour EMA(55)/EMA(200) trend filter and sized via
risk-percentage position sizing at <strong>3.0x</strong> notional exposure.
Overlays include a drawdown governor (15%/25% triggers), a VIX/Fear-Greed regime
proxy, per-asset daily EMA-200 quarter-sizing, and a breakout conviction boost.</p>

<p>The backtest was <strong>audited for forward-looking bias</strong> — all signals use
completed bars, entries fill at next-bar open with slippage, and overlay lookups
apply 1-day lags. No look-ahead bias was detected.</p>

<!-- ═══════ PERFORMANCE ═══════ -->
<h2 class="page-break">2. Performance Analysis</h2>

<h3>2.1 Equity Curve &amp; Drawdowns</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_equity}}" alt="Equity Curve"></div>

<h3>2.2 Monthly P&amp;L Heatmap</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_heatmap}}" alt="Monthly Heatmap"></div>

<h3>2.3 Quarterly Performance</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_quarterly}}" alt="Quarterly"></div>

<h3>2.4 Rolling Sharpe Ratio</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_rolling_sharpe}}" alt="Rolling Sharpe"></div>

<!-- ═══════ RISK ANALYSIS ═══════ -->
<h2 class="page-break">3. Risk Analysis</h2>

<h3>3.1 Key Risk Metrics</h3>
<div class="two-col">
<table class="risk-table">
  <tr><th colspan="2">Return Risk</th></tr>
  <tr><td>VaR (95%, per trade)</td><td>{{m.var95}}</td></tr>
  <tr><td>CVaR / Expected Shortfall (95%)</td><td>{{m.cvar95}}</td></tr>
  <tr><td>VaR (99%, per trade)</td><td>{{m.var99}}</td></tr>
  <tr><td>CVaR / Expected Shortfall (99%)</td><td>{{m.cvar99}}</td></tr>
  <tr><td>Return Skewness</td><td>{{m.skewness}}</td></tr>
  <tr><td>Excess Kurtosis</td><td>{{m.kurtosis}}</td></tr>
</table>
<table class="risk-table">
  <tr><th colspan="2">Portfolio Characteristics</th></tr>
  <tr><td>Average Trade P&L</td><td>{{m.avg_trade}}</td></tr>
  <tr><td>Median Trade P&L</td><td>{{m.median_trade}}</td></tr>
  <tr><td>Long Trades</td><td>{{m.long}}</td></tr>
  <tr><td>Short Trades</td><td>{{m.short}}</td></tr>
  <tr><td>Max Drawdown</td><td>{{m.max_dd}}</td></tr>
  <tr><td>Profit Factor</td><td>{{m.profit_factor}}</td></tr>
</table>
</div>

<h3>3.2 Trade Return Distribution</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_distribution}}" alt="Return Distribution"></div>

<h3>3.3 Win/Loss Analysis</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_winloss}}" alt="Win/Loss"></div>

<!-- ═══════ EXPOSURE ═══════ -->
<h2 class="page-break">4. Exposure &amp; Leverage Analysis</h2>

<h3>4.1 Asset Class Exposure &amp; P&L Contribution</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_exposure}}" alt="Exposure"></div>

<h3>4.2 Leverage &amp; Concurrent Positions Over Time</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_leverage}}" alt="Leverage"></div>

<h3>4.3 Holding Period Distribution</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_holding}}" alt="Holding Period"></div>

<h3>4.4 Per-Ticker P&L Attribution</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_tickers}}" alt="Ticker P&L"></div>

<!-- ═══════ CORRELATION ═══════ -->
<h2 class="page-break">5. Correlation &amp; Diversification</h2>

<h3>5.1 Asset Class Correlation</h3>
<p class="section-note">Weekly P&L correlations between asset class buckets. Low cross-bucket
correlation indicates genuine diversification benefit.</p>
<div class="plot"><img src="data:image/png;base64,{{plot_bucket_corr}}" alt="Bucket Correlation"></div>

<h3>5.2 Ticker-Level Correlation Matrix</h3>
<p class="section-note">Pairwise weekly P&L correlations across all active tickers.
High correlation pairs represent concentration risk.</p>
<div class="plot"><img src="data:image/png;base64,{{plot_corr}}" alt="Correlation Matrix"></div>

<!-- ═══════ REGIME ANALYSIS ═══════ -->
<h2 class="page-break">6. Regime &amp; Overlay Analysis</h2>

<h3>6.1 VIX/Fear-Greed &amp; EMA Regime Performance</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_regime}}" alt="Regime"></div>

<!-- ═══════ STRESS TESTING ═══════ -->
<h2 class="page-break">7. Stress Testing &amp; Forward Simulation</h2>

<h3>7.1 Monte Carlo Forward Simulation (1-Year)</h3>
<p class="section-note">5,000 bootstrapped paths projecting forward from current equity.
Trade P&L samples drawn with replacement from historical distribution.</p>
<div class="plot"><img src="data:image/png;base64,{{plot_mc}}" alt="Monte Carlo"></div>

<h3>7.2 Simulated Terminal Wealth Distribution</h3>
<div class="plot"><img src="data:image/png;base64,{{plot_mc_dist}}" alt="MC Distribution"></div>

<h3>7.3 Simulated Maximum Drawdown Distribution</h3>
<p class="section-note">Distribution of worst drawdowns across 10,000 simulated 1-year periods.
Provides probabilistic bounds on expected peak-to-trough losses.</p>
<div class="plot"><img src="data:image/png;base64,{{plot_mc_dd}}" alt="MC Drawdown"></div>

<!-- ═══════ APPENDIX ═══════ -->
<h2 class="page-break">8. Appendix</h2>

<h3>8.1 Strategy Parameters</h3>
<table>
<tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
<tr><td>Channel Period</td><td>10/5</td><td>Entry/exit Donchian channel lookback (session bars)</td></tr>
<tr><td>Exposure Multiplier</td><td>3.0x</td><td>Maximum notional leverage</td></tr>
<tr><td>Base Risk %</td><td>5.0%</td><td>Risk per trade as % of equity</td></tr>
<tr><td>Fixed Stop</td><td>10.0%</td><td>Hard stop-loss distance</td></tr>
<tr><td>Trend Filter</td><td>EMA 55/200 (4H)</td><td>Directional filter — longs above, shorts below</td></tr>
<tr><td>DD Governor T1</td><td>15% → 1.5x</td><td>Reduce exposure at 15% drawdown</td></tr>
<tr><td>DD Governor T2</td><td>25% → 0.5x</td><td>Severely reduce at 25% drawdown</td></tr>
<tr><td>Conviction Boost</td><td>Max 1.25x</td><td>4-factor composite: volume, ratio, breakout, close location</td></tr>
<tr><td>VIX/FG Proxy</td><td>1-day lag</td><td>VIX ≤15 risk-on, ≥25 risk-off; FG ≥60 greed, ≤30 fear</td></tr>
<tr><td>EMA-200 Overlay</td><td>0.25x quarter-sizing</td><td>Counter-trend trades sized at 25% of normal</td></tr>
<tr><td>Portfolio Cap</td><td>90%</td><td>Maximum portfolio notional as % of equity &times; exposure</td></tr>
</table>

<h3>8.2 Universe Composition</h3>
<table>
<tr><th>Asset Class</th><th>Tickers</th></tr>
<tr><td>Crypto</td><td>BTC-USD, ETH-USD, SOL-USD, PAXG-USD</td></tr>
<tr><td>Equity</td><td>NVDA, META, GOOGL, AMZN, TSLA, PLTR, INTC, HOOD, COIN, MSTR, CRCL</td></tr>
<tr><td>Gold</td><td>PAXG-USD (via crypto sessions)</td></tr>
<tr><td>Metals</td><td>SLV, PPLT, COPPER-USD</td></tr>
<tr><td>Energy</td><td>BRENT, NATGAS-USD</td></tr>
</table>

<h3>8.3 Methodology Notes</h3>
<ul>
<li>All backtests use 5-minute bar data from Binance (crypto) and Tiingo (equities/metals/energy).</li>
<li>Entries execute at the <strong>next bar's open price</strong> with slippage applied.</li>
<li>Per-asset signals are generated independently; portfolio allocation occurs chronologically.</li>
<li>Monte Carlo simulations use i.i.d. bootstrapping from the empirical trade P&L distribution.</li>
<li>Sharpe/Sortino ratios are annualised using observed trades-per-year (~133).</li>
<li>VaR/CVaR are computed at the individual trade level relative to portfolio equity at entry.</li>
</ul>

<div class="disclaimer">
<strong>Important Disclosures:</strong> This document is provided for informational and research
purposes only. Past performance, whether actual or simulated, is not indicative of future
results. The strategy results shown are based on backtested data and do not account for
all potential real-world factors including but not limited to: slippage beyond modelled
estimates, exchange downtime, liquidity constraints, regulatory changes, or funding costs.
All figures assume reinvestment of profits. This is not investment advice. Prospective
investors should conduct their own due diligence and consult with qualified advisors
before making any investment decisions.
</div>

<div class="footer">
  Confidential &mdash; Prepared for qualified investors only &mdash; {{report_date}}
</div>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df, s = load()

    print("Computing risk metrics...")
    m = compute_risk_metrics(df, s)

    print("Generating plots...")
    plot_equity, _, _ = plot_equity_curve(df, s)
    plot_heatmap = plot_monthly_returns_heatmap(df)
    plot_distribution = plot_return_distribution(df)
    plot_winloss = plot_win_loss_analysis(df)
    plot_exposure = plot_exposure_breakdown(df, s)
    plot_corr = plot_correlation_matrix(df)
    plot_b_corr = plot_bucket_correlation(df)
    plot_rs = plot_rolling_sharpe(df, s)
    plot_mc_img, mc_paths = plot_monte_carlo(df, s)
    plot_mc_d = plot_mc_distribution(mc_paths, s)
    plot_mc_dd = plot_max_drawdown_distribution(df, s)
    plot_reg = plot_regime_analysis(df)
    plot_tick = plot_top_tickers(df)
    plot_qtr = plot_quarterly_performance(df)
    plot_hold = plot_holding_period(df)
    plot_lev = plot_leverage_timeline(df, s)

    print("Rendering HTML report...")
    from jinja2 import Template
    tmpl = Template(HTML_TEMPLATE)
    html = tmpl.render(
        report_date=datetime.now().strftime("%B %d, %Y"),
        start_date=s["start_date"][:10],
        end_date=s["end_date"][:10],
        m=m,
        plot_equity=plot_equity,
        plot_heatmap=plot_heatmap,
        plot_distribution=plot_distribution,
        plot_winloss=plot_winloss,
        plot_exposure=plot_exposure,
        plot_corr=plot_corr,
        plot_bucket_corr=plot_b_corr,
        plot_rolling_sharpe=plot_rs or "",
        plot_mc=plot_mc_img,
        plot_mc_dist=plot_mc_d,
        plot_mc_dd=plot_mc_dd,
        plot_regime=plot_reg,
        plot_tickers=plot_tick,
        plot_quarterly=plot_qtr,
        plot_holding=plot_hold,
        plot_leverage=plot_lev,
    )

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_HTML}")
    print(f"File size: {OUTPUT_HTML.stat().st_size / 1024 / 1024:.1f} MB")
    print("Open in browser and print to PDF for distribution.")


if __name__ == "__main__":
    main()
