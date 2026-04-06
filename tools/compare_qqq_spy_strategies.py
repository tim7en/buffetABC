"""
tools/compare_qqq_spy_strategies.py

Compares 7 strategies on QQQ & SPY over the full data period (Mar 2021 → Mar 2026).

Strategies
----------
1. Buy & Hold QQQ
2. Buy & Hold SPY
3. DCA Monthly          — invest equal installments each month into 50/50 QQQ+SPY
4. Donchian 20/10 QQQ   — long-only daily Donchian channel on QQQ
5. Donchian 20/10 SPY   — long-only daily Donchian channel on SPY
6. SMA-200 Filter SPY   — hold SPY when close > 200-day SMA, else cash
7. QQQ/SPY Rotation     — monthly pick whichever has stronger 63-day momentum

All strategies start with $10 000.  DCA splits the $10 000 into equal monthly
installments over the full period (the undeployed portion sits in cash at 0%).

Output
------
  reports/qqq_spy_comparison/strategy_comparison.png
  reports/qqq_spy_comparison/strategy_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TIINGO = ROOT / "cache" / "cache" / "tiingo"
OUT = ROOT / "reports" / "qqq_spy_comparison"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_daily(symbol: str) -> pd.DataFrame:
    """Load 5-min parquet → daily OHLCV (US/Eastern date)."""
    path = TIINGO / f"{symbol}_5m.parquet"
    tbl = pq.read_table(path, columns=["time", "o", "h", "l", "c", "v"])
    df = tbl.to_pandas()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # convert to US/Eastern for correct date grouping
    df["date"] = df["time"].dt.tz_convert("America/New_York").dt.date
    daily = (
        df.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def _metrics(equity: pd.Series, label: str) -> dict:
    """Compute CAGR, max drawdown, Sharpe, total return from an equity series."""
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()

    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    return {
        "Strategy": label,
        "Total Return": f"{total_ret * 100:.1f}%",
        "CAGR": f"{cagr * 100:.1f}%",
        "Max Drawdown": f"{max_dd * 100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Final Value": f"${eq.iloc[-1]:,.0f}",
    }


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading QQQ …")
qqq = _load_daily("QQQ")
print("Loading SPY …")
spy = _load_daily("SPY")

# align to common dates
common = sorted(set(qqq["date"]) & set(spy["date"]))
qqq = qqq[qqq["date"].isin(common)].set_index("date").sort_index()
spy = spy[spy["date"].isin(common)].set_index("date").sort_index()

dates = qqq.index
n = len(dates)
print(f"Common daily bars: {n}  ({dates[0].date()} to {dates[-1].date()})")

# ── Strategy 1 & 2: Buy & Hold ────────────────────────────────────────────────

def buy_and_hold(price_series: pd.Series, capital: float = INITIAL_CAPITAL) -> pd.Series:
    shares = capital / price_series.iloc[0]
    return (shares * price_series).rename(price_series.name)

eq_bh_qqq = buy_and_hold(qqq["close"]).rename("Buy&Hold QQQ")
eq_bh_spy = buy_and_hold(spy["close"]).rename("Buy&Hold SPY")

# ── Strategy 3: DCA Monthly (50/50 QQQ+SPY) ──────────────────────────────────

def dca_monthly(qqq_c: pd.Series, spy_c: pd.Series, capital: float = INITIAL_CAPITAL) -> pd.Series:
    """
    Split capital into equal monthly installments.
    Each month's first trading day: buy 50% QQQ + 50% SPY with that installment.
    Remaining cash earns 0%.
    """
    months = pd.PeriodIndex(qqq_c.index, freq="M").unique()
    n_months = len(months)
    installment = capital / n_months

    qqq_shares = 0.0
    spy_shares = 0.0
    cash = capital  # all capital starts as cash

    qqq_shares_ts: list[float] = []
    spy_shares_ts: list[float] = []
    cash_ts: list[float] = []

    for date, qc, sc in zip(qqq_c.index, qqq_c.values, spy_c.values):
        period = pd.Period(date, freq="M")
        if period in months:
            # first trading day of this month? — check if previous bar was in prior month
            idx = qqq_c.index.get_loc(date)
            is_first = (idx == 0) or (pd.Period(qqq_c.index[idx - 1], freq="M") != period)
            if is_first and cash >= installment:
                half = installment / 2
                qqq_shares += half / qc
                spy_shares += half / sc
                cash -= installment
        qqq_shares_ts.append(qqq_shares)
        spy_shares_ts.append(spy_shares)
        cash_ts.append(cash)

    eq = (
        np.array(qqq_shares_ts) * qqq_c.values
        + np.array(spy_shares_ts) * spy_c.values
        + np.array(cash_ts)
    )
    return pd.Series(eq, index=qqq_c.index, name="DCA Monthly 50/50")

eq_dca = dca_monthly(qqq["close"], spy["close"])

# ── Strategy 4 & 5: Donchian 20/10 Long-Only ─────────────────────────────────

def donchian_long_only(
    price: pd.Series,
    entry_period: int = 20,
    exit_period: int = 10,
    capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """
    Daily Donchian channel (long-only, 100% in/out).
    Entry : close breaks above highest close of prior `entry_period` bars
    Exit  : close breaks below lowest  close of prior `exit_period`  bars
    """
    closes = price.values.astype(float)
    eq = np.full(n, np.nan)
    equity = capital
    shares = 0.0
    in_trade = False

    for i in range(entry_period, n):
        c = closes[i]
        prev_closes = closes[max(0, i - entry_period): i]
        entry_level = prev_closes.max() if len(prev_closes) else c

        if in_trade:
            prev_exit = closes[max(0, i - exit_period): i]
            exit_level = prev_exit.min() if len(prev_exit) else c
            if c < exit_level:
                equity = shares * c
                shares = 0.0
                in_trade = False
        else:
            if c > entry_level:
                shares = equity / c
                in_trade = True

        eq[i] = shares * c if in_trade else equity

    # fill warm-up period with starting capital
    eq[:entry_period] = capital
    # forward-fill any nans
    series = pd.Series(eq, index=price.index, name=price.name)
    return series.ffill()

eq_don_qqq = donchian_long_only(qqq["close"]).rename("Donchian 20/10 QQQ")
eq_don_spy = donchian_long_only(spy["close"]).rename("Donchian 20/10 SPY")

# ── Strategy 4b & 5b: Donchian 55/20 (Turtle System 2) ───────────────────────

eq_don55_qqq = donchian_long_only(qqq["close"], entry_period=55, exit_period=20).rename("Donchian 55/20 QQQ")
eq_don55_spy = donchian_long_only(spy["close"], entry_period=55, exit_period=20).rename("Donchian 55/20 SPY")

# ── Strategy 4c & 5c: Donchian 55/20 + SMA-200 filter ────────────────────────

def donchian_with_sma_filter(
    price: pd.Series,
    entry_period: int = 55,
    exit_period: int = 20,
    sma_period: int = 200,
    capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """
    Donchian long-only with SMA trend gate:
    - Only enter if close > SMA(sma_period)
    - Exit as normal on Donchian exit signal OR if close drops below SMA
    """
    closes = price.values.astype(float)
    warmup = max(entry_period, sma_period)
    sma = np.array([closes[max(0, i - sma_period + 1): i + 1].mean() for i in range(n)])

    eq = np.full(n, np.nan)
    equity = capital
    shares = 0.0
    in_trade = False

    for i in range(warmup, n):
        c = closes[i]
        above_sma = c > sma[i]
        prev_entry = closes[max(0, i - entry_period): i]
        entry_level = prev_entry.max() if len(prev_entry) else c

        if in_trade:
            prev_exit = closes[max(0, i - exit_period): i]
            exit_level = prev_exit.min() if len(prev_exit) else c
            if c < exit_level or not above_sma:
                equity = shares * c
                shares = 0.0
                in_trade = False
        else:
            if c > entry_level and above_sma:
                shares = equity / c
                in_trade = True

        eq[i] = shares * c if in_trade else equity

    eq[:warmup] = capital
    series = pd.Series(eq, index=price.index)
    return series.ffill()

eq_don55_sma_qqq = donchian_with_sma_filter(qqq["close"]).rename("Donchian 55/20+SMA200 QQQ")
eq_don55_sma_spy = donchian_with_sma_filter(spy["close"]).rename("Donchian 55/20+SMA200 SPY")

# ── Strategy 6: SMA-200 Trend Filter on SPY ───────────────────────────────────

def sma_filter(price: pd.Series, sma_period: int = 200, capital: float = INITIAL_CAPITAL) -> pd.Series:
    """Long when close > SMA(200), else cash (0% rate)."""
    closes = price.values.astype(float)
    sma = np.array([closes[max(0, i - sma_period + 1): i + 1].mean() for i in range(n)])

    eq = np.full(n, np.nan)
    equity = capital
    shares = 0.0
    in_trade = False

    for i in range(sma_period, n):
        c = closes[i]
        above = c > sma[i]

        if in_trade and not above:
            equity = shares * c
            shares = 0.0
            in_trade = False
        elif not in_trade and above:
            shares = equity / c
            in_trade = True

        eq[i] = shares * c if in_trade else equity

    eq[:sma_period] = capital
    series = pd.Series(eq, index=price.index, name="SMA-200 Filter SPY")
    return series.ffill()

eq_sma = sma_filter(spy["close"])

# ── Strategy 7: QQQ/SPY Momentum Rotation ────────────────────────────────────

def momentum_rotation(
    qqq_c: pd.Series,
    spy_c: pd.Series,
    lookback: int = 63,   # ~3 months
    capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """
    Monthly rebalance: hold whichever ETF has stronger 63-day return.
    If both are negative, go to cash.
    """
    qc = qqq_c.values.astype(float)
    sc = spy_c.values.astype(float)

    eq = np.full(n, np.nan)
    equity = capital
    qqq_shares = 0.0
    spy_shares = 0.0
    current = "cash"  # "qqq" | "spy" | "cash"

    prev_month = None

    for i in range(lookback, n):
        date = qqq_c.index[i]
        month = (date.year, date.month)

        # rebalance on first bar of each new month
        if month != prev_month:
            qqq_ret = qc[i] / qc[i - lookback] - 1
            spy_ret = sc[i] / sc[i - lookback] - 1

            # liquidate current position
            if current == "qqq":
                equity = qqq_shares * qc[i]
                qqq_shares = 0.0
            elif current == "spy":
                equity = spy_shares * sc[i]
                spy_shares = 0.0

            # allocate
            if qqq_ret <= 0 and spy_ret <= 0:
                current = "cash"
            elif qqq_ret >= spy_ret:
                qqq_shares = equity / qc[i]
                current = "qqq"
            else:
                spy_shares = equity / sc[i]
                current = "spy"

            prev_month = month

        # mark-to-market
        if current == "qqq":
            eq[i] = qqq_shares * qc[i]
        elif current == "spy":
            eq[i] = spy_shares * sc[i]
        else:
            eq[i] = equity

    eq[:lookback] = capital
    series = pd.Series(eq, index=qqq_c.index, name="QQQ/SPY Rotation")
    return series.ffill()

eq_rot = momentum_rotation(qqq["close"], spy["close"])

# ── collect all equity curves ─────────────────────────────────────────────────
all_eq = [
    eq_bh_qqq, eq_bh_spy, eq_dca,
    eq_don_qqq, eq_don_spy,
    eq_don55_qqq, eq_don55_spy,
    eq_don55_sma_qqq, eq_don55_sma_spy,
    eq_sma, eq_rot,
]

# ── compute metrics ───────────────────────────────────────────────────────────
rows = [_metrics(eq, eq.name) for eq in all_eq]
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT / "strategy_metrics.csv", index=False)
print("\n" + metrics_df.to_string(index=False))

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    "Buy&Hold QQQ":           "#1565C0",
    "Buy&Hold SPY":           "#42A5F5",
    "DCA Monthly 50/50":      "#2E7D32",
    "Donchian 20/10 QQQ":     "#F57F17",
    "Donchian 20/10 SPY":     "#FF8F00",
    "Donchian 55/20 QQQ":     "#BF360C",
    "Donchian 55/20 SPY":     "#FF7043",
    "Donchian 55/20+SMA200 QQQ": "#880E4F",
    "Donchian 55/20+SMA200 SPY": "#E91E63",
    "SMA-200 Filter SPY":     "#558B2F",
    "QQQ/SPY Rotation":       "#6A1B9A",
}
STYLES = {
    "Buy&Hold QQQ":           "-",
    "Buy&Hold SPY":           "--",
    "DCA Monthly 50/50":      "-",
    "Donchian 20/10 QQQ":     "-",
    "Donchian 20/10 SPY":     "--",
    "Donchian 55/20 QQQ":     "-",
    "Donchian 55/20 SPY":     "--",
    "Donchian 55/20+SMA200 QQQ": "-",
    "Donchian 55/20+SMA200 SPY": "--",
    "SMA-200 Filter SPY":     "-",
    "QQQ/SPY Rotation":       "-",
}

fig = plt.figure(figsize=(20, 18))
fig.suptitle(
    "QQQ & SPY — Strategy Comparison  (Mar 2021 to Mar 2026, $10 000 start)",
    fontsize=14, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40, height_ratios=[3, 2, 1.8])
ax_eq = fig.add_subplot(gs[0])
ax_dd = fig.add_subplot(gs[1])
ax_tb = fig.add_subplot(gs[2])
ax_tb.axis("off")

# equity curves (normalised to 1)
for eq in all_eq:
    norm = eq / eq.iloc[0]
    ax_eq.plot(
        eq.index, norm,
        color=COLORS[eq.name], linestyle=STYLES[eq.name],
        linewidth=1.6, label=eq.name,
    )
ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
ax_eq.set_ylabel("Normalised Value (start = 1.0)")
ax_eq.set_title("Equity Curves")
ax_eq.legend(fontsize=9, ncol=2, loc="upper left")
ax_eq.grid(alpha=0.25)

# drawdown curves
for eq in all_eq:
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    ax_dd.plot(
        eq.index, dd,
        color=COLORS[eq.name], linestyle=STYLES[eq.name],
        linewidth=1.2, label=eq.name,
    )
ax_dd.axhline(0, color="black", linewidth=0.6)
ax_dd.fill_between(dates, ax_dd.get_ylim()[0] if ax_dd.get_ylim()[0] < 0 else -1, 0,
                   alpha=0.04, color="red")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=8, ncol=2, loc="lower left")
ax_dd.grid(alpha=0.25)

# metrics table
col_labels = list(metrics_df.columns)
cell_text = metrics_df.values.tolist()
tbl = ax_tb.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
# colour header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#263238")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# alternate row shading
for i in range(1, len(cell_text) + 1):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor("#ECEFF1" if i % 2 == 0 else "white")

out_path = OUT / "strategy_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
