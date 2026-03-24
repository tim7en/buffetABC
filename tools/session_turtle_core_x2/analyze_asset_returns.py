"""Asset-level PnL breakdown for the x3 variant."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUT_DIR = Path("reports/capacity_test_core_x2/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "reports/capacity_test_core_x2/trades_high_exp_risk_5_exp_3x.csv",
    parse_dates=["entry_ts", "exit_ts"],
)
df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
df["equity_after_exit"] = pd.to_numeric(df["equity_after_exit"], errors="coerce")

BUCKET_COLORS = {
    "equity":  "#1565C0",
    "crypto":  "#6A1B9A",
    "gold":    "#F9A825",
    "metals":  "#558B2F",
}

# ── per-ticker totals ──────────────────────────────────────────────────────
ticker_stats = (
    df.groupby(["ticker", "asset_bucket"])
    .agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        avg_win=("net_pnl", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        avg_loss=("net_pnl", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
    )
    .reset_index()
    .sort_values("total_pnl", ascending=False)
)

# ── per-bucket totals ──────────────────────────────────────────────────────
bucket_stats = (
    df.groupby("asset_bucket")
    .agg(trades=("net_pnl", "count"), total_pnl=("net_pnl", "sum"),
         win_rate=("net_pnl", lambda x: (x > 0).mean() * 100))
    .reset_index()
    .sort_values("total_pnl", ascending=False)
)

# ── long vs short per ticker ───────────────────────────────────────────────
dir_stats = (
    df.groupby(["ticker", "direction", "asset_bucket"])
    .agg(trades=("net_pnl", "count"), total_pnl=("net_pnl", "sum"))
    .reset_index()
)

# ── equity curve per bucket ────────────────────────────────────────────────
df_sorted = df.sort_values("exit_ts")
bucket_cum = {}
for bkt in ["equity", "crypto", "gold", "metals"]:
    sub = df_sorted[df_sorted["asset_bucket"] == bkt]["net_pnl"]
    bucket_cum[bkt] = sub.cumsum().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Asset Returns — Session Turtle Core x3", fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_bar   = fig.add_subplot(gs[0, :])   # ticker PnL bar
ax_dir   = fig.add_subplot(gs[1, 0])   # long vs short heatmap
ax_bkt   = fig.add_subplot(gs[1, 1])   # bucket donut
ax_cum   = fig.add_subplot(gs[2, :])   # cumulative PnL by bucket

# 1. Ticker PnL horizontal bar ─────────────────────────────────────────────
tickers = ticker_stats["ticker"].tolist()
pnls    = ticker_stats["total_pnl"].tolist()
colors  = [BUCKET_COLORS[b] for b in ticker_stats["asset_bucket"]]
y = np.arange(len(tickers))
bars = ax_bar.barh(y, pnls, color=colors, alpha=0.85, edgecolor="white")
ax_bar.axvline(0, color="black", linewidth=0.8)
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(tickers, fontsize=10)
ax_bar.set_xlabel("Total Net PnL ($)")
ax_bar.set_title("Total PnL by Ticker (x3 variant, all directions combined)", fontsize=11)
ax_bar.grid(alpha=0.3, axis="x")
for bar, v in zip(bars, pnls):
    offset = 30 if v >= 0 else -30
    ha = "left" if v >= 0 else "right"
    ax_bar.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"${v:+,.0f}", va="center", ha=ha, fontsize=8)
# legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=b) for b, c in BUCKET_COLORS.items()]
ax_bar.legend(handles=legend_els, loc="lower right", fontsize=9)

# 2. Long vs short PnL per ticker ──────────────────────────────────────────
pivot = dir_stats.pivot_table(index="ticker", columns="direction",
                               values="total_pnl", fill_value=0)
pivot = pivot.reindex(ticker_stats["ticker"].tolist())
x = np.arange(len(pivot))
w = 0.38
long_vals  = pivot.get("long",  pd.Series(0, index=pivot.index)).values
short_vals = pivot.get("short", pd.Series(0, index=pivot.index)).values
ax_dir.bar(x - w/2, long_vals,  w, label="Long",  color="#42A5F5", alpha=0.85)
ax_dir.bar(x + w/2, short_vals, w, label="Short", color="#EF5350", alpha=0.85)
ax_dir.axhline(0, color="black", linewidth=0.8)
ax_dir.set_xticks(x)
ax_dir.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
ax_dir.set_title("PnL by Direction per Ticker", fontsize=11)
ax_dir.set_ylabel("Net PnL ($)")
ax_dir.legend(fontsize=9)
ax_dir.grid(alpha=0.3, axis="y")

# 3. Bucket contribution donut ─────────────────────────────────────────────
pos_bkt = bucket_stats[bucket_stats["total_pnl"] > 0]
wedge_colors = [BUCKET_COLORS[b] for b in pos_bkt["asset_bucket"]]
wedges, texts, autotexts = ax_bkt.pie(
    pos_bkt["total_pnl"], labels=pos_bkt["asset_bucket"],
    colors=wedge_colors, autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.5), textprops={"fontsize": 10},
)
ax_bkt.set_title("Bucket Share of Total PnL", fontsize=11)
# add bucket totals in center
total = bucket_stats["total_pnl"].sum()
ax_bkt.text(0, 0, f"${total:,.0f}", ha="center", va="center",
            fontsize=12, fontweight="bold")

# add win rate annotation next to each bucket label
for bucket_row in bucket_stats.itertuples():
    pass  # already shown in bar chart; keep donut clean

# 4. Cumulative PnL by bucket ──────────────────────────────────────────────
for bkt, series in bucket_cum.items():
    ax_cum.plot(series.values, label=bkt, color=BUCKET_COLORS[bkt], linewidth=1.8)
ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_cum.set_title("Cumulative PnL by Bucket (trade sequence order)", fontsize=11)
ax_cum.set_xlabel("Trade #")
ax_cum.set_ylabel("Cumulative Net PnL ($)")
ax_cum.legend(fontsize=9)
ax_cum.grid(alpha=0.3)

fig.savefig(OUT_DIR / "asset_returns_x3.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_DIR / 'asset_returns_x3.png'}")
plt.close(fig)

# ── print summary table ────────────────────────────────────────────────────
print(f"\n{'Ticker':<12} {'Bucket':<8} {'Trades':>7} {'Total PnL':>10} {'Win%':>6} {'Avg Win':>9} {'Avg Loss':>9}")
print("-" * 68)
for _, r in ticker_stats.iterrows():
    flag = " ** NEGATIVE **" if r["total_pnl"] < 0 else ""
    print(f"{r['ticker']:<12} {r['asset_bucket']:<8} {r['trades']:>7} "
          f"${r['total_pnl']:>9,.0f} {r['win_rate']:>5.1f}% "
          f"${r['avg_win']:>8,.0f} ${r['avg_loss']:>8,.0f}{flag}")

print(f"\n{'Bucket':<10} {'Trades':>7} {'Total PnL':>10} {'Win%':>6}")
print("-" * 36)
for _, r in bucket_stats.iterrows():
    print(f"{r['asset_bucket']:<10} {r['trades']:>7} ${r['total_pnl']:>9,.0f} {r['win_rate']:>5.1f}%")
