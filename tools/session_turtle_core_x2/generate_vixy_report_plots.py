"""
VIXY Impact Report — Session Turtle Core x2
Generates a comprehensive set of plots showing how VIXY integration
affected strategy performance.
"""

import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_DIR  = ROOT / "reports" / "vixy_micro_backtest_core_x2"
WF_DIR      = ROOT / "reports" / "vixy_micro_walkforward_core_x2_bias_reduced"
VIX_PARQUET = ROOT / "cache" / "cache" / "tiingo" / "VIX_5m.parquet"
OUT_DIR     = ROOT / "reports" / "vixy_impact_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline":           "#4c72b0",
    "VIXY conservative":  "#55a868",
    "VIXY asymmetric":    "#dd8452",
    "VIXY strict filter": "#c44e52",
}
VARIANT_ORDER = ["Baseline", "VIXY conservative", "VIXY asymmetric", "VIXY strict filter"]

# ── load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(REPORT_DIR / "vixy_micro_comparison_summary.csv")
summary["variant_label"] = summary["variant_label"].str.strip()

trades_files = {
    "Baseline":           "trades_baseline.csv",
    "VIXY conservative":  "trades_vixy_conservative.csv",
    "VIXY asymmetric":    "trades_vixy_asymmetric.csv",
    "VIXY strict filter": "trades_vixy_strict_filter.csv",
}
trades = {k: pd.read_csv(REPORT_DIR / v) for k, v in trades_files.items()}

wf_all      = pd.read_csv(WF_DIR / "walkforward_variant_results.csv")
wf_selected = pd.read_csv(WF_DIR / "walkforward_selected_results.csv")

# load VIX 5m parquet
vix_raw = pq.read_table(VIX_PARQUET).to_pandas()
vix_raw.columns = [col.lower() for col in vix_raw.columns]
# the datetime column is 'time'; the close column is 'c'
vix_raw["ts"] = pd.to_datetime(vix_raw["time"], utc=True)
vix_raw = vix_raw.sort_values("ts").reset_index(drop=True)

# restrict to backtest window Feb 2022 – Mar 2026
start_ts = pd.Timestamp("2022-02-09", tz="UTC")
end_ts   = pd.Timestamp("2026-03-03", tz="UTC")
vix = vix_raw[(vix_raw["ts"] >= start_ts) & (vix_raw["ts"] <= end_ts)].copy()

# close column is 'c'
vix["close"] = vix["c"].astype(float)

# compute SMAs for regime overlay
vix["sma78"]  = vix["close"].rolling(78,  min_periods=1).mean()
vix["sma390"] = vix["close"].rolling(390, min_periods=1).mean()

def regime(row):
    c, s, l = row["close"], row["sma78"], row["sma390"]
    if c <= s <= l:
        return "risk_on_micro"
    elif c > s > l:
        return "risk_off_micro"
    return "neutral_micro"

vix["regime"] = vix.apply(regime, axis=1)

# daily resample for price-action plot
vix_daily = vix.set_index("ts")["close"].resample("1D").last().dropna().reset_index()

# ============================================================
# helper
# ============================================================
def _bar_chart(ax, variants, values, title, ylabel, fmt=".1f", highlight="VIXY asymmetric"):
    bars = ax.bar(variants, values, color=[COLORS[v] for v in variants], width=0.5, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    baseline_val = values[0]
    for bar, var, val in zip(bars, variants, values):
        label = f"{val:{fmt}}"
        if var != "Baseline":
            delta = val - baseline_val
            sign = "+" if delta >= 0 else ""
            label += f"\n({sign}{delta:{fmt}})"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                label, ha="center", va="bottom", fontsize=7.5)
    if highlight in variants:
        idx = variants.index(highlight)
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2)
    return bars


# ============================================================
# FIG 1 – Performance Overview (6 metrics)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("VIXY Micro-Regime Overlay — Performance Overview\n(Session Turtle Core x2 · Feb 2022 – Mar 2026)",
             fontsize=13, fontweight="bold", y=1.01)

s = summary.set_index("variant_label").loc[VARIANT_ORDER]

metrics = [
    ("total_return_pct",          "Total Return (%)",      ".1f"),
    ("cagr_pct",                  "CAGR (%)",              ".2f"),
    ("max_realized_drawdown_pct", "Max Drawdown (%)",      ".2f"),
    ("profit_factor",             "Profit Factor",         ".2f"),
    ("win_rate_pct",              "Win Rate (%)",          ".1f"),
    ("executed_trades",           "Executed Trades",       ".0f"),
]

for ax, (col, label, fmt) in zip(axes.flat, metrics):
    vals = [float(s.loc[v, col]) for v in VARIANT_ORDER]
    _bar_chart(ax, VARIANT_ORDER, vals, label, label, fmt=fmt)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_performance_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_performance_overview.png")


# ============================================================
# FIG 2 – Trade Breakdown (long / short / win / loss)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("VIXY Overlay — Trade Breakdown by Variant",
             fontsize=13, fontweight="bold")

# 2a stacked bar: long vs short
ax = axes[0]
longs  = [float(s.loc[v, "long_trades"])  for v in VARIANT_ORDER]
shorts = [float(s.loc[v, "short_trades"]) for v in VARIANT_ORDER]
x = np.arange(len(VARIANT_ORDER))
bars1 = ax.bar(x, longs,  0.45, label="Long",  color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.9)
bars2 = ax.bar(x, shorts, 0.45, label="Short", color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.45,
               hatch="///")
ax.set_xticks(x); ax.set_xticklabels(VARIANT_ORDER, fontsize=8)
ax.set_title("Long vs Short Trades", fontweight="bold")
ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)

# 2b stacked bar: win vs loss
ax = axes[1]
wins   = [float(s.loc[v, "winning_trades"]) for v in VARIANT_ORDER]
losses = [float(s.loc[v, "losing_trades"])  for v in VARIANT_ORDER]
ax.bar(x, wins,   0.45, label="Winning", color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.9)
ax.bar(x, losses, 0.45, label="Losing",  color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.45,
       hatch="xxx")
ax.set_xticks(x); ax.set_xticklabels(VARIANT_ORDER, fontsize=8)
ax.set_title("Winning vs Losing Trades", fontweight="bold")
ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)

# 2c skipped trades
ax = axes[2]
sk_ticker   = [float(s.loc[v, "skipped_same_ticker"])  for v in VARIANT_ORDER]
sk_capacity = [float(s.loc[v, "skipped_no_capacity"])  for v in VARIANT_ORDER]
ax.bar(x, sk_ticker,   0.45, label="Same ticker", color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.9)
ax.bar(x, sk_capacity, 0.45, label="No capacity", color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.45,
       hatch="...")
ax.set_xticks(x); ax.set_xticklabels(VARIANT_ORDER, fontsize=8)
ax.set_title("Skipped Trades (Capacity / Ticker)", fontweight="bold")
ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_trade_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_trade_breakdown.png")


# ============================================================
# FIG 3 – VIXY Regime Distribution & Overlay Stats
# ============================================================
# The overlay touched 20 trades; regime distribution from the summary cols
risk_on  = float(s.loc["VIXY asymmetric", "entries_intraday_volatility_risk_on_micro"])
neutral  = float(s.loc["VIXY asymmetric", "entries_intraday_volatility_neutral_micro"])
risk_off = float(s.loc["VIXY asymmetric", "entries_intraday_volatility_risk_off_micro"])
scaled   = float(s.loc["VIXY asymmetric", "entries_intraday_volatility_proxy_scaled"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("VIXY Micro-Regime — Overlay Statistics",
             fontsize=13, fontweight="bold")

# 3a regime pie
ax = axes[0]
sizes  = [risk_on, neutral, risk_off]
labels = ["risk_on_micro", "neutral_micro", "risk_off_micro"]
clrs   = ["#2ecc71", "#f39c12", "#e74c3c"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=clrs,
                                   autopct="%1.1f%%", startangle=90,
                                   textprops={"fontsize": 9})
ax.set_title("Regime Distribution at Entry\n(VIXY asymmetric)", fontweight="bold")

# 3b average sizing mult across variants
ax = axes[1]
avg_mults = [float(s.loc[v, "avg_intraday_volatility_proxy_mult"]) for v in VARIANT_ORDER]
bars = ax.bar(VARIANT_ORDER, avg_mults, color=[COLORS[v] for v in VARIANT_ORDER], width=0.5)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="No scaling (1.0)")
ax.set_ylim(0.88, 1.05)
ax.set_title("Avg VIXY Size Multiplier", fontweight="bold")
ax.set_ylabel("Multiplier"); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, avg_mults):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax.legend(fontsize=8)

# 3c trades affected vs total
ax = axes[2]
x2 = np.arange(len(VARIANT_ORDER))
totals   = [float(s.loc[v, "executed_trades"]) for v in VARIANT_ORDER]
affected = [float(s.loc[v, "entries_intraday_volatility_proxy_scaled"]) for v in VARIANT_ORDER]
ax.bar(x2, totals,   0.45, label="Total executed",  color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.5)
ax.bar(x2, affected, 0.45, label="VIXY-scaled",     color=[COLORS[v] for v in VARIANT_ORDER], alpha=0.95)
ax.set_xticks(x2); ax.set_xticklabels(VARIANT_ORDER, fontsize=8)
ax.set_title("VIXY-Scaled vs Total Trades", fontweight="bold")
ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_regime_stats.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_regime_stats.png")


# ============================================================
# FIG 4 – Asset Class PnL Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Asset Class PnL — Baseline vs VIXY Variants",
             fontsize=13, fontweight="bold")

buckets = ["crypto", "gold", "equity", "metals"]
pnl_cols = [f"{b}_pnl" for b in buckets]
bucket_colors = ["#9b59b6", "#f1c40f", "#3498db", "#e67e22"]

# 4a grouped bars
ax = axes[0]
x3 = np.arange(len(buckets))
width = 0.18
for i, var in enumerate(VARIANT_ORDER):
    vals = [float(s.loc[var, c]) for c in pnl_cols]
    ax.bar(x3 + i * width - 1.5 * width, vals, width,
           label=var, color=COLORS[var], alpha=0.85)
ax.set_xticks(x3); ax.set_xticklabels(buckets, fontsize=9)
ax.set_title("PnL by Asset Class", fontweight="bold")
ax.set_ylabel("PnL ($)"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.5)
ax.axhline(0, color="black", linewidth=0.8)

# 4b delta from baseline
ax = axes[1]
baseline_pnl = [float(s.loc["Baseline", c]) for c in pnl_cols]
for i, var in enumerate(VARIANT_ORDER[1:], 1):
    deltas = [float(s.loc[var, c]) - b for c, b in zip(pnl_cols, baseline_pnl)]
    ax.bar(x3 + (i-1) * width - width, deltas, width,
           label=var, color=COLORS[var], alpha=0.85)
ax.set_xticks(x3); ax.set_xticklabels(buckets, fontsize=9)
ax.set_title("PnL Delta vs Baseline", fontweight="bold")
ax.set_ylabel("ΔPnL ($)"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.5)
ax.axhline(0, color="black", linewidth=1.2)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_asset_class_pnl.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_asset_class_pnl.png")


# ============================================================
# FIG 5 – VIX/VIXY 5-minute Price Action with Regime Overlay
# ============================================================
# Use a 3-month sample for readability + a full-period daily view

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.8], hspace=0.35)

# ── top: daily VIX close (full period)
ax_daily = fig.add_subplot(gs[0])
ax_daily.plot(vix_daily["ts"], vix_daily["close"], color="#2c3e50", linewidth=1, label="VIX close (daily)")
ax_daily.axhline(20, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7, label="VIX = 20 (stress threshold)")
ax_daily.fill_between(vix_daily["ts"], vix_daily["close"], alpha=0.12, color="#2c3e50")
ax_daily.set_title("VIX Daily Close — Full Backtest Period (Feb 2022 – Mar 2026)",
                   fontweight="bold", fontsize=10)
ax_daily.set_ylabel("VIX Close")
ax_daily.grid(alpha=0.3)
ax_daily.legend(fontsize=8)

# highlight years
for year in [2022, 2023, 2024, 2025, 2026]:
    ax_daily.axvline(pd.Timestamp(f"{year}-01-01", tz="UTC"), color="grey",
                     linewidth=0.5, linestyle=":", alpha=0.6)

# ── bottom: 5m with regime colour bands — pick a 3-month window of interest
# choose a period with mixed regimes: Aug-Oct 2022 (volatile)
sample_start = pd.Timestamp("2022-08-01", tz="UTC")
sample_end   = pd.Timestamp("2022-11-01", tz="UTC")
vix_sample = vix[(vix["ts"] >= sample_start) & (vix["ts"] <= sample_end)].copy()

ax_5m = fig.add_subplot(gs[1])
# background colour bands per regime
regime_colors_bg = {
    "risk_on_micro":  "#d5f5e3",
    "neutral_micro":  "#fef9e7",
    "risk_off_micro": "#fadbd8",
}
cur_regime = None
band_start = None
for _, row in vix_sample.iterrows():
    r = row["regime"]
    if r != cur_regime:
        if cur_regime is not None:
            ax_5m.axvspan(band_start, row["ts"], alpha=0.35,
                          color=regime_colors_bg[cur_regime], linewidth=0)
        cur_regime  = r
        band_start  = row["ts"]
if cur_regime and band_start:
    ax_5m.axvspan(band_start, vix_sample["ts"].iloc[-1], alpha=0.35,
                  color=regime_colors_bg[cur_regime], linewidth=0)

ax_5m.plot(vix_sample["ts"], vix_sample["close"],  color="#2c3e50", linewidth=0.7,  label="VIX 5m close", zorder=3)
ax_5m.plot(vix_sample["ts"], vix_sample["sma78"],  color="#e74c3c", linewidth=1.2,  label="SMA-78 (short)", zorder=4)
ax_5m.plot(vix_sample["ts"], vix_sample["sma390"], color="#3498db", linewidth=1.2,  label="SMA-390 (long)",  zorder=4)

# legend patches
patch_on  = mpatches.Patch(color="#d5f5e3", alpha=0.8, label="risk_on_micro  (VIXY ≤ SMA78 ≤ SMA390)")
patch_neu = mpatches.Patch(color="#fef9e7", alpha=0.8, label="neutral_micro")
patch_off = mpatches.Patch(color="#fadbd8", alpha=0.8, label="risk_off_micro (VIXY > SMA78 > SMA390)")
handles, labels_ = ax_5m.get_legend_handles_labels()
ax_5m.legend(handles=handles + [patch_on, patch_neu, patch_off], fontsize=7.5, ncol=2)

ax_5m.set_title("VIX 5-Minute Bars with VIXY Micro-Regime Overlay\n(Aug – Oct 2022 sample)",
                fontweight="bold", fontsize=10)
ax_5m.set_ylabel("VIX 5m Close")
ax_5m.grid(alpha=0.25)

plt.savefig(OUT_DIR / "fig5_vixy_price_action.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_vixy_price_action.png")


# ============================================================
# FIG 6 – Regime fraction over time (monthly rolling)
# ============================================================
vix["ym"] = vix["ts"].dt.to_period("M")
regime_monthly = (
    vix.groupby(["ym", "regime"])
       .size()
       .unstack(fill_value=0)
       .apply(lambda row: row / row.sum(), axis=1)
)
# ensure all three cols exist
for col in ["risk_on_micro", "neutral_micro", "risk_off_micro"]:
    if col not in regime_monthly.columns:
        regime_monthly[col] = 0.0

regime_monthly.index = regime_monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(regime_monthly.index,
             regime_monthly["risk_on_micro"],
             regime_monthly["neutral_micro"],
             regime_monthly["risk_off_micro"],
             labels=["risk_on_micro", "neutral_micro", "risk_off_micro"],
             colors=["#2ecc71", "#f39c12", "#e74c3c"],
             alpha=0.75)
ax.set_title("Monthly VIXY Micro-Regime Composition\n(fraction of 5m bars per month)",
             fontweight="bold", fontsize=11)
ax.set_ylabel("Fraction of Bars")
ax.set_ylim(0, 1)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(12))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig6_regime_fraction_over_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6_regime_fraction_over_time.png")


# ============================================================
# FIG 7 – Walk-Forward Out-of-Sample Results
# ============================================================
wf_test = wf_all[wf_all["split"] == "test"].copy()
wf_test["variant_label"] = wf_test["variant_label"].str.strip()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Walk-Forward Out-of-Sample — VIXY Variant Performance",
             fontsize=13, fontweight="bold")

for fold_id, ax in zip([1, 2], [axes[0], axes[1]]):
    fold_data = wf_test[wf_test["fold"] == fold_id]
    fold_data  = fold_data.set_index("variant_label").loc[
        [v for v in VARIANT_ORDER if v in fold_data["variant_label"].values]]
    vals = fold_data["total_return_pct"].tolist()
    vars_ = fold_data.index.tolist()
    bars = ax.bar(vars_, vals, color=[COLORS[v] for v in vars_], width=0.5)
    test_range = f"{fold_data.iloc[0]['window_start']} → {fold_data.iloc[0]['window_end']}"
    ax.set_title(f"Fold {fold_id} – Test Period\n{test_range}", fontweight="bold", fontsize=9)
    ax.set_ylabel("Total Return (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    # mark selected
    sel_row = wf_selected[wf_selected["fold"] == fold_id].iloc[0]
    sel_var = sel_row["selected_variant"].strip()
    if sel_var in vars_:
        idx = vars_.index(sel_var)
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2.5)
        ax.text(bars[idx].get_x() + bars[idx].get_width()/2, -5,
                "▲ selected", ha="center", fontsize=7, color="black")

# 7c: delta table plot
ax = axes[2]
ax.axis("off")
fold1_sel = wf_selected[wf_selected["fold"] == 1].iloc[0]
fold2_sel = wf_selected[wf_selected["fold"] == 2].iloc[0]

table_data = [
    ["Fold", "Selected Variant", "OOS Return", "Baseline OOS", "Delta"],
    ["1", fold1_sel["selected_variant"],
     f"{fold1_sel['selected_test_return_pct']:.1f}%",
     f"{fold1_sel['baseline_test_return_pct']:.1f}%",
     f"{fold1_sel['test_return_delta_pct']:+.2f}%"],
    ["2", fold2_sel["selected_variant"],
     f"{fold2_sel['selected_test_return_pct']:.1f}%",
     f"{fold2_sel['baseline_test_return_pct']:.1f}%",
     f"{fold2_sel['test_return_delta_pct']:+.2f}%"],
]

tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center",
               bbox=[0, 0.2, 1, 0.6])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")

ax.set_title("Walk-Forward Summary", fontweight="bold", fontsize=10, pad=30)
ax.text(0.5, 0.05,
        "Selected = best in-sample variant. Delta = OOS vs Baseline OOS.",
        ha="center", va="bottom", transform=ax.transAxes, fontsize=7.5,
        color="grey")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_walkforward_oos.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig7_walkforward_oos.png")


# ============================================================
# FIG 8 – Individual Trade PnL: Baseline vs VIXY asymmetric
# (sorted by entry date, shows where sizing changed)
# ============================================================
bl = trades["Baseline"][["ticker", "direction", "entry_ts", "net_pnl", "intraday_vol_proxy_regime"]].copy()
va = trades["VIXY asymmetric"][["ticker", "direction", "entry_ts", "net_pnl",
                                  "intraday_vol_proxy_regime", "intraday_vol_proxy_mult"]].copy()

bl["entry_ts"] = pd.to_datetime(bl["entry_ts"])
va["entry_ts"] = pd.to_datetime(va["entry_ts"])
bl = bl.sort_values("entry_ts").reset_index(drop=True)
va = va.sort_values("entry_ts").reset_index(drop=True)

# merge on entry_ts + ticker + direction (outer to see extra trades in VIXY)
merged = pd.merge(bl, va, on=["entry_ts", "ticker", "direction"],
                  how="outer", suffixes=("_bl", "_va"))
merged = merged.sort_values("entry_ts").reset_index(drop=True)
merged["pnl_delta"] = merged["net_pnl_va"].fillna(0) - merged["net_pnl_bl"].fillna(0)
merged["cumsum_bl"] = merged["net_pnl_bl"].fillna(0).cumsum()
merged["cumsum_va"] = merged["net_pnl_va"].fillna(0).cumsum()

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle("Trade-Level Comparison — Baseline vs VIXY Asymmetric",
             fontsize=13, fontweight="bold")

# 8a cumulative PnL
ax = axes[0]
ax.plot(merged.index, merged["cumsum_bl"], color=COLORS["Baseline"],       linewidth=1.8, label="Baseline")
ax.plot(merged.index, merged["cumsum_va"], color=COLORS["VIXY asymmetric"], linewidth=1.8, label="VIXY asymmetric")
ax.fill_between(merged.index, merged["cumsum_bl"], merged["cumsum_va"],
                where=merged["cumsum_va"] >= merged["cumsum_bl"],
                alpha=0.2, color=COLORS["VIXY asymmetric"], label="VIXY ahead")
ax.fill_between(merged.index, merged["cumsum_bl"], merged["cumsum_va"],
                where=merged["cumsum_va"] < merged["cumsum_bl"],
                alpha=0.2, color=COLORS["Baseline"], label="Baseline ahead")
ax.set_title("Cumulative PnL ($) — Trade Sequence", fontweight="bold", fontsize=9)
ax.set_ylabel("Cumulative PnL ($)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 8b per-trade PnL delta
ax = axes[1]
clr_delta = ["#2ecc71" if d >= 0 else "#e74c3c" for d in merged["pnl_delta"]]
ax.bar(merged.index, merged["pnl_delta"], color=clr_delta, width=0.8, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Per-Trade PnL Delta (VIXY asymmetric − Baseline)", fontweight="bold", fontsize=9)
ax.set_ylabel("ΔPnL ($)"); ax.grid(axis="y", alpha=0.3)

# 8c per-trade PnL scatter coloured by regime
ax = axes[2]
regime_point_colors = {
    "risk_on_micro":  "#2ecc71",
    "risk_off_micro": "#e74c3c",
    "neutral_micro":  "#f39c12",
    float("nan"):     "#bdc3c7",
}
for reg in ["risk_on_micro", "neutral_micro", "risk_off_micro"]:
    mask = va["intraday_vol_proxy_regime"].fillna("none") == reg
    sub = va[mask].reset_index(drop=True)
    ax.scatter(sub.index, sub["net_pnl"], color=regime_point_colors[reg],
               alpha=0.7, s=25, label=reg, zorder=3)
no_regime_mask = va["intraday_vol_proxy_regime"].isna()
sub_nr = va[no_regime_mask].reset_index(drop=True)
ax.scatter(sub_nr.index, sub_nr["net_pnl"], color="#bdc3c7", alpha=0.5, s=20, label="no VIXY signal")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("VIXY Asymmetric — Trade PnL by Entry Regime", fontweight="bold", fontsize=9)
ax.set_ylabel("Net PnL ($)"); ax.set_xlabel("Trade #"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig8_trade_pnl_detail.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig8_trade_pnl_detail.png")


# ============================================================
# FIG 9 – Summary Dashboard
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#f8f9fa")
gs9 = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle("VIXY Micro-Regime Overlay — Summary Dashboard\nSession Turtle Core x2 · Feb 2022 – Mar 2026",
             fontsize=14, fontweight="bold", y=1.01)

# top row: 4 KPI boxes
kpi_metrics = [
    ("Total Return",  "total_return_pct",          "{:.1f}%"),
    ("CAGR",          "cagr_pct",                  "{:.2f}%"),
    ("Max Drawdown",  "max_realized_drawdown_pct",  "{:.2f}%"),
    ("Profit Factor", "profit_factor",              "{:.2f}"),
]
for i, (name, col, fmt) in enumerate(kpi_metrics):
    ax_kpi = fig.add_subplot(gs9[0, i])
    ax_kpi.set_facecolor("white")
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")
    ax_kpi.text(0.5, 0.85, name, ha="center", va="top", fontsize=9, color="#7f8c8d",
                transform=ax_kpi.transAxes)
    for j, var in enumerate(VARIANT_ORDER):
        val = float(s.loc[var, col])
        color = COLORS[var]
        ax_kpi.text(0.5, 0.62 - j * 0.14, fmt.format(val),
                    ha="center", va="top", fontsize=10.5, color=color, fontweight="bold",
                    transform=ax_kpi.transAxes)
        ax_kpi.text(0.05, 0.62 - j * 0.14, var, ha="left", va="top",
                    fontsize=6.5, color=color, transform=ax_kpi.transAxes)
    rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                                    linewidth=1.5, edgecolor="#bdc3c7", facecolor="none",
                                    transform=ax_kpi.transAxes)
    ax_kpi.add_patch(rect)

# middle-left: return bar
ax_ret = fig.add_subplot(gs9[1, :2])
vals_ret = [float(s.loc[v, "total_return_pct"]) for v in VARIANT_ORDER]
_bar_chart(ax_ret, VARIANT_ORDER, vals_ret, "Total Return (%)", "Return (%)", fmt=".1f")

# middle-right: profit factor bar
ax_pf = fig.add_subplot(gs9[1, 2:])
vals_pf = [float(s.loc[v, "profit_factor"]) for v in VARIANT_ORDER]
_bar_chart(ax_pf, VARIANT_ORDER, vals_pf, "Profit Factor", "PF", fmt=".2f")

# bottom row: regime pie + walk-forward
ax_pie = fig.add_subplot(gs9[2, :2])
sizes  = [risk_on, neutral, risk_off]
labels_pie = [f"risk_on_micro\n{int(risk_on)} bars", f"neutral_micro\n{int(neutral)} bars", f"risk_off_micro\n{int(risk_off)} bars"]
ax_pie.pie(sizes, labels=labels_pie, colors=["#2ecc71","#f39c12","#e74c3c"],
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
ax_pie.set_title("VIXY Regime at Entry\n(VIXY asymmetric)", fontweight="bold", fontsize=9)

ax_wf = fig.add_subplot(gs9[2, 2:])
ax_wf.axis("off")
wf_rows = [
    ["Fold", "Period", "Selected", "OOS Return", "Baseline", "Delta"],
    ["1", "Jul–Dec 2024", "VIXY asymm.", "51.1%", "60.1%", "−9.0%"],
    ["2", "Sep 2025–Feb 2026", "Baseline", "66.8%", "66.8%", "0.0%"],
]
tbl2 = ax_wf.table(cellText=wf_rows[1:], colLabels=wf_rows[0],
                    cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.85])
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")
ax_wf.set_title("Walk-Forward OOS Summary", fontweight="bold", fontsize=9, pad=10)

plt.savefig(OUT_DIR / "fig9_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig9_dashboard.png")

print(f"\nAll plots saved to: {OUT_DIR}")
print("Files:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
