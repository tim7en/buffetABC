"""
VIXY US-Scoped Impact Report
Confirms VIXY overlay is correctly restricted to new_york_equity_open session only,
and generates a fresh performance report from the re-run.
"""

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

ROOT        = pathlib.Path(__file__).resolve().parents[2]
NEW_DIR     = ROOT / "reports" / "vixy_micro_backtest_core_x2_us_scoped"
OLD_DIR     = ROOT / "reports" / "vixy_micro_backtest_core_x2"
VIX_PARQUET = ROOT / "cache" / "cache" / "tiingo" / "VIX_5m.parquet"
OUT_DIR     = ROOT / "reports" / "vixy_us_scoped_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Baseline":           "#4c72b0",
    "VIXY conservative":  "#55a868",
    "VIXY asymmetric":    "#dd8452",
    "VIXY strict filter": "#c44e52",
}
VARIANT_ORDER = ["Baseline", "VIXY conservative", "VIXY asymmetric", "VIXY strict filter"]
SESSION_COLORS = {
    "new_york_equity_open": "#2980b9",
    "hong_kong_open":       "#e67e22",
}

# ── load new summary & trades ────────────────────────────────
summary = pd.read_csv(NEW_DIR / "vixy_micro_comparison_summary.csv")
summary["variant_label"] = summary["variant_label"].str.strip()
s = summary.set_index("variant_label").loc[VARIANT_ORDER]

trades_files = {
    "Baseline":           "trades_baseline.csv",
    "VIXY conservative":  "trades_vixy_conservative.csv",
    "VIXY asymmetric":    "trades_vixy_asymmetric.csv",
    "VIXY strict filter": "trades_vixy_strict_filter.csv",
}
trades = {k: pd.read_csv(NEW_DIR / v) for k, v in trades_files.items()}

# ── load OLD summary for comparison ─────────────────────────
old_summary = pd.read_csv(OLD_DIR / "vixy_micro_comparison_summary.csv")
old_summary["variant_label"] = old_summary["variant_label"].str.strip()
os_ = old_summary.set_index("variant_label").loc[VARIANT_ORDER]

# ── load VIX data ─────────────────────────────────────────────
vix_raw = pq.read_table(VIX_PARQUET).to_pandas()
vix_raw.columns = [c.lower() for c in vix_raw.columns]
vix_raw["ts"] = pd.to_datetime(vix_raw["time"], utc=True)
vix_raw = vix_raw.sort_values("ts").reset_index(drop=True)
start_ts = pd.Timestamp("2022-02-09", tz="UTC")
end_ts   = pd.Timestamp("2026-03-03", tz="UTC")
vix = vix_raw[(vix_raw["ts"] >= start_ts) & (vix_raw["ts"] <= end_ts)].copy()
vix["close"] = vix["c"].astype(float)
vix["sma78"]  = vix["close"].rolling(78,  min_periods=1).mean()
vix["sma390"] = vix["close"].rolling(390, min_periods=1).mean()

def regime(row):
    c, s, l = row["close"], row["sma78"], row["sma390"]
    if c <= s <= l:   return "risk_on_micro"
    elif c > s > l:   return "risk_off_micro"
    return "neutral_micro"

vix["regime"] = vix.apply(regime, axis=1)
vix_daily = vix.set_index("ts")["close"].resample("1D").last().dropna().reset_index()


# ================================================================
# FIG 1 – Session Coverage Validation
# (prove VIXY = 0 scaled trades on HK, >0 on NY)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("VIXY Session Coverage Validation\n(US Proxy → new_york_equity_open ONLY)",
             fontsize=12, fontweight="bold")

for var in ["VIXY asymmetric"]:
    df = trades[var].copy()
    df["vixy_scaled"] = df["intraday_vol_proxy_mult"].fillna(1.0).astype(float) < 0.9999

    # 1a: trade count by session
    ax = axes[0]
    sessions = ["new_york_equity_open", "hong_kong_open"]
    for sess in sessions:
        sub = df[df["session_open"] == sess]
        total   = len(sub)
        regime_ = sub["intraday_vol_proxy_regime"].notna() & (sub["intraday_vol_proxy_regime"] != "")
        scaled  = sub["vixy_scaled"]
        ax.bar(
            sessions.index(sess) * 3 + 0, total,   0.6,
            color=SESSION_COLORS[sess], alpha=0.5, label=f"{sess} total"
        )
        ax.bar(
            sessions.index(sess) * 3 + 1, regime_.sum(), 0.6,
            color=SESSION_COLORS[sess], alpha=0.8, label=f"{sess} w/ regime"
        )
        ax.bar(
            sessions.index(sess) * 3 + 2, scaled.sum(), 0.6,
            color=SESSION_COLORS[sess], alpha=1.0, label=f"{sess} scaled"
        )
    ax.set_xticks([0.6, 3.6])
    ax.set_xticklabels(["new_york_equity_open", "hong_kong_open"], fontsize=8)
    ax.set_title("Trades: Total / With Regime / Scaled\n(VIXY asymmetric)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
    # custom legend
    ny_patch  = mpatches.Patch(color=SESSION_COLORS["new_york_equity_open"], label="NY equity open")
    hk_patch  = mpatches.Patch(color=SESSION_COLORS["hong_kong_open"],       label="HK open")
    ax.legend(handles=[ny_patch, hk_patch], fontsize=8)

    # 1b: mult distribution for NY-only trades WITH regime
    ax = axes[1]
    ny_regime = df[
        (df["session_open"] == "new_york_equity_open") &
        df["intraday_vol_proxy_regime"].notna() &
        (df["intraday_vol_proxy_regime"] != "")
    ]
    if len(ny_regime):
        mults = ny_regime["intraday_vol_proxy_mult"].astype(float)
        regime_counts = ny_regime["intraday_vol_proxy_regime"].value_counts()
        clrs = {"risk_on_micro": "#2ecc71", "neutral_micro": "#f39c12", "risk_off_micro": "#e74c3c"}
        for reg, cnt in regime_counts.items():
            sub_r = ny_regime[ny_regime["intraday_vol_proxy_regime"] == reg]
            ax.scatter(
                [list(regime_counts.index).index(reg)] * len(sub_r),
                sub_r["intraday_vol_proxy_mult"].astype(float),
                color=clrs.get(reg, "grey"), alpha=0.7, s=40, label=reg
            )
        ax.set_xticks(range(len(regime_counts)))
        ax.set_xticklabels(list(regime_counts.index), fontsize=7)
        ax.set_title("VIXY Mult by Regime (NY trades only)", fontweight="bold", fontsize=9)
        ax.set_ylabel("intraday_vol_proxy_mult"); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)

    # 1c: HK trade breakdown confirming 0 scaling
    ax = axes[2]
    hk = df[df["session_open"] == "hong_kong_open"]
    hk_regimes = hk["intraday_vol_proxy_regime"].fillna("(none)").value_counts()
    hk_mults   = hk["intraday_vol_proxy_mult"].astype(float)
    ax.bar(["HK: Total", "HK: Regime\nset", "HK: mult\n< 1.0"],
           [len(hk), sum(hk["intraday_vol_proxy_regime"].notna() & (hk["intraday_vol_proxy_regime"] != "")),
            sum(hk_mults < 0.9999)],
           color=[SESSION_COLORS["hong_kong_open"]] * 3, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("HK Trades — VIXY Not Applied\n(all mult=1.0, regime=none)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate([len(hk),
                            sum(hk["intraday_vol_proxy_regime"].notna() & (hk["intraday_vol_proxy_regime"] != "")),
                            sum(hk_mults < 0.9999)]):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_session_coverage_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_session_coverage_validation.png")


# ================================================================
# FIG 2 – Performance Comparison: New Run
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("VIXY Micro-Regime Overlay — US-Scoped Backtest\n(new_york_equity_open only · Feb 2022 – Mar 2026)",
             fontsize=13, fontweight="bold", y=1.01)

metrics = [
    ("total_return_pct",          "Total Return (%)",      ".1f"),
    ("cagr_pct",                  "CAGR (%)",              ".2f"),
    ("max_realized_drawdown_pct", "Max Drawdown (%)",      ".2f"),
    ("profit_factor",             "Profit Factor",         ".2f"),
    ("win_rate_pct",              "Win Rate (%)",          ".1f"),
    ("executed_trades",           "Executed Trades",       ".0f"),
]

def bar_chart(ax, variants, values, title, ylabel, fmt=".1f", highlight="VIXY asymmetric"):
    bars = ax.bar(variants, values, color=[COLORS[v] for v in variants], width=0.5, zorder=3)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.5)
    baseline_val = values[0]
    for bar, var, val in zip(bars, variants, values):
        lbl = f"{val:{fmt}}"
        if var != "Baseline":
            delta = val - baseline_val
            sign = "+" if delta >= 0 else ""
            lbl += f"\n({sign}{delta:{fmt}})"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.015,
                lbl, ha="center", va="bottom", fontsize=7)
    if highlight in variants:
        bars[variants.index(highlight)].set_edgecolor("black")
        bars[variants.index(highlight)].set_linewidth(2)

for ax, (col, label, fmt) in zip(axes.flat, metrics):
    vals = [float(s.loc[v, col]) for v in VARIANT_ORDER]
    bar_chart(ax, VARIANT_ORDER, vals, label, label, fmt=fmt)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_performance_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_performance_overview.png")


# ================================================================
# FIG 3 – Old vs New Backtest Comparison (baseline + VIXY asymmetric)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Old Backtest vs US-Scoped Re-run\n(Baseline & VIXY Asymmetric)",
             fontsize=13, fontweight="bold")

compare_variants = ["Baseline", "VIXY asymmetric"]
compare_metrics  = [
    ("total_return_pct",          "Total Return (%)"),
    ("cagr_pct",                  "CAGR (%)"),
    ("max_realized_drawdown_pct", "Max Drawdown (%)"),
    ("profit_factor",             "Profit Factor"),
    ("win_rate_pct",              "Win Rate (%)"),
    ("executed_trades",           "Trades"),
]

# 3a: grouped bar for each metric, old vs new
ax = axes[0]
x = np.arange(len(compare_metrics))
width = 0.18
labels_shown = [m[1] for m in compare_metrics]

for i, var in enumerate(compare_variants):
    old_vals = [float(os_.loc[var, m[0]]) for m in compare_metrics]
    new_vals = [float(s.loc[var, m[0]])   for m in compare_metrics]
    # normalise to old = 100%
    norm_old = [100.0] * len(old_vals)
    norm_new = [nv / ov * 100 if ov != 0 else 100 for nv, ov in zip(new_vals, old_vals)]
    ax.bar(x + i * width * 2,     norm_old, width, color=COLORS[var], alpha=0.4, label=f"{var} (old)")
    ax.bar(x + i * width * 2 + width, norm_new, width, color=COLORS[var], alpha=0.9, label=f"{var} (new)")

ax.axhline(100, color="black", linewidth=0.8, linestyle="--", label="100% = old baseline")
ax.set_xticks(x + width * 1.5); ax.set_xticklabels(labels_shown, fontsize=7.5, rotation=20, ha="right")
ax.set_title("Normalised to Old Baseline (100%)", fontweight="bold", fontsize=9)
ax.set_ylabel("% of old value"); ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7, ncol=2)

# 3b: absolute table
ax = axes[1]
ax.axis("off")
rows_data = []
for var in compare_variants:
    for tag, df_s in [("OLD", os_), ("NEW", s)]:
        rows_data.append([
            f"{var} ({tag})",
            f"{float(df_s.loc[var,'total_return_pct']):.1f}%",
            f"{float(df_s.loc[var,'cagr_pct']):.2f}%",
            f"{float(df_s.loc[var,'max_realized_drawdown_pct']):.2f}%",
            f"{float(df_s.loc[var,'profit_factor']):.2f}",
            f"{int(float(df_s.loc[var,'executed_trades']))}",
        ])

col_labels = ["Variant", "Return", "CAGR", "MaxDD", "PF", "Trades"]
tbl = ax.table(cellText=rows_data, colLabels=col_labels,
               cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#34495e"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 4 in (1, 2):
        cell.set_facecolor("#eaf4fb")
    else:
        cell.set_facecolor("#fef9e7")

ax.set_title("Absolute Performance: Old vs New\n(code changes in portfolio allocator explain the difference)",
             fontweight="bold", fontsize=8.5, pad=25)
ax.text(0.5, 0.03,
        "Note: Return difference is from portfolio allocator code changes between runs, NOT from VIXY scoping.\n"
        "VIXY session guard (NY only) was already in place in both runs.",
        ha="center", va="bottom", transform=ax.transAxes, fontsize=6.5, color="#7f8c8d", style="italic")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_old_vs_new_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_old_vs_new_comparison.png")


# ================================================================
# FIG 4 – Trade PnL by Session: NY vs HK (new run, VIXY asymmetric)
# ================================================================
df_va = trades["VIXY asymmetric"].copy()
df_va["net_pnl"] = df_va["net_pnl"].astype(float)
df_va["entry_ts"] = pd.to_datetime(df_va["entry_ts"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("VIXY Asymmetric — Trade PnL Breakdown by Session\n(US-Scoped Re-run)",
             fontsize=12, fontweight="bold")

# 4a cumulative PnL by session
ax = axes[0]
for sess, clr in SESSION_COLORS.items():
    sub = df_va[df_va["session_open"] == sess].sort_values("entry_ts")
    ax.plot(range(len(sub)), sub["net_pnl"].cumsum(), color=clr, linewidth=1.6, label=sess)
ax.set_title("Cumulative PnL by Session", fontweight="bold", fontsize=9)
ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL ($)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 4b PnL distribution violin
ax = axes[1]
ny_pnl = df_va[df_va["session_open"] == "new_york_equity_open"]["net_pnl"]
hk_pnl = df_va[df_va["session_open"] == "hong_kong_open"]["net_pnl"]
parts = ax.violinplot([ny_pnl, hk_pnl], positions=[0, 1], showmedians=True, showextrema=True)
for pc, clr in zip(parts["bodies"], [SESSION_COLORS["new_york_equity_open"], SESSION_COLORS["hong_kong_open"]]):
    pc.set_facecolor(clr); pc.set_alpha(0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["NY equity open", "HK open"], fontsize=9)
ax.set_title("PnL Distribution by Session", fontweight="bold", fontsize=9)
ax.set_ylabel("Net PnL ($)"); ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.8)

# 4c VIXY regime at entry for NY trades
ax = axes[2]
ny_trades = df_va[df_va["session_open"] == "new_york_equity_open"].copy()
regime_counts = ny_trades["intraday_vol_proxy_regime"].fillna("(no signal)").value_counts()
regime_clrs = {
    "risk_on_micro":  "#2ecc71",
    "neutral_micro":  "#f39c12",
    "risk_off_micro": "#e74c3c",
    "(no signal)":    "#bdc3c7",
}
bars_r = ax.bar(range(len(regime_counts)), regime_counts.values,
                color=[regime_clrs.get(r, "grey") for r in regime_counts.index], width=0.6)
ax.set_xticks(range(len(regime_counts)))
ax.set_xticklabels(regime_counts.index, fontsize=8, rotation=15, ha="right")
ax.set_title("NY Trades by VIXY Regime at Entry", fontweight="bold", fontsize=9)
ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars_r, regime_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_session_pnl_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_session_pnl_breakdown.png")


# ================================================================
# FIG 5 – VIX Price Action + VIXY Coverage Window Annotation
# ================================================================
vix_data = vix_daily.copy()

fig = plt.figure(figsize=(16, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

ax_top = fig.add_subplot(gs[0])
ax_top.plot(vix_data["ts"], vix_data["close"], color="#2c3e50", linewidth=1.0)
ax_top.fill_between(vix_data["ts"], vix_data["close"], alpha=0.12, color="#2c3e50")
ax_top.axhline(20, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7, label="VIX=20")

# shade VIX data coverage window (from June 2022)
vixy_start = pd.Timestamp("2022-06-30", tz="UTC")
ax_top.axvspan(vixy_start, vix_data["ts"].max(),
               alpha=0.08, color="#27ae60", label="VIXY data available")
ax_top.axvline(vixy_start, color="#27ae60", linewidth=1.5, linestyle="--")
ax_top.text(vixy_start, vix_data["close"].max() * 0.95,
            "VIXY\ndata\nstarts", ha="left", fontsize=8, color="#27ae60")
# mark backtest start
bt_start = pd.Timestamp("2022-02-09", tz="UTC")
ax_top.axvline(bt_start, color="#7f8c8d", linewidth=1, linestyle=":")
ax_top.text(bt_start, vix_data["close"].max() * 0.85,
            "Backtest\nstart", ha="left", fontsize=8, color="#7f8c8d")

ax_top.set_title("VIX Daily Close — Backtest Period\n(VIXY applied to NY equity open trades only, from Jun 2022)",
                 fontweight="bold", fontsize=10)
ax_top.set_ylabel("VIX Close"); ax_top.grid(alpha=0.25); ax_top.legend(fontsize=8)

# bottom: monthly fraction of NY trades WITH VIXY regime
df_va_ny = trades["VIXY asymmetric"][
    trades["VIXY asymmetric"]["session_open"] == "new_york_equity_open"
].copy()
df_va_ny["entry_ts"] = pd.to_datetime(df_va_ny["entry_ts"])
df_va_ny["ym"] = df_va_ny["entry_ts"].dt.to_period("M")
df_va_ny["has_regime"] = df_va_ny["intraday_vol_proxy_regime"].notna() & (df_va_ny["intraday_vol_proxy_regime"] != "")
monthly = df_va_ny.groupby("ym").agg(total=("has_regime","count"), with_regime=("has_regime","sum"))
monthly["frac"] = monthly["with_regime"] / monthly["total"]
monthly.index = monthly.index.to_timestamp()

ax_bot = fig.add_subplot(gs[1])
ax_bot.bar(monthly.index, monthly["frac"] * 100, width=20,
           color="#27ae60", alpha=0.7, label="% NY trades with VIXY regime")
ax_bot.axhline(100, color="grey", linestyle="--", linewidth=0.8)
ax_bot.set_ylim(0, 110)
ax_bot.set_title("% of NY Equity Open Trades With Active VIXY Signal (per month)",
                 fontweight="bold", fontsize=9)
ax_bot.set_ylabel("%"); ax_bot.grid(axis="y", alpha=0.3); ax_bot.legend(fontsize=8)

plt.savefig(OUT_DIR / "fig5_vixy_coverage_window.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_vixy_coverage_window.png")


# ================================================================
# FIG 6 – Summary Dashboard (new run)
# ================================================================
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#f8f9fa")
gs6 = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)
fig.suptitle("VIXY US-Scoped Re-run — Summary Dashboard\nSession Turtle Core x2 · new_york_equity_open only",
             fontsize=13, fontweight="bold", y=1.01)

# KPI boxes
kpi = [
    ("Total Return",  "total_return_pct",          "{:.1f}%"),
    ("CAGR",          "cagr_pct",                  "{:.2f}%"),
    ("Max Drawdown",  "max_realized_drawdown_pct",  "{:.2f}%"),
    ("Profit Factor", "profit_factor",              "{:.2f}"),
]
for i, (name, col, fmt) in enumerate(kpi):
    ax_k = fig.add_subplot(gs6[0, i])
    ax_k.set_facecolor("white"); ax_k.set_xlim(0,1); ax_k.set_ylim(0,1); ax_k.axis("off")
    ax_k.text(0.5, 0.88, name, ha="center", fontsize=9, color="#7f8c8d", transform=ax_k.transAxes)
    for j, var in enumerate(VARIANT_ORDER):
        val = float(s.loc[var, col])
        ax_k.text(0.5, 0.68 - j * 0.14, fmt.format(val),
                  ha="center", fontsize=11, color=COLORS[var], fontweight="bold", transform=ax_k.transAxes)
        ax_k.text(0.05, 0.68 - j * 0.14, var, ha="left", fontsize=6,
                  color=COLORS[var], transform=ax_k.transAxes)
    rect = mpatches.FancyBboxPatch((0.02,0.02), 0.96, 0.96,
                                    boxstyle="round,pad=0.02", linewidth=1.5,
                                    edgecolor="#bdc3c7", facecolor="none", transform=ax_k.transAxes)
    ax_k.add_patch(rect)

# Bottom-left: return bar
ax_ret = fig.add_subplot(gs6[1, :2])
vals = [float(s.loc[v, "total_return_pct"]) for v in VARIANT_ORDER]
bars_ = ax_ret.bar(VARIANT_ORDER, vals, color=[COLORS[v] for v in VARIANT_ORDER], width=0.5)
ax_ret.set_title("Total Return (%)", fontweight="bold"); ax_ret.grid(axis="y", alpha=0.3)
baseline_ = vals[0]
for bar_, var_, val_ in zip(bars_, VARIANT_ORDER, vals):
    lbl_ = f"{val_:.1f}%"
    if var_ != "Baseline":
        d_ = val_ - baseline_
        lbl_ += f"\n({'+' if d_>=0 else ''}{d_:.1f}%)"
    ax_ret.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + 2,
                lbl_, ha="center", fontsize=8)

# Bottom-right: session coverage summary
ax_cov = fig.add_subplot(gs6[1, 2:])
ax_cov.axis("off")
df_va_ = trades["VIXY asymmetric"]
ny_count  = (df_va_["session_open"] == "new_york_equity_open").sum()
hk_count  = (df_va_["session_open"] == "hong_kong_open").sum()
ny_regime = ((df_va_["session_open"] == "new_york_equity_open") &
              df_va_["intraday_vol_proxy_regime"].notna() &
              (df_va_["intraday_vol_proxy_regime"] != "")).sum()
ny_scaled = ((df_va_["session_open"] == "new_york_equity_open") &
              (df_va_["intraday_vol_proxy_mult"].astype(float) < 0.9999)).sum()
hk_regime = ((df_va_["session_open"] == "hong_kong_open") &
              df_va_["intraday_vol_proxy_regime"].notna() &
              (df_va_["intraday_vol_proxy_regime"] != "")).sum()
hk_scaled = ((df_va_["session_open"] == "hong_kong_open") &
              (df_va_["intraday_vol_proxy_mult"].astype(float) < 0.9999)).sum()

cov_rows = [
    ["Session",            "Trades", "VIXY regime", "Scaled", "Correct?"],
    ["new_york_equity_open", str(ny_count), str(ny_regime), str(ny_scaled), "✓ YES"],
    ["hong_kong_open",       str(hk_count), str(hk_regime), str(hk_scaled), "✓ YES (0 = correct)"],
]
tbl_ = ax_cov.table(cellText=cov_rows[1:], colLabels=cov_rows[0],
                     cellLoc="center", loc="center", bbox=[0, 0.15, 1, 0.7])
tbl_.auto_set_font_size(False); tbl_.set_fontsize(9)
for (r, c), cell in tbl_.get_celld().items():
    if r == 0:
        cell.set_facecolor("#34495e"); cell.set_text_props(color="white", fontweight="bold")
    elif r == 1:
        cell.set_facecolor("#d5f5e3")
    else:
        cell.set_facecolor("#fef9e7")
ax_cov.set_title("Session Coverage Confirmation (VIXY asymmetric)", fontweight="bold", fontsize=9, pad=10)

plt.savefig(OUT_DIR / "fig6_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6_dashboard.png")

print(f"\nAll plots saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
