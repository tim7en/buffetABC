"""
Asset selection analysis — characterise why winners won and losers lost,
then derive quantitative rules for future asset inclusion/exclusion.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
from pathlib import Path

OUT_DIR = Path("reports/capacity_test_core_x2/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "reports/capacity_test_core_x2/trades_high_exp_risk_5_exp_3x.csv",
    parse_dates=["entry_ts", "exit_ts"],
)
df["net_pnl"]    = pd.to_numeric(df["net_pnl"],    errors="coerce")
df["notional"]   = pd.to_numeric(df["notional"],   errors="coerce")
df["hold_days"]  = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 86400
df["year"]       = df["entry_ts"].dt.year

BUCKET_COLORS = {"equity": "#1565C0", "crypto": "#6A1B9A",
                 "gold": "#F9A825",   "metals": "#558B2F"}

# ── ticker-level stats (all directions combined) ──────────────────────────
def ticker_stats(grp):
    wins  = grp[grp > 0]
    losses = grp[grp < 0]
    return pd.Series({
        "trades":       len(grp),
        "total_pnl":    grp.sum(),
        "win_rate":     (grp > 0).mean() * 100,
        "avg_win":      wins.mean()         if len(wins)   else 0.0,
        "avg_loss":     losses.mean()       if len(losses) else 0.0,   # negative
        "payoff_ratio": (wins.mean() / abs(losses.mean()))
                        if len(wins) and len(losses) else (999.0 if len(wins) else 0.0),
        "expectancy":   grp.mean(),
        "pnl_per_day":  grp.sum() / df.loc[grp.index, "hold_days"].sum()
                        if df.loc[grp.index, "hold_days"].sum() > 0 else 0.0,
        "years_positive": (df.loc[grp.index].groupby(df.loc[grp.index,"year"])["net_pnl"]
                           .sum() > 0).sum(),
        "years_traded":   df.loc[grp.index, "year"].nunique(),
        "max_single_loss": losses.min() if len(losses) else 0.0,
    })

ts = (
    df.groupby(["ticker", "asset_bucket"])["net_pnl"]
    .apply(ticker_stats)
    .unstack()
    .reset_index()
)
ts["years_positive_pct"] = ts["years_positive"] / ts["years_traded"] * 100

# ── year-by-year heatmap data ─────────────────────────────────────────────
yr_pivot = df.groupby(["ticker", "year"])["net_pnl"].sum().unstack(fill_value=0)
# ensure all 5 years present
for y in [2022, 2023, 2024, 2025, 2026]:
    if y not in yr_pivot.columns:
        yr_pivot[y] = 0.0
yr_pivot = yr_pivot[[2022, 2023, 2024, 2025, 2026]]
# sort by total
yr_pivot["total"] = yr_pivot.sum(axis=1)
yr_pivot = yr_pivot.sort_values("total", ascending=False)
yr_pivot = yr_pivot.drop(columns="total")

# ── selection scorecard ───────────────────────────────────────────────────
# Rules:
#   PASS  payoff_ratio >= 2.0
#   PASS  expectancy   >= 20   (avg $ per trade)
#   PASS  pnl_per_day  >= 1.0  ($ per day held)
#   PASS  years_positive_pct >= 50%
#   PASS  win_rate     >= 25%  (trend-following minimum)
ts["r_payoff"]   = ts["payoff_ratio"]   >= 2.0
ts["r_expect"]   = ts["expectancy"]     >= 20.0
ts["r_pnlday"]   = ts["pnl_per_day"]    >= 1.0
ts["r_yr_pos"]   = ts["years_positive_pct"] >= 50.0
ts["r_winrate"]  = ts["win_rate"]       >= 25.0
rule_cols = ["r_payoff", "r_expect", "r_pnlday", "r_yr_pos", "r_winrate"]
ts["rules_passed"] = ts[rule_cols].sum(axis=1)
ts["verdict"] = ts["rules_passed"].map(
    lambda n: "KEEP" if n >= 4 else ("MARGINAL" if n == 3 else "DROP")
)

# ─────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.suptitle("Asset Selection Analysis — Session Turtle Core x3",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

ax_heat  = fig.add_subplot(gs[0, :])   # year heatmap
ax_pay   = fig.add_subplot(gs[1, 0])   # payoff ratio scatter
ax_score = fig.add_subplot(gs[1, 1])   # scorecard bars
ax_cum   = fig.add_subplot(gs[2, :])   # cumulative PnL per ticker

# 1. Year-by-year heatmap ──────────────────────────────────────────────────
data   = yr_pivot.values
tickers_ord = yr_pivot.index.tolist()
vmax   = np.percentile(np.abs(data[data != 0]), 95) if (data != 0).any() else 1
norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax_heat.imshow(data.T, aspect="auto", cmap="RdYlGn", norm=norm)
ax_heat.set_xticks(range(len(tickers_ord)))
ax_heat.set_xticklabels(tickers_ord, rotation=45, ha="right", fontsize=10)
ax_heat.set_yticks(range(5))
ax_heat.set_yticklabels([2022, 2023, 2024, 2025, 2026])
ax_heat.set_title("Annual PnL per Ticker (green=profit, red=loss)", fontsize=11)
for i, ticker in enumerate(tickers_ord):
    for j, yr in enumerate([2022, 2023, 2024, 2025, 2026]):
        v = yr_pivot.loc[ticker, yr]
        if v != 0:
            ax_heat.text(i, j, f"${v:,.0f}", ha="center", va="center",
                         fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")
plt.colorbar(im, ax=ax_heat, label="Net PnL ($)", fraction=0.02, pad=0.02)

# 2. Payoff ratio vs win rate scatter ──────────────────────────────────────
for _, r in ts.iterrows():
    color  = BUCKET_COLORS.get(r["asset_bucket"], "grey")
    marker = "o" if r["verdict"] == "KEEP" else ("^" if r["verdict"] == "MARGINAL" else "x")
    ms     = 120 if r["verdict"] == "KEEP" else 80
    ax_pay.scatter(r["win_rate"], r["payoff_ratio"], color=color,
                   marker=marker, s=ms, zorder=3,
                   edgecolors="black" if r["verdict"] == "KEEP" else "none", linewidths=0.8)
    ax_pay.annotate(r["ticker"], (r["win_rate"], r["payoff_ratio"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
ax_pay.axhline(2.0,  color="orange", linewidth=1.0, linestyle="--", label="Payoff threshold (2.0)")
ax_pay.axvline(25.0, color="steelblue", linewidth=1.0, linestyle="--", label="Win rate threshold (25%)")
ax_pay.set_xlabel("Win Rate %")
ax_pay.set_ylabel("Payoff Ratio (avg win / avg loss)")
ax_pay.set_title("Win Rate vs Payoff Ratio\n(o=KEEP  ^=MARGINAL  x=DROP)", fontsize=11)
ax_pay.set_ylim(0, min(ts["payoff_ratio"].replace(999, 0).max() * 1.2, 20))
ax_pay.legend(fontsize=8)
ax_pay.grid(alpha=0.3)

# 3. Scorecard ─────────────────────────────────────────────────────────────
sc_order = ts.sort_values("rules_passed", ascending=True)
colors_sc = ["#43A047" if v == "KEEP" else ("#FFA726" if v == "MARGINAL" else "#E53935")
             for v in sc_order["verdict"]]
bars = ax_score.barh(range(len(sc_order)), sc_order["rules_passed"],
                     color=colors_sc, alpha=0.85)
ax_score.set_yticks(range(len(sc_order)))
ax_score.set_yticklabels(
    [f"{r['ticker']} ({r['verdict']})" for _, r in sc_order.iterrows()], fontsize=9)
ax_score.set_xlim(0, 5.5)
ax_score.axvline(4, color="green",  linewidth=1.2, linestyle="--", label="KEEP threshold")
ax_score.axvline(3, color="orange", linewidth=1.0, linestyle=":",  label="MARGINAL threshold")
ax_score.set_xlabel("Rules Passed (out of 5)")
ax_score.set_title("Asset Selection Scorecard\n(payoff≥2 | expect≥$20 | $/day≥$1 | 50%+yrs positive | WR≥25%)",
                   fontsize=10)
ax_score.legend(fontsize=8)
ax_score.grid(alpha=0.3, axis="x")
for bar, v in zip(bars, sc_order["rules_passed"]):
    ax_score.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                  str(int(v)), va="center", fontsize=9)

# 4. Cumulative PnL per ticker ─────────────────────────────────────────────
df_s = df.sort_values("exit_ts")
for _, r in ts.sort_values("total_pnl", ascending=False).iterrows():
    sub   = df_s[df_s["ticker"] == r["ticker"]]["net_pnl"].cumsum().reset_index(drop=True)
    color = BUCKET_COLORS.get(r["asset_bucket"], "grey")
    ls    = "-" if r["verdict"] == "KEEP" else ("--" if r["verdict"] == "MARGINAL" else ":")
    lw    = 1.8 if r["verdict"] == "KEEP" else 1.2
    ax_cum.plot(sub.values, label=f"{r['ticker']} ({r['verdict']})",
                color=color, linestyle=ls, linewidth=lw)
ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_cum.set_title("Cumulative PnL per Ticker (solid=KEEP, dashed=MARGINAL, dotted=DROP)", fontsize=11)
ax_cum.set_xlabel("Trade sequence")
ax_cum.set_ylabel("Cumulative Net PnL ($)")
ax_cum.legend(fontsize=7, ncol=3)
ax_cum.grid(alpha=0.3)

fig.savefig(OUT_DIR / "asset_selection_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_DIR / 'asset_selection_analysis.png'}")
plt.close(fig)

# ── print scorecard table ─────────────────────────────────────────────────
print(f"\n{'Ticker':<12} {'Bucket':<8} {'Verdict':<10} {'Pass':>5} "
      f"{'Payoff':>7} {'Expect':>8} {'$/day':>7} {'Yr+%':>7} {'WR%':>6} {'Total PnL':>10}")
print("-" * 90)
for _, r in ts.sort_values("total_pnl", ascending=False).iterrows():
    flags = "".join(
        ("Y" if r[c] else "N")
        for c in rule_cols
    )
    print(f"{r['ticker']:<12} {r['asset_bucket']:<8} {r['verdict']:<10} "
          f"{int(r['rules_passed']):>3}/5 [{flags}] "
          f"{min(r['payoff_ratio'],99):>6.2f}x "
          f"${r['expectancy']:>7,.0f} "
          f"${r['pnl_per_day']:>6,.1f} "
          f"{r['years_positive_pct']:>6.0f}% "
          f"{r['win_rate']:>5.1f}% "
          f"${r['total_pnl']:>9,.0f}")

print("\nFlags: [payoff>=2 | expect>=$20 | $/day>=$1 | 50%+yrs | WR>=25%]")
print("\nSummary:")
for verdict in ["KEEP", "MARGINAL", "DROP"]:
    names = ts[ts["verdict"] == verdict]["ticker"].tolist()
    print(f"  {verdict:<10}: {', '.join(names)}")
