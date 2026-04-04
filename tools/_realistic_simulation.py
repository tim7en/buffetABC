"""
Realistic Strategy Simulation
==============================
Re-simulates backtest results under more conservative assumptions:
1. Fixed notional (no compounding / reinvestment)
2. Higher slippage/costs
3. Funding cost for leverage
4. Position-size cap
5. Excludes Q1 2026 outlier period

Outputs a realistic_simulation.json for the investor report.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

TRADES_CSV = Path(r"d:\buffetABC\reports\pruned_grouped_backtest_20260404\trades.csv")
SUMMARY_JSON = Path(r"d:\buffetABC\reports\pruned_grouped_backtest_20260404\summary.json")
OUTPUT_JSON = Path(r"d:\buffetABC\reports\pruned_grouped_backtest_20260404\realistic_simulation.json")

df = pd.read_csv(TRADES_CSV, parse_dates=["entry_ts", "exit_ts"])
df = df.sort_values("exit_ts").reset_index(drop=True)
with open(SUMMARY_JSON) as f:
    summary = json.load(f)["summary"]

years = (df["exit_ts"].max() - df["entry_ts"].min()).days / 365.25
initial_capital = summary["initial_capital"]

print("=" * 80)
print("  REALISTIC STRATEGY SIMULATIONS")
print("=" * 80)

results = {}

# ─────────────────────────────────────────────────────────────────────
# Scenario 1: Baseline (as-is from backtest)
# ─────────────────────────────────────────────────────────────────────
baseline_pnl = df["net_pnl"].sum()
baseline_return = baseline_pnl / initial_capital * 100
baseline_cagr = ((initial_capital + baseline_pnl) / initial_capital) ** (1 / years) - 1

results["baseline"] = {
    "label": "Backtest (as-is, 3x, compounding)",
    "total_pnl": round(baseline_pnl, 2),
    "total_return_pct": round(baseline_return, 1),
    "cagr_pct": round(baseline_cagr * 100, 1),
    "final_equity": round(initial_capital + baseline_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 2: Fixed notional — scale every trade's PnL to $3,000
# notional (as if we started with $1,000 at 3x and never grew)
# ─────────────────────────────────────────────────────────────────────
fixed_notional = initial_capital * 3  # $3,000
fixed_pnl = 0
for _, trade in df.iterrows():
    actual_notional = float(trade["notional"])
    pnl = float(trade["net_pnl"])
    if actual_notional > 0:
        fixed_pnl += pnl * (fixed_notional / actual_notional)
    else:
        fixed_pnl += pnl

fixed_return = fixed_pnl / initial_capital * 100
fixed_cagr = ((initial_capital + fixed_pnl) / initial_capital) ** (1 / years) - 1

results["fixed_notional"] = {
    "label": "Fixed $3,000 notional (no compounding)",
    "total_pnl": round(fixed_pnl, 2),
    "total_return_pct": round(fixed_return, 1),
    "cagr_pct": round(fixed_cagr * 100, 1),
    "final_equity": round(initial_capital + fixed_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 3: Fixed notional at 1x (no leverage, no compounding)
# ─────────────────────────────────────────────────────────────────────
fixed_1x_notional = initial_capital  # $1,000
fixed_1x_pnl = 0
for _, trade in df.iterrows():
    actual_notional = float(trade["notional"])
    pnl = float(trade["net_pnl"])
    if actual_notional > 0:
        fixed_1x_pnl += pnl * (fixed_1x_notional / actual_notional)
    else:
        fixed_1x_pnl += pnl

fixed_1x_return = fixed_1x_pnl / initial_capital * 100
fixed_1x_cagr = ((initial_capital + fixed_1x_pnl) / initial_capital) ** (1 / years) - 1

results["fixed_1x"] = {
    "label": "Fixed $1,000 notional (1x, no compounding)",
    "total_pnl": round(fixed_1x_pnl, 2),
    "total_return_pct": round(fixed_1x_return, 1),
    "cagr_pct": round(fixed_1x_cagr * 100, 1),
    "final_equity": round(initial_capital + fixed_1x_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 4: Add funding cost (8% annualized on leveraged portion)
# Each trade's notional is borrowed at 8% p.a., charged pro-rata
# ─────────────────────────────────────────────────────────────────────
FUNDING_RATE = 0.08  # 8% annual
funding_pnl = 0
total_funding_cost = 0
for _, trade in df.iterrows():
    pnl = float(trade["net_pnl"])
    notional = float(trade["notional"])
    holding_hours = (trade["exit_ts"] - trade["entry_ts"]).total_seconds() / 3600
    # Funding cost = notional * (2/3 borrowed at 3x) * rate * time
    borrowed_fraction = max(0, 1 - initial_capital / notional) if notional > 0 else 0
    borrowed_amount = notional * borrowed_fraction
    funding_cost = borrowed_amount * FUNDING_RATE * (holding_hours / 8766)
    funding_pnl += pnl - funding_cost
    total_funding_cost += funding_cost

funding_return = funding_pnl / initial_capital * 100
funding_cagr = ((initial_capital + funding_pnl) / initial_capital) ** (1 / years) - 1

results["with_funding"] = {
    "label": "Backtest + 8% funding cost on borrowed capital",
    "total_pnl": round(funding_pnl, 2),
    "total_return_pct": round(funding_return, 1),
    "cagr_pct": round(funding_cagr * 100, 1),
    "total_funding_cost": round(total_funding_cost, 2),
    "final_equity": round(initial_capital + funding_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 5: Higher slippage (5 bps instead of 2 bps per side)
# Add 3 bps extra each way = 6 bps additional round-trip cost
# ─────────────────────────────────────────────────────────────────────
EXTRA_COST_BPS = 6  # 3 bps extra per side × 2
extra_cost_pnl = 0
total_extra_cost = 0
for _, trade in df.iterrows():
    pnl = float(trade["net_pnl"])
    notional = float(trade["notional"])
    extra = notional * EXTRA_COST_BPS / 10000
    extra_cost_pnl += pnl - extra
    total_extra_cost += extra

extra_return = extra_cost_pnl / initial_capital * 100
extra_cagr = ((initial_capital + extra_cost_pnl) / initial_capital) ** (1 / years) - 1

results["higher_costs"] = {
    "label": "Backtest + 3 bps extra slippage per side",
    "total_pnl": round(extra_cost_pnl, 2),
    "total_return_pct": round(extra_return, 1),
    "cagr_pct": round(extra_cagr * 100, 1),
    "total_extra_cost": round(total_extra_cost, 2),
    "final_equity": round(initial_capital + extra_cost_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 6: "Conservative realistic" — fixed notional + funding + extra cost
# ─────────────────────────────────────────────────────────────────────
conservative_pnl = 0
for _, trade in df.iterrows():
    actual_notional = float(trade["notional"])
    pnl = float(trade["net_pnl"])
    holding_hours = (trade["exit_ts"] - trade["entry_ts"]).total_seconds() / 3600
    if actual_notional > 0:
        # Scale to fixed notional
        scale = fixed_notional / actual_notional
        adj_pnl = pnl * scale
        adj_notional = fixed_notional
    else:
        adj_pnl = pnl
        adj_notional = 0
    # Add funding
    borrowed_amount = adj_notional * 2 / 3  # 2/3 borrowed at 3x
    funding = borrowed_amount * FUNDING_RATE * (holding_hours / 8766)
    # Add extra slippage
    extra = adj_notional * EXTRA_COST_BPS / 10000
    conservative_pnl += adj_pnl - funding - extra

conservative_return = conservative_pnl / initial_capital * 100
conservative_cagr = ((initial_capital + conservative_pnl) / initial_capital) ** (1 / years) - 1

results["conservative"] = {
    "label": "CONSERVATIVE: fixed notional + funding + extra slippage",
    "total_pnl": round(conservative_pnl, 2),
    "total_return_pct": round(conservative_return, 1),
    "cagr_pct": round(conservative_cagr * 100, 1),
    "final_equity": round(initial_capital + conservative_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# Scenario 7: Exclude 2026 (in-sample only through 2025)
# ─────────────────────────────────────────────────────────────────────
df_pre2026 = df[df["exit_ts"].dt.year < 2026]
years_pre2026 = (df_pre2026["exit_ts"].max() - df_pre2026["entry_ts"].min()).days / 365.25
pre2026_pnl = df_pre2026["net_pnl"].sum()
pre2026_return = pre2026_pnl / initial_capital * 100
pre2026_cagr = ((initial_capital + pre2026_pnl) / initial_capital) ** (1 / max(years_pre2026, 0.1)) - 1

results["exclude_2026"] = {
    "label": "Exclude Q1 2026 (4-year only)",
    "total_pnl": round(pre2026_pnl, 2),
    "total_return_pct": round(pre2026_return, 1),
    "cagr_pct": round(pre2026_cagr * 100, 1),
    "trades": len(df_pre2026),
    "final_equity": round(initial_capital + pre2026_pnl, 2),
}

# ─────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'Scenario':<55} {'Return':>10} {'CAGR':>8} {'Final $':>12}")
print("-" * 90)
for key, r in results.items():
    print(f"  {r['label']:<53} {r['total_return_pct']:>8.1f}% {r['cagr_pct']:>6.1f}% ${r['final_equity']:>10,.2f}")

# ─────────────────────────────────────────────────────────────────────
# WINNER CONCENTRATION TABLE
# ─────────────────────────────────────────────────────────────────────
print(f"\n--- WINNER DEPENDENCY ANALYSIS ---")
wins = df[df["net_pnl"] > 0].sort_values("net_pnl", ascending=False)
for n in [1, 3, 5, 10, 20]:
    top_pnl = wins.head(n)["net_pnl"].sum()
    remaining = baseline_pnl - top_pnl
    remaining_pf = (df["net_pnl"].sum() - top_pnl + abs(df[df["net_pnl"] <= 0]["net_pnl"].sum())) / abs(df[df["net_pnl"] <= 0]["net_pnl"].sum())
    print(f"  Remove top {n:2d} wins: PnL ${remaining:>10,.2f} "
          f"({remaining/initial_capital*100:>7.1f}% return)")

# ─────────────────────────────────────────────────────────────────────
# YEARLY PROFIT FACTOR STABILITY
# ─────────────────────────────────────────────────────────────────────
print(f"\n--- YEARLY EDGE CONSISTENCY (fixed-notional basis) ---")
for year in sorted(df["exit_ts"].dt.year.unique()):
    ydf = df[df["exit_ts"].dt.year == year]
    fixed_year_pnl = 0
    for _, trade in ydf.iterrows():
        notional = float(trade["notional"])
        pnl = float(trade["net_pnl"])
        if notional > 0:
            fixed_year_pnl += pnl * (fixed_notional / notional)
    year_wins = [pnl * (fixed_notional / n) for pnl, n in
                 zip(ydf[ydf["net_pnl"] > 0]["net_pnl"], ydf[ydf["net_pnl"] > 0]["notional"]) if n > 0]
    year_losses = [pnl * (fixed_notional / n) for pnl, n in
                   zip(ydf[ydf["net_pnl"] <= 0]["net_pnl"], ydf[ydf["net_pnl"] <= 0]["notional"]) if n > 0]
    gross_win = sum(year_wins)
    gross_loss = abs(sum(year_losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(ydf[ydf["net_pnl"] > 0]) / len(ydf) * 100 if len(ydf) > 0 else 0
    print(f"  {year}: {len(ydf):>3d} trades, fixed PnL ${fixed_year_pnl:>8,.2f}, "
          f"PF {pf:>5.2f}, WR {win_rate:>5.1f}%")

# Save results
output_data = {
    "scenarios": results,
    "analysis": {
        "top_10_win_concentration_pct": round(wins.head(10)["net_pnl"].sum() / df[df["net_pnl"] > 0]["net_pnl"].sum() * 100, 1),
        "late_quarter_pnl_pct": round(df[df["exit_ts"] >= df["exit_ts"].quantile(0.75)]["net_pnl"].sum() / baseline_pnl * 100, 1),
        "total_years": round(years, 2),
        "total_trades": len(df),
        "free_parameters": 26,
        "trades_per_parameter": round(len(df) / 26, 1),
    }
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {OUTPUT_JSON}")
