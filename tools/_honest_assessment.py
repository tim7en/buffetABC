"""Quick analysis of compounding, concentration, and leverage effects."""
import pandas as pd
import numpy as np
import json

df = pd.read_csv(r"d:\buffetABC\reports\pruned_grouped_backtest_20260404\trades.csv",
                 parse_dates=["entry_ts", "exit_ts"])
with open(r"d:\buffetABC\reports\pruned_grouped_backtest_20260404\summary.json") as f:
    summary = json.load(f)["summary"]

print("=" * 80)
print("  HONEST STRATEGY ASSESSMENT")
print("=" * 80)

years = (df["exit_ts"].max() - df["entry_ts"].min()).days / 365.25
total_pnl = df["net_pnl"].sum()
print(f"\n  Period: {df['entry_ts'].min().date()} -> {df['exit_ts'].max().date()} ({years:.2f} years)")
print(f"  Trades: {len(df)}")

# 1. COMPOUNDING EFFECT
print(f"\n--- COMPOUNDING ANALYSIS ---")
print(f"  Sum of all trade PnL (what you'd get if position sizes didn't grow): ${total_pnl:,.2f}")
print(f"  Simple return (no compounding): {total_pnl/1000*100:.1f}%")
print(f"  Compounded return (backtest): {summary['total_return_pct']:.1f}%")
print(f"  COMPOUNDING MULTIPLIER: {summary['total_return_pct'] / (total_pnl/1000*100):.2f}x")

# 2. LEVERAGE EFFECT - simulate at 1x
equity_1x = 1000.0
equity_3x = 1000.0
for _, trade in df.iterrows():
    notional = float(trade["notional"])
    scale = float(trade["scale"])
    pnl = float(trade["net_pnl"])
    if scale > 0 and notional > 0:
        # The trade's PnL at 1x would be PnL * (1/3) approximately
        # More precisely: the trade was sized at 3x, so at 1x it's 1/3
        pnl_1x = pnl / 3.0
        equity_1x += pnl_1x
    equity_3x += pnl

print(f"\n--- LEVERAGE EFFECT ---")
print(f"  At 1x leverage: ${equity_1x:,.2f} ({(equity_1x/1000-1)*100:.1f}%)")
print(f"  At 3x leverage: ${equity_3x:,.2f} ({(equity_3x/1000-1)*100:.1f}%)")
print(f"  Leverage amplification: {(equity_3x/1000-1)/(max(equity_1x/1000-1, 0.01)):.1f}x")

# 3. WINNER CONCENTRATION
wins = df[df["net_pnl"] > 0].copy()
losses = df[df["net_pnl"] <= 0].copy()
top5_pnl = wins.nlargest(5, "net_pnl")["net_pnl"].sum()
top10_pnl = wins.nlargest(10, "net_pnl")["net_pnl"].sum()
total_profit = wins["net_pnl"].sum()

print(f"\n--- WINNER CONCENTRATION ---")
print(f"  Avg win: ${wins['net_pnl'].mean():.2f}, Avg loss: ${losses['net_pnl'].mean():.2f}")
print(f"  Win/Loss ratio: {abs(wins['net_pnl'].mean() / losses['net_pnl'].mean()):.2f}")
print(f"  Largest single win: ${wins['net_pnl'].max():,.2f}")
print(f"  Largest single loss: ${losses['net_pnl'].min():,.2f}")
print(f"  Top 5 wins: ${top5_pnl:,.2f} ({top5_pnl/total_profit*100:.1f}% of all profit)")
print(f"  Top 10 wins: ${top10_pnl:,.2f} ({top10_pnl/total_profit*100:.1f}% of all profit)")
print(f"  Total gross profit: ${total_profit:,.2f}")
print(f"  Total gross loss: ${losses['net_pnl'].sum():,.2f}")

# 4. TEMPORAL CONCENTRATION - are returns front/back loaded?
print(f"\n--- TEMPORAL DISTRIBUTION ---")
df["exit_year"] = df["exit_ts"].dt.year
for year in sorted(df["exit_year"].unique()):
    ydf = df[df["exit_year"] == year]
    ypnl = ydf["net_pnl"].sum()
    avg_notional = ydf["notional"].mean()
    print(f"  {year}: {len(ydf)} trades, PnL ${ypnl:,.2f}, "
          f"avg notional ${avg_notional:,.2f}, "
          f"PF {ydf[ydf['net_pnl']>0]['net_pnl'].sum()/max(abs(ydf[ydf['net_pnl']<=0]['net_pnl'].sum()),1):.2f}")

# 5. EXPOSURE ANALYSIS
print(f"\n--- EXPOSURE AT ENTRY ---")
for mult in sorted(df["entry_exposure_mult"].unique()):
    subset = df[df["entry_exposure_mult"] == mult]
    print(f"  {mult}x: {len(subset)} trades, PnL ${subset['net_pnl'].sum():,.2f}, "
          f"avg PnL ${subset['net_pnl'].mean():.2f}")

# 6. LATE-PERIOD MEGA-TRADE ANALYSIS
print(f"\n--- LATE-PERIOD BIG TRADES (last 25% of timeline) ---")
cutoff = df["exit_ts"].quantile(0.75)
late = df[df["exit_ts"] >= cutoff]
early = df[df["exit_ts"] < cutoff]
print(f"  Early (75%): {len(early)} trades, PnL ${early['net_pnl'].sum():,.2f}, avg notional ${early['notional'].mean():,.2f}")
print(f"  Late (25%):  {len(late)} trades, PnL ${late['net_pnl'].sum():,.2f}, avg notional ${late['notional'].mean():,.2f}")
print(f"  Late trades generate {late['net_pnl'].sum()/total_pnl*100:.1f}% of total PnL")

# 7. WHAT IF WE CAP POSITION SIZE?
print(f"\n--- POSITION SIZE CAPPED AT $5,000 MAX NOTIONAL ---")
capped_pnl = 0
for _, trade in df.iterrows():
    notional = float(trade["notional"])
    pnl = float(trade["net_pnl"])
    if notional > 5000:
        capped_pnl += pnl * (5000 / notional)
    else:
        capped_pnl += pnl
print(f"  Capped PnL: ${capped_pnl:,.2f} (return: {capped_pnl/1000*100:.1f}%)")
print(f"  vs Actual PnL: ${total_pnl:,.2f} (return: {total_pnl/1000*100:.1f}%)")

# 8. DEGREES OF FREEDOM / OVERFITTING RISK
n_params = 0
params = [
    "channel period selection (2 groups)", "exit channel period (2)",
    "exposure_mult", "DD trigger 1", "DD trigger 2", "DD mult 1", "DD mult 2",
    "base_risk_pct", "fixed_stop_pct", "directional_volume_risk_pct",
    "conviction_max_mult", "trend_fast", "trend_slow",
    "vix_on_thresh", "vix_off_thresh", "fg_greed_thresh", "fg_fear_thresh",
    "ema200_above_long", "ema200_above_short", "ema200_below_long", "ema200_below_short",
    "entry_window_minutes", "lookback_years",
    "vix_long_risk_off_mult", "vix_short_risk_on_mult",
    "universe selection (21 of 27)",
]
print(f"\n--- OVERFITTING RISK ---")
print(f"  Estimated free parameters: {len(params)}")
print(f"  Executed trades: {len(df)}")
print(f"  Trades per parameter: {len(df)/len(params):.1f}")
print(f"  Rule of thumb: need 20+ trades per parameter (you have {len(df)/len(params):.0f})")
print(f"  Unique assets trading: {df['ticker'].nunique()}")

# Key params list
for i, p in enumerate(params, 1):
    print(f"    {i:2d}. {p}")
