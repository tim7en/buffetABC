# Session Turtle Trend Strategy Guide

**Version:** Core x3 — Pruned & Grouped  
**Date:** April 4, 2026  
**Backtest Period:** Feb 2022 — Mar 2026 (4.1 years)

---

## 1. Strategy Overview

Session Turtle Trend is a **Donchian channel breakout** strategy operating on **5-minute bars**, aggregated into **session bars** (1 session = 1 trading day measured from session open to session open).

**Core mechanic:** When price breaks above/below the N-session high/low, enter a position in the breakout direction. Exit when price crosses back through the shorter exit channel, or when the stop loss is hit.

### Donchian Channel Periods

| Parameter | Meaning |
|-----------|---------|
| **Channel period 10** | Highest high / lowest low of the last **10 sessions** (~2 weeks) |
| **Channel period 20** | Highest high / lowest low of the last **20 sessions** (~4 weeks) |
| **Exit channel 5** | Opposite 5-session channel triggers trade exit |
| **Exit channel 10** | Opposite 10-session channel triggers trade exit |

**10/5** = "Enter on a 2-week breakout, exit on a 1-week reversal."  
**20/10** = "Enter on a 4-week breakout, exit on a 2-week reversal."

---

## 2. Session Definitions

| Session | UTC Time | Used By | Purpose |
|---------|----------|---------|---------|
| `hong_kong_open` | 01:30 UTC | Crypto only | Asian session anchor for 24/7 markets |
| `new_york_equity_open` | 09:30 ET | All assets | Primary session for all instruments |

- Crypto assets get **2 daily entry windows** (HK open + NY open)
- All other assets trade **NY open only**
- **Entry window:** First 480 minutes (8 hours) from session open — no new entries after that

---

## 3. Universe: 21 Symbols in 2 Groups

### Group A — Fast Breakout (Channel 10/5) — 13 symbols

*Session: `new_york_equity_open` only*

#### Commodity (6 symbols)
| Symbol | Engine Ticker | Source | Bucket | Notes |
|--------|--------------|--------|--------|-------|
| BRENT | BRENT | tiingo | energy | Proxy: BZ-USD → BRENT |
| NATGAS-USD | NATGAS-USD | tiingo | energy | |
| COPPER-USD | COPPER-USD | tiingo | metals | |
| XPD-USD | XPD-USD | tiingo | metals | Palladium |
| PPLT | PPLT | tiingo | metals | Proxy: XPT-USD → PPLT |
| SLV | SLV | tiingo | metals | Proxy: XAG-USD → SLV |

**Why 10/5:** Physical commodities trend sharply on short timeframes. 10-session breakout captures supply/demand shocks and macro-driven moves effectively.

#### High-Beta Equity (7 symbols)
| Symbol | Engine Ticker | Source | Bucket |
|--------|--------------|--------|--------|
| COIN | COIN | tiingo | equity |
| CRCL | CRCL | tiingo | equity |
| HOOD | HOOD | tiingo | equity |
| INTC | INTC | tiingo | equity |
| MSTR | MSTR | tiingo | equity |
| PLTR | PLTR | tiingo | equity |
| TSLA | TSLA | tiingo | equity |

**Why 10/5:** High-momentum equities with fat-tailed moves. Fast breakout captures earnings, news, and sector rotation-driven trends.

### Group B — Slow Breakout (Channel 20/10) — 8 symbols

#### Crypto (4 symbols)
*Sessions: `hong_kong_open` + `new_york_equity_open`*

| Symbol | Engine Ticker | Source | Bucket |
|--------|--------------|--------|--------|
| BTC-USD | BTC-USD | binance | crypto |
| ETH-USD | ETH-USD | binance | crypto |
| SOL-USD | SOL-USD | binance | crypto |
| PAXG-USD | PAXG-USD | binance | gold |

**Why 20/10:** 24/7 markets generate excessive noise. Wider channel filters false breakouts. At 10/5, crypto was *negative* (-5.27% return). At 20/10: +643% return, PF 2.02.

#### Mega-Cap Equity (4 symbols)
*Session: `new_york_equity_open` only*

| Symbol | Engine Ticker | Source | Bucket |
|--------|--------------|--------|--------|
| GOOGL | GOOGL | tiingo | equity |
| META | META | tiingo | equity |
| TSM | TSM | tiingo | equity |
| EWY | EWY | tiingo | equity |

**Why 20/10:** Lower-volatility, higher-liquidity names need wider windows to confirm trend. 20-session breakout avoids mean-reversion traps that plague mega-caps at shorter horizons.

### Removed (6 symbols — zero diversification loss)

| Symbol | PF | Return | Reason |
|--------|-----|--------|--------|
| AAPL | 0.03 | -$1,326 | 6.2% win rate; structurally mean-reverting for breakout |
| AMZN | 0.06 | -$398 | Persistent loser, no breakout edge |
| NVDA | 0.60 | -$1,017 | Negative, but unique AI exposure — monitor |
| SPY | 0.28 | -$126 | Redundant index (composite of held tickers) |
| QQQ | 0.81 | -$58 | Redundant index (composite of held tickers) |
| EWJ | 0.53 | -$219 | Genuine Japan diversifier but unprofitable |

---

## 4. Portfolio Parameters (x3 Production)

### Position Sizing & Risk
| Parameter | Value |
|-----------|-------|
| Exposure multiple | 3.0x notional |
| Base risk per trade | 5% of capital |
| Directional volume risk (upgraded) | 7% of capital |
| Fixed stop loss | 10% |
| Max position (portfolio cap) | 90% |

### Drawdown Governor
| Parameter | Value |
|-----------|-------|
| Trigger 1 | 15% drawdown → reduce exposure to 1.5x |
| Trigger 2 | 25% drawdown → reduce exposure to 0.5x |

### Trend Filter (4H timeframe)
| Parameter | Value |
|-----------|-------|
| Fast EMA | 55 periods |
| Slow EMA | 200 periods |
| Logic | Long only if fast > slow; short only if fast < slow |

### Conviction Boost
| Parameter | Value |
|-----------|-------|
| Max multiplier | 1.25x |
| Inputs | Relative volume, volume ratio, breakout penetration, close location |

### Extended Hours VIX/Fear-Greed Proxy (1-day lag)
| Parameter | Value |
|-----------|-------|
| VIX risk-on | < 15 |
| VIX risk-off | > 25 |
| Fear & Greed greed threshold | > 60 |
| Fear & Greed fear threshold | < 30 |
| Long risk-off mult | 0.5x |
| Short risk-on mult | ~0 (blocked) |

### Per-Asset Technical Overlay (Daily EMA-200, 1-day lag)
| Condition | Long Mult | Short Mult |
|-----------|-----------|------------|
| Price above EMA-200 | 1.0x | 0.25x |
| Price below EMA-200 | 0.25x | 1.0x |

### Asset Class Caps
| Bucket | Cap |
|--------|-----|
| Crypto | 1.0x |
| Gold | 1.0x |
| Metals | 1.0x |
| Energy | 1.0x |
| Equity | None (uncapped) |

### Costs (embedded in backtest)
| Cost | Value |
|------|-------|
| Slippage | 2.0 bps per side |
| Commission | 1.0 bps per side |

---

## 5. Backtest Results (Pruned Universe, 21 Symbols)

| Metric | Value |
|--------|-------|
| **Total Return** | +5,288% |
| **CAGR** | 162.3% |
| **Max Drawdown** | 21.1% |
| **Profit Factor** | 3.85 |
| **Win Rate** | 41.4% |
| **Total Trades** | 447 |
| **Initial Capital** | $1,000 |
| **Final Equity** | $53,879 |

### P&L by Asset Bucket
| Bucket | P&L |
|--------|-----|
| Equity | $14,362 |
| Energy | $10,526 |
| Crypto | $8,078 |
| Metals | $5,336 |
| Gold | $333 |

### Per-Asset Performance (sorted by P&L)

| Ticker | Group | Trades | WR% | PF | P&L | Long P&L | Short P&L |
|--------|-------|--------|-----|-----|------|----------|-----------|
| BRENT | COMMODITY | 37 | 37.8% | 8.97 | +$10,526 | +$11,101 | -$575 |
| META | MEGA_ETF | 17 | 76.5% | 65.60 | +$6,499 | +$1,551 | +$4,947 |
| HOOD | HIGH_BETA | 18 | 44.4% | 4.96 | +$4,835 | +$4,941 | -$105 |
| SOL-USD | CRYPTO | 26 | 38.5% | 5.40 | +$4,643 | +$605 | +$4,038 |
| SLV | COMMODITY | 21 | 52.4% | 12.36 | +$4,262 | +$4,279 | -$16 |
| EWY | MEGA_ETF | 15 | 66.7% | 16.09 | +$4,096 | +$4,080 | +$17 |
| COIN | HIGH_BETA | 15 | 40.0% | 4.46 | +$3,248 | -$505 | +$3,753 |
| INTC | HIGH_BETA | 17 | 29.4% | 2.74 | +$3,021 | +$2,229 | +$793 |
| BTC-USD | CRYPTO | 27 | 40.7% | 3.52 | +$3,018 | +$1,425 | +$1,593 |
| CRCL | HIGH_BETA | 2 | 50.0% | ∞ | +$2,040 | $0 | +$2,040 |
| XPD-USD | COMMODITY | 35 | 42.9% | 1.88 | +$1,268 | +$890 | +$378 |
| GOOGL | MEGA_ETF | 19 | 42.1% | 2.67 | +$1,158 | +$1,157 | +$1 |
| PPLT | COMMODITY | 21 | 38.1% | 2.86 | +$1,073 | +$1,289 | -$216 |
| TSLA | HIGH_BETA | 18 | 33.3% | 3.04 | +$1,070 | +$1,410 | -$340 |
| TSM | MEGA_ETF | 14 | 71.4% | 10.69 | +$1,031 | +$874 | +$157 |
| ETH-USD | CRYPTO | 31 | 22.6% | 1.20 | +$418 | -$1,574 | +$1,992 |
| PAXG-USD | CRYPTO | 22 | 36.4% | 1.83 | +$333 | +$329 | +$4 |
| PLTR | HIGH_BETA | 20 | 55.0% | 1.23 | +$251 | +$251 | $0 |
| NATGAS-USD | COMMODITY | 28 | 35.7% | 1.14 | +$223 | +$963 | -$740 |
| COPPER-USD | COMMODITY | 30 | 26.7% | 0.96 | -$30 | +$113 | -$143 |
| MSTR | HIGH_BETA | 14 | 35.7% | 0.89 | -$103 | -$428 | +$324 |

### Watchlist (marginal assets, kept for diversification)
- **COPPER-USD** (PF 0.96, -$30): Basically flat. Genuine industrial metals diversifier — no substitute in universe.
- **MSTR** (PF 0.89, -$103): Leveraged BTC proxy. Weakest link — first removal candidate if stays below PF 1.0.

---

## 6. Correlation & Stress Analysis

### Cross-Group Average Correlation (Full Period)

|  | CRYPTO | COMMODITY | MEGA_ETF | HIGH_BETA |
|--|--------|-----------|----------|-----------|
| **CRYPTO** | 0.438 | 0.175 | 0.205 | 0.278 |
| **COMMODITY** | 0.175 | 0.275 | 0.127 | 0.096 |
| **MEGA_ETF** | 0.205 | 0.127 | 0.369 | 0.280 |
| **HIGH_BETA** | 0.278 | 0.096 | 0.280 | 0.357 |

### Stress-Period Correlation (worst 176 days)

|  | CRYPTO | COMMODITY | MEGA_ETF | HIGH_BETA |
|--|--------|-----------|----------|-----------|
| **CRYPTO** | 0.342 | 0.043 | -0.094 | 0.086 |
| **COMMODITY** | 0.043 | 0.238 | -0.057 | -0.043 |
| **MEGA_ETF** | -0.094 | -0.057 | 0.106 | 0.156 |
| **HIGH_BETA** | 0.086 | -0.043 | 0.156 | 0.226 |

### Correlation Shift (Stress minus Normal)

All negative — groups **decorrelate** during stress, the opposite of what kills most portfolios:

|  | CRYPTO | COMMODITY | MEGA_ETF | HIGH_BETA |
|--|--------|-----------|----------|-----------|
| **CRYPTO** | -0.044 | -0.076 | **-0.259** | -0.117 |
| **COMMODITY** | -0.076 | -0.006 | **-0.145** | -0.080 |
| **MEGA_ETF** | **-0.259** | **-0.145** | -0.300 | -0.036 |
| **HIGH_BETA** | -0.117 | -0.080 | -0.036 | -0.058 |

### High-Correlation Pairs (|ρ| > 0.5)

| Pair | Correlation | Group |
|------|-------------|-------|
| BTC-USD ↔ ETH-USD | +0.836 | Intra-crypto |
| BTC-USD ↔ SOL-USD | +0.745 | Intra-crypto |
| ETH-USD ↔ SOL-USD | +0.743 | Intra-crypto |
| SLV ↔ PAXG-USD | +0.705 | Cross: commodity/crypto |
| PPLT ↔ SLV | +0.706 | Intra-commodity (metals) |
| XPD-USD ↔ PPLT | +0.688 | Intra-commodity (metals) |
| COIN ↔ HOOD | +0.659 | Intra-high-beta |
| COIN ↔ CRCL | +0.579 | Intra-high-beta |
| COIN ↔ MSTR | +0.577 | Intra-high-beta |
| COIN ↔ BTC-USD | +0.569 | Cross: high-beta/crypto |
| PPLT ↔ PAXG-USD | +0.559 | Cross: commodity/crypto |
| TSM ↔ EWY | +0.558 | Intra-mega-cap (Asia) |
| XPD-USD ↔ SLV | +0.551 | Intra-commodity (metals) |
| COIN ↔ ETH-USD | +0.534 | Cross: high-beta/crypto |
| MSTR ↔ BTC-USD | +0.518 | Cross: high-beta/crypto |

### Diversification Metrics
- **Average pairwise correlation:** 0.222
- **Effective independent bets:** 12.8 / 21
- **Most correlated pair:** BTC-USD ↔ ETH-USD (0.836)

---

## 7. Allocation Mechanism

1. All groups generate trade candidates independently at their respective channel periods
2. Candidates are merged and sorted chronologically by `entry_ts` (FIFO)
3. Pro-rata scaling is applied per bucket (crypto, gold, metals, energy, equity) then overall
4. Drawdown governor checks portfolio-level drawdown and reduces exposure accordingly
5. No priority ranking between groups — allocation is purely chronological

---

## 8. Key Findings & Design Rationale

### Why different channel periods per group

| Group | Optimal | Suboptimal | Insight |
|-------|---------|------------|---------|
| CRYPTO | 20/10 (+643%) | 10/5 (-5.27%) | 24/7 noise kills short channels; wider filter needed |
| COMMODITY | 10/5 (+102%) | 20/10 (-24%) | Physical commodities trend fast; short window captures supply shocks |
| MEGA_ETF | 20/10 (+247%) | 10/5 (+129%) | Slower, mean-reverting; needs wider confirmation window |
| HIGH_BETA | 10/5 (+2,224%) | 20/10 (+376%) | Momentum-driven; fast breakouts capture fat-tail moves |

### Impact of pruning

| Config | Return | CAGR | MaxDD | PF |
|--------|--------|------|-------|-----|
| All 27 symbols (10/5 baseline) | +206% | 31.1% | 37.2% | 1.30 |
| 27 symbols BEST_PF grouped | +2,556% | 120.7% | 21.5% | 2.60 |
| **21 symbols pruned + grouped** | **+5,288%** | **162.3%** | **21.1%** | **3.85** |

Removing 6 unprofitable assets doubled returns and nearly doubled PF because freed allocation slots flow to winners.

---

## 9. Evolution Log

| Date | Change | Result |
|------|--------|--------|
| Baseline | All 27 symbols, uniform 10/5 channel | +206%, PF 1.30 |
| Group optimization | 4 groups with optimal channels | +2,556%, PF 2.60 |
| Pruned universe | Removed 6 losers (AAPL, AMZN, NVDA, SPY, QQQ, EWJ) | +5,288%, PF 3.85 |
