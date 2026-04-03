# Universe Backtest Audit — Session Turtle X3

**Strategy:** x3 exposure, DD Governor 15%/25%, live production config  
**New assets (soon-to-add):** QQQ, SPY, TSM, AAPL included in FULL_27 baseline  
**Run date:** 2026-04-03 22:39  
**Universes tested:** 7

## Universe Comparison

| Universe | Symbols | Trades | Final $ | Return% | CAGR% | MaxDD% | PF | WR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_27 | 27 | 554 | $4,078.39 | +307.84% | 40.43% | 37.23% | 1.31 | 37.4% |
| ORIGINAL_23 | 23 | 581 | $35,406.07 | +3440.61% | 137.10% | 24.99% | 2.72 | 39.6% |
| HIGHBETA_ONLY | 11 | 250 | $2,512.98 | +151.30% | 25.04% | 33.13% | 1.35 | 36.4% |
| EQUITIES_ONLY | 17 | 354 | $7,951.94 | +695.19% | 65.67% | 30.84% | 1.95 | 39.3% |
| NO_BROAD_ETF | 25 | 508 | $5,281.09 | +428.11% | 49.48% | 40.35% | 1.34 | 35.6% |
| NO_DRAG | 22 | 477 | $11,352.85 | +1035.28% | 79.83% | 27.97% | 1.66 | 40.2% |
| HIGHBETA_PLUS | 13 | 301 | $2,232.13 | +123.21% | 21.40% | 36.25% | 1.25 | 34.5% |

---
## FULL_27  (27 symbols)

**Tickers:** BTC-USD, ETH-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, BZ-USD, NATGAS-USD, AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY, QQQ, SPY, TSM, AAPL

| Metric | Value |
|---|---|
| Total Return | +307.84% |
| CAGR | 40.43% |
| Max Drawdown | 37.23% |
| Profit Factor | 1.31 |
| Win Rate | 37.4% |
| Executed Trades | 554 |
| Long Trades | 317 |
| Short Trades | 237 |
| Final Equity | $4,078.39 |
| Entries @ Base Exposure | 404 |
| Entries @ DD Exposure 1 | 101 |
| Entries @ DD Exposure 2 | 49 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 288 | +3,497.96 | 42.0% | 1.74 |
| gold | 35 | +137.79 | 40.0% | 1.36 |
| metals | 80 | +15.10 | 33.8% | 1.01 |
| energy | 52 | +5.27 | 30.8% | 1.00 |
| crypto | 99 | -577.71 | 29.3% | 0.75 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 7 | 111 | +811.91 | 36.0% | 1.29 |
| very_high | 5 | 90 | +1,641.14 | 34.4% | 1.64 |
| high | 8 | 179 | +951.46 | 38.0% | 1.38 |
| medium | 7 | 174 | -326.10 | 39.1% | 0.84 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| PLTR | 64% | very_high | 20 | 45.0% | 2.78 | +1,176.86 | +1,176.86 | +0.00 |
| HOOD | 62% | very_high | 10 | 60.0% | 3.62 | +558.22 | +599.13 | -40.91 |
| MSTR | 150% | extreme | 4 | 75.0% | 4.28 | +456.51 | +386.92 | +69.60 |
| COIN | 81% | extreme | 8 | 50.0% | 2.28 | +408.05 | -142.90 | +550.95 |
| NVDA | 128% | extreme | 20 | 50.0% | 2.25 | +356.48 | +439.31 | -82.83 |
| BRENT | 33% | high | 29 | 41.4% | 1.82 | +355.67 | +535.33 | -179.66 |
| TSLA | 58% | very_high | 14 | 35.7% | 2.28 | +345.17 | +406.68 | -61.50 |
| TSM | 36% | high | 22 | 31.8% | 2.28 | +250.38 | -23.10 | +273.47 |
| SLV | 39% | high | 16 | 43.8% | 2.36 | +226.46 | +237.19 | -10.73 |
| INTC | 54% | very_high | 13 | 15.4% | 1.43 | +212.58 | -197.83 | +410.41 |
| EWY | 26% | medium | 23 | 56.5% | 1.65 | +147.95 | -37.20 | +185.15 |
| PAXG-USD | 15% | medium | 35 | 40.0% | 1.36 | +137.79 | +78.35 | +59.44 |
| XPD-USD | 42% | high | 27 | 48.1% | 1.35 | +128.57 | +124.95 | +3.61 |
| META | 43% | high | 22 | 45.5% | 1.41 | +125.69 | +163.39 | -37.70 |
| SOL-USD | 81% | extreme | 31 | 32.3% | 1.12 | +84.15 | -24.65 | +108.79 |
| AMZN | 31% | high | 15 | 40.0% | 1.23 | +31.20 | +6.95 | +24.25 |
| SPY | 16% | medium | 18 | 55.6% | 1.50 | +19.23 | +20.16 | -0.93 |
| CRCL | 113% | extreme | 1 | 0.0% | inf | +0.00 | +0.00 | +0.00 |
| BTC-USD | 43% | high | 35 | 28.6% | 0.98 | -10.17 | +82.47 | -92.64 |
| QQQ | 22% | medium | 23 | 56.5% | 0.88 | -15.01 | +56.96 | -71.97 |
| GOOGL | 136% | extreme | 24 | 37.5% | 0.66 | -142.88 | -221.14 | +78.25 |
| PPLT | 33% | high | 13 | 23.1% | 0.30 | -156.34 | -98.21 | -58.13 |
| COPPER-USD | 27% | medium | 24 | 16.7% | 0.43 | -183.59 | -88.09 | -95.50 |
| AAPL | 27% | medium | 28 | 32.1% | 0.62 | -215.68 | +61.35 | -277.03 |
| EWJ | 18% | medium | 23 | 21.7% | 0.32 | -216.79 | -230.21 | +13.42 |
| NATGAS-USD | 89% | extreme | 23 | 17.4% | 0.62 | -350.40 | +280.08 | -630.48 |
| ETH-USD | 58% | very_high | 33 | 27.3% | 0.28 | -651.69 | -380.92 | -270.77 |

---
## ORIGINAL_23  (23 symbols)

**Tickers:** BTC-USD, ETH-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, BZ-USD, NATGAS-USD, AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY

| Metric | Value |
|---|---|
| Total Return | +3440.61% |
| CAGR | 137.10% |
| Max Drawdown | 24.99% |
| Profit Factor | 2.72 |
| Win Rate | 39.6% |
| Executed Trades | 581 |
| Long Trades | 336 |
| Short Trades | 245 |
| Final Equity | $35,406.07 |
| Entries @ Base Exposure | 511 |
| Entries @ DD Exposure 1 | 70 |
| Entries @ DD Exposure 2 | 0 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 249 | +19,573.87 | 42.6% | 2.97 |
| metals | 102 | +8,566.37 | 41.2% | 6.57 |
| energy | 64 | +2,199.57 | 35.9% | 1.71 |
| gold | 45 | +2,123.27 | 44.4% | 3.95 |
| crypto | 121 | +1,942.99 | 32.2% | 1.42 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 7 | 145 | +10,344.28 | 37.2% | 2.55 |
| very_high | 5 | 118 | +4,688.79 | 34.7% | 1.70 |
| high | 7 | 190 | +16,072.48 | 43.2% | 4.46 |
| medium | 4 | 128 | +3,300.52 | 41.4% | 2.69 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| SLV | 39% | high | 21 | 52.4% | 34.71 | +6,011.26 | +6,011.45 | -0.19 |
| COIN | 81% | extreme | 13 | 46.2% | 12.46 | +5,833.67 | +2,798.96 | +3,034.70 |
| META | 43% | high | 30 | 43.3% | 7.41 | +4,100.71 | +479.67 | +3,621.04 |
| BRENT | 33% | high | 36 | 44.4% | 4.15 | +3,326.39 | +3,763.26 | -436.87 |
| INTC | 54% | very_high | 16 | 25.0% | 3.15 | +2,765.49 | +2,085.18 | +680.31 |
| SOL-USD | 81% | extreme | 37 | 37.8% | 2.88 | +2,441.22 | -4.60 | +2,445.82 |
| MSTR | 150% | extreme | 10 | 50.0% | 5.22 | +2,131.02 | +99.87 | +2,031.15 |
| PAXG-USD | 15% | medium | 45 | 44.4% | 3.95 | +2,123.27 | +2,110.61 | +12.66 |
| EWY | 26% | medium | 24 | 54.2% | 7.59 | +1,445.86 | +1,234.97 | +210.89 |
| PPLT | 33% | high | 21 | 38.1% | 4.37 | +1,354.12 | +1,512.49 | -158.37 |
| PLTR | 64% | very_high | 24 | 41.7% | 1.80 | +1,228.20 | +1,228.20 | +0.00 |
| HOOD | 62% | very_high | 18 | 55.6% | 2.58 | +1,115.59 | +1,252.47 | -136.88 |
| XPD-USD | 42% | high | 27 | 48.1% | 2.97 | +1,043.10 | +577.58 | +465.52 |
| CRCL | 113% | extreme | 2 | 50.0% | inf | +913.85 | +0.00 | +913.85 |
| TSLA | 58% | very_high | 14 | 42.9% | 2.42 | +900.36 | +1,173.99 | -273.63 |
| BTC-USD | 43% | high | 38 | 36.8% | 1.96 | +822.62 | +882.76 | -60.14 |
| NVDA | 128% | extreme | 28 | 46.4% | 1.52 | +609.31 | +45.20 | +564.11 |
| COPPER-USD | 27% | medium | 33 | 30.3% | 1.37 | +157.89 | +203.31 | -45.41 |
| EWJ | 18% | medium | 26 | 38.5% | 0.27 | -426.50 | -462.50 | +36.00 |
| GOOGL | 136% | extreme | 27 | 29.6% | 0.61 | -457.97 | -603.24 | +145.27 |
| AMZN | 31% | high | 17 | 41.2% | 0.40 | -585.72 | -75.39 | -510.33 |
| NATGAS-USD | 89% | extreme | 28 | 25.0% | 0.45 | -1,126.82 | +495.71 | -1,622.53 |
| ETH-USD | 58% | very_high | 46 | 23.9% | 0.47 | -1,320.85 | -118.64 | -1,202.21 |

---
## HIGHBETA_ONLY  (11 symbols)

**Tickers:** BTC-USD, ETH-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, COIN, CRCL, MSTR

| Metric | Value |
|---|---|
| Total Return | +151.30% |
| CAGR | 25.04% |
| Max Drawdown | 33.13% |
| Profit Factor | 1.35 |
| Win Rate | 36.4% |
| Executed Trades | 250 |
| Long Trades | 139 |
| Short Trades | 111 |
| Final Equity | $2,512.98 |
| Entries @ Base Exposure | 100 |
| Entries @ DD Exposure 1 | 79 |
| Entries @ DD Exposure 2 | 71 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| metals | 84 | +719.84 | 36.9% | 1.78 |
| equity | 21 | +420.76 | 42.9% | 1.46 |
| crypto | 103 | +389.89 | 35.9% | 1.18 |
| gold | 42 | -17.52 | 33.3% | 0.94 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 4 | 53 | +711.84 | 43.4% | 1.40 |
| very_high | 1 | 36 | -275.48 | 27.8% | 0.58 |
| high | 4 | 93 | +1,133.35 | 38.7% | 1.84 |
| medium | 2 | 68 | -56.74 | 32.4% | 0.89 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| SLV | 39% | high | 15 | 53.3% | 11.71 | +514.35 | +521.55 | -7.20 |
| MSTR | 150% | extreme | 11 | 45.5% | 1.85 | +415.46 | +682.01 | -266.54 |
| BTC-USD | 43% | high | 35 | 37.1% | 1.59 | +374.29 | +418.75 | -44.46 |
| SOL-USD | 81% | extreme | 32 | 43.8% | 1.34 | +291.08 | +389.91 | -98.83 |
| XPD-USD | 42% | high | 28 | 39.3% | 1.45 | +182.04 | +150.91 | +31.14 |
| PPLT | 33% | high | 15 | 26.7% | 1.25 | +62.67 | +213.21 | -150.54 |
| COIN | 81% | extreme | 9 | 44.4% | 1.01 | +5.30 | +66.05 | -60.75 |
| CRCL | 113% | extreme | 1 | 0.0% | inf | +0.00 | +0.00 | +0.00 |
| PAXG-USD | 15% | medium | 42 | 33.3% | 0.94 | -17.52 | +15.93 | -33.45 |
| COPPER-USD | 27% | medium | 26 | 30.8% | 0.82 | -39.22 | +6.21 | -45.43 |
| ETH-USD | 58% | very_high | 36 | 27.8% | 0.58 | -275.48 | -341.09 | +65.61 |

---
## EQUITIES_ONLY  (17 symbols)

**Tickers:** AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY, QQQ, SPY, TSM, AAPL

| Metric | Value |
|---|---|
| Total Return | +695.19% |
| CAGR | 65.67% |
| Max Drawdown | 30.84% |
| Profit Factor | 1.95 |
| Win Rate | 39.3% |
| Executed Trades | 354 |
| Long Trades | 212 |
| Short Trades | 142 |
| Final Equity | $7,951.94 |
| Entries @ Base Exposure | 258 |
| Entries @ DD Exposure 1 | 68 |
| Entries @ DD Exposure 2 | 28 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 354 | +6,951.93 | 39.3% | 1.95 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 5 | 90 | +2,693.79 | 37.8% | 1.83 |
| very_high | 4 | 68 | +3,740.23 | 39.7% | 3.11 |
| high | 3 | 60 | +740.01 | 40.0% | 1.85 |
| medium | 5 | 136 | -222.10 | 39.7% | 0.84 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HOOD | 62% | very_high | 15 | 66.7% | 20.07 | +2,854.01 | +2,498.41 | +355.61 |
| COIN | 81% | extreme | 13 | 46.2% | 5.98 | +1,648.20 | +218.46 | +1,429.75 |
| PLTR | 64% | very_high | 21 | 38.1% | 2.75 | +778.28 | +604.70 | +173.57 |
| META | 43% | high | 25 | 32.0% | 2.79 | +655.70 | -126.75 | +782.45 |
| MSTR | 150% | extreme | 19 | 47.4% | 1.62 | +630.45 | +44.74 | +585.71 |
| GOOGL | 136% | extreme | 32 | 25.0% | 1.64 | +526.95 | +588.83 | -61.88 |
| EWY | 26% | medium | 27 | 51.9% | 3.76 | +419.73 | +307.52 | +112.21 |
| INTC | 54% | very_high | 17 | 29.4% | 1.40 | +245.75 | +111.38 | +134.37 |
| TSM | 36% | high | 19 | 52.6% | 1.80 | +198.86 | +47.15 | +151.71 |
| NVDA | 128% | extreme | 24 | 41.7% | 1.27 | +161.07 | -16.00 | +177.08 |
| EWJ | 18% | medium | 30 | 36.7% | 1.19 | +49.88 | +13.29 | +36.59 |
| SPY | 16% | medium | 23 | 47.8% | 0.37 | -85.53 | -65.69 | -19.84 |
| QQQ | 22% | medium | 25 | 44.0% | 0.44 | -113.01 | -84.83 | -28.18 |
| AMZN | 31% | high | 16 | 37.5% | 0.56 | -114.55 | -11.39 | -103.16 |
| TSLA | 58% | very_high | 15 | 26.7% | 0.75 | -137.81 | -77.59 | -60.22 |
| CRCL | 113% | extreme | 2 | 50.0% | 0.43 | -272.88 | +0.00 | -272.88 |
| AAPL | 27% | medium | 31 | 22.6% | 0.21 | -493.17 | -90.37 | -402.80 |

---
## NO_BROAD_ETF  (25 symbols)

**Tickers:** BTC-USD, ETH-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, BZ-USD, NATGAS-USD, AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY, TSM, AAPL

| Metric | Value |
|---|---|
| Total Return | +428.11% |
| CAGR | 49.48% |
| Max Drawdown | 40.35% |
| Profit Factor | 1.34 |
| Win Rate | 35.6% |
| Executed Trades | 508 |
| Long Trades | 291 |
| Short Trades | 217 |
| Final Equity | $5,281.09 |
| Entries @ Base Exposure | 340 |
| Entries @ DD Exposure 1 | 119 |
| Entries @ DD Exposure 2 | 49 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 255 | +4,954.99 | 40.4% | 1.82 |
| energy | 50 | +374.08 | 32.0% | 1.26 |
| metals | 74 | +139.97 | 32.4% | 1.11 |
| gold | 33 | +84.56 | 30.3% | 1.16 |
| crypto | 96 | -1,272.53 | 29.2% | 0.62 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 7 | 111 | +687.87 | 34.2% | 1.18 |
| very_high | 5 | 94 | +2,226.34 | 33.0% | 1.62 |
| high | 8 | 172 | +1,923.94 | 40.7% | 1.64 |
| medium | 5 | 131 | -557.08 | 32.1% | 0.75 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| PLTR | 64% | very_high | 18 | 44.4% | 3.18 | +1,801.43 | +1,801.43 | +0.00 |
| NVDA | 128% | extreme | 19 | 52.6% | 3.94 | +847.58 | +980.59 | -133.01 |
| COIN | 81% | extreme | 10 | 50.0% | 2.64 | +834.10 | +61.95 | +772.15 |
| TSM | 36% | high | 22 | 45.5% | 4.60 | +736.12 | +311.05 | +425.08 |
| HOOD | 62% | very_high | 13 | 53.8% | 2.62 | +714.07 | +777.95 | -63.88 |
| BRENT | 33% | high | 30 | 43.3% | 2.50 | +711.05 | +933.50 | -222.45 |
| TSLA | 58% | very_high | 14 | 35.7% | 2.65 | +509.14 | +559.16 | -50.02 |
| SLV | 39% | high | 14 | 42.9% | 2.92 | +370.69 | +371.54 | -0.85 |
| INTC | 54% | very_high | 15 | 20.0% | 1.48 | +304.88 | -222.59 | +527.48 |
| BTC-USD | 43% | high | 33 | 33.3% | 1.16 | +129.12 | +211.66 | -82.54 |
| XPD-USD | 42% | high | 25 | 44.0% | 1.25 | +119.14 | +212.26 | -93.13 |
| AMZN | 31% | high | 13 | 46.2% | 1.65 | +98.76 | +65.11 | +33.66 |
| PAXG-USD | 15% | medium | 33 | 30.3% | 1.16 | +84.56 | +30.21 | +54.36 |
| EWY | 26% | medium | 24 | 54.2% | 1.17 | +52.86 | -144.22 | +197.08 |
| META | 43% | high | 21 | 42.9% | 1.05 | +16.01 | +113.66 | -97.65 |
| CRCL | 113% | extreme | 1 | 0.0% | inf | +0.00 | +0.00 | +0.00 |
| COPPER-USD | 27% | medium | 21 | 14.3% | 0.68 | -92.91 | -29.35 | -63.56 |
| MSTR | 150% | extreme | 6 | 33.3% | 0.75 | -97.37 | -107.64 | +10.28 |
| EWJ | 18% | medium | 23 | 30.4% | 0.32 | -252.80 | -274.49 | +21.69 |
| PPLT | 33% | high | 14 | 28.6% | 0.21 | -256.95 | -118.91 | -138.04 |
| GOOGL | 136% | extreme | 26 | 34.6% | 0.55 | -261.00 | -286.15 | +25.16 |
| SOL-USD | 81% | extreme | 29 | 31.0% | 0.74 | -298.47 | -58.44 | -240.03 |
| NATGAS-USD | 89% | extreme | 20 | 15.0% | 0.65 | -336.97 | +363.26 | -700.23 |
| AAPL | 27% | medium | 30 | 30.0% | 0.52 | -348.79 | +71.13 | -419.92 |
| ETH-USD | 58% | very_high | 34 | 23.5% | 0.19 | -1,103.18 | -660.81 | -442.37 |

---
## NO_DRAG  (22 symbols)

**Tickers:** BTC-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, BZ-USD, COIN, CRCL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWY, QQQ, SPY, TSM, AAPL

| Metric | Value |
|---|---|
| Total Return | +1035.28% |
| CAGR | 79.83% |
| Max Drawdown | 27.97% |
| Profit Factor | 1.66 |
| Win Rate | 40.2% |
| Executed Trades | 477 |
| Long Trades | 279 |
| Short Trades | 198 |
| Final Equity | $11,352.85 |
| Entries @ Base Exposure | 365 |
| Entries @ DD Exposure 1 | 96 |
| Entries @ DD Exposure 2 | 16 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 255 | +7,178.02 | 44.7% | 1.75 |
| energy | 33 | +1,146.78 | 42.4% | 1.98 |
| metals | 81 | +961.01 | 33.3% | 1.58 |
| gold | 36 | +611.54 | 36.1% | 1.86 |
| crypto | 72 | +455.48 | 33.3% | 1.17 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 4 | 78 | +2,028.01 | 38.5% | 1.63 |
| very_high | 4 | 59 | +3,355.76 | 39.0% | 1.72 |
| high | 7 | 176 | +4,456.35 | 40.9% | 1.92 |
| medium | 6 | 164 | +512.71 | 40.9% | 1.17 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| PLTR | 64% | very_high | 22 | 50.0% | 2.22 | +2,329.24 | +2,329.24 | +0.00 |
| NVDA | 128% | extreme | 28 | 50.0% | 4.94 | +1,997.60 | +1,394.72 | +602.89 |
| HOOD | 62% | very_high | 11 | 63.6% | 3.08 | +1,209.69 | +1,386.35 | -176.65 |
| BRENT | 33% | high | 33 | 42.4% | 1.98 | +1,146.78 | +1,571.82 | -425.04 |
| SLV | 39% | high | 16 | 43.8% | 4.84 | +1,004.37 | +985.23 | +19.14 |
| META | 43% | high | 27 | 44.4% | 2.05 | +843.09 | -245.77 | +1,088.86 |
| TSM | 36% | high | 21 | 42.9% | 2.90 | +789.98 | +247.98 | +542.00 |
| BTC-USD | 43% | high | 40 | 37.5% | 1.59 | +717.78 | +695.13 | +22.65 |
| PAXG-USD | 15% | medium | 36 | 36.1% | 1.86 | +611.54 | +569.42 | +42.12 |
| EWY | 26% | medium | 26 | 53.8% | 1.90 | +402.19 | +150.17 | +252.02 |
| COIN | 81% | extreme | 9 | 33.3% | 1.41 | +316.89 | -125.00 | +441.89 |
| QQQ | 22% | medium | 27 | 55.6% | 1.45 | +126.13 | +212.56 | -86.43 |
| PPLT | 33% | high | 13 | 23.1% | 1.29 | +109.52 | +327.29 | -217.77 |
| SPY | 16% | medium | 23 | 56.5% | 1.26 | +31.72 | +96.47 | -64.75 |
| TSLA | 58% | very_high | 11 | 27.3% | 1.01 | +6.58 | +153.00 | -146.43 |
| COPPER-USD | 27% | medium | 26 | 19.2% | 1.01 | +2.29 | +151.21 | -148.92 |
| MSTR | 150% | extreme | 9 | 44.4% | 0.95 | -24.18 | -11.59 | -12.59 |
| XPD-USD | 42% | high | 26 | 46.2% | 0.73 | -155.17 | +195.35 | -350.51 |
| INTC | 54% | very_high | 15 | 13.3% | 0.87 | -189.75 | -646.07 | +456.32 |
| SOL-USD | 81% | extreme | 32 | 28.1% | 0.82 | -262.30 | -66.52 | -195.78 |
| AAPL | 27% | medium | 26 | 26.9% | 0.40 | -661.16 | -52.95 | -608.22 |

---
## HIGHBETA_PLUS  (13 symbols)

**Tickers:** BTC-USD, ETH-USD, SOL-USD, PAXG-USD, COPPER-USD, XAG-USD, XPD-USD, XPT-USD, COIN, CRCL, MSTR, TSM, AAPL

| Metric | Value |
|---|---|
| Total Return | +123.21% |
| CAGR | 21.40% |
| Max Drawdown | 36.25% |
| Profit Factor | 1.25 |
| Win Rate | 34.5% |
| Executed Trades | 301 |
| Long Trades | 171 |
| Short Trades | 130 |
| Final Equity | $2,232.13 |
| Entries @ Base Exposure | 163 |
| Entries @ DD Exposure 1 | 57 |
| Entries @ DD Exposure 2 | 81 |

### Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 72 | +760.12 | 40.3% | 1.56 |
| metals | 79 | +504.14 | 34.2% | 1.58 |
| gold | 38 | -2.30 | 28.9% | 0.99 |
| crypto | 112 | -29.81 | 33.0% | 0.99 |

### Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| extreme | 4 | 52 | +510.37 | 36.5% | 1.30 |
| very_high | 1 | 36 | -373.48 | 27.8% | 0.44 |
| high | 5 | 120 | +1,359.80 | 40.8% | 1.86 |
| medium | 3 | 93 | -264.54 | 28.0% | 0.71 |

### Per-Asset Performance

| Ticker | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Long PnL | Short PnL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| TSM | 36% | high | 25 | 56.0% | 3.56 | +514.12 | +337.07 | +177.05 |
| SLV | 39% | high | 15 | 46.7% | 12.56 | +370.82 | +374.96 | -4.14 |
| MSTR | 150% | extreme | 9 | 33.3% | 1.67 | +317.77 | +420.22 | -102.45 |
| BTC-USD | 43% | high | 41 | 34.1% | 1.27 | +213.72 | +403.32 | -189.60 |
| XPD-USD | 42% | high | 24 | 41.7% | 1.53 | +187.04 | +118.80 | +68.23 |
| SOL-USD | 81% | extreme | 35 | 37.1% | 1.14 | +129.95 | +323.57 | -193.62 |
| PPLT | 33% | high | 15 | 26.7% | 1.34 | +74.10 | +190.16 | -116.06 |
| COIN | 81% | extreme | 7 | 42.9% | 1.22 | +62.65 | +143.09 | -80.45 |
| CRCL | 113% | extreme | 1 | 0.0% | inf | +0.00 | +0.00 | +0.00 |
| PAXG-USD | 15% | medium | 38 | 28.9% | 0.99 | -2.30 | +7.58 | -9.88 |
| COPPER-USD | 27% | medium | 25 | 24.0% | 0.53 | -127.82 | -65.70 | -62.12 |
| AAPL | 27% | medium | 30 | 30.0% | 0.66 | -134.42 | +54.19 | -188.62 |
| ETH-USD | 58% | very_high | 36 | 27.8% | 0.44 | -373.48 | -384.64 | +11.15 |
