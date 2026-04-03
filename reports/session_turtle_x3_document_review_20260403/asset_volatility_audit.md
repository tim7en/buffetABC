# Asset Universe — Volatility & Theme Audit

**Strategy:** Session Turtle X3  |  **Run date:** 2026-04-03

## Per-Asset Table

| Ticker | Theme | Bucket | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Core |
|---|---|---|---:|---|---:|---:|---:|---:|:---:|
| MSTR | Crypto-Proxy Equity | equity | 150% | extreme (80%+) | 10 | 50% | 5.22 | +2,131.02 | YES |
| GOOGL | Mega-Cap Tech | equity | 136% | extreme (80%+) | 27 | 30% | 0.61 | -457.97 | YES |
| NVDA | Mega-Cap Tech | equity | 128% | extreme (80%+) | 28 | 46% | 1.52 | +609.31 | YES |
| CRCL | Crypto-Proxy Equity | equity | 113% | extreme (80%+) | 2 | 50% | inf | +913.85 | YES |
| NATGAS-USD | Energy / Commodities | energy | 89% | extreme (80%+) | 28 | 25% | 0.45 | -1,126.82 | YES |
| SOL-USD | Crypto L1 | crypto | 81% | extreme (80%+) | 37 | 38% | 2.88 | +2,441.22 | YES |
| COIN | Crypto-Proxy Equity | equity | 81% | extreme (80%+) | 13 | 46% | 12.46 | +5,833.67 | YES |
| PLTR | AI / Disruptive Tech | equity | 64% | very high (50-80%) | 24 | 42% | 1.80 | +1,228.20 | YES |
| HOOD | Fintech High-Beta | equity | 62% | very high (50-80%) | 18 | 56% | 2.58 | +1,115.59 | YES |
| ETH-USD | Crypto L1 | crypto | 58% | very high (50-80%) | 46 | 24% | 0.47 | -1,320.85 | YES |
| TSLA | EV / Disruptive | equity | 58% | very high (50-80%) | 14 | 43% | 2.42 | +900.36 | YES |
| INTC | Mega-Cap Tech | equity | 54% | very high (50-80%) | 16 | 25% | 3.15 | +2,765.49 | YES |
| META | Mega-Cap Tech | equity | 43% | high (30-50%) | 30 | 43% | 7.41 | +4,100.71 | YES |
| BTC-USD | Crypto L1 | crypto | 43% | high (30-50%) | 38 | 37% | 1.96 | +822.62 | YES |
| XPD-USD | Precious Metals | metals | 42% | high (30-50%) | 27 | 48% | 2.97 | +1,043.10 | YES |
| XAG-USD* | Precious Metals | metals | 39% | high (30-50%) | 21 | 52% | 34.71 | +6,011.26 | YES |
| TSM | International Semi | equity | 36% | high (30-50%) | 0 | — | — | +0.00 | — |
| BZ-USD* | Energy / Commodities | energy | 33% | high (30-50%) | 36 | 44% | 4.15 | +3,326.39 | YES |
| XPT-USD* | Precious Metals | metals | 33% | high (30-50%) | 21 | 38% | 4.37 | +1,354.12 | YES |
| AMZN | Mega-Cap Tech | equity | 31% | high (30-50%) | 17 | 41% | 0.40 | -585.72 | YES |
| AAPL | Mega-Cap Tech | equity | 27% | medium (15-30%) | 0 | — | — | +0.00 | — |
| COPPER-USD | Industrial Metals | metals | 27% | medium (15-30%) | 33 | 30% | 1.37 | +157.89 | YES |
| EWY | International ETF | equity | 26% | medium (15-30%) | 24 | 54% | 7.59 | +1,445.86 | YES |
| QQQ | Broad Market ETF | equity | 22% | medium (15-30%) | 0 | — | — | +0.00 | — |
| EWJ | International ETF | equity | 18% | medium (15-30%) | 26 | 38% | 0.27 | -426.50 | YES |
| SPY | Broad Market ETF | equity | 16% | medium (15-30%) | 0 | — | — | +0.00 | — |
| PAXG-USD | Crypto-Gold Hybrid | gold | 15% | medium (15-30%) | 45 | 44% | 3.95 | +2,123.27 | YES |

> \* Proxy ticker used (see engine_ticker in JSON for mapping)

## Theme Summary (Core Universe)

| Theme | Assets | Trades | Total PnL |
|---|---:|---:|---:|
| Crypto-Proxy Equity | 3 | 25 | +8,878.54 |
| Precious Metals | 3 | 69 | +8,408.48 |
| Mega-Cap Tech | 5 | 118 | +6,431.82 |
| Energy / Commodities | 2 | 64 | +2,199.57 |
| Crypto-Gold Hybrid | 1 | 45 | +2,123.27 |
| Crypto L1 | 3 | 121 | +1,942.99 |
| AI / Disruptive Tech | 1 | 24 | +1,228.20 |
| Fintech High-Beta | 1 | 18 | +1,115.59 |
| International ETF | 2 | 50 | +1,019.36 |
| EV / Disruptive | 1 | 14 | +900.36 |
| Industrial Metals | 1 | 33 | +157.89 |

## Key Hypotheses

| # | Hypothesis | Evidence |
|---|---|---|
| 1 | High-beta/crypto assets drive most of the PnL | Check extreme + very-high vol tier PnL share |
| 2 | Mega-cap ETFs (SPY/QQQ) dilute returns | Expanded what-if: -33% to -97% CAGR across all scenarios |
| 3 | Crypto-proxy equities (MSTR, COIN, CRCL) add alpha beyond raw crypto | Compare theme PnL vs crypto L1 |
| 4 | International ETFs (EWJ, EWY) are low-vol drag | Check medium vol tier contribution |
