# Session Turtle X3 — Backtest Research Compendium

**Strategy:** Session Turtle Trend X3 (Donchian channel breakout, 5-min session bars, 3x exposure)  
**Document compiled:** 2026-04-04  
**Period covered by backtests:** Feb 2022 – Apr 2026 (≈4.1 years)

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Most Recent Run: Donchian Group A vs B (2026-04-03)](#1-most-recent-run-donchian-group-ab-split-20260403)
3. [Universe Composition Audit (2026-04-03)](#2-universe-composition-audit-20260403)
4. [Document Review / Production Baseline (2026-04-03)](#3-document-review--production-baseline-20260403)
5. [Channel Granularity Comparison: Raw 5m vs Session 10/5 vs Session 20/10](#4-channel-granularity-comparison)
6. [Donchian vs MA Cross (Channel Type Study)](#5-donchian-vs-ma-cross)
7. [Donchian 20 — Directional Volume & Pyramiding](#6-donchian-20--directional-volume--pyramiding)
8. [Donchian 20 — Relative Volume Sizing](#7-donchian-20--relative-volume-sizing)
9. [Donchian 20 — Low-Risk Pyramiding](#8-donchian-20--low-risk-pyramiding)
10. [Asset Universe Volatility & Theme Audit](#9-asset-universe-volatility--theme-audit)
11. [Key Findings & Decision Log](#key-findings--decision-log)

---

## Strategy Overview

The **Session Turtle Trend X3** strategy is a systematic trend-following system built on Donchian channel breakouts computed on 5-minute intraday bars ("session bars"). The core mechanics:

| Component | Detail |
|---|---|
| Signal | Donchian channel breakout (configurable period) |
| Bar resolution | 5-minute session bars |
| Entry | Next-bar open after confirmed breakout bar |
| Exit | Opposite Donchian channel (exit period = entry / 2 by default) |
| Position sizing | Risk-% per trade, notional exposure multiplier |
| Exposure multiplier | 3.0x (X3) |
| Trend filter | 4h EMA-55 / EMA-200 cross (must be aligned) |
| Per-asset overlay | Daily EMA-200 gate (quarter-size below) |
| Drawdown Governor | Two-tier: −15% → 1.5x exposure; −25% → 0.5x |
| Extended-hours proxy | VIX ≤ 15 + FG ≥ 60 → risk-on; VIX ≥ 25 or FG ≤ 30 → risk-off |
| Conviction boost | Breakout vol, close location, RVOL ratio → up to 1.25x size |
| Asset class caps | Crypto / Gold / Metals / Energy (configurable per run) |

Direction on short suppression: longs are suppressed in risk-off regime (mult = ~0); shorts are suppressed in risk-on regime.

---

## 1. Most Recent Run: Donchian Group A/B Split (2026-04-03)

**Report file:** `reports/donchian_group_backtest_20260403/report.md`  
**Run timestamp:** 2026-04-03 23:38

### Motivation

The key research question was: *can we improve returns and reduce drawdown by using different Donchian channel periods for volatile vs stable assets?*

- **Group A** (high-beta, fast-moving): 10-bar entry / 5-bar exit  
- **Group B** (steadier, lower-vol): 20-bar entry / 10-bar exit

### Group Definitions

| Group | Period | Tickers |
|---|---|---|
| A | 10 / 5 | BTC-USD, ETH-USD, SOL-USD, PAXG-USD, SLV (XAG), XPD-USD, PPLT (XPT), BRENT (BZ), NATGAS-USD, COIN, CRCL, HOOD, INTC, META, MSTR, PLTR, TSLA |
| B | 20 / 10 | COPPER-USD, AMZN, GOOGL, NVDA, EWJ, EWY, QQQ, SPY, AAPL, TSM |

### Three-Way Comparison

| Run | Syms | Trades | Return% | CAGR% | MaxDD% | PF | WR% | Final$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE (10/5, all 27) | 27 | 465 | +206.40% | 31.06% | 37.23% | 1.30 | 38.5% | $3,064 |
| **DUAL_PERIOD (A:10/5, B:20/10)** | **27** | **481** | **+3,064.64%** | **130.36%** | **24.65%** | **2.67** | **41.6%** | **$31,646** |
| GROUP_B_ONLY (20/10, 10 syms) | 10 | 146 | −0.63% | −0.15% | 33.41% | 1.00 | 34.9% | $994 |

### Key Takeaway

The **Dual Period split is by far the strongest result** for this 27-symbol universe: CAGR jumps from 31% to 130%, max drawdown falls from 37% to 25%, and profit factor rises from 1.30 to 2.67. Running Group B alone at 20/10 is essentially flat — those assets only contribute when combined with Group A's high-beta PnL engine.

### DUAL_PERIOD Top Performers

| Ticker | Group | Trades | WR% | PF | Total PnL |
|---|---|---:|---:|---:|---:|
| COIN | A | 8 | 62.5% | 22.05 | +$5,065 |
| BRENT | A | 31 | 45.2% | 6.12 | +$4,335 |
| META | A | 25 | 48.0% | 9.71 | +$3,498 |
| SLV | A | 16 | 31.2% | 8.46 | +$2,704 |
| SOL-USD | A | 35 | 40.0% | 3.21 | +$2,374 |
| PAXG-USD | A | 40 | 50.0% | 5.94 | +$2,320 |
| MSTR | A | 8 | 75.0% | 9.12 | +$2,306 |
| INTC | A | 12 | 33.3% | 2.82 | +$2,078 |
| EWY | B | 14 | 64.3% | 5.99 | +$2,027 |
| PLTR | A | 18 | 55.6% | 2.42 | +$1,780 |

### DUAL_PERIOD Laggards

| Ticker | Group | Trades | WR% | PF | Total PnL |
|---|---|---:|---:|---:|---:|
| NATGAS-USD | A | 22 | 27.3% | 0.41 | −$804 |
| ETH-USD | A | 42 | 33.3% | 0.56 | −$1,001 |
| AAPL | B | 16 | 18.8% | 0.13 | −$1,292 |
| AMZN | B | 7 | 14.3% | 0.02 | −$594 |

### BASELINE vs DUAL_PERIOD — Delta Analysis

Switching from a uniform 10/5 to the split period:
- COIN PnL: +$303 → +$5,065 (+$4,762)
- META PnL: −$64 → +$3,498 (+$3,562, largely short-side)
- SLV PnL: +$247 → +$2,704 (+$2,457)
- BRENT PnL: +$231 → +$4,335 (+$4,104)
- AAPL PnL: −$97 → −$1,292 (worse under 20/10)
- AMZN PnL: +$58 → −$594 (worse under 20/10)

---

## 2. Universe Composition Audit (2026-04-03)

**Report file:** `reports/universe_backtest_audit_20260403/report.md`  
**Run timestamp:** 2026-04-03 22:39  
**Strategy config:** X3, DD Governor 15%/25%, production config

### Motivation

Exploring what happens to performance as we add "steady" assets (QQQ, SPY, TSM, AAPL) to the original 23-symbol universe.

### Universe Comparison

| Universe | Symbols | Trades | Final$ | Return% | CAGR% | MaxDD% | PF | WR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_27 | 27 | 554 | $4,078 | +307.84% | 40.43% | 37.23% | 1.31 | 37.4% |
| **ORIGINAL_23** | **23** | **581** | **$35,406** | **+3,440.61%** | **137.10%** | **24.99%** | **2.72** | **39.6%** |
| HIGHBETA_ONLY | 11 | 250 | $2,513 | +151.30% | 25.04% | 33.13% | 1.35 | 36.4% |
| EQUITIES_ONLY | 17 | 354 | $7,952 | +695.19% | 65.67% | 30.84% | 1.95 | 39.3% |
| NO_BROAD_ETF | 25 | 508 | $5,281 | +428.11% | 49.48% | 40.35% | 1.34 | 35.6% |
| NO_DRAG | 22 | 477 | $11,353 | +1,035.28% | 79.83% | 27.97% | 1.66 | 40.2% |
| HIGHBETA_PLUS | 13 | 301 | $2,232 | +123.21% | 21.40% | 36.25% | 1.25 | 34.5% |

### Asset Universe Definitions

| Label | Tickers included |
|---|---|
| FULL_27 | All 27: includes QQQ, SPY, TSM, AAPL (new assets) |
| ORIGINAL_23 | Core 23: BTC, ETH, SOL, PAXG, COPPER, XAG, XPD, XPT, BZ, NATGAS, AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY |
| HIGHBETA_ONLY | 11 alts: BTC, ETH, SOL, PAXG, COPPER, XAG, XPD, XPT, COIN, CRCL, MSTR |
| EQUITIES_ONLY | 17 equities: AMZN, COIN, CRCL, GOOGL, HOOD, INTC, META, MSTR, NVDA, PLTR, TSLA, EWJ, EWY, QQQ, SPY, TSM, AAPL |
| NO_BROAD_ETF | 25 = FULL_27 minus QQQ and SPY |
| NO_DRAG | 22 = ORIGINAL_23 minus ETH and EWJ |
| HIGHBETA_PLUS | 13 = HIGHBETA_ONLY + PLTR, TSLA, HOOD, META, INTC (no raw crypto ETFs) |

### ORIGINAL_23 Bucket Breakdown

| Bucket | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|
| equity | 249 | +$19,574 | 42.6% | 2.97 |
| metals | 102 | +$8,566 | 41.2% | 6.57 |
| energy | 64 | +$2,200 | 35.9% | 1.71 |
| gold | 45 | +$2,123 | 44.4% | 3.95 |
| crypto | 121 | +$1,943 | 32.2% | 1.42 |

### ORIGINAL_23 Volatility Tier Breakdown

| Tier | Assets | Trades | Total PnL | WR% | PF |
|---|---:|---:|---:|---:|---:|
| high (30–50%) | 7 | 190 | +$16,072 | 43.2% | 4.46 |
| extreme (80%+) | 7 | 145 | +$10,344 | 37.2% | 2.55 |
| very_high (50–80%) | 5 | 118 | +$4,689 | 34.7% | 1.70 |
| medium (15–30%) | 4 | 128 | +$3,301 | 41.4% | 2.69 |

### Key Takeaway

Adding QQQ, SPY, TSM, AAPL (FULL_27) *collapses* CAGR from 137% → 40%. The drag is concentrated in broad-market ETFs and AAPL/AMZN. The ORIGINAL_23 universe, built around high-beta and commodity-linked assets, is the strongest configuration for this strategy.

---

## 3. Document Review / Production Baseline (2026-04-03)

**Report file:** `reports/session_turtle_x3_document_review_20260403/`  
**Run timestamp:** 2026-04-03  
**Universe:** ORIGINAL_23 (23 symbols)

This run was the formal production-style document for investors, using the standard DD Governor, VIX/FG proxy, per-asset EMA-200 filter, asset class caps, and breakout conviction boost.

### Headline KPIs

| Metric | Value |
|---|---|
| Total Return | +3,440.61% |
| CAGR | 137.10% |
| Max Drawdown | 24.99% |
| Profit Factor | 2.72 |
| Win Rate | 39.59% |
| Executed Trades | 581 |
| Final Equity | $35,406 |
| Backtest Period | Feb 2022 – Mar 2026 (4.1 yr) |

### Bias Audit Result

> *"The backtest was audited for forward-looking bias — all signals use completed bars, entries fill at next-bar open with slippage, and overlay lookups apply 1-day lags. No look-ahead bias was detected."*

### Key Configuration

| Parameter | Value |
|---|---|
| Channel period | 10 (entry) / 5 (exit) |
| Exposure mult | 3.0x |
| DD Governor | 15% / 25% triggers |
| VIX thresholds | Risk-on ≤ 15, risk-off ≥ 25 |
| FG thresholds | Greed ≥ 60, fear ≤ 30 |
| EMA overlay | 200-period daily |
| Trend filter | 55/200 4h cross |
| Conviction boost | Max 1.25x, avg ~1.17x |
| Asset class caps | Crypto 1.0x, Gold 0.8x, Metals 0.8x, Energy 0.8x |

### Bucket PnL (Production Baseline)

| Bucket | PnL |
|---|---|
| Equities | +$36,448 |
| Metals | +$9,675 |
| Energy | +$17,037 |
| Gold | +$1,703 |
| Crypto | +$7,790 |

---

## 4. Channel Granularity Comparison

**Report file:** `reports/channel_granularity_comparison/current_baseline/`

Compares three channel computation approaches on the same 25-symbol universe:

| Variant | Channel | Entry P | Exit P | Trades | Return% | CAGR% | MaxDD% | PF | Final$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Session 20/10 | Session bars | 20 | 10 | 330 | +7,265% | 182.91% | 36.97% | 2.39 | $73,652 |
| Session 10/5 | Session bars | 10 | 5 | 588 | +10,990% | 212.56% | 31.87% | 2.89 | $110,896 |
| Raw 5m 20/10 | Raw 5-min bars | 20 | 10 | 20,789 | +101.77% | 18.43% | 66.44% | 1.04 | $2,018 |

> Note: This run did **not** include the DD Governor (drawdown governor disabled). Results therefore differ from the governed runs above.

### Key Takeaway

- **Session bars (4h bars derived from session data) dramatically outperform raw 5-minute granularity.** Raw 5m at 20/10 generates 20k+ trades with a profit factor of only 1.04 and a 66% max drawdown — essentially noise.
- Session 10/5 slightly edges out Session 20/10 on CAGR (213% vs 183%) with lower drawdown (32% vs 37%).
- The Donchian group split experiment (Section 1) effectively applies 10/5 to high-beta assets and 20/10 to steadier ones to capture complementary dynamics.

---

## 5. Donchian vs MA Cross

**Report file:** `reports/donchian_vs_ma/`  
**Universe:** Shared account core assets

### Comparison: Signal Generation Method

| Scenario | Entry Signal | Trades | Return% | CAGR% | MaxDD% | Final$ |
|---|---|---:|---:|---:|---:|---:|
| **Donchian 20 + 4h EMA Filter** | D20 breakout, EMA filter | 137 | +486.69% | 55.14% | 19.45% | $5,867 |
| Donchian 55 + 4h EMA Filter | D55 breakout, EMA filter | 68 | +250.82% | 38.83% | 28.93% | $3,508 |
| EMA 55/200 Cross | MA cross signal | 141 | +241.07% | 35.39% | 26.35% | $3,411 |
| Donchian 20 Pure | D20, no EMA filter | 167 | +62.24% | 12.76% | 44.07% | $1,622 |
| Donchian 55 Pure | D55, no EMA filter | 79 | +61.92% | 13.43% | 29.24% | $1,619 |

### Key Takeaway

- **The 4h EMA trend filter is essential** — Donchian 20 without it drops from 55% CAGR to 13%.
- Donchian 20 + EMA filter beats MA cross (55% vs 35% CAGR) and achieves lower max drawdown (19% vs 26%).
- Donchian 55 is more conservative (fewer trades, higher drawdown) — it works but is less effective for the high-beta universe.

---

## 6. Donchian 20 — Directional Volume & Pyramiding

**Report file:** `reports/donchian20_directional_pyramid/`  
**Period:** Feb 2022 – Mar 2026

Tests four sizing/pyramiding variants on Donchian 20 + EMA filter baseline:

| Scenario | Trades (parent) | Add-ons | Lots | Return% | CAGR% | MaxDD% | Final$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Directional Volume (5%→7%) | 250 | 0 | 250 | +578.43% | 60.84% | 21.0% | $6,784 |
| **Baseline** | **268** | **0** | **268** | **+437.59%** | **51.81%** | **18.1%** | **$5,376** |
| Directional + Pyramiding | 170 | 52 | 222 | +312.66% | 42.17% | 17.67% | $4,127 |
| Pyramiding only | 161 | 47 | 208 | +69.18% | 13.94% | 28.16% | $1,692 |

> "Directional Volume" = size longs at 5% risk, shorts at 7% risk (or vice versa depending on regime).

### Key Takeaway

- **Directional volume boost alone** (+2% risk on trend-aligned direction) improves CAGR from 52% → 61% while slightly increasing drawdown (18% → 21%).
- **Pyramiding hurts** when combined with directional boost (312% vs 578%) — add-on capacity is consumed by the larger initial positions, leaving fewer follow-on slots.
- **Pure pyramiding** without volume boost drops return to 69% — worse than baseline.
- Best single-layer enhancement: directional volume adjustment.

---

## 7. Donchian 20 — Relative Volume Sizing

**Report file:** `reports/donchian20_volume/`  
**Period:** Feb 2022 – Mar 2026

Tests whether scaling position size by relative volume (RVOL) at entry improves risk-adjusted returns.

| Scenario | Trades | Return% | CAGR% | MaxDD% | Avg RVOL | Avg Vol Scale | Final$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Donchian 20 + RVOL Sizing** | 131 | +530.52% | 57.94% | 24.47% | 5.21x | 1.37x | $6,305 |
| Baseline (Donchian 20 + EMA) | 137 | +486.69% | 55.14% | 19.45% | 5.43x | 1.00x | $5,867 |

### Key Takeaway

- RVOL sizing provides a modest CAGR improvement (+2.8pp) but increases max drawdown from 19% to 24%.
- The avg 1.37x scale means entries on high-volume breakouts receive ~37% larger positions.
- Risk-adjusted improvement is marginal; the higher drawdown may not justify the complexity.

---

## 8. Donchian 20 — Low-Risk Pyramiding

**Report file:** `reports/donchian20_pyramid_lowrisk/`  
**Period:** Feb 2022 – Mar 2026

Tests pyramiding at a reduced base risk (2.5% instead of 5%) to allow add-ons without over-sizing:

| Scenario | Parent Trades | Add-ons | Lots | Return% | CAGR% | MaxDD% | Final$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2.5%→3.5% Directional + Pyramiding | 164 | 81 | 245 | +41.84% | 9.06% | 24.79% | $1,418 |
| Base 2.5% + Pyramiding | 175 | 90 | 265 | +39.14% | 8.54% | 26.25% | $1,391 |

### Key Takeaway

- Halving base risk to 2.5% to accommodate pyramiding dramatically reduces returns (9% CAGR vs 52% at 5%).
- Pyramiding at lower risk does not recover the edge lost from smaller position sizes.
- Conclusion: for this strategy, larger single-entry sizing (standard 5%) is better than fragmented pyramid entries at lower sizes.

---

## 9. Asset Universe Volatility & Theme Audit

**Report file:** `reports/session_turtle_x3_document_review_20260403/asset_volatility_audit.md`  
**Source:** ORIGINAL_23 production run

### Per-Asset Classification

| Ticker | Theme | Vol Tier | Ann.Vol% | WR% | PF | Total PnL |
|---|---|---|---:|---:|---:|---:|
| MSTR | Crypto-Proxy Equity | extreme (80%+) | 150% | 50% | 5.22 | +$2,131 |
| GOOGL | Mega-Cap Tech | extreme (80%+) | 136% | 30% | 0.61 | −$458 |
| NVDA | Mega-Cap Tech | extreme (80%+) | 128% | 46% | 1.52 | +$609 |
| CRCL | Crypto-Proxy Equity | extreme (80%+) | 113% | 50% | inf | +$914 |
| NATGAS-USD | Energy / Commodities | extreme (80%+) | 89% | 25% | 0.45 | −$1,127 |
| SOL-USD | Crypto L1 | extreme (80%+) | 81% | 38% | 2.88 | +$2,441 |
| COIN | Crypto-Proxy Equity | extreme (80%+) | 81% | 46% | 12.46 | +$5,834 |
| PLTR | AI / Disruptive Tech | very high (50–80%) | 64% | 42% | 1.80 | +$1,228 |
| HOOD | Fintech High-Beta | very high (50–80%) | 62% | 56% | 2.58 | +$1,116 |
| ETH-USD | Crypto L1 | very high (50–80%) | 58% | 24% | 0.47 | −$1,321 |
| TSLA | EV / Disruptive | very high (50–80%) | 58% | 43% | 2.42 | +$900 |
| INTC | Mega-Cap Tech | very high (50–80%) | 54% | 25% | 3.15 | +$2,765 |
| META | Mega-Cap Tech | high (30–50%) | 43% | 43% | 7.41 | +$4,101 |
| BTC-USD | Crypto L1 | high (30–50%) | 43% | 37% | 1.96 | +$823 |
| XPD-USD | Precious Metals | high (30–50%) | 42% | 48% | 2.97 | +$1,043 |
| XAG-USD (SLV) | Precious Metals | high (30–50%) | 39% | 52% | 34.71 | +$6,011 |
| BZ-USD (BRENT) | Energy / Commodities | high (30–50%) | 33% | 44% | 4.15 | +$3,326 |
| XPT-USD (PPLT) | Precious Metals | high (30–50%) | 33% | 38% | 4.37 | +$1,354 |
| AMZN | Mega-Cap Tech | high (30–50%) | 31% | 41% | 0.40 | −$586 |
| COPPER-USD | Industrial Metals | medium (15–30%) | 27% | 30% | 1.37 | +$158 |
| EWY | International ETF | medium (15–30%) | 26% | 54% | 7.59 | +$1,446 |
| EWJ | International ETF | medium (15–30%) | 18% | 38% | 0.27 | −$427 |
| PAXG-USD | Crypto-Gold Hybrid | medium (15–30%) | 15% | 44% | 3.95 | +$2,123 |

### Theme PnL Summary (ORIGINAL_23)

| Theme | Assets | Total PnL |
|---|---:|---:|
| Crypto-Proxy Equity (MSTR, COIN, CRCL) | 3 | +$8,879 |
| Precious Metals (XAG, XPD, XPT) | 3 | +$8,408 |
| Mega-Cap Tech (GOOGL, NVDA, INTC, META, AMZN) | 5 | +$6,432 |
| Energy (NATGAS, BRENT) | 2 | +$2,200 |
| Crypto-Gold Hybrid (PAXG) | 1 | +$2,123 |
| Crypto L1 (BTC, ETH, SOL) | 3 | +$1,943 |
| AI / Disruptive (PLTR) | 1 | +$1,228 |
| Fintech High-Beta (HOOD) | 1 | +$1,116 |
| International ETF (EWJ, EWY) | 2 | +$1,019 |
| EV / Disruptive (TSLA) | 1 | +$900 |
| Industrial Metals (COPPER) | 1 | +$158 |

### Problematic Assets (Consistent Drag)

| Ticker | Vol Tier | PnL Across Runs | Notes |
|---|---|---|---|
| ETH-USD | very_high | Consistently negative | High vol but low directional persistence |
| NATGAS-USD | extreme | Consistently negative | Whipsaw; extreme vol, low trend quality |
| GOOGL | extreme | Negative | High IV, mean-reverts frequently |
| AMZN | high | Slightly negative/flat | Moderate trend quality |
| EWJ | medium | Negative | Low vol, poor trend signal at 10/5 |
| AAPL | medium | Negative whenever included | Dilutive; low vol with poor fit |
| QQQ | medium | Negative whenever included | Broad ETF = dilution |
| SPY | medium | Near-zero | Broad ETF = no edge |

---

## Key Findings & Decision Log

### Finding 1: Channel Period Should Match Asset Volatility

**Evidence:** The Donchian Group A/B split (Section 1) shows that assigning shorter periods (10/5) to high-beta assets and longer periods (20/10) to steadier assets produces a dramatic improvement: 3,064% return vs 206% for uniform 10/5. The shorter period captures the faster mean-reverting/breakout patterns of volatile assets, while the longer period filters noise for lower-vol assets.

**Decision:** Use **Group A (10/5)** for high-beta crypto, metals, and high-vol equities; **Group B (20/10)** for large-cap equities and index proxies.

---

### Finding 2: Universe Composition Is a Primary Return Driver

**Evidence:** Adding QQQ, SPY, AAPL, TSM to the ORIGINAL_23 (Section 2) drops CAGR from 137% to 40%. These assets consume capacity slots and dilute sizing on alpha-generating positions.

**Decision:** Keep the core universe limited to **high-beta, trend-friendly assets**. Broad ETFs and low-vol large-caps should be excluded from the primary strategy.

---

### Finding 3: Session Bars Are Critical — Raw 5m Is Noise

**Evidence:** Raw 5-minute channel (20/10) generates 20k trades with PF = 1.04 and 66% drawdown (Section 4). Session bars (same period) → 330 trades, PF = 2.39, 37% drawdown.

**Decision:** Always compute Donchian channels on **session bars** (aggregated to 4h via session open/close), not raw 5-minute candles.

---

### Finding 4: The 4h EMA Trend Filter Is Not Optional

**Evidence:** Donchian 20 without trend filter = 13% CAGR / 44% drawdown. With EMA filter = 55% CAGR / 19% drawdown (Section 5). The trend filter eliminates ~80% of counter-trend noise trades.

**Decision:** The 55/200 4h EMA filter is a **hard requirement** of the strategy.

---

### Finding 5: Pyramiding Hurts at Standard Risk Sizing

**Evidence:** Adding pyramiding to standard 5% risk entries either hurts or is flat (Section 6). Low-risk (2.5%) pyramid is dramatically worse (Section 8). Directional volume adjustment (+2% in trend direction) gives the best incremental gain.

**Decision:** Do **not** use pyramiding. Use **directional volume sizing** if enhancement is needed.

---

### Finding 6: RVOL Sizing Adds Marginal Value With Higher DrawdownCost

**Evidence:** RVOL sizing adds ~3pp CAGR but increases max drawdown from 19% to 24% (Section 7). Avg RVOL at entry is 5.2x normal — breakouts already occur on elevated volume naturally.

**Decision:** RVOL sizing is **optional / deprioritized**. The conviction boost mechanism in the main strategy already partially captures this.

---

### Finding 7: Precious Metals Are Underrated Alpha Sources

**Evidence:** XAG (SLV) = PF 34.71, PPLT (XPT) = PF 4.37, XPD = PF 2.97 in ORIGINAL_23 (Section 9). Metals bucket = $8,566 total PnL, second only to equities. They're in Group A (10/5) and contribute consistently.

**Decision:** Maintain metals allocation. Consider reviewing PPLT proxy vs direct XPT access.

---

### Finding 8: ETH and NATGAS Are Consistent Losers

**Evidence:** ETH = −$1,321 (PF 0.47); NATGAS = −$1,127 (PF 0.45) across ORIGINAL_23 and FULL_27 runs. In the DUAL_PERIOD run ETH = −$1,001 and NATGAS = −$804.

**Decision:** ETH and NATGAS are **candidates for removal** from the universe. The NO_DRAG universe (ORIGINAL_23 minus ETH and EWJ) delivers 79.83% CAGR vs 40% for FULL_27, confirming drag reduction value.

---

## Run Chronology

| Date | Report Folder | What Was Tested |
|---|---|---|
| 2026-04-03 22:39 | `universe_backtest_audit_20260403` | 7 universe variants; confirmed ORIGINAL_23 superiority |
| 2026-04-03 23:38 | `donchian_group_backtest_20260403` | **Group A (10/5) vs Group B (20/10) split** — strongest result found |
| 2026-04-03 | `session_turtle_x3_document_review_20260403` | Production investor report on ORIGINAL_23 |
| (earlier) | `channel_granularity_comparison` | Session vs raw 5m bars; session 10/5 vs 20/10 |
| (earlier) | `donchian_vs_ma` | Donchian 20 vs 55 vs EMA cross |
| (earlier) | `donchian20_directional_pyramid` | Directional vol boost, pyramiding variants |
| (earlier) | `donchian20_volume` | RVOL entry scaling |
| (earlier) | `donchian20_pyramid_lowrisk` | Low-risk pyramid experiment |

---

## Next Research Questions

Based on findings, the open questions are:

1. **Is the DUAL_PERIOD result robust out-of-sample?** The 3,064% return on DUAL_PERIOD is extraordinary — further walk-forward or regime analysis is needed to confirm it isn't period-specific.
2. **Should ETH and NATGAS be dropped?** The NO_DRAG experiment (−ETH, −EWJ) improves CAGR to 80% vs 40%. Dropping just NATGAS and ETH from ORIGINAL_23 could be tested.
3. **What is the optimal Group B list?** GROUP_B_ONLY at 20/10 is flat — Group B assets only add value in the combined dual-period run. Which B-assets are drag and which are additive?
4. **AAPL specifically at 20/10**: AAPL is the worst performer in DUAL_PERIOD (−$1,292). Does AAPL improve at a different period, or should it be excluded?
5. **DD Governor calibration for DUAL_PERIOD**: The governor was active in some runs but tuned for the 10/5 uniform configuration. Re-calibrating triggers for the split-period universe may further reduce drawdown.
