# Session Turtle Core x3 — Live Strategy Integration Guide

**Strategy**: Session Turtle Core x2 → upgraded to x3 exposure
**Chosen configuration**: `exposure_mult=3.0`, three-layer overlay (VIX daily + Crypto F&G + daily EMA(200) scaling)
**Basis**: Backtested Feb 2022 – Mar 2026 · 315 trades · CAGR 148.4% · Max DD 36.0% · PF 2.10

> **Note**: VIXY intraday (5m SMA) is a backtest research layer only and is **not used in live execution**. It requires a live 5-minute data feed and real-time regime classification that is not operationally practical. The three daily-refresh layers below are sufficient for live use.

---

## 1. Architecture Overview

```
Every entry decision passes through three sequential filters:

  Entry signal
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 1: VIX daily macro (non-crypto, all sessions)     │
  │  Source   : cache/sentiment/vix_closes.json  (yfinance)  │
  │  Signal   : prior-day VIX close vs 15 / 25 thresholds    │
  │  Applies  : equity, gold, metals buckets                  │
  └──────────────────────────┬───────────────────────────────┘
                             │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 2: Crypto Fear & Greed (crypto only, all sessions) │
  │  Source   : cache/sentiment/crypto_fg_scores.json         │
  │  Signal   : prior-day F&G score vs 60 / 30 thresholds    │
  │  Applies  : crypto bucket only                            │
  └──────────────────────────┬───────────────────────────────┘
                             │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 3: Daily EMA(200) scaling (per-asset, all         │
  │           sessions)                                       │
  │  Source   : local OHLCV bars (daily close vs EMA200)     │
  │  Signal   : price vs prior-day EMA(200) — 1-day lag      │
  │  Applies  : all assets independently                      │
  └──────────────────────────┬───────────────────────────────┘
                             │
       ▼
  Position size = base_size
                × layer1_mult   (VIX or F&G macro signal)
                × layer3_mult   (EMA trend alignment, per-asset)
```

All three multipliers stack multiplicatively. If VIX is in risk_off (0.5×) and price is below EMA200 on a long entry (0.5×), the combined size is 0.25× of base. There is no additive blending — each layer independently scales the same position.

---

## 2. Regime Rules (exact thresholds)

### Layer 1 — VIX Daily Macro (equity / gold / metals)

| Prior-day VIX close | Regime      | Long mult | Short mult |
|---------------------|-------------|-----------|------------|
| ≤ 15                | `risk_on`   | **1.0×**  | **1e-9×** (fully suppressed) |
| 15 < VIX < 25       | `neutral`   | 1.0×      | 1.0×       |
| ≥ 25                | `risk_off`  | **0.5×**  | 1.0×       |

### Layer 2 — Crypto Fear & Greed (crypto only)

| Prior-day F&G score | Regime           | Long mult | Short mult |
|---------------------|------------------|-----------|------------|
| ≥ 60                | `greed/risk_on`  | **1.0×**  | **1e-9×** (fully suppressed) |
| 30 < F&G < 60       | `neutral`        | 1.0×      | 1.0×       |
| ≤ 30                | `fear/risk_off`  | **0.5×**  | 1.0×       |

### Layer 3 — Daily EMA(200) Scaling (per-asset)

| Prior-day price vs EMA(200) | Direction | Mult  | Interpretation |
|-----------------------------|-----------|-------|----------------|
| price > EMA(200)            | Long      | 1.0×  | With-trend — full size |
| price > EMA(200)            | Short     | 0.5×  | Counter-trend — cautious |
| price < EMA(200)            | Long      | 0.5×  | Counter-trend — cautious |
| price < EMA(200)            | Short     | 1.0×  | With-trend — full size |

EMA period is 200 daily bars (industry standard). A 1-day lag is applied — the EMA signal from date T applies to entries on date T+1. This is the same lag policy as Layers 1 and 2.

**Lag policy**: All three layers use a strict 1-day lag. The signal observed on date T applies only to entries on date T+1.

**Why EMA(200) does not overfit**: 200-period daily EMA is the most widely used trend indicator in systematic trading. It is applied identically to every asset in the universe with no asset-specific tuning. The scaling rule (with-trend full, counter-trend half) is a structural bias, not an optimised parameter.

---

## 3. Data Pipeline

### 3a. VIX Daily Closes

**Source**: CBOE VIX via [yfinance](https://github.com/ranaroussi/yfinance) (`^VIX` ticker)
**Fetch function**: `edgar.services.sentiment_data.load_vix_closes()`
**Cache file**: `cache/sentiment/vix_closes.json`
**Format**: `{"YYYY-MM-DD": float, ...}` — raw VIX close (e.g. `{"2026-03-20": 26.78}`)
**Coverage**: 2021-01-04 onwards (1,309+ days as of Mar 2026)

```python
from edgar.services.sentiment_data import load_vix_closes

# First call downloads + caches; subsequent calls load from cache
vix_closes = load_vix_closes(start="2021-01-01", force_refresh=False)

# Force a fresh download (run daily after market close)
vix_closes = load_vix_closes(force_refresh=True)
```

**Live refresh schedule**: Once per trading day, after 4:30 PM ET (after VIX close is published).

---

### 3b. Crypto Fear & Greed Index

**Source**: [alternative.me public API](https://alternative.me/crypto/fear-and-greed-index/)
**Endpoint**: `https://api.alternative.me/fng/?limit=0&format=json`
**Fetch function**: `edgar.services.sentiment_data.load_crypto_fg_scores()`
**Cache file**: `cache/sentiment/crypto_fg_scores.json`
**Format**: `{"YYYY-MM-DD": float, ...}` — raw score 0–100 (e.g. `{"2026-03-23": 8.0}`)
**Coverage**: 2018-02-01 onwards (2,969+ days as of Mar 2026)
**No API key required.** The API returns full history in one request.

```python
from edgar.services.sentiment_data import load_crypto_fg_scores

# Returns full history; force_refresh fetches latest
crypto_fg = load_crypto_fg_scores(force_refresh=True)
```

**Live refresh schedule**: Once per day after 01:00 UTC (F&G index updates at midnight UTC).

---

### 3c. Daily OHLCV Bars (EMA(200) scaling)

**Source**: Local Tiingo/Binance bar cache (same data used for backtest and candidate generation)
**Fetch function**: `edgar.services.session_turtle_portfolio.build_per_asset_technical_state()`
**No external fetch required.** The function reads from the existing local bar cache.

The state object is computed once per session and covers all tickers in the universe. It requires at least 200 trading days of daily close history per asset (automatically satisfied by the 5-year lookback in the bar cache).

---

## 4. Building the Overlay State Objects

Computed once per day and passed into the report generator:

```python
import json
from pathlib import Path
from edgar.services.sentiment_data import load_vix_closes, load_crypto_fg_scores
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    CORE_SESSION_TURTLE_UNIVERSE,
)

root = Path(".")  # project root

# --- Layer 1 + 2: Refresh macro data ---
vix_closes = load_vix_closes(force_refresh=True)
crypto_fg  = load_crypto_fg_scores(force_refresh=True)

# Build combined VIX + F&G state object
macro_state = build_extended_hours_proxy_state(
    daily_vix_closes=vix_closes,
    crypto_fg_scores=crypto_fg,
)

# --- Layer 3: Build per-asset EMA state ---
universe = list(dict.fromkeys(
    (ticker, side, session)
    for ticker, side, session in CORE_SESSION_TURTLE_UNIVERSE
))
tech_state = build_per_asset_technical_state(
    universe=universe,
    lookback_years=5.0,
    warmup_days=300,
    ema_period=200,
    adx_period=14,   # loaded but not used in live (ADX gate inactive)
)
```

---

## 5. Full Live Run Call (x3 configuration)

```python
from edgar.services.session_turtle_portfolio import (
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)

# Step 1: build candidate trade set
candidates = build_session_turtle_shared_account_candidates(
    basket="core",
    initial_capital=<current_account_equity>,
    lookback_years=4.1,
    channel_period=20,
    base_risk_pct=0.05,
    fixed_stop_pct=0.10,
    directional_volume_risk_pct=0.07,
    trend_fast_period=55,
    trend_slow_period=200,
)

# Step 2: run the allocator with three-layer overlay
result = generate_session_turtle_shared_account_report(
    # core params
    basket="core",
    exposure_mult=3.0,
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    base_risk_pct=0.05,
    fixed_stop_pct=0.10,
    directional_volume_risk_pct=0.07,
    precomputed_candidates=candidates,

    # Layer 1+2: VIX daily macro + Crypto F&G (all sessions)
    use_extended_hours_proxy=True,
    extended_hours_proxy_state=macro_state,
    extended_hours_proxy_lag_days=1,
    extended_hours_vix_risk_on_threshold=15.0,
    extended_hours_vix_risk_off_threshold=25.0,
    extended_hours_fg_greed_threshold=60.0,
    extended_hours_fg_fear_threshold=30.0,
    extended_hours_long_risk_on_mult=1.0,
    extended_hours_long_neutral_mult=1.0,
    extended_hours_long_risk_off_mult=0.5,
    extended_hours_short_risk_on_mult=1e-9,   # suppress shorts in calm/bull environments
    extended_hours_short_neutral_mult=1.0,
    extended_hours_short_risk_off_mult=1.0,

    # Layer 3: Daily EMA(200) per-asset scaling
    use_per_asset_technical_overlay=True,
    per_asset_technical_state=tech_state,
    per_asset_ema_lag_days=1,
    per_asset_ema_above_long_mult=1.0,    # price above EMA: full long, half short
    per_asset_ema_above_short_mult=0.5,
    per_asset_ema_below_long_mult=0.5,    # price below EMA: half long, full short
    per_asset_ema_below_short_mult=1.0,
    per_asset_use_adx_gate=False,         # ADX gate is redundant at breakout entries
)

trades  = result["trades"]
summary = result["summary"]
```

---

## 6. How the Three Layers Interact

The three layers are **independent and multiplicative**. Each layer applies its own multiplier to the same position size. The final size is:

```
target_position_size = base_size
    × ext_hours_proxy_mult     # Layer 1 (VIX) or Layer 2 (F&G), depending on asset bucket
    × technical_ema_mult       # Layer 3 (daily EMA per-asset)
```

### Examples

| Asset  | VIX regime   | Price vs EMA(200) | Combined mult | Scenario |
|--------|-------------|-------------------|---------------|----------|
| SPY    | neutral      | above EMA (long)  | 1.0 × 1.0 = **1.0×** | Full-size long |
| SPY    | risk_off     | above EMA (long)  | 0.5 × 1.0 = **0.5×** | VIX chop halves it |
| SPY    | neutral      | below EMA (long)  | 1.0 × 0.5 = **0.5×** | Counter-trend caution |
| SPY    | risk_off     | below EMA (long)  | 0.5 × 0.5 = **0.25×** | Double caution |
| BTC    | F&G greed    | above EMA (long)  | 1.0 × 1.0 = **1.0×** | Full-size long |
| BTC    | F&G greed    | above EMA (short) | 1e-9 × 0.5 ≈ **0×** | Short fully suppressed in greed |
| BTC    | F&G fear     | below EMA (short) | 1.0 × 1.0 = **1.0×** | Full-size short |

VIX (Layer 1) never applies to crypto; F&G (Layer 2) never applies to equities. Layer 3 (EMA) applies to all assets regardless of bucket.

---

## 7. Daily Operational Checklist

```
Time (ET)    Action
─────────────────────────────────────────────────────────────────
04:30 PM     VIX daily close published → refresh vix_closes.json
             load_vix_closes(force_refresh=True)

~08:00 PM    Crypto F&G daily score published → refresh cache
(01:00 UTC)  load_crypto_fg_scores(force_refresh=True)

Pre-session  Re-run candidate generation with current account equity

Pre-session  Re-build macro_state and tech_state (see Section 4)

Pre-session  Re-run report with three-layer overlay (exposure_mult=3.0)

Pre-session  Review new entries / exits from result["trades"]
             and execute in the brokerage
```

---

## 8. Data Freshness Validation

Before each live run, verify coverage:

```python
from edgar.services.sentiment_data import coverage_report
from datetime import date

today = date.today().strftime("%Y-%m-%d")
coverage_report(vix_closes, "VIX closes",  backtest_end=today)
coverage_report(crypto_fg,  "Crypto F&G",  backtest_end=today)
```

Both should report **100% coverage** up to yesterday's date. Any gap means the fetch failed — re-run before proceeding.

---

## 9. Key Parameter Summary

| Parameter | Value | Purpose |
|---|---|---|
| `exposure_mult` | **3.0** | Total portfolio leverage |
| `base_risk_pct` | 0.05 | 5% capital risked per trade |
| `fixed_stop_pct` | 0.10 | 10% ATR-based stop loss |
| `directional_volume_risk_pct` | 0.07 | Volume-adjusted risk cap |
| `channel_period` | 20 | Donchian breakout lookback |
| `trend_fast_period` | 55 | Trend filter fast EMA |
| `trend_slow_period` | 200 | Trend filter slow EMA |
| VIX risk_on threshold | **≤ 15** | Suppress shorts for non-crypto |
| VIX risk_off threshold | **≥ 25** | Halve longs for non-crypto |
| F&G greed threshold | **≥ 60** | Suppress shorts for crypto |
| F&G fear threshold | **≤ 30** | Halve longs for crypto |
| `short_risk_on_mult` | **1e-9** | Shorts fully suppressed in calm/bull environments |
| `per_asset_ema_period` | **200** | Daily EMA trend alignment filter |
| `per_asset_ema_above_long_mult` | 1.0 | Full-size longs when price above EMA |
| `per_asset_ema_above_short_mult` | 0.5 | Half-size shorts when price above EMA |
| `per_asset_ema_below_long_mult` | 0.5 | Half-size longs when price below EMA |
| `per_asset_ema_below_short_mult` | 1.0 | Full-size shorts when price below EMA |

---

## 10. Why x3 (not x2 or x4)

| | x2 | **x3** | x4 |
|---|---|---|---|
| CAGR | 81.8% | **132.7%** | 195.7% |
| Max DD | 24.7% | **36.0%** | 40.8% |
| Sharpe | 1.10 | **1.11** | 0.94 |
| Calmar | 3.31 | **3.69** | 4.80 |
| Trades | 263 | **321** | 348 |

- Sharpe stays flat (1.10 → 1.11) — risk-adjusted quality preserved
- Calmar improves 11% vs baseline
- Max DD 36% stays within institutional tolerance (< 40%)
- x4 Sharpe degrades to 0.94 — marginal entries are lower quality

*Note: numbers above are for the two-layer baseline without EMA scaling. With daily EMA(200) added, CAGR increases to ~148.4% (315 trades, PF 2.10) with no increase in max drawdown.*

---

## 11. Risk Warnings

1. **80% of days are spent below peak equity** — structural to trend-following. Do not reduce exposure during normal drawdown periods.

2. **Max single drawdown was 36%** in backtest. Live conditions (slippage, wider spreads) may push this higher. Size the account so a 45% drawdown is survivable.

3. **VIX > 25 currently** (Mar 2026): strategy is in `risk_off` for equity/gold/metals — long sizes are halved. This is by design.

4. **F&G currently at extreme fear (8–12)**: crypto longs are halved, shorts enabled at full size. Strategy is defensively positioned until F&G recovers above 30.

5. **Short suppression is intentional**: In calm/bull environments (VIX ≤ 15, F&G ≥ 60), shorts are fully suppressed (`1e-9`). This reflects the observation that most assets exhibit strong positive drift in these regimes, and false short signals in rising markets erode performance significantly.
