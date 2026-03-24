# Session Turtle Core x3 — Live Strategy Integration Guide

**Strategy**: Session Turtle Core x2 → upgraded to x3 exposure
**Chosen configuration**: `exposure_mult=3.0`, two-layer overlay (VIX daily + Crypto F&G)
**Basis**: Backtested Feb 2022 – Mar 2026 · 321 trades · CAGR 132.7% · Max DD 36.0% · Sharpe 1.11 · Calmar 3.69

> **Note**: VIXY intraday (5m SMA) is a backtest research layer only and is **not used in live execution**. It requires a live 5-minute data feed and real-time regime classification that is not operationally practical. The two daily-refresh layers below are sufficient for live use.

---

## 1. Architecture Overview

```
Every entry decision passes through two sequential filters:

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
  Position size = base_size × layer1_mult × layer2_mult
```

---

## 2. Regime Rules (exact thresholds)

### Layer 1 — VIX Daily Macro (equity / gold / metals)

| Prior-day VIX close | Regime      | Long mult | Short mult |
|---------------------|-------------|-----------|------------|
| ≤ 15                | `risk_on`   | **1.0×**  | **1e-9×** (suppressed) |
| 15 < VIX < 25       | `neutral`   | 1.0×      | 1.0×       |
| ≥ 25                | `risk_off`  | **0.5×**  | 1.0×       |

### Layer 2 — Crypto Fear & Greed (crypto only)

| Prior-day F&G score | Regime           | Long mult | Short mult |
|---------------------|------------------|-----------|------------|
| ≥ 60                | `greed/risk_on`  | **1.0×**  | **1e-9×** (suppressed) |
| 30 < F&G < 60       | `neutral`        | 1.0×      | 1.0×       |
| ≤ 30                | `fear/risk_off`  | **0.5×**  | 1.0×       |

**Lag policy**: Both layers use a strict 1-day lag — the signal observed on date T applies only to entries on date T+1. Up to 5-calendar-day fallback applies for weekends and holidays with no data.

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

## 4. Building the Overlay State Object

Computed once per day and passed into the report generator:

```python
import json
from pathlib import Path
from edgar.services.sentiment_data import load_vix_closes, load_crypto_fg_scores
from edgar.services.session_turtle_portfolio import build_extended_hours_proxy_state

root = Path(".")  # project root

# Refresh data (or load from cache if already fresh)
vix_closes = load_vix_closes(force_refresh=True)
crypto_fg  = load_crypto_fg_scores(force_refresh=True)

# Build combined state object
macro_state = build_extended_hours_proxy_state(
    daily_vix_closes=vix_closes,
    crypto_fg_scores=crypto_fg,
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

# Step 2: run the allocator with two-layer overlay
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

    # VIX daily macro + Crypto F&G (all sessions)
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
    extended_hours_short_risk_on_mult=1e-9,
    extended_hours_short_neutral_mult=1.0,
    extended_hours_short_risk_off_mult=1.0,
)

trades  = result["trades"]
summary = result["summary"]
```

---

## 6. Daily Operational Checklist

```
Time (ET)    Action
─────────────────────────────────────────────────────────────────
04:30 PM     VIX daily close published → refresh vix_closes.json
             load_vix_closes(force_refresh=True)

~08:00 PM    Crypto F&G daily score published → refresh cache
(01:00 UTC)  load_crypto_fg_scores(force_refresh=True)

Pre-session  Re-run candidate generation with current account equity

Pre-session  Re-run report with two-layer overlay (exposure_mult=3.0)

Pre-session  Review new entries / exits from result["trades"]
             and execute in the brokerage
```

---

## 7. Data Freshness Validation

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

## 8. Key Parameter Summary

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

---

## 9. Why x3 (not x2 or x4)

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

---

## 10. Risk Warnings

1. **80% of days are spent below peak equity** — structural to trend-following. Do not reduce exposure during normal drawdown periods.

2. **Max single drawdown was 36%** in backtest. Live conditions (slippage, wider spreads) may push this higher. Size the account so a 45% drawdown is survivable.

3. **VIX > 25 currently** (Mar 2026): strategy is in `risk_off` for equity/gold/metals — long sizes are halved. This is by design.

4. **F&G currently at extreme fear (8–12)**: crypto longs are halved, shorts enabled at full size. Strategy is defensively positioned until F&G recovers above 30.
