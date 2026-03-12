# Session Turtle Trend Core x2 With Asset Class Caps

This document is the canonical prompt-ready specification for the current best production strategy in this repo, based on the saved report in `reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_summary.csv`.

Use this as:
- a strategy brief for future research
- a prompt seed for another AI system
- a production spec reference when rebuilding the backtest elsewhere

It is a backtest specification, not a promise of future performance.

## Strategy Identity

- Canonical name: `Session Turtle Trend Core x2 With Asset Class Caps`
- Strategy family: multi-asset session breakout / trend-following
- Execution timeframe: `5m`
- Trend filter timeframe: `4h`
- Breakout lookback: `20` session bars
- Portfolio mode: shared-account allocator across a fixed core universe
- Production status in this repo: best current baseline among saved production-style candidates

## Why This Was Chosen

This strategy outperformed the simpler Donchian-only variants and also outperformed the later experimental overlays that added leadership-based adaptive sizing.

Saved production baseline metrics:
- Start date: `2022-02-09T19:20:00`
- End date: `2026-03-02T16:25:00`
- Initial capital: `1000.0`
- Final equity: `10665.67`
- Total return: `966.57%`
- CAGR: `79.22%`
- Max realized drawdown: `27.58%`
- Profit factor: `1.91`
- Executed trades: `375`
- Win rate: `41.33%`

Interpretation:
- This is not a high-win-rate system.
- It behaves like a classic trend system: many losses, fewer but larger winners.
- The edge comes from capturing persistent directional moves in high-beta assets and controlling portfolio concentration.

## Source Files

Primary implementation:
- `edgar/services/session_turtle_trend_strategy.py`
- `edgar/services/session_turtle_portfolio.py`
- `edgar/management/commands/generate_session_turtle_plots.py`

Saved production report:
- `reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_summary.csv`
- `reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_asset_summary.csv`
- `reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_yearly_returns.csv`

## Core Thesis

The strategy assumes:
- breakout moves that align with a higher-timeframe trend are more likely to continue
- strong breakouts with convincing volume deserve larger risk allocation
- the best returns come from a basket of high-beta assets rather than one symbol
- portfolio concentration must be limited, because the universe contains correlated assets

In plain English:
- it looks for a 20-session breakout
- only takes it if the 4-hour trend agrees
- sizes more aggressively if the breakout bar is strong and volume-confirmed
- then lets a diversified but capped portfolio decide how much of each candidate can actually be held

## Universe

### Core Universe

The production `core` basket is:

Crypto / token-like metals from Binance cache:
- `BTC-USD` at `hong_kong_open`
- `BTC-USD` at `new_york_equity_open`
- `ETH-USD` at `hong_kong_open`
- `ETH-USD` at `new_york_equity_open`
- `SOL-USD` at `hong_kong_open`
- `SOL-USD` at `new_york_equity_open`
- `PAXG-USD` at `hong_kong_open`
- `PAXG-USD` at `new_york_equity_open`

Tiingo assets at New York equity open:
- `AMZN`
- `COIN`
- `COPPER`
- `CRCL`
- `GLD`
- `HOOD`
- `INTC`
- `MSTR`
- `PLTR`
- `PPLT`
- `SLV`
- `TSLA`

### Important Clarifications

- `QQQ` and `SPY` are not part of the `core` basket. They are only in the separate `index` basket.
- ETFs are included in `core`: `GLD`, `PPLT`, and `SLV`.
- `COPPER` is included in `core`.
- The same ticker can appear as multiple session streams in the candidate set, but the portfolio layer only allows one live position per ticker at a time.

## Asset Buckets

These matter for concentration limits.

- `crypto`: `BTC-USD`, `ETH-USD`, `SOL-USD`
- `gold`: `PAXG-USD`, `GLD`
- `equity`: `AMZN`, `COIN`, `CRCL`, `HOOD`, `INTC`, `MSTR`, `PLTR`, `TSLA`
- `metals`: `COPPER`, `PPLT`, `SLV`

## Data and Clock

### Bar Source

- Binance local cache for crypto / PAXG
- Tiingo local `5m` parquet cache for equities / ETFs / metals

### Execution Frequency

- The engine evaluates every `5m` bar.

### Session Anchors

- `hong_kong_open`: `01:30 UTC`
- `new_york_equity_open`: `09:30 America/New_York`, converted to UTC

### Session Bar Construction

Raw `5m` bars are aggregated into session bars anchored to the selected session open.

These session bars are used to compute:
- 20-session breakout channels
- 10-session exit channels
- 20-period session ATR

### 4-Hour Trend Bars

The trend filter is built by aggregating raw bars into `4h` buckets.

These `4h` bars are used to compute:
- `EMA(55)`
- `EMA(200)`

## Production Parameter Set

These are the canonical settings for the best production baseline.

### Single-Asset Candidate Engine

- `interval = "5m"`
- `lookback_years = 4.1`
- `channel_period = 20`
- `exit_channel_period = 10` (derived default for channel 20)
- `atr_period = 20`
- `atr_stop_mult = 2.0`
- `fixed_stop_pct = 0.10`
- `entry_window_minutes = 480`
- `entry_buffer_bps = 0.0`
- `base_risk_pct = 0.05`
- `max_position_pct = 0.90`
- `use_volume_risk_scaling = False`
- `volume_period = 40`
- `use_directional_volume_risk_boost = True`
- `directional_volume_min_rel_volume = 1.25`
- `directional_volume_close_location_threshold = 0.65`
- `directional_volume_risk_pct = 0.07`
- `enable_pyramiding = False`
- `slippage_bps = 2.0`
- `commission_bps = 1.0`
- `allow_longs = True`
- `allow_shorts = True`
- `use_break_even_stop = False`
- `use_4h_trend_filter = True`
- `trend_fast_period = 55`
- `trend_slow_period = 200`
- `use_chandelier_exit = False`

### Shared-Account Portfolio Layer

- `basket = "core"`
- `exposure_mult = 2.0`
- `use_drawdown_governor = False`
- `base_portfolio_cap_pct = 0.90`
- `crypto_cap_mult = 1.0`
- `gold_cap_mult = 0.8`
- `metals_cap_mult = 0.8`
- `equity_cap_mult = None`

### Non-Production Research Variants

These are not the chosen baseline:
- leadership / recent-performance sizing overlay
- extended-hours protective-exit mode
- drawdown governor variants
- expanded and index baskets

## Exact Entry Logic

The strategy only enters if all of the following are true.

### 1. Enough History Exists

It must have enough completed data for:
- breakout channel
- exit channel
- ATR
- 4h EMA trend filter

### 2. Entry Must Occur Inside the Allowed Entry Window

- New entries are only allowed while `minutes_since_session_open < entry_window_minutes`
- In production this is `480` minutes

### 3. Higher-Timeframe Trend Must Agree

For longs:
- completed `4h` close must be above `EMA55`
- `EMA55` must be above `EMA200`

For shorts:
- completed `4h` close must be below `EMA55`
- `EMA55` must be below `EMA200`

This prevents taking breakouts against the higher-timeframe structure.

### 4. 20-Session Breakout Must Trigger

For longs:
- current `5m` close must close above the prior completed 20-session high
- previous `5m` close must be at or below that trigger

For shorts:
- current `5m` close must close below the prior completed 20-session low
- previous `5m` close must be at or above that trigger

This means:
- no intrabar anticipation
- no repeated same-direction firing while already above or below the level

### 5. Entry Happens On The Next Bar

If the breakout is confirmed on bar `i`, actual entry is on bar `i+1` open, adjusted for slippage.

This is important:
- signal bar confirms breakout
- execution happens on the next bar

## Volume Logic

Volume does not create the signal. It only changes risk.

### Relative Volume

- `rel_volume = current_bar_volume / SMA(volume, 40)`

### Close Location

For longs:
- `(close - low) / (high - low) >= 0.65`

For shorts:
- close must be in the bottom `35%` of the bar

### Directional Volume Confirmation

A breakout is considered volume-confirmed if:
- `rel_volume >= 1.25`
- and the bar closes near its extreme in the breakout direction

### Effect On Risk

If not volume-confirmed:
- risk target stays at `5%`

If volume-confirmed:
- risk target is boosted to `7%`

Production note:
- the strategy does not use continuous RVOL scaling in the chosen baseline
- it only uses a discrete directional boost

## Stop, Exit, and Trade Management

### Initial Stop

Production uses:
- fixed stop distance of `10%` from entry price

For longs:
- `stop = entry_price * (1 - 0.10)`

For shorts:
- `stop = entry_price * (1 + 0.10)`

### Exit Channel

With a 20-session breakout channel, the default exit channel is 10 sessions.

For longs:
- protective stop becomes the higher of:
  - current active stop
  - 10-session session-low channel

For shorts:
- protective stop becomes the lower of:
  - current active stop
  - 10-session session-high channel

In production:
- break-even stop is disabled
- chandelier exit is disabled
- pyramiding is disabled

### What This Means Practically

The trade exits when:
- the fixed/protective stop is hit
- or the shorter exit channel overtakes the stop and is hit
- or end-of-data closes the position

### Slippage and Fees

- slippage: `2 bps`
- commission: `1 bp`

Both are modeled in entry and exit calculations.

## Position Sizing

### Single-Asset Candidate Sizing

Risk amount:
- `capital * target_risk_pct`

Shares:
- `risk_amount / stop_distance`

Notional:
- `shares * entry_price`

Per-asset candidate notional cap:
- `capital * 0.90`

Because production uses a 10% fixed stop:
- a `5%` risk target roughly implies `50%` gross notional before caps
- a `7%` risk target roughly implies `70%` gross notional before caps

### Shared-Account Portfolio Scaling

Every asset is first backtested independently.

Then all candidate trades are merged chronologically into one portfolio.

This is critical:
- candidate signals are generated independently
- final execution is constrained by one shared capital pool

Portfolio cap:
- `shared_capital * 0.90 * exposure_mult`
- production uses `exposure_mult = 2.0`
- so gross portfolio cap is effectively `180%` of capital

Additional asset bucket caps:
- crypto: `90%` of capital
- gold: `72%` of capital
- metals: `72%` of capital
- equity: uncapped beyond the main portfolio cap

### Ticker Conflict Rule

If a ticker already has an open position:
- new candidates for that same ticker are skipped

This matters because:
- `BTC-USD`, `ETH-USD`, `SOL-USD`, and `PAXG-USD` exist in multiple session streams
- only one live trade per ticker can exist at a time

## Candidate Ordering and Conflict Resolution

Candidates are sorted by:
- `entry_ts`
- `combo_idx`
- `trade_idx`

At each new candidate time:
- positions whose exits are due are closed first
- same-ticker conflicts are rejected
- portfolio notional capacity is checked
- bucket cap capacity is checked
- if enough capacity exists, the candidate is accepted or scaled down

## Performance Summary By Year

- `2022`: `+18.39%`
- `2023`: `+51.72%`
- `2024`: `+230.35%`
- `2025`: `+44.38%`
- `2026` through March 2: `+24.49%`

Interpretation:
- 2024 was the breakout year
- 2025 remained strong but less explosive
- the strategy appears most effective in sustained trending regimes

## Asset Contribution Summary

From the saved production report:

- `SOL-USD`: `+1726.3426`
- `HOOD`: `+1723.3820`
- `COIN`: `+1266.6818`
- `PLTR`: `+1198.9788`
- `MSTR`: `+1100.0816`
- `TSLA`: `+642.1953`
- `SLV`: `+402.4285`
- `GLD`: `+358.8256`
- `CRCL`: `+330.4247`
- `BTC-USD`: `+313.5664`
- `ETH-USD`: `+253.8462`
- `PPLT`: `+202.9278`
- `PAXG-USD`: `+200.5485`
- `COPPER`: `+51.0890`
- `INTC`: `-3.0528`
- `AMZN`: `-102.5960`

### What The Best Assets Had In Common

The biggest winners shared these traits:
- high beta
- sustained trend persistence
- strong participation on breakout bars
- ability to keep moving after already looking extended

The strategy is not primarily finding cheap assets.
It is primarily finding assets that continue trending after a confirmed breakout.

### What Copper Did

`COPPER` was included and profitable, but it was not a main performance engine.
Its role in the production report was more diversification than dominant alpha.

## What The Strategy Is Not

It is not:
- a mean reversion strategy
- an overnight gap fade strategy
- a whole-day free-form intraday scalper
- a pure RVOL strategy
- a pure Donchian-only system

It is:
- a session-anchored breakout system
- filtered by higher-timeframe trend
- risk-boosted by breakout quality
- controlled by portfolio caps

## Regimes Where It Should Work Best

- broad, persistent trend regimes
- momentum-led tape
- leadership concentrated in high-beta instruments
- environments where breakouts continue after confirmation

## Regimes Where It Can Struggle

- violent mean-reversion conditions
- false-breakout environments
- low-liquidity or noisy price action
- regimes where trends reverse immediately after breakout
- correlation spikes where many assets fail together

## Operational Caveats

- Tiingo assets in the current repo are regular-session oriented
- the baseline does not assume true 24/7 equity trading
- extended-hours protective exits were implemented as research, but current cached Tiingo data does not materially change the baseline result because it does not yet provide the tokenized-equity style overnight feed needed to activate that mode
- leadership-based adaptive sizing was also implemented as research and did not beat the production baseline

## Canonical Prompt For Future AI Use

Copy and paste the section below into another AI system when you want it to reason about, implement, review, or extend this strategy without re-discovering the details.

---

You are working with a trading strategy called `Session Turtle Trend Core x2 With Asset Class Caps`. Treat the following as the canonical specification unless I explicitly override something.

Strategy identity:
- Multi-asset shared-account breakout / trend-following system
- Best current production baseline in the project
- Execution timeframe: 5-minute bars
- Trend filter timeframe: 4-hour bars
- Breakout channel: 20 session bars
- Exit channel: 10 session bars

Universe:
- Binance session streams:
  - BTC-USD at hong_kong_open
  - BTC-USD at new_york_equity_open
  - ETH-USD at hong_kong_open
  - ETH-USD at new_york_equity_open
  - SOL-USD at hong_kong_open
  - SOL-USD at new_york_equity_open
  - PAXG-USD at hong_kong_open
  - PAXG-USD at new_york_equity_open
- Tiingo New York equity session assets:
  - AMZN
  - COIN
  - COPPER
  - CRCL
  - GLD
  - HOOD
  - INTC
  - MSTR
  - PLTR
  - PPLT
  - SLV
  - TSLA
- QQQ and SPY are not in the core basket
- ETFs GLD, PPLT, and SLV are included in the core basket

Asset buckets:
- crypto: BTC-USD, ETH-USD, SOL-USD
- gold: PAXG-USD, GLD
- equity: AMZN, COIN, CRCL, HOOD, INTC, MSTR, PLTR, TSLA
- metals: COPPER, PPLT, SLV

Single-asset signal engine settings:
- interval = 5m
- lookback_years = 4.1
- channel_period = 20
- exit_channel_period = 10
- atr_period = 20
- fixed_stop_pct = 10%
- entry_window_minutes = 480
- base_risk_pct = 5%
- max_position_pct = 90%
- use_4h_trend_filter = true
- trend_fast_period = 55
- trend_slow_period = 200
- use_directional_volume_risk_boost = true
- directional_volume_min_rel_volume = 1.25
- directional_volume_close_location_threshold = 0.65
- directional_volume_risk_pct = 7%
- use_volume_risk_scaling = false
- enable_pyramiding = false
- use_break_even_stop = false
- use_chandelier_exit = false
- slippage_bps = 2
- commission_bps = 1

Entry logic:
- Build session bars from 5m data using the configured session open
- Build rolling 20-session highs and lows from completed session bars
- Build 4h EMA55 and EMA200 from completed 4h bars
- Long only when completed 4h close > EMA55 > EMA200
- Short only when completed 4h close < EMA55 < EMA200
- Long signal when current 5m close crosses above the prior completed 20-session high
- Short signal when current 5m close crosses below the prior completed 20-session low
- Entry executes on the next 5m bar open with slippage

Volume logic:
- rel_volume = current bar volume / SMA(volume, 40)
- directional_volume_confirmed requires rel_volume >= 1.25 and breakout bar closing near its extreme
- if not confirmed, use 5% risk
- if confirmed, use 7% risk

Exit logic:
- initial stop is fixed at 10% from entry
- long protective stop is max(active stop, 10-session low exit channel)
- short protective stop is min(active stop, 10-session high exit channel)
- no pyramiding, no break-even stop, no chandelier exit in the production baseline

Portfolio layer:
- Merge all candidate trades chronologically into one shared account
- exposure_mult = 2.0
- base_portfolio_cap_pct = 0.90
- crypto_cap_mult = 1.0
- gold_cap_mult = 0.8
- metals_cap_mult = 0.8
- equity_cap_mult = none
- one live position per ticker max
- if a candidate exceeds available portfolio or bucket capacity, scale it down proportionally

Saved production metrics:
- start = 2022-02-09T19:20:00
- end = 2026-03-02T16:25:00
- initial capital = 1000
- final equity = 10665.67
- total return = 966.57%
- CAGR = 79.22%
- max realized drawdown = 27.58%
- profit factor = 1.91
- executed trades = 375
- win rate = 41.33%

Behavioral interpretation:
- This is a session-anchored trend breakout strategy for high-beta assets
- The core edge is breakout plus 4h trend alignment plus selective volume-based risk boost
- Portfolio caps are a critical part of the production design
- Do not simplify it into "just momentum" or "just Donchian"; the portfolio construction and volume gating matter

If I ask you to modify or review this strategy:
- preserve the baseline behavior unless I explicitly request a variant
- compare any new variant against this exact production baseline
- focus on changes to risk, exits, session handling, portfolio concentration, and regime sensitivity

---

## Recommended Short Name

If you need a compact name in future prompts, use:

`Session Turtle Core x2 Capped`

## Recommended Research Labels

If you want better terminology for future variants:
- cross-sectional momentum sizing
- leadership persistence overlay
- extended-hours protective exit mode
- tokenized-equity overnight risk model
- breakout follow-through filter
