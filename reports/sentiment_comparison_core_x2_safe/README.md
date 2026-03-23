# Sentiment Governor Review For Session Turtle Core x2

## Executive Summary

This report summarizes the sentiment-governor backtest run against the current best saved baseline in the repo:

- Baseline strategy: `Session Turtle Trend Core x2 With Asset Class Caps`
- Basket: `core`
- Exposure multiplier: `2.0`
- Asset caps: `crypto=1.0`, `gold=0.8`, `metals=0.8`
- Sentiment combine mode: `average`
- Sentiment lag: `1 day`
- Thresholds: `45 / 25`
- Reduced multiplier: `1.0`
- Floor multiplier: `0.5`
- Reversal window: `10 days`
- Reversal min rise: `10 points`
- Reversal multiplier: `1.0`

Best observed result in this ordered cumulative test:

- Full composite (`VIX + Crypto F&G + CNN F&G`) produced the best overall outcome.
- Total return improved from `966.57%` to `1005.73%`.
- CAGR improved from `79.22%` to `80.82%`.
- Max realized drawdown improved from `27.58%` to `27.23%`.
- Profit factor improved from `1.91` to `1.97`.

Short answer:

- Highest overall impact: the full 3-source composite.
- Highest marginal added impact in this specific ordered test: `Crypto F&G`, when added on top of `VIX`.

Important nuance:

- This was an ordered cumulative test, not a full ablation/permutation study.
- So `Crypto F&G` had the biggest marginal contribution in the tested sequence, but that is not yet proof it is the universally most important source under all orderings.

## Coverage

All loaded sources had full backtest-window coverage for `2022-02-09` through `2026-03-02`.

- `VIX`: 100.0%
- `Crypto F&G`: 100.0%
- `CNN F&G`: 100.0%

## Results

| Variant | Return % | CAGR % | Max DD % | PF | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (no sentiment) | 966.57 | 79.22 | 27.58 | 1.91 | 375 |
| Cumulative average: VIX | 981.45 | 79.83 | 27.58 | 1.93 | 376 |
| Cumulative average: VIX + Crypto F&G | 1000.55 | 80.61 | 27.23 | 1.96 | 362 |
| Cumulative average: VIX + Crypto F&G + CNN F&G | 1005.73 | 80.82 | 27.23 | 1.97 | 357 |

Delta vs baseline:

- `VIX`: `+14.88` return points, `+0.61` CAGR points, drawdown unchanged, PF `+0.02`
- `VIX + Crypto F&G`: `+33.98` return points, `+1.39` CAGR points, drawdown `-0.35`, PF `+0.05`
- `VIX + Crypto F&G + CNN F&G`: `+39.16` return points, `+1.60` CAGR points, drawdown `-0.35`, PF `+0.06`

Marginal contribution by step in the tested order:

- Add `VIX`: `+14.88` return points
- Add `Crypto F&G` on top of `VIX`: `+19.10` return points
- Add `CNN F&G` on top of `VIX + Crypto F&G`: `+5.18` return points

Interpretation:

- `VIX` helped, but only modestly.
- `Crypto F&G` was the biggest incremental improvement in the tested sequence.
- `CNN F&G` still added value, but it was a smaller final refinement than the crypto sentiment layer.

## Did We Resize Positions Accordingly?

Yes, but with an important implementation detail:

- The current sentiment governor resizes exposure at the **portfolio gross-cap level**, not as a direct multiplier on every raw candidate position size.
- In code, sentiment is converted into `active_exposure_mult`, and then portfolio capacity is set through:
  - `portfolio_cap = capital * base_portfolio_cap_pct * active_exposure_mult`
- Actual trade notional is then clipped by available capacity:
  - `scaled_position_size = min(target_position_size, available_notional)`

Practical meaning:

- When sentiment is weak, the system allows less total gross exposure in the book.
- If the lower cap binds, positions are clipped smaller or skipped.
- If the cap does **not** bind, an individual trade can still go on at its normal candidate size.
- So this is a **book-level exposure governor**, not a guaranteed per-trade direct downsizer.

What changed in the results that confirms the governor was active:

- Trades dropped from `375` to `357` in the best composite run.
- `skipped_no_capacity` rose from `67` to `87`.
- Non-base exposure entries increased:
  - baseline: `0`
  - VIX: `4`
  - VIX + Crypto F&G: `43`
  - VIX + Crypto F&G + CNN F&G: `47`

Audit fields now written into trade output:

- `entry_exposure_mult`
- `entry_sentiment_score`

That makes it possible to inspect exactly when the governor reduced capacity.

## Forward-Looking Bias Check

The current implementation is intentionally conservative:

- Sentiment is lagged by `1 calendar day`.
- That means a trade on day `T` uses sentiment known by day `T-1`.
- This avoids using same-day daily sentiment values that may only be known after the market close.

This is especially important for:

- `VIX` daily-close-derived scores
- `CNN Fear & Greed`
- any once-per-day sentiment feed

## What Had The Highest Impact?

There are two defensible answers depending on what you mean by "impact":

### 1. Highest final impact

If "impact" means best final backtest result, then the winner was:

- `VIX + Crypto F&G + CNN F&G`

### 2. Highest marginal impact in the tested sequence

If "impact" means biggest incremental improvement when added next, then the winner was:

- `Crypto F&G`

Why:

- It delivered the largest step-up after `VIX` had already been included.
- It also coincided with lower drawdown and fewer trades, which suggests useful risk filtering rather than just extra turnover.

## Practical Approaches

### 1. Production-leaning approach

Use the full composite:

- `average(VIX, Crypto F&G, CNN F&G)`
- keep the current 1-day lag
- keep the current thresholds for now

Reason:

- It gave the best combined return, CAGR, PF, and a small drawdown improvement.

### 2. Simpler robust approach

Use only:

- `VIX + Crypto F&G`

Reason:

- Most of the improvement was already captured before adding CNN.
- This is a simpler stack with less dependency on the CNN feed.

### 3. Stronger direct sizing approach

If the goal is "resize every position directly when sentiment is weak", the current implementation is only a partial match.

Recommended enhancement:

- multiply `target_position_size` by a sentiment sizing multiplier directly
- keep the gross portfolio cap governor as a second safety layer

That would make sentiment affect both:

- book-level gross capacity
- per-trade initial notional

### 4. Better research approach

Run a proper ablation/permutation matrix:

- each source individually
- each pair
- all 3 together
- multiple source orders

Reason:

- the current run answers "what happened when we added them in this order"
- it does not fully answer "which source is intrinsically best"

### 5. Regime-aware approach

Consider source-specific usage:

- use `VIX` as the default macro risk brake
- use `Crypto F&G` more heavily for crypto-heavy exposure
- use `CNN F&G` as an equity-risk refinement layer

This may be more natural than forcing a flat equal-weight average across all assets.

## Recommended Next Steps

1. Keep the current full-composite result as the leading candidate.
2. Run standalone and pairwise ablations to isolate source importance more cleanly.
3. Test a direct per-trade sentiment sizing multiplier in addition to the current gross-cap governor.
4. Break out results by asset bucket (`crypto`, `gold`, `equity`, `metals`) to see where each sentiment source actually helps.
5. Walk-forward the thresholds and lag rather than treating `45/25` as fixed forever.

## Output Files

- `sentiment_comparison_summary.csv`
- `trades_baseline_no_sentiment.csv`
- `trades_cumulative_average_vix.csv`
- `trades_cumulative_average_vix_crypto_f_g.csv`
- `trades_cumulative_average_vix_crypto_f_g_cnn_f_g.csv`
