# VIXY Micro Backtest

## Setup

This rerun applies the `VIXY` micro-regime overlay to the current best baseline:

- strategy: `Session Turtle Trend Core x2 With Asset Class Caps`
- baseline runner path: `core x2`
- proxy source: local `cache/cache/tiingo/VIX_5m.parquet`
- proxy meaning: `VIXY`, not raw `VIX`
- bar frequency: `5m`
- freshness rule: last proxy bar must be `<= 60` minutes old
- anti-look-ahead rule: use `1` completed 5-minute bar of lag
- regime state:
  - `risk_on_micro` = `VIXY <= SMA78 <= SMA390`
  - `risk_off_micro` = `VIXY > SMA78 > SMA390`
  - otherwise `neutral_micro`

This overlay only changes sizing. It does not change the underlying entry logic.

## Main Result

Yes, the finer-grained `VIXY` overlay improved the real backtest.

| Variant | Return | CAGR | Max DD | PF | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 966.57% | 79.22% | 27.58% | 1.91 | 375 |
| VIXY conservative | 1003.81% | 80.74% | 25.49% | 1.99 | 377 |
| VIXY asymmetric | 1023.67% | 81.53% | 25.49% | 2.02 | 377 |
| VIXY strict filter | 1043.16% | 82.31% | 25.49% | 2.08 | 377 |

Relative to baseline:

- conservative: `+37.24%` return, `+1.52%` CAGR, `-2.09%` drawdown, `+0.08` PF
- asymmetric: `+57.10%` return, `+2.31%` CAGR, `-2.09%` drawdown, `+0.11` PF
- strict: `+76.59%` return, `+3.09%` CAGR, `-2.09%` drawdown, `+0.17` PF

## What The Overlay Actually Did

The overlay touched `20` executed trades in each non-baseline variant.

Fresh detected micro regimes inside the run:

- `risk_on_micro`: `39`
- `neutral_micro`: `44`
- `risk_off_micro`: `36`

Average sizing multiplier applied:

- conservative: `0.9735`
- asymmetric: `0.9655`
- strict: `0.9469`

So this is not a huge always-on reduction. It is a targeted filter that only intervened on a small slice of trades, but that slice mattered.

## Variant Definitions

`VIXY conservative`

- longs in `risk_off_micro` -> `0.5x`
- shorts in `risk_on_micro` -> `0.5x`

`VIXY asymmetric`

- longs in `risk_off_micro` -> `0.5x`
- shorts in `risk_on_micro` -> `0.0x`

`VIXY strict filter`

- longs in `risk_off_micro` -> `0.0x`
- shorts in `risk_on_micro` -> `0.0x`

## Why The Real Backtest Helped More Than The Proxy Study

The earlier trade-level study already hinted that mismatch trades were the weak ones.

The full rerun helped more because once those trades were reduced or removed:

- bad trades stopped consuming portfolio capacity
- later trades could still enter
- that capital reuse created a bigger lift than the simple trade-level approximation suggested

That is why all three live reruns beat the proxy-only what-if study.

## Practical Recommendation

I would not ship the `strict filter` as the default yet even though it won this sample.

Reason:

- it is the most brittle
- it fully deletes trades
- the current `VIXY` history is still discontinuous

The better first production candidate is `VIXY asymmetric`:

- keep longs at `0.5x` in `risk_off_micro`
- fully suppress shorts in `risk_on_micro`
- leave neutral and aligned regimes unchanged

Why this one:

- strong improvement over baseline
- smaller overfitting risk than the strict binary filter
- keeps the logic intuitive

## Important Caveat

This is still using the current partial `VIXY` cache, not a continuous full-history proxy.

So the result is promising, but it should be treated as:

- strong evidence that the overlay is worth keeping
- not final proof that the strict version should become default production behavior

## Files

- `vixy_micro_comparison_summary.csv`
- `trades_baseline.csv`
- `trades_vixy_conservative.csv`
- `trades_vixy_asymmetric.csv`
- `trades_vixy_strict_filter.csv`

## Bottom Line

Yes, we should use `VIXY` in backtests.

The finer granularity helped, the no-look-ahead version still worked, and the overlay improved:

- return
- CAGR
- profit factor
- and drawdown

The best next step is to keep the `VIXY` micro overlay in code as an optional path, and treat `VIXY asymmetric` as the best default candidate for the next round of validation.
