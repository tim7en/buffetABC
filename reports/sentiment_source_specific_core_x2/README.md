# Source-Specific Sentiment Sizing Review

## Test Goal

This run tested a more explicit source-specific design:

- Use `VIX` as the macro book-level risk brake.
- Use `Crypto F&G` only for direct crypto position sizing.
- Compare that against:
  - the plain baseline
  - `VIX` macro brake only
  - the earlier full average composite (`VIX + Crypto F&G + CNN F&G`)

## Result

The source-specific direct crypto sizing variant underperformed.

| Variant | Return % | CAGR % | Max DD % | PF | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (no sentiment) | 966.57 | 79.22 | 27.58 | 1.91 | 375 |
| Macro brake only: VIX | 981.45 | 79.83 | 27.58 | 1.93 | 376 |
| Macro VIX + direct crypto sizing (Crypto F&G) | 937.88 | 78.02 | 28.47 | 1.92 | 375 |
| Full average composite | 1005.73 | 80.82 | 27.23 | 1.97 | 357 |

Delta vs baseline:

- `VIX` macro only: `+14.88` return points
- `VIX + direct crypto sizing`: `-28.69` return points
- `Full average composite`: `+39.16` return points

## What Happened

The direct crypto sizing layer did what it was supposed to do mechanically, but the result was worse:

- `use_direct_bucket_sentiment_sizing = True`
- `entries_direct_sentiment_downscaled = 49`
- `avg_direct_sentiment_size_mult = 0.9633`

That means the model actively reduced crypto trade size, but those reductions did not improve the portfolio.

The biggest visible damage was in crypto PnL:

- Baseline crypto PnL: `2293.76`
- `VIX` macro only crypto PnL: `2407.19`
- `VIX + direct crypto sizing` crypto PnL: `2035.32`
- Full composite crypto PnL: `2546.76`

So the direct sizing layer cut crypto exposure often enough to reduce one of the strategy's strongest return contributors.

## Interpretation

The current evidence suggests:

- `VIX` works well as a macro brake.
- `Crypto F&G` is useful as part of the broader composite governor.
- `Crypto F&G` was **not** helpful when converted into a direct per-trade crypto downsizer with the tested settings.

Why this likely happened:

- the strategy's edge already depends heavily on large crypto trend winners
- direct sizing reduced those winners before the portfolio cap became binding
- the macro VIX layer was already doing a cleaner job of controlling overall risk

In other words:

- the book-level governor helped
- the direct crypto clipper over-filtered

## Practical Recommendation

For now, the better practical hierarchy is:

1. Keep `VIX` as the default macro brake.
2. Prefer the full average composite over the direct crypto sizing variant.
3. Do **not** promote the direct crypto-sizing experiment in its current form.

## If We Want To Keep Researching This Direction

The next sensible variants would be:

1. Use softer direct multipliers, for example `0.9 / 0.75` instead of `0.75 / 0.5`.
2. Apply direct crypto sizing only in extreme fear, not at the first threshold.
3. Restrict direct crypto sizing to the most volatile names (`SOL`, maybe `ETH`), not all crypto bucket names.
4. Test direct sentiment sizing only when the macro brake is already active.
5. Compare direct sizing against a crypto-only portfolio-cap adjustment instead of a per-trade clip.

## Implementation Note

This experiment added a true direct sizing path in the allocator:

- macro governor still controls gross book capacity
- direct bucket sentiment sizing now multiplies the raw target position size before the normal capacity clip

So this test was a real implementation test, not just a paper what-if.
