# VIX + VIXY Persistence Backtest

## Strategy Idea

This overlay uses:

- previous-day raw `VIX`
- current intraday `VIXY` from the most recent completed 5-minute bar

The tested normalized persistence ratio is:

`R = (VIXY_t-1bar / EMA_78_5m(VIXY)) / (VIX_T-1 / EMA_20_daily(VIX))`

Interpretation:

- `R > 1` means intraday `VIXY` is carrying more stress than the previous-day `VIX` baseline implies
- `R < 1` means the futures/ETF proxy is underconfirming the spot-vol stress

The official first pass used:

- `ratio_upper = 1.05`
- `ratio_lower = 0.95`
- `daily_stress_min_rel = 1.0`

So the persistence overlay only activates when previous-day `VIX` is at or above its own 20-day EMA.

## No-Look-Ahead Rules

- Daily `VIX` input is lagged by `1` day.
- Intraday `VIXY` uses the last completed 5-minute bar only.
- Intraday `VIXY` uses a `1` bar lag.
- Intraday signal must be no older than `60` minutes.

## Official Backtest Results

From `vix_vixy_persistence_summary.csv`:

| Variant | Return | CAGR | Max DD | PF | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 966.57% | 79.22% | 27.58% | 1.91 | 375 |
| VIXY asymmetric | 1023.67% | 81.53% | 25.49% | 2.02 | 377 |
| Persistence conservative | 918.95% | 77.21% | 28.12% | 1.88 | 376 |
| Persistence asymmetric | 878.26% | 75.44% | 27.99% | 1.86 | 377 |
| Persistence strict | 878.26% | 75.44% | 27.99% | 1.86 | 377 |

So the new persistence overlay underperformed both:

- the plain baseline
- and the simpler `VIXY asymmetric` micro overlay

## Why It Underperformed

The most important diagnostic result is this:

- `persistent_stress` triggers: `0`
- `fading_stress` triggers: `23`
- scaled trades: `13`

That means the initial parameterization almost never generated the regime we actually wanted:

- suppress longs when stress persistence is strong

Instead, it mostly generated:

- underconfirmed stress / `fading_stress`

and then reduced or removed short trades.

That turned out to be the wrong side to cut in this system.

## What The Overlay Actually Hit

Scaled trades were:

- `10` equity shorts
- `1` crypto short
- `1` gold short
- `1` metals short

In other words, the first pass mostly removed short exposure during moments where the strategy still benefited from it.

## Follow-Up Sweep

I also ran a compact follow-up sweep closer to the intended Donchian use case:

- use the persistence signal only as a long-side brake
- do not suppress shorts on `fading_stress`

Results:

| Variant | Return | CAGR | Max DD | PF |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 966.57% | 79.22% | 27.58% | 1.91 |
| Long-only persistence, `R > 1.00`, long `0.5x` | 966.02% | 79.19% | 27.58% | 1.91 |
| Long-only persistence, `R > 0.98`, long `0.5x` | 966.85% | 79.23% | 27.56% | 1.91 |

That is much better than the two-sided version, but still not a meaningful edge.

## Practical Conclusion

The concept is intellectually sound, but in this backtest it did **not** improve the strategy enough to justify replacing the simpler `VIXY` micro-regime overlay.

Best current ordering:

1. `VIXY asymmetric` micro overlay
2. baseline with no persistence ratio
3. `VIX + VIXY persistence` ratio overlay

## Recommendation

Do not promote the persistence ratio overlay into the default strategy path right now.

Keep it as a research branch only.

If we revisit it later, the next sensible directions would be:

- apply it only to equity-beta longs, not the whole New York-session book
- use it as a breakout-quality gate instead of a hard size multiplier
- add `VIXY` volume confirmation before calling a regime `persistent_stress`
- test rebased-index or z-score spread versions as secondary features, not primary governors

## Files

- `vix_vixy_persistence_summary.csv`
- `trades_baseline.csv`
- `trades_vixy_asymmetric.csv`
- `trades_persistence_conservative.csv`
- `trades_persistence_asymmetric.csv`
- `trades_persistence_strict.csv`

## Bottom Line

We developed it and backtested it.

The result was clear:

- the `VIX/VIXY` persistence ratio did not beat the simpler `VIXY` micro-regime model
- the first-pass implementation mostly suppressed the wrong trades
- even a more concept-faithful long-only version was basically flat to baseline

So the better production path remains:

- daily VIX as the macro brake
- intraday `VIXY` as the fine-grained sizing/filter layer
