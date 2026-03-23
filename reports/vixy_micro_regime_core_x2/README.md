# VIXY Micro-Regime Overlay Study

## What Was Verified First

- The local proxy file exists at `cache/cache/tiingo/VIX_5m.parquet`.
- It is populated: `30,000` five-minute bars from `2022-06-30 18:40 UTC` through `2026-02-23 20:55 UTC`.
- Coverage is partial, not continuous:
  - proxy sessions: `387`
  - SPY sessions in the same window: `952`
  - day coverage vs SPY: `40.55%`
- The file comes in three contiguous blocks, not a full uninterrupted history:
  - `2022-06-30` to `2022-12-27`
  - `2024-07-02` to `2024-12-27`
  - `2025-08-27` to `2026-02-23`

## Proxy Definition

This is intentionally a `VIXY` study, not a raw `^VIX` study.

The current local file is named `VIX_5m.parquet`, but it is being used as `VIXY` in practice:

- `VIXY` is free and available as intraday tradable data
- it gives us 5-minute granularity instead of only a daily macro read
- it includes volume, which makes it more useful for a fine-grained stress proxy
- its adjusted history handles reverse splits and long-run decay better than a raw unadjusted ETF chart

Sanity check:

- local file close on `2026-02-23 20:55 UTC`: `28.59`
- Yahoo daily `VIXY` close on `2026-02-23`: `28.47`
- Yahoo daily `^VIX` close on `2026-02-23`: `21.01`

So the correct interpretation is:

- macro layer: daily VIX-style fear regime
- intraday layer here: `VIXY` as a tradable, volume-bearing approximation of volatility stress

## Scope Of This Study

This was an overlay study on the current best baseline:

- baseline trade log: `reports/session_turtle_trend_x2_selective_asset_caps_core/shared_account_trades.csv`
- volatility proxy: `cache/cache/tiingo/VIX_5m.parquet`
- fresh signal only:
  - `session_open == new_york_equity_open`
  - last proxy bar no older than `60` minutes
- micro trend filters:
  - short SMA = `78` bars = about `1` US session
  - long SMA = `390` bars = about `5` US sessions

Regime definitions:

- `risk_on_micro` = `proxy_close <= SMA78 <= SMA390`
- `risk_off_micro` = `proxy_close > SMA78 > SMA390`
- `neutral_micro` = everything in between

This is a trade-level proxy analysis, not a full capital-recycled portfolio re-simulation. So the overlay deltas below are useful directionally, but they are not final production backtest numbers.

Because `VIXY` is an adjusted ETF series with structural decay, the safest use is with relative features, not absolute level targets:

- price vs moving average
- moving-average slope or crossover state
- short-horizon percent change
- volume expansion during stress

That is why this study focuses on intraday regime state, not fixed absolute price thresholds.

## Coverage Impact

- baseline trades: `375`
- trades with a fresh same-session proxy read: `118`
- fresh-signal coverage of baseline trades: `31.47%`

So the signal is real enough to study, but the current cache is too sparse to make it an always-on production governor yet.

## What The Signal Said

From `regime_direction_summary.csv`:

| Direction | Regime | Trades | Win Rate | Net PnL |
| --- | --- | ---: | ---: | ---: |
| Long | risk_on_micro | 33 | 42.42% | +916.53 |
| Long | neutral_micro | 27 | 55.56% | +646.24 |
| Long | risk_off_micro | 13 | 30.77% | -106.50 |
| Short | risk_on_micro | 6 | 16.67% | -299.14 |
| Short | neutral_micro | 16 | 37.50% | +416.86 |
| Short | risk_off_micro | 23 | 43.48% | +760.39 |

Interpretation:

- Longs clearly degraded in `risk_off_micro`.
- Shorts clearly degraded in `risk_on_micro`.
- The proxy is behaving like a directional intraday bias filter, not just a generic volatility throttle.

## What Did Not Work Well

Pure 1-hour panic chasing was weak.

- `shock_up` rule = 1-hour proxy return `>= +5%`
- matched trades: `8`
- all `8` were shorts
- net PnL: `-381.23`

So a raw intraday volatility spike by itself is not a good "press the short button now" signal.

Relief was better than panic impulse:

- `relief_down` rule = 1-hour proxy return `<= -3%`
- matched trades: `5`
- all `5` were longs
- net PnL: `+219.63`

That suggests trend state is more useful than a one-bar volatility shock.

## Proxy Overlay What-If

From `overlay_proxy_summary.csv`:

| Overlay | Trade-Level Proxy PnL | Delta vs Baseline |
| --- | ---: | ---: |
| baseline_no_overlay | 9665.67 | 0.00 |
| half_long_risk_off | 9718.92 | +53.25 |
| skip_long_risk_off | 9772.17 | +106.50 |
| half_short_risk_on | 9815.24 | +149.57 |
| skip_short_risk_on | 9964.81 | +299.14 |
| half_both_mismatched | 9868.49 | +202.82 |
| half_long_risk_off_plus_skip_short_risk_on | 10018.06 | +352.39 |
| skip_both_mismatched | 10071.31 | +405.64 |

Read this carefully:

- The strongest ex-post improvement came from removing direction/regime mismatches:
  - avoid longs in `risk_off_micro`
  - avoid shorts in `risk_on_micro`
- But the strongest variant is also the easiest to overfit.
- The safer starting point is `half_both_mismatched`, not `skip_both_mismatched`.

## Recommended Fine-Grained Strategy

Use the existing daily VIX composite as the macro brake, then add `VIXY` as an intraday directional filter only for the New York equity session.

Recommended first implementation:

- Keep current daily macro governor unchanged.
- Add a `micro_vol_proxy_mult` only when a fresh same-session proxy bar exists.
- Apply it only to `new_york_equity_open` trades.
- Do not apply it to Hong Kong-open crypto logic.
- Do not apply it when the proxy bar is older than `60` minutes.

Suggested starter multipliers:

| Direction | risk_on_micro | neutral_micro | risk_off_micro |
| --- | ---: | ---: | ---: |
| Long | 1.0x | 1.0x | 0.5x |
| Short | 0.5x | 1.0x | 1.0x |

Why this version:

- It respects the directional evidence.
- It is more conservative than full skipping.
- It should be easier to validate without overfitting to the sparse proxy history.
- It uses `VIXY` in a way that is robust to adjusted-price decay: relative regime, not raw price level.

## Why VIXY Is Useful Here

`VIXY` is not a perfect substitute for spot VIX, but it has practical advantages for this strategy work:

- it is free to source
- it is available intraday
- it has volume
- it is directly aligned with short-term volatility stress

For a tactical portfolio overlay, those strengths matter more than spot-index purity.

The tradeoff is that `VIXY` has roll and decay effects, so we should avoid rules like:

- "if VIXY > fixed number, do X"

and prefer rules like:

- "if VIXY is above its own short MA and that short MA is above its longer MA, treat the tape as stressed"
- "if VIXY is falling below its own short MA and stress is easing, stop suppressing longs"

## Practical Production Rules

1. Compute the proxy regime from the most recent completed 5-minute bar only.
2. Require bar freshness `<= 60` minutes.
3. Use only prior data for the moving averages. No forward-looking reads.
4. If the proxy is missing or stale, fall back to the current daily governor only.
5. Do not use the 1-hour shock spike as a standalone short trigger.
6. Do not increase leverage just because the proxy is calm. Use it as a filter first.
7. Prefer normalized or relative `VIXY` features over absolute level thresholds because the ETF decays through time.

## Data Hygiene Before Wiring It In

Before turning this into a real strategy input:

- rename or alias the file clearly as `VIXY_5m.parquet` so the symbol meaning is not ambiguous
- build a continuous legal history, because the current file only covers `40.55%` of SPY sessions in-window
- rerun a proper shared-account backtest with the overlay inside `session_turtle_portfolio.py`
- validate that capital reuse and trade competition do not erase the trade-level proxy gains
- once coverage is complete, test adding `VIXY` volume confirmation to the regime state

## Tiingo Usage Note

This analysis used the local parquet only. It did not make any new Tiingo calls.

That matters because the current cache is discontinuous. So the next step should be:

- verify what your current Tiingo plan allows
- fill history in a plan-compliant way
- cache locally and analyze from disk, not by re-pulling during every backtest

## Files Produced

- `coverage_summary.json`
- `coverage_blocks.csv`
- `regime_direction_summary.csv`
- `overlay_proxy_summary.csv`
- `trade_entries_with_vixy_regime.csv`

## Bottom Line

Yes, the finer-grained volatility proxy contains real data and it is useful.

But the real edge here is not "buy or sell because volatility moved fast." The better use is:

- suppress longs when the proxy is in a rising stress trend
- suppress shorts when the proxy is in a calm falling trend
- keep the current daily VIX governor above it as the macro brake

So the best next production version is a `two-layer volatility control`:

- daily VIX = macro gross-cap governor
- 5-minute VIXY proxy = intraday directional sizing filter for New York equity trades
