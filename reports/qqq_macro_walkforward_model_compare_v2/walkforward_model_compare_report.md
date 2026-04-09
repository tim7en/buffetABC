# Walk-forward Model Comparison

## Scope

- Dataset used: `..\reports\qqq_macro_ml_regime_analysis_v2\aligned_daily_dataset.csv`
- Regime models compared: `gmm, logistic, random_forest`
- Common strategy start: `2014-11-03`
- Contribution cadence: `monthly` at `$100.00` per event.

## No-Lookahead Safeguards

- Daily GMM refits train only on rows strictly before each refit date.
- Supervised monthly models train only on month-end rows strictly before the prediction month.
- Supervised monthly models purge any training rows whose forward target window overlaps the prediction date.
- All daily leverage backtests trade lagged signals only; no same-day prediction is traded on the same bar.

## Coverage

- GMM first refit: `2009-08-03`
- GMM last refit: `2026-04-01`
- GMM refit count: `201`
- Logistic first monthly prediction: `2014-10-31`
- Logistic last monthly prediction: `2026-04-07`
- Logistic prediction count: `139`
- Random Forest first monthly prediction: `2014-10-31`
- Random Forest last monthly prediction: `2026-04-07`
- Random Forest prediction count: `139`

## Comparable Window Metrics

| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_logistic_riskon_2x_prob_regime_dca | $124,827 | 19.9% | 20.0% | -46.8% | 5.27x |
| walkforward_logistic_riskon_3x_prob_regime_dca | $124,692 | 19.9% | 20.3% | -62.8% | 5.26x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $121,710 | 19.6% | 19.1% | -62.7% | 5.14x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $121,324 | 19.5% | 19.2% | -46.8% | 5.12x |
| plain_dca | $104,371 | 17.7% | 17.5% | -33.8% | 4.40x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $103,414 | 17.6% | 17.4% | -49.1% | 4.36x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $88,777 | 15.8% | 15.4% | -65.5% | 3.75x |
