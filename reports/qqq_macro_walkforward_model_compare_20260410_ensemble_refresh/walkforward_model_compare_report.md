# Walk-forward Model Comparison

## Scope

- Dataset used: `reports\qqq_macro_ml_regime_analysis_20260409_feature_refresh\aligned_daily_dataset.csv`
- Regime models compared: `Logistic, Ensemble Blend, Random Forest, Ensemble Majority`
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
- Ensemble Blend first monthly prediction: `1999-03-31`
- Ensemble Blend last monthly prediction: `2026-04-09`
- Ensemble Blend prediction count: `326`
- Logistic first monthly prediction: `2014-10-31`
- Logistic last monthly prediction: `2026-04-09`
- Logistic prediction count: `139`
- Ensemble Majority first monthly prediction: `2014-10-31`
- Ensemble Majority last monthly prediction: `2026-04-09`
- Ensemble Majority prediction count: `139`
- Random Forest first monthly prediction: `2014-10-31`
- Random Forest last monthly prediction: `2026-04-09`
- Random Forest prediction count: `139`

## Comparable Window Metrics

| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_ensemble_blend_riskon_3x_prob_regime_dca | $133,063 | 20.6% | 20.8% | -62.8% | 5.61x |
| walkforward_ensemble_blend_riskon_2x_prob_regime_dca | $130,862 | 20.4% | 20.4% | -46.8% | 5.52x |
| walkforward_logistic_riskon_2x_prob_regime_dca | $128,950 | 20.2% | 20.3% | -46.8% | 5.44x |
| walkforward_logistic_riskon_3x_prob_regime_dca | $128,811 | 20.2% | 20.6% | -62.8% | 5.44x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $125,766 | 19.9% | 19.4% | -62.7% | 5.31x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $125,367 | 19.9% | 19.5% | -46.8% | 5.29x |
| plain_dca | $107,852 | 18.1% | 17.8% | -33.8% | 4.55x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $106,863 | 18.0% | 17.7% | -49.1% | 4.51x |
| walkforward_ensemble_majority_riskon_2x_prob_regime_dca | $100,307 | 17.2% | 17.2% | -46.8% | 4.23x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $91,738 | 16.2% | 15.7% | -65.5% | 3.87x |
| walkforward_ensemble_majority_riskon_3x_prob_regime_dca | $79,772 | 14.5% | 14.5% | -69.6% | 3.37x |
