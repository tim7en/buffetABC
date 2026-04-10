# Walk-forward Model Comparison

## Scope

- Dataset used: `reports\sp500_macro_regime_with_qqq_20260410\analysis\aligned_daily_dataset.csv`
- Regime models compared: `GMM, Logistic, Random Forest, Ensemble Majority, Ensemble Blend`
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
- Ensemble Blend first monthly prediction: `2014-10-31`
- Ensemble Blend last monthly prediction: `2026-04-09`
- Ensemble Blend prediction count: `139`
- Ensemble Majority first monthly prediction: `2014-10-31`
- Ensemble Majority last monthly prediction: `2026-04-09`
- Ensemble Majority prediction count: `139`
- Logistic first monthly prediction: `2014-10-31`
- Logistic last monthly prediction: `2026-04-09`
- Logistic prediction count: `139`
- Random Forest first monthly prediction: `2014-10-31`
- Random Forest last monthly prediction: `2026-04-09`
- Random Forest prediction count: `139`

## Comparable Window Metrics

| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_ensemble_majority_riskon_3x_prob_regime_dca | $130,540 | 20.4% | 19.8% | -43.3% | 5.51x |
| walkforward_ensemble_majority_riskon_2x_prob_regime_dca | $99,514 | 17.1% | 16.7% | -33.5% | 4.20x |
| walkforward_logistic_riskon_3x_prob_regime_dca | $95,498 | 16.7% | 16.3% | -71.0% | 4.03x |
| walkforward_logistic_riskon_2x_prob_regime_dca | $89,367 | 15.9% | 15.4% | -54.6% | 3.77x |
| walkforward_ensemble_blend_riskon_3x_prob_regime_dca | $88,941 | 15.8% | 15.3% | -71.0% | 3.75x |
| walkforward_ensemble_blend_riskon_2x_prob_regime_dca | $85,659 | 15.4% | 14.9% | -54.6% | 3.61x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $85,613 | 15.4% | 14.9% | -33.5% | 3.61x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $79,335 | 14.5% | 14.0% | -33.5% | 3.35x |
| plain_dca | $73,309 | 13.5% | 13.1% | -33.5% | 3.09x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $63,821 | 11.9% | 11.5% | -40.9% | 2.69x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $51,213 | 9.3% | 8.8% | -54.1% | 2.16x |
