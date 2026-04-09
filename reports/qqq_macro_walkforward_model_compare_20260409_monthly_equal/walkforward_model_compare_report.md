# Walk-forward Model Comparison

## Scope

- Dataset used: `D:\buffetABC\reports\qqq_macro_ml_regime_analysis\aligned_daily_dataset.csv`
- Regime models compared: `gmm, logistic, random_forest`
- Common strategy start: `2009-01-02`
- Contribution cadence: `monthly` at `$100.00` per event.

## No-Lookahead Safeguards

- Daily GMM refits train only on rows strictly before each refit date.
- Supervised monthly models train only on month-end rows strictly before the prediction month.
- Supervised monthly models purge any training rows whose forward target window overlaps the prediction date.
- All daily leverage backtests trade lagged signals only; no same-day prediction is traded on the same bar.

## Coverage

- GMM first refit: `2003-10-01`
- GMM last refit: `2026-04-01`
- GMM refit count: `271`
- Random Forest first monthly prediction: `2008-12-31`
- Random Forest last monthly prediction: `2026-04-07`
- Random Forest prediction count: `209`
- Logistic first monthly prediction: `2008-12-31`
- Logistic last monthly prediction: `2026-04-07`
- Logistic prediction count: `209`

## Comparable Window Metrics

| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_logistic_riskon_3x_prob_regime_dca | $2,067,666 | 33.5% | 34.4% | -69.8% | 67.35x |
| walkforward_logistic_riskon_2x_prob_regime_dca | $1,039,309 | 28.0% | 28.6% | -52.0% | 33.85x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $378,785 | 20.0% | 20.4% | -61.0% | 12.34x |
| plain_dca | $344,251 | 19.3% | 19.6% | -34.7% | 11.21x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $343,603 | 19.3% | 19.7% | -46.9% | 11.19x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $309,287 | 18.4% | 18.9% | -77.6% | 10.07x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $286,325 | 17.8% | 18.5% | -62.8% | 9.33x |
