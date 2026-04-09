# Walk-forward Model Comparison

## Scope

- Dataset used: `D:\buffetABC\reports\qqq_macro_ml_regime_analysis\aligned_daily_dataset.csv`
- Regime models compared: `gmm, logistic, random_forest`
- Common strategy start: `2009-03-02`
- Contribution cadence: `monthly` at `$100.00` per event.

## No-Lookahead Safeguards

- Daily GMM refits train only on rows strictly before each refit date.
- Supervised monthly models train only on month-end rows strictly before the prediction month.
- Supervised monthly models purge any training rows whose forward target window overlaps the prediction date.
- All daily leverage backtests trade lagged signals only; no same-day prediction is traded on the same bar.

## Coverage

- GMM first refit: `2004-01-02`
- GMM last refit: `2026-04-01`
- GMM refit count: `268`
- Random Forest first monthly prediction: `2009-02-27`
- Random Forest last monthly prediction: `2026-04-07`
- Random Forest prediction count: `207`
- Logistic first monthly prediction: `2009-02-27`
- Logistic last monthly prediction: `2026-04-07`
- Logistic prediction count: `207`

## Comparable Window Metrics

| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_logistic_riskon_3x_prob_regime_dca | $1,424,117 | 30.8% | 31.8% | -69.8% | 46.69x |
| walkforward_logistic_riskon_2x_prob_regime_dca | $841,171 | 26.6% | 27.4% | -52.0% | 27.58x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $714,422 | 25.3% | 26.2% | -51.8% | 23.42x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $568,130 | 23.5% | 24.3% | -40.7% | 18.63x |
| plain_dca | $375,575 | 20.2% | 20.9% | -34.8% | 12.31x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $334,550 | 19.3% | 20.2% | -46.9% | 10.97x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $258,982 | 17.3% | 18.5% | -62.8% | 8.49x |
