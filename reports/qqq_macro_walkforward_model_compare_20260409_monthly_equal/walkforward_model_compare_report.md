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
| walkforward_logistic_riskon_3x_prob_regime_dca | $1,605,812 | 31.8% | 32.7% | -69.8% | 52.65x |
| walkforward_logistic_riskon_2x_prob_regime_dca | $886,670 | 27.0% | 27.8% | -52.0% | 29.07x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $583,501 | 23.7% | 24.6% | -58.0% | 19.13x |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $518,780 | 22.7% | 23.5% | -40.3% | 17.01x |
| plain_dca | $375,575 | 20.2% | 20.9% | -34.8% | 12.31x |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $369,190 | 20.1% | 21.0% | -46.9% | 12.10x |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $306,727 | 18.6% | 19.8% | -62.8% | 10.06x |
