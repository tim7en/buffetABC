# Investor-Friendly ML Backtest Report

## Bottom Line

- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.
- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.
- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.

## Backtest Snapshot

| Strategy | Final Value | XIRR | TWR CAGR | Max DD |
|---|---:|---:|---:|---:|
| walkforward_logistic_riskon_3x_prob_regime_dca | $2,067,666 | 33.5% | 34.4% | -69.8% |
| walkforward_logistic_riskon_2x_prob_regime_dca | $1,039,309 | 28.0% | 28.6% | -52.0% |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $378,785 | 20.0% | 20.4% | -61.0% |
| plain_dca | $344,251 | 19.3% | 19.6% | -34.7% |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $343,603 | 19.3% | 19.7% | -46.9% |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $309,287 | 18.4% | 18.9% | -77.6% |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $286,325 | 17.8% | 18.5% | -62.8% |

## Model Validation

These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.

| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |
|---|---|---:|---:|---:|---:|
| Jump-in | Logistic | 0.558 | 0.462 | 0.340 | 0.541 |
| Jump-in | Random Forest | 0.628 | 0.540 | 0.237 | 0.535 |
| Risk-off | Logistic | 0.592 | 0.378 | 0.226 | 0.500 |
| Risk-off | Random Forest | 0.574 | 0.336 | 0.204 | 0.547 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$2,067,666` with `-69.8%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$2,539,633` with `-69.8%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| High-yield spread 3-month change | 1.293 |
| High-yield spread | 1.234 |
| 10Y Treasury yield level | 1.052 |
| Financial conditions level | 0.811 |
| QQQ 1-month realized volatility | 0.739 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -2.128 |
| QQQ 1-year drawdown | -0.960 |
| Latent sentiment | -0.656 |
| External shock score | -0.547 |
| US dollar 3-month return | -0.115 |

| Strongest random forest features | Importance |
|---|---:|
| Inflation YoY | 0.0242 |
| Unemployment rate | 0.0162 |
| Inflation 3-month change | 0.0159 |
| 10Y yield 3-month change | 0.0069 |
| Latent sentiment | 0.0050 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 2.311 |
| QQQ 1-year drawdown | 0.687 |
| Yield curve 10Y-2Y | 0.396 |
| External shock score | 0.111 |
| US dollar 3-month return | 0.069 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| High-yield spread 3-month change | -0.867 |
| Financial conditions level | -0.778 |
| 10Y-3M curve | -0.645 |
| QQQ 1-month realized volatility | -0.612 |
| Unemployment rate | -0.432 |

| Strongest random forest features | Importance |
|---|---:|
| VIX level | 0.0259 |
| QQQ vs 200-day trend | 0.0231 |
| Inflation YoY | 0.0209 |
| QQQ 1-year drawdown | 0.0209 |
| QQQ 1-month return | 0.0164 |

## Practical Take

- The logistic strategy looks more like an aggressive risk-on timing overlay than a defensive capital-preservation model.
- Plain DCA still has the cleaner risk story. Logistic 3x may be interesting for research, but its drawdown profile is severe enough that I would not present it as a conservative investor solution.
- GMM remains the more intuitive macro-regime framework, but in the current corrected run it does not match logistic on return.

## Files

- `walkforward_model_validation_metrics.csv`: purged validation scores
- `walkforward_model_feature_importance.csv`: raw model feature weights and permutation importance
- `walkforward_feature_importance_compare_risk_off.png`: risk-off feature comparison chart
- `walkforward_feature_importance_compare_jump_in.png`: jump-in feature comparison chart
- `walkforward_logistic_regimes_full_common_window.png`: logistic regime chart on QQQ