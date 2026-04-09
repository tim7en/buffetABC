# Investor-Friendly ML Backtest Report

## Bottom Line

- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.
- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.
- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.

## Backtest Snapshot

| Strategy | Final Value | XIRR | TWR CAGR | Max DD |
|---|---:|---:|---:|---:|
| walkforward_logistic_riskon_2x_prob_regime_dca | $124,827 | 19.9% | 20.0% | -46.8% |
| walkforward_logistic_riskon_3x_prob_regime_dca | $124,692 | 19.9% | 20.3% | -62.8% |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $121,710 | 19.6% | 19.1% | -62.7% |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $121,324 | 19.5% | 19.2% | -46.8% |
| plain_dca | $104,371 | 17.7% | 17.5% | -33.8% |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $103,414 | 17.6% | 17.4% | -49.1% |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $88,777 | 15.8% | 15.4% | -65.5% |

## Model Validation

These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.

| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |
|---|---|---:|---:|---:|---:|
| Jump-in | Logistic | 0.605 | 0.504 | 0.267 | 0.619 |
| Jump-in | Random Forest | 0.602 | 0.523 | 0.241 | 0.519 |
| Risk-off | Logistic | 0.583 | 0.325 | 0.526 | 0.545 |
| Risk-off | Random Forest | 0.650 | 0.444 | 0.201 | 0.578 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$124,692` with `-62.8%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$138,354` with `-62.8%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

## What Drives Forward QQQ Returns

| Ridge features linked to stronger returns | Coef |
|---|---:|
| QQQ 65-day trend level | 0.495 |
| VIX level | 0.167 |
| Yield curve 10Y-2Y | 0.121 |
| QQQ 1-year drawdown | 0.055 |
| US dollar 3-month return | 0.005 |

| Ridge features linked to weaker returns | Coef |
|---|---:|
| QQQ 222-day trend level | -0.476 |
| High-yield spread | -0.224 |
| QQQ vs 200-day trend | -0.146 |
| 10Y-3M curve | -0.124 |
| copper gold ratio level | -0.083 |

| Strongest random forest return features | Importance |
|---|---:|
| Inflation YoY | 0.0016 |
| VIX level | 0.0009 |
| QQQ 1-year drawdown | 0.0006 |
| Shiller CAPE | 0.0005 |
| QQQ 1-month realized volatility | 0.0004 |

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| High-yield spread | 1.546 |
| 10Y Treasury yield level | 1.371 |
| High-yield spread 3-month change | 1.320 |
| 10Y-3M curve | 1.141 |
| QQQ 222-day trend level | 1.009 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -1.340 |
| copper gold 63d return | -1.039 |
| Latent sentiment | -0.906 |
| Financial conditions 3-month change | -0.707 |
| Inflation 3-month change | -0.048 |

| Strongest random forest features | Importance |
|---|---:|
| Inflation YoY | 0.0429 |
| Latent sentiment | 0.0316 |
| Shiller CAPE | 0.0265 |
| Inflation 3-month change | 0.0196 |
| High-yield spread 3-month change | 0.0154 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 1.806 |
| QQQ 65-day trend level | 0.627 |
| sector rotation 63d change | 0.585 |
| Yield curve 10Y-2Y | 0.517 |
| QQQ 1-year drawdown | 0.481 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| 10Y-3M curve | -1.045 |
| copper gold ratio level | -0.971 |
| High-yield spread 3-month change | -0.865 |
| QQQ 222-day trend level | -0.804 |
| High-yield spread | -0.792 |

| Strongest random forest features | Importance |
|---|---:|
| Inflation YoY | 0.0199 |
| QQQ vs 200-day trend | 0.0169 |
| VIX level | 0.0167 |
| QQQ 1-year drawdown | 0.0165 |
| QQQ 1-month realized volatility | 0.0146 |

## Practical Take

- The logistic strategy looks more like an aggressive risk-on timing overlay than a defensive capital-preservation model.
- Plain DCA still has the cleaner risk story. Logistic 3x may be interesting for research, but its drawdown profile is severe enough that I would not present it as a conservative investor solution.
- GMM remains the more intuitive macro-regime framework, but in the current corrected run it does not match logistic on return.

## Files

- `walkforward_model_validation_metrics.csv`: purged validation scores
- `walkforward_model_feature_importance.csv`: raw model feature weights and permutation importance
- `walkforward_feature_importance_compare_returns.png`: forward-return feature comparison chart
- `walkforward_feature_importance_compare_risk_off.png`: risk-off feature comparison chart
- `walkforward_feature_importance_compare_jump_in.png`: jump-in feature comparison chart
- `walkforward_logistic_regimes_full_common_window.png`: logistic regime chart on QQQ