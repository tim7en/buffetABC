# Investor-Friendly ML Backtest Report

## Bottom Line

- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.
- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.
- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.

## Backtest Snapshot

| Strategy | Final Value | XIRR | TWR CAGR | Max DD |
|---|---:|---:|---:|---:|
| walkforward_logistic_riskon_3x_prob_regime_dca | $1,424,117 | 30.8% | 31.8% | -69.8% |
| walkforward_logistic_riskon_2x_prob_regime_dca | $841,171 | 26.6% | 27.4% | -52.0% |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $714,422 | 25.3% | 26.2% | -51.8% |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $568,130 | 23.5% | 24.3% | -40.7% |
| plain_dca | $375,575 | 20.2% | 20.9% | -34.8% |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $334,550 | 19.3% | 20.2% | -46.9% |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $258,982 | 17.3% | 18.5% | -62.8% |

## Model Validation

These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.

| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |
|---|---|---:|---:|---:|---:|
| Jump-in | Logistic | 0.518 | 0.466 | 0.326 | 0.573 |
| Jump-in | Random Forest | 0.583 | 0.504 | 0.246 | 0.519 |
| Risk-off | Logistic | 0.542 | 0.285 | 0.520 | 0.550 |
| Risk-off | Random Forest | 0.593 | 0.360 | 0.210 | 0.557 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$1,424,117` with `-69.8%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$1,645,899` with `-69.8%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

## What Drives Forward QQQ Returns

| Ridge features linked to stronger returns | Coef |
|---|---:|
| QQQ 65-day trend level | 0.554 |
| VIX level | 0.151 |
| Yield curve 10Y-2Y | 0.103 |
| Shiller CAPE | 0.058 |
| Latent sentiment | 0.010 |

| Ridge features linked to weaker returns | Coef |
|---|---:|
| QQQ 222-day trend level | -0.535 |
| High-yield spread | -0.194 |
| QQQ vs 200-day trend | -0.147 |
| 10Y-3M curve | -0.088 |
| Wilshire / GDP valuation proxy | -0.079 |

| Strongest random forest return features | Importance |
|---|---:|
| Inflation YoY | 0.0019 |
| VIX level | 0.0017 |
| Shiller CAPE | 0.0006 |
| QQQ 1-month realized volatility | 0.0002 |
| Unemployment rate | 0.0002 |

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| 10Y Treasury yield level | 1.882 |
| High-yield spread 3-month change | 1.371 |
| High-yield spread | 1.197 |
| QQQ 222-day trend level | 1.103 |
| 10Y-3M curve | 0.862 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -1.753 |
| Latent sentiment | -0.909 |
| QQQ 1-year drawdown | -0.781 |
| External shock score | -0.520 |
| QQQ 65-day trend level | -0.212 |

| Strongest random forest features | Importance |
|---|---:|
| Shiller CAPE | 0.0282 |
| Inflation YoY | 0.0171 |
| Inflation 3-month change | 0.0139 |
| High-yield spread 3-month change | 0.0111 |
| Unemployment rate | 0.0102 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 2.133 |
| QQQ 1-year drawdown | 0.824 |
| QQQ 65-day trend level | 0.759 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| QQQ 222-day trend level | -0.899 |
| High-yield spread 3-month change | -0.809 |
| 10Y-3M curve | -0.638 |
| Financial conditions level | -0.583 |
| QQQ 1-month realized volatility | -0.579 |

| Strongest random forest features | Importance |
|---|---:|
| QQQ vs 200-day trend | 0.0361 |
| VIX level | 0.0252 |
| QQQ 1-year drawdown | 0.0182 |
| Inflation YoY | 0.0181 |
| Financial conditions level | 0.0147 |

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