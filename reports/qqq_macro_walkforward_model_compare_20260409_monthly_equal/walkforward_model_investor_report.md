# Investor-Friendly ML Backtest Report

## Bottom Line

- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.
- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.
- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.

## Backtest Snapshot

| Strategy | Final Value | XIRR | TWR CAGR | Max DD |
|---|---:|---:|---:|---:|
| walkforward_logistic_riskon_3x_prob_regime_dca | $1,605,812 | 31.8% | 32.7% | -69.8% |
| walkforward_logistic_riskon_2x_prob_regime_dca | $886,670 | 27.0% | 27.8% | -52.0% |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $583,501 | 23.7% | 24.6% | -58.0% |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $518,780 | 22.7% | 23.5% | -40.3% |
| plain_dca | $375,575 | 20.2% | 20.9% | -34.8% |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $369,190 | 20.1% | 21.0% | -46.9% |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $306,727 | 18.6% | 19.8% | -62.8% |

## Model Validation

These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.

| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |
|---|---|---:|---:|---:|---:|
| Jump-in | Logistic | 0.518 | 0.479 | 0.338 | 0.554 |
| Jump-in | Random Forest | 0.615 | 0.531 | 0.242 | 0.525 |
| Risk-off | Logistic | 0.541 | 0.282 | 0.522 | 0.569 |
| Risk-off | Random Forest | 0.603 | 0.376 | 0.205 | 0.564 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$1,605,812` with `-69.8%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$2,180,733` with `-69.8%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| 10Y Treasury yield level | 1.807 |
| High-yield spread 3-month change | 1.330 |
| High-yield spread | 1.238 |
| QQQ 222-day trend level | 1.160 |
| 10Y-3M curve | 0.868 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -1.753 |
| Latent sentiment | -0.961 |
| QQQ 1-year drawdown | -0.932 |
| US dollar 3-month return | -0.167 |
| Inflation 3-month change | -0.039 |

| Strongest random forest features | Importance |
|---|---:|
| Inflation YoY | 0.0303 |
| Shiller CAPE | 0.0253 |
| Inflation 3-month change | 0.0245 |
| Unemployment rate | 0.0164 |
| Latent sentiment | 0.0131 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 2.220 |
| QQQ 1-year drawdown | 0.646 |
| QQQ 65-day trend level | 0.630 |
| Latent sentiment | 0.214 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| QQQ 222-day trend level | -0.949 |
| High-yield spread 3-month change | -0.926 |
| Financial conditions level | -0.625 |
| Unemployment rate | -0.612 |
| 10Y Treasury yield level | -0.609 |

| Strongest random forest features | Importance |
|---|---:|
| QQQ vs 200-day trend | 0.0323 |
| Inflation YoY | 0.0291 |
| QQQ 1-year drawdown | 0.0194 |
| VIX level | 0.0164 |
| QQQ 1-month return | 0.0132 |

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