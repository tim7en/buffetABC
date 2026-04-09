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
| Jump-in | Logistic | 0.524 | 0.457 | 0.362 | 0.523 |
| Jump-in | Random Forest | 0.616 | 0.528 | 0.239 | 0.504 |
| Risk-off | Logistic | 0.512 | 0.267 | 0.521 | 0.545 |
| Risk-off | Random Forest | 0.616 | 0.359 | 0.199 | 0.522 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$1,605,812` with `-69.8%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$2,180,733` with `-69.8%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| 10Y Treasury yield level | 1.788 |
| High-yield spread | 1.110 |
| High-yield spread 3-month change | 1.098 |
| QQQ 222-day trend level | 1.022 |
| 10Y-3M curve | 0.628 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -1.471 |
| Market cap to GDP 1-year drift | -0.871 |
| Latent sentiment | -0.707 |
| Financial conditions 3-month change | -0.621 |
| US dollar 3-month return | -0.179 |

| Strongest random forest features | Importance |
|---|---:|
| Market cap to GDP 1-year drift | 0.0462 |
| Shiller CAPE | 0.0296 |
| Inflation YoY | 0.0260 |
| Inflation 3-month change | 0.0165 |
| QQQ 1-month realized volatility | 0.0157 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 2.176 |
| Market cap to GDP | 1.258 |
| qqq volume | 0.465 |
| QQQ 1-year drawdown | 0.345 |
| High-yield spread | 0.109 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| QQQ 222-day trend level | -1.168 |
| 10Y Treasury yield level | -0.837 |
| High-yield spread 3-month change | -0.834 |
| QQQ feedback | -0.624 |
| QQQ 1-month realized volatility | -0.447 |

| Strongest random forest features | Importance |
|---|---:|
| QQQ vs 200-day trend | 0.0291 |
| Inflation YoY | 0.0225 |
| VIX level | 0.0223 |
| QQQ 1-year drawdown | 0.0159 |
| Financial conditions level | 0.0151 |

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