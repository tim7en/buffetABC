# Investor-Friendly ML Backtest Report

## Bottom Line

- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.
- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.
- Ensemble variants in this report are simple combinations of already-generated walk-forward signals; they do not add a second fitting stage.
- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.

## Backtest Snapshot

| Strategy | Final Value | XIRR | TWR CAGR | Max DD |
|---|---:|---:|---:|---:|
| walkforward_ensemble_majority_riskon_3x_prob_regime_dca | $130,540 | 20.4% | 19.8% | -43.3% |
| walkforward_ensemble_majority_riskon_2x_prob_regime_dca | $99,514 | 17.1% | 16.7% | -33.5% |
| walkforward_logistic_riskon_3x_prob_regime_dca | $95,498 | 16.7% | 16.3% | -71.0% |
| walkforward_logistic_riskon_2x_prob_regime_dca | $89,367 | 15.9% | 15.4% | -54.6% |
| walkforward_ensemble_blend_riskon_3x_prob_regime_dca | $88,941 | 15.8% | 15.3% | -71.0% |
| walkforward_ensemble_blend_riskon_2x_prob_regime_dca | $85,659 | 15.4% | 14.9% | -54.6% |
| walkforward_random_forest_riskon_3x_prob_regime_dca | $85,613 | 15.4% | 14.9% | -33.5% |
| walkforward_random_forest_riskon_2x_prob_regime_dca | $79,335 | 14.5% | 14.0% | -33.5% |
| plain_dca | $73,309 | 13.5% | 13.1% | -33.5% |
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $63,821 | 11.9% | 11.5% | -40.9% |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $51,213 | 9.3% | 8.8% | -54.1% |

## Model Validation

These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.

| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |
|---|---|---:|---:|---:|---:|
| Jump-in | Logistic | 0.692 | 0.562 | 0.250 | 0.586 |
| Jump-in | Random Forest | 0.624 | 0.474 | 0.233 | 0.514 |
| Risk-off | Logistic | 0.577 | 0.245 | 0.320 | 0.536 |
| Risk-off | Random Forest | 0.584 | 0.285 | 0.168 | 0.518 |

## Threshold Check

- The current logistic 3x setting (`risk_off=0.45`, `jump_in=0.55`) finished at `$95,498` with `-71.0%` max drawdown.
- The best nearby 3x threshold in the local grid finished at `$95,498` with `-71.0%` max drawdown.
- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.

## What Drives The Logistic Model

The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.

## What Drives Forward QQQ Returns

| Ridge features linked to stronger returns | Coef |
|---|---:|
| VIX level | 0.072 |
| Yield curve 10Y-2Y | 0.028 |
| QQQ 222-day trend level | 0.020 |
| QQQ 1-year drawdown | 0.014 |
| sector rotation 63d change | 0.007 |

| Ridge features linked to weaker returns | Coef |
|---|---:|
| High-yield spread | -0.078 |
| 10Y-3M curve | -0.064 |
| Wilshire / GDP valuation proxy | -0.047 |
| copper gold ratio level | -0.038 |
| High-yield spread 3-month change | -0.025 |

| Strongest random forest return features | Importance |
|---|---:|
| Inflation YoY | 0.0006 |
| growth qqq realized vol 21d | 0.0004 |
| copper gold ratio level | 0.0003 |
| QQQ 1-year drawdown | 0.0002 |
| VIX level | 0.0002 |

### Risk-off

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| High-yield spread | 1.817 |
| 10Y Treasury yield level | 1.300 |
| 10Y-3M curve | 1.242 |
| High-yield spread 3-month change | 1.177 |
| qqq volume | 0.980 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| VIX level | -1.225 |
| QQQ vs 200-day trend | -1.208 |
| copper gold 63d return | -1.191 |
| Inflation YoY | -0.427 |
| vix term structure | -0.358 |

| Strongest random forest features | Importance |
|---|---:|
| Inflation 3-month change | 0.0090 |
| efa 63d return | 0.0075 |
| VIX 1-month change | 0.0055 |
| Inflation YoY | 0.0054 |
| US dollar 3-month return | 0.0036 |

### Jump-in

| Logistic coefficients pushing odds higher | Coef |
|---|---:|
| VIX level | 1.774 |
| growth qqq drawdown 252d | 1.052 |
| sector rotation 63d change | 0.808 |
| Unemployment rate | 0.669 |
| growth qqq beta ratio 63d | 0.272 |

| Logistic coefficients pushing odds lower | Coef |
|---|---:|
| Wilshire / GDP valuation proxy | -1.596 |
| growth qqq realized vol 21d | -1.076 |
| Financial conditions 3-month change | -1.009 |
| 10Y-3M curve | -0.968 |
| Financial conditions level | -0.849 |

| Strongest random forest features | Importance |
|---|---:|
| QQQ 1-year drawdown | 0.0158 |
| copper gold ratio level | 0.0153 |
| Unemployment rate | 0.0103 |
| US dollar 3-month return | 0.0085 |
| VIX level | 0.0079 |

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
- `walkforward_ensemble_blend_regimes_full_common_window.png`: probability-blend ensemble chart on QQQ
- `walkforward_ensemble_majority_regimes_full_common_window.png`: majority-vote ensemble chart on QQQ