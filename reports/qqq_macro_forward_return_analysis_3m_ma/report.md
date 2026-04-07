# QQQ Macro Forward Return Analysis

This is a descriptive research audit, not a forecast or investment recommendation.

## Data

- QQQ source: `/Users/timursabitov/Development/Python_Projects/buffetABC/cache/cache/cache/QQQ_daily.parquet`
- Macro source: `/Users/timursabitov/Development/Python_Projects/buffetABC/cache/cache/macro_daily_1999.parquet`
- Daily aligned range: `1999-03-10` to `2026-04-06`
- Month-end observations: `301`
- Macro values are forward-filled to QQQ trading days, so signals use the latest value known on or before the observation date.
- Outcomes are QQQ adjusted-close forward CAGRs over 252 and 504 trading days.
- Features use simple levels, 3-month changes, 12-month changes, and yield-curve spreads.
- Macro smoothing: `63` trading-day trailing moving average before feature derivation.

## Strongest Univariate Relationships

### 252 Trading Days

| Feature | Spearman | Low Third Avg | Mid Third Avg | High Third Avg | High-Low |
|---|---:|---:|---:|---:|---:|
| us10y_level | -0.34 | 18.8% | 20.3% | -2.5% | -21.3% |
| us30y_level | -0.32 | 18.2% | 19.4% | -0.9% | -19.1% |
| us2y_level | -0.23 | 16.5% | 20.1% | 0.1% | -16.4% |
| wti_12m_return | -0.29 | 18.7% | 17.5% | 2.4% | -16.4% |
| wti_3m_return | -0.24 | 17.7% | 13.0% | 4.5% | -13.2% |
| curve_30y2y_level | 0.08 | 11.5% | 3.9% | 21.1% | 9.6% |
| dxy_3m_return | -0.09 | 13.0% | 16.2% | 6.0% | -7.0% |
| wti_level | 0.08 | 7.9% | 14.4% | 14.3% | 6.3% |

### 504 Trading Days

| Feature | Spearman | Low Third Avg | Mid Third Avg | High Third Avg | High-Low |
|---|---:|---:|---:|---:|---:|
| us30y_level | -0.48 | 18.5% | 19.2% | -4.3% | -22.8% |
| us10y_level | -0.48 | 17.8% | 19.7% | -4.3% | -22.0% |
| us2y_level | -0.30 | 15.0% | 19.5% | -1.0% | -16.0% |
| wti_level | 0.28 | 2.3% | 14.1% | 16.9% | 14.7% |
| curve_30y2y_level | 0.13 | 6.7% | 9.3% | 17.2% | 10.5% |
| wti_3m_return | -0.26 | 16.1% | 11.9% | 6.2% | -9.8% |
| wti_12m_return | -0.26 | 17.4% | 13.4% | 7.8% | -9.6% |
| curve_10y2y_level | 0.07 | 6.3% | 11.0% | 15.9% | 9.6% |

## Best Logical Rules

| Rule | Horizon | On Avg CAGR | Off Avg CAGR | Difference | On Count |
|---|---:|---:|---:|---:|---:|
| WTI falling YoY | 252 | 17.8% | 9.8% | 8.0% | 116 |
| 10Y-2Y curve flattening YoY | 252 | 14.3% | 11.3% | 3.0% | 157 |
| DXY rising YoY | 252 | 13.7% | 12.1% | 1.6% | 148 |
| 2Y yield rising YoY | 252 | 13.1% | 12.6% | 0.5% | 157 |
| Rates rising, curve flattening | 252 | 13.1% | 12.7% | 0.4% | 117 |
| 10Y yield falling YoY | 252 | 12.9% | 12.8% | 0.1% | 160 |
| Rates rising and dollar rising | 252 | 12.8% | 12.9% | -0.1% | 60 |
| 10Y yield rising YoY | 252 | 12.8% | 12.9% | -0.2% | 126 |
| DXY rising YoY | 504 | 15.6% | 6.3% | 9.2% | 148 |
| 10Y-2Y curve flattening YoY | 504 | 14.7% | 6.7% | 8.0% | 157 |
| WTI falling YoY | 504 | 15.7% | 7.9% | 7.9% | 116 |
| 10Y yield falling YoY | 504 | 13.3% | 8.1% | 5.2% | 160 |
| 2Y yield rising YoY | 504 | 13.4% | 8.2% | 5.2% | 157 |
| Rates rising and dollar rising | 504 | 14.6% | 10.0% | 4.7% | 60 |
| Rates rising, curve flattening | 504 | 13.5% | 9.2% | 4.3% | 117 |
| 2Y yield falling YoY | 504 | 12.2% | 9.9% | 2.3% | 129 |

## Robustness Notes

| Feature | Horizon | Full Spearman | 1999-2011 Spearman | 2012+ Spearman | December-Only Spearman |
|---|---:|---:|---:|---:|---:|
| us10y_level | 252 | -0.34 | -0.49 | 0.06 | -0.36 |
| us10y_level | 504 | -0.48 | -0.79 | 0.32 | -0.39 |
| us2y_level | 252 | -0.23 | -0.47 | 0.22 | -0.23 |
| us2y_level | 504 | -0.30 | -0.79 | 0.57 | -0.23 |
| wti_12m_return | 252 | -0.29 | -0.10 | -0.30 | -0.34 |
| wti_12m_return | 504 | -0.26 | -0.07 | -0.15 | -0.26 |
| dxy_12m_return | 252 | 0.06 | -0.12 | 0.04 | 0.09 |
| dxy_12m_return | 504 | 0.27 | -0.06 | 0.34 | 0.25 |
| curve_10y2y_level | 252 | 0.02 | 0.43 | -0.38 | 0.03 |
| curve_10y2y_level | 504 | 0.07 | 0.68 | -0.47 | 0.06 |

## Purged Walk-Forward ML Importance

- Models: shallow random forest permutation importance and ridge regression standardized coefficients.
- Split rule: each test fold is in the future, and training rows are excluded if their forward-return window overlaps the test fold.
- Treat this as a variable screen, not a fitted trading model.

| Feature | Horizon | Combined Rank | RF Permutation Importance | Ridge Signed Coef |
|---|---:|---:|---:|---:|
| curve_10y2y_12m_change_pp | 252 | 4.0 | 0.0005 | -0.1112 |
| us2y_level | 252 | 6.5 | 0.0025 | -0.0360 |
| wti_3m_return | 252 | 7.5 | 0.0007 | -0.0417 |
| curve_30y2y_12m_change_pp | 252 | 7.5 | 0.0005 | 0.0483 |
| wti_12m_return | 252 | 8.0 | 0.0014 | -0.0253 |
| dxy_12m_return | 252 | 8.0 | 0.0007 | -0.0358 |
| us10y_12m_change_pp | 252 | 8.0 | -0.0000 | -0.1066 |
| curve_30y2y_level | 252 | 9.0 | 0.0002 | -0.0220 |
| us10y_12m_change_pp | 504 | 5.0 | 0.0002 | -0.0272 |
| us2y_level | 504 | 5.0 | 0.0002 | -0.0342 |
| us30y_level | 504 | 5.5 | 0.0000 | -0.1617 |
| us10y_level | 504 | 6.0 | 0.0000 | 0.0560 |
| wti_12m_return | 504 | 9.0 | 0.0003 | 0.0009 |
| curve_10y2y_12m_change_pp | 504 | 9.5 | 0.0000 | -0.0203 |
| us2y_12m_change_pp | 504 | 10.0 | 0.0001 | -0.0009 |
| wti_level | 504 | 10.0 | -0.0002 | -0.0557 |

### ML Fold Metrics

| Horizon | Model | Folds | Avg MAE | Avg R2 | Avg Spearman Pred/Actual |
|---|---:|---:|---:|---:|---:|
| 252 | random_forest | 5 | 0.186 | -1.36 | 0.16 |
| 252 | ridge | 5 | 0.244 | -6.57 | 0.26 |
| 504 | random_forest | 5 | 0.144 | -5.44 | -0.09 |
| 504 | ridge | 5 | 0.278 | -21.84 | 0.22 |

## Files

- `aligned_daily_dataset.csv`: daily aligned QQQ and macro features
- `month_end_sample.csv`: month-end sample used for the audit
- `feature_audit.csv`: univariate feature correlations and tercile buckets
- `logical_rule_audit.csv`: fixed simple rule outcomes
- `robustness_audit.csv`: era split and December-only robustness checks
- `ml_importance_raw.csv`: fold-level ridge and random forest importances
- `ml_importance.csv`: aggregated ML importance ranking
- `ml_fold_metrics.csv`: out-of-sample fold metrics
