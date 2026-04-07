# QQQ Macro Forward Return Analysis

This is a descriptive research audit, not a forecast or investment recommendation.

## Data

- QQQ source: `D:\buffetABC\cache\cache\cache\QQQ_daily.parquet`
- Macro source: `D:\buffetABC\cache\cache\macro_daily_1999.parquet`
- Daily aligned range: `1999-03-10` to `2026-04-06`
- Month-end observations: `301`
- Macro values are forward-filled to QQQ trading days, so signals use the latest value known on or before the observation date.
- Outcomes are QQQ adjusted-close forward CAGRs over 252 and 504 trading days.
- Features use simple levels, 3-month changes, 12-month changes, and yield-curve spreads.

## Strongest Univariate Relationships

### 252 Trading Days

| Feature | Spearman | Low Third Avg | Mid Third Avg | High Third Avg | High-Low |
|---|---:|---:|---:|---:|---:|
| wti_12m_return | -0.32 | 20.6% | 15.6% | 0.3% | -20.3% |
| us10y_level | -0.32 | 19.3% | 19.7% | -0.5% | -19.8% |
| us30y_level | -0.29 | 18.7% | 17.7% | 2.2% | -16.5% |
| us2y_level | -0.23 | 17.8% | 18.3% | 2.6% | -15.2% |
| dxy_3m_return | -0.10 | 13.2% | 17.1% | 6.3% | -7.0% |
| wti_3m_return | -0.16 | 16.0% | 11.1% | 9.4% | -6.6% |
| curve_30y2y_level | 0.07 | 13.5% | 5.6% | 19.4% | 5.8% |
| us2y_3m_change_pp | -0.14 | 12.5% | 15.9% | 8.1% | -4.4% |

### 504 Trading Days

| Feature | Spearman | Low Third Avg | Mid Third Avg | High Third Avg | High-Low |
|---|---:|---:|---:|---:|---:|
| us10y_level | -0.50 | 18.1% | 19.1% | -4.7% | -22.8% |
| us30y_level | -0.50 | 18.6% | 18.2% | -4.0% | -22.7% |
| us2y_level | -0.33 | 15.4% | 19.0% | -1.7% | -17.1% |
| wti_level | 0.26 | 2.2% | 14.5% | 16.1% | 14.0% |
| wti_12m_return | -0.30 | 18.9% | 12.3% | 5.6% | -13.4% |
| curve_30y2y_level | 0.14 | 6.7% | 8.9% | 17.0% | 10.3% |
| curve_10y2y_level | 0.07 | 7.2% | 10.1% | 15.4% | 8.1% |
| curve_10y2y_12m_change_pp | -0.25 | 14.2% | 14.2% | 8.3% | -6.0% |

## Best Logical Rules

| Rule | Horizon | On Avg CAGR | Off Avg CAGR | Difference | On Count |
|---|---:|---:|---:|---:|---:|
| WTI falling YoY | 252 | 19.1% | 8.9% | 10.2% | 116 |
| 2Y yield falling YoY | 252 | 13.1% | 12.7% | 0.4% | 136 |
| DXY falling YoY | 252 | 12.4% | 13.2% | -0.8% | 137 |
| 10Y-2Y curve flattening YoY | 252 | 12.2% | 13.5% | -1.3% | 155 |
| 10Y yield rising YoY | 252 | 12.1% | 13.4% | -1.3% | 126 |
| 10Y-2Y curve steepening YoY | 252 | 12.0% | 13.5% | -1.5% | 134 |
| 10Y yield falling YoY | 252 | 12.1% | 13.8% | -1.7% | 162 |
| DXY rising YoY | 252 | 11.9% | 13.8% | -2.0% | 152 |
| WTI falling YoY | 504 | 16.9% | 7.1% | 9.8% | 116 |
| DXY rising YoY | 504 | 14.6% | 7.1% | 7.5% | 152 |
| 10Y-2Y curve flattening YoY | 504 | 13.9% | 7.7% | 6.2% | 155 |
| 10Y yield falling YoY | 504 | 12.4% | 9.1% | 3.4% | 162 |
| 2Y yield falling YoY | 504 | 12.5% | 9.6% | 2.8% | 136 |
| Rates rising and dollar rising | 504 | 12.8% | 10.4% | 2.5% | 63 |
| 2Y yield rising YoY | 504 | 12.0% | 9.8% | 2.2% | 152 |
| 10Y yield rising YoY | 504 | 11.9% | 10.1% | 1.8% | 126 |

## Robustness Notes

| Feature | Horizon | Full Spearman | 1999-2011 Spearman | 2012+ Spearman | December-Only Spearman |
|---|---:|---:|---:|---:|---:|
| us10y_level | 252 | -0.32 | -0.44 | 0.02 | -0.43 |
| us10y_level | 504 | -0.50 | -0.80 | 0.26 | -0.45 |
| us2y_level | 252 | -0.23 | -0.41 | 0.13 | -0.29 |
| us2y_level | 504 | -0.33 | -0.80 | 0.50 | -0.26 |
| wti_12m_return | 252 | -0.32 | -0.14 | -0.31 | -0.17 |
| wti_12m_return | 504 | -0.30 | -0.10 | -0.19 | -0.11 |
| dxy_12m_return | 252 | 0.00 | -0.14 | -0.11 | -0.02 |
| dxy_12m_return | 504 | 0.27 | -0.05 | 0.27 | 0.15 |
| curve_10y2y_level | 252 | 0.01 | 0.37 | -0.33 | 0.00 |
| curve_10y2y_level | 504 | 0.07 | 0.67 | -0.46 | 0.03 |

## Files

- `aligned_daily_dataset.csv`: daily aligned QQQ and macro features
- `month_end_sample.csv`: month-end sample used for the audit
- `feature_audit.csv`: univariate feature correlations and tercile buckets
- `logical_rule_audit.csv`: fixed simple rule outcomes
- `robustness_audit.csv`: era split and December-only robustness checks
