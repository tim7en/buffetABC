# QQQ Macro ML Regime Analysis

This is a research audit, not investment advice or a live trading recommendation.

## Method

- Daily aligned sample: `1999-03-10` to `2026-04-06`.
- Main supervised regime horizon: `63` trading days.
- CPI and unemployment are lagged by `45` calendar days before forward fill.
- OLS impact tests use standardized features and Newey-West standard errors on month-end observations.
- OLS impact tests drop high-VIF terms above `20` before significance scoring; the full VIF audit is still saved.
- ML validation is chronological with purge/embargo of overlapping forward-return windows.
- The latent sentiment variable is a black-box proxy, not an observed sentiment dataset.

## Data Sources

- QQQ parquet: `D:\buffetABC\cache\cache\cache\QQQ_daily.parquet`
- Macro parquet: `D:\buffetABC\cache\cache\macro_daily_1999.parquet`
- FRED `VIXCLS` as `vix`: `loaded`
- FRED `BAMLH0A0HYM2` as `hy_oas`: `loaded`
- FRED `NFCI` as `nfci`: `loaded`
- FRED `T10Y3M` as `t10y3m`: `loaded`

## Current Snapshot

- As of: `2026-04-06`
- QQQ adjusted close: `588.50`
- GMM regime: `risk_off`
- Latent sentiment index: `-2.65`
- External shock score: `1.13`
- Logistic current risk-off probability: `16.8%`
- Logistic current jump-in probability: `80.2%`
- Research allocation label: `risk_off_reserve_cash`
- Research target equity allocation: `25.0%`

## Strongest Significant Impact Tests

| Horizon | Feature | Coef pp / 1 sd | p-value | q-value |
|---:|---|---:|---:|---:|
| 252 | qqq_drawdown_252d | 23.12 | 0.0051 | 0.0206 |
| 252 | vix_level | 19.13 | 0.0000 | 0.0010 |
| 252 | hy_oas_level | 16.84 | 0.0216 | 0.0553 |
| 252 | qqq_vs_sma200 | -13.96 | 0.0035 | 0.0168 |
| 126 | vix_level | 13.55 | 0.0000 | 0.0001 |
| 126 | t10y3m_level | -13.41 | 0.0018 | 0.0143 |
| 252 | nfci_level | -12.56 | 0.0225 | 0.0553 |
| 252 | qqq_realized_vol_21d | -12.08 | 0.0029 | 0.0168 |
| 63 | t10y3m_level | -11.76 | 0.0000 | 0.0003 |
| 63 | curve_10y2y_level | 11.65 | 0.0000 | 0.0003 |
| 63 | vix_level | 9.53 | 0.0000 | 0.0000 |
| 126 | curve_10y2y_level | 9.39 | 0.0323 | 0.0969 |

## Sentiment Black-Box Tests

| Test | Outcome | Term | Coef | p-value | q-value |
|---|---|---|---:|---:|---:|
| forward_return_with_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0163 | 0.2879 | 0.5335 |
| forward_return_with_sentiment | qqq_fwd_63d_return | latent_sentiment_index | -0.0055 | 0.7253 | 0.8704 |
| forward_return_with_sentiment | qqq_fwd_63d_return | qqq_feedback_score | 0.0011 | 0.9304 | 0.9304 |
| forward_return_without_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0128 | 0.2145 | 0.3932 |
| forward_return_without_sentiment | qqq_fwd_63d_return | qqq_feedback_score | -0.0005 | 0.9548 | 0.9548 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | external_shock_score | -0.7051 | 0.0000 | 0.0000 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | qqq_feedback_score | 0.5270 | 0.0000 | 0.0000 |

## Holdout ML Validation

| Target | Model | Train N | Test N | AUC/R2 | MAE/Brier | Spearman/Recall |
|---|---|---:|---:|---:|---:|---:|
| qqq_fwd_63d_return | ridge | 222 | 97 | -1.642 | 0.113 | 0.078 |
| qqq_fwd_63d_return | random_forest | 222 | 97 | -0.133 | 0.080 | 0.141 |
| risk_off_target | logistic | 222 | 97 | 0.592 | 0.226 | 0.000 |
| risk_off_target | random_forest | 222 | 97 | 0.581 | 0.206 | 0.154 |
| jump_in_target | logistic | 222 | 97 | 0.557 | 0.341 | 0.902 |
| jump_in_target | random_forest | 222 | 97 | 0.617 | 0.238 | 0.244 |

## DCA Backtest

| Strategy | Final | Total Contributed | Profit/Contrib | XIRR | Max DD | Avg Allocation |
|---|---:|---:|---:|---:|---:|---:|
| Plain DCA 100% QQQ | $1,704,650 | $237,000 | 619.3% | 17.5% | -39.1% | 100.0% |
| Static 70/30 DCA | $927,642 | $237,000 | 291.4% | 12.5% | -26.1% | 70.1% |
| ML Regime DCA Cash Reserve | $859,037 | $237,000 | 262.5% | 11.8% | -18.0% | 62.8% |

## Files

- `aligned_daily_dataset.csv`
- `month_end_model_sample.csv`
- `ols_newey_west_impact.csv`
- `sentiment_mediation_tests.csv`
- `feature_correlation_spearman.csv`
- `feature_vif.csv`
- `ols_feature_vif_filter.csv`
- `gmm_regime_summary.csv`
- `shock_forward_return_tests.csv`
- `model_validation_metrics.csv`
- `model_feature_importance.csv`
- `walkforward_allocation_signal.csv`
- `dca_backtest_metrics.csv`
- `dca_equity_curves.csv`
- `dca_allocations.csv`
- `current_signal.json`
- `plots/`

## Caveats

- Significance is historical association, not proof of causality.
- FRED monthly macro data is not true point-in-time ALFRED vintage data; the release lag is a conservative approximation.
- The black-box sentiment proxy is intentionally transparent enough to audit, but it is still a proxy.
- DCA results depend on contribution timing, cash yield assumption, transaction cost, and thresholds.
- Treat allocation labels as hypotheses for review, not as automatic execution instructions.
