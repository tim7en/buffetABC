# QQQ Macro ML Regime Analysis

This is a research audit, not investment advice or a live trading recommendation.

## Method

- Daily aligned sample: `1999-03-10` to `2026-04-07`.
- Main supervised regime horizon: `63` trading days.
- CPI and unemployment are lagged by `45` calendar days before forward fill.
- Shiller CAPE is treated as available on its stated observation date from the downloaded monthly table.
- OLS impact tests use standardized features and Newey-West standard errors on month-end observations.
- OLS impact tests drop high-VIF terms above `20` before significance scoring; the full VIF audit is still saved.
- ML validation is chronological with purge/embargo of overlapping forward-return windows.
- Allocation/backtest decisions use walk-forward logistic probabilities and walk-forward GMM regimes; the full-sample GMM remains descriptive only.
- The latent sentiment variable is a black-box proxy, not an observed sentiment dataset.

## Data Sources

- QQQ parquet: `D:\buffetABC\cache\cache\cache\QQQ_daily.parquet`
- Macro parquet: `D:\buffetABC\cache\cache\macro_daily_1999.parquet`
- FRED `VIXCLS` as `vix`: `loaded`
- FRED `BAMLH0A0HYM2` as `hy_oas`: `loaded`
- FRED `NFCI` as `nfci`: `loaded`
- FRED `T10Y3M` as `t10y3m`: `loaded`

## Current Snapshot

- As of: `2026-04-07`
- QQQ adjusted close: `588.59`
- Full-sample descriptive GMM regime: `risk_off`
- Walk-forward GMM regime used for allocation: `risk_on`
- Latent sentiment index: `-2.49`
- External shock score: `0.93`
- Gold price proxy: `4657.10`
- Shiller CAPE ratio: `39.20`
- Logistic current risk-off probability: `22.3%`
- Logistic current jump-in probability: `77.1%`
- Research allocation label: `risk_off_reserve_cash`
- Research target equity allocation: `25.0%`

## Strongest Significant Impact Tests

| Horizon | Feature | Coef pp / 1 sd | p-value | q-value |
|---:|---|---:|---:|---:|
| 252 | cape_level | -28.75 | 0.0000 | 0.0000 |
| 252 | vix_level | 15.46 | 0.0000 | 0.0000 |
| 126 | cape_level | -15.25 | 0.0000 | 0.0000 |
| 252 | qqq_sma222 | 13.59 | 0.0000 | 0.0000 |
| 252 | nfci_level | -13.43 | 0.0000 | 0.0000 |
| 126 | vix_level | 10.32 | 0.0002 | 0.0015 |
| 252 | unemployment_rate_pct | -9.82 | 0.0004 | 0.0016 |
| 252 | qqq_vs_sma200 | -9.47 | 0.0493 | 0.0832 |
| 252 | t10y3m_level | -9.02 | 0.0001 | 0.0008 |
| 126 | nfci_level | -8.77 | 0.0003 | 0.0015 |
| 63 | cape_level | -8.24 | 0.0000 | 0.0003 |
| 126 | qqq_sma222 | 7.61 | 0.0000 | 0.0002 |

## Sentiment Black-Box Tests

| Test | Outcome | Term | Coef | p-value | q-value |
|---|---|---|---:|---:|---:|
| forward_return_with_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0163 | 0.2879 | 0.5335 |
| forward_return_with_sentiment | qqq_fwd_63d_return | latent_sentiment_index | -0.0055 | 0.7253 | 0.8704 |
| forward_return_with_sentiment | qqq_fwd_63d_return | qqq_feedback_score | 0.0011 | 0.9304 | 0.9304 |
| forward_return_without_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0128 | 0.2145 | 0.3932 |
| forward_return_without_sentiment | qqq_fwd_63d_return | qqq_feedback_score | -0.0005 | 0.9548 | 0.9548 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | external_shock_score | -0.7035 | 0.0000 | 0.0000 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | qqq_feedback_score | 0.5273 | 0.0000 | 0.0000 |

## Holdout ML Validation

| Target | Model | Train N | Test N | AUC/R2 | MAE/Brier | Spearman/Recall |
|---|---|---:|---:|---:|---:|---:|
| qqq_fwd_63d_return | ridge | 222 | 97 | -20.427 | 0.309 | 0.127 |
| qqq_fwd_63d_return | random_forest | 222 | 97 | -0.232 | 0.083 | 0.075 |
| risk_off_target | logistic | 222 | 97 | 0.541 | 0.522 | 0.885 |
| risk_off_target | random_forest | 222 | 97 | 0.603 | 0.205 | 0.269 |
| jump_in_target | logistic | 222 | 97 | 0.518 | 0.338 | 0.268 |
| jump_in_target | random_forest | 222 | 97 | 0.615 | 0.242 | 0.122 |

## DCA Backtest

| Strategy | Final | Total Contributed | Profit/Contrib | XIRR | Max DD | Avg Allocation |
|---|---:|---:|---:|---:|---:|---:|
| Plain DCA 100% QQQ | $1,704,911 | $237,000 | 619.4% | 17.5% | -39.1% | 100.0% |
| Static 70/30 DCA | $927,741 | $237,000 | 291.5% | 12.5% | -26.1% | 70.1% |
| ML Regime DCA Cash Reserve | $525,934 | $237,000 | 121.9% | 7.5% | -12.7% | 45.3% |

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
- Shiller CAPE comes from the downloadable Multpl table rather than a point-in-time vintage database.
- Gold uses Yahoo Finance front-month futures, which is a liquid proxy but not a perfect spot series.
- The black-box sentiment proxy is intentionally transparent enough to audit, but it is still a proxy.
- DCA results depend on contribution timing, cash yield assumption, transaction cost, and thresholds.
- Treat allocation labels as hypotheses for review, not as automatic execution instructions.
