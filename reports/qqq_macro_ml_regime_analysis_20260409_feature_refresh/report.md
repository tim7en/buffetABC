# QQQ Macro ML Regime Analysis

This is a research audit, not investment advice or a live trading recommendation.

## Method

- Daily aligned sample: `1999-03-10` to `2026-04-09`.
- Main supervised regime horizon: `63` trading days.
- CPI and unemployment are lagged by `45` calendar days before forward fill.
- Nominal GDP is lagged by `45` calendar days before forward fill.
- The annual market-cap-to-GDP anchor is lagged by `365` calendar days before forward fill.
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

- As of: `2026-04-09`
- QQQ adjusted close: `608.22`
- Full-sample descriptive GMM regime: `neutral`
- Walk-forward GMM regime used for allocation: `risk_on`
- Macro cycle classifier: `expansion` (high confidence)
- Latent sentiment index: `-2.17`
- External shock score: `1.39`
- Wilshire total-market proxy: `67945.49`
- Nominal GDP (lagged SAAR): `31442.48`
- Wilshire / GDP valuation proxy: `2.1609`
- Wilshire / GDP rolling z-score: `1.32`
- Gold price proxy: `4764.60`
- Shiller CAPE ratio: `39.14`
- Official market cap to GDP anchor: `194.89`
- Logistic current risk-off probability: `78.5%`
- Logistic current jump-in probability: `42.0%`
- Research allocation label: `risk_off_reserve_cash`
- Research target equity allocation: `25.0%`

## Strongest Significant Impact Tests

| Horizon | Feature | Coef pp / 1 sd | p-value | q-value |
|---:|---|---:|---:|---:|
| 252 | nfci_level | -16.35 | 0.0000 | 0.0000 |
| 126 | nfci_level | -14.66 | 0.0000 | 0.0003 |
| 252 | cape_level | -13.57 | 0.0000 | 0.0001 |
| 252 | t10y3m_level | -12.91 | 0.0000 | 0.0000 |
| 252 | latent_sentiment_index | 12.72 | 0.0000 | 0.0000 |
| 252 | qqq_drawdown_252d | -12.27 | 0.0001 | 0.0004 |
| 126 | qqq_drawdown_252d | -11.63 | 0.0015 | 0.0074 |
| 126 | cape_level | -9.13 | 0.0001 | 0.0014 |
| 252 | cpi_yoy_pct | -7.95 | 0.0001 | 0.0008 |
| 63 | nfci_level | -7.70 | 0.0009 | 0.0152 |
| 63 | vix_level | 7.54 | 0.0078 | 0.0514 |
| 126 | t10y3m_level | -7.07 | 0.0000 | 0.0000 |

## Sentiment Black-Box Tests

| Test | Outcome | Term | Coef | p-value | q-value |
|---|---|---|---:|---:|---:|
| forward_return_with_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0139 | 0.4068 | 0.6101 |
| forward_return_with_sentiment | qqq_fwd_63d_return | latent_sentiment_index | -0.0029 | 0.8595 | 0.9535 |
| forward_return_with_sentiment | qqq_fwd_63d_return | qqq_feedback_score | -0.0007 | 0.9535 | 0.9535 |
| forward_return_without_sentiment | qqq_fwd_63d_return | external_shock_score | -0.0125 | 0.2577 | 0.4173 |
| forward_return_without_sentiment | qqq_fwd_63d_return | qqq_feedback_score | -0.0014 | 0.8869 | 0.8869 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | external_shock_score | -0.7293 | 0.0000 | 0.0000 |
| sentiment_driver_feedback_and_shocks | latent_sentiment_index | qqq_feedback_score | 0.4424 | 0.0000 | 0.0000 |

## Holdout ML Validation

| Target | Model | Train N | Test N | AUC/R2 | MAE/Brier | Spearman/Recall |
|---|---|---:|---:|---:|---:|---:|
| qqq_fwd_63d_return | ridge | 222 | 97 | -12.035 | 0.267 | 0.153 |
| qqq_fwd_63d_return | random_forest | 222 | 97 | -0.114 | 0.079 | 0.080 |
| risk_off_target | logistic | 222 | 97 | 0.583 | 0.526 | 0.808 |
| risk_off_target | random_forest | 222 | 97 | 0.650 | 0.201 | 0.269 |
| jump_in_target | logistic | 222 | 97 | 0.605 | 0.267 | 0.488 |
| jump_in_target | random_forest | 222 | 97 | 0.602 | 0.241 | 0.073 |

## DCA Backtest

| Strategy | Final | Total Contributed | Profit/Contrib | XIRR | Max DD | Avg Allocation |
|---|---:|---:|---:|---:|---:|---:|
| Plain DCA 100% QQQ | $1,761,773 | $237,000 | 643.4% | 17.8% | -39.1% | 100.0% |
| Static 70/30 DCA | $949,448 | $237,000 | 300.6% | 12.6% | -26.1% | 70.1% |
| ML Regime DCA Cash Reserve | $619,501 | $237,000 | 161.4% | 9.0% | -12.8% | 48.4% |

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
- `analysis_variable_inventory.csv`
- `analysis_variable_inventory.md`
- `macro_cycle_daily.csv`
- `current_market_environment.csv`
- `current_market_environment.md`
- `current_signal.json`
- `plots/`

## Caveats

- Significance is historical association, not proof of causality.
- FRED monthly macro data is not true point-in-time ALFRED vintage data; the release lag is a conservative approximation.
- FRED GDP is quarterly and the timing lag is still an approximation rather than a full point-in-time vintage release history.
- Shiller CAPE comes from the downloadable Multpl table rather than a point-in-time vintage database.
- The Wilshire / GDP valuation feature is a timely proxy for the Buffett indicator, not a direct total-market-cap series.
- The official market cap to GDP series comes from the World Bank via FRED and is annual, so it is kept as a slow anchor rather than a live timing feature.
- Gold uses Yahoo Finance front-month futures, which is a liquid proxy but not a perfect spot series.
- The black-box sentiment proxy is intentionally transparent enough to audit, but it is still a proxy.
- DCA results depend on contribution timing, cash yield assumption, transaction cost, and thresholds.
- Treat allocation labels as hypotheses for review, not as automatic execution instructions.
