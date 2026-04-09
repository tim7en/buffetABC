# Walk-forward GMM Backtest Audit

## Scope

- Dataset used: `reports/qqq_macro_ml_regime_analysis_audit_20260409/aligned_daily_dataset.csv`
- Contribution cadence: `monthly` at `$100.00` per event.
- Backtest under audit: dedicated daily walk-forward GMM leverage script.

## Findings

- PASS: walk-forward GMM refits train only on rows strictly before each refit date.
- PASS: predictions are limited to the current month of each refit and then traded with a one-day lag.
- PASS: leverage backtest uses `wf_gmm_regime_signal_lag1`, not the full-sample descriptive GMM labels.
- NOTE: the analysis dataset contains full-sample descriptive GMM fields for reporting only.
- NOTE: black-box PCA fields were audited separately; overlap with traded GMM features is `none`.

## Refit Coverage

- First refit: `2003-10-01` trained through `2003-09-30`.
- Last refit: `2026-04-01` trained through `2026-03-31`.
- Refit count: `271`.

## Comparable Window Metrics

| Strategy | Final Value | TWR | TWR CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca | $308,027 | 1354.3% | 15.3% | -47.5% | 9.42x |
| plain_dca | $300,668 | 1346.5% | 15.2% | -47.4% | 9.19x |
| walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca | $267,664 | 1101.1% | 14.1% | -53.1% | 8.19x |
