# SP500 Macro Regime With QQQ Variables

## Scope

- Separate experiment folder: `reports/sp500_macro_regime_with_qqq_20260410`.
- Target/backtest proxy: `SPY` adjusted close as the tradable S&P 500 proxy.
- Compatibility note: inside the reused walk-forward engine, `qqq_close` and `qqq_fwd_*` represent the SP500/SPY target for this folder only.
- Actual QQQ inputs are included separately as `growth_qqq_*` features.
- Common walk-forward strategy start: `2014-11-03`.
- Supervised monthly prediction count: `139`.

## Full-Window Backtest

| Strategy | Final Value | XIRR | CAGR | Max DD | Avg Leverage | Risk-On Months |
|---|---:|---:|---:|---:|---:|---:|
| ensemble_majority_3x | $130,540 | 20.39% | 19.85% | -43.30% | 1.14 | 10 |
| ensemble_majority_2x | $99,514 | 17.15% | 16.67% | -33.48% | 1.07 | 10 |
| logistic_3x | $95,498 | 16.66% | 16.26% | -70.99% | 1.41 | 28 |
| logistic_2x | $89,367 | 15.87% | 15.45% | -54.60% | 1.21 | 28 |
| ensemble_blend_2x | $85,659 | 15.37% | 14.89% | -54.60% | 1.14 | 19 |
| random_forest_2x | $79,335 | 14.46% | 14.02% | -33.46% | 1.03 | 4 |
| plain_dca_1x | $73,309 | 13.52% | 13.12% | -33.46% | 1.00 |  |
| gmm_2x | $63,821 | 11.88% | 11.50% | -40.89% | 1.30 | 44 |

## Current Signal

- As of `2026-04-09`, logistic, random forest, ensemble blend, and ensemble majority all classify the SP500/SPY regime as `risk_off`.
- Logistic latest risk-off probability: `99.57%`.
- Random forest latest risk-off probability: `56.41%`.
- Ensemble blend latest risk-off probability: `77.99%`.
- SP500-named logistic traded-regime file: `compare/walkforward_sp500_logistic_traded_regimes_daily.csv`.

## QQQ Variable Check

- QQQ growth variables were included in the corrected compare run through `--extra-model-features`.
- The strongest QQQ-linked logistic coefficients were `growth_qqq_realized_vol_21d`, `growth_qqq_drawdown_252d`, `growth_qqq_126d_return`, and `growth_qqq_vs_sma200`.
- This means the corrected SP500 run is not just a relabeled QQQ run; QQQ is used as a separate growth-expression input.

## Overfit Read

- The best headline row is `ensemble_majority_3x`, but it only spends `10` months risk-on, so it is not automatically production-ready.
- Subwindow winners rotate: `ensemble_blend_2x` wins pre-COVID, `ensemble_majority_2x` wins the COVID/recovery window, and `ensemble_blend_2x` slightly wins 2022-2026.
- The safer takeaway is that SP500 as the target appears more economically sensible than QQQ for macro regimes, but the deployment candidate needs frozen thresholds and a stricter stability/holdout test before going live.
