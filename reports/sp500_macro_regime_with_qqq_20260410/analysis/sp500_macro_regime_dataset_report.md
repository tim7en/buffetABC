# SP500 Macro Regime Dataset With QQQ Variables

- Target/backtest proxy: `SPY` adjusted close, saved in `sp500_close`.
- Compatibility note: `qqq_close` and `qqq_fwd_*` are SP500 target columns in this folder only, so existing walk-forward code can run without a broad refactor.
- Actual QQQ inputs are stored separately as `growth_qqq_*` columns and can be passed through `--extra-model-features`.
- This keeps QQQ as a higher-beta growth expression rather than the primary regime target.

## Latest Row

- Date: `2026-04-09`
- SP500 proxy close: `679.86`
- QQQ close: `608.22`
- SP500 63d return: `-1.14%`
- QQQ 63d return: `-2.41%`

## Output Files

- `aligned_daily_dataset.csv`: SP500 target dataset plus QQQ growth variables.
- `sp500_supervised_feature_manifest.csv`: feature roles for the supervised models.
- `qqq_growth_feature_args.txt`: space-separated QQQ features for the compare script.