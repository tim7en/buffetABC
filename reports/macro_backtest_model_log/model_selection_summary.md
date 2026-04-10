# Macro Backtest Model Log

- Generated at UTC: `2026-04-10T03:52:43+00:00`
- Source of truth CSV: `backtest_registry.csv`
- Validation CSV: `validation_registry.csv`
- This log is intended to be rerun after each macro-regime backtest so model selection is auditable.

## Current Leaders

### qqq_20260410_ensemble_refresh

- Target: `QQQ`
- Best final value: `ensemble_blend_3x` at $133,063, XIRR 20.62%, max DD -62.76%.
- Best 2x candidate: `ensemble_blend_2x` at $130,862, XIRR 20.42%, max DD -46.82%.
- Best XIRR/drawdown candidate: `ensemble_blend_2x` with score 0.436.
- Caution flags present: `severe_drawdown`

### sp500_with_qqq_20260410

- Target: `SP500_SPY`
- Best final value: `ensemble_majority_3x` at $130,540, XIRR 20.39%, max DD -43.30%.
- Best 2x candidate: `ensemble_majority_2x` at $99,514, XIRR 17.15%, max DD -33.48%.
- Best XIRR/drawdown candidate: `ensemble_majority_2x` with score 0.512.
- Caution flags present: `few_risk_on_months, severe_drawdown`

## Selection Rule Of Thumb

- Prefer candidates that beat plain DCA on XIRR and final value without severe drawdown flags.
- Prefer 2x over 3x when the XIRR improvement is small relative to drawdown increase.
- Treat `few_risk_on_months` as a fragility warning, not a rejection by itself.
- Do not promote a model solely because it wins one full-window run; check `subwindow_win_count_2x` and validation metrics.

## Refresh Command

```powershell
python tools/update_macro_backtest_log.py
```
