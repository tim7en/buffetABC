# Macro Backtest Logs

This folder records model-selection evidence for macro-regime experiments.

Files:

- `backtest_registry.csv`: full-window strategy metrics across configured experiments.
- `validation_registry.csv`: model validation metrics copied from each compare run.
- `model_selection_leaders.csv`: compact best-final, best-2x, and practical-candidate rows.
- `model_selection_summary.md`: compact leader/caution summary.
- `latest_model_selection.json`: best final-value, best 2x, and best XIRR/drawdown choices by experiment.

Refresh after rerunning backtests:

```powershell
python tools/update_macro_backtest_log.py
```