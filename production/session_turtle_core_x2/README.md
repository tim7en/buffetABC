# Session Turtle Core x2 Production Package

This folder is a standalone copy of the current production backtest path for the best saved strategy in the repo:

- `Session Turtle Trend Core x2 With Asset Class Caps`

It is designed to run without the Django management command path. The code still reads the existing local cache directories from the repo root.

## Files

- `run_production_backtest.py`: direct CLI runner
- `session_turtle_portfolio.py`: shared-account portfolio allocator
- `session_turtle_trend_strategy.py`: single-asset session turtle engine
- `binance_data.py`: local Binance cache loader
- `local_tiingo_data.py`: local Tiingo parquet loader
- `session_open_utils.py`: session anchor and aggregation helpers
- `indicator_helpers.py`: local EMA helper
- `strategy_helpers.py`: local ATR / stop / rolling-window helpers
- `backtest_plotting.py`: PNG report writers

## Default Production Replay

The runner defaults to the current production baseline:

- `basket=core`
- `exposure_mult=2.0`
- `crypto_cap_mult=1.0`
- `gold_cap_mult=0.8`
- `metals_cap_mult=0.8`
- `base_risk_pct=5.0`
- `fixed_stop_pct=10.0`
- `directional_volume_risk_pct=7.0`

Run it from the repo root:

```powershell
python production/session_turtle_core_x2/run_production_backtest.py
```

## Data Requirements

The package uses the existing local caches:

- Tiingo parquet under `cache/cache/tiingo/`
- Binance 5m csv.gz cache under the repo cache roots already used by the project

## Notes

- This package keeps the current research toggles for the leadership overlay and extended-hours protective exits.
- The default run is still the simpler production baseline without those research overlays enabled.
