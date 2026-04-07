# QQQ Turtle vs DCA Backtest

This is a research backtest, not a promise of future performance.

## Data

- Source: `D:\buffetABC\cache\cache\cache\QQQ_daily.parquet`
- Daily bars used: `6810`
- Date range: `1999-03-10` to `2026-04-06`
- Daily parquet OHLC was adjusted with `adj_c / c` when `adj_c` was available

## Assumptions

- Initial capital: `$10,000.00`
- Max leverage: `3x`
- Borrow rate on leverage above 1x: `5.50%` annual
- Trading cost: `2` bps slippage + `1` bps commission per exposure change
- Turtle entry/exit: `55`-day breakout / `20`-day exit channel
- Trend filter: close above/below `200`-day SMA
- RVOL confirmation: volume >= `1.25x` prior `40`-day average and close location threshold `0.65`
- Tactical MA DCA: cheap when `60`-day SMA / `220`-day SMA < `1`
- Tactical entry ramp phase days: `0, 63, 126, 252`; `0` means immediate max leverage
- Tactical RVOL scan thresholds: `0, 1.25, 1.5`; `0` means no RVOL confirmation
- Shorting enabled: `False`

## Results

| Strategy | Final | CAGR | Max DD | Sharpe | vs DCA | Trades |
|---|---:|---:|---:|---:|---:|---:|
| DCA Monthly QQQ 3x | $245,263 | 12.5% | -82.1% | 0.49 | 143.6% | 0 |
| Buy & Hold QQQ 1x | $136,624 | 10.1% | -83.0% | 0.49 | 35.7% | 0 |
| DCA Monthly QQQ 1x | $100,677 | 8.9% | -34.4% | 0.66 | 0.0% | 0 |
| MA60/220 Tactical Sleeve DCA 3x | $91,852 | 8.5% | -34.4% | 0.65 | -8.8% | 0 |
| MA60/220 Tactical Portfolio DCA 3x | $89,377 | 8.4% | -73.4% | 0.46 | -11.2% | 0 |
| Turtle ATR 3x Cap | $21,030 | 2.8% | -42.2% | 0.27 | -79.1% | 67 |
| Turtle ATR 3x Cap RVOL Boost | $18,108 | 2.2% | -49.0% | 0.22 | -82.0% | 67 |
| Turtle ATR Pyramid 3x Cap RVOL Boost | $17,056 | 2.0% | -71.9% | 0.21 | -83.1% | 69 |
| Turtle Fixed 3x | $5,449 | -2.2% | -92.1% | 0.11 | -94.6% | 67 |
| Turtle Fixed 3x RVOL Filter | $3,744 | -3.6% | -75.1% | -0.04 | -96.3% | 32 |

## Files

- `metrics.csv`: numeric strategy metrics
- `equity_curves.csv`: daily equity for each benchmark and strategy
- `leverage.csv`: daily signed leverage for turtle strategies
- `trades.csv`: entry/exit log for turtle strategies
- `equity_curves.png`: visual summary
- `summary.json`: config and data metadata
- `grid_results.csv`: optional fixed-leverage parameter scan
- `tactical_ma_grid_results.csv`: optional tactical MA DCA parameter scan
- `post_reclaim_exit_grid_results.csv`: optional post-reclaim de-leveraging scan
