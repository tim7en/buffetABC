# QQQ Turtle vs DCA Backtest

This is a research backtest, not a promise of future performance.

## Data

- Source: `D:\buffetABC\cache\cache\tiingo\QQQ_5m.parquet`
- Daily bars used: `1301`
- Date range: `2021-03-12` to `2026-03-10`
- Partial days dropped when fewer than `60` 5-minute bars were present

## Assumptions

- Initial capital: `$10,000.00`
- Max leverage: `5x`
- Borrow rate on leverage above 1x: `5.50%` annual
- Trading cost: `2` bps slippage + `1` bps commission per exposure change
- Turtle entry/exit: `55`-day breakout / `20`-day exit channel
- Trend filter: close above/below `200`-day SMA
- RVOL confirmation: volume >= `1.25x` prior `40`-day average and close location threshold `0.65`
- Tactical MA DCA: cheap when `60`-day SMA / `220`-day SMA < `1`
- Shorting enabled: `False`

## Results

| Strategy | Final | CAGR | Max DD | Sharpe | vs DCA | Trades |
|---|---:|---:|---:|---:|---:|---:|
| MA60/220 Tactical Portfolio DCA 5x | $23,459 | 18.6% | -35.5% | 0.85 | 51.4% | 0 |
| Turtle ATR Pyramid 5x Cap RVOL Boost | $19,347 | 14.1% | -54.9% | 0.52 | 24.9% | 8 |
| Buy & Hold QQQ 1x | $19,263 | 14.0% | -35.6% | 0.69 | 24.3% | 0 |
| DCA Monthly QQQ 5x | $19,172 | 13.9% | -73.3% | 0.51 | 23.7% | 0 |
| Turtle Fixed 5x | $18,095 | 12.6% | -62.8% | 0.48 | 16.8% | 8 |
| Turtle Fixed 5x RVOL Filter | $15,697 | 9.4% | -53.8% | 0.42 | 1.3% | 6 |
| DCA Monthly QQQ 1x | $15,493 | 9.2% | -19.7% | 0.76 | 0.0% | 0 |
| MA60/220 Tactical Sleeve DCA 5x | $15,010 | 8.5% | -19.5% | 0.65 | -3.1% | 0 |
| Turtle ATR 5x Cap RVOL Boost | $14,533 | 7.8% | -22.7% | 0.52 | -6.2% | 8 |
| Turtle ATR 5x Cap | $13,390 | 6.0% | -20.2% | 0.51 | -13.6% | 8 |

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
