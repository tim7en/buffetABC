# QQQ Turtle vs DCA Backtest

This is a research backtest, not a promise of future performance.

## Data

- Source: `D:\buffetABC\cache\cache\tiingo\QQQ_5m.parquet`
- Daily bars used: `1301`
- Date range: `2021-03-12` to `2026-03-10`
- Partial days dropped when fewer than `60` 5-minute bars were present

## Assumptions

- Initial capital: `$10,000.00`
- Max leverage: `3x`
- Borrow rate on leverage above 1x: `5.50%` annual
- Trading cost: `2` bps slippage + `1` bps commission per exposure change
- Turtle entry/exit: `55`-day breakout / `20`-day exit channel
- Trend filter: close above/below `200`-day SMA
- RVOL confirmation: volume >= `1.25x` prior `40`-day average and close location threshold `0.65`
- Tactical MA DCA: cheap when `50`-day SMA / `200`-day SMA < `1`
- Shorting enabled: `False`

## Results

| Strategy | Final | CAGR | Max DD | Sharpe | vs DCA | Trades |
|---|---:|---:|---:|---:|---:|---:|
| DCA Monthly QQQ 3x | $20,596 | 15.6% | -52.4% | 0.57 | 32.9% | 0 |
| Buy & Hold QQQ 1x | $19,263 | 14.0% | -35.6% | 0.69 | 24.3% | 0 |
| Turtle ATR Pyramid 3x Cap RVOL Boost | $18,830 | 13.5% | -35.9% | 0.58 | 21.5% | 8 |
| Turtle Fixed 3x | $17,476 | 11.8% | -41.4% | 0.51 | 12.8% | 8 |
| MA50/200 Tactical Portfolio DCA 3x | $16,671 | 10.8% | -24.1% | 0.67 | 7.6% | 0 |
| Turtle Fixed 3x RVOL Filter | $15,546 | 9.2% | -33.7% | 0.45 | 0.3% | 6 |
| DCA Monthly QQQ 1x | $15,493 | 9.2% | -19.7% | 0.76 | 0.0% | 0 |
| MA50/200 Tactical Sleeve DCA 3x | $14,914 | 8.3% | -19.5% | 0.67 | -3.7% | 0 |
| Turtle ATR 3x Cap RVOL Boost | $14,533 | 7.8% | -22.7% | 0.52 | -6.2% | 8 |
| Turtle ATR 3x Cap | $13,390 | 6.0% | -20.2% | 0.51 | -13.6% | 8 |

## Files

- `metrics.csv`: numeric strategy metrics
- `equity_curves.csv`: daily equity for each benchmark and strategy
- `leverage.csv`: daily signed leverage for turtle strategies
- `trades.csv`: entry/exit log for turtle strategies
- `equity_curves.png`: visual summary
- `summary.json`: config and data metadata
- `grid_results.csv`: optional fixed-leverage parameter scan
- `tactical_ma_grid_results.csv`: optional tactical MA DCA parameter scan
