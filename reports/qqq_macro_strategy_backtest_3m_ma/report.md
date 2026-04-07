# QQQ Macro-Regime DCA and Moving-Average Strategy Backtest

This is a research backtest, not investment advice or a live trading recommendation.

## Method

- Uses the 3-month-smoothed macro risk-regime dataset from the prior audit.
- Signals are shifted by one trading day before applying returns to reduce look-ahead bias.
- Monthly DCA splits initial capital into equal monthly installments; undeployed cash earns 0%.
- Leverage is modeled as daily rebalanced exposure with borrow cost on gross exposure above 1x.
- Short exposure is charged a simple annualized short-borrow cost and can lose money quickly in rallies.

## Configuration

- Date range: `1999-03-10` to `2026-04-06`
- Initial capital: `$10,000`
- Borrow rate: `5.5%` annual
- Short borrow rate: `2.0%` annual
- Trading cost: `3.0` bps per notional exposure change
- Latest QQQ adjusted close: `588.50`
- Latest macro regime: `neutral`
- Latest risk score 0-100: `40.3`
- Latest major shock: `False`
- Latest active oil-up shock: `True`

## Ranked Results

| Strategy | Final | CAGR | Max DD | Sharpe | Avg Gross Exp | Short Days |
|---|---:|---:|---:|---:|---:|---:|
| DCA Regime Scaled Long Only | $83,028 | 8.1% | -25.6% | 0.67 | 57.9% | 0.0% |
| Regime Scaled Long Only | $197,211 | 11.6% | -48.1% | 0.67 | 91.5% | 0.0% |
| DCA Buy & Hold QQQ 1x | $100,646 | 8.9% | -34.4% | 0.66 | 57.4% | 0.0% |
| DCA Regime + SMA Long/Flat | $60,910 | 6.9% | -28.4% | 0.62 | 52.5% | 0.0% |
| Regime + SMA Long/Flat | $96,896 | 8.7% | -38.2% | 0.57 | 80.0% | 0.0% |
| DCA Buy & Hold QQQ 2x | $219,813 | 12.1% | -63.8% | 0.54 | 112.0% | 0.0% |
| Regime Only Long/Short | $127,461 | 9.9% | -41.6% | 0.51 | 112.5% | 38.7% |
| SMA200 Long/Cash | $64,308 | 7.1% | -58.6% | 0.50 | 71.3% | 0.0% |
| Buy & Hold QQQ 1x | $136,583 | 10.1% | -83.0% | 0.49 | 100.0% | 0.0% |
| SMA50/200 Trend 2x | $129,236 | 9.9% | -68.4% | 0.46 | 131.2% | 0.0% |
| DCA Regime + SMA Long/Short | $47,578 | 5.9% | -42.0% | 0.42 | 77.2% | 15.8% |
| Buy & Hold QQQ 2x | $59,136 | 6.8% | -99.0% | 0.39 | 200.0% | 0.0% |
| Regime + SMA Long/Short | $58,788 | 6.8% | -58.5% | 0.38 | 126.1% | 21.2% |
| SMA200 Long/Short 50% | $25,882 | 3.6% | -58.4% | 0.28 | 85.7% | 28.6% |

## Current Targets

| Strategy | Current Target Exposure | Prior-Day Applied Exposure |
|---|---:|---:|
| Buy & Hold QQQ 1x | 1.00x | 1.00x |
| Buy & Hold QQQ 2x | 2.00x | 2.00x |
| SMA200 Long/Cash | 0.00x | 0.00x |
| SMA200 Long/Short 50% | -0.50x | -0.50x |
| SMA50/200 Trend 2x | 0.00x | 0.00x |
| Regime Scaled Long Only | 1.00x | 1.00x |
| Regime + SMA Long/Flat | 0.00x | 0.00x |
| Regime + SMA Long/Short | 0.00x | 0.00x |
| Regime Only Long/Short | 0.75x | 0.75x |

## Interpretation

- Use the table as a hypothesis generator, not as a final allocation model.
- Compare Sharpe, max drawdown, and short-day exposure before comparing final equity.
- Leveraged strategies need stress testing beyond this simple daily-rebalanced model.
- Short variants are included because the account can short; they should be treated as risk overlays, not default DCA behavior.

## Files

- `metrics.csv`: numeric performance metrics for each strategy
- `equity_curves.csv`: daily equity curves
- `target_exposure.csv`: daily signed target exposure before one-day signal shift
- `realized_exposure.csv`: daily signed exposure actually applied in the simulation
- `current_targets.csv`: latest target exposures
- `yearly_returns.csv`: calendar-year return table
- `summary.json`: run configuration and latest regime snapshot
