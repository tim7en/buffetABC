# QQQ Macro Regime and Shock Audit

This is a descriptive regime audit, not a forecast or investment recommendation.

## Method

- Macro smoothing: `63` trading-day trailing moving average.
- Outcomes: QQQ adjusted-close forward CAGRs over 252 and 504 trading days.
- Regime score direction: negative is risk-on, positive is risk-off.
- Industry-standard proxies: VIX, high-yield OAS, NFCI, 10Y-3M curve, QQQ 200D trend, QQQ 3M momentum, 10Y rate change, DXY, and WTI.
- Shock flags use unsmoothed rapid changes or threshold breaks; regime classification uses smoothed inputs.

## Current Snapshot

- As of: `2026-04-06`
- QQQ adjusted close: `588.50`
- Regime: `neutral`
- Risk score: `-0.194` on a -1 risk-on to +1 risk-off scale
- Risk score 0-100: `40.3`
- Major shock active: `False`
- Active shocks: `oil_up_shock`

## Current Forward-Return Context

| Horizon | Context | Count | Mean CAGR | Median CAGR | P25 | P75 | Positive Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 252 | regime | 99 | 13.6% | 21.2% | -6.2% | 32.6% | 73.7% |
| 252 | shock_state | 192 | 17.1% | 18.5% | 5.8% | 28.2% | 90.6% |
| 504 | regime | 98 | 9.6% | 18.8% | -1.1% | 23.1% | 72.4% |
| 504 | shock_state | 183 | 12.9% | 14.2% | 7.6% | 21.1% | 86.9% |

## Current Active Shock Context

| Horizon | Active Shock | Shock Count | Shock Avg | No-Shock Avg | Difference | Shock Positive Rate |
|---|---|---:|---:|---:|---:|---:|
| 252 | oil_up_shock | 78 | 8.2% | 14.8% | -6.6% | 65.4% |
| 504 | oil_up_shock | 78 | 6.0% | 12.6% | -6.6% | 67.9% |

## Regime Forward Returns

| Horizon | Group Type | Group | Count | Mean CAGR | Median CAGR | P25 | P75 | Positive Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 252 | risk_regime | neutral | 99 | 13.6% | 21.2% | -6.2% | 32.6% | 73.7% |
| 252 | risk_regime | risk_off | 37 | 0.1% | -12.8% | -35.4% | 33.1% | 45.9% |
| 252 | risk_regime | risk_on | 174 | 14.6% | 14.3% | 5.1% | 24.3% | 90.2% |
| 252 | risk_regime | unknown | 3 | 78.5% | 76.4% | 62.9% | 93.0% | 100.0% |
| 252 | major_shock | no_major_shock | 192 | 17.1% | 18.5% | 5.8% | 28.2% | 90.6% |
| 252 | major_shock | major_shock | 121 | 6.9% | 13.0% | -25.5% | 31.8% | 62.8% |
| 504 | risk_regime | neutral | 98 | 9.6% | 18.8% | -1.1% | 23.1% | 72.4% |
| 504 | risk_regime | risk_off | 37 | 4.5% | 5.7% | -18.4% | 33.3% | 59.5% |
| 504 | risk_regime | risk_on | 163 | 13.5% | 13.4% | 7.9% | 20.5% | 90.2% |
| 504 | risk_regime | unknown | 3 | -9.0% | -7.3% | -10.6% | -6.6% | 0.0% |
| 504 | major_shock | no_major_shock | 183 | 12.9% | 14.2% | 7.6% | 21.1% | 86.9% |
| 504 | major_shock | major_shock | 118 | 7.8% | 12.1% | -2.7% | 23.2% | 68.6% |

## Shock Forward Returns

| Horizon | Shock | Shock Count | Shock Avg | No-Shock Avg | Difference | Shock Median |
|---|---|---:|---:|---:|---:|---:|
| 252 | equity_drawdown_shock | 53 | 4.4% | 14.9% | -10.5% | 9.2% |
| 252 | credit_spread_shock | 102 | 6.2% | 16.5% | -10.4% | 14.8% |
| 252 | dollar_up_shock | 38 | 7.1% | 14.0% | -6.9% | 9.3% |
| 252 | oil_up_shock | 78 | 8.2% | 14.8% | -6.6% | 12.0% |
| 252 | financial_conditions_shock | 25 | 12.4% | 13.2% | -0.9% | 18.3% |
| 252 | curve_inversion_shock | 65 | 12.6% | 13.3% | -0.7% | 23.0% |
| 252 | rate_up_shock | 45 | 14.5% | 12.9% | 1.5% | 15.8% |
| 252 | oil_down_shock | 16 | 26.0% | 12.5% | 13.5% | 23.5% |
| 252 | volatility_shock | 32 | 30.8% | 11.2% | 19.7% | 33.5% |
| 504 | curve_inversion_shock | 55 | 4.3% | 12.4% | -8.1% | 15.4% |
| 504 | oil_up_shock | 78 | 6.0% | 12.6% | -6.6% | 9.2% |
| 504 | equity_drawdown_shock | 52 | 7.9% | 11.5% | -3.7% | 13.6% |
| 504 | credit_spread_shock | 102 | 9.1% | 11.8% | -2.6% | 16.8% |
| 504 | dollar_up_shock | 37 | 9.0% | 11.2% | -2.2% | 13.2% |
| 504 | rate_up_shock | 43 | 12.1% | 10.7% | 1.5% | 13.2% |
| 504 | financial_conditions_shock | 25 | 16.2% | 10.4% | 5.8% | 19.4% |
| 504 | volatility_shock | 32 | 22.1% | 9.6% | 12.6% | 21.4% |
| 504 | oil_down_shock | 16 | 24.2% | 10.1% | 14.0% | 28.1% |

## Recent Major Shock Month-Ends

| Date | Regime | Shock Count | Active Shocks | 1Y CAGR | 2Y CAGR |
|---|---|---:|---|---:|---:|
| 2021-04-30 | risk_on | 2 | oil_up_shock, rate_up_shock | -6.9% | -2.2% |
| 2022-02-28 | neutral | 4 | volatility_shock, equity_drawdown_shock, oil_up_shock, rate_up_shock | -15.4% | 14.2% |
| 2022-03-31 | neutral | 2 | oil_up_shock, rate_up_shock | -11.0% | 10.4% |
| 2022-04-29 | neutral | 4 | volatility_shock, equity_drawdown_shock, dollar_up_shock, rate_up_shock | 2.7% | 17.6% |
| 2022-05-31 | neutral | 2 | equity_drawdown_shock, rate_up_shock | 15.1% | 22.1% |
| 2022-06-30 | risk_off | 4 | credit_spread_shock, equity_drawdown_shock, dollar_up_shock, rate_up_shock | 33.1% | 33.3% |
| 2022-07-29 | risk_off | 2 | equity_drawdown_shock, curve_inversion_shock | 22.2% | 21.6% |
| 2022-08-31 | risk_off | 4 | equity_drawdown_shock, dollar_up_shock, rate_up_shock, curve_inversion_shock | 27.1% | 24.9% |
| 2022-09-30 | risk_off | 6 | volatility_shock, equity_drawdown_shock, oil_down_shock, dollar_up_shock, rate_up_shock, curve_inversion_shock | 33.7% | 35.2% |
| 2022-10-31 | risk_off | 3 | equity_drawdown_shock, rate_up_shock, curve_inversion_shock | 29.3% | 33.3% |
| 2022-11-30 | risk_off | 2 | rate_up_shock, curve_inversion_shock | 33.8% | 33.6% |
| 2022-12-30 | risk_off | 2 | equity_drawdown_shock, curve_inversion_shock | 50.6% | 40.4% |
| 2023-06-30 | neutral | 2 | rate_up_shock, curve_inversion_shock | 32.7% | 23.0% |
| 2023-07-31 | risk_on | 3 | oil_up_shock, rate_up_shock, curve_inversion_shock | 23.6% | 22.0% |
| 2023-09-29 | risk_on | 3 | oil_up_shock, rate_up_shock, curve_inversion_shock | 35.3% | 30.5% |
| 2023-10-31 | neutral | 2 | rate_up_shock, curve_inversion_shock | 38.8% | 33.7% |
| 2024-04-30 | risk_on | 2 | rate_up_shock, curve_inversion_shock | 15.8% |  |
| 2024-12-31 | risk_on | 2 | dollar_up_shock, rate_up_shock | 21.5% |  |
| 2025-03-31 | neutral | 2 | equity_drawdown_shock, curve_inversion_shock | 25.2% |  |
| 2025-04-30 | neutral | 2 | credit_spread_shock, curve_inversion_shock |  |  |

## Files

- `derived_daily_regime_dataset.csv`: derived regime features, scores, shocks, and forward returns
- `month_end_regime_sample.csv`: month-end sample used for historical return tracking
- `regime_forward_returns.csv`: forward returns by risk-on/neutral/risk-off and major-shock states
- `shock_forward_returns.csv`: forward returns after specific shock flags
- `recent_shocks.csv`: recent month-end rows with active major shock flags
- `current_snapshot.json`: latest regime and historical forward-return context
