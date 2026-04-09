# Macro Regime Edge Audit

## Anti-Leakage Discipline

- QQQ prices are aligned on actual trading dates only.
- Monthly supervised models train only on earlier month-end observations.
- Any row whose forward target window overlaps the prediction date is purged from training.
- Allocation and leverage simulations trade lagged regime signals only.
- Quarterly GDP and annual market-cap-to-GDP anchor data are lagged before forward-fill, so the proxy is not using future macro releases.

## Current Read

- As of `2026-04-07` the macro cycle is `expansion` with `medium` confidence.
- Expansion / late-cycle / contraction scores are `4` / `3` / `0`.
- The Wilshire / GDP proxy is `2.1079` with rolling z-score `1.08`.
- The traded regime is `risk_off` with target allocation `25%`.

## Regime Accuracy

| Model | Label | Precision | Recall | Balanced Accuracy | False Positive Rate | Overall Accuracy |
|---|---|---:|---:|---:|---:|---:|
| consensus | neutral | 0.400 | 0.322 | 0.494 | 0.333 | 0.296 |
| consensus | risk_off | 0.209 | 0.558 | 0.511 | 0.535 | 0.296 |
| consensus | risk_on | 0.393 | 0.133 | 0.501 | 0.131 | 0.296 |
| gmm | neutral | 0.396 | 0.375 | 0.477 | 0.421 | 0.333 |
| gmm | risk_off | 0.275 | 0.241 | 0.531 | 0.180 | 0.333 |
| gmm | risk_on | 0.299 | 0.340 | 0.450 | 0.441 | 0.333 |
| logistic | neutral | 0.517 | 0.366 | 0.567 | 0.231 | 0.365 |
| logistic | risk_off | 0.193 | 0.421 | 0.507 | 0.406 | 0.365 |
| logistic | risk_on | 0.452 | 0.337 | 0.527 | 0.283 | 0.365 |

## Expected Returns By Predicted Regime

| Model | Predicted Regime | N | Avg 21D | Avg 63D | Avg 126D | Positive 63D Rate | Risk-off Event Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| consensus | neutral | 71 | 1.1% | 5.6% | 12.0% | 80.3% | 18.6% |
| consensus | risk_off | 117 | 1.7% | 3.9% | 7.7% | 68.4% | 20.9% |
| consensus | risk_on | 29 | 3.4% | 6.1% | 10.2% | 89.7% | 21.4% |
| gmm | neutral | 109 | 1.2% | 5.1% | 9.5% | 71.6% | 19.8% |
| gmm | risk_off | 51 | 1.6% | 3.5% | 7.5% | 72.5% | 27.5% |
| gmm | risk_on | 108 | 1.3% | 3.2% | 7.1% | 74.1% | 21.5% |
| logistic | neutral | 58 | 1.3% | 4.6% | 11.4% | 81.0% | 12.1% |
| logistic | risk_off | 85 | 1.8% | 4.4% | 8.6% | 68.2% | 19.3% |
| logistic | risk_on | 64 | 2.5% | 7.0% | 11.7% | 84.4% | 24.2% |

## Plain DCA vs Macro-Aware 1x

| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| plain_dca_1x | $375,575 | 20.2% | 20.9% | -34.8% | 12.31x |
| gmm_macro_aware_1x_dca | $213,606 | 15.8% | 16.1% | -22.7% | 7.00x |
| logistic_macro_aware_1x_dca | $161,815 | 13.6% | 14.2% | -28.5% | 5.31x |
| consensus_macro_aware_1x_dca | $141,362 | 12.5% | 13.1% | -20.5% | 4.63x |

## Plain DCA vs Macro-Aware 2x

| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| logistic_macro_aware_2x_dca | $841,171 | 26.6% | 27.4% | -52.0% | 27.58x |
| consensus_macro_aware_2x_dca | $773,729 | 25.9% | 26.6% | -37.1% | 25.37x |
| gmm_macro_aware_2x_dca | $568,130 | 23.5% | 24.3% | -40.7% | 18.63x |
| plain_dca_1x | $375,575 | 20.2% | 20.9% | -34.8% | 12.31x |

## Sensitivity

- Best ex-post logistic threshold in the audited 1x grid: risk-off `0.50`, jump-in `0.55`, final value `$167,162`, max DD `-28.5%`.
- Best ex-post logistic threshold in the audited 2x grid: risk-off `0.40`, jump-in `0.55`, final value `$890,630`, max DD `-52.0%`.
- Best ex-post 1x allocation grid for logistic used risk-off `50%`, neutral `100%`, finishing at `$238,373`.

## Quarter-End / Window-Dressing Diagnostics


### is_quarter_end

| Model | Regime | Flag | N | Avg 21D | Avg 63D | Positive 21D Rate |
|---|---|---|---:|---:|---:|---:|
| consensus | neutral | False | 47 | 0.4% | 5.6% | 51.1% |
| consensus | neutral | True | 24 | 2.6% | 5.5% | 66.7% |
| consensus | risk_off | False | 75 | 1.4% | 4.3% | 62.7% |
| consensus | risk_off | True | 42 | 2.2% | 3.3% | 69.0% |
| consensus | risk_on | False | 22 | 2.5% | 4.7% | 72.7% |
| consensus | risk_on | True | 7 | 6.0% | 10.4% | 100.0% |
| gmm | neutral | False | 74 | 0.7% | 5.0% | 59.5% |
| gmm | neutral | True | 35 | 2.4% | 5.1% | 68.6% |
| gmm | risk_off | False | 34 | 1.3% | 3.6% | 61.8% |
| gmm | risk_off | True | 17 | 2.0% | 3.4% | 64.7% |
| gmm | risk_on | False | 71 | 1.2% | 3.2% | 60.6% |
| gmm | risk_on | True | 37 | 1.6% | 3.3% | 64.9% |
| logistic | neutral | False | 38 | 0.8% | 5.1% | 55.3% |
| logistic | neutral | True | 20 | 2.4% | 3.7% | 70.0% |
| logistic | risk_off | False | 55 | 1.4% | 4.7% | 63.6% |
| logistic | risk_off | True | 30 | 2.5% | 3.8% | 73.3% |
| logistic | risk_on | False | 45 | 1.9% | 6.1% | 64.4% |
| logistic | risk_on | True | 19 | 3.8% | 9.0% | 73.7% |

### is_turn_of_quarter

| Model | Regime | Flag | N | Avg 21D | Avg 63D | Positive 21D Rate |
|---|---|---|---:|---:|---:|---:|
| consensus | neutral | False | 46 | 1.4% | 5.6% | 56.5% |
| consensus | neutral | True | 25 | 0.5% | 5.5% | 56.0% |
| consensus | risk_off | False | 83 | 1.5% | 3.3% | 66.3% |
| consensus | risk_off | True | 34 | 2.1% | 5.3% | 61.8% |
| consensus | risk_on | False | 17 | 2.6% | 7.6% | 76.5% |
| consensus | risk_on | True | 12 | 4.5% | 3.8% | 83.3% |
| gmm | neutral | False | 70 | 1.5% | 4.8% | 64.3% |
| gmm | neutral | True | 39 | 0.7% | 5.6% | 59.0% |
| gmm | risk_off | False | 37 | 1.4% | 3.3% | 62.2% |
| gmm | risk_off | True | 14 | 2.1% | 4.3% | 64.3% |
| gmm | risk_on | False | 71 | 1.1% | 3.6% | 62.0% |
| gmm | risk_on | True | 37 | 1.9% | 2.4% | 62.2% |
| logistic | neutral | False | 41 | 1.6% | 4.4% | 58.5% |
| logistic | neutral | True | 17 | 0.7% | 5.2% | 64.7% |
| logistic | risk_off | False | 58 | 1.7% | 4.0% | 70.7% |
| logistic | risk_off | True | 27 | 1.9% | 5.1% | 59.3% |
| logistic | risk_on | False | 39 | 2.2% | 8.0% | 66.7% |
| logistic | risk_on | True | 25 | 2.8% | 5.3% | 68.0% |

## Feature Takeaways

| Feature | 252D OLS Coef | 252D q-value | Ridge Return Coef | Logistic Risk-off Coef | Logistic Jump-in Coef |
|---|---:|---:|---:|---:|---:|
| Wilshire / GDP 1-year drift | 3.89 | 0.333 | -0.057 | -0.304 | -0.477 |
| Wilshire / GDP valuation proxy | nan | nan | -0.079 | 0.286 | -0.169 |
| Shiller CAPE | -30.92 | 0.000 | 0.058 | -0.134 | 0.293 |
| Inflation YoY | -4.05 | 0.199 | -0.033 | 0.448 | -0.207 |
| High-yield spread | nan | nan | -0.194 | 1.197 | -0.456 |
| Latent sentiment | 6.76 | 0.023 | 0.010 | -0.909 | 0.209 |
| QQQ 222-day trend level | 13.04 | 0.000 | -0.535 | 1.103 | -0.899 |
| QQQ 65-day trend level | nan | nan | 0.554 | -0.212 | 0.759 |
| Unemployment rate | -11.55 | 0.001 | -0.046 | 0.355 | -0.497 |
| VIX level | 15.86 | 0.000 | 0.151 | -1.753 | 2.133 |

## View

- Macro awareness does appear to add information, but most of the edge comes from combining valuation, stress, inflation, and trend rather than any single magic series.
- The Wilshire / GDP proxy behaves like a medium-horizon valuation headwind, not a fast crash alarm.
- Stress and liquidity proxies such as VIX, credit spreads, and financial conditions remain more useful for tactical regime shifts.
- If the objective is to beat plain DCA without leverage, the bar is high and the evidence must come from stable out-of-sample allocation rules, not from the single best ex-post parameter cell.
- If the objective allows 2x leverage, the evidence currently supports only measured use during confirmed risk-on states. A human macro-aware investor likely gets more benefit from cutting exposure in bad regimes than from maximizing leverage in good ones.
- The professional playbook here is to use monthly walk-forward classification, daily lagged execution, disciplined reserve deployment, and explicit review of false positives around risk-off calls.