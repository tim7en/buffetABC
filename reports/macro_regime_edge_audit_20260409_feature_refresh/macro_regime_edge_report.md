# Macro Regime Edge Audit

## Anti-Leakage Discipline

- QQQ prices are aligned on actual trading dates only.
- Monthly supervised models train only on earlier month-end observations.
- Any row whose forward target window overlaps the prediction date is purged from training.
- Allocation and leverage simulations trade lagged regime signals only.
- Quarterly GDP and annual market-cap-to-GDP anchor data are lagged before forward-fill, so the proxy is not using future macro releases.

## Current Read

- As of `2026-04-09` the macro cycle is `expansion` with `high` confidence.
- Expansion / late-cycle / contraction scores are `5` / `3` / `0`.
- The Wilshire / GDP proxy is `2.1609` with rolling z-score `1.32`.
- The traded regime is `risk_off` with target allocation `25%`.

## Regime Accuracy

| Model | Label | Precision | Recall | Balanced Accuracy | False Positive Rate | Overall Accuracy |
|---|---|---:|---:|---:|---:|---:|
| consensus | neutral | 0.415 | 0.315 | 0.524 | 0.267 | 0.299 |
| consensus | risk_off | 0.226 | 0.594 | 0.507 | 0.580 | 0.299 |
| consensus | risk_on | 0.368 | 0.121 | 0.491 | 0.140 | 0.299 |
| gmm | neutral | 0.370 | 0.333 | 0.468 | 0.397 | 0.335 |
| gmm | risk_off | 0.216 | 0.211 | 0.514 | 0.182 | 0.335 |
| gmm | risk_on | 0.356 | 0.397 | 0.463 | 0.471 | 0.335 |
| logistic | neutral | 0.367 | 0.229 | 0.505 | 0.218 | 0.311 |
| logistic | risk_off | 0.254 | 0.531 | 0.523 | 0.485 | 0.311 |
| logistic | risk_on | 0.368 | 0.255 | 0.477 | 0.300 | 0.311 |

## Expected Returns By Predicted Regime

| Model | Predicted Regime | N | Avg 21D | Avg 63D | Avg 126D | Positive 63D Rate | Risk-off Event Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| consensus | neutral | 41 | 1.6% | 6.2% | 10.9% | 90.2% | 14.6% |
| consensus | risk_off | 88 | 1.8% | 4.1% | 9.0% | 67.0% | 22.6% |
| consensus | risk_on | 19 | 1.2% | 4.6% | 9.2% | 84.2% | 36.8% |
| gmm | neutral | 77 | 1.6% | 4.3% | 9.7% | 66.2% | 20.5% |
| gmm | risk_off | 37 | 2.9% | 5.4% | 10.9% | 75.7% | 21.6% |
| gmm | risk_on | 87 | 1.3% | 5.2% | 9.4% | 85.1% | 17.2% |
| logistic | neutral | 30 | 2.3% | 6.3% | 12.3% | 86.7% | 13.3% |
| logistic | risk_off | 71 | 1.2% | 3.5% | 8.3% | 63.4% | 25.4% |
| logistic | risk_on | 38 | 1.8% | 5.2% | 9.7% | 86.8% | 28.9% |

## Plain DCA vs Macro-Aware 1x

| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| plain_dca_1x | $107,852 | 18.1% | 17.8% | -33.8% | 4.55x |
| logistic_macro_aware_1x_dca | $58,269 | 10.8% | 11.2% | -26.4% | 2.46x |
| gmm_macro_aware_1x_dca | $53,319 | 9.7% | 9.8% | -28.3% | 2.25x |
| consensus_macro_aware_1x_dca | $43,931 | 7.4% | 7.8% | -26.3% | 1.85x |

## Plain DCA vs Macro-Aware 2x

| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |
|---|---:|---:|---:|---:|---:|
| logistic_macro_aware_2x_dca | $128,950 | 20.2% | 20.3% | -46.8% | 5.44x |
| plain_dca_1x | $107,852 | 18.1% | 17.8% | -33.8% | 4.55x |
| gmm_macro_aware_2x_dca | $106,863 | 18.0% | 17.7% | -49.1% | 4.51x |
| consensus_macro_aware_2x_dca | $98,173 | 17.0% | 17.1% | -46.8% | 4.14x |

## Sensitivity

- Best ex-post logistic threshold in the audited 1x grid: risk-off `0.50`, jump-in `0.50`, final value `$61,379`, max DD `-28.3%`.
- Best ex-post logistic threshold in the audited 2x grid: risk-off `0.50`, jump-in `0.55`, final value `$136,286`, max DD `-46.8%`.
- Best ex-post 1x allocation grid for logistic used risk-off `50%`, neutral `100%`, finishing at `$78,629`.

## Quarter-End / Window-Dressing Diagnostics


### is_quarter_end

| Model | Regime | Flag | N | Avg 21D | Avg 63D | Positive 21D Rate |
|---|---|---|---:|---:|---:|---:|
| consensus | neutral | False | 29 | 0.8% | 6.0% | 62.1% |
| consensus | neutral | True | 12 | 3.7% | 6.7% | 75.0% |
| consensus | risk_off | False | 58 | 1.5% | 4.5% | 60.3% |
| consensus | risk_off | True | 30 | 2.4% | 3.3% | 70.0% |
| consensus | risk_on | False | 13 | 0.7% | 3.3% | 53.8% |
| consensus | risk_on | True | 6 | 2.2% | 7.7% | 83.3% |
| gmm | neutral | False | 49 | 0.6% | 3.3% | 55.1% |
| gmm | neutral | True | 28 | 3.4% | 6.0% | 71.4% |
| gmm | risk_off | False | 23 | 2.8% | 7.9% | 69.6% |
| gmm | risk_off | True | 14 | 3.0% | 1.3% | 85.7% |
| gmm | risk_on | False | 62 | 1.2% | 5.0% | 61.3% |
| gmm | risk_on | True | 25 | 1.4% | 5.4% | 64.0% |
| logistic | neutral | False | 20 | 1.3% | 7.2% | 65.0% |
| logistic | neutral | True | 10 | 4.1% | 4.6% | 80.0% |
| logistic | risk_off | False | 47 | 0.9% | 3.6% | 55.3% |
| logistic | risk_off | True | 24 | 1.8% | 3.4% | 62.5% |
| logistic | risk_on | False | 26 | 1.3% | 4.7% | 61.5% |
| logistic | risk_on | True | 12 | 2.8% | 6.4% | 83.3% |

### is_turn_of_quarter

| Model | Regime | Flag | N | Avg 21D | Avg 63D | Positive 21D Rate |
|---|---|---|---:|---:|---:|---:|
| consensus | neutral | False | 27 | 1.3% | 5.5% | 63.0% |
| consensus | neutral | True | 14 | 2.2% | 7.6% | 71.4% |
| consensus | risk_off | False | 55 | 1.6% | 4.0% | 67.3% |
| consensus | risk_off | True | 33 | 2.2% | 4.2% | 57.6% |
| consensus | risk_on | False | 14 | 1.0% | 5.9% | 64.3% |
| consensus | risk_on | True | 5 | 1.7% | 1.2% | 60.0% |
| gmm | neutral | False | 52 | 1.7% | 4.7% | 63.5% |
| gmm | neutral | True | 25 | 1.3% | 3.4% | 56.0% |
| gmm | risk_off | False | 23 | 2.8% | 4.9% | 82.6% |
| gmm | risk_off | True | 14 | 3.1% | 6.2% | 64.3% |
| gmm | risk_on | False | 59 | 1.1% | 4.9% | 59.3% |
| gmm | risk_on | True | 28 | 1.8% | 5.7% | 67.9% |
| logistic | neutral | False | 20 | 2.8% | 5.6% | 75.0% |
| logistic | neutral | True | 10 | 1.2% | 7.7% | 60.0% |
| logistic | risk_off | False | 45 | 0.8% | 3.5% | 60.0% |
| logistic | risk_off | True | 26 | 1.8% | 3.5% | 53.8% |
| logistic | risk_on | False | 27 | 1.0% | 5.2% | 63.0% |
| logistic | risk_on | True | 11 | 3.8% | 5.3% | 81.8% |

## Feature Takeaways

| Feature | 252D OLS Coef | 252D q-value | Ridge Return Coef | Logistic Risk-off Coef | Logistic Jump-in Coef |
|---|---:|---:|---:|---:|---:|
| Wilshire / GDP 1-year drift | 2.41 | 0.572 | -0.050 | 0.039 | -0.451 |
| Wilshire / GDP valuation proxy | nan | nan | -0.014 | 0.097 | 0.001 |
| Shiller CAPE | -13.57 | 0.000 | -0.074 | 0.191 | -0.143 |
| Inflation YoY | -7.95 | 0.001 | -0.037 | 0.638 | -0.260 |
| High-yield spread | nan | nan | -0.224 | 1.546 | -0.792 |
| Latent sentiment | 12.72 | 0.000 | -0.012 | -0.906 | 0.041 |
| QQQ 222-day trend level | nan | nan | -0.476 | 1.009 | -0.803 |
| QQQ 65-day trend level | nan | nan | 0.495 | -0.166 | 0.627 |
| Unemployment rate | 0.54 | 0.901 | -0.049 | 0.022 | -0.201 |
| VIX level | 2.77 | 0.700 | 0.167 | -1.340 | 1.806 |

## View

- Macro awareness does appear to add information, but most of the edge comes from combining valuation, stress, inflation, and trend rather than any single magic series.
- The Wilshire / GDP proxy behaves like a medium-horizon valuation headwind, not a fast crash alarm.
- Stress and liquidity proxies such as VIX, credit spreads, and financial conditions remain more useful for tactical regime shifts.
- If the objective is to beat plain DCA without leverage, the bar is high and the evidence must come from stable out-of-sample allocation rules, not from the single best ex-post parameter cell.
- If the objective allows 2x leverage, the evidence currently supports only measured use during confirmed risk-on states. A human macro-aware investor likely gets more benefit from cutting exposure in bad regimes than from maximizing leverage in good ones.
- The professional playbook here is to use monthly walk-forward classification, daily lagged execution, disciplined reserve deployment, and explicit review of false positives around risk-off calls.