# Short-Horizon Hedge Audit

## Anti-Leakage Discipline

- Hedge targets are built from future 21-day and 42-day path pain, but model features use only information available at each month-end.
- Walk-forward hedge models train on earlier month-end rows only.
- Any row whose forward path window overlaps the prediction date is purged from training.
- Daily hedge execution uses one-day-lagged monthly states.
- Slow macro releases remain lagged because the source dataset already enforces release discipline.

## Target Design

- `hedge_strong_target`: next 21d min return <= -10% or 21d path-CVaR20 <= -7.5%; event rate `13.3%`.
- `hedge_light_target`: strong target or next 42d min return <= -8% or 42d path-CVaR20 <= -5.5%; event rate `28.8%`.
- Average next 21d min return = `-3.8%`; average next 42d path-CVaR20 = `-3.5%`.

## Current Hedge Read

- As of `2026-04-07`, logistic strong-hedge probability = `5.9%` and light-hedge probability = `49.1%`.
- Random forest strong-hedge probability = `31.5%` and light-hedge probability = `47.3%`.
- Consensus hedge state = `hedge_0_6`.

## Validation

| Target | Model | Train N | Test N | AUC | Avg Precision | Brier | Precision@50 | Recall@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hedge_light_target | logistic | 224 | 97 | 0.561 | 0.348 | 0.487 | 0.303 | 0.714 |
| hedge_light_target | random_forest | 224 | 97 | 0.629 | 0.423 | 0.202 | 0.562 | 0.321 |
| hedge_strong_target | logistic | 225 | 98 | 0.506 | 0.115 | 0.546 | 0.132 | 0.818 |
| hedge_strong_target | random_forest | 225 | 98 | 0.749 | 0.272 | 0.119 | 0.000 | 0.000 |

## Expected Path Pain By Predicted State

| Model | State | N | Avg 21D Return | Avg 42D Return | Avg 21D Min | Avg 42D Min | 42D CVaR20 | Strong Event Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| consensus | hedge_0_3 | 8 | -0.2% | 1.9% | -7.6% | -9.0% | -6.1% | 37.5% |
| consensus | hedge_0_6 | 117 | 1.8% | 3.5% | -3.3% | -4.7% | -2.6% | 7.8% |
| consensus | unhedged | 95 | 1.4% | 2.7% | -2.7% | -4.1% | -2.3% | 7.4% |
| logistic | hedge_0_3 | 42 | 2.0% | 3.1% | -3.4% | -4.6% | -2.7% | 11.9% |
| logistic | hedge_0_6 | 60 | 1.7% | 3.0% | -3.0% | -4.6% | -2.6% | 6.8% |
| logistic | unhedged | 118 | 1.3% | 3.1% | -3.2% | -4.6% | -2.6% | 8.5% |
| random_forest | hedge_0_3 | 14 | -2.9% | -0.3% | -8.9% | -11.0% | -7.7% | 35.7% |
| random_forest | hedge_0_6 | 46 | 3.5% | 5.4% | -3.2% | -4.4% | -2.0% | 9.1% |
| random_forest | unhedged | 160 | 1.4% | 2.7% | -2.7% | -4.1% | -2.4% | 6.2% |

## Strategy Results

| Strategy | Base Beta | Final Value | XIRR | CAGR | Max DD | Avg Beta | Delta vs Same-Beta Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_beta_1x | 1.0 | $293,619 | 16.9% | 20.4% | -46.7% | 1.00 | $0 |
| random_forest_hedge_base_1x | 1.0 | $246,105 | 15.6% | 19.3% | -28.2% | 0.87 | $-47,514 |
| consensus_hedge_base_1x | 1.0 | $175,975 | 13.1% | 17.1% | -30.2% | 0.76 | $-117,643 |
| logistic_hedge_base_1x | 1.0 | $145,310 | 11.7% | 15.9% | -37.8% | 0.76 | $-148,309 |
| baseline_beta_2x | 2.0 | $705,317 | 23.3% | 26.4% | -76.1% | 2.00 | $0 |
| random_forest_hedge_base_2x | 2.0 | $504,000 | 20.9% | 24.1% | -46.3% | 1.61 | $-201,317 |
| consensus_hedge_base_2x | 2.0 | $265,381 | 16.2% | 19.8% | -42.3% | 1.20 | $-439,937 |
| logistic_hedge_base_2x | 2.0 | $191,603 | 13.8% | 17.6% | -64.3% | 1.30 | $-513,715 |
| baseline_beta_3x | 3.0 | $838,699 | 24.6% | 27.6% | -90.3% | 3.00 | $0 |
| random_forest_hedge_base_3x | 3.0 | $705,488 | 23.4% | 26.4% | -60.8% | 2.34 | $-133,211 |
| consensus_hedge_base_3x | 3.0 | $322,858 | 17.6% | 21.1% | -53.3% | 1.64 | $-515,841 |
| logistic_hedge_base_3x | 3.0 | $179,485 | 13.3% | 17.2% | -82.9% | 1.84 | $-659,214 |
| random_forest_hedge_base_5x | 5.0 | $439,777 | 19.9% | 23.1% | -81.4% | 3.81 | $259,661 |
| consensus_hedge_base_5x | 5.0 | $260,009 | 16.0% | 19.6% | -72.0% | 2.51 | $79,894 |
| baseline_beta_5x | 5.0 | $180,116 | 13.3% | 17.2% | -98.8% | 5.00 | $0 |
| logistic_hedge_base_5x | 5.0 | $66,126 | 5.8% | 11.0% | -96.5% | 2.91 | $-113,990 |

## Sensitivity

- Best ex-post hedge configuration in the audited grid: base beta `2.0`, light threshold `0.50`, strong threshold `0.55`, final value `$228,175`, max DD `-64.3%`.

## Feature Takeaways

| Feature | 42D Min q-value | 42D CVaR q-value | Logistic Light | Logistic Strong |
|---|---:|---:|---:|---:|
| Wilshire / GDP valuation proxy | nan | nan | 0.372 | 0.500 |
| Shiller CAPE | 0.001 | 0.003 | -0.068 | 0.263 |
| Inflation YoY | 0.345 | 0.190 | 0.648 | 0.359 |
| High-yield spread | nan | nan | 1.159 | 1.073 |
| Latent sentiment | 0.594 | 0.482 | -0.746 | -0.640 |
| Financial conditions level | 0.001 | 0.003 | 0.077 | 0.164 |
| QQQ 222-day trend level | 0.262 | 0.284 | 0.701 | 0.651 |
| QQQ 65-day trend level | nan | nan | -0.202 | -0.073 |
| Unemployment rate | 0.114 | 0.055 | 0.841 | 0.612 |
| VIX level | 0.146 | 0.049 | -1.544 | -1.633 |

## View

- A hedge model should be judged on path pain, not just forward return. Predicting drawdown and path-CVaR is more aligned with actual hedging decisions.
- The defensive edge will likely come from avoiding the worst path episodes while keeping most of the market beta in ordinary environments.
- Quarter-end and turn-of-quarter effects are worth monitoring, but they should remain context features rather than dominant hedge triggers.
- A hedge overlay that only reduces net beta to 0.6 or 0.3 is structurally safer than a fully short tactical model, because false positives remain expensive.
- The honest benchmark is the same-beta unhedged baseline, not just plain 1x DCA. A useful hedge should improve path risk without giving away too much terminal wealth.