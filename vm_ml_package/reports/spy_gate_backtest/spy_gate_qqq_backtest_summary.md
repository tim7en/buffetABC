# SPY Gate / QQQ Ensemble Backtest

## Policy

- Broad market gate: `SPY ensemble_blend` lagged daily signal.
- Beta expression: `QQQ ensemble_blend` lagged daily signal.
- Combined rule:
  - `risk_off` when SPY gate is `risk_off`.
  - `risk_on` only when SPY gate is `risk_on` and QQQ signal is not `risk_off`.
  - `neutral` otherwise.
- Leverage rule: `risk_on => 2x`, `neutral => 1x`, `risk_off => no leverage + reserve cash for new contributions`.
- Additional tested variant: `risk_on => 3x`, `neutral => 1x`, `risk_off => 1x + reserve cash`.
- Accelerated DCA trigger for the 3x variant: on the contribution date, double the monthly DCA if the lagged policy is `risk_on`, lagged `VIX >= 20`, lagged `VIX 21d change > 0`, and lagged `QQQ 21d return < 0`.

## Scope

- QQQ price source: `reports\qqq_analysis\aligned_daily_dataset.csv`
- QQQ signal source: `reports\qqq_compare\walkforward_model_signals_daily.csv`
- SPY signal source: `reports\spy_compare\walkforward_model_signals_daily.csv`
- QQQ signal first valid date: `2014-11-03`
- SPY signal first valid date: `2014-11-03`
- Common backtest start: `2014-11-03`

## Full-Window Metrics

| strategy                                 |   final_value |     xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   risk_on_months |   neutral_months |   risk_off_months |   final_delta_vs_plain_dca |
|:-----------------------------------------|--------------:|---------:|---------------------:|---------------:|----------------------:|-----------------:|-----------------:|------------------:|---------------------------:|
| spy_gate_qqq_ensemble_blend_3x_vix_accel |        163804 | 0.230073 |             0.224472 |      -0.626857 |               1.23243 |               16 |               76 |                46 |                    55951.5 |
| spy_gate_qqq_ensemble_blend_3x           |        161725 | 0.229703 |             0.224475 |      -0.627622 |               1.23243 |               16 |               76 |                46 |                    53872.2 |
| spy_gate_qqq_ensemble_blend_2x           |        141648 | 0.213714 |             0.209321 |      -0.468118 |               1.11621 |               16 |               76 |                46 |                    33795.4 |
| qqq_ensemble_blend_3x                    |        133063 | 0.2062   |             0.208297 |      -0.627631 |               1.51287 |               35 |               41 |                62 |                    25210.1 |
| qqq_ensemble_blend_2x                    |        130862 | 0.204198 |             0.204029 |      -0.468159 |               1.25644 |               35 |               41 |                62 |                    23009   |
| plain_dca_1x                             |        107852 | 0.181067 |             0.178346 |      -0.337991 |               1       |                  |                  |                   |                        0   |

## Subwindow Metrics

| window                   | strategy                                 |   final_value |     xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   risk_on_months |   neutral_months |   risk_off_months |   final_delta_vs_plain_dca |
|:-------------------------|:-----------------------------------------|--------------:|---------:|---------------------:|---------------:|----------------------:|-----------------:|-----------------:|------------------:|---------------------------:|
| pre_covid_2014_2019      | plain_dca_1x                             |       31864.7 | 0.169983 |             0.16503  |      -0.220606 |               1       |                  |                  |                   |                      0     |
| pre_covid_2014_2019      | qqq_ensemble_blend_2x                    |       40119.5 | 0.230917 |             0.228927 |      -0.421399 |               1.3572  |               22 |               30 |                10 |                   8254.79  |
| pre_covid_2014_2019      | spy_gate_qqq_ensemble_blend_2x           |       38886.2 | 0.222538 |             0.212211 |      -0.220606 |               1.04619 |                3 |               56 |                 3 |                   7021.49  |
| pre_covid_2014_2019      | qqq_ensemble_blend_3x                    |       47659.7 | 0.277849 |             0.279166 |      -0.583653 |               1.7144  |               22 |               30 |                10 |                  15795     |
| pre_covid_2014_2019      | spy_gate_qqq_ensemble_blend_3x           |       47110.6 | 0.274653 |             0.25926  |      -0.238626 |               1.09238 |                3 |               56 |                 3 |                  15245.9   |
| pre_covid_2014_2019      | spy_gate_qqq_ensemble_blend_3x_vix_accel |       47317.1 | 0.275116 |             0.259258 |      -0.235184 |               1.09238 |                3 |               56 |                 3 |                  15452.4   |
| covid_recovery_2020_2021 | plain_dca_1x                             |       21785.1 | 0.364767 |             0.364631 |      -0.278527 |               1       |                  |                  |                   |                      0     |
| covid_recovery_2020_2021 | qqq_ensemble_blend_2x                    |       18487.8 | 0.249335 |             0.237371 |      -0.464205 |               1.21584 |                5 |                4 |                15 |                  -3297.39  |
| covid_recovery_2020_2021 | spy_gate_qqq_ensemble_blend_2x           |       21078.1 | 0.340804 |             0.334092 |      -0.464205 |               1.21188 |                5 |                7 |                12 |                   -707.046 |
| covid_recovery_2020_2021 | qqq_ensemble_blend_3x                    |       14663.3 | 0.101444 |             0.072181 |      -0.624719 |               1.43168 |                5 |                4 |                15 |                  -7121.84  |
| covid_recovery_2020_2021 | spy_gate_qqq_ensemble_blend_3x           |       18778.2 | 0.2599   |             0.241259 |      -0.624719 |               1.42376 |                5 |                7 |                12 |                  -3006.96  |
| covid_recovery_2020_2021 | spy_gate_qqq_ensemble_blend_3x_vix_accel |       19442.6 | 0.268087 |             0.241326 |      -0.621046 |               1.42376 |                5 |                7 |                12 |                  -2342.58  |
| inflation_ai_2022_2026   | plain_dca_1x                             |       23350.6 | 0.127981 |             0.109053 |      -0.283246 |               1       |                  |                  |                   |                      0     |
| inflation_ai_2022_2026   | qqq_ensemble_blend_2x                    |       26987.7 | 0.172705 |             0.151304 |      -0.358359 |               1.15327 |                8 |                7 |                37 |                   3637.12  |
| inflation_ai_2022_2026   | spy_gate_qqq_ensemble_blend_2x           |       26714.5 | 0.169522 |             0.148298 |      -0.358775 |               1.15607 |                8 |               13 |                31 |                   3363.9   |
| inflation_ai_2022_2026   | qqq_ensemble_blend_3x                    |       30074.8 | 0.206966 |             0.183427 |      -0.481313 |               1.30654 |                8 |                7 |                37 |                   6724.25  |
| inflation_ai_2022_2026   | spy_gate_qqq_ensemble_blend_3x           |       29063   | 0.196065 |             0.172585 |      -0.481479 |               1.31215 |                8 |               13 |                31 |                   5712.44  |
| inflation_ai_2022_2026   | spy_gate_qqq_ensemble_blend_3x_vix_accel |       29205.6 | 0.19631  |             0.172586 |      -0.478822 |               1.31215 |                8 |               13 |                31 |                   5854.98  |

## Read

- Combined policy finished at `$141,648` vs `$107,852` plain DCA and `$130,862` standalone QQQ blend 2x.
- Combined policy XIRR was `21.37%` with max drawdown `-46.81%`.
- 3x gated policy finished at `$161,725` with XIRR `22.97%` and max drawdown `-62.76%`.
- 3x gated policy with accelerated DCA finished at `$163,804` with XIRR `23.01%` and max drawdown `-62.69%`.
- The accelerated-DCA trigger was active on `71` daily rows spanning `8` contribution months.
- Compared with standalone QQQ blend 2x, the gate changed the path by `+10,786` of final value.
- Policy daily state counts: risk_on `334`, neutral `1588`, risk_off `952`.

## Assumptions

- Neutral means standard 1x long exposure, not partial cash.
- Risk-off keeps existing exposure unlevered and parks new contributions in reserve until the next non-risk-off regime.
- The backtest uses the existing lagged-signal engine, trading costs, and borrow cost assumptions from the QQQ walk-forward scripts.