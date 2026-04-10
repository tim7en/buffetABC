# Aggressive SPY-Gated QQQ Variants

## Variants

- `spy_gate_qqq_ensemble_blend_2x`: current combined policy benchmark.
- `spy_gate_full_cash_3x`: `risk_off -> full cash`, `neutral -> 1x QQQ`, `risk_on -> 3x QQQ`.
- `spy_gate_full_cash_3x_vix_spike_dca2x`: same as above, but on monthly contribution dates inside `risk_on` plus lagged VIX spike plus lagged QQQ 21-day return <= `-3%`, the contribution is multiplied by `2.0x`.
- VIX spike uses the existing lagged volatility-shock definition: `VIX >= 25` or `VIX 21d change >= 5`.

## Full-Window Metrics

| strategy                              |   final_value |     xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   risk_on_months |   neutral_months |   risk_off_months |   accelerated_contribution_events |   accelerated_extra_contribution |   final_delta_vs_plain_dca |
|:--------------------------------------|--------------:|---------:|---------------------:|---------------:|----------------------:|-----------------:|-----------------:|------------------:|----------------------------------:|---------------------------------:|---------------------------:|
| spy_gate_qqq_ensemble_blend_2x        |        141648 | 0.213714 |             0.209321 |      -0.468118 |              1.11621  |               16 |               76 |                46 |                                 0 |                                0 |                  33795.4   |
| plain_dca_1x                          |        107852 | 0.181067 |             0.178346 |      -0.337991 |              1        |                  |                  |                   |                                 0 |                                0 |                      0     |
| spy_gate_full_cash_3x_vix_spike_dca2x |        107181 | 0.178795 |             0.177699 |      -0.626861 |              0.901183 |               16 |               76 |                46 |                                 5 |                              500 |                   -671.041 |
| spy_gate_full_cash_3x                 |        105652 | 0.178609 |             0.177701 |      -0.627623 |              0.901183 |               16 |               76 |                46 |                                 0 |                                0 |                  -2200.96  |

## Subwindow Metrics

| window                   | strategy                              |   final_value |      xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   accelerated_contribution_events |   accelerated_extra_contribution |   final_delta_vs_plain_dca |
|:-------------------------|:--------------------------------------|--------------:|----------:|---------------------:|---------------:|----------------------:|----------------------------------:|---------------------------------:|---------------------------:|
| pre_covid_2014_2019      | plain_dca_1x                          |       31864.7 | 0.169983  |            0.16503   |      -0.220606 |              1        |                                 0 |                                0 |                      0     |
| pre_covid_2014_2019      | spy_gate_qqq_ensemble_blend_2x        |       38886.2 | 0.222538  |            0.212211  |      -0.220606 |              1.04619  |                                 0 |                                0 |                   7021.49  |
| pre_covid_2014_2019      | spy_gate_full_cash_3x                 |       47211.7 | 0.275244  |            0.259966  |      -0.238647 |              1.04234  |                                 0 |                                0 |                  15347     |
| pre_covid_2014_2019      | spy_gate_full_cash_3x_vix_spike_dca2x |       47418.2 | 0.275706  |            0.259964  |      -0.235213 |              1.04234  |                                 1 |                              100 |                  15553.5   |
| covid_recovery_2020_2021 | plain_dca_1x                          |       21785.1 | 0.364767  |            0.364631  |      -0.278527 |              1        |                                 0 |                                0 |                      0     |
| covid_recovery_2020_2021 | spy_gate_qqq_ensemble_blend_2x        |       21078.1 | 0.340804  |            0.334092  |      -0.464205 |              1.21188  |                                 0 |                                0 |                   -707.046 |
| covid_recovery_2020_2021 | spy_gate_full_cash_3x                 |       14047.8 | 0.0758822 |            0.0574995 |      -0.624719 |              0.916832 |                                 0 |                                0 |                  -7737.3   |
| covid_recovery_2020_2021 | spy_gate_full_cash_3x_vix_spike_dca2x |       14696.2 | 0.0841174 |            0.0574719 |      -0.621046 |              0.916832 |                                 4 |                              400 |                  -7088.93  |
| inflation_ai_2022_2026   | plain_dca_1x                          |       23350.6 | 0.127981  |            0.109053  |      -0.283246 |              1        |                                 0 |                                0 |                      0     |
| inflation_ai_2022_2026   | spy_gate_qqq_ensemble_blend_2x        |       26714.5 | 0.169522  |            0.148298  |      -0.358775 |              1.15607  |                                 0 |                                0 |                   3363.9   |
| inflation_ai_2022_2026   | spy_gate_full_cash_3x                 |       24660.1 | 0.144699  |            0.137631  |      -0.483106 |              0.72243  |                                 0 |                                0 |                   1309.48  |
| inflation_ai_2022_2026   | spy_gate_full_cash_3x_vix_spike_dca2x |       24660.1 | 0.144699  |            0.137631  |      -0.483106 |              0.72243  |                                 0 |                                0 |                   1309.48  |