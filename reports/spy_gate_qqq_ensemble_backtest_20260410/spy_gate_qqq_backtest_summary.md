# SPY Gate / QQQ Ensemble Backtest

## Policy

- Broad market gate: `SPY ensemble_blend` lagged daily signal.
- Beta expression: `QQQ ensemble_blend` lagged daily signal.
- Combined rule:
  - `risk_off` when SPY gate is `risk_off`.
  - `risk_on` only when SPY gate is `risk_on` and QQQ signal is not `risk_off`.
  - `neutral` otherwise.
- Leverage rule: `risk_on => 2x`, `neutral => 1x`, `risk_off => no leverage + reserve cash for new contributions`.

## Scope

- QQQ price source: `D:\buffetABC\reports\qqq_macro_ml_regime_analysis_20260409_feature_refresh\aligned_daily_dataset.csv`
- QQQ signal source: `D:\buffetABC\reports\qqq_macro_walkforward_model_compare_20260410_ensemble_refresh\walkforward_model_signals_daily.csv`
- SPY signal source: `D:\buffetABC\reports\sp500_macro_regime_with_qqq_20260410\compare\walkforward_model_signals_daily.csv`
- QQQ signal first valid date: `2014-11-03`
- SPY signal first valid date: `2014-11-03`
- Common backtest start: `2014-11-03`

## Full-Window Metrics

| strategy                       |   final_value |     xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   risk_on_months |   neutral_months |   risk_off_months |   final_delta_vs_plain_dca |
|:-------------------------------|--------------:|---------:|---------------------:|---------------:|----------------------:|-----------------:|-----------------:|------------------:|---------------------------:|
| spy_gate_qqq_ensemble_blend_2x |        141648 | 0.213714 |             0.209321 |      -0.468118 |               1.11621 |               16 |               76 |                46 |                    33795.4 |
| qqq_ensemble_blend_2x          |        130862 | 0.204198 |             0.204029 |      -0.468159 |               1.25644 |               35 |               41 |                62 |                    23009   |
| plain_dca_1x                   |        107852 | 0.181067 |             0.178346 |      -0.337991 |               1       |                  |                  |                   |                        0   |

## Subwindow Metrics

| window                   | strategy                       |   final_value |     xirr |   time_weighted_cagr |   max_drawdown |   avg_target_leverage |   risk_on_months |   neutral_months |   risk_off_months |   final_delta_vs_plain_dca |
|:-------------------------|:-------------------------------|--------------:|---------:|---------------------:|---------------:|----------------------:|-----------------:|-----------------:|------------------:|---------------------------:|
| pre_covid_2014_2019      | plain_dca_1x                   |       31864.7 | 0.169983 |             0.16503  |      -0.220606 |               1       |                  |                  |                   |                      0     |
| pre_covid_2014_2019      | qqq_ensemble_blend_2x          |       40119.5 | 0.230917 |             0.228927 |      -0.421399 |               1.3572  |               22 |               30 |                10 |                   8254.79  |
| pre_covid_2014_2019      | spy_gate_qqq_ensemble_blend_2x |       38886.2 | 0.222538 |             0.212211 |      -0.220606 |               1.04619 |                3 |               56 |                 3 |                   7021.49  |
| covid_recovery_2020_2021 | plain_dca_1x                   |       21785.1 | 0.364767 |             0.364631 |      -0.278527 |               1       |                  |                  |                   |                      0     |
| covid_recovery_2020_2021 | qqq_ensemble_blend_2x          |       18487.8 | 0.249335 |             0.237371 |      -0.464205 |               1.21584 |                5 |                4 |                15 |                  -3297.39  |
| covid_recovery_2020_2021 | spy_gate_qqq_ensemble_blend_2x |       21078.1 | 0.340804 |             0.334092 |      -0.464205 |               1.21188 |                5 |                7 |                12 |                   -707.046 |
| inflation_ai_2022_2026   | plain_dca_1x                   |       23350.6 | 0.127981 |             0.109053 |      -0.283246 |               1       |                  |                  |                   |                      0     |
| inflation_ai_2022_2026   | qqq_ensemble_blend_2x          |       26987.7 | 0.172705 |             0.151304 |      -0.358359 |               1.15327 |                8 |                7 |                37 |                   3637.12  |
| inflation_ai_2022_2026   | spy_gate_qqq_ensemble_blend_2x |       26714.5 | 0.169522 |             0.148298 |      -0.358775 |               1.15607 |                8 |               13 |                31 |                   3363.9   |

## Read

- Combined policy finished at `$141,648` vs `$107,852` plain DCA and `$130,862` standalone QQQ blend 2x.
- Combined policy XIRR was `21.37%` with max drawdown `-46.81%`.
- Compared with standalone QQQ blend 2x, the gate changed the path by `+10,786` of final value.
- Policy daily state counts: risk_on `334`, neutral `1588`, risk_off `952`.

## Assumptions

- Neutral means standard 1x long exposure, not partial cash.
- Risk-off keeps existing exposure unlevered and parks new contributions in reserve until the next non-risk-off regime.
- The backtest uses the existing lagged-signal engine, trading costs, and borrow cost assumptions from the QQQ walk-forward scripts.