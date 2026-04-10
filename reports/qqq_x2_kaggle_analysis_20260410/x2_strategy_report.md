# QQQ x2 Kaggle-Style Analysis

## Verdict

- Best overall x2 strategy in this rerun: `SPY Gate + QQQ Blend 2x` with final value `$ 141,648`, XIRR `21.37%`, and max drawdown `-46.81%`.
- Best QQQ-only x2 strategy: `QQQ Ensemble Blend 2x` at `$ 130,862`, XIRR `20.42%`, max drawdown `-46.82%`.
- Plain DCA baseline finished at `$ 107,852` with XIRR `18.11%` and max drawdown `-33.80%`.

## Why The Winner Won

- The SPY gate improved final value by `$ 10,786` over standalone QQQ 2x while using lower average leverage (`1.12x` vs `1.26x`).
- It cut risk-on months from `35` to `16`, which means the edge came from selectivity rather than simply pressing leverage harder.
- Relative to plain DCA, the winner added `$ 33,795` of terminal wealth but accepted an extra `13.0` percentage points of peak-to-trough pain.

## Robustness Read

- `2014-2019`: best was `QQQ Ensemble Blend 2x` at `$ 40,120` with max drawdown `-42.14%`.
- `2020-2021`: best was `Plain DCA 1x` at `$ 21,785` with max drawdown `-27.85%`.
- `2022-2026`: best was `QQQ Ensemble Blend 2x` at `$ 26,988` with max drawdown `-35.84%`.

## Sensitivity

- Logistic 2x threshold grid peaked at risk-off `0.50` and jump-in `0.55`, finishing at `$ 131,453`.
- The threshold surface is fairly smooth rather than cliff-like, which is a good sign for model stability.
- The bigger fragility is model-choice sensitivity: GMM and ensemble-majority both underperformed plain DCA even at 2x.

## Variables That Matter Most

- `us10y_level` (10Y Treasury yield level): composite importance `1.596`
- `qqq_sma65` (QQQ 65-day trend level): composite importance `0.956` [trend]
- `curve_10y2y_level` (Yield curve 10Y-2Y): composite importance `0.859`
- `cape_63d_change` (Shiller CAPE 3-month change): composite importance `0.680` [valuation]
- `vix_level` (VIX level): composite importance `0.657` [stress]
- `hy_oas_level` (High-yield spread): composite importance `0.531` [stress]
- `hy_oas_63d_change_pp` (High-yield spread 3-month change): composite importance `0.444` [stress]
- `cpi_yoy_pct` (Inflation YoY): composite importance `0.406` [stress]

## Production Caveat

- This is good enough to treat as a research-backed overlay for DCA, not yet a fully trusted production autopilot.
- The key remaining risk is structural: the SPY gate uses features that include QQQ-derived information, so the gate/expression stack is not perfectly disentangled.
- Before production, the clean next test is a pure-SPY-feature gate controlling QQQ leverage on a locked-forward holdout.
