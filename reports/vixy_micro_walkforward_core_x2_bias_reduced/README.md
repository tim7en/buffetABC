# VIXY Micro Walk-Forward

## Setup

- investable universe as of: `2023-01-03`
- folds: `2`
- selection metric: profit factor, then total return, then lower max drawdown

## Coverage Blocks

- block 1: `2022-06-30` -> `2022-12-27`
- block 2: `2024-07-02` -> `2024-12-27`
- block 3: `2025-08-27` -> `2026-02-23`

## Out-Of-Sample Results

- fold 1: selected `VIXY asymmetric` on train, test return `51.05%` vs baseline `60.07%`, delta `-9.02%`, test PF `2.08` vs baseline `2.31`
- fold 2: selected `Baseline` on train, test return `66.75%` vs baseline `66.75%`, delta `+0.00%`, test PF `2.93` vs baseline `2.93`

## Bottom Line

This report reduces tuning bias by choosing the VIXY variant only on prior coverage blocks, then judging it on the next unseen block.
