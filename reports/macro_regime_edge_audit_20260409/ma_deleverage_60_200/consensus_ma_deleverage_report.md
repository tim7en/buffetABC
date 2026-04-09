# Consensus MA Deleveraging Audit

- Trend rule under test: lagged `60d < 200d` on QQQ closes.
- No-leakage treatment: the moving-average state is shifted by one trading day before it can change leverage.
- Reserve behavior is unchanged: only `risk_off` parks new cash in reserve; non-`risk_off` regimes deploy reserve cash.

## Headline read

- `Consensus 2x`: baseline stayed best on terminal value at `$ 773,729` with `-37.1%` max drawdown.
- Both MA deleveraging variants reduced return more than drawdown for `2x`.
- `Consensus 3x`: baseline stayed best on terminal value at `$ 1,495,677` with `-49.7%` max drawdown.
- Both MA deleveraging variants reduced return more than drawdown for `3x`.
- `Consensus 5x`: baseline stayed best on terminal value at `$ 4,355,735` with `-69.2%` max drawdown.
- Both MA deleveraging variants reduced return more than drawdown for `5x`.

## Trend signal stats

- Lagged bearish-state share: `15.8%` of traded days.
- Raw cross-below events: `9`.
- Raw cross-above events: `10`.

## Conclusion

- A hard 1x cap during bearish trend states is too blunt for this consensus long strategy.
- A softer step-down rule preserves more upside, but still gave up too much return for only modest drawdown relief in this backtest.
- If we want trend-aware deleveraging to help, the next sensible test is a smaller haircut such as `5x -> 4x`, `3x -> 2.5x`, `2x -> 1.5x` or using the trend filter only after stress confirmation.