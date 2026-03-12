from __future__ import annotations


def _ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    ema_val = sum(values[:period]) / period
    out[period - 1] = ema_val
    for i in range(period, len(values)):
        ema_val = (values[i] * alpha) + (ema_val * (1.0 - alpha))
        out[i] = ema_val
    return out
