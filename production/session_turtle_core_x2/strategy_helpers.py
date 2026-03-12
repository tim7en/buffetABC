from __future__ import annotations


def _sma(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    rolling = sum(values[:period])
    out[period - 1] = rolling / period
    for i in range(period, n):
        rolling += values[i] - values[i - period]
        out[i] = rolling / period
    return out


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period + 1:
        return out

    tr: list[float] = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = sum(tr[1: period + 1]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = ((atr * (period - 1)) + tr[i]) / period
        out[i] = atr
    return out


def _rolling_highest(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    for i in range(period - 1, n):
        out[i] = max(values[i - period + 1: i + 1])
    return out


def _rolling_lowest(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    for i in range(period - 1, n):
        out[i] = min(values[i - period + 1: i + 1])
    return out


def _chandelier_stop_candidate(
    direction: str,
    idx: int,
    rolling_highs: list[float | None],
    rolling_lows: list[float | None],
    atr_values: list[float | None],
    atr_mult: float,
) -> float | None:
    if idx < 0 or idx >= len(atr_values):
        return None
    atr_now = atr_values[idx]
    if atr_now is None or atr_now <= 0:
        return None
    if direction == "long":
        hh = rolling_highs[idx]
        if hh is None:
            return None
        return hh - (atr_now * atr_mult)
    ll = rolling_lows[idx]
    if ll is None:
        return None
    return ll + (atr_now * atr_mult)


def _tighten_stop(
    direction: str,
    current_stop: float,
    candidate_stop: float | None,
    take_profit: float | None = None,
) -> float:
    if candidate_stop is None:
        return current_stop
    if direction == "long":
        new_stop = max(current_stop, candidate_stop)
        if take_profit is not None:
            new_stop = min(new_stop, take_profit - 1e-6)
        return new_stop
    new_stop = min(current_stop, candidate_stop)
    if take_profit is not None:
        new_stop = max(new_stop, take_profit + 1e-6)
    return new_stop


def _break_even_stop_candidate(
    direction: str,
    entry_price: float,
    initial_stop: float,
    bar_high: float,
    bar_low: float,
    trigger_r: float,
) -> float | None:
    if trigger_r <= 0:
        return None
    initial_risk = abs(entry_price - initial_stop)
    if initial_risk <= 0:
        return None
    trigger_distance = initial_risk * trigger_r
    if direction == "long":
        return entry_price if bar_high >= (entry_price + trigger_distance) else None
    return entry_price if bar_low <= (entry_price - trigger_distance) else None


def _stop_is_trailed(direction: str, initial_stop: float, active_stop: float, tolerance: float = 1e-8) -> bool:
    if direction == "long":
        return active_stop > (initial_stop + tolerance)
    return active_stop < (initial_stop - tolerance)


def _stop_is_break_even_or_better(
    direction: str,
    entry_price: float,
    active_stop: float,
    tolerance: float = 1e-8,
) -> bool:
    if direction == "long":
        return active_stop >= (entry_price - tolerance)
    return active_stop <= (entry_price + tolerance)
