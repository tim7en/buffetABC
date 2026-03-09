"""Hourly swing-failure + lower-timeframe FVG retest strategy.

Transcript-aligned process:
1) Use confirmed hourly swing structure to infer the day bias.
2) Around the New York equity open, wait for a raid of a confirmed hourly swing.
3) Require a lower-timeframe swing failure pattern (sweep + close back through the level).
4) Wait for a same-direction fair value gap to form.
5) Enter on the first retest of that FVG, with the stop beyond the middle candle wick.

Notes:
- Built for lower-timeframe execution bars (5m/15m/30m).
- The NY open filter uses real `America/New_York` time so DST shifts are handled.
- Hourly pivots are confirmed with right-side bars only; no future bars are used for
  the active bias, swing levels, or entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from edgar.services.binance_data import fetch_binance_klines
from edgar.services.market_mechanics_strategy import (
    _bars_per_day,
    _fetch_intraday_bars,
    _interval_to_minutes,
    _max_lookback_days_for_interval,
    _pivot_levels,
    _recent_pivots,
    _resolve_effective_interval,
)
from edgar.services.strategy import (
    _atr,
    _break_even_stop_candidate,
    _chandelier_stop_candidate,
    _resolve_bar_bracket_exit,
    _rolling_highest,
    _rolling_lowest,
    _sma,
    _stop_is_break_even_or_better,
    _stop_is_trailed,
    _tighten_stop,
)


_NY_TZ = ZoneInfo("America/New_York")


@dataclass
class _ExecutionFVG:
    kind: str
    idx: int
    zone_low: float
    zone_high: float
    stop_ref: float


@dataclass
class _SfpEvent:
    session_date: date
    direction: str
    level: float
    sfp_idx: int
    sweep_extreme: float
    hourly_support: float | None
    hourly_resistance: float | None


@dataclass
class _PendingRetest:
    session_date: date
    direction: str
    level: float
    sfp_idx: int
    sweep_extreme: float
    hourly_support: float | None
    hourly_resistance: float | None
    zone_low: float
    zone_high: float
    stop_ref: float
    fvg_idx: int
    expires_idx: int


@dataclass
class _Trade:
    direction: str
    entry_ts: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    position_size: float
    shares: float
    entry_index: int
    liquidity_level: float
    fvg_low: float
    fvg_high: float
    hourly_support: float | None
    hourly_resistance: float | None
    active_stop_loss: float | None = None
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0
    intrabar_conflict: bool = False
    exit_fill_policy: str = "stop_first"
    entry_rel_volume: float = 0.0
    volume_confirmed: bool = False
    sizing_tier: str = "standard"
    signal_quality: str = "A"
    hold_bars: int = 0
    stop_source: str = "fvg_mid_wick"
    target_liquidity: float | None = None
    target_liquidity_type: str = ""


@dataclass
class _SessionBar:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _bars_per_day_24x7(interval: str) -> int:
    minutes = max(_interval_to_minutes(interval), 1)
    return max(int((24 * 60) / minutes), 1)


def _resolve_market_data_source(market_data_source: str, ticker: str) -> str:
    source = (market_data_source or "auto").strip().lower()
    if source in {"yfinance", "yf"}:
        return "yfinance"
    if source in {"binance", "binance_spot"}:
        return "binance"
    if source == "auto":
        text = (ticker or "").strip().upper()
        if text.startswith("BTC") or text.startswith("ETH") or text.startswith("PAXG"):
            return "binance"
        return "yfinance"
    raise ValueError("market_data_source must be one of ['auto', 'yfinance', 'binance']")


def _to_new_york(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc).astimezone(_NY_TZ)


def _is_ny_open_window(ts: datetime, window_minutes: int) -> bool:
    local = _to_new_york(ts)
    start = local.replace(hour=9, minute=30, second=0, microsecond=0)
    end = start + timedelta(minutes=max(window_minutes, 1))
    return start <= local < end


def _hour_bucket_local(ts: datetime) -> datetime:
    local = _to_new_york(ts)
    return local.replace(minute=0, second=0, microsecond=0)


def _aggregate_hourly(bars: list[dict]) -> tuple[list[dict], list[int]]:
    hourly: list[dict] = []
    bar_to_hour_idx: list[int] = []
    current_key: datetime | None = None
    current: dict | None = None

    for bar in bars:
        hour_key = _hour_bucket_local(bar["timestamp"])
        if current_key != hour_key:
            if current is not None:
                hourly.append(current)
            current_key = hour_key
            current = {
                "timestamp": hour_key,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", 0.0) or 0.0),
            }
        else:
            current["high"] = max(float(current["high"]), float(bar["high"]))
            current["low"] = min(float(current["low"]), float(bar["low"]))
            current["close"] = float(bar["close"])
            current["volume"] = float(current["volume"]) + float(bar.get("volume", 0.0) or 0.0)
        bar_to_hour_idx.append(len(hourly))

    if current is not None:
        hourly.append(current)

    return hourly, bar_to_hour_idx


def _aggregate_sessions(bars: list[dict]) -> tuple[list[_SessionBar], list[int]]:
    sessions: list[_SessionBar] = []
    bar_to_session_idx: list[int] = []
    current_key: date | None = None
    current: _SessionBar | None = None

    for bar in bars:
        session_key = _to_new_york(bar["timestamp"]).date()
        if current_key != session_key:
            if current is not None:
                sessions.append(current)
            current_key = session_key
            current = _SessionBar(
                session_date=session_key,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar.get("volume", 0.0) or 0.0),
            )
        else:
            current.high = max(float(current.high), float(bar["high"]))
            current.low = min(float(current.low), float(bar["low"]))
            current.close = float(bar["close"])
            current.volume = float(current.volume) + float(bar.get("volume", 0.0) or 0.0)
        bar_to_session_idx.append(len(sessions))

    if current is not None:
        sessions.append(current)

    return sessions, bar_to_session_idx


def _collect_execution_fvgs(highs: list[float], lows: list[float]) -> list[_ExecutionFVG]:
    out: list[_ExecutionFVG] = []
    for i in range(2, len(highs)):
        if lows[i] > highs[i - 2]:
            out.append(
                _ExecutionFVG(
                    kind="bullish",
                    idx=i,
                    zone_low=highs[i - 2],
                    zone_high=lows[i],
                    stop_ref=lows[i - 1],
                )
            )
        if highs[i] < lows[i - 2]:
            out.append(
                _ExecutionFVG(
                    kind="bearish",
                    idx=i,
                    zone_low=highs[i],
                    zone_high=lows[i - 2],
                    stop_ref=highs[i - 1],
                )
            )
    return out


def _hourly_bias(
    piv_hi: list[float | None],
    piv_lo: list[float | None],
    usable_end_idx: int,
    buffer: float,
) -> tuple[str | None, float | None, float | None]:
    highs_recent = _recent_pivots(piv_hi, usable_end_idx, count=2)
    lows_recent = _recent_pivots(piv_lo, usable_end_idx, count=2)
    if len(highs_recent) < 2 or len(lows_recent) < 2:
        return None, None, None

    old_hi = highs_recent[0][1]
    new_hi = highs_recent[1][1]
    old_lo = lows_recent[0][1]
    new_lo = lows_recent[1][1]
    if new_hi > (old_hi * (1.0 + buffer)) and new_lo > (old_lo * (1.0 + buffer)):
        return "long", new_lo, new_hi
    if new_hi < (old_hi * (1.0 - buffer)) and new_lo < (old_lo * (1.0 - buffer)):
        return "short", new_lo, new_hi
    return None, new_lo, new_hi


def _session_bias_from_previous_two(
    sessions: list[_SessionBar],
    current_session_idx: int,
    buffer: float,
) -> tuple[str | None, _SessionBar | None, _SessionBar | None]:
    if current_session_idx < 2:
        return None, None, None

    older = sessions[current_session_idx - 2]
    recent = sessions[current_session_idx - 1]
    bullish_structure = (
        recent.high > (older.high * (1.0 + buffer))
        and recent.low > (older.low * (1.0 + buffer))
    )
    bearish_structure = (
        recent.high < (older.high * (1.0 - buffer))
        and recent.low < (older.low * (1.0 - buffer))
    )
    bullish_close = recent.close > recent.open and recent.close >= older.close
    bearish_close = recent.close < recent.open and recent.close <= older.close

    if bullish_structure and bullish_close:
        return "long", older, recent
    if bearish_structure and bearish_close:
        return "short", older, recent
    return None, older, recent


def _recent_confirmed_levels(
    levels: list[float | None],
    usable_end_idx: int,
    lookback_hours: int,
    max_levels: int = 8,
) -> list[tuple[int, float]]:
    if usable_end_idx < 0:
        return []
    start = max(0, usable_end_idx - max(lookback_hours, 1) + 1)
    out: list[tuple[int, float]] = []
    for i in range(usable_end_idx, start - 1, -1):
        level = levels[i]
        if level is None:
            continue
        out.append((i, float(level)))
        if len(out) >= max_levels:
            break
    return out


def _latest_confirmed_level(
    levels: list[float | None],
    usable_end_idx: int,
) -> float | None:
    if usable_end_idx < 0:
        return None
    for i in range(usable_end_idx, -1, -1):
        level = levels[i]
        if level is not None:
            return float(level)
    return None


def _swept_level(
    direction: str,
    levels: list[tuple[int, float]],
    high_price: float,
    low_price: float,
    close_price: float,
    sweep_buffer: float,
    reclaim_buffer: float,
) -> float | None:
    matched: list[float] = []
    for _, level in levels:
        if direction == "long":
            swept = low_price < (level * (1.0 - sweep_buffer))
            reclaimed = close_price > (level * (1.0 + reclaim_buffer))
        else:
            swept = high_price > (level * (1.0 + sweep_buffer))
            reclaimed = close_price < (level * (1.0 - reclaim_buffer))
        if swept and reclaimed:
            matched.append(level)
    if not matched:
        return None
    return max(matched) if direction == "long" else min(matched)


def _bar_overlaps_zone(bar_low: float, bar_high: float, zone_low: float, zone_high: float) -> bool:
    return bar_high >= zone_low and bar_low <= zone_high


def _next_hourly_liquidity_target(
    direction: str,
    pivot_levels: list[float | None],
    usable_end_idx: int,
    reference_price: float,
    search_window: int,
) -> tuple[float | None, str]:
    if usable_end_idx < 0:
        return None, ""
    start = max(0, usable_end_idx - max(search_window, 1) + 1)
    candidate: float | None = None
    for i in range(start, usable_end_idx + 1):
        level = pivot_levels[i]
        if level is None:
            continue
        level_f = float(level)
        if direction == "long":
            if level_f <= reference_price:
                continue
            if candidate is None or level_f < candidate:
                candidate = level_f
        else:
            if level_f >= reference_price:
                continue
            if candidate is None or level_f > candidate:
                candidate = level_f
    return candidate, ("hourly_pivot" if candidate is not None else "")


def _next_session_liquidity_target(
    direction: str,
    older_session: _SessionBar | None,
    recent_session: _SessionBar | None,
    reference_price: float,
) -> tuple[float | None, str]:
    sessions = [s for s in (older_session, recent_session) if s is not None]
    candidate: float | None = None
    source = ""
    for idx, session in enumerate(sessions):
        level = session.high if direction == "long" else session.low
        if direction == "long":
            if level <= reference_price:
                continue
            if candidate is None or level < candidate:
                candidate = float(level)
                source = "prior_session_high" if idx == len(sessions) - 1 else "two_sessions_back_high"
        else:
            if level >= reference_price:
                continue
            if candidate is None or level > candidate:
                candidate = float(level)
                source = "prior_session_low" if idx == len(sessions) - 1 else "two_sessions_back_low"
    return candidate, source


def _resolve_next_liquidity_target(
    direction: str,
    reference_price: float,
    hourly_targets: tuple[float | None, str],
    session_targets: tuple[float | None, str],
) -> tuple[float | None, str]:
    candidates = []
    hourly_level, hourly_source = hourly_targets
    session_level, session_source = session_targets
    if hourly_level is not None:
        candidates.append((float(hourly_level), hourly_source))
    if session_level is not None:
        candidates.append((float(session_level), session_source))
    if not candidates:
        return None, ""
    if direction == "long":
        return min(candidates, key=lambda item: item[0])
    return max(candidates, key=lambda item: item[0])


def run_session_sfp_fvg_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    auto_adjust_for_yf_limits: bool = True,
    hourly_bias_pivot_window: int = 2,
    hourly_level_pivot_window: int = 1,
    swing_lookback_hours: int = 48,
    session_trigger_window_minutes: int = 60,
    sfp_structure_buffer_bps: float = 0.0,
    sfp_sweep_buffer_bps: float = 3.0,
    sfp_reclaim_buffer_bps: float = 0.0,
    fvg_search_bars: int = 18,
    fvg_retest_bars: int = 18,
    fvg_reclaim_buffer_bps: float = 0.0,
    use_target_room_filter: bool = True,
    min_target_room_ratio: float = 1.0,
    stop_buffer_bps: float = 5.0,
    rr_multiple: float = 2.0,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    slippage_bps: float = 4.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
    use_break_even_stop: bool = False,
    break_even_trigger_r: float = 1.0,
    use_chandelier_exit: bool = False,
    chandelier_period: int = 22,
    chandelier_atr_period: int = 22,
    chandelier_atr_mult: float = 3.0,
    exit_fill_policy: str = "stop_first",
) -> dict:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if _interval_to_minutes(interval) >= 60:
        raise ValueError("hourly_sfp_fvg requires an execution interval below 60 minutes")
    if hourly_bias_pivot_window < 1 or hourly_level_pivot_window < 1:
        raise ValueError("hourly pivot windows must be >= 1")
    if fvg_search_bars < 1 or fvg_retest_bars < 1:
        raise ValueError("fvg_search_bars and fvg_retest_bars must be >= 1")
    if session_trigger_window_minutes < 1:
        raise ValueError("session_trigger_window_minutes must be >= 1")
    if min_target_room_ratio <= 0:
        raise ValueError("min_target_room_ratio must be positive")
    if exit_fill_policy not in {"stop_first", "target_first"}:
        raise ValueError("exit_fill_policy must be one of ['stop_first', 'target_first']")
    if chandelier_period < 2 or chandelier_atr_period < 2:
        raise ValueError("chandelier periods must be >= 2")

    resolved_source = _resolve_market_data_source(market_data_source=market_data_source, ticker=ticker)
    requested_interval = interval
    interval_adjustment = None
    if resolved_source == "yfinance":
        effective_interval, interval_adjustment = _resolve_effective_interval(
            requested_interval=requested_interval,
            lookback_years=lookback_years,
            auto_adjust_for_yf_limits=auto_adjust_for_yf_limits,
        )
        if _interval_to_minutes(effective_interval) >= 60:
            raise ValueError("hourly_sfp_fvg requires an execution interval below 60 minutes")
        lookback_days = max(int(365.25 * lookback_years), 1)
        max_days = _max_lookback_days_for_interval(effective_interval)
        if max_days is not None and lookback_days > max_days:
            raise ValueError(
                f"Yahoo Finance limit for interval={effective_interval} is about {max_days} days. "
                f"Requested {lookback_days} days (~{lookback_years}y). "
                "Use a shorter window or Binance for multi-year 5m crypto history."
            )
    else:
        effective_interval = requested_interval

    bars_per_day = (
        max(_bars_per_day(effective_interval), 1)
        if resolved_source == "yfinance"
        else _bars_per_day_24x7(effective_interval)
    )
    interval_minutes = max(_interval_to_minutes(effective_interval), 1)
    warmup_bars = max(
        volume_period + 20,
        int((swing_lookback_hours + hourly_bias_pivot_window + hourly_level_pivot_window + 6) * (60 / interval_minutes)),
        720,
    )
    warmup_days = max(int(warmup_bars / bars_per_day) + 10, 20)

    resolved_symbol = ticker.upper()
    if resolved_source == "binance":
        bars, resolved_symbol = fetch_binance_klines(
            ticker=ticker,
            interval=effective_interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
    else:
        bars = _fetch_intraday_bars(
            ticker=ticker,
            interval=effective_interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
        )
    if len(bars) < warmup_bars + 30:
        raise ValueError(f"Insufficient intraday data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b.get("volume", 0.0) for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    hourly_bars, bar_to_hour_idx = _aggregate_hourly(bars)
    session_bars, bar_to_session_idx = _aggregate_sessions(bars)
    hourly_highs = [b["high"] for b in hourly_bars]
    hourly_lows = [b["low"] for b in hourly_bars]
    piv_hi_hourly, piv_lo_hourly = _pivot_levels(hourly_highs, hourly_lows, window=hourly_level_pivot_window)

    fvgs = _collect_execution_fvgs(highs, lows)
    fvg_by_idx: dict[int, list[_ExecutionFVG]] = {}
    for fvg in fvgs:
        fvg_by_idx.setdefault(fvg.idx, []).append(fvg)

    vol_sma = _sma(volumes, volume_period)
    chandelier_atr_vals = _atr(highs, lows, closes, period=chandelier_atr_period) if use_chandelier_exit else []
    chandelier_highs = _rolling_highest(highs, chandelier_period) if use_chandelier_exit else []
    chandelier_lows = _rolling_lowest(lows, chandelier_period) if use_chandelier_exit else []

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0
    structure_buffer = max(0.0, sfp_structure_buffer_bps) / 10_000.0
    sweep_buffer = max(0.0, sfp_sweep_buffer_bps) / 10_000.0
    reclaim_buffer = max(0.0, sfp_reclaim_buffer_bps) / 10_000.0
    fvg_reclaim_buffer = max(0.0, fvg_reclaim_buffer_bps) / 10_000.0
    stop_buffer = max(0.0, stop_buffer_bps) / 10_000.0

    capital = float(initial_capital)
    trades: list[_Trade] = []
    equity_curve: list[dict] = []
    open_trade: _Trade | None = None
    peak_equity = capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0
    session_traded: set[date] = set()
    pending_event: _SfpEvent | None = None
    pending_retest: _PendingRetest | None = None

    def _close_trade(trade: _Trade, idx: int, raw_exit_price: float, reason: str) -> None:
        nonlocal capital, total_fees
        if trade.direction == "long":
            exit_price = raw_exit_price * (1.0 - slippage_rate)
            gross_pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            exit_price = raw_exit_price * (1.0 + slippage_rate)
            gross_pnl = (trade.entry_price - exit_price) * trade.shares

        exit_fee = abs(trade.shares * exit_price) * commission_rate
        fee_total = trade.fees_paid + exit_fee
        net_pnl = gross_pnl - fee_total

        trade.exit_ts = timestamps[idx]
        trade.exit_price = round(exit_price, 4)
        trade.pnl = round(net_pnl, 4)
        trade.exit_reason = reason
        trade.fees_paid = round(fee_total, 4)
        trade.hold_bars = max(idx - trade.entry_index, 0)

        capital += net_pnl
        total_fees += fee_total
        trades.append(trade)

    for i in range(start_idx, len(bars)):
        ts = timestamps[i]
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        local = _to_new_york(ts)
        session_date = local.date()

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            raw_exit, hit_sl, hit_tp, intrabar_conflict = _resolve_bar_bracket_exit(
                direction=open_trade.direction,
                bar_high=high_i,
                bar_low=low_i,
                stop_loss=current_stop,
                take_profit=open_trade.take_profit,
                fill_policy=exit_fill_policy,
            )
            if raw_exit is not None:
                open_trade.intrabar_conflict = intrabar_conflict
                trailed_stop = _stop_is_trailed(open_trade.direction, open_trade.stop_loss, current_stop)
                break_even_stop = use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
                    direction=open_trade.direction,
                    entry_price=open_trade.entry_price,
                    active_stop=current_stop,
                )
                _close_trade(
                    open_trade,
                    i,
                    raw_exit,
                    (
                        "break_even_stop"
                        if hit_sl and break_even_stop
                        else (
                            "chandelier_stop"
                            if hit_sl and use_chandelier_exit and trailed_stop
                            else ("stop_loss" if hit_sl else "take_profit")
                        )
                    ),
                )
                open_trade = None
            else:
                next_stop = current_stop
                if use_break_even_stop:
                    next_stop = _tighten_stop(
                        direction=open_trade.direction,
                        current_stop=next_stop,
                        candidate_stop=_break_even_stop_candidate(
                            direction=open_trade.direction,
                            entry_price=open_trade.entry_price,
                            initial_stop=open_trade.stop_loss,
                            bar_high=high_i,
                            bar_low=low_i,
                            trigger_r=break_even_trigger_r,
                        ),
                        take_profit=open_trade.take_profit,
                    )
                tr_stop = None
                if use_chandelier_exit:
                    tr_stop = _chandelier_stop_candidate(
                        direction=open_trade.direction,
                        idx=i,
                        rolling_highs=chandelier_highs,
                        rolling_lows=chandelier_lows,
                        atr_values=chandelier_atr_vals,
                        atr_mult=chandelier_atr_mult,
                    )
                open_trade.active_stop_loss = round(
                    _tighten_stop(
                        direction=open_trade.direction,
                        current_stop=next_stop,
                        candidate_stop=tr_stop,
                        take_profit=open_trade.take_profit,
                    ),
                    4,
                )

        unrealized = 0.0
        if open_trade is not None:
            if open_trade.direction == "long":
                marked = close_i * (1.0 - slippage_rate)
                unrealized = (marked - open_trade.entry_price) * open_trade.shares - open_trade.fees_paid
            else:
                marked = close_i * (1.0 + slippage_rate)
                unrealized = (open_trade.entry_price - marked) * open_trade.shares - open_trade.fees_paid

        if ts >= period_start:
            equity = capital + unrealized
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, dd)
            equity_curve.append(
                {
                    "date": ts.isoformat(),
                    "equity": round(equity, 4),
                    "capital": round(capital, 4),
                }
            )

        if open_trade is not None:
            continue
        if ts < period_start or i >= len(bars) - 1:
            continue

        if pending_event is not None and (pending_event.session_date != session_date or (i - pending_event.sfp_idx) > fvg_search_bars):
            pending_event = None
        if pending_retest is not None and (pending_retest.session_date != session_date or i > pending_retest.expires_idx):
            pending_retest = None

        current_hour_idx = bar_to_hour_idx[i]
        current_session_idx = bar_to_session_idx[i]
        completed_hour_idx = current_hour_idx - 1
        level_usable_end = completed_hour_idx - hourly_level_pivot_window
        if level_usable_end < 1 or current_session_idx < 2:
            continue

        bias_direction, older_session, recent_session = _session_bias_from_previous_two(
            sessions=session_bars,
            current_session_idx=current_session_idx,
            buffer=structure_buffer,
        )
        hourly_support = _latest_confirmed_level(
            levels=piv_lo_hourly,
            usable_end_idx=level_usable_end,
        )
        hourly_resistance = _latest_confirmed_level(
            levels=piv_hi_hourly,
            usable_end_idx=level_usable_end,
        )

        if pending_retest is not None:
            invalidated = (
                low_i < pending_retest.sweep_extreme if pending_retest.direction == "long" else high_i > pending_retest.sweep_extreme
            )
            if invalidated:
                pending_retest = None
            else:
                zone_low = min(pending_retest.zone_low, pending_retest.zone_high)
                zone_high = max(pending_retest.zone_low, pending_retest.zone_high)
                overlaps = _bar_overlaps_zone(low_i, high_i, zone_low, zone_high)
                direction_ok = (
                    close_i >= (zone_low * (1.0 + fvg_reclaim_buffer))
                    if pending_retest.direction == "long"
                    else close_i <= (zone_high * (1.0 - fvg_reclaim_buffer))
                )
                if overlaps and direction_ok:
                    rel_volume = 1.0
                    if vol_sma[i] is not None and vol_sma[i] > 0:
                        rel_volume = volumes[i] / vol_sma[i]
                    if use_volume_filter and rel_volume < min_rel_volume:
                        continue

                    next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
                    if next_open <= 0:
                        continue
                    direction = pending_retest.direction
                    entry_price = next_open * (1.0 + slippage_rate) if direction == "long" else next_open * (1.0 - slippage_rate)
                    if direction == "long":
                        stop_ref = min(pending_retest.stop_ref, pending_retest.sweep_extreme)
                        stop_loss = stop_ref * (1.0 - stop_buffer)
                        sl_distance = entry_price - stop_loss
                        take_profit = entry_price + (sl_distance * rr_multiple)
                    else:
                        stop_ref = max(pending_retest.stop_ref, pending_retest.sweep_extreme)
                        stop_loss = stop_ref * (1.0 + stop_buffer)
                        sl_distance = stop_loss - entry_price
                        take_profit = entry_price - (sl_distance * rr_multiple)
                    if sl_distance <= 0 or take_profit <= 0:
                        pending_retest = None
                        continue

                    target_level, target_type = _resolve_next_liquidity_target(
                        direction=direction,
                        reference_price=entry_price,
                        hourly_targets=_next_hourly_liquidity_target(
                            direction=direction,
                            pivot_levels=piv_hi_hourly if direction == "long" else piv_lo_hourly,
                            usable_end_idx=level_usable_end,
                            reference_price=entry_price,
                            search_window=max(swing_lookback_hours, 24),
                        ),
                        session_targets=_next_session_liquidity_target(
                            direction=direction,
                            older_session=older_session,
                            recent_session=recent_session,
                            reference_price=entry_price,
                        ),
                    )
                    if use_target_room_filter:
                        if target_level is None:
                            pending_retest = None
                            continue
                        target_distance = (
                            target_level - entry_price if direction == "long" else entry_price - target_level
                        )
                        required_room = sl_distance * rr_multiple * min_target_room_ratio
                        if target_distance < required_room:
                            pending_retest = None
                            continue

                    risk_pct = base_risk_pct
                    risk_amount = capital * risk_pct
                    shares = risk_amount / sl_distance
                    position_size = shares * entry_price
                    max_notional = capital * max_position_pct
                    sizing_tier = "standard"
                    signal_quality = "A"
                    if position_size > max_notional and entry_price > 0:
                        shares = max_notional / entry_price
                        position_size = max_notional
                        risk_amount = shares * sl_distance
                        risk_pct = risk_amount / capital if capital > 0 else 0.0
                        sizing_tier = "standard_capped"
                    if shares <= 0 or position_size <= 0:
                        pending_retest = None
                        continue

                    entry_fee = position_size * commission_rate
                    open_trade = _Trade(
                        direction=direction,
                        entry_ts=timestamps[i + 1],
                        entry_price=round(entry_price, 4),
                        stop_loss=round(stop_loss, 4),
                        take_profit=round(take_profit, 4),
                        risk_pct=round(risk_pct, 6),
                        position_size=round(position_size, 4),
                        shares=round(shares, 6),
                        entry_index=i + 1,
                        liquidity_level=round(pending_retest.level, 4),
                        fvg_low=round(zone_low, 4),
                        fvg_high=round(zone_high, 4),
                        hourly_support=round(pending_retest.hourly_support, 4) if pending_retest.hourly_support is not None else None,
                        hourly_resistance=round(pending_retest.hourly_resistance, 4) if pending_retest.hourly_resistance is not None else None,
                        active_stop_loss=round(stop_loss, 4),
                        fees_paid=round(entry_fee, 4),
                        exit_fill_policy=exit_fill_policy,
                        entry_rel_volume=round(rel_volume, 3),
                        volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
                        sizing_tier=sizing_tier,
                        signal_quality=signal_quality,
                        target_liquidity=round(target_level, 4) if target_level is not None else None,
                        target_liquidity_type=target_type,
                    )
                    session_traded.add(session_date)
                    pending_retest = None
                    pending_event = None
            continue

        if pending_event is not None:
            desired_kind = "bullish" if pending_event.direction == "long" else "bearish"
            fvg_candidates = [
                fvg
                for fvg in fvg_by_idx.get(i, [])
                if fvg.kind == desired_kind and fvg.idx > pending_event.sfp_idx
            ]
            if fvg_candidates:
                chosen = fvg_candidates[-1]
                pending_retest = _PendingRetest(
                    session_date=pending_event.session_date,
                    direction=pending_event.direction,
                    level=pending_event.level,
                    sfp_idx=pending_event.sfp_idx,
                    sweep_extreme=pending_event.sweep_extreme,
                    hourly_support=pending_event.hourly_support,
                    hourly_resistance=pending_event.hourly_resistance,
                    zone_low=chosen.zone_low,
                    zone_high=chosen.zone_high,
                    stop_ref=chosen.stop_ref,
                    fvg_idx=chosen.idx,
                    expires_idx=min(chosen.idx + fvg_retest_bars, len(bars) - 2),
                )
                pending_event = None
            continue

        if session_date in session_traded or not _is_ny_open_window(ts, session_trigger_window_minutes):
            continue

        long_level = None
        if allow_longs and bias_direction == "long":
            long_levels = _recent_confirmed_levels(
                levels=piv_lo_hourly,
                usable_end_idx=level_usable_end,
                lookback_hours=swing_lookback_hours,
                max_levels=8,
            )
            long_level = _swept_level(
                direction="long",
                levels=long_levels,
                high_price=high_i,
                low_price=low_i,
                close_price=close_i,
                sweep_buffer=sweep_buffer,
                reclaim_buffer=reclaim_buffer,
            )
        short_level = None
        if allow_shorts and bias_direction == "short":
            short_levels = _recent_confirmed_levels(
                levels=piv_hi_hourly,
                usable_end_idx=level_usable_end,
                lookback_hours=swing_lookback_hours,
                max_levels=8,
            )
            short_level = _swept_level(
                direction="short",
                levels=short_levels,
                high_price=high_i,
                low_price=low_i,
                close_price=close_i,
                sweep_buffer=sweep_buffer,
                reclaim_buffer=reclaim_buffer,
            )

        if long_level is not None and short_level is not None:
            continue
        if long_level is not None:
            pending_event = _SfpEvent(
                session_date=session_date,
                direction="long",
                level=long_level,
                sfp_idx=i,
                sweep_extreme=low_i,
                hourly_support=hourly_support,
                hourly_resistance=hourly_resistance,
            )
        elif short_level is not None:
            pending_event = _SfpEvent(
                session_date=session_date,
                direction="short",
                level=short_level,
                sfp_idx=i,
                sweep_extreme=high_i,
                hourly_support=hourly_support,
                hourly_resistance=hourly_resistance,
            )

    if open_trade is not None:
        _close_trade(open_trade, len(bars) - 1, closes[-1], "end_of_data")

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    gross_profit = sum(t.pnl for t in winning)
    gross_loss_abs = abs(sum(t.pnl for t in losing))
    total_return = ((capital - initial_capital) / initial_capital) * 100.0

    start_ts = timestamps[start_idx]
    end_ts = timestamps[-1]
    years = max((end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    cagr = ((capital / initial_capital) ** (1 / years) - 1.0) * 100.0 if capital > 0 else 0.0
    avg_trade_return = (
        sum((t.pnl / t.position_size) * 100.0 for t in trades if t.position_size > 0) / len(trades)
        if trades
        else 0.0
    )
    exposure = (bars_in_position / bars_in_period * 100.0) if bars_in_period > 0 else 0.0
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "ticker": ticker.upper(),
        "data_mode": "intraday",
        "interval": effective_interval,
        "requested_interval": requested_interval,
        "effective_interval": effective_interval,
        "interval_adjustment": interval_adjustment,
        "market_data_source": resolved_source,
        "market_data_symbol": resolved_symbol,
        "strategy_variant": "hourly_sfp_fvg",
        "bias_model": "previous_two_sessions",
        "entry_session": "new_york_equity_open",
        "use_target_room_filter": use_target_room_filter,
        "min_target_room_ratio": min_target_room_ratio,
        "use_chandelier_exit": use_chandelier_exit,
        "use_break_even_stop": use_break_even_stop,
        "break_even_trigger_r": break_even_trigger_r,
        "chandelier_period": chandelier_period,
        "chandelier_atr_period": chandelier_atr_period,
        "chandelier_atr_mult": chandelier_atr_mult,
        "exit_fill_policy": exit_fill_policy,
        "lookback_years": lookback_years,
        "bar_count": len(equity_curve),
        "start_date": start_ts.isoformat(),
        "end_date": end_ts.isoformat(),
        "initial_capital": initial_capital,
        "final_capital": round(capital, 4),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round((len(winning) / len(trades)) * 100.0, 1) if trades else 0.0,
        "max_drawdown_pct": round(max_drawdown * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "cagr_pct": round(cagr, 2),
        "avg_trade_return_pct": round(avg_trade_return, 2),
        "exposure_pct": round(exposure, 2),
        "total_fees": round(total_fees, 4),
        "price_series": [
            {
                "date": timestamps[j].isoformat(),
                "open": round(opens[j], 4),
                "high": round(highs[j], 4),
                "low": round(lows[j], 4),
                "close": round(closes[j], 4),
                "volume": round(volumes[j], 4),
            }
            for j in range(start_idx, len(bars))
        ],
        "trades": [
            {
                "direction": t.direction,
                "entry_date": t.entry_ts.isoformat(),
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "active_stop_loss": t.active_stop_loss,
                "risk_pct": t.risk_pct,
                "position_size": t.position_size,
                "shares": t.shares,
                "exit_date": t.exit_ts.isoformat() if t.exit_ts else None,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "fees_paid": t.fees_paid,
                "intrabar_conflict": t.intrabar_conflict,
                "exit_fill_policy": t.exit_fill_policy,
                "entry_rel_volume": t.entry_rel_volume,
                "volume_confirmed": t.volume_confirmed,
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "fractal_high": t.fvg_high,
                "fractal_low": t.fvg_low,
                "liquidity_level": t.liquidity_level,
                "target_liquidity": t.target_liquidity,
                "target_liquidity_type": t.target_liquidity_type,
                "hourly_support": t.hourly_support,
                "hourly_resistance": t.hourly_resistance,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
