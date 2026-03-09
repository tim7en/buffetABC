"""Session range breakout strategy.

Process:
1) Build a pre-open trading range from the bars leading into the New York equity open.
2) During the opening breakout window, wait for a clean close outside that range.
3) Enter on the next bar in the breakout direction.
4) Place the stop beyond the breakout candle wick, with a configurable buffer.
5) Target a fixed R multiple, with optional room-to-liquidity filtering.

Notes:
- Built for lower-timeframe execution bars (5m/15m/30m).
- Uses real `America/New_York` time for the open anchor so DST shifts are handled.
- Designed to be less selective than the hourly SFP + FVG reversal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from edgar.services.binance_data import fetch_binance_klines
from edgar.services.market_mechanics_strategy import (
    _bars_per_day,
    _fetch_intraday_bars,
    _interval_to_minutes,
    _max_lookback_days_for_interval,
    _pivot_levels,
    _resolve_effective_interval,
)
from edgar.services.session_sfp_fvg_strategy import (
    _aggregate_hourly,
    _aggregate_sessions,
    _bars_per_day_24x7,
    _next_hourly_liquidity_target,
    _next_session_liquidity_target,
    _resolve_market_data_source,
    _resolve_next_liquidity_target,
    _to_new_york,
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
    range_low: float
    range_high: float
    breakout_level: float
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
    stop_source: str = "breakout_wick"
    target_liquidity: float | None = None
    target_liquidity_type: str = ""


def _session_clock(ts: datetime, session_open: str) -> datetime:
    aware_utc = ts.replace(tzinfo=timezone.utc)
    if session_open == "new_york_equity_open":
        return _to_new_york(ts)
    if session_open == "asia_open":
        return aware_utc
    raise ValueError("session_open must be one of ['new_york_equity_open', 'asia_open']")


def _session_open_bounds_local(local: datetime, session_open: str, breakout_window_minutes: int) -> tuple[datetime, datetime]:
    if session_open == "new_york_equity_open":
        start = local.replace(hour=9, minute=30, second=0, microsecond=0)
    elif session_open == "asia_open":
        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("session_open must be one of ['new_york_equity_open', 'asia_open']")
    end = start + timedelta(minutes=max(1, breakout_window_minutes))
    return start, end


def _is_session_breakout_window(local: datetime, session_open: str, breakout_window_minutes: int) -> bool:
    start, end = _session_open_bounds_local(local, session_open, breakout_window_minutes)
    return start <= local < end


def _pre_open_range(
    session_times: list[datetime],
    highs: list[float],
    lows: list[float],
    idx: int,
    range_lookback_minutes: int,
    session_open: str,
) -> tuple[float | None, float | None, int]:
    if idx < 0:
        return None, None, 0
    local = session_times[idx]
    open_start, _ = _session_open_bounds_local(local, session_open, 1)
    range_start = open_start - timedelta(minutes=max(1, range_lookback_minutes))
    collected: list[int] = []
    j = idx
    while j >= 0:
        lt = session_times[j]
        if lt >= open_start:
            j -= 1
            continue
        if lt < range_start:
            break
        collected.append(j)
        j -= 1
    if len(collected) < 3:
        return None, None, 0
    return (
        min(lows[j] for j in collected),
        max(highs[j] for j in collected),
        len(collected),
    )


def run_session_range_breakout_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    auto_adjust_for_yf_limits: bool = True,
    session_open: str = "new_york_equity_open",
    range_lookback_minutes: int = 180,
    breakout_window_minutes: int = 90,
    breakout_buffer_bps: float = 0.0,
    breakout_close_buffer_bps: float = 0.0,
    use_target_room_filter: bool = False,
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
    if session_open not in {"new_york_equity_open", "asia_open"}:
        raise ValueError("session_open must be one of ['new_york_equity_open', 'asia_open']")
    if _interval_to_minutes(interval) >= 60:
        raise ValueError("session_range_breakout requires an execution interval below 60 minutes")
    if range_lookback_minutes < 30:
        raise ValueError("range_lookback_minutes must be >= 30")
    if breakout_window_minutes < 1:
        raise ValueError("breakout_window_minutes must be >= 1")
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
            raise ValueError("session_range_breakout requires an execution interval below 60 minutes")
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
    warmup_bars = max(volume_period + 20, int((range_lookback_minutes / interval_minutes) * 3), 720)
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
    session_times = [_session_clock(ts, session_open) for ts in timestamps]
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
    piv_hi_hourly, piv_lo_hourly = _pivot_levels(hourly_highs, hourly_lows, window=1)

    vol_sma = _sma(volumes, volume_period)
    capital = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0
    equity_curve: list[dict] = []
    trades: list[_Trade] = []
    open_trade: _Trade | None = None
    session_traded: set[date] = set()

    commission_rate = max(commission_bps, 0.0) / 10_000.0
    slippage_rate = max(slippage_bps, 0.0) / 10_000.0
    breakout_buffer = max(breakout_buffer_bps, 0.0) / 10_000.0
    close_buffer = max(breakout_close_buffer_bps, 0.0) / 10_000.0
    stop_buffer = max(stop_buffer_bps, 0.0) / 10_000.0

    chandelier_atr_vals = _atr(highs, lows, closes, chandelier_atr_period) if use_chandelier_exit else []
    chandelier_highs = _rolling_highest(highs, chandelier_period) if use_chandelier_exit else []
    chandelier_lows = _rolling_lowest(lows, chandelier_period) if use_chandelier_exit else []

    def _close_trade(trade: _Trade, idx: int, raw_exit_price: float, reason: str) -> None:
        nonlocal capital, total_fees
        exit_price = raw_exit_price * (1.0 - slippage_rate) if trade.direction == "long" else raw_exit_price * (1.0 + slippage_rate)
        exit_notional = abs(trade.shares) * exit_price
        exit_fee = exit_notional * commission_rate
        gross_pnl = (
            (exit_price - trade.entry_price) * trade.shares
            if trade.direction == "long"
            else (trade.entry_price - exit_price) * trade.shares
        )
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
        open_i = opens[i]
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        local = session_times[i]
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
            equity_curve.append({"date": ts.isoformat(), "equity": round(equity, 4), "capital": round(capital, 4)})

        if open_trade is not None:
            continue
        if ts < period_start or i >= len(bars) - 1:
            continue

        current_hour_idx = bar_to_hour_idx[i]
        current_session_idx = bar_to_session_idx[i]
        completed_hour_idx = current_hour_idx - 1
        if completed_hour_idx < 1:
            continue
        hourly_support = next((float(piv_lo_hourly[j]) for j in range(completed_hour_idx, -1, -1) if piv_lo_hourly[j] is not None), None)
        hourly_resistance = next((float(piv_hi_hourly[j]) for j in range(completed_hour_idx, -1, -1) if piv_hi_hourly[j] is not None), None)

        if session_date in session_traded or not _is_session_breakout_window(local, session_open, breakout_window_minutes):
            continue

        range_low, range_high, range_bar_count = _pre_open_range(
            session_times=session_times,
            highs=highs,
            lows=lows,
            idx=i,
            range_lookback_minutes=range_lookback_minutes,
            session_open=session_open,
        )
        if range_low is None or range_high is None or range_bar_count < 3 or range_high <= range_low:
            continue

        prev_close = closes[i - 1] if i > 0 else close_i
        broke_up = high_i > (range_high * (1.0 + breakout_buffer))
        broke_down = low_i < (range_low * (1.0 - breakout_buffer))
        close_up = close_i > (range_high * (1.0 + close_buffer))
        close_down = close_i < (range_low * (1.0 - close_buffer))
        prev_inside = (range_low <= prev_close <= range_high)
        long_signal = allow_longs and prev_inside and broke_up and close_up and not broke_down
        short_signal = allow_shorts and prev_inside and broke_down and close_down and not broke_up
        if long_signal == short_signal:
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        if next_open <= 0:
            continue

        direction = "long" if long_signal else "short"
        entry_price = next_open * (1.0 + slippage_rate) if direction == "long" else next_open * (1.0 - slippage_rate)
        if direction == "long":
            stop_loss = low_i * (1.0 - stop_buffer)
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * rr_multiple)
            breakout_level = range_high
        else:
            stop_loss = high_i * (1.0 + stop_buffer)
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * rr_multiple)
            breakout_level = range_low
        if sl_distance <= 0 or take_profit <= 0:
            continue

        target_level = None
        target_type = ""
        if use_target_room_filter:
            older_session = session_bars[current_session_idx - 2] if current_session_idx >= 2 else None
            recent_session = session_bars[current_session_idx - 1] if current_session_idx >= 1 else None
            target_level, target_type = _resolve_next_liquidity_target(
                direction=direction,
                reference_price=entry_price,
                hourly_targets=_next_hourly_liquidity_target(
                    direction=direction,
                    pivot_levels=piv_hi_hourly if direction == "long" else piv_lo_hourly,
                    usable_end_idx=completed_hour_idx,
                    reference_price=entry_price,
                    search_window=max(24, int(range_lookback_minutes / 60) * 6),
                ),
                session_targets=_next_session_liquidity_target(
                    direction=direction,
                    older_session=older_session,
                    recent_session=recent_session,
                    reference_price=entry_price,
                ),
            )
            if target_level is None:
                continue
            target_distance = target_level - entry_price if direction == "long" else entry_price - target_level
            required_room = sl_distance * rr_multiple * min_target_room_ratio
            if target_distance < required_room:
                continue

        risk_amount = capital * base_risk_pct
        shares = risk_amount / sl_distance
        position_size = shares * entry_price
        max_notional = capital * max_position_pct
        sizing_tier = "standard"
        if position_size > max_notional and entry_price > 0:
            shares = max_notional / entry_price
            position_size = max_notional
            risk_amount = shares * sl_distance
            sizing_tier = "standard_capped"
        if shares <= 0 or position_size <= 0:
            continue

        entry_fee = position_size * commission_rate
        open_trade = _Trade(
            direction=direction,
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_pct=round(risk_amount / capital if capital > 0 else 0.0, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            range_low=round(range_low, 4),
            range_high=round(range_high, 4),
            breakout_level=round(breakout_level, 4),
            hourly_support=round(hourly_support, 4) if hourly_support is not None else None,
            hourly_resistance=round(hourly_resistance, 4) if hourly_resistance is not None else None,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            exit_fill_policy=exit_fill_policy,
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality="A" if range_bar_count >= 12 else "B",
            target_liquidity=round(target_level, 4) if target_level is not None else None,
            target_liquidity_type=target_type,
        )
        session_traded.add(session_date)

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
        "strategy_variant": "session_range_breakout",
        "bias_model": "none",
        "entry_session": f"{session_open}_breakout",
        "session_open": session_open,
        "range_lookback_minutes": range_lookback_minutes,
        "breakout_window_minutes": breakout_window_minutes,
        "breakout_buffer_bps": breakout_buffer_bps,
        "breakout_close_buffer_bps": breakout_close_buffer_bps,
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
                "fractal_high": t.range_high,
                "fractal_low": t.range_low,
                "liquidity_level": t.breakout_level,
                "target_liquidity": t.target_liquidity,
                "target_liquidity_type": t.target_liquidity_type,
                "hourly_support": t.hourly_support,
                "hourly_resistance": t.hourly_resistance,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
