"""Session-constrained turtle trend-following strategy with long and short legs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from edgar.services.binance_data import load_local_binance_klines
from edgar.services.local_tiingo_data import load_local_tiingo_klines
from edgar.services.session_open_utils import (
    SESSION_OPEN_UTC,
    aggregate_session_bars,
    bars_per_day_24x7,
    minutes_since_session_open,
)
from edgar.services.intraday_strategy import _ema
from edgar.services.strategy import (
    _atr,
    _break_even_stop_candidate,
    _chandelier_stop_candidate,
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
    take_profit: float | None
    risk_pct: float
    position_size: float
    shares: float
    entry_index: int
    session_label: str
    channel_period: int
    exit_channel_period: int
    breakout_level: float
    exit_channel_level: float | None
    atr_at_entry: float
    highest_price_since_entry: float
    lowest_price_since_entry: float
    unit_count: int
    add_count: int
    next_add_price: float | None
    active_stop_loss: float | None = None
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0
    intrabar_conflict: bool = False
    exit_fill_policy: str = "stop_first"
    sizing_tier: str = "standard"
    signal_quality: str = "A"
    hold_bars: int = 0
    stop_source: str = "daily_atr_stop"
    strategy_leg: str = "session_turtle_trend"
    entry_rel_volume: float = 1.0
    volume_risk_scale: float = 1.0
    directional_volume_confirmed: bool = False
    risk_model: str = "base"
    initial_shares: float = 0.0
    initial_position_size: float = 0.0
    add_events: list[dict] = field(default_factory=list)


def _bars_per_day_for_source(interval: str, source_label: str) -> int:
    if source_label == "local_tiingo_cache":
        text = (interval or "").strip().lower()
        if text == "5m":
            return 78
        if text == "15m":
            return 26
    return bars_per_day_24x7(interval)


def _bucket_start(ts: datetime, interval_minutes: int) -> datetime:
    total_minutes = ts.hour * 60 + ts.minute
    bucket_minutes = (total_minutes // interval_minutes) * interval_minutes
    return ts.replace(
        hour=bucket_minutes // 60,
        minute=bucket_minutes % 60,
        second=0,
        microsecond=0,
    )


def _aggregate_bars_with_index_mapping(
    bars: list[dict],
    interval_minutes: int,
) -> tuple[list[dict], list[int]]:
    aggregated: list[dict] = []
    bar_to_bucket_idx: list[int] = []
    current_bucket: datetime | None = None
    current_bar: dict | None = None

    for bar in bars:
        bucket = _bucket_start(bar["timestamp"], interval_minutes)
        if bucket != current_bucket:
            if current_bar is not None:
                aggregated.append(current_bar)
            current_bucket = bucket
            current_bar = {
                "timestamp": bucket,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", 0.0) or 0.0),
            }
        else:
            current_bar["high"] = max(float(current_bar["high"]), float(bar["high"]))
            current_bar["low"] = min(float(current_bar["low"]), float(bar["low"]))
            current_bar["close"] = float(bar["close"])
            current_bar["volume"] = float(current_bar["volume"]) + float(bar.get("volume", 0.0) or 0.0)
        bar_to_bucket_idx.append(len(aggregated))

    if current_bar is not None:
        aggregated.append(current_bar)
    return aggregated, bar_to_bucket_idx


def _load_local_bars(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_source: str,
    market_data_symbol: str | None,
) -> tuple[list[dict], str, str, str]:
    source = (market_data_source or "auto").strip().lower()
    if source in {"binance", "local_binance_cache"}:
        bars, symbol = load_local_binance_klines(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_binance_cache", ""
    if source in {"tiingo", "local_tiingo_cache"}:
        bars, symbol, path = load_local_tiingo_klines(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_tiingo_cache", path
    try:
        bars, symbol = load_local_binance_klines(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_binance_cache", ""
    except Exception:
        bars, symbol, path = load_local_tiingo_klines(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_tiingo_cache", path


def run_session_turtle_trend_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "15m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    session_open: str = "tokyo_open",
    channel_period: int = 20,
    exit_channel_period: int | None = None,
    atr_period: int = 20,
    atr_stop_mult: float = 2.0,
    fixed_stop_pct: float | None = None,
    entry_window_minutes: int = 480,
    core_session_minutes: int | None = None,
    entry_buffer_bps: float = 0.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    use_volume_risk_scaling: bool = False,
    volume_period: int = 40,
    volume_risk_floor: float = 0.5,
    volume_risk_cap: float = 1.5,
    use_directional_volume_risk_boost: bool = False,
    directional_volume_min_rel_volume: float = 1.25,
    directional_volume_close_location_threshold: float = 0.65,
    directional_volume_risk_pct: float = 0.07,
    enable_pyramiding: bool = False,
    pyramid_add_atr: float = 0.5,
    max_units: int = 4,
    slippage_bps: float = 2.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
    use_break_even_stop: bool = False,
    break_even_trigger_r: float = 1.0,
    use_4h_trend_filter: bool = False,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    use_extended_hours_protective_exits_only: bool = False,
    use_chandelier_exit: bool = False,
    chandelier_period: int = 22,
    chandelier_atr_period: int = 22,
    chandelier_atr_mult: float = 3.0,
    exit_fill_policy: str = "stop_first",
) -> dict:
    del exit_fill_policy
    source = (market_data_source or "auto").strip().lower()
    if source in {"yfinance", "yf"}:
        raise ValueError("session_turtle_trend uses only local cached market data")
    valid_sessions = sorted(list(SESSION_OPEN_UTC) + ["new_york_equity_open"])
    if session_open not in SESSION_OPEN_UTC and session_open != "new_york_equity_open":
        raise ValueError(f"session_open must be one of {valid_sessions}")
    if interval.strip().lower() not in {"5m", "15m"}:
        raise ValueError("session_turtle_trend requires interval='5m' or '15m'")
    if not allow_longs and not allow_shorts:
        raise ValueError("session_turtle_trend requires at least one of allow_longs or allow_shorts")
    if channel_period not in {20, 55}:
        raise ValueError("channel_period must be 20 or 55")
    if atr_period < 5:
        raise ValueError("atr_period must be >= 5")
    if entry_window_minutes <= 0:
        raise ValueError("entry_window_minutes must be positive")
    if core_session_minutes is not None and core_session_minutes <= 0:
        raise ValueError("core_session_minutes must be positive when provided")
    if volume_period < 2:
        raise ValueError("volume_period must be >= 2")
    if atr_stop_mult <= 0:
        raise ValueError("atr_stop_mult must be positive")
    if fixed_stop_pct is not None and fixed_stop_pct <= 0:
        raise ValueError("fixed_stop_pct must be positive when provided")
    if volume_risk_floor <= 0:
        raise ValueError("volume_risk_floor must be positive")
    if volume_risk_cap < volume_risk_floor:
        raise ValueError("volume_risk_cap must be >= volume_risk_floor")
    if directional_volume_min_rel_volume <= 0:
        raise ValueError("directional_volume_min_rel_volume must be positive")
    if not 0.5 <= directional_volume_close_location_threshold <= 1.0:
        raise ValueError("directional_volume_close_location_threshold must be between 0.5 and 1.0")
    if directional_volume_risk_pct < base_risk_pct:
        raise ValueError("directional_volume_risk_pct must be >= base_risk_pct")
    if max_units < 1:
        raise ValueError("max_units must be >= 1")
    if trend_fast_period < 2 or trend_slow_period < 2:
        raise ValueError("trend_fast_period and trend_slow_period must be >= 2")
    if trend_fast_period >= trend_slow_period:
        raise ValueError("trend_fast_period must be smaller than trend_slow_period")

    core_session_window = int(core_session_minutes) if core_session_minutes is not None else int(entry_window_minutes)
    active_entry_window = (
        min(int(entry_window_minutes), core_session_window)
        if use_extended_hours_protective_exits_only
        else int(entry_window_minutes)
    )
    exit_channel = exit_channel_period if exit_channel_period is not None else (10 if channel_period == 20 else 20)
    preload_warmup_days = max(channel_period + exit_channel + atr_period + 20, 75)
    if use_4h_trend_filter:
        preload_warmup_days = max(preload_warmup_days, trend_slow_period + 40)

    bars, resolved_symbol, resolved_source, market_data_path = _load_local_bars(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=preload_warmup_days,
        market_data_source=market_data_source,
        market_data_symbol=market_data_symbol,
    )
    bars_per_day = _bars_per_day_for_source(interval=interval, source_label=resolved_source)
    if resolved_source == "local_tiingo_cache" and session_open != "new_york_equity_open":
        raise ValueError("Local Tiingo equity cache only supports session_open='new_york_equity_open'")
    warmup_bars = max(
        (channel_period + exit_channel + atr_period) * bars_per_day,
        chandelier_period + 10,
        1800,
    )
    if len(bars) < warmup_bars + 30:
        raise ValueError(f"Insufficient cached market data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [float(b["open"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    closes = [float(b["close"]) for b in bars]
    volumes = [float(b.get("volume", 0.0) or 0.0) for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 90)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    session_bars, bar_to_session_idx = aggregate_session_bars(bars, session_open)
    session_highs = [float(bar.high) for bar in session_bars]
    session_lows = [float(bar.low) for bar in session_bars]
    session_closes = [float(bar.close) for bar in session_bars]
    daily_atr = _atr(session_highs, session_lows, session_closes, atr_period)
    channel_highs = _rolling_highest(session_highs, channel_period)
    channel_lows = _rolling_lowest(session_lows, channel_period)
    exit_highs = _rolling_highest(session_highs, exit_channel)
    exit_lows = _rolling_lowest(session_lows, exit_channel)
    vol_sma = _sma(volumes, volume_period)
    trend_bars, bar_to_trend_idx = _aggregate_bars_with_index_mapping(bars, interval_minutes=240)
    trend_closes = [float(bar["close"]) for bar in trend_bars]
    trend_ema_fast = _ema(trend_closes, trend_fast_period) if use_4h_trend_filter else []
    trend_ema_slow = _ema(trend_closes, trend_slow_period) if use_4h_trend_filter else []
    chandelier_atr_vals = _atr(highs, lows, closes, chandelier_atr_period) if use_chandelier_exit else []
    chandelier_highs = _rolling_highest(highs, chandelier_period) if use_chandelier_exit else []
    chandelier_lows = _rolling_lowest(lows, chandelier_period) if use_chandelier_exit else []

    capital = float(initial_capital)
    peak_equity = capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0
    trades: list[_Trade] = []
    equity_curve: list[dict] = []
    open_trade: _Trade | None = None

    commission_rate = max(commission_bps, 0.0) / 10_000.0
    slippage_rate = max(slippage_bps, 0.0) / 10_000.0
    entry_buffer = max(entry_buffer_bps, 0.0) / 10_000.0

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
        open_i = opens[i]
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        minutes_open = minutes_since_session_open(ts, session_open)
        current_session_idx = bar_to_session_idx[i]
        completed_session_idx = current_session_idx - 1
        outside_core_session = use_extended_hours_protective_exits_only and minutes_open >= core_session_window

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            if open_trade.direction == "long":
                current_exit_channel = (
                    float(exit_lows[completed_session_idx])
                    if completed_session_idx >= exit_channel and exit_lows[completed_session_idx] is not None
                    else open_trade.exit_channel_level
                )
                protective_stop = current_stop
                stop_reason = "stop_loss"
                if not outside_core_session and current_exit_channel is not None and current_exit_channel > protective_stop:
                    protective_stop = current_exit_channel
                    stop_reason = "exit_channel"
                if low_i <= protective_stop:
                    if outside_core_session:
                        reason = "extended_hours_protective_stop"
                    else:
                        trailed_stop = _stop_is_trailed("long", open_trade.stop_loss, current_stop)
                        break_even_stop = use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
                            direction="long",
                            entry_price=open_trade.entry_price,
                            active_stop=current_stop,
                        )
                        if stop_reason == "stop_loss" and break_even_stop:
                            reason = "break_even_stop"
                        elif stop_reason == "stop_loss" and use_chandelier_exit and trailed_stop:
                            reason = "chandelier_stop"
                        else:
                            reason = stop_reason
                    _close_trade(open_trade, i, protective_stop, reason)
                    open_trade = None
                else:
                    open_trade.highest_price_since_entry = max(open_trade.highest_price_since_entry, high_i)
                    atr_now = (
                        float(daily_atr[completed_session_idx])
                        if completed_session_idx >= 0 and daily_atr[completed_session_idx] is not None
                        else open_trade.atr_at_entry
                    )
                    if not outside_core_session:
                        next_stop = current_stop
                        if use_break_even_stop:
                            next_stop = _tighten_stop(
                                direction="long",
                                current_stop=next_stop,
                                candidate_stop=_break_even_stop_candidate(
                                    direction="long",
                                    entry_price=open_trade.entry_price,
                                    initial_stop=open_trade.stop_loss,
                                    bar_high=high_i,
                                    bar_low=low_i,
                                    trigger_r=break_even_trigger_r,
                                ),
                                take_profit=None,
                            )
                        if fixed_stop_pct is None:
                            trend_stop = open_trade.highest_price_since_entry - (atr_now * atr_stop_mult)
                            next_stop = _tighten_stop(
                                direction="long",
                                current_stop=next_stop,
                                candidate_stop=trend_stop,
                                take_profit=None,
                            )
                        tr_stop = None
                        if use_chandelier_exit:
                            tr_stop = _chandelier_stop_candidate(
                                direction="long",
                                idx=i,
                                rolling_highs=chandelier_highs,
                                rolling_lows=chandelier_lows,
                                atr_values=chandelier_atr_vals,
                                atr_mult=chandelier_atr_mult,
                            )
                        open_trade.active_stop_loss = round(
                            _tighten_stop(
                                direction="long",
                                current_stop=next_stop,
                                candidate_stop=tr_stop,
                                take_profit=None,
                            ),
                            4,
                        )
                        open_trade.exit_channel_level = current_exit_channel

                    if (
                        enable_pyramiding
                        and open_trade.unit_count < max_units
                        and open_trade.next_add_price is not None
                        and close_i >= open_trade.next_add_price
                        and minutes_open < active_entry_window
                        and i < len(bars) - 1
                        and completed_session_idx >= 0
                    ):
                        atr_add = (
                            float(daily_atr[completed_session_idx])
                            if daily_atr[completed_session_idx] is not None
                            else open_trade.atr_at_entry
                        )
                        if atr_add > 0:
                            add_entry = (opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]) * (1.0 + slippage_rate)
                            add_risk = atr_add * atr_stop_mult
                            add_shares = (capital * base_risk_pct) / add_risk if add_risk > 0 else 0.0
                            add_notional = add_shares * add_entry
                            cap_notional = capital * max_position_pct
                            available_notional = max(cap_notional - open_trade.position_size, 0.0)
                            if add_notional > available_notional and add_entry > 0:
                                add_shares = available_notional / add_entry
                                add_notional = add_shares * add_entry
                            if add_shares > 0 and add_notional > 0:
                                fee = add_notional * commission_rate
                                total_shares = open_trade.shares + add_shares
                                weighted_entry = (
                                    (open_trade.entry_price * open_trade.shares) + (add_entry * add_shares)
                                ) / total_shares
                                open_trade.entry_price = round(weighted_entry, 4)
                                open_trade.shares = round(total_shares, 6)
                                open_trade.position_size = round(open_trade.position_size + add_notional, 4)
                                open_trade.fees_paid = round(open_trade.fees_paid + fee, 4)
                                open_trade.unit_count += 1
                                open_trade.add_count += 1
                                open_trade.add_events.append(
                                    {
                                        "timestamp": timestamps[i + 1],
                                        "entry_price": round(add_entry, 4),
                                        "shares": round(add_shares, 6),
                                        "notional": round(add_notional, 4),
                                        "fee": round(fee, 4),
                                    }
                                )
                                open_trade.next_add_price = (
                                    round(add_entry + (atr_add * pyramid_add_atr), 4)
                                    if open_trade.unit_count < max_units
                                    else None
                                )
                                open_trade.active_stop_loss = round(
                                    _tighten_stop(
                                        direction="long",
                                        current_stop=open_trade.active_stop_loss,
                                        candidate_stop=add_entry - (atr_add * atr_stop_mult),
                                        take_profit=None,
                                    ),
                                    4,
                                )
            else:
                current_exit_channel = (
                    float(exit_highs[completed_session_idx])
                    if completed_session_idx >= exit_channel and exit_highs[completed_session_idx] is not None
                    else open_trade.exit_channel_level
                )
                protective_stop = current_stop
                stop_reason = "stop_loss"
                if not outside_core_session and current_exit_channel is not None and current_exit_channel < protective_stop:
                    protective_stop = current_exit_channel
                    stop_reason = "exit_channel"
                if high_i >= protective_stop:
                    if outside_core_session:
                        reason = "extended_hours_protective_stop"
                    else:
                        trailed_stop = _stop_is_trailed("short", open_trade.stop_loss, current_stop)
                        break_even_stop = use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
                            direction="short",
                            entry_price=open_trade.entry_price,
                            active_stop=current_stop,
                        )
                        if stop_reason == "stop_loss" and break_even_stop:
                            reason = "break_even_stop"
                        elif stop_reason == "stop_loss" and use_chandelier_exit and trailed_stop:
                            reason = "chandelier_stop"
                        else:
                            reason = stop_reason
                    _close_trade(open_trade, i, protective_stop, reason)
                    open_trade = None
                else:
                    open_trade.lowest_price_since_entry = min(open_trade.lowest_price_since_entry, low_i)
                    atr_now = (
                        float(daily_atr[completed_session_idx])
                        if completed_session_idx >= 0 and daily_atr[completed_session_idx] is not None
                        else open_trade.atr_at_entry
                    )
                    if not outside_core_session:
                        next_stop = current_stop
                        if use_break_even_stop:
                            next_stop = _tighten_stop(
                                direction="short",
                                current_stop=next_stop,
                                candidate_stop=_break_even_stop_candidate(
                                    direction="short",
                                    entry_price=open_trade.entry_price,
                                    initial_stop=open_trade.stop_loss,
                                    bar_high=high_i,
                                    bar_low=low_i,
                                    trigger_r=break_even_trigger_r,
                                ),
                                take_profit=None,
                            )
                        if fixed_stop_pct is None:
                            trend_stop = open_trade.lowest_price_since_entry + (atr_now * atr_stop_mult)
                            next_stop = _tighten_stop(
                                direction="short",
                                current_stop=next_stop,
                                candidate_stop=trend_stop,
                                take_profit=None,
                            )
                        tr_stop = None
                        if use_chandelier_exit:
                            tr_stop = _chandelier_stop_candidate(
                                direction="short",
                                idx=i,
                                rolling_highs=chandelier_highs,
                                rolling_lows=chandelier_lows,
                                atr_values=chandelier_atr_vals,
                                atr_mult=chandelier_atr_mult,
                            )
                        open_trade.active_stop_loss = round(
                            _tighten_stop(
                                direction="short",
                                current_stop=next_stop,
                                candidate_stop=tr_stop,
                                take_profit=None,
                            ),
                            4,
                        )
                        open_trade.exit_channel_level = current_exit_channel

                    if (
                        enable_pyramiding
                        and open_trade.unit_count < max_units
                        and open_trade.next_add_price is not None
                        and close_i <= open_trade.next_add_price
                        and minutes_open < active_entry_window
                        and i < len(bars) - 1
                        and completed_session_idx >= 0
                    ):
                        atr_add = (
                            float(daily_atr[completed_session_idx])
                            if daily_atr[completed_session_idx] is not None
                            else open_trade.atr_at_entry
                        )
                        if atr_add > 0:
                            add_entry = (opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]) * (1.0 - slippage_rate)
                            add_risk = atr_add * atr_stop_mult
                            add_shares = (capital * base_risk_pct) / add_risk if add_risk > 0 else 0.0
                            add_notional = add_shares * add_entry
                            cap_notional = capital * max_position_pct
                            available_notional = max(cap_notional - open_trade.position_size, 0.0)
                            if add_notional > available_notional and add_entry > 0:
                                add_shares = available_notional / add_entry
                                add_notional = add_shares * add_entry
                            if add_shares > 0 and add_notional > 0:
                                fee = add_notional * commission_rate
                                total_shares = open_trade.shares + add_shares
                                weighted_entry = (
                                    (open_trade.entry_price * open_trade.shares) + (add_entry * add_shares)
                                ) / total_shares
                                open_trade.entry_price = round(weighted_entry, 4)
                                open_trade.shares = round(total_shares, 6)
                                open_trade.position_size = round(open_trade.position_size + add_notional, 4)
                                open_trade.fees_paid = round(open_trade.fees_paid + fee, 4)
                                open_trade.unit_count += 1
                                open_trade.add_count += 1
                                open_trade.add_events.append(
                                    {
                                        "timestamp": timestamps[i + 1],
                                        "entry_price": round(add_entry, 4),
                                        "shares": round(add_shares, 6),
                                        "notional": round(add_notional, 4),
                                        "fee": round(fee, 4),
                                    }
                                )
                                open_trade.next_add_price = (
                                    round(add_entry - (atr_add * pyramid_add_atr), 4)
                                    if open_trade.unit_count < max_units
                                    else None
                                )
                                open_trade.active_stop_loss = round(
                                    _tighten_stop(
                                        direction="short",
                                        current_stop=open_trade.active_stop_loss,
                                        candidate_stop=add_entry + (atr_add * atr_stop_mult),
                                        take_profit=None,
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
        if ts < period_start or i >= len(bars) - 1 or minutes_open >= active_entry_window:
            continue
        if completed_session_idx < max(channel_period, atr_period, exit_channel):
            continue

        atr_now = daily_atr[completed_session_idx]
        long_breakout = channel_highs[completed_session_idx]
        short_breakout = channel_lows[completed_session_idx]
        exit_channel_low = exit_lows[completed_session_idx]
        exit_channel_high = exit_highs[completed_session_idx]
        if atr_now is None or atr_now <= 0:
            continue

        prev_close = closes[i - 1]
        direction: str | None = None
        breakout_level: float | None = None
        exit_channel_level: float | None = None
        allow_long_signal = allow_longs
        allow_short_signal = allow_shorts

        if use_4h_trend_filter:
            completed_trend_idx = bar_to_trend_idx[i] - 1
            if completed_trend_idx < trend_slow_period - 1:
                continue
            fast_now = trend_ema_fast[completed_trend_idx]
            slow_now = trend_ema_slow[completed_trend_idx]
            trend_close = trend_closes[completed_trend_idx]
            if fast_now is None or slow_now is None:
                continue
            allow_long_signal = allow_longs and trend_close > float(fast_now) and float(fast_now) > float(slow_now)
            allow_short_signal = allow_shorts and trend_close < float(fast_now) and float(fast_now) < float(slow_now)

        if allow_long_signal and long_breakout is not None:
            trigger_high = float(long_breakout) * (1.0 + entry_buffer)
            if close_i > trigger_high and prev_close <= trigger_high:
                direction = "long"
                breakout_level = float(long_breakout)
                exit_channel_level = float(exit_channel_low) if exit_channel_low is not None else None

        if direction is None and allow_short_signal and short_breakout is not None:
            trigger_low = float(short_breakout) * (1.0 - entry_buffer)
            if close_i < trigger_low and prev_close >= trigger_low:
                direction = "short"
                breakout_level = float(short_breakout)
                exit_channel_level = float(exit_channel_high) if exit_channel_high is not None else None

        if direction is None or breakout_level is None:
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and float(vol_sma[i]) > 0:
            rel_volume = float(volumes[i]) / float(vol_sma[i])
        volume_risk_scale = 1.0
        if use_volume_risk_scaling:
            volume_risk_scale = min(max(rel_volume, volume_risk_floor), volume_risk_cap)
        bar_range = max(high_i - low_i, 1e-8)
        close_location = (close_i - low_i) / bar_range
        directional_volume_confirmed = rel_volume >= directional_volume_min_rel_volume and (
            close_location >= directional_volume_close_location_threshold
            if direction == "long"
            else close_location <= (1.0 - directional_volume_close_location_threshold)
        )
        target_risk_pct = base_risk_pct * volume_risk_scale
        risk_model = "base"
        if use_volume_risk_scaling and volume_risk_scale != 1.0:
            risk_model = "rvol_scaled"
        if use_directional_volume_risk_boost and directional_volume_confirmed:
            target_risk_pct = max(target_risk_pct, directional_volume_risk_pct)
            risk_model = "directional_volume_boost"

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        fixed_stop_rate = float(fixed_stop_pct) if fixed_stop_pct is not None else None
        if direction == "long":
            entry_price = next_open * (1.0 + slippage_rate)
            if fixed_stop_rate is not None:
                stop_loss = entry_price * (1.0 - fixed_stop_rate)
            else:
                stop_loss = entry_price - (float(atr_now) * atr_stop_mult)
            next_add_price = (
                round(entry_price + (float(atr_now) * pyramid_add_atr), 4)
                if enable_pyramiding and max_units > 1
                else None
            )
        else:
            entry_price = next_open * (1.0 - slippage_rate)
            if fixed_stop_rate is not None:
                stop_loss = entry_price * (1.0 + fixed_stop_rate)
            else:
                stop_loss = entry_price + (float(atr_now) * atr_stop_mult)
            next_add_price = (
                round(entry_price - (float(atr_now) * pyramid_add_atr), 4)
                if enable_pyramiding and max_units > 1
                else None
            )

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            continue
        risk_amount = capital * target_risk_pct
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
        stop_source = "fixed_pct_stop" if fixed_stop_rate is not None else "daily_atr_stop"
        open_trade = _Trade(
            direction=direction,
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=None,
            risk_pct=round(risk_amount / capital if capital > 0 else 0.0, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            session_label=session_open,
            channel_period=channel_period,
            exit_channel_period=exit_channel,
            breakout_level=round(breakout_level, 4),
            exit_channel_level=round(exit_channel_level, 4) if exit_channel_level is not None else None,
            atr_at_entry=round(float(atr_now), 4),
            highest_price_since_entry=round(max(high_i, entry_price), 4),
            lowest_price_since_entry=round(min(low_i, entry_price), 4),
            unit_count=1,
            add_count=0,
            next_add_price=next_add_price,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            sizing_tier=sizing_tier,
            signal_quality="A" if channel_period == 55 else "B",
            stop_source=stop_source,
            entry_rel_volume=round(rel_volume, 4),
            volume_risk_scale=round(volume_risk_scale, 4),
            directional_volume_confirmed=directional_volume_confirmed,
            risk_model=risk_model,
            initial_shares=round(shares, 6),
            initial_position_size=round(position_size, 4),
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
        "strategy_name": "Session Turtle Trend",
        "data_mode": "intraday",
        "interval": interval,
        "requested_interval": interval,
        "effective_interval": interval,
        "interval_adjustment": None,
        "market_data_source": resolved_source,
        "market_data_symbol": resolved_symbol,
        "market_data_path": market_data_path,
        "strategy_variant": "session_turtle_trend",
        "bias_model": f"{channel_period}d_breakout_long_short",
        "entry_session": session_open,
        "session_open": session_open,
        "channel_period": channel_period,
        "exit_channel_period": exit_channel,
        "atr_period": atr_period,
        "atr_stop_mult": atr_stop_mult,
        "fixed_stop_pct": fixed_stop_pct,
        "entry_window_minutes": entry_window_minutes,
        "core_session_minutes": core_session_window,
        "allow_longs": allow_longs,
        "allow_shorts": allow_shorts,
        "use_4h_trend_filter": use_4h_trend_filter,
        "trend_filter_interval": "4h" if use_4h_trend_filter else None,
        "trend_fast_period": trend_fast_period,
        "trend_slow_period": trend_slow_period,
        "use_extended_hours_protective_exits_only": use_extended_hours_protective_exits_only,
        "use_volume_risk_scaling": use_volume_risk_scaling,
        "volume_period": volume_period,
        "volume_risk_floor": volume_risk_floor,
        "volume_risk_cap": volume_risk_cap,
        "use_directional_volume_risk_boost": use_directional_volume_risk_boost,
        "directional_volume_min_rel_volume": directional_volume_min_rel_volume,
        "directional_volume_close_location_threshold": directional_volume_close_location_threshold,
        "directional_volume_risk_pct": directional_volume_risk_pct,
        "enable_pyramiding": enable_pyramiding,
        "pyramid_add_atr": pyramid_add_atr,
        "max_units": max_units,
        "use_chandelier_exit": use_chandelier_exit,
        "use_break_even_stop": use_break_even_stop,
        "break_even_trigger_r": break_even_trigger_r,
        "chandelier_period": chandelier_period,
        "chandelier_atr_period": chandelier_atr_period,
        "chandelier_atr_mult": chandelier_atr_mult,
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
                "strategy_leg": t.strategy_leg,
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
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "entry_rel_volume": t.entry_rel_volume,
                "volume_risk_scale": t.volume_risk_scale,
                "directional_volume_confirmed": t.directional_volume_confirmed,
                "risk_model": t.risk_model,
                "session_label": t.session_label,
                "breakout_level": t.breakout_level,
                "exit_channel_level": t.exit_channel_level,
                "atr_at_entry": t.atr_at_entry,
                "highest_price_since_entry": t.highest_price_since_entry,
                "lowest_price_since_entry": t.lowest_price_since_entry,
                "initial_shares": t.initial_shares,
                "initial_position_size": t.initial_position_size,
                "unit_count": t.unit_count,
                "add_count": t.add_count,
                "next_add_price": t.next_add_price,
                "add_events": [
                    {
                        "timestamp": event["timestamp"].isoformat(),
                        "entry_price": event["entry_price"],
                        "shares": event["shares"],
                        "notional": event["notional"],
                        "fee": event["fee"],
                    }
                    for event in t.add_events
                ],
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
