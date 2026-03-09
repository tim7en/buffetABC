"""Opening Pressure + Turtle hybrid strategy.

Design:
- ORB short leg hunts downside momentum after the U.S. cash open.
- Turtle long leg buys 15m Donchian breakouts when trend and volume confirm.
- Both legs flatten at the U.S. session close.

Notes:
- Built for 5m or 15m execution data.
- 15m state is aggregated from the execution bars and only completed 15m bars are used.
- Daily trend uses completed New York session bars only.
- The service supports Binance crypto pairs and compatible Yahoo intraday windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from edgar.services.binance_data import fetch_binance_klines
from edgar.services.intraday_strategy import _ema
from edgar.services.market_mechanics_strategy import (
    _bars_per_day,
    _fetch_intraday_bars,
    _interval_to_minutes,
    _max_lookback_days_for_interval,
    _resolve_effective_interval,
)
from edgar.services.session_sfp_fvg_strategy import (
    _bars_per_day_24x7,
    _resolve_market_data_source,
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
class _SessionBar:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class _Trade:
    direction: str
    leg: str
    entry_ts: datetime
    entry_price: float
    stop_loss: float
    take_profit: float | None
    risk_pct: float
    position_size: float
    shares: float
    entry_index: int
    capital_at_entry: float
    orb_high: float | None
    orb_low: float | None
    donchian_upper: float | None
    donchian_lower: float | None
    trend_ema: float | None
    active_stop_loss: float | None
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
    stop_source: str = ""
    session_label: str = "other"
    full_trail_mode: bool = False
    use_chandelier_exit: bool = False
    use_break_even_stop: bool = False
    portfolio_gate_active: bool = False
    short_meta_gate_active: bool = False
    signal_score: int = 0
    time_stop_minutes: int = 0


def _aggregate_timeframe(bars: list[dict], minutes: int) -> tuple[list[dict], list[int]]:
    bucket_ms = max(minutes, 1) * 60 * 1000
    aggregated: list[dict] = []
    bar_to_bucket: list[int] = []
    current_bucket: int | None = None
    current: dict | None = None

    for bar in bars:
        ts = bar["timestamp"].replace(tzinfo=timezone.utc)
        bucket = int(ts.timestamp() * 1000) // bucket_ms * bucket_ms
        bucket_ts = datetime.fromtimestamp(bucket / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        if bucket != current_bucket:
            if current is not None:
                aggregated.append(current)
            current_bucket = bucket
            current = {
                "timestamp": bucket_ts,
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
        bar_to_bucket.append(len(aggregated))

    if current is not None:
        aggregated.append(current)
    return aggregated, bar_to_bucket


def _aggregate_ny_sessions(bars: list[dict]) -> tuple[list[_SessionBar], list[int]]:
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


def _is_new_tf_bucket(bar_to_bucket: list[int], idx: int) -> bool:
    if idx <= 0 or idx >= len(bar_to_bucket):
        return False
    return bar_to_bucket[idx] != bar_to_bucket[idx - 1]


def _session_label(ts: datetime) -> str:
    utc_ts = ts.replace(tzinfo=timezone.utc)
    ny_ts = _to_new_york(ts)
    if 0 <= utc_ts.hour < 8:
        return "asia"
    if (ny_ts.hour, ny_ts.minute) >= (9, 30) and (ny_ts.hour, ny_ts.minute) < (16, 0):
        return "new_york"
    return "other"


def _bar_crosses_us_close(ts: datetime, interval_minutes: int) -> bool:
    local = _to_new_york(ts)
    close_ts = local.replace(hour=16, minute=0, second=0, microsecond=0)
    bar_end = local + timedelta(minutes=max(interval_minutes, 1))
    return local < close_ts <= bar_end


def _minutes_since_ny_open(ts: datetime) -> int:
    local = _to_new_york(ts)
    open_ts = local.replace(hour=9, minute=30, second=0, microsecond=0)
    return int((local - open_ts).total_seconds() / 60)


def _minutes_until_us_close(ts: datetime) -> int:
    local = _to_new_york(ts)
    close_ts = local.replace(hour=16, minute=0, second=0, microsecond=0)
    return int((close_ts - local).total_seconds() / 60)


def _build_opening_ranges(
    timestamps: list[datetime],
    highs: list[float],
    lows: list[float],
    orb_window_minutes: int,
) -> dict[date, dict]:
    out: dict[date, dict] = {}
    for idx, ts in enumerate(timestamps):
        local = _to_new_york(ts)
        session_date = local.date()
        open_ts = local.replace(hour=9, minute=30, second=0, microsecond=0)
        orb_end = open_ts + timedelta(minutes=max(orb_window_minutes, 1))
        entry = out.setdefault(
            session_date,
            {
                "high": None,
                "low": None,
                "count": 0,
                "open_ts": open_ts,
                "orb_end": orb_end,
            },
        )
        if open_ts <= local < orb_end:
            entry["high"] = highs[idx] if entry["high"] is None else max(entry["high"], highs[idx])
            entry["low"] = lows[idx] if entry["low"] is None else min(entry["low"], lows[idx])
            entry["count"] += 1
    return out


def _orb_signal_score(
    rel_volume: float,
    breakdown_close: float,
    orb_low: float,
    orb_high: float,
    ema_15m: float | None,
    daily_slope_negative: bool,
    minutes_since_open: int,
) -> tuple[int, str]:
    score = 0
    orb_range = max(orb_high - orb_low, 1e-8)
    extension = max((orb_low - breakdown_close) / orb_range, 0.0)
    if rel_volume >= 1.25:
        score += 1
    if extension >= 0.25:
        score += 1
    if ema_15m is not None and breakdown_close < ema_15m:
        score += 1
    if daily_slope_negative:
        score += 1
    if 0 <= minutes_since_open <= 30:
        score += 1
    quality = "A" if score >= 4 else ("B" if score >= 2 else "C")
    return score, quality


def _recent_closed_return_pct(closed_meta: list[dict], session_filter: set[str], lookback: int, capital: float) -> float:
    if lookback <= 0 or capital <= 0:
        return 0.0
    selected = [m for m in closed_meta if m.get("entry_session") in session_filter][-lookback:]
    if not selected:
        return 0.0
    return (sum(float(m.get("pnl", 0.0)) for m in selected) / capital) * 100.0


def _recent_short_loss_count(closed_meta: list[dict], lookback: int) -> int:
    if lookback <= 0:
        return 0
    recent = [m for m in closed_meta if m.get("leg") == "orb_short"][-lookback:]
    return sum(1 for m in recent if float(m.get("pnl", 0.0)) < 0)


def run_orb_turtle_hybrid_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    auto_adjust_for_yf_limits: bool = True,
    allow_longs: bool = True,
    allow_shorts: bool = True,
    orb_window_minutes: int = 15,
    use_five_minute_or: bool = False,
    orb_entry_window_minutes: int = 120,
    orb_break_buffer_bps: float = 0.0,
    donchian_period: int = 200,
    ema_period: int = 200,
    daily_ema_period: int = 5,
    rvol_period: int = 40,
    min_rel_volume: float = 1.0,
    long_risk_pct: float = 0.015,
    short_risk_pct: float = 0.03,
    max_position_pct: float = 0.30,
    stop_buffer_bps: float = 5.0,
    orb_rr_multiple: float = 2.0,
    short_time_stop_minutes: int = 120,
    short_time_stop_min_r: float = 0.5,
    short_full_trail_score_threshold: int = 3,
    turtle_initial_stop_atr_period: int = 20,
    turtle_initial_stop_atr_mult: float = 2.0,
    use_break_even_stop: bool = False,
    break_even_trigger_r: float = 1.0,
    chandelier_period: int = 22,
    chandelier_atr_period: int = 22,
    chandelier_atr_mult: float = 3.0,
    portfolio_gate_lookback: int = 8,
    portfolio_gate_threshold_pct: float = -1.0,
    portfolio_gate_risk_scale: float = 0.5,
    short_meta_lookback: int = 4,
    short_meta_loss_threshold: int = 2,
    short_meta_risk_scale: float = 0.5,
    slippage_bps: float = 4.0,
    commission_bps: float = 1.0,
    exit_fill_policy: str = "stop_first",
) -> dict:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if exit_fill_policy not in {"stop_first", "target_first"}:
        raise ValueError("exit_fill_policy must be one of ['stop_first', 'target_first']")
    minutes = _interval_to_minutes(interval)
    if minutes not in {5, 15}:
        raise ValueError("orb_turtle_hybrid requires a 5m or 15m execution interval")
    if use_five_minute_or:
        orb_window_minutes = 5
    if orb_window_minutes not in {5, 15}:
        raise ValueError("orb_window_minutes must be 5 or 15")
    if minutes > orb_window_minutes:
        raise ValueError("Execution interval cannot be larger than the selected opening range window")
    if donchian_period < 20 or ema_period < 20:
        raise ValueError("donchian_period and ema_period must be >= 20")

    resolved_source = _resolve_market_data_source(market_data_source=market_data_source, ticker=ticker)
    requested_interval = interval
    interval_adjustment = None
    if resolved_source == "yfinance":
        effective_interval, interval_adjustment = _resolve_effective_interval(
            requested_interval=requested_interval,
            lookback_years=lookback_years,
            auto_adjust_for_yf_limits=auto_adjust_for_yf_limits,
        )
        if _interval_to_minutes(effective_interval) not in {5, 15}:
            raise ValueError(
                "orb_turtle_hybrid requires 5m or 15m data. "
                "Use Binance for multi-year crypto intraday history or shorten the Yahoo window."
            )
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

    effective_minutes = _interval_to_minutes(effective_interval)
    if effective_minutes not in {5, 15}:
        raise ValueError("orb_turtle_hybrid requires a 5m or 15m effective interval")

    bars_per_day = (
        max(_bars_per_day(effective_interval), 1)
        if resolved_source == "yfinance"
        else _bars_per_day_24x7(effective_interval)
    )
    warmup_bars = max(donchian_period * 3, ema_period * 3, rvol_period * 4, 2200)
    warmup_days = max(int(warmup_bars / bars_per_day) + 20, 45)

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
    if len(bars) < warmup_bars + 50:
        raise ValueError(f"Insufficient intraday data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [float(b["open"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    closes = [float(b["close"]) for b in bars]
    volumes = [float(b.get("volume", 0.0) or 0.0) for b in bars]
    session_labels = [_session_label(ts) for ts in timestamps]
    ny_dates = [_to_new_york(ts).date() for ts in timestamps]

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)

    tf_bars, bar_to_tf_idx = _aggregate_timeframe(bars, 15)
    tf_highs = [float(b["high"]) for b in tf_bars]
    tf_lows = [float(b["low"]) for b in tf_bars]
    tf_closes = [float(b["close"]) for b in tf_bars]
    tf_volumes = [float(b.get("volume", 0.0) or 0.0) for b in tf_bars]
    tf_ema = _ema(tf_closes, ema_period)
    tf_atr = _atr(tf_highs, tf_lows, tf_closes, turtle_initial_stop_atr_period)
    tf_vol_sma = _sma(tf_volumes, rvol_period)
    tf_donchian_high = _rolling_highest(tf_highs, donchian_period)
    tf_donchian_low = _rolling_lowest(tf_lows, donchian_period)
    tf_donchian_prev_high: list[float | None] = [None] * len(tf_bars)
    tf_donchian_prev_low: list[float | None] = [None] * len(tf_bars)
    for idx in range(1, len(tf_bars)):
        tf_donchian_prev_high[idx] = tf_donchian_high[idx - 1]
        tf_donchian_prev_low[idx] = tf_donchian_low[idx - 1]

    session_bars, bar_to_session_idx = _aggregate_ny_sessions(bars)
    session_closes = [float(b.close) for b in session_bars]
    daily_ema = _ema(session_closes, daily_ema_period)

    opening_ranges = _build_opening_ranges(
        timestamps=timestamps,
        highs=highs,
        lows=lows,
        orb_window_minutes=orb_window_minutes,
    )

    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    capital = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0
    equity_curve: list[dict] = []
    trades: list[_Trade] = []
    closed_meta: list[dict] = []
    open_trade: _Trade | None = None
    day_leg_taken: set[tuple[date, str]] = set()

    commission_rate = max(commission_bps, 0.0) / 10_000.0
    slippage_rate = max(slippage_bps, 0.0) / 10_000.0
    stop_buffer = max(stop_buffer_bps, 0.0) / 10_000.0
    orb_break_buffer = max(orb_break_buffer_bps, 0.0) / 10_000.0

    chandelier_atr_vals = _atr(highs, lows, closes, chandelier_atr_period)
    chandelier_highs = _rolling_highest(highs, chandelier_period)
    chandelier_lows = _rolling_lowest(lows, chandelier_period)

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
        closed_meta.append(
            {
                "leg": trade.leg,
                "direction": trade.direction,
                "entry_session": trade.session_label,
                "pnl": net_pnl,
                "capital_at_entry": trade.capital_at_entry,
            }
        )

    for i in range(start_idx, len(bars)):
        ts = timestamps[i]
        open_i = opens[i]
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        session_date = ny_dates[i]
        session_label = session_labels[i]

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            if open_trade.take_profit is not None:
                raw_exit, hit_sl, hit_tp, intrabar_conflict = _resolve_bar_bracket_exit(
                    direction=open_trade.direction,
                    bar_high=high_i,
                    bar_low=low_i,
                    stop_loss=current_stop,
                    take_profit=open_trade.take_profit,
                    fill_policy=exit_fill_policy,
                )
            else:
                if open_trade.direction == "long":
                    hit_sl = low_i <= current_stop
                else:
                    hit_sl = high_i >= current_stop
                hit_tp = False
                intrabar_conflict = False
                raw_exit = current_stop if hit_sl else None
            if raw_exit is not None:
                open_trade.intrabar_conflict = intrabar_conflict
                trailed_stop = _stop_is_trailed(open_trade.direction, open_trade.stop_loss, current_stop)
                break_even_stop = open_trade.use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
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
                            if hit_sl and open_trade.use_chandelier_exit and trailed_stop
                            else ("stop_loss" if hit_sl else "take_profit")
                        )
                    ),
                )
                open_trade = None
            else:
                elapsed_minutes = max((i - open_trade.entry_index + 1) * effective_minutes, 0)
                if open_trade.leg == "orb_short" and elapsed_minutes >= short_time_stop_minutes:
                    initial_risk = max(abs(open_trade.entry_price - open_trade.stop_loss), 1e-8)
                    current_r = (open_trade.entry_price - close_i) / initial_risk
                    if current_r < short_time_stop_min_r:
                        _close_trade(open_trade, i, close_i, "time_stop")
                        open_trade = None
                if open_trade is not None and _bar_crosses_us_close(ts, effective_minutes):
                    _close_trade(open_trade, i, close_i, "session_close")
                    open_trade = None
                if open_trade is not None:
                    next_stop = current_stop
                    if open_trade.use_break_even_stop:
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
                    if open_trade.use_chandelier_exit:
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
        if _minutes_until_us_close(ts) < effective_minutes:
            continue

        portfolio_gate_return = _recent_closed_return_pct(
            closed_meta=closed_meta,
            session_filter={"asia", "new_york"},
            lookback=portfolio_gate_lookback,
            capital=capital,
        )
        portfolio_gate_active = portfolio_gate_return <= portfolio_gate_threshold_pct
        portfolio_risk_scale = portfolio_gate_risk_scale if portfolio_gate_active else 1.0

        current_session_idx = bar_to_session_idx[i]
        completed_session_idx = current_session_idx - 1
        daily_slope_positive = False
        daily_slope_negative = False
        if completed_session_idx >= 1 and daily_ema[completed_session_idx] is not None and daily_ema[completed_session_idx - 1] is not None:
            slope = float(daily_ema[completed_session_idx]) - float(daily_ema[completed_session_idx - 1])
            daily_slope_positive = slope > 0
            daily_slope_negative = slope < 0
        else:
            slope = None

        # ORB short leg: first decisive break of the U.S. opening range.
        if allow_shorts and (session_date, "orb_short") not in day_leg_taken:
            orb = opening_ranges.get(session_date)
            if orb and orb.get("count", 0) > 0:
                local = _to_new_york(ts)
                orb_end = orb["orb_end"]
                entry_end = orb_end + timedelta(minutes=max(orb_entry_window_minutes, 1))
                prev_close = closes[i - 1] if i > 0 else close_i
                orb_high = float(orb["high"])
                orb_low = float(orb["low"])
                if orb_end <= local < entry_end:
                    broke_down = low_i < (orb_low * (1.0 - orb_break_buffer))
                    close_down = close_i < (orb_low * (1.0 - orb_break_buffer))
                    prev_above = prev_close >= orb_low
                    if broke_down and close_down and prev_above:
                        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
                        if next_open > 0:
                            tf_idx = max(bar_to_tf_idx[i] - 1, 0) if _is_new_tf_bucket(bar_to_tf_idx, i) else max(bar_to_tf_idx[i] - 1, 0)
                            ema_now = tf_ema[tf_idx] if 0 <= tf_idx < len(tf_ema) else None
                            rel_volume = 1.0
                            tf_bar_idx = bar_to_tf_idx[i]
                            if 0 <= tf_bar_idx < len(tf_bars) and tf_vol_sma[tf_bar_idx] is not None and tf_vol_sma[tf_bar_idx] > 0:
                                rel_volume = tf_volumes[tf_bar_idx] / tf_vol_sma[tf_bar_idx]
                            score, quality = _orb_signal_score(
                                rel_volume=rel_volume,
                                breakdown_close=close_i,
                                orb_low=orb_low,
                                orb_high=orb_high,
                                ema_15m=float(ema_now) if ema_now is not None else None,
                                daily_slope_negative=daily_slope_negative,
                                minutes_since_open=_minutes_since_ny_open(ts),
                            )
                            short_meta_active = _recent_short_loss_count(closed_meta, short_meta_lookback) >= short_meta_loss_threshold
                            short_risk_scale = short_meta_risk_scale if short_meta_active else 1.0
                            full_trail_mode = score >= short_full_trail_score_threshold
                            entry_price = next_open * (1.0 - slippage_rate)
                            stop_ref = max(high_i, orb_high)
                            stop_loss = stop_ref * (1.0 + stop_buffer)
                            sl_distance = stop_loss - entry_price
                            if sl_distance > 0:
                                take_profit = None if full_trail_mode else entry_price - (sl_distance * orb_rr_multiple)
                                if take_profit is None or take_profit > 0:
                                    effective_risk_pct = short_risk_pct * portfolio_risk_scale * short_risk_scale
                                    risk_amount = capital * effective_risk_pct
                                    shares = risk_amount / sl_distance
                                    position_size = shares * entry_price
                                    max_notional = capital * max_position_pct
                                    sizing_tier = "standard"
                                    if position_size > max_notional and entry_price > 0:
                                        shares = max_notional / entry_price
                                        position_size = max_notional
                                        risk_amount = shares * sl_distance
                                        effective_risk_pct = risk_amount / capital if capital > 0 else 0.0
                                        sizing_tier = "short_capped"
                                    if shares > 0 and position_size > 0:
                                        entry_fee = position_size * commission_rate
                                        open_trade = _Trade(
                                            direction="short",
                                            leg="orb_short",
                                            entry_ts=timestamps[i + 1],
                                            entry_price=round(entry_price, 4),
                                            stop_loss=round(stop_loss, 4),
                                            take_profit=round(take_profit, 4) if take_profit is not None else None,
                                            risk_pct=round(effective_risk_pct, 6),
                                            position_size=round(position_size, 4),
                                            shares=round(shares, 6),
                                            entry_index=i + 1,
                                            capital_at_entry=capital,
                                            orb_high=round(orb_high, 4),
                                            orb_low=round(orb_low, 4),
                                            donchian_upper=None,
                                            donchian_lower=None,
                                            trend_ema=round(float(ema_now), 4) if ema_now is not None else None,
                                            active_stop_loss=round(stop_loss, 4),
                                            fees_paid=round(entry_fee, 4),
                                            exit_fill_policy=exit_fill_policy,
                                            entry_rel_volume=round(rel_volume, 3),
                                            volume_confirmed=rel_volume >= min_rel_volume,
                                            sizing_tier=sizing_tier,
                                            signal_quality=quality,
                                            stop_source="orb_high_wick",
                                            session_label="new_york",
                                            full_trail_mode=full_trail_mode,
                                            use_chandelier_exit=full_trail_mode,
                                            use_break_even_stop=use_break_even_stop,
                                            portfolio_gate_active=portfolio_gate_active,
                                            short_meta_gate_active=short_meta_active,
                                            signal_score=score,
                                            time_stop_minutes=short_time_stop_minutes,
                                        )
                                        day_leg_taken.add((session_date, "orb_short"))
        if open_trade is not None:
            continue

        # Turtle long leg: completed 15m Donchian breakout with trend and RVOL filters.
        if allow_longs and (session_date, "turtle_long") not in day_leg_taken and session_label in {"asia", "new_york"}:
            if _is_new_tf_bucket(bar_to_tf_idx, i):
                completed_tf_idx = bar_to_tf_idx[i] - 1
                if completed_tf_idx >= max(donchian_period, ema_period, rvol_period):
                    donchian_upper = tf_donchian_prev_high[completed_tf_idx]
                    donchian_lower = tf_donchian_prev_low[completed_tf_idx]
                    ema_now = tf_ema[completed_tf_idx]
                    atr_now = tf_atr[completed_tf_idx]
                    vol_sma_now = tf_vol_sma[completed_tf_idx]
                    rel_volume = (
                        tf_volumes[completed_tf_idx] / vol_sma_now
                        if vol_sma_now is not None and vol_sma_now > 0
                        else 1.0
                    )
                    close_tf = tf_closes[completed_tf_idx]
                    if (
                        donchian_upper is not None
                        and ema_now is not None
                        and atr_now is not None
                        and daily_slope_positive
                        and rel_volume >= min_rel_volume
                        and close_tf > donchian_upper
                        and close_tf > ema_now
                    ):
                        next_open = opens[i] if opens[i] > 0 else closes[i]
                        if next_open > 0:
                            entry_price = next_open * (1.0 + slippage_rate)
                            wick_stop = tf_lows[completed_tf_idx] * (1.0 - stop_buffer)
                            atr_stop = entry_price - (float(atr_now) * turtle_initial_stop_atr_mult)
                            stop_loss = min(wick_stop, atr_stop)
                            sl_distance = entry_price - stop_loss
                            if sl_distance > 0:
                                effective_risk_pct = long_risk_pct * portfolio_risk_scale
                                risk_amount = capital * effective_risk_pct
                                shares = risk_amount / sl_distance
                                position_size = shares * entry_price
                                max_notional = capital * max_position_pct
                                sizing_tier = "standard"
                                if position_size > max_notional and entry_price > 0:
                                    shares = max_notional / entry_price
                                    position_size = max_notional
                                    risk_amount = shares * sl_distance
                                    effective_risk_pct = risk_amount / capital if capital > 0 else 0.0
                                    sizing_tier = "long_capped"
                                if shares > 0 and position_size > 0:
                                    entry_fee = position_size * commission_rate
                                    open_trade = _Trade(
                                        direction="long",
                                        leg="turtle_long",
                                        entry_ts=timestamps[i],
                                        entry_price=round(entry_price, 4),
                                        stop_loss=round(stop_loss, 4),
                                        take_profit=None,
                                        risk_pct=round(effective_risk_pct, 6),
                                        position_size=round(position_size, 4),
                                        shares=round(shares, 6),
                                        entry_index=i,
                                        capital_at_entry=capital,
                                        orb_high=None,
                                        orb_low=None,
                                        donchian_upper=round(float(donchian_upper), 4),
                                        donchian_lower=round(float(donchian_lower), 4) if donchian_lower is not None else None,
                                        trend_ema=round(float(ema_now), 4),
                                        active_stop_loss=round(stop_loss, 4),
                                        fees_paid=round(entry_fee, 4),
                                        exit_fill_policy=exit_fill_policy,
                                        entry_rel_volume=round(rel_volume, 3),
                                        volume_confirmed=True,
                                        sizing_tier=sizing_tier,
                                        signal_quality="A" if rel_volume >= 1.5 else "B",
                                        stop_source="breakout_bar_or_atr",
                                        session_label=session_label,
                                        full_trail_mode=True,
                                        use_chandelier_exit=True,
                                        use_break_even_stop=use_break_even_stop,
                                        portfolio_gate_active=portfolio_gate_active,
                                        short_meta_gate_active=False,
                                        signal_score=3 + (1 if rel_volume >= 1.5 else 0),
                                        time_stop_minutes=0,
                                    )
                                    day_leg_taken.add((session_date, "turtle_long"))

    if open_trade is not None:
        _close_trade(open_trade, len(bars) - 1, closes[-1], "end_of_data")

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    orb_trades = [t for t in trades if t.leg == "orb_short"]
    turtle_trades = [t for t in trades if t.leg == "turtle_long"]
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
        "strategy_variant": "orb_turtle_hybrid",
        "strategy_name": "Opening Pressure + Turtle",
        "bias_model": "15m_donchian_plus_daily_ema",
        "entry_session": "asia_and_new_york",
        "orb_window_minutes": orb_window_minutes,
        "use_five_minute_or": use_five_minute_or,
        "donchian_period": donchian_period,
        "ema_period": ema_period,
        "daily_ema_period": daily_ema_period,
        "min_rel_volume": min_rel_volume,
        "short_risk_pct": short_risk_pct,
        "long_risk_pct": long_risk_pct,
        "short_time_stop_minutes": short_time_stop_minutes,
        "short_full_trail_score_threshold": short_full_trail_score_threshold,
        "use_chandelier_exit": True,
        "use_break_even_stop": use_break_even_stop,
        "break_even_trigger_r": break_even_trigger_r,
        "chandelier_period": chandelier_period,
        "chandelier_atr_period": chandelier_atr_period,
        "chandelier_atr_mult": chandelier_atr_mult,
        "portfolio_gate_lookback": portfolio_gate_lookback,
        "portfolio_gate_threshold_pct": portfolio_gate_threshold_pct,
        "portfolio_gate_risk_scale": portfolio_gate_risk_scale,
        "short_meta_lookback": short_meta_lookback,
        "short_meta_loss_threshold": short_meta_loss_threshold,
        "short_meta_risk_scale": short_meta_risk_scale,
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
        "orb_short_trades": len(orb_trades),
        "turtle_long_trades": len(turtle_trades),
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
                "strategy_leg": t.leg,
                "entry_session": t.session_label,
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
                "fractal_high": t.orb_high if t.orb_high is not None else t.donchian_upper,
                "fractal_low": t.orb_low if t.orb_low is not None else t.donchian_lower,
                "liquidity_level": t.orb_low if t.direction == "short" and t.orb_low is not None else t.donchian_upper,
                "target_liquidity": None,
                "target_liquidity_type": "",
                "hourly_support": t.donchian_lower,
                "hourly_resistance": t.trend_ema,
                "use_chandelier_exit": t.use_chandelier_exit,
                "use_break_even_stop": t.use_break_even_stop,
                "full_trail_mode": t.full_trail_mode,
                "portfolio_gate_active": t.portfolio_gate_active,
                "short_meta_gate_active": t.short_meta_gate_active,
                "signal_score": t.signal_score,
                "time_stop_minutes": t.time_stop_minutes,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
