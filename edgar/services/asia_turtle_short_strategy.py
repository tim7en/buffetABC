"""Asia-session turtle short on 20/55-day breakdowns using the local Binance cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from edgar.services.binance_data import load_local_binance_klines
from edgar.services.session_open_utils import (
    SESSION_OPEN_UTC,
    aggregate_session_bars,
    bars_per_day_24x7,
    minutes_since_session_open,
)
from edgar.services.strategy import (
    _atr,
    _break_even_stop_candidate,
    _chandelier_stop_candidate,
    _rolling_highest,
    _rolling_lowest,
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
    exit_channel_high: float | None
    atr_at_entry: float
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
    strategy_leg: str = "asia_turtle_short"


def run_asia_turtle_short_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "15m",
    lookback_years: float = 2.0,
    market_data_source: str = "binance",
    market_data_symbol: str | None = None,
    session_open: str = "tokyo_open",
    channel_period: int = 20,
    exit_channel_period: int | None = None,
    atr_period: int = 20,
    atr_stop_mult: float = 2.0,
    entry_window_minutes: int = 480,
    entry_buffer_bps: float = 0.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    enable_pyramiding: bool = False,
    pyramid_add_atr: float = 0.5,
    max_units: int = 4,
    slippage_bps: float = 2.0,
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
    del allow_longs
    del exit_fill_policy
    source = (market_data_source or "binance").strip().lower()
    if source in {"yfinance", "yf"}:
        raise ValueError("asia_turtle_short uses only the local Binance cache")
    if session_open not in SESSION_OPEN_UTC:
        raise ValueError(f"session_open must be one of {sorted(SESSION_OPEN_UTC)}")
    if interval.strip().lower() not in {"5m", "15m"}:
        raise ValueError("asia_turtle_short requires interval='5m' or '15m'")
    if not allow_shorts:
        raise ValueError("asia_turtle_short is a short-only strategy; enable allow_shorts")
    if channel_period not in {20, 55}:
        raise ValueError("channel_period must be 20 or 55")
    if atr_period < 5:
        raise ValueError("atr_period must be >= 5")
    if atr_stop_mult <= 0:
        raise ValueError("atr_stop_mult must be positive")
    if max_units < 1:
        raise ValueError("max_units must be >= 1")

    exit_channel = exit_channel_period if exit_channel_period is not None else (10 if channel_period == 20 else 20)
    bars_per_day = bars_per_day_24x7(interval)
    warmup_bars = max((channel_period + exit_channel + atr_period) * bars_per_day, chandelier_period + 10, 1800)
    warmup_days = max(int(warmup_bars / bars_per_day) + 15, channel_period + exit_channel + 20)

    bars, resolved_symbol = load_local_binance_klines(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
        market_data_symbol=market_data_symbol,
    )
    if len(bars) < warmup_bars + 30:
        raise ValueError(f"Insufficient cached Binance data for {ticker}: {len(bars)} bars")

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
    channel_lows = _rolling_lowest(session_lows, channel_period)
    exit_highs = _rolling_highest(session_highs, exit_channel)
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
        exit_price = raw_exit_price * (1.0 + slippage_rate)
        exit_fee = abs(trade.shares * exit_price) * commission_rate
        gross_pnl = (trade.entry_price - exit_price) * trade.shares
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

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            current_exit_channel = (
                float(exit_highs[completed_session_idx])
                if completed_session_idx >= exit_channel and exit_highs[completed_session_idx] is not None
                else open_trade.exit_channel_high
            )
            protective_stop = current_stop
            stop_reason = "stop_loss"
            if current_exit_channel is not None and current_exit_channel < protective_stop:
                protective_stop = current_exit_channel
                stop_reason = "exit_channel"
            if high_i >= protective_stop:
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
                open_trade.exit_channel_high = current_exit_channel

                if (
                    enable_pyramiding
                    and open_trade.unit_count < max_units
                    and open_trade.next_add_price is not None
                    and close_i <= open_trade.next_add_price
                    and minutes_open < entry_window_minutes
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
        if ts < period_start or i >= len(bars) - 1 or minutes_open >= entry_window_minutes:
            continue
        if completed_session_idx < max(channel_period, atr_period, exit_channel):
            continue

        breakout_level = channel_lows[completed_session_idx]
        atr_now = daily_atr[completed_session_idx]
        exit_channel_high = exit_highs[completed_session_idx]
        if breakout_level is None or atr_now is None or atr_now <= 0:
            continue

        trigger_level = float(breakout_level) * (1.0 - entry_buffer)
        prev_close = closes[i - 1]
        broke_down = close_i < trigger_level and prev_close >= trigger_level
        if not broke_down:
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        entry_price = next_open * (1.0 - slippage_rate)
        sl_distance = float(atr_now) * atr_stop_mult
        if sl_distance <= 0:
            continue
        stop_loss = entry_price + sl_distance
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
        next_add_price = (
            round(entry_price - (float(atr_now) * pyramid_add_atr), 4)
            if enable_pyramiding and max_units > 1
            else None
        )
        open_trade = _Trade(
            direction="short",
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
            breakout_level=round(float(breakout_level), 4),
            exit_channel_high=round(float(exit_channel_high), 4) if exit_channel_high is not None else None,
            atr_at_entry=round(float(atr_now), 4),
            lowest_price_since_entry=round(min(low_i, entry_price), 4),
            unit_count=1,
            add_count=0,
            next_add_price=next_add_price,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            sizing_tier=sizing_tier,
            signal_quality="A" if channel_period == 55 else "B",
        )

    if open_trade is not None:
        _close_trade(open_trade, len(bars) - 1, closes[-1], "end_of_data")

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
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
        "strategy_name": "Asia Turtle Short",
        "data_mode": "intraday",
        "interval": interval,
        "requested_interval": interval,
        "effective_interval": interval,
        "interval_adjustment": None,
        "market_data_source": "local_binance_cache",
        "market_data_symbol": resolved_symbol,
        "strategy_variant": "asia_turtle_short",
        "bias_model": f"{channel_period}d_breakdown",
        "entry_session": session_open,
        "session_open": session_open,
        "channel_period": channel_period,
        "exit_channel_period": exit_channel,
        "atr_period": atr_period,
        "atr_stop_mult": atr_stop_mult,
        "entry_window_minutes": entry_window_minutes,
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
        "long_trades": 0,
        "short_trades": len(trades),
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
                "session_label": t.session_label,
                "breakout_level": t.breakout_level,
                "exit_channel_high": t.exit_channel_high,
                "atr_at_entry": t.atr_at_entry,
                "lowest_price_since_entry": t.lowest_price_since_entry,
                "unit_count": t.unit_count,
                "add_count": t.add_count,
                "next_add_price": t.next_add_price,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
