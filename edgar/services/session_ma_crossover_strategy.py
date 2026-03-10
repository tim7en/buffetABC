"""Session-constrained 4h EMA crossover trend-following strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from edgar.services.intraday_strategy import _ema
from edgar.services.session_open_utils import SESSION_OPEN_UTC, minutes_since_session_open
from edgar.services.session_turtle_trend_strategy import (
    _aggregate_bars_with_index_mapping,
    _load_local_bars,
)
from edgar.services.strategy import (
    _atr,
    _break_even_stop_candidate,
    _chandelier_stop_candidate,
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
    trend_fast_value: float
    trend_slow_value: float
    crossover_ts: datetime
    crossover_index: int
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
    stop_source: str = "trend_atr_stop"
    strategy_leg: str = "session_ma_crossover"
    use_break_even_stop: bool = False
    use_chandelier_exit: bool = False


def run_session_ma_crossover_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    session_open: str = "tokyo_open",
    entry_window_minutes: int = 480,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    trend_interval_minutes: int = 240,
    atr_period: int = 20,
    atr_stop_mult: float = 2.0,
    fixed_stop_pct: float | None = None,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
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
) -> dict:
    source = (market_data_source or "auto").strip().lower()
    if source in {"yfinance", "yf"}:
        raise ValueError("session_ma_crossover uses only local cached market data")
    valid_sessions = sorted(list(SESSION_OPEN_UTC) + ["new_york_equity_open"])
    if session_open not in SESSION_OPEN_UTC and session_open != "new_york_equity_open":
        raise ValueError(f"session_open must be one of {valid_sessions}")
    if interval.strip().lower() not in {"5m", "15m"}:
        raise ValueError("session_ma_crossover requires interval='5m' or '15m'")
    if not allow_longs and not allow_shorts:
        raise ValueError("session_ma_crossover requires at least one of allow_longs or allow_shorts")
    if trend_fast_period < 2 or trend_slow_period < 2:
        raise ValueError("trend_fast_period and trend_slow_period must be >= 2")
    if trend_fast_period >= trend_slow_period:
        raise ValueError("trend_fast_period must be smaller than trend_slow_period")
    if trend_interval_minutes != 240:
        raise ValueError("session_ma_crossover currently supports trend_interval_minutes=240 only")
    if atr_period < 2:
        raise ValueError("atr_period must be >= 2")
    if atr_stop_mult <= 0:
        raise ValueError("atr_stop_mult must be positive")
    if fixed_stop_pct is not None and fixed_stop_pct <= 0:
        raise ValueError("fixed_stop_pct must be positive when provided")

    preload_warmup_days = max(trend_slow_period + atr_period + chandelier_atr_period + 40, 250)
    bars, resolved_symbol, resolved_source, market_data_path = _load_local_bars(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=preload_warmup_days,
        market_data_source=market_data_source,
        market_data_symbol=market_data_symbol,
    )
    if resolved_source == "local_tiingo_cache" and session_open != "new_york_equity_open":
        raise ValueError("Local Tiingo equity cache only supports session_open='new_york_equity_open'")
    if len(bars) < 1800:
        raise ValueError(f"Insufficient cached market data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [float(b["open"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    closes = [float(b["close"]) for b in bars]
    volumes = [float(b.get("volume", 0.0) or 0.0) for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 90)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))

    trend_bars, bar_to_trend_idx = _aggregate_bars_with_index_mapping(
        bars,
        interval_minutes=trend_interval_minutes,
    )
    trend_closes = [float(bar["close"]) for bar in trend_bars]
    trend_ema_fast = _ema(trend_closes, trend_fast_period)
    trend_ema_slow = _ema(trend_closes, trend_slow_period)
    trend_highs = [float(bar["high"]) for bar in trend_bars]
    trend_lows = [float(bar["low"]) for bar in trend_bars]
    trend_atr = _atr(trend_highs, trend_lows, trend_closes, atr_period)
    chandelier_atr_vals = _atr(highs, lows, closes, chandelier_atr_period) if use_chandelier_exit else []
    chandelier_highs = []
    chandelier_lows = []
    if use_chandelier_exit:
        from edgar.services.strategy import _rolling_highest, _rolling_lowest

        chandelier_highs = _rolling_highest(highs, chandelier_period)
        chandelier_lows = _rolling_lowest(lows, chandelier_period)

    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    warmup_bars = max(1800, (trend_slow_period + atr_period + 5) * max(trend_interval_minutes // 5, 1))
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    capital = float(initial_capital)
    peak_equity = capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0
    trades: list[_Trade] = []
    equity_curve: list[dict] = []
    open_trade: _Trade | None = None
    pending_direction: str | None = None
    pending_cross_idx: int | None = None
    pending_crossover_ts: datetime | None = None
    pending_fast_value: float | None = None
    pending_slow_value: float | None = None

    commission_rate = max(commission_bps, 0.0) / 10_000.0
    slippage_rate = max(slippage_bps, 0.0) / 10_000.0

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

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        completed_trend_idx = bar_to_trend_idx[i] - 1
        current_regime: str | None = None
        crossover_direction: str | None = None
        fast_now = None
        slow_now = None
        atr_now = None
        if completed_trend_idx >= trend_slow_period:
            fast_prev = trend_ema_fast[completed_trend_idx - 1]
            slow_prev = trend_ema_slow[completed_trend_idx - 1]
            fast_now = trend_ema_fast[completed_trend_idx]
            slow_now = trend_ema_slow[completed_trend_idx]
            atr_now = trend_atr[completed_trend_idx]
            if fast_now is not None and slow_now is not None:
                if float(fast_now) > float(slow_now):
                    current_regime = "long"
                elif float(fast_now) < float(slow_now):
                    current_regime = "short"
            if (
                fast_prev is not None
                and slow_prev is not None
                and fast_now is not None
                and slow_now is not None
            ):
                if float(fast_prev) <= float(slow_prev) and float(fast_now) > float(slow_now) and allow_longs:
                    crossover_direction = "long"
                elif float(fast_prev) >= float(slow_prev) and float(fast_now) < float(slow_now) and allow_shorts:
                    crossover_direction = "short"
        if crossover_direction is not None:
            pending_direction = crossover_direction
            pending_cross_idx = completed_trend_idx
            pending_crossover_ts = trend_bars[completed_trend_idx]["timestamp"]
            pending_fast_value = float(fast_now)
            pending_slow_value = float(slow_now)

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            hit_stop = low_i <= current_stop if open_trade.direction == "long" else high_i >= current_stop
            if hit_stop:
                trailed_stop = _stop_is_trailed(open_trade.direction, open_trade.stop_loss, current_stop)
                break_even_stop = open_trade.use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
                    direction=open_trade.direction,
                    entry_price=open_trade.entry_price,
                    active_stop=current_stop,
                )
                _close_trade(
                    open_trade,
                    i,
                    current_stop,
                    (
                        "break_even_stop"
                        if break_even_stop
                        else (
                            "chandelier_stop"
                            if open_trade.use_chandelier_exit and trailed_stop
                            else "stop_loss"
                        )
                    ),
                )
                open_trade = None
            elif crossover_direction is not None and crossover_direction != open_trade.direction:
                _close_trade(open_trade, i, close_i, "opposite_crossover")
                open_trade = None
            else:
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
                        take_profit=None,
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
        if ts < period_start or i >= len(bars) - 1 or minutes_open >= entry_window_minutes:
            continue
        if pending_direction is None or pending_cross_idx is None or pending_crossover_ts is None:
            continue
        if current_regime != pending_direction:
            if crossover_direction is None:
                pending_direction = None
                pending_cross_idx = None
                pending_crossover_ts = None
                pending_fast_value = None
                pending_slow_value = None
            continue
        if fixed_stop_pct is None and (atr_now is None or atr_now <= 0):
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        fixed_stop_rate = float(fixed_stop_pct) if fixed_stop_pct is not None else None
        if pending_direction == "long":
            entry_price = next_open * (1.0 + slippage_rate)
            stop_loss = (
                entry_price * (1.0 - fixed_stop_rate)
                if fixed_stop_rate is not None
                else entry_price - (float(atr_now) * atr_stop_mult)
            )
        else:
            entry_price = next_open * (1.0 - slippage_rate)
            stop_loss = (
                entry_price * (1.0 + fixed_stop_rate)
                if fixed_stop_rate is not None
                else entry_price + (float(atr_now) * atr_stop_mult)
            )

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
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
        stop_source = "fixed_pct_stop" if fixed_stop_rate is not None else "trend_atr_stop"
        open_trade = _Trade(
            direction=pending_direction,
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=None,
            risk_pct=round(risk_amount / capital if capital > 0 else 0.0, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            session_label=session_open,
            trend_fast_value=round(float(pending_fast_value), 4) if pending_fast_value is not None else 0.0,
            trend_slow_value=round(float(pending_slow_value), 4) if pending_slow_value is not None else 0.0,
            crossover_ts=pending_crossover_ts,
            crossover_index=pending_cross_idx,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            sizing_tier=sizing_tier,
            signal_quality="A" if pending_direction == "long" else "B",
            stop_source=stop_source,
            use_break_even_stop=use_break_even_stop,
            use_chandelier_exit=use_chandelier_exit,
        )
        pending_direction = None
        pending_cross_idx = None
        pending_crossover_ts = None
        pending_fast_value = None
        pending_slow_value = None

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
        "strategy_name": "Session MA Crossover",
        "data_mode": "intraday",
        "interval": interval,
        "requested_interval": interval,
        "effective_interval": interval,
        "interval_adjustment": None,
        "market_data_source": resolved_source,
        "market_data_symbol": resolved_symbol,
        "market_data_path": market_data_path,
        "strategy_variant": "session_ma_crossover",
        "bias_model": f"ema{trend_fast_period}_ema{trend_slow_period}_cross",
        "entry_session": session_open,
        "session_open": session_open,
        "entry_window_minutes": entry_window_minutes,
        "trend_filter_interval": "4h",
        "trend_interval_minutes": trend_interval_minutes,
        "trend_fast_period": trend_fast_period,
        "trend_slow_period": trend_slow_period,
        "atr_period": atr_period,
        "atr_stop_mult": atr_stop_mult,
        "fixed_stop_pct": fixed_stop_pct,
        "base_risk_pct": base_risk_pct,
        "max_position_pct": max_position_pct,
        "slippage_bps": slippage_bps,
        "commission_bps": commission_bps,
        "use_break_even_stop": use_break_even_stop,
        "break_even_trigger_r": break_even_trigger_r,
        "use_chandelier_exit": use_chandelier_exit,
        "chandelier_period": chandelier_period,
        "chandelier_atr_period": chandelier_atr_period,
        "chandelier_atr_mult": chandelier_atr_mult,
        "allow_longs": allow_longs,
        "allow_shorts": allow_shorts,
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
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "trend_fast_value": t.trend_fast_value,
                "trend_slow_value": t.trend_slow_value,
                "crossover_ts": t.crossover_ts.isoformat(),
                "crossover_index": t.crossover_index,
                "use_chandelier_exit": t.use_chandelier_exit,
                "use_break_even_stop": t.use_break_even_stop,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
