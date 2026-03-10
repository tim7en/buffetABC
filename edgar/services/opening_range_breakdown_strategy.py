"""Asia-session opening-range breakdown short with a higher-timeframe bearish filter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from edgar.services.binance_data import load_local_binance_klines
from edgar.services.session_open_utils import (
    SESSION_OPEN_UTC,
    aggregate_session_bars,
    bars_per_day_24x7,
    minutes_since_session_open,
    session_anchor_for_ts,
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
from edgar.services.intraday_strategy import _ema


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
    session_label: str
    time_stop_ts: datetime
    range_high: float
    range_low: float
    breakdown_level: float
    session_vwap_at_entry: float
    stop_ref: float
    filter_label: str
    filter_snapshot: str
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
    signal_quality: str = "B"
    hold_bars: int = 0
    stop_source: str = "range_or_vwap"
    strategy_leg: str = "opening_range_breakdown_short"


_VALID_FILTERS = {
    "below_ema20",
    "below_20d_low",
    "lower_highs",
    "ema20_and_lower_highs",
    "below_20d_low_and_lower_highs",
}


def _filter_snapshot(
    session_highs: list[float],
    session_closes: list[float],
    ema_vals: list[float | None],
    prior_low_breaks: list[float | None],
    completed_session_idx: int,
) -> dict[str, bool]:
    prev_idx = completed_session_idx
    prev2_idx = completed_session_idx - 1
    below_ema20 = ema_vals[prev_idx] is not None and session_closes[prev_idx] < float(ema_vals[prev_idx])
    below_20d_low = (
        prev_idx >= 1
        and prior_low_breaks[prev_idx - 1] is not None
        and session_closes[prev_idx] < float(prior_low_breaks[prev_idx - 1])
    )
    lower_highs = prev2_idx >= 0 and session_highs[prev_idx] < session_highs[prev2_idx]
    return {
        "below_ema20": below_ema20,
        "below_20d_low": below_20d_low,
        "lower_highs": lower_highs,
    }


def _trend_filter_pass(snapshot: dict[str, bool], mode: str) -> bool:
    if mode == "below_ema20":
        return snapshot["below_ema20"]
    if mode == "below_20d_low":
        return snapshot["below_20d_low"]
    if mode == "lower_highs":
        return snapshot["lower_highs"]
    if mode == "below_20d_low_and_lower_highs":
        return snapshot["below_20d_low"] and snapshot["lower_highs"]
    return snapshot["below_ema20"] and snapshot["lower_highs"]


def _quality_from_snapshot(snapshot: dict[str, bool]) -> str:
    score = sum(1 for passed in snapshot.values() if passed)
    if score >= 3:
        return "A"
    if score == 2:
        return "B"
    return "C"


def run_opening_range_breakdown_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "binance",
    market_data_symbol: str | None = None,
    session_open: str = "tokyo_open",
    opening_range_minutes: int = 30,
    entry_window_minutes: int = 90,
    breakdown_buffer_bps: float = 2.0,
    breakdown_close_buffer_bps: float = 3.0,
    require_wick_retest: bool = True,
    retest_tolerance_bps: float = 5.0,
    trend_filter_mode: str = "ema20_and_lower_highs",
    daily_ema_period: int = 20,
    lookback_low_period: int = 20,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    stop_buffer_bps: float = 5.0,
    rr_multiple: float = 2.0,
    max_hold_minutes: int = 240,
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
    source = (market_data_source or "binance").strip().lower()
    if source in {"yfinance", "yf"}:
        raise ValueError("opening_range_breakdown_short uses only the local Binance cache")
    if session_open not in SESSION_OPEN_UTC:
        raise ValueError(f"session_open must be one of {sorted(SESSION_OPEN_UTC)}")
    if trend_filter_mode not in _VALID_FILTERS:
        raise ValueError(f"trend_filter_mode must be one of {sorted(_VALID_FILTERS)}")
    if interval.strip().lower() != "5m":
        raise ValueError("opening_range_breakdown_short requires interval='5m'")
    if not allow_shorts:
        raise ValueError("opening_range_breakdown_short is a short-only strategy; enable allow_shorts")
    if opening_range_minutes < 10:
        raise ValueError("opening_range_minutes must be >= 10")
    if entry_window_minutes <= opening_range_minutes:
        raise ValueError("entry_window_minutes must be greater than opening_range_minutes")
    if rr_multiple <= 0:
        raise ValueError("rr_multiple must be positive")

    bars_per_day = bars_per_day_24x7(interval)
    warmup_bars = max(
        (daily_ema_period + lookback_low_period) * bars_per_day,
        volume_period * 3,
        chandelier_period + 10,
        chandelier_atr_period + 10,
        1200,
    )
    warmup_days = max(int(warmup_bars / bars_per_day) + 10, 45)

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

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    session_bars, bar_to_session_idx = aggregate_session_bars(bars, session_open)
    session_highs = [float(bar.high) for bar in session_bars]
    session_lows = [float(bar.low) for bar in session_bars]
    session_closes = [float(bar.close) for bar in session_bars]
    daily_ema = _ema(session_closes, daily_ema_period)
    prior_low_breaks = _rolling_lowest(session_lows, lookback_low_period)
    vol_sma = _sma(volumes, volume_period)
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
    breakdown_buffer = max(breakdown_buffer_bps, 0.0) / 10_000.0
    close_buffer = max(breakdown_close_buffer_bps, 0.0) / 10_000.0
    stop_buffer = max(stop_buffer_bps, 0.0) / 10_000.0
    retest_tolerance = max(retest_tolerance_bps, 0.0) / 10_000.0

    current_anchor: datetime | None = None
    range_high: float | None = None
    range_low: float | None = None
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    session_traded = False

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

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            current_stop = open_trade.active_stop_loss if open_trade.active_stop_loss is not None else open_trade.stop_loss
            raw_exit, hit_sl, hit_tp, intrabar_conflict = _resolve_bar_bracket_exit(
                direction="short",
                bar_high=high_i,
                bar_low=low_i,
                stop_loss=current_stop,
                take_profit=open_trade.take_profit,
                fill_policy=exit_fill_policy,
            )
            if raw_exit is None and ts >= open_trade.time_stop_ts:
                raw_exit = close_i
                hit_sl = False
                hit_tp = False
                intrabar_conflict = False
                reason = "time_stop"
            elif raw_exit is not None:
                trailed_stop = _stop_is_trailed("short", open_trade.stop_loss, current_stop)
                break_even_stop = use_break_even_stop and trailed_stop and _stop_is_break_even_or_better(
                    direction="short",
                    entry_price=open_trade.entry_price,
                    active_stop=current_stop,
                )
                reason = (
                    "break_even_stop"
                    if hit_sl and break_even_stop
                    else (
                        "chandelier_stop"
                        if hit_sl and use_chandelier_exit and trailed_stop
                        else ("stop_loss" if hit_sl else "take_profit")
                    )
                )
            else:
                reason = ""
            if raw_exit is not None:
                open_trade.intrabar_conflict = intrabar_conflict
                _close_trade(open_trade, i, raw_exit, reason)
                open_trade = None
            else:
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
                        take_profit=open_trade.take_profit,
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
                        take_profit=open_trade.take_profit,
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

        session_anchor = session_anchor_for_ts(ts, session_open)
        if session_anchor != current_anchor:
            current_anchor = session_anchor
            range_high = None
            range_low = None
            cumulative_pv = 0.0
            cumulative_volume = 0.0
            session_traded = False

        typical_price = (high_i + low_i + close_i) / 3.0
        cumulative_pv += typical_price * volumes[i]
        cumulative_volume += max(volumes[i], 0.0)
        session_vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else close_i
        minutes_open = minutes_since_session_open(ts, session_open)

        if 0 <= minutes_open < opening_range_minutes:
            range_high = high_i if range_high is None else max(range_high, high_i)
            range_low = low_i if range_low is None else min(range_low, low_i)

        if open_trade is not None:
            continue
        if ts < period_start or i >= len(bars) - 1 or session_traded:
            continue
        if range_high is None or range_low is None or range_high <= range_low:
            continue
        if minutes_open < opening_range_minutes or minutes_open >= entry_window_minutes:
            continue

        current_session_idx = bar_to_session_idx[i]
        completed_session_idx = current_session_idx - 1
        if completed_session_idx < max(daily_ema_period, lookback_low_period, 2):
            continue

        snapshot = _filter_snapshot(
            session_highs=session_highs,
            session_closes=session_closes,
            ema_vals=daily_ema,
            prior_low_breaks=prior_low_breaks,
            completed_session_idx=completed_session_idx,
        )
        if not _trend_filter_pass(snapshot, trend_filter_mode):
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        prev_close = closes[i - 1]
        prev_inside = range_low <= prev_close <= range_high
        broke_down = low_i < (range_low * (1.0 - breakdown_buffer))
        close_down = close_i < (range_low * (1.0 - close_buffer))
        wick_retested = (not require_wick_retest) or high_i >= (range_low * (1.0 - retest_tolerance))
        bearish_body = close_i < open_i
        below_vwap = close_i < session_vwap
        if not (prev_inside and broke_down and close_down and wick_retested and bearish_body and below_vwap):
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        entry_price = next_open * (1.0 - slippage_rate)
        stop_ref = max(range_high, session_vwap)
        stop_loss = stop_ref * (1.0 + stop_buffer)
        sl_distance = stop_loss - entry_price
        if sl_distance <= 0:
            continue
        take_profit = entry_price - (sl_distance * rr_multiple)
        if take_profit <= 0:
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

        filter_label = ",".join(name for name, passed in snapshot.items() if passed) or "none"
        snapshot_text = "|".join(f"{name}:{int(passed)}" for name, passed in snapshot.items())
        entry_fee = position_size * commission_rate
        open_trade = _Trade(
            direction="short",
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_pct=round(risk_amount / capital if capital > 0 else 0.0, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            session_label=session_open,
            time_stop_ts=timestamps[i + 1] + timedelta(minutes=max_hold_minutes),
            range_high=round(range_high, 4),
            range_low=round(range_low, 4),
            breakdown_level=round(range_low, 4),
            session_vwap_at_entry=round(session_vwap, 4),
            stop_ref=round(stop_ref, 4),
            filter_label=filter_label,
            filter_snapshot=snapshot_text,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            exit_fill_policy=exit_fill_policy,
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality=_quality_from_snapshot(snapshot),
        )
        session_traded = True

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
        "strategy_name": "Opening-Range Breakdown Short",
        "data_mode": "intraday",
        "interval": "5m",
        "requested_interval": interval,
        "effective_interval": "5m",
        "interval_adjustment": None,
        "market_data_source": "local_binance_cache",
        "market_data_symbol": resolved_symbol,
        "strategy_variant": "opening_range_breakdown_short",
        "bias_model": trend_filter_mode,
        "entry_session": session_open,
        "session_open": session_open,
        "opening_range_minutes": opening_range_minutes,
        "entry_window_minutes": entry_window_minutes,
        "breakdown_buffer_bps": breakdown_buffer_bps,
        "breakdown_close_buffer_bps": breakdown_close_buffer_bps,
        "require_wick_retest": require_wick_retest,
        "retest_tolerance_bps": retest_tolerance_bps,
        "trend_filter_mode": trend_filter_mode,
        "daily_ema_period": daily_ema_period,
        "lookback_low_period": lookback_low_period,
        "max_hold_minutes": max_hold_minutes,
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
                "exit_fill_policy": t.exit_fill_policy,
                "entry_rel_volume": t.entry_rel_volume,
                "volume_confirmed": t.volume_confirmed,
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "fractal_high": t.range_high,
                "fractal_low": t.range_low,
                "liquidity_level": t.breakdown_level,
                "session_label": t.session_label,
                "session_vwap_at_entry": t.session_vwap_at_entry,
                "stop_ref": t.stop_ref,
                "filter_label": t.filter_label,
                "filter_snapshot": t.filter_snapshot,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
