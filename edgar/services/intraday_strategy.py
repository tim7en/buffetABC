"""Intraday strategy runner (15m default, 2-year lookback).

This service pulls intraday bars from Yahoo Finance in chunks (to handle
provider window limits) and runs the same price/volume + fractal logic used
in the daily strategy, with intraday-friendly defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from edgar.services.strategy import _atr, _sma, _stochastic_rsi, _williams_fractals

logger = logging.getLogger(__name__)


def _bars_per_day(interval: str) -> int:
    if interval == "1m":
        return 390
    if interval == "2m":
        return 195
    if interval == "5m":
        return 78
    if interval == "15m":
        return 26
    if interval == "30m":
        return 13
    if interval == "60m":
        return 7
    if interval == "90m":
        return 5
    return 26


def _chunk_days_for_interval(interval: str) -> int:
    # Yahoo intraday usually supports up to ~60 days per request for <=60m.
    if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}:
        return 58
    return 365


def _max_lookback_days_for_interval(interval: str) -> int | None:
    # yfinance/yahoo practical limits for intraday intervals.
    if interval == "1m":
        return 7
    if interval in {"2m", "5m", "15m", "30m"}:
        return 60
    if interval in {"60m", "90m"}:
        return 730
    return None


def _fetch_intraday_bars(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
) -> list[dict]:
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance and pandas are required for intraday backtest") from exc

    lookback_days = max(int(365.25 * lookback_years), 30)
    total_days = lookback_days + max(warmup_days, 30)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None:
        total_days = min(total_days, max_days)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=total_days)

    ticker_obj = yf.Ticker(ticker.upper())
    chunk_days = _chunk_days_for_interval(interval)
    cursor = start_dt
    frames = []

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            hist = ticker_obj.history(
                start=cursor,
                end=chunk_end + timedelta(minutes=1),
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            logger.warning(
                "intraday chunk failed for %s (%s to %s): %s",
                ticker,
                cursor.isoformat(),
                chunk_end.isoformat(),
                exc,
            )
            hist = None
        if hist is not None and not hist.empty:
            frames.append(hist)
        cursor = chunk_end

    if not frames:
        return []

    all_hist = pd.concat(frames).sort_index()
    all_hist = all_hist[~all_hist.index.duplicated(keep="last")]
    all_hist = all_hist.dropna(subset=["Close"])

    bars: list[dict] = []
    for idx, row in all_hist.iterrows():
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        bars.append(
            {
                "timestamp": ts,
                "open": float(row["Open"]) if row.get("Open") is not None else float(row["Close"]),
                "high": float(row["High"]) if row.get("High") is not None else float(row["Close"]),
                "low": float(row["Low"]) if row.get("Low") is not None else float(row["Close"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            }
        )

    return bars


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
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0
    entry_rel_volume: float = 0.0
    volume_confirmed: bool = False
    sizing_tier: str = ""
    signal_quality: str = ""
    hold_bars: int = 0
    stop_source: str = ""
    fractal_high: float | None = None
    fractal_low: float | None = None
    entry_index: int = 0


def run_intraday_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "15m",
    lookback_years: float = 2.0,
    sma_fast_period: int = 130,
    sma_slow_period: int = 520,
    stoch_rsi_period: int = 14,
    oversold: float = 20.0,
    overbought: float = 80.0,
    atr_period: int = 14,
    stop_atr_mult: float = 2.4,
    fractal_period: int = 2,
    require_fractal_confirmation: bool = True,
    require_fractal_breakout: bool = False,
    fractal_break_buffer_atr: float = 0.1,
    min_fractal_stop_atr: float = 0.8,
    max_fractal_stop_atr: float = 5.0,
    take_profit_rr: float = 2.0,
    trail_atr_mult: float = 2.2,
    volume_period: int = 40,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.008,
    max_risk_pct: float = 0.015,
    max_position_pct: float = 0.25,
    slippage_bps: float = 4.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
) -> dict:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if sma_fast_period >= sma_slow_period:
        raise ValueError("sma_fast_period must be smaller than sma_slow_period")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if fractal_period < 1:
        raise ValueError("fractal_period must be >= 1")
    lookback_days = max(int(365.25 * lookback_years), 1)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None and lookback_days > max_days:
        raise ValueError(
            f"Yahoo Finance limit for interval={interval} is about {max_days} days. "
            f"Requested {lookback_days} days (~{lookback_years}y). "
            "Use a shorter window, or use interval=60m for multi-year runs."
        )

    bars_per_day = max(_bars_per_day(interval), 1)
    warmup_days = max(int(sma_slow_period / bars_per_day) + 20, 60)
    bars = _fetch_intraday_bars(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
    )
    if len(bars) < (sma_slow_period + 200):
        raise ValueError(
            f"Insufficient intraday data for {ticker}: {len(bars)} bars (need {sma_slow_period + 200}+)"
        )

    timestamps = [b["timestamp"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    warmup = max(sma_slow_period + 2, stoch_rsi_period * 3, atr_period + 2, volume_period + 2)
    start_idx = max(first_period_idx, warmup)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough intraday bars after warmup for backtest window")

    sma_fast = _sma(closes, sma_fast_period)
    sma_slow = _sma(closes, sma_slow_period)
    stoch_k, stoch_d = _stochastic_rsi(
        closes,
        rsi_period=stoch_rsi_period,
        stoch_period=stoch_rsi_period,
    )
    atr_vals = _atr(highs, lows, closes, period=atr_period)
    vol_sma = _sma(volumes, volume_period)
    frac_hi, frac_lo = _williams_fractals(highs, lows, period=fractal_period)

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0

    capital = float(initial_capital)
    trades: list[_Trade] = []
    equity_curve: list[dict] = []
    open_trade: _Trade | None = None
    peak_equity = capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0

    def _close_trade(trade: _Trade, idx: int, raw_exit_price: float, reason: str) -> None:
        nonlocal capital, total_fees
        if trade.direction == "long":
            exit_price = raw_exit_price * (1 - slippage_rate)
            gross_pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            exit_price = raw_exit_price * (1 + slippage_rate)
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

    slope_lookback = 20
    for i in range(start_idx, len(bars)):
        ts = timestamps[i]
        close_i = closes[i]
        high_i = highs[i]
        low_i = lows[i]

        if ts >= period_start:
            bars_in_period += 1
        if open_trade is not None and ts >= period_start:
            bars_in_position += 1

        if open_trade is not None:
            hit_sl = False
            hit_tp = False
            raw_exit: float | None = None
            if open_trade.direction == "long":
                hit_sl = low_i <= open_trade.stop_loss
                hit_tp = high_i >= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif hit_tp:
                    raw_exit = open_trade.take_profit
            else:
                hit_sl = high_i >= open_trade.stop_loss
                hit_tp = low_i <= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif hit_tp:
                    raw_exit = open_trade.take_profit

            if raw_exit is not None:
                _close_trade(
                    trade=open_trade,
                    idx=i,
                    raw_exit_price=raw_exit,
                    reason="take_profit" if hit_tp and not hit_sl else "stop_loss",
                )
                open_trade = None
            else:
                atr_now = atr_vals[i]
                if atr_now is not None and atr_now > 0:
                    if open_trade.direction == "long":
                        tr_stop = close_i - (atr_now * trail_atr_mult)
                        if tr_stop > open_trade.stop_loss:
                            open_trade.stop_loss = round(min(tr_stop, open_trade.take_profit - 1e-6), 4)
                    else:
                        tr_stop = close_i + (atr_now * trail_atr_mult)
                        if tr_stop < open_trade.stop_loss:
                            open_trade.stop_loss = round(max(tr_stop, open_trade.take_profit + 1e-6), 4)

        unrealized = 0.0
        if open_trade is not None:
            if open_trade.direction == "long":
                marked = close_i * (1 - slippage_rate)
                unrealized = (marked - open_trade.entry_price) * open_trade.shares - open_trade.fees_paid
            else:
                marked = close_i * (1 + slippage_rate)
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
        if ts < period_start:
            continue
        if i >= len(bars) - 1:
            continue

        atr_now = atr_vals[i]
        sf = sma_fast[i]
        ss = sma_slow[i]
        vk = stoch_k[i]
        vd = stoch_d[i]
        pk = stoch_k[i - 1] if i - 1 >= 0 else None
        pd = stoch_d[i - 1] if i - 1 >= 0 else None
        sv = vol_sma[i]
        if None in (atr_now, sf, ss, vk, vd, pk, pd, sv):
            continue
        if atr_now <= 0 or sv <= 0:
            continue

        ss_prev = sma_slow[i - slope_lookback] if i - slope_lookback >= 0 else None
        if ss_prev is None:
            continue

        confirmed_idx = i - fractal_period
        if confirmed_idx < 0:
            continue
        last_frac_hi = None
        last_frac_lo = None
        for j in range(confirmed_idx, -1, -1):
            if last_frac_hi is None and frac_hi[j] is not None:
                last_frac_hi = frac_hi[j]
            if last_frac_lo is None and frac_lo[j] is not None:
                last_frac_lo = frac_lo[j]
            if last_frac_hi is not None and last_frac_lo is not None:
                break

        trend_long = close_i > ss and sf > ss and ss > ss_prev
        trend_short = close_i < ss and sf < ss and ss < ss_prev
        long_momo = pk <= oversold and vk > oversold and pk <= pd and vk > vd
        short_momo = pk >= overbought and vk < overbought and pk >= pd and vk < vd

        long_signal = allow_longs and trend_long and long_momo
        short_signal = allow_shorts and trend_short and short_momo

        if require_fractal_breakout:
            break_buffer = atr_now * max(fractal_break_buffer_atr, 0.0)
            if long_signal:
                long_signal = last_frac_hi is not None and close_i > (last_frac_hi + break_buffer)
            if short_signal:
                short_signal = last_frac_lo is not None and close_i < (last_frac_lo - break_buffer)
        if long_signal == short_signal:
            continue

        rel_volume = volumes[i] / sv if sv else 0.0
        if rel_volume < min_rel_volume:
            continue

        trend_distance = abs((close_i / ss) - 1.0)
        if trend_distance < 0.003:
            continue

        if rel_volume >= 1.8 and trend_distance >= 0.02:
            tier = "high_conviction"
            signal_quality = "A"
            risk_mult = 1.5
        elif rel_volume >= 1.25 and trend_distance >= 0.01:
            tier = "standard"
            signal_quality = "B"
            risk_mult = 1.0
        else:
            tier = "conservative"
            signal_quality = "C"
            risk_mult = 0.65

        direction = "long" if long_signal else "short"
        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        if next_open <= 0:
            continue
        entry_price = next_open * (1 + slippage_rate) if direction == "long" else next_open * (1 - slippage_rate)

        atr_fallback_stop = atr_now * stop_atr_mult
        if atr_fallback_stop <= 0:
            continue
        min_stop = atr_now * max(min_fractal_stop_atr, 0.0)
        max_stop = atr_now * max(max_fractal_stop_atr, min_fractal_stop_atr)
        stop_source = "atr"
        stop_distance = atr_fallback_stop

        if direction == "long":
            if last_frac_lo is not None and last_frac_lo < entry_price:
                fractal_distance = entry_price - last_frac_lo
                if fractal_distance > max_stop:
                    if require_fractal_confirmation:
                        continue
                elif fractal_distance < min_stop:
                    stop_distance = min_stop
                    stop_source = "fractal_floor"
                else:
                    stop_distance = fractal_distance
                    stop_source = "fractal"
            elif require_fractal_confirmation:
                continue
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * take_profit_rr)
            if stop_loss <= 0:
                continue
        else:
            if last_frac_hi is not None and last_frac_hi > entry_price:
                fractal_distance = last_frac_hi - entry_price
                if fractal_distance > max_stop:
                    if require_fractal_confirmation:
                        continue
                elif fractal_distance < min_stop:
                    stop_distance = min_stop
                    stop_source = "fractal_floor"
                else:
                    stop_distance = fractal_distance
                    stop_source = "fractal"
            elif require_fractal_confirmation:
                continue
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * take_profit_rr)
            if take_profit <= 0:
                continue
        if stop_distance > max_stop:
            continue

        risk_pct = min(max_risk_pct, max(base_risk_pct * 0.4, base_risk_pct * risk_mult))
        risk_amount = capital * risk_pct
        shares = risk_amount / stop_distance
        position_size = shares * entry_price
        max_notional = capital * max_position_pct
        if position_size > max_notional and entry_price > 0:
            shares = max_notional / entry_price
            position_size = max_notional
            risk_amount = shares * stop_distance
            risk_pct = risk_amount / capital if capital > 0 else 0.0
            tier = f"{tier}_capped"

        if shares <= 0 or position_size <= 0:
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
            fees_paid=round(entry_fee, 4),
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=True,
            sizing_tier=tier,
            signal_quality=signal_quality,
            stop_source=stop_source,
            fractal_high=round(last_frac_hi, 4) if last_frac_hi is not None else None,
            fractal_low=round(last_frac_lo, 4) if last_frac_lo is not None else None,
            entry_index=i + 1,
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
        "interval": interval,
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
        "trades": [
            {
                "direction": t.direction,
                "entry_date": t.entry_ts.isoformat(),
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "risk_pct": t.risk_pct,
                "position_size": t.position_size,
                "shares": t.shares,
                "exit_date": t.exit_ts.isoformat() if t.exit_ts else None,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "fees_paid": t.fees_paid,
                "entry_rel_volume": t.entry_rel_volume,
                "volume_confirmed": t.volume_confirmed,
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "fractal_high": t.fractal_high,
                "fractal_low": t.fractal_low,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
