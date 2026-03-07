"""Intraday Williams Fractal strategy variants (transcript-aligned).

Supported variants:
- fractal_breakout_ema200:
  Trend filter by EMA200, trade breakout of latest confirmed fractal.
  Long: close > EMA200 and close breaks above latest top fractal.
  Short: close < EMA200 and close breaks below latest bottom fractal.
  Stop: signal candle low/high. TP: 1.5R by default.

- alligator_stoch_fractal:
  Trend filter by alligator-style lines + Stoch RSI + newly confirmed fractal.
  Long: uptrend + new bottom fractal + Stoch RSI oversold.
  Short: downtrend + new top fractal + Stoch RSI overbought.
  Stop: fractal candle low/high. TP: 1.5R fallback, plus alligator mid-line exit.

Implementation is explicitly bias-safe:
- fractals are used only after confirmation lag (period bars)
- signals are generated on bar i, entries happen at next bar open (i+1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from edgar.services.strategy import _sma, _stochastic_rsi, _williams_fractals

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
    if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}:
        return 58
    return 365


def _max_lookback_days_for_interval(interval: str) -> int | None:
    # Practical Yahoo Finance limits for intraday bars.
    if interval == "1m":
        return 7
    if interval in {"2m", "5m", "15m", "30m"}:
        return 60
    if interval in {"60m", "90m"}:
        return 730
    return None


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


def _smma(values: list[float], period: int) -> list[float | None]:
    # Wilder-style smoothed moving average (used for Alligator-like lines).
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    smma = sum(values[:period]) / period
    out[period - 1] = smma
    for i in range(period, len(values)):
        smma = ((smma * (period - 1)) + values[i]) / period
        out[i] = smma
    return out


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
        # Keep a small safety margin to avoid boundary rejections.
        total_days = min(total_days, max(max_days - 2, 1))

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
                "intraday chunk failed for %s (%s -> %s): %s",
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
                "open": float(row.get("Open", row["Close"])),
                "high": float(row.get("High", row["Close"])),
                "low": float(row.get("Low", row["Close"])),
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
    entry_index: int
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


def run_intraday_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "60m",
    lookback_years: float = 2.0,
    strategy_variant: str = "fractal_breakout_ema200",
    ema_period: int = 200,
    fractal_window: int = 9,
    breakout_buffer_bps: float = 0.0,
    rr_multiple: float = 1.5,
    stoch_rsi_period: int = 14,
    oversold: float = 20.0,
    overbought: float = 80.0,
    alligator_jaw_period: int = 13,
    alligator_teeth_period: int = 8,
    alligator_lips_period: int = 5,
    alligator_slope_bars: int = 3,
    alligator_min_gap_pct: float = 0.001,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_risk_pct: float = 0.02,
    max_position_pct: float = 0.30,
    slippage_bps: float = 4.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
) -> dict:
    valid_variants = {"fractal_breakout_ema200", "alligator_stoch_fractal"}
    if strategy_variant not in valid_variants:
        raise ValueError(f"strategy_variant must be one of {sorted(valid_variants)}")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if fractal_window < 5 or fractal_window % 2 == 0:
        raise ValueError("fractal_window must be odd and >= 5 (e.g. 5 or 9)")

    lookback_days = max(int(365.25 * lookback_years), 1)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None and lookback_days > max_days:
        raise ValueError(
            f"Yahoo Finance limit for interval={interval} is about {max_days} days. "
            f"Requested {lookback_days} days (~{lookback_years}y). "
            "Use a shorter window, or use interval=60m for multi-year runs."
        )

    fractal_period = (fractal_window - 1) // 2
    bars_per_day = max(_bars_per_day(interval), 1)
    warmup_bars = max(
        ema_period + 5,
        stoch_rsi_period * 3 + 5,
        alligator_jaw_period + alligator_slope_bars + 5,
        volume_period + 5,
        fractal_period * 4 + 5,
    )
    warmup_days = max(int(warmup_bars / bars_per_day) + 15, 45)

    bars = _fetch_intraday_bars(
        ticker=ticker,
        interval=interval,
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
    volumes = [b["volume"] for b in bars]

    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=max(int(365.25 * lookback_years), 30)))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    ema200 = _ema(closes, ema_period)
    stoch_k, stoch_d = _stochastic_rsi(
        closes,
        rsi_period=stoch_rsi_period,
        stoch_period=stoch_rsi_period,
    )
    vol_sma = _sma(volumes, volume_period)
    frac_hi, frac_lo = _williams_fractals(highs, lows, period=fractal_period)

    hl2 = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    jaw = _smma(hl2, alligator_jaw_period)
    teeth = _smma(hl2, alligator_teeth_period)
    lips = _smma(hl2, alligator_lips_period)

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0
    breakout_buffer = max(0.0, breakout_buffer_bps) / 10_000.0

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
            line_exit = False
            raw_exit: float | None = None

            if open_trade.direction == "long":
                hit_sl = low_i <= open_trade.stop_loss
                hit_tp = high_i >= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif strategy_variant == "alligator_stoch_fractal" and teeth[i] is not None and close_i <= teeth[i]:
                    line_exit = True
                    raw_exit = close_i
                elif hit_tp:
                    raw_exit = open_trade.take_profit
            else:
                hit_sl = high_i >= open_trade.stop_loss
                hit_tp = low_i <= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif strategy_variant == "alligator_stoch_fractal" and teeth[i] is not None and close_i >= teeth[i]:
                    line_exit = True
                    raw_exit = close_i
                elif hit_tp:
                    raw_exit = open_trade.take_profit

            if raw_exit is not None:
                if hit_sl:
                    reason = "stop_loss"
                elif line_exit:
                    reason = "alligator_line_exit"
                else:
                    reason = "take_profit"
                _close_trade(open_trade, i, raw_exit, reason)
                open_trade = None

        unrealized = 0.0
        if open_trade is not None:
            if open_trade.direction == "long":
                mark = close_i * (1.0 - slippage_rate)
                unrealized = (mark - open_trade.entry_price) * open_trade.shares - open_trade.fees_paid
            else:
                mark = close_i * (1.0 + slippage_rate)
                unrealized = (open_trade.entry_price - mark) * open_trade.shares - open_trade.fees_paid

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

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        long_signal = False
        short_signal = False
        stop_source = ""
        stop_anchor_low = None
        stop_anchor_high = None

        if strategy_variant == "fractal_breakout_ema200":
            ema_i = ema200[i]
            if ema_i is None:
                continue
            if allow_longs and last_frac_hi is not None:
                long_signal = close_i > ema_i and close_i > (last_frac_hi * (1.0 + breakout_buffer))
            if allow_shorts and last_frac_lo is not None:
                short_signal = close_i < ema_i and close_i < (last_frac_lo * (1.0 - breakout_buffer))
            stop_source = "signal_candle"
            stop_anchor_low = low_i
            stop_anchor_high = high_i
        else:  # alligator_stoch_fractal
            if None in (jaw[i], teeth[i], lips[i], stoch_k[i]):
                continue
            j_prev = i - alligator_slope_bars
            if j_prev < 0 or None in (jaw[j_prev], teeth[j_prev], lips[j_prev]):
                continue

            gap_ok = min(abs(lips[i] - teeth[i]), abs(teeth[i] - jaw[i])) / max(close_i, 1e-9) >= alligator_min_gap_pct
            uptrend = (
                lips[i] > teeth[i] > jaw[i]
                and lips[i] > lips[j_prev]
                and teeth[i] > teeth[j_prev]
                and jaw[i] > jaw[j_prev]
                and gap_ok
            )
            downtrend = (
                lips[i] < teeth[i] < jaw[i]
                and lips[i] < lips[j_prev]
                and teeth[i] < teeth[j_prev]
                and jaw[i] < jaw[j_prev]
                and gap_ok
            )

            new_bottom_fractal = frac_lo[confirmed_idx] is not None
            new_top_fractal = frac_hi[confirmed_idx] is not None

            if allow_longs and uptrend and new_bottom_fractal and stoch_k[i] <= oversold:
                long_signal = True
            if allow_shorts and downtrend and new_top_fractal and stoch_k[i] >= overbought:
                short_signal = True

            stop_source = "fractal_candle"
            stop_anchor_low = lows[confirmed_idx]
            stop_anchor_high = highs[confirmed_idx]

        if long_signal == short_signal:
            continue

        direction = "long" if long_signal else "short"
        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        if next_open <= 0:
            continue
        entry_price = next_open * (1.0 + slippage_rate) if direction == "long" else next_open * (1.0 - slippage_rate)

        if direction == "long":
            if stop_anchor_low is None:
                continue
            stop_loss = float(stop_anchor_low)
            sl_distance = entry_price - stop_loss
            if sl_distance <= 0:
                continue
            take_profit = entry_price + (sl_distance * rr_multiple)
        else:
            if stop_anchor_high is None:
                continue
            stop_loss = float(stop_anchor_high)
            sl_distance = stop_loss - entry_price
            if sl_distance <= 0:
                continue
            take_profit = entry_price - (sl_distance * rr_multiple)
            if take_profit <= 0:
                continue

        if use_volume_filter:
            if rel_volume >= 1.8:
                risk_pct = min(max_risk_pct, base_risk_pct * 1.5)
                sizing_tier = "high_conviction"
                signal_quality = "A"
            elif rel_volume >= 1.2:
                risk_pct = min(max_risk_pct, base_risk_pct)
                sizing_tier = "standard"
                signal_quality = "B"
            else:
                risk_pct = max(base_risk_pct * 0.6, base_risk_pct * 0.4)
                sizing_tier = "conservative"
                signal_quality = "C"
        else:
            risk_pct = min(max_risk_pct, base_risk_pct)
            sizing_tier = "standard"
            signal_quality = "B"

        risk_amount = capital * risk_pct
        shares = risk_amount / sl_distance
        position_size = shares * entry_price
        max_notional = capital * max_position_pct
        if position_size > max_notional and entry_price > 0:
            shares = max_notional / entry_price
            position_size = max_notional
            risk_amount = shares * sl_distance
            risk_pct = risk_amount / capital if capital > 0 else 0.0
            sizing_tier = f"{sizing_tier}_capped"

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
            entry_index=i + 1,
            fees_paid=round(entry_fee, 4),
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality=signal_quality,
            stop_source=stop_source,
            fractal_high=round(last_frac_hi, 4) if last_frac_hi is not None else None,
            fractal_low=round(last_frac_lo, 4) if last_frac_lo is not None else None,
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
        "strategy_variant": strategy_variant,
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
