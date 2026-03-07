"""Manipulation + inverse FVG strategy backtest service.

Transcript-aligned logic:
1) Identify obvious liquidity levels from confirmed swing pivots.
2) Wait for manipulation (liquidity sweep and rejection back).
3) Find inverse FVG confirmation near the manipulation.
4) Enter on confirmation close (filled at next bar open), SL beyond gap, TP at 2R.

Designed for intraday bars (e.g., 60m over ~2 years from Yahoo/yfinance limits).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from edgar.services.strategy import _sma

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
    return 7


def _chunk_days_for_interval(interval: str) -> int:
    if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}:
        return 58
    return 365


def _max_lookback_days_for_interval(interval: str) -> int | None:
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
        raise RuntimeError("yfinance and pandas are required for manipulation backtest") from exc

    lookback_days = max(int(365.25 * lookback_years), 30)
    total_days = lookback_days + max(warmup_days, 30)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None:
        total_days = min(total_days, max(max_days - 2, 1))

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=total_days)
    ticker_obj = yf.Ticker(ticker.upper())

    cursor = start_dt
    chunk_days = _chunk_days_for_interval(interval)
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
                "manipulation fetch chunk failed for %s (%s -> %s): %s",
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
class _FVG:
    kind: str  # "bullish" (gap up) | "bearish" (gap down)
    idx: int
    zone_low: float
    zone_high: float


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
    ifvg_low: float
    ifvg_high: float
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
    stop_source: str = "ifvg"


def _pivot_levels(
    highs: list[float],
    lows: list[float],
    window: int,
) -> tuple[list[float | None], list[float | None]]:
    n = len(highs)
    piv_hi: list[float | None] = [None] * n
    piv_lo: list[float | None] = [None] * n
    if window < 1 or n < (window * 2 + 1):
        return piv_hi, piv_lo

    for i in range(window, n - window):
        h = highs[i]
        l = lows[i]
        left_h = highs[i - window: i]
        right_h = highs[i + 1: i + window + 1]
        left_l = lows[i - window: i]
        right_l = lows[i + 1: i + window + 1]
        if all(h > x for x in left_h) and all(h > x for x in right_h):
            piv_hi[i] = h
        if all(l < x for x in left_l) and all(l < x for x in right_l):
            piv_lo[i] = l
    return piv_hi, piv_lo


def _collect_fvgs(highs: list[float], lows: list[float]) -> list[_FVG]:
    # Standard 3-candle FVG definitions.
    # Bullish FVG (gap up): low[i] > high[i-2]
    # Bearish FVG (gap down): high[i] < low[i-2]
    out: list[_FVG] = []
    for i in range(2, len(highs)):
        if lows[i] > highs[i - 2]:
            out.append(_FVG(kind="bullish", idx=i, zone_low=highs[i - 2], zone_high=lows[i]))
        if highs[i] < lows[i - 2]:
            out.append(_FVG(kind="bearish", idx=i, zone_low=highs[i], zone_high=lows[i - 2]))
    return out


def _latest_confirmed_level(
    levels: list[float | None],
    i: int,
    confirm_lag: int,
    search_window: int,
) -> float | None:
    end_idx = i - confirm_lag
    if end_idx < 0:
        return None
    start_idx = max(0, end_idx - search_window)
    for j in range(end_idx, start_idx - 1, -1):
        if levels[j] is not None:
            return levels[j]
    return None


def run_manipulation_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "60m",
    lookback_years: float = 2.0,
    pivot_window: int = 3,
    liquidity_search_window: int = 240,
    manipulation_max_age_bars: int = 14,
    ifvg_proximity_bars: int = 16,
    sweep_buffer_bps: float = 0.0,
    recovery_buffer_bps: float = 0.0,
    ifvg_break_buffer_bps: float = 0.0,
    stop_buffer_bps: float = 3.0,
    rr_multiple: float = 2.0,
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
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if pivot_window < 2:
        raise ValueError("pivot_window must be >= 2")

    lookback_days = max(int(365.25 * lookback_years), 1)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None and lookback_days > max_days:
        raise ValueError(
            f"Yahoo Finance limit for interval={interval} is about {max_days} days. "
            f"Requested {lookback_days} days (~{lookback_years}y). "
            "Use a shorter window, or use interval=60m for multi-year runs."
        )

    warmup_days = max(int((liquidity_search_window + 40) / max(_bars_per_day(interval), 1)), 45)
    bars = _fetch_intraday_bars(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
    )
    if len(bars) < 250:
        raise ValueError(f"Insufficient intraday data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    warmup = max(liquidity_search_window // 2, volume_period + 5, pivot_window * 4 + 10)
    start_idx = max(first_period_idx, warmup)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    piv_hi, piv_lo = _pivot_levels(highs, lows, window=pivot_window)
    fvgs = _collect_fvgs(highs, lows)
    fvg_by_idx: dict[int, list[_FVG]] = {}
    for f in fvgs:
        fvg_by_idx.setdefault(f.idx, []).append(f)

    vol_sma = _sma(volumes, volume_period)

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0
    sweep_buffer = max(0.0, sweep_buffer_bps) / 10_000.0
    recovery_buffer = max(0.0, recovery_buffer_bps) / 10_000.0
    break_buffer = max(0.0, ifvg_break_buffer_bps) / 10_000.0
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

    bull_event: dict | None = None
    bear_event: dict | None = None

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

    def _find_ifvg(event_idx: int, i: int, desired_kind: str) -> _FVG | None:
        start = max(0, event_idx - 2)
        best = None
        for j in range(i, start - 1, -1):
            for fvg in fvg_by_idx.get(j, []):
                if fvg.kind != desired_kind:
                    continue
                if abs(fvg.idx - event_idx) > ifvg_proximity_bars:
                    continue
                best = fvg
                break
            if best is not None:
                break
        return best

    for i in range(start_idx, len(bars)):
        ts = timestamps[i]
        close_i = closes[i]
        high_i = highs[i]
        low_i = lows[i]
        prev_close = closes[i - 1] if i > 0 else close_i

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
                _close_trade(open_trade, i, raw_exit, "stop_loss" if hit_sl else "take_profit")
                open_trade = None

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

        support = _latest_confirmed_level(
            levels=piv_lo,
            i=i,
            confirm_lag=pivot_window,
            search_window=liquidity_search_window,
        )
        resistance = _latest_confirmed_level(
            levels=piv_hi,
            i=i,
            confirm_lag=pivot_window,
            search_window=liquidity_search_window,
        )

        if allow_longs and support is not None:
            swept = low_i < (support * (1.0 - sweep_buffer))
            recovered = close_i >= (support * (1.0 + recovery_buffer))
            if swept and recovered:
                bull_event = {"idx": i, "level": support}
        if allow_shorts and resistance is not None:
            swept = high_i > (resistance * (1.0 + sweep_buffer))
            recovered = close_i <= (resistance * (1.0 - recovery_buffer))
            if swept and recovered:
                bear_event = {"idx": i, "level": resistance}

        if bull_event is not None and i - bull_event["idx"] > manipulation_max_age_bars:
            bull_event = None
        if bear_event is not None and i - bear_event["idx"] > manipulation_max_age_bars:
            bear_event = None

        long_signal = False
        short_signal = False
        chosen_ifvg: _FVG | None = None
        chosen_level = None

        if allow_longs and bull_event is not None:
            ifvg = _find_ifvg(bull_event["idx"], i, desired_kind="bearish")
            if ifvg is not None:
                broke = prev_close <= (ifvg.zone_high * (1.0 + break_buffer)) and close_i > (ifvg.zone_high * (1.0 + break_buffer))
                if broke:
                    long_signal = True
                    chosen_ifvg = ifvg
                    chosen_level = float(bull_event["level"])

        if allow_shorts and bear_event is not None:
            ifvg = _find_ifvg(bear_event["idx"], i, desired_kind="bullish")
            if ifvg is not None:
                broke = prev_close >= (ifvg.zone_low * (1.0 - break_buffer)) and close_i < (ifvg.zone_low * (1.0 - break_buffer))
                if broke:
                    short_signal = True
                    chosen_ifvg = ifvg
                    chosen_level = float(bear_event["level"])

        if long_signal and short_signal:
            continue
        if not long_signal and not short_signal:
            continue
        if chosen_ifvg is None or chosen_level is None:
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        direction = "long" if long_signal else "short"
        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        if next_open <= 0:
            continue
        entry_price = next_open * (1.0 + slippage_rate) if direction == "long" else next_open * (1.0 - slippage_rate)

        if direction == "long":
            stop_loss = chosen_ifvg.zone_low * (1.0 - stop_buffer)
            sl_distance = entry_price - stop_loss
            if sl_distance <= 0:
                continue
            take_profit = entry_price + (sl_distance * rr_multiple)
        else:
            stop_loss = chosen_ifvg.zone_high * (1.0 + stop_buffer)
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
            liquidity_level=round(chosen_level, 4),
            ifvg_low=round(chosen_ifvg.zone_low, 4),
            ifvg_high=round(chosen_ifvg.zone_high, 4),
            fees_paid=round(entry_fee, 4),
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality=signal_quality,
        )
        bull_event = None
        bear_event = None

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
        "strategy_variant": "manipulation_ifvg",
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
                "fractal_high": t.ifvg_high,
                "fractal_low": t.ifvg_low,
                "liquidity_level": t.liquidity_level,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
