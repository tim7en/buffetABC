"""Three-step price action strategy (direction, location, execution).

Transcript-aligned framework:
1) Direction (higher-timeframe structure bias).
2) Location (point-of-interest in premium/discount context).
3) Execution (rejection + break/close + failure-to-continue confluences).

Design notes:
- Built for intraday bars while remaining bias-safe (no forward-looking access).
- Entries are placed on next bar open after all execution rules pass.
- Stop is anchored to the rejection candle; take-profit is fixed R-multiple.
- With 60m base bars, the 4h/1h/15m framework is approximated by scaled windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from edgar.services.strategy import _sma

logger = logging.getLogger(__name__)


def _interval_to_minutes(interval: str) -> int:
    text = (interval or "").strip().lower()
    if text.endswith("m"):
        try:
            return int(text[:-1])
        except ValueError:
            return 60
    if text.endswith("h"):
        try:
            return int(text[:-1]) * 60
        except ValueError:
            return 60
    return 60


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
        raise RuntimeError("yfinance and pandas are required for market mechanics backtest") from exc

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
                "market mechanics fetch chunk failed for %s (%s -> %s): %s",
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
        if all(h > x for x in highs[i - window:i]) and all(h > x for x in highs[i + 1:i + window + 1]):
            piv_hi[i] = h
        if all(l < x for x in lows[i - window:i]) and all(l < x for x in lows[i + 1:i + window + 1]):
            piv_lo[i] = l
    return piv_hi, piv_lo


def _recent_pivots(levels: list[float | None], end_idx: int, count: int = 2) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    if end_idx < 0:
        return out
    for i in range(end_idx, -1, -1):
        level = levels[i]
        if level is not None:
            out.append((i, float(level)))
            if len(out) >= count:
                break
    out.reverse()
    return out


def _avg_body(opens: list[float], closes: list[float], period: int = 20) -> list[float | None]:
    bodies = [abs(c - o) for o, c in zip(opens, closes)]
    return _sma(bodies, period)


def _bullish_engulfing(opens: list[float], closes: list[float], i: int) -> bool:
    if i < 1:
        return False
    return (
        closes[i] > opens[i]
        and closes[i - 1] < opens[i - 1]
        and closes[i] >= opens[i - 1]
        and opens[i] <= closes[i - 1]
    )


def _bearish_engulfing(opens: list[float], closes: list[float], i: int) -> bool:
    if i < 1:
        return False
    return (
        closes[i] < opens[i]
        and closes[i - 1] > opens[i - 1]
        and closes[i] <= opens[i - 1]
        and opens[i] >= closes[i - 1]
    )


def _bullish_pin_with_displacement(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    avg_body: list[float | None],
    i: int,
    wick_to_body: float = 1.5,
    displacement_mult: float = 1.2,
) -> bool:
    if i < 1:
        return False
    p = i - 1
    pin_body = max(abs(closes[p] - opens[p]), 1e-8)
    lower_wick = min(opens[p], closes[p]) - lows[p]
    upper_wick = highs[p] - max(opens[p], closes[p])
    pin_ok = lower_wick >= (pin_body * wick_to_body) and upper_wick <= (pin_body * 1.2)

    body_now = abs(closes[i] - opens[i])
    avg_now = avg_body[i] if avg_body[i] is not None else body_now
    displacement = (
        closes[i] > opens[i]
        and body_now >= max(avg_now * displacement_mult, pin_body * 1.1)
        and closes[i] > highs[p]
    )
    return pin_ok and displacement


def _bearish_pin_with_displacement(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    avg_body: list[float | None],
    i: int,
    wick_to_body: float = 1.5,
    displacement_mult: float = 1.2,
) -> bool:
    if i < 1:
        return False
    p = i - 1
    pin_body = max(abs(closes[p] - opens[p]), 1e-8)
    upper_wick = highs[p] - max(opens[p], closes[p])
    lower_wick = min(opens[p], closes[p]) - lows[p]
    pin_ok = upper_wick >= (pin_body * wick_to_body) and lower_wick <= (pin_body * 1.2)

    body_now = abs(closes[i] - opens[i])
    avg_now = avg_body[i] if avg_body[i] is not None else body_now
    displacement = (
        closes[i] < opens[i]
        and body_now >= max(avg_now * displacement_mult, pin_body * 1.1)
        and closes[i] < lows[p]
    )
    return pin_ok and displacement


def _rejection_signal(
    direction: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    avg_body: list[float | None],
    i: int,
) -> tuple[bool, int | None]:
    if direction == "long":
        if _bullish_engulfing(opens, closes, i):
            return True, i
        if _bullish_pin_with_displacement(opens, highs, lows, closes, avg_body, i):
            return True, i - 1
        return False, None

    if _bearish_engulfing(opens, closes, i):
        return True, i
    if _bearish_pin_with_displacement(opens, highs, lows, closes, avg_body, i):
        return True, i - 1
    return False, None


def _failure_to_continue(
    direction: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    i: int,
    zone_low: float,
    zone_high: float,
    activated_idx: int,
    lookback: int,
) -> bool:
    start = max(activated_idx, i - lookback)
    if direction == "long":
        for j in range(start, i):
            attempted_lower = closes[j] < opens[j] and lows[j] <= zone_high
            if attempted_lower and closes[i] > highs[j]:
                return True
        return False

    for j in range(start, i):
        attempted_higher = closes[j] > opens[j] and highs[j] >= zone_low
        if attempted_higher and closes[i] < lows[j]:
            return True
    return False


def _in_zone(low: float, high: float, zone_low: float, zone_high: float) -> bool:
    return low <= zone_high and high >= zone_low


def _find_latest_zone(
    direction: str,
    i: int,
    search_window: int,
    confirm_lag: int,
    piv_hi: list[float | None],
    piv_lo: list[float | None],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    equilibrium: float,
) -> tuple[float, float, int] | None:
    end_idx = i - confirm_lag
    if end_idx < 0:
        return None
    start_idx = max(0, end_idx - search_window)

    if direction == "long":
        for j in range(end_idx, start_idx - 1, -1):
            if piv_lo[j] is None:
                continue
            zone_low = lows[j]
            zone_high = max(opens[j], closes[j])
            if zone_high <= zone_low:
                zone_high = zone_low + max((highs[j] - lows[j]) * 0.5, 1e-4)
            zone_mid = (zone_low + zone_high) / 2.0
            if zone_mid >= equilibrium:
                continue
            # Unmitigated zone: no prior touch between creation and current bar.
            touched = any(_in_zone(lows[k], highs[k], zone_low, zone_high) for k in range(j + 1, i))
            if touched:
                continue
            return (zone_low, zone_high, j)
        return None

    for j in range(end_idx, start_idx - 1, -1):
        if piv_hi[j] is None:
            continue
        zone_high = highs[j]
        zone_low = min(opens[j], closes[j])
        if zone_high <= zone_low:
            zone_low = zone_high - max((highs[j] - lows[j]) * 0.5, 1e-4)
        zone_mid = (zone_low + zone_high) / 2.0
        if zone_mid <= equilibrium:
            continue
        touched = any(_in_zone(lows[k], highs[k], zone_low, zone_high) for k in range(j + 1, i))
        if touched:
            continue
        return (zone_low, zone_high, j)
    return None


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
    zone_low: float
    zone_high: float
    zone_created_idx: int
    rejection_idx: int
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0
    entry_rel_volume: float = 0.0
    volume_confirmed: bool = False
    sizing_tier: str = "standard"
    signal_quality: str = "A"
    hold_bars: int = 0
    stop_source: str = "rejection_candle"


def run_market_mechanics_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "60m",
    lookback_years: float = 2.0,
    rr_multiple: float = 3.0,
    direction_pivot_window: int = 3,
    location_pivot_window: int = 2,
    direction_search_window: int = 320,
    location_search_window: int = 220,
    zone_expiry_bars: int = 10,
    failure_lookback_bars: int = 6,
    bos_buffer_bps: float = 0.0,
    break_buffer_bps: float = 0.0,
    stop_buffer_bps: float = 5.0,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    slippage_bps: float = 4.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
) -> dict:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if rr_multiple <= 0:
        raise ValueError("rr_multiple must be positive")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")

    lookback_days = max(int(365.25 * lookback_years), 1)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is not None and lookback_days > max_days:
        raise ValueError(
            f"Yahoo Finance limit for interval={interval} is about {max_days} days. "
            f"Requested {lookback_days} days (~{lookback_years}y). "
            "Use a shorter window, or use interval=60m for multi-year runs."
        )

    base_minutes = max(_interval_to_minutes(interval), 1)
    htf_factor = max(1, int(round(240 / base_minutes)))  # 4h proxy
    mtf_factor = max(1, int(round(60 / base_minutes)))   # 1h proxy

    dir_window = max(2, direction_pivot_window * htf_factor)
    loc_window = max(2, location_pivot_window * mtf_factor)
    bars_per_day = max(_bars_per_day(interval), 1)
    warmup_bars = max(dir_window * 8, loc_window * 6, volume_period + 20, 140)
    warmup_days = max(int(warmup_bars / bars_per_day) + 20, 45)

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

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    piv_hi_dir, piv_lo_dir = _pivot_levels(highs, lows, window=dir_window)
    piv_hi_loc, piv_lo_loc = _pivot_levels(highs, lows, window=loc_window)
    vol_sma = _sma(volumes, volume_period)
    body_sma = _avg_body(opens, closes, period=20)

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0
    bos_buffer = max(0.0, bos_buffer_bps) / 10_000.0
    break_buffer = max(0.0, break_buffer_bps) / 10_000.0
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

    bias_direction: str | None = None
    swing_low = None
    swing_high = None
    pending_long: dict | None = None
    pending_short: dict | None = None

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

        # ---- Step 1: Direction (higher-timeframe structure proxy)
        piv_end = i - dir_window
        highs_recent = _recent_pivots(piv_hi_dir, piv_end, count=2)
        lows_recent = _recent_pivots(piv_lo_dir, piv_end, count=2)
        if len(highs_recent) == 2 and len(lows_recent) == 2:
            old_hi = highs_recent[0][1]
            new_hi = highs_recent[1][1]
            old_lo = lows_recent[0][1]
            new_lo = lows_recent[1][1]
            is_bull = new_hi > (old_hi * (1.0 + bos_buffer)) and new_lo > (old_lo * (1.0 + bos_buffer))
            is_bear = new_hi < (old_hi * (1.0 - bos_buffer)) and new_lo < (old_lo * (1.0 - bos_buffer))
            if is_bull:
                bias_direction = "long"
                swing_low = new_lo
                swing_high = new_hi
            elif is_bear:
                bias_direction = "short"
                swing_low = new_lo
                swing_high = new_hi

        if bias_direction is None or swing_low is None or swing_high is None:
            continue
        if swing_high <= swing_low:
            continue

        equilibrium = swing_low + ((swing_high - swing_low) * 0.5)

        # ---- Step 2: Location (POI in premium/discount)
        if allow_longs and bias_direction == "long":
            zone = _find_latest_zone(
                direction="long",
                i=i,
                search_window=location_search_window,
                confirm_lag=loc_window,
                piv_hi=piv_hi_loc,
                piv_lo=piv_lo_loc,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                equilibrium=equilibrium,
            )
            if zone is not None and _in_zone(low_i, high_i, zone[0], zone[1]):
                pending_long = {
                    "zone_low": zone[0],
                    "zone_high": zone[1],
                    "zone_idx": zone[2],
                    "activated_idx": i,
                    "expires_idx": i + max(zone_expiry_bars, 1),
                }

        if allow_shorts and bias_direction == "short":
            zone = _find_latest_zone(
                direction="short",
                i=i,
                search_window=location_search_window,
                confirm_lag=loc_window,
                piv_hi=piv_hi_loc,
                piv_lo=piv_lo_loc,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                equilibrium=equilibrium,
            )
            if zone is not None and _in_zone(low_i, high_i, zone[0], zone[1]):
                pending_short = {
                    "zone_low": zone[0],
                    "zone_high": zone[1],
                    "zone_idx": zone[2],
                    "activated_idx": i,
                    "expires_idx": i + max(zone_expiry_bars, 1),
                }

        if pending_long is not None and i > pending_long["expires_idx"]:
            pending_long = None
        if pending_short is not None and i > pending_short["expires_idx"]:
            pending_short = None

        # ---- Step 3: Execution (three confluences)
        signal_direction: str | None = None
        rejection_idx: int | None = None
        active_ctx: dict | None = None

        if (
            allow_longs
            and bias_direction == "long"
            and pending_long is not None
            and i >= pending_long["activated_idx"]
        ):
            strong_rej, rej_idx = _rejection_signal("long", opens, highs, lows, closes, body_sma, i)
            break_close = i >= 1 and close_i > (highs[i - 1] * (1.0 + break_buffer))
            failed_down = _failure_to_continue(
                direction="long",
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                i=i,
                zone_low=pending_long["zone_low"],
                zone_high=pending_long["zone_high"],
                activated_idx=pending_long["activated_idx"],
                lookback=max(failure_lookback_bars, 2),
            )
            if strong_rej and break_close and failed_down and rej_idx is not None:
                signal_direction = "long"
                rejection_idx = rej_idx
                active_ctx = pending_long

        if (
            signal_direction is None
            and allow_shorts
            and bias_direction == "short"
            and pending_short is not None
            and i >= pending_short["activated_idx"]
        ):
            strong_rej, rej_idx = _rejection_signal("short", opens, highs, lows, closes, body_sma, i)
            break_close = i >= 1 and close_i < (lows[i - 1] * (1.0 - break_buffer))
            failed_up = _failure_to_continue(
                direction="short",
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                i=i,
                zone_low=pending_short["zone_low"],
                zone_high=pending_short["zone_high"],
                activated_idx=pending_short["activated_idx"],
                lookback=max(failure_lookback_bars, 2),
            )
            if strong_rej and break_close and failed_up and rej_idx is not None:
                signal_direction = "short"
                rejection_idx = rej_idx
                active_ctx = pending_short

        if signal_direction is None or rejection_idx is None or active_ctx is None:
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        if next_open <= 0:
            continue
        entry_price = next_open * (1.0 + slippage_rate) if signal_direction == "long" else next_open * (1.0 - slippage_rate)

        if signal_direction == "long":
            stop_ref = lows[rejection_idx]
            stop_loss = stop_ref * (1.0 - stop_buffer)
            sl_distance = entry_price - stop_loss
            if sl_distance <= 0:
                continue
            take_profit = entry_price + (sl_distance * rr_multiple)
        else:
            stop_ref = highs[rejection_idx]
            stop_loss = stop_ref * (1.0 + stop_buffer)
            sl_distance = stop_loss - entry_price
            if sl_distance <= 0:
                continue
            take_profit = entry_price - (sl_distance * rr_multiple)
            if take_profit <= 0:
                continue

        risk_pct = max(base_risk_pct, 0.0001)
        risk_amount = capital * risk_pct
        shares = risk_amount / sl_distance
        position_size = shares * entry_price
        max_notional = capital * max_position_pct
        sizing_tier = "standard"
        if position_size > max_notional and entry_price > 0:
            shares = max_notional / entry_price
            position_size = max_notional
            risk_amount = shares * sl_distance
            risk_pct = risk_amount / capital if capital > 0 else 0.0
            sizing_tier = "capped"

        if shares <= 0 or position_size <= 0:
            continue

        entry_fee = position_size * commission_rate
        open_trade = _Trade(
            direction=signal_direction,
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_pct=round(risk_pct, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            zone_low=round(active_ctx["zone_low"], 4),
            zone_high=round(active_ctx["zone_high"], 4),
            zone_created_idx=int(active_ctx["zone_idx"]),
            rejection_idx=int(rejection_idx),
            fees_paid=round(entry_fee, 4),
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality="A",
        )
        pending_long = None
        pending_short = None

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
        "strategy_variant": "price_action_3step",
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
                # Reuse existing table columns for zone bounds.
                "fractal_high": t.zone_high,
                "fractal_low": t.zone_low,
                "liquidity_level": round((t.zone_low + t.zone_high) / 2.0, 4),
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
