"""Multi-timeframe liquidity flow strategy backtest service.

Transcript-aligned process:
1) Higher timeframe narrative (4h proxy): identify swing structure bias.
2) Medium timeframe refinement (15m proxy): follow internal orderflow and POI.
3) Lower timeframe execution (5m proxy): aggressive liquidation entry or
   conservative market-shift entry.

Notes:
- Designed for intraday bars; uses interval-scaled windows to approximate 4h/15m/5m.
- No forward-looking bias: confirmed pivots only, entry on next bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from edgar.services.binance_data import fetch_binance_klines
from edgar.services.market_mechanics_strategy import (
    _bars_per_day,
    _fetch_intraday_bars,
    _in_zone,
    _interval_to_minutes,
    _max_lookback_days_for_interval,
    _pivot_levels,
    _recent_pivots,
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
    _sweep_exhaustion_confirmed,
    _tighten_stop,
)


def _bars_per_day_24x7(interval: str) -> int:
    minutes = max(_interval_to_minutes(interval), 1)
    return max(int((24 * 60) / minutes), 1)


def _resolve_market_data_source(market_data_source: str, ticker: str) -> str:
    source = (market_data_source or "auto").strip().lower()
    if source in {"yfinance", "yf"}:
        return "yfinance"
    if source in {"binance", "binance_spot"}:
        return "binance"
    if source == "auto":
        text = (ticker or "").strip().upper()
        if text.startswith("BTC") or text.startswith("ETH") or text.startswith("PAXG"):
            return "binance"
        return "yfinance"
    raise ValueError("market_data_source must be one of ['auto', 'yfinance', 'binance']")


def _entry_session_allowed(ts: datetime, entry_session: str) -> bool:
    session = (entry_session or "all").strip().lower()
    if session == "all":
        return True
    hour = ts.hour
    if session == "asia":
        return 0 <= hour < 8
    if session == "london":
        return 7 <= hour < 13
    if session in {"new_york", "ny"}:
        return 13 <= hour < 20
    if session in {"overlap", "london_ny_overlap"}:
        return 13 <= hour < 16
    raise ValueError("entry_session must be one of ['all', 'asia', 'london', 'new_york', 'overlap']")


def _internal_direction(
    piv_hi: list[float | None],
    piv_lo: list[float | None],
    end_idx: int,
    bos_buffer: float,
) -> tuple[str | None, list[tuple[int, float]], list[tuple[int, float]]]:
    highs_recent = _recent_pivots(piv_hi, end_idx, count=2)
    lows_recent = _recent_pivots(piv_lo, end_idx, count=2)
    if len(highs_recent) < 2 or len(lows_recent) < 2:
        return None, highs_recent, lows_recent

    old_hi = highs_recent[0][1]
    new_hi = highs_recent[1][1]
    old_lo = lows_recent[0][1]
    new_lo = lows_recent[1][1]
    bullish = new_hi > (old_hi * (1.0 + bos_buffer)) and new_lo > (old_lo * (1.0 + bos_buffer))
    bearish = new_hi < (old_hi * (1.0 - bos_buffer)) and new_lo < (old_lo * (1.0 - bos_buffer))
    if bullish:
        return "bullish", highs_recent, lows_recent
    if bearish:
        return "bearish", highs_recent, lows_recent
    return None, highs_recent, lows_recent


def _has_nearby_liquidity(
    direction: str,
    pivot_levels: list[float | None],
    ref_level: float,
    end_idx: int,
    lookback: int,
    tolerance: float,
) -> bool:
    start = max(0, end_idx - lookback)
    for i in range(start, end_idx):
        level = pivot_levels[i]
        if level is None:
            continue
        rel = abs(float(level) - ref_level) / max(ref_level, 1e-8)
        if rel <= tolerance:
            return True
    return False


def _compute_target(
    direction: str,
    entry_price: float,
    stop_loss: float,
    weak_level: float | None,
    rr_multiple: float,
) -> float | None:
    if direction == "short":
        sl_distance = stop_loss - entry_price
        if sl_distance <= 0:
            return None
        rr_target = entry_price - (sl_distance * rr_multiple)
        if weak_level is not None and weak_level < entry_price:
            # Use nearest weak low only if it gives at least ~1R.
            if (entry_price - weak_level) >= sl_distance:
                return weak_level
        return rr_target

    sl_distance = entry_price - stop_loss
    if sl_distance <= 0:
        return None
    rr_target = entry_price + (sl_distance * rr_multiple)
    if weak_level is not None and weak_level > entry_price:
        if (weak_level - entry_price) >= sl_distance:
            return weak_level
    return rr_target


def _resolve_effective_interval(
    requested_interval: str,
    lookback_years: float,
    auto_adjust_for_yf_limits: bool,
) -> tuple[str, str | None]:
    interval = (requested_interval or "60m").strip()
    lookback_days = max(int(365.25 * lookback_years), 1)
    max_days = _max_lookback_days_for_interval(interval)
    if max_days is None or lookback_days <= max_days:
        return interval, None

    if not auto_adjust_for_yf_limits:
        raise ValueError(
            f"Yahoo Finance limit for interval={interval} is about {max_days} days. "
            f"Requested {lookback_days} days (~{lookback_years}y). "
            "Enable auto_adjust_for_yf_limits or use a coarser interval."
        )

    fallback_order = ["60m", "90m"]
    for fb in fallback_order:
        fb_max = _max_lookback_days_for_interval(fb)
        if fb_max is None or lookback_days <= fb_max:
            note = (
                f"Adjusted interval from {interval} to {fb} due to Yahoo Finance "
                f"intraday history limits for lookback {lookback_days} days."
            )
            return fb, note

    raise ValueError(
        f"Requested lookback {lookback_days} days exceeds supported intraday history. "
        "Use <=730 days with 60m/90m, or reduce lookback."
    )


@dataclass
class _Setup:
    direction: str
    zone_low: float
    zone_high: float
    zone_idx: int
    weak_level: float | None
    activated_idx: int
    expires_idx: int
    sweep_idx: int | None = None
    sweep_reference_volume: float = 0.0
    sweep_reference_pressure: float = 0.0
    sweep_exhaustion_confirmed: bool = False


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
    signal_model: str
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
    stop_source: str = "liquidation_candle"
    volume_exhaustion_confirmed: bool = False


def run_mtf_liquidity_flow_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "60m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",  # auto | yfinance | binance
    market_data_symbol: str | None = None,  # e.g. BTCUSDT / PAXGUSDT
    auto_adjust_for_yf_limits: bool = True,
    trend_alignment_mode: str = "aligned",  # aligned | pullback
    entry_session: str = "all",  # UTC entry window
    entry_model: str = "hybrid",  # aggressive | conservative | hybrid
    rr_multiple: float = 2.0,
    htf_pivot_window: int = 3,
    internal_pivot_window: int = 2,
    structure_search_bars: int = 200,
    zone_expiry_bars: int = 12,
    liquidity_lookback_bars: int = 60,
    equal_level_tolerance_bps: float = 12.0,
    sweep_buffer_bps: float = 3.0,
    break_buffer_bps: float = 0.0,
    trigger_buffer_bps: float = 1.0,
    stop_buffer_bps: float = 5.0,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    use_volume_exhaustion_filter: bool = True,
    max_sweep_rel_volume: float = 3.5,
    min_reversal_pressure_ratio: float = 0.25,
    min_rejection_wick_ratio: float = 0.15,
    exhaustion_lookback_bars: int = 24,
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
    if rr_multiple <= 0:
        raise ValueError("rr_multiple must be positive")
    if entry_model not in {"aggressive", "conservative", "hybrid"}:
        raise ValueError("entry_model must be one of ['aggressive', 'conservative', 'hybrid']")
    if trend_alignment_mode not in {"aligned", "pullback"}:
        raise ValueError("trend_alignment_mode must be one of ['aligned', 'pullback']")
    if (entry_session or "all").strip().lower() not in {"all", "asia", "london", "new_york", "ny", "overlap", "london_ny_overlap"}:
        raise ValueError("entry_session must be one of ['all', 'asia', 'london', 'new_york', 'overlap']")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if exit_fill_policy not in {"stop_first", "target_first"}:
        raise ValueError("exit_fill_policy must be one of ['stop_first', 'target_first']")
    if chandelier_period < 2 or chandelier_atr_period < 2:
        raise ValueError("chandelier periods must be >= 2")
    if exhaustion_lookback_bars < 2:
        raise ValueError("exhaustion_lookback_bars must be >= 2")

    resolved_source = _resolve_market_data_source(market_data_source=market_data_source, ticker=ticker)

    requested_interval = interval
    interval_adjustment = None
    if resolved_source == "yfinance":
        effective_interval, interval_adjustment = _resolve_effective_interval(
            requested_interval=requested_interval,
            lookback_years=lookback_years,
            auto_adjust_for_yf_limits=auto_adjust_for_yf_limits,
        )

        lookback_days = max(int(365.25 * lookback_years), 1)
        max_days = _max_lookback_days_for_interval(effective_interval)
        if max_days is not None and lookback_days > max_days:
            raise ValueError(
                f"Yahoo Finance limit for interval={effective_interval} is about {max_days} days. "
                f"Requested {lookback_days} days (~{lookback_years}y). "
                "Use a shorter window, or allow auto-adjust with 60m/90m."
            )
    else:
        effective_interval = (requested_interval or "5m").strip().lower()
        if effective_interval not in {"1m", "3m", "5m", "15m", "30m", "60m"}:
            raise ValueError(
                "Binance source supports intervals ['1m','3m','5m','15m','30m','60m']"
            )

    base_minutes = max(_interval_to_minutes(effective_interval), 1)
    htf_factor = max(1, int(round(240 / base_minutes)))  # 4h proxy
    mtf_factor = max(1, int(round(15 / base_minutes)))   # 15m proxy
    exec_factor = max(1, int(round(5 / base_minutes)))   # 5m proxy

    htf_window = max(2, htf_pivot_window * htf_factor)
    internal_window = max(2, internal_pivot_window * mtf_factor)
    warmup_bars = max(htf_window * 8, internal_window * 8, volume_period + 20, 160)
    bars_per_day = (
        max(_bars_per_day(effective_interval), 1)
        if resolved_source == "yfinance"
        else _bars_per_day_24x7(effective_interval)
    )
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

    piv_hi_htf, piv_lo_htf = _pivot_levels(highs, lows, window=htf_window)
    piv_hi_int, piv_lo_int = _pivot_levels(highs, lows, window=internal_window)
    vol_sma = _sma(volumes, volume_period)
    chandelier_atr_vals = _atr(highs, lows, closes, period=chandelier_atr_period) if use_chandelier_exit else []
    chandelier_highs = _rolling_highest(highs, chandelier_period) if use_chandelier_exit else []
    chandelier_lows = _rolling_lowest(lows, chandelier_period) if use_chandelier_exit else []

    commission_rate = max(0.0, commission_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0
    sweep_buffer = max(0.0, sweep_buffer_bps) / 10_000.0
    break_buffer = max(0.0, break_buffer_bps) / 10_000.0
    trigger_buffer = max(0.0, trigger_buffer_bps) / 10_000.0
    stop_buffer = max(0.0, stop_buffer_bps) / 10_000.0
    eq_tolerance = max(0.0, equal_level_tolerance_bps) / 10_000.0

    capital = float(initial_capital)
    trades: list[_Trade] = []
    equity_curve: list[dict] = []
    open_trade: _Trade | None = None
    active_setup: _Setup | None = None
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
        high_i = highs[i]
        low_i = lows[i]
        close_i = closes[i]
        open_i = opens[i]

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

        # Expire old setup.
        if active_setup is not None and i > active_setup.expires_idx:
            active_setup = None

        # Step 1: higher timeframe narrative.
        htf_end = i - htf_window
        htf_dir, _, _ = _internal_direction(piv_hi_htf, piv_lo_htf, htf_end, bos_buffer=0.0)
        if htf_dir is None:
            continue

        # Step 2: medium timeframe internal orderflow and POI.
        int_end = i - internal_window
        int_dir, int_highs, int_lows = _internal_direction(piv_hi_int, piv_lo_int, int_end, bos_buffer=0.0)
        if int_dir is None or len(int_highs) < 2 or len(int_lows) < 2:
            continue

        # Default path: only trade 5m sweeps when 4h and 15m structure agree.
        if trend_alignment_mode == "aligned":
            if htf_dir != int_dir:
                if active_setup is not None:
                    active_setup = None
                continue
            preferred = "long" if htf_dir == "bullish" else "short"
        else:
            preferred = "short" if htf_dir == "bullish" else "long"
            if preferred == "short" and int_dir != "bearish":
                if active_setup is not None:
                    active_setup = None
                continue
            if preferred == "long" and int_dir != "bullish":
                if active_setup is not None:
                    active_setup = None
                continue
        if preferred == "short" and not allow_shorts:
            continue
        if preferred == "long" and not allow_longs:
            continue
        if active_setup is not None and active_setup.direction != preferred:
            active_setup = None

        if active_setup is None:
            if preferred == "short":
                zone_idx = int_highs[-1][0]
                if i - zone_idx > max(structure_search_bars, internal_window * 2):
                    continue
                zone_high = highs[zone_idx]
                zone_low = min(opens[zone_idx], closes[zone_idx])
                if zone_high <= zone_low:
                    zone_low = zone_high - max((highs[zone_idx] - lows[zone_idx]) * 0.6, 1e-4)
                has_liq = _has_nearby_liquidity(
                    direction="short",
                    pivot_levels=piv_hi_int,
                    ref_level=zone_high,
                    end_idx=zone_idx,
                    lookback=max(liquidity_lookback_bars, internal_window * 3),
                    tolerance=eq_tolerance,
                )
                if has_liq and _in_zone(low_i, high_i, zone_low, zone_high):
                    active_setup = _Setup(
                        direction="short",
                        zone_low=zone_low,
                        zone_high=zone_high,
                        zone_idx=zone_idx,
                        weak_level=int_lows[-1][1] if int_lows else None,
                        activated_idx=i,
                        expires_idx=i + max(zone_expiry_bars, exec_factor * 4),
                    )
            else:
                zone_idx = int_lows[-1][0]
                if i - zone_idx > max(structure_search_bars, internal_window * 2):
                    continue
                zone_low = lows[zone_idx]
                zone_high = max(opens[zone_idx], closes[zone_idx])
                if zone_high <= zone_low:
                    zone_high = zone_low + max((highs[zone_idx] - lows[zone_idx]) * 0.6, 1e-4)
                has_liq = _has_nearby_liquidity(
                    direction="long",
                    pivot_levels=piv_lo_int,
                    ref_level=zone_low,
                    end_idx=zone_idx,
                    lookback=max(liquidity_lookback_bars, internal_window * 3),
                    tolerance=eq_tolerance,
                )
                if has_liq and _in_zone(low_i, high_i, zone_low, zone_high):
                    active_setup = _Setup(
                        direction="long",
                        zone_low=zone_low,
                        zone_high=zone_high,
                        zone_idx=zone_idx,
                        weak_level=int_highs[-1][1] if int_highs else None,
                        activated_idx=i,
                        expires_idx=i + max(zone_expiry_bars, exec_factor * 4),
                    )

        if active_setup is None:
            continue
        if i < active_setup.activated_idx:
            continue

        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        next_high = highs[i + 1]
        next_low = lows[i + 1]
        if next_open <= 0:
            continue
        if not _entry_session_allowed(timestamps[i + 1], entry_session):
            continue

        entry_price = None
        stop_loss = None
        take_profit = None
        signal_model = None
        stop_source = None
        exhaustion_confirmed = active_setup.sweep_exhaustion_confirmed

        # Step 3A: aggressive liquidation entry.
        if entry_model in {"aggressive", "hybrid"} and _in_zone(low_i, high_i, active_setup.zone_low, active_setup.zone_high):
            if active_setup.direction == "short":
                recent_high = max(highs[max(0, i - max(liquidity_lookback_bars, 2)):i] or [high_i])
                strong_liquidation = high_i > (recent_high * (1.0 + sweep_buffer)) and close_i < open_i
                if strong_liquidation:
                    exhaustion_confirmed, reference_volume, reference_pressure = _sweep_exhaustion_confirmed(
                        direction="short",
                        volumes=volumes,
                        opens=opens,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        idx=i,
                        lookback=max(exhaustion_lookback_bars, exec_factor * 4),
                        max_sweep_rel_volume=max_sweep_rel_volume,
                        min_reversal_pressure_ratio=min_reversal_pressure_ratio,
                        min_rejection_wick_ratio=min_rejection_wick_ratio,
                        vol_sma=vol_sma,
                    )
                    active_setup.sweep_idx = i
                    active_setup.sweep_reference_volume = reference_volume
                    active_setup.sweep_reference_pressure = reference_pressure
                    active_setup.sweep_exhaustion_confirmed = exhaustion_confirmed
                    trigger = low_i * (1.0 - trigger_buffer)
                    fill = None
                    if next_open <= trigger:
                        fill = next_open
                    elif next_low <= trigger:
                        fill = trigger
                    if fill is not None and ((not use_volume_exhaustion_filter) or exhaustion_confirmed):
                        entry_price = fill * (1.0 - slippage_rate)
                        stop_loss = high_i * (1.0 + stop_buffer)
                        take_profit = _compute_target("short", entry_price, stop_loss, active_setup.weak_level, rr_multiple)
                        signal_model = "aggressive_liquidation"
                        stop_source = "liquidation_candle"
            else:
                recent_low = min(lows[max(0, i - max(liquidity_lookback_bars, 2)):i] or [low_i])
                strong_liquidation = low_i < (recent_low * (1.0 - sweep_buffer)) and close_i > open_i
                if strong_liquidation:
                    exhaustion_confirmed, reference_volume, reference_pressure = _sweep_exhaustion_confirmed(
                        direction="long",
                        volumes=volumes,
                        opens=opens,
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        idx=i,
                        lookback=max(exhaustion_lookback_bars, exec_factor * 4),
                        max_sweep_rel_volume=max_sweep_rel_volume,
                        min_reversal_pressure_ratio=min_reversal_pressure_ratio,
                        min_rejection_wick_ratio=min_rejection_wick_ratio,
                        vol_sma=vol_sma,
                    )
                    active_setup.sweep_idx = i
                    active_setup.sweep_reference_volume = reference_volume
                    active_setup.sweep_reference_pressure = reference_pressure
                    active_setup.sweep_exhaustion_confirmed = exhaustion_confirmed
                    trigger = high_i * (1.0 + trigger_buffer)
                    fill = None
                    if next_open >= trigger:
                        fill = next_open
                    elif next_high >= trigger:
                        fill = trigger
                    if fill is not None and ((not use_volume_exhaustion_filter) or exhaustion_confirmed):
                        entry_price = fill * (1.0 + slippage_rate)
                        stop_loss = low_i * (1.0 - stop_buffer)
                        take_profit = _compute_target("long", entry_price, stop_loss, active_setup.weak_level, rr_multiple)
                        signal_model = "aggressive_liquidation"
                        stop_source = "liquidation_candle"

        # Step 3B: conservative market-shift entry only after a prior sweep exists.
        if entry_price is None and entry_model in {"conservative", "hybrid"}:
            if active_setup.direction == "short":
                weak_low = active_setup.weak_level
                shifted = (
                    active_setup.sweep_idx is not None
                    and i > active_setup.sweep_idx
                    and weak_low is not None
                    and close_i < (weak_low * (1.0 - break_buffer))
                    and ((not use_volume_exhaustion_filter) or active_setup.sweep_exhaustion_confirmed)
                )
                if shifted:
                    entry_price = next_open * (1.0 - slippage_rate)
                    stop_loss = max(high_i, active_setup.zone_high) * (1.0 + stop_buffer)
                    take_profit = _compute_target("short", entry_price, stop_loss, weak_low, rr_multiple)
                    signal_model = "conservative_shift"
                    stop_source = "shift_candle"
            else:
                weak_high = active_setup.weak_level
                shifted = (
                    active_setup.sweep_idx is not None
                    and i > active_setup.sweep_idx
                    and weak_high is not None
                    and close_i > (weak_high * (1.0 + break_buffer))
                    and ((not use_volume_exhaustion_filter) or active_setup.sweep_exhaustion_confirmed)
                )
                if shifted:
                    entry_price = next_open * (1.0 + slippage_rate)
                    stop_loss = min(low_i, active_setup.zone_low) * (1.0 - stop_buffer)
                    take_profit = _compute_target("long", entry_price, stop_loss, weak_high, rr_multiple)
                    signal_model = "conservative_shift"
                    stop_source = "shift_candle"

        if entry_price is None or stop_loss is None or take_profit is None:
            continue
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        if active_setup.direction == "short":
            sl_distance = stop_loss - entry_price
            if sl_distance <= 0 or take_profit >= entry_price:
                continue
        else:
            sl_distance = entry_price - stop_loss
            if sl_distance <= 0 or take_profit <= entry_price:
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
            direction=active_setup.direction,
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_pct=round(risk_pct, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            zone_low=round(active_setup.zone_low, 4),
            zone_high=round(active_setup.zone_high, 4),
            signal_model=signal_model or "model",
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            exit_fill_policy=exit_fill_policy,
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality="A" if signal_model == "aggressive_liquidation" else "B",
            stop_source=stop_source or "zone",
            volume_exhaustion_confirmed=bool(exhaustion_confirmed),
        )
        active_setup = None

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
        "strategy_variant": "mtf_liquidity_flow",
        "trend_alignment_mode": trend_alignment_mode,
        "entry_session": "new_york" if entry_session == "ny" else ("overlap" if entry_session == "london_ny_overlap" else entry_session),
        "entry_model": entry_model,
        "use_chandelier_exit": use_chandelier_exit,
        "use_break_even_stop": use_break_even_stop,
        "break_even_trigger_r": break_even_trigger_r,
        "use_volume_exhaustion_filter": use_volume_exhaustion_filter,
        "max_sweep_rel_volume": max_sweep_rel_volume,
        "min_reversal_pressure_ratio": min_reversal_pressure_ratio,
        "min_rejection_wick_ratio": min_rejection_wick_ratio,
        "exhaustion_lookback_bars": exhaustion_lookback_bars,
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
                "date": timestamps[i].isoformat(),
                "open": round(opens[i], 4),
                "high": round(highs[i], 4),
                "low": round(lows[i], 4),
                "close": round(closes[i], 4),
            }
            for i in range(start_idx, len(bars))
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
                "volume_exhaustion_confirmed": t.volume_exhaustion_confirmed,
                "volume_divergence_confirmed": t.volume_exhaustion_confirmed,
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_bars,
                "stop_source": t.stop_source,
                "fractal_high": t.zone_high,
                "fractal_low": t.zone_low,
                "liquidity_level": round((t.zone_low + t.zone_high) / 2.0, 4),
                "signal_model": t.signal_model,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
