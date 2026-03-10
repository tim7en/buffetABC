"""Opening-shock fade short strategy using only local real-market caches.

Design:
1) Read only the cached 5-minute local market files already stored on disk.
2) Anchor a session open at Tokyo 09:00 (00:00 UTC), Hong Kong 09:30 (01:30 UTC),
   or the U.S. cash open for Tiingo equities.
3) Measure the first opening range, then require a strong upside shock in the early session.
4) Short the failure when price loses the opening-range midpoint and the running session VWAP.
5) Stop above the opening spike high and target the nearest lower mean-reversion level
   among session open / session VWAP / prior close.

Notes:
- This is intentionally a short-only research model.
- It uses next-bar execution and never reads future bars for the trigger.
- It is tied to the existing local caches so the backtest can run without
  fresh network data.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from edgar.services.binance_data import resolve_binance_symbol
from edgar.services.local_tiingo_data import load_local_tiingo_klines
from edgar.services.session_open_utils import (
    SESSION_OPEN_UTC,
    bars_per_day_24x7,
    minutes_since_session_open as _shared_minutes_since_session_open,
    session_anchor_for_ts as _shared_session_anchor_for_ts,
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

_SESSION_OPEN_UTC: dict[str, tuple[int, int]] = dict(SESSION_OPEN_UTC)


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
    session_open_price: float
    prior_close: float
    opening_range_high: float
    opening_range_low: float
    opening_range_mid: float
    spike_high: float
    session_vwap_at_entry: float
    primary_target_label: str
    secondary_target: float | None
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
    stop_source: str = "opening_spike_high"
    strategy_leg: str = "opening_shock_short"


def _bars_per_day_for_source(interval: str, source_label: str) -> int:
    if (interval or "").strip().lower() != "5m":
        raise ValueError("opening_shock_fade uses only 5m local cache data")
    if source_label == "local_tiingo_cache":
        return 78
    return bars_per_day_24x7(interval)


def _project_root() -> Path:
    try:
        from django.conf import settings

        return Path(settings.BASE_DIR)
    except Exception:
        return Path(__file__).resolve().parents[2]


def _locate_cached_binance_file(symbol: str, interval: str) -> Path:
    if (interval or "").strip().lower() != "5m":
        raise ValueError("opening_shock_fade requires interval='5m'")

    root = _project_root()
    search_roots = [
        root / "cache" / "binance_asia_orb",
        root / "cache" / "cache" / "cache" / "binance_asia_orb",
    ]
    pattern = f"{symbol}_*_{interval}.csv.gz"
    for base in search_roots:
        matches = sorted(base.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_size)
    raise FileNotFoundError(
        f"No local cached Binance file found for {symbol} interval={interval}. "
        "Expected it under cache/binance_asia_orb/."
    )


def _load_cached_binance_bars(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_symbol: str | None = None,
) -> tuple[list[dict], str, str]:
    symbol = resolve_binance_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
    path = _locate_cached_binance_file(symbol=symbol, interval=interval)

    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                open_time = int(row["open_time"])
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).replace(tzinfo=None),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue

    if not rows:
        return [], symbol, str(path)

    lookback_days = max(int(365.25 * lookback_years), 30)
    total_days = lookback_days + max(int(warmup_days), 1)
    cutoff = rows[-1]["timestamp"] - timedelta(days=total_days)
    bars = [row for row in rows if row["timestamp"] >= cutoff]
    return bars, symbol, str(path)


def _session_anchor_for_ts(ts: datetime, session_open: str) -> datetime:
    return _shared_session_anchor_for_ts(ts, session_open)


def _minutes_since_session_open(ts: datetime, session_open: str) -> int:
    return _shared_minutes_since_session_open(ts, session_open)


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
        bars, symbol, path = _load_cached_binance_bars(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_binance_cache", path
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
        bars, symbol, path = _load_cached_binance_bars(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_binance_cache", path
    except Exception:
        bars, symbol, path = load_local_tiingo_klines(
            ticker=ticker,
            interval=interval,
            lookback_years=lookback_years,
            warmup_days=warmup_days,
            market_data_symbol=market_data_symbol,
        )
        return bars, symbol, "local_tiingo_cache", path


def _session_target_candidates(
    entry_price: float,
    session_open_price: float,
    session_vwap: float,
    prior_close: float,
) -> tuple[tuple[str, float] | None, float | None]:
    raw_candidates = [
        ("session_vwap", float(session_vwap)),
        ("session_open", float(session_open_price)),
        ("prior_close", float(prior_close)),
    ]
    candidates = [(name, level) for name, level in raw_candidates if level > 0 and level < entry_price]
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[1], reverse=True)
    primary = candidates[0]
    secondary = candidates[1][1] if len(candidates) > 1 else None
    return primary, secondary


def run_opening_shock_fade_backtest(
    ticker: str,
    initial_capital: float = 10_000.0,
    interval: str = "5m",
    lookback_years: float = 2.0,
    market_data_source: str = "auto",
    market_data_symbol: str | None = None,
    session_open: str = "tokyo_open",
    opening_range_minutes: int = 15,
    shock_window_minutes: int = 30,
    entry_window_minutes: int = 60,
    min_shock_bps: float = 35.0,
    min_shock_atr_mult: float = 0.8,
    reclaim_buffer_bps: float = 0.0,
    shock_atr_period: int = 20,
    volume_period: int = 40,
    use_volume_filter: bool = False,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    stop_buffer_bps: float = 5.0,
    max_hold_minutes: int = 180,
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
    source = (market_data_source or "auto").strip().lower()
    if source in {"yfinance", "yf"}:
        raise ValueError("opening_shock_fade uses only local cached market data")
    valid_sessions = sorted(list(_SESSION_OPEN_UTC) + ["new_york_equity_open"])
    if session_open not in _SESSION_OPEN_UTC and session_open != "new_york_equity_open":
        raise ValueError(f"session_open must be one of {valid_sessions}")
    if interval.strip().lower() != "5m":
        raise ValueError("opening_shock_fade requires interval='5m' because the local cache is 5-minute only")
    if not allow_shorts:
        raise ValueError("opening_shock_fade is a short-only strategy; enable allow_shorts")
    if opening_range_minutes < 5:
        raise ValueError("opening_range_minutes must be >= 5")
    if shock_window_minutes < opening_range_minutes:
        raise ValueError("shock_window_minutes must be >= opening_range_minutes")
    if entry_window_minutes < opening_range_minutes:
        raise ValueError("entry_window_minutes must be >= opening_range_minutes")
    if max_hold_minutes < 5:
        raise ValueError("max_hold_minutes must be >= 5")
    if shock_atr_period < 2 or volume_period < 2:
        raise ValueError("shock_atr_period and volume_period must be >= 2")
    if exit_fill_policy not in {"stop_first", "target_first"}:
        raise ValueError("exit_fill_policy must be one of ['stop_first', 'target_first']")
    if chandelier_period < 2 or chandelier_atr_period < 2:
        raise ValueError("chandelier periods must be >= 2")

    preload_bars_per_day = 78
    warmup_bars = max(
        shock_atr_period * 3,
        volume_period * 2,
        chandelier_period + 5,
        chandelier_atr_period + 5,
        int(entry_window_minutes / 5) * 4,
        144,
    )
    warmup_days = max(int(warmup_bars / preload_bars_per_day) + 5, 7)

    bars, resolved_symbol, resolved_source, cache_path = _load_local_bars(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
        market_data_source=market_data_source,
        market_data_symbol=market_data_symbol,
    )
    if resolved_source == "local_tiingo_cache" and session_open != "new_york_equity_open":
        raise ValueError("Local Tiingo equity cache only supports session_open='new_york_equity_open'")
    bars_per_day = _bars_per_day_for_source(interval=interval, source_label=resolved_source)
    warmup_bars = max(
        shock_atr_period * 3,
        volume_period * 2,
        chandelier_period + 5,
        chandelier_atr_period + 5,
        int(entry_window_minutes / 5) * 4,
        max(int(bars_per_day / 2), 72),
    )
    if len(bars) < warmup_bars + 30:
        raise ValueError(f"Insufficient cached market data for {ticker}: {len(bars)} bars")

    timestamps = [b["timestamp"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [float(b.get("volume", 0.0) or 0.0) for b in bars]

    lookback_days = max(int(365.25 * lookback_years), 30)
    period_start = max(timestamps[0], timestamps[-1] - timedelta(days=lookback_days))
    first_period_idx = next((i for i, ts in enumerate(timestamps) if ts >= period_start), len(timestamps) - 1)
    start_idx = max(first_period_idx, warmup_bars)
    if start_idx >= len(bars) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

    shock_atr_vals = _atr(highs, lows, closes, period=shock_atr_period)
    vol_sma = _sma(volumes, volume_period)
    chandelier_atr_vals = _atr(highs, lows, closes, period=chandelier_atr_period) if use_chandelier_exit else []
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
    reclaim_buffer = max(reclaim_buffer_bps, 0.0) / 10_000.0
    stop_buffer = max(stop_buffer_bps, 0.0) / 10_000.0

    current_anchor: datetime | None = None
    session_open_price: float | None = None
    prior_close: float | None = None
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    session_spike_high: float | None = None
    session_spike_idx: int | None = None
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

        session_anchor = _session_anchor_for_ts(ts, session_open)
        if session_anchor != current_anchor:
            current_anchor = session_anchor
            session_open_price = None
            prior_close = closes[i - 1] if i > 0 else None
            opening_range_high = None
            opening_range_low = None
            cumulative_pv = 0.0
            cumulative_volume = 0.0
            session_spike_high = None
            session_spike_idx = None
            session_traded = False

        if ts < current_anchor:
            continue

        typical_price = (high_i + low_i + close_i) / 3.0
        cumulative_pv += typical_price * volumes[i]
        cumulative_volume += max(volumes[i], 0.0)
        session_vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else close_i
        if session_open_price is None:
            session_open_price = open_i
        minutes_since_open = _minutes_since_session_open(ts, session_open)
        if 0 <= minutes_since_open < opening_range_minutes:
            opening_range_high = high_i if opening_range_high is None else max(opening_range_high, high_i)
            opening_range_low = low_i if opening_range_low is None else min(opening_range_low, low_i)
        if session_spike_high is None or high_i > session_spike_high:
            session_spike_high = high_i
            session_spike_idx = i

        if open_trade is not None:
            continue
        if ts < period_start or i >= len(bars) - 1 or session_traded:
            continue
        if (
            prior_close is None
            or session_open_price is None
            or opening_range_high is None
            or opening_range_low is None
            or session_spike_high is None
            or session_spike_idx is None
        ):
            continue
        if opening_range_high <= opening_range_low:
            continue
        if minutes_since_open < opening_range_minutes or minutes_since_open >= entry_window_minutes:
            continue

        atr_now = shock_atr_vals[i]
        if atr_now is None or atr_now <= 0:
            continue
        rel_volume = 1.0
        if vol_sma[i] is not None and vol_sma[i] > 0:
            rel_volume = volumes[i] / vol_sma[i]
        if use_volume_filter and rel_volume < min_rel_volume:
            continue

        opening_mid = (opening_range_high + opening_range_low) / 2.0
        prev_close = closes[i - 1]
        shock_distance = session_spike_high - session_open_price
        shock_threshold = max(session_open_price * (min_shock_bps / 10_000.0), atr_now * max(min_shock_atr_mult, 0.0))
        if shock_distance < shock_threshold:
            continue
        if (ts - timestamps[session_spike_idx]).total_seconds() / 60 > shock_window_minutes:
            continue

        fresh_mid_loss = prev_close >= (opening_mid * (1.0 - reclaim_buffer)) and close_i < (opening_mid * (1.0 - reclaim_buffer))
        vwap_lost = close_i < (session_vwap * (1.0 - reclaim_buffer))
        bearish_reversal_bar = close_i < open_i
        if not (fresh_mid_loss and vwap_lost and bearish_reversal_bar):
            continue

        next_open = opens[i + 1] if opens[i + 1] > 0 else closes[i + 1]
        entry_price = next_open * (1.0 - slippage_rate)
        primary_target, secondary_target = _session_target_candidates(
            entry_price=entry_price,
            session_open_price=session_open_price,
            session_vwap=session_vwap,
            prior_close=prior_close,
        )
        if primary_target is None:
            continue

        stop_loss = session_spike_high * (1.0 + stop_buffer)
        sl_distance = stop_loss - entry_price
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

        shock_multiple = shock_distance / shock_threshold if shock_threshold > 0 else 0.0
        signal_quality = "A" if shock_multiple >= 1.5 and rel_volume >= 1.0 else ("B" if shock_multiple >= 1.0 else "C")
        entry_fee = position_size * commission_rate
        open_trade = _Trade(
            direction="short",
            entry_ts=timestamps[i + 1],
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(primary_target[1], 4),
            risk_pct=round(risk_amount / capital if capital > 0 else 0.0, 6),
            position_size=round(position_size, 4),
            shares=round(shares, 6),
            entry_index=i + 1,
            session_label=session_open,
            time_stop_ts=timestamps[i + 1] + timedelta(minutes=max_hold_minutes),
            session_open_price=round(session_open_price, 4),
            prior_close=round(prior_close, 4),
            opening_range_high=round(opening_range_high, 4),
            opening_range_low=round(opening_range_low, 4),
            opening_range_mid=round(opening_mid, 4),
            spike_high=round(session_spike_high, 4),
            session_vwap_at_entry=round(session_vwap, 4),
            primary_target_label=primary_target[0],
            secondary_target=round(secondary_target, 4) if secondary_target is not None else None,
            active_stop_loss=round(stop_loss, 4),
            fees_paid=round(entry_fee, 4),
            exit_fill_policy=exit_fill_policy,
            entry_rel_volume=round(rel_volume, 3),
            volume_confirmed=(not use_volume_filter) or rel_volume >= min_rel_volume,
            sizing_tier=sizing_tier,
            signal_quality=signal_quality,
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
        "strategy_name": "Opening Shock Fade",
        "data_mode": "intraday",
        "interval": "5m",
        "requested_interval": interval,
        "effective_interval": "5m",
        "interval_adjustment": None,
        "market_data_source": resolved_source,
        "market_data_symbol": resolved_symbol,
        "market_data_path": cache_path,
        "strategy_variant": "opening_shock_fade",
        "bias_model": "opening_shock_failure_short",
        "entry_session": session_open,
        "session_open": session_open,
        "opening_range_minutes": opening_range_minutes,
        "shock_window_minutes": shock_window_minutes,
        "entry_window_minutes": entry_window_minutes,
        "min_shock_bps": min_shock_bps,
        "min_shock_atr_mult": min_shock_atr_mult,
        "reclaim_buffer_bps": reclaim_buffer_bps,
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
                "fractal_high": t.spike_high,
                "fractal_low": t.opening_range_low,
                "liquidity_level": t.opening_range_mid,
                "session_label": t.session_label,
                "session_open_price": t.session_open_price,
                "prior_close": t.prior_close,
                "opening_range_high": t.opening_range_high,
                "opening_range_low": t.opening_range_low,
                "opening_range_mid": t.opening_range_mid,
                "session_vwap": t.session_vwap_at_entry,
                "target_label": t.primary_target_label,
                "secondary_target": t.secondary_target,
            }
            for t in trades
        ],
        "equity_curve": equity_curve,
    }
