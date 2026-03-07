"""Price-action strategy with momentum, volatility and volume filters.

Strategy design (daily bars, no fundamental overlays):
- Regime filter: 50 SMA vs 200 SMA plus 200 SMA slope
- Trigger: Stochastic RSI cross from oversold/overbought bands
- Structure filter: Williams Fractal breakout confirmation (optional)
- Risk model: Fractal-anchored stop with ATR guardrails and optional ATR trailing stop
- Volume gate: relative volume (today / 20-day avg) confirms entries
- Position sizing: risk tier adjusted by trend strength + relative volume

Backtest defaults target the last 5 years of daily data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from edgar.models import EdgarCompany, StockPrice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average. Returns list aligned with input."""
    n = len(values)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period:
        return out
    rolling = sum(values[:period])
    out[period - 1] = rolling / period
    for i in range(period, n):
        rolling += values[i] - values[i - period]
        out[i] = rolling / period
    return out


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Wilder RSI."""
    n = len(closes)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period + 1:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + (avg_gain / avg_loss)))
    return out


def _stochastic_rsi(
    closes: list[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[list[float | None], list[float | None]]:
    """Stochastic RSI. Returns (%K, %D) aligned with input."""
    rsi_vals = _rsi(closes, period=rsi_period)
    n = len(closes)
    raw: list[float | None] = [None] * n

    for i in range(n):
        if rsi_vals[i] is None:
            continue
        ws = max(0, i - stoch_period + 1)
        window = [v for v in rsi_vals[ws: i + 1] if v is not None]
        if len(window) < stoch_period:
            continue
        lo = min(window)
        hi = max(window)
        raw[i] = 50.0 if hi == lo else ((rsi_vals[i] - lo) / (hi - lo) * 100.0)

    k: list[float | None] = [None] * n
    for i in range(n):
        if raw[i] is None:
            continue
        ws = max(0, i - k_smooth + 1)
        window = [v for v in raw[ws: i + 1] if v is not None]
        if len(window) >= k_smooth:
            k[i] = sum(window) / len(window)

    d: list[float | None] = [None] * n
    for i in range(n):
        if k[i] is None:
            continue
        ws = max(0, i - d_smooth + 1)
        window = [v for v in k[ws: i + 1] if v is not None]
        if len(window) >= d_smooth:
            d[i] = sum(window) / len(window)

    return k, d


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder ATR aligned with input."""
    n = len(closes)
    out: list[float | None] = [None] * n
    if period <= 0 or n < period + 1:
        return out

    tr: list[float] = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = sum(tr[1: period + 1]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = ((atr * (period - 1)) + tr[i]) / period
        out[i] = atr
    return out


def _williams_fractals(
    highs: list[float],
    lows: list[float],
    period: int = 2,
) -> tuple[list[float | None], list[float | None]]:
    """Williams Fractal highs/lows, aligned with input bars.

    Fractal at index i is confirmed only after `period` future bars exist.
    Callers should use it with lag (e.g. confirmed index = i - period).
    """
    n = len(highs)
    frac_hi: list[float | None] = [None] * n
    frac_lo: list[float | None] = [None] * n
    if period < 1 or n < (period * 2 + 1):
        return frac_hi, frac_lo

    for i in range(period, n - period):
        is_high = True
        is_low = True
        for j in range(1, period + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
        if is_high:
            frac_hi[i] = highs[i]
        if is_low:
            frac_lo[i] = lows[i]
    return frac_hi, frac_lo


# ---------------------------------------------------------------------------
# Backtest structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    direction: str
    entry_date: date
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    position_size: float
    shares: float
    exit_date: date | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0
    entry_rel_volume: float = 0.0
    volume_confirmed: bool = False
    sizing_tier: str = ""
    signal_quality: str = ""
    hold_days: int = 0
    stop_source: str = ""
    fractal_high: float | None = None
    fractal_low: float | None = None


@dataclass
class BacktestResult:
    ticker: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float
    profit_factor: float = 0.0
    cagr_pct: float = 0.0
    avg_trade_return_pct: float = 0.0
    exposure_pct: float = 0.0
    total_fees: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _fetch_price_data(company: EdgarCompany) -> list[dict]:
    prices = (
        StockPrice.objects.filter(company=company)
        .order_by("date")
        .values("date", "open", "high", "low", "close", "volume")
    )
    return [p for p in prices if p.get("close") is not None]


def _normalize_fetch_period(fetch_period: str, minimum_years: int) -> str:
    period = (fetch_period or "").strip().lower() or f"{minimum_years}y"
    if period.endswith("y") and period[:-1].isdigit():
        years = int(period[:-1])
        if years < minimum_years:
            return f"{minimum_years}y"
    return period


def run_backtest(
    ticker: str,
    initial_capital: float = 100.0,
    sma_fast_period: int = 50,
    sma_slow_period: int = 200,
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
    volume_period: int = 20,
    min_rel_volume: float = 1.0,
    base_risk_pct: float = 0.01,
    max_risk_pct: float = 0.02,
    max_position_pct: float = 0.30,
    slippage_bps: float = 3.0,
    commission_bps: float = 1.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
    lookback_years: int = 5,
    force_fetch: bool = False,
    fetch_period: str = "5y",
) -> BacktestResult:
    """Run a daily backtest using technical indicators and relative volume."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if sma_fast_period >= sma_slow_period:
        raise ValueError("sma_fast_period must be smaller than sma_slow_period")
    if not allow_longs and not allow_shorts:
        raise ValueError("At least one of allow_longs / allow_shorts must be true")
    if fractal_period < 1:
        raise ValueError("fractal_period must be >= 1")

    try:
        company = EdgarCompany.objects.get(ticker=ticker.upper())
    except EdgarCompany.DoesNotExist as exc:
        raise ValueError(f"Company {ticker} not found in database") from exc

    prices = _fetch_price_data(company)
    minimum_years = max(lookback_years + 2, 5)
    need_refresh = force_fetch or len(prices) < (sma_slow_period + 300)
    if prices:
        stale_days = (date.today() - prices[-1]["date"]).days
        if stale_days > 10:
            need_refresh = True
    if need_refresh:
        from edgar.services.stock_price import fetch_and_store_prices

        normalized_period = _normalize_fetch_period(fetch_period, minimum_years)
        saved = fetch_and_store_prices(company, period=normalized_period)
        logger.info(
            "price refresh for %s (%s): %s rows",
            company.ticker,
            normalized_period,
            saved,
        )
        prices = _fetch_price_data(company)

    if len(prices) < (sma_slow_period + 120):
        raise ValueError(
            f"Insufficient price data for {ticker}: {len(prices)} bars (need {sma_slow_period + 120}+)"
        )

    dates = [p["date"] for p in prices]
    opens = [float(p["open"] if p["open"] is not None else p["close"]) for p in prices]
    highs = [float(p["high"] if p["high"] is not None else p["close"]) for p in prices]
    lows = [float(p["low"] if p["low"] is not None else p["close"]) for p in prices]
    closes = [float(p["close"]) for p in prices]
    volumes = [float(p["volume"] or 0) for p in prices]

    last_date = dates[-1]
    period_start = max(dates[0], last_date - timedelta(days=int(365.25 * lookback_years)))
    first_period_idx = next((i for i, d in enumerate(dates) if d >= period_start), len(dates) - 1)
    warmup = max(
        sma_slow_period + 2,
        stoch_rsi_period * 3,
        atr_period + 2,
        volume_period + 2,
    )
    start_idx = max(first_period_idx, warmup)
    if start_idx >= len(prices) - 2:
        raise ValueError("Not enough bars after warmup for backtest window")

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
    trades: list[Trade] = []
    equity_curve: list[dict] = []
    open_trade: Trade | None = None
    peak_equity = capital
    max_drawdown = 0.0
    total_fees = 0.0
    bars_in_period = 0
    bars_in_position = 0

    def close_trade(trade: Trade, exit_day: date, raw_exit_price: float, reason: str) -> None:
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

        trade.exit_date = exit_day
        trade.exit_price = round(exit_price, 4)
        trade.pnl = round(net_pnl, 4)
        trade.exit_reason = reason
        trade.fees_paid = round(fee_total, 4)
        trade.hold_days = max((exit_day - trade.entry_date).days, 0)

        capital += net_pnl
        total_fees += fee_total
        trades.append(trade)

    slope_lookback = 10
    for i in range(start_idx, len(prices)):
        today = dates[i]
        today_close = closes[i]
        today_high = highs[i]
        today_low = lows[i]

        if today >= period_start:
            bars_in_period += 1

        if open_trade is not None and today >= period_start:
            bars_in_position += 1

        # 1) Manage open position exits with conservative intraday assumption:
        #    if both TP and SL are touched in one bar, SL is filled first.
        if open_trade is not None:
            hit_sl = False
            hit_tp = False
            raw_exit: float | None = None

            if open_trade.direction == "long":
                hit_sl = today_low <= open_trade.stop_loss
                hit_tp = today_high >= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif hit_tp:
                    raw_exit = open_trade.take_profit
            else:
                hit_sl = today_high >= open_trade.stop_loss
                hit_tp = today_low <= open_trade.take_profit
                if hit_sl:
                    raw_exit = open_trade.stop_loss
                elif hit_tp:
                    raw_exit = open_trade.take_profit

            if raw_exit is not None:
                close_trade(
                    trade=open_trade,
                    exit_day=today,
                    raw_exit_price=raw_exit,
                    reason="take_profit" if hit_tp and not hit_sl else "stop_loss",
                )
                open_trade = None
            else:
                atr_now = atr_vals[i]
                if atr_now is not None and atr_now > 0:
                    if open_trade.direction == "long":
                        tr_stop = today_close - (atr_now * trail_atr_mult)
                        if tr_stop > open_trade.stop_loss:
                            open_trade.stop_loss = round(
                                min(tr_stop, open_trade.take_profit - 1e-6),
                                4,
                            )
                    else:
                        tr_stop = today_close + (atr_now * trail_atr_mult)
                        if tr_stop < open_trade.stop_loss:
                            open_trade.stop_loss = round(
                                max(tr_stop, open_trade.take_profit + 1e-6),
                                4,
                            )

        unrealized = 0.0
        if open_trade is not None:
            if open_trade.direction == "long":
                marked = today_close * (1 - slippage_rate)
                unrealized = (marked - open_trade.entry_price) * open_trade.shares - open_trade.fees_paid
            else:
                marked = today_close * (1 + slippage_rate)
                unrealized = (open_trade.entry_price - marked) * open_trade.shares - open_trade.fees_paid

        if today >= period_start:
            current_equity = capital + unrealized
            peak_equity = max(peak_equity, current_equity)
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, dd)
            equity_curve.append(
                {
                    "date": today.isoformat(),
                    "equity": round(current_equity, 4),
                    "capital": round(capital, 4),
                }
            )

        if open_trade is not None:
            continue
        if today < period_start:
            continue
        if i >= len(prices) - 1:
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

        trend_long = today_close > ss and sf > ss and ss > ss_prev
        trend_short = today_close < ss and sf < ss and ss < ss_prev

        long_momo = pk <= oversold and vk > oversold and pk <= pd and vk > vd
        short_momo = pk >= overbought and vk < overbought and pk >= pd and vk < vd

        long_signal = allow_longs and trend_long and long_momo
        short_signal = allow_shorts and trend_short and short_momo

        if require_fractal_breakout:
            break_buffer = atr_now * max(fractal_break_buffer_atr, 0.0)
            if long_signal:
                long_signal = last_frac_hi is not None and today_close > (last_frac_hi + break_buffer)
            if short_signal:
                short_signal = last_frac_lo is not None and today_close < (last_frac_lo - break_buffer)
        if long_signal == short_signal:
            continue

        rel_volume = volumes[i] / sv if sv else 0.0
        if rel_volume < min_rel_volume:
            continue

        trend_distance = abs((today_close / ss) - 1.0)
        if trend_distance < 0.005:
            continue

        if rel_volume >= 1.8 and trend_distance >= 0.04:
            tier = "high_conviction"
            signal_quality = "A"
            risk_mult = 1.5
        elif rel_volume >= 1.3 and trend_distance >= 0.02:
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
        entry_price = (
            next_open * (1 + slippage_rate)
            if direction == "long"
            else next_open * (1 - slippage_rate)
        )

        atr_fallback_stop = atr_now * stop_atr_mult
        if atr_fallback_stop <= 0:
            continue

        min_stop = atr_now * max(min_fractal_stop_atr, 0.0)
        max_stop = atr_now * max(max_fractal_stop_atr, min_fractal_stop_atr)
        stop_source = "atr"
        stop_distance = atr_fallback_stop
        chosen_frac_hi = last_frac_hi
        chosen_frac_lo = last_frac_lo

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
        open_trade = Trade(
            direction=direction,
            entry_date=dates[i + 1],
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
            fractal_high=round(chosen_frac_hi, 4) if chosen_frac_hi is not None else None,
            fractal_low=round(chosen_frac_lo, 4) if chosen_frac_lo is not None else None,
        )

    if open_trade is not None:
        close_trade(
            trade=open_trade,
            exit_day=dates[-1],
            raw_exit_price=closes[-1],
            reason="end_of_data",
        )

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    gross_profit = sum(t.pnl for t in winning)
    gross_loss_abs = abs(sum(t.pnl for t in losing))

    total_return = ((capital - initial_capital) / initial_capital) * 100.0
    start_date = dates[start_idx]
    end_date = dates[-1]
    years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    if initial_capital > 0 and capital > 0:
        cagr = ((capital / initial_capital) ** (1 / years) - 1.0) * 100.0
    else:
        cagr = 0.0

    avg_trade_return = (
        sum((t.pnl / t.position_size) * 100.0 for t in trades if t.position_size > 0) / len(trades)
        if trades
        else 0.0
    )
    exposure = (bars_in_position / bars_in_period * 100.0) if bars_in_period > 0 else 0.0
    profit_factor = (
        (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    )

    return BacktestResult(
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=round(capital, 4),
        total_return_pct=round(total_return, 2),
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=round((len(winning) / len(trades)) * 100.0, 1) if trades else 0.0,
        max_drawdown_pct=round(max_drawdown * 100.0, 2),
        profit_factor=round(profit_factor, 2),
        cagr_pct=round(cagr, 2),
        avg_trade_return_pct=round(avg_trade_return, 2),
        exposure_pct=round(exposure, 2),
        total_fees=round(total_fees, 4),
        long_trades=len(long_trades),
        short_trades=len(short_trades),
        trades=trades,
        equity_curve=equity_curve,
    )


def backtest_to_dict(result: BacktestResult) -> dict:
    """Serialize BacktestResult for API responses."""
    return {
        "ticker": result.ticker,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate": result.win_rate,
        "max_drawdown_pct": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "cagr_pct": result.cagr_pct,
        "avg_trade_return_pct": result.avg_trade_return_pct,
        "exposure_pct": result.exposure_pct,
        "total_fees": result.total_fees,
        "long_trades": result.long_trades,
        "short_trades": result.short_trades,
        "trades": [
            {
                "direction": t.direction,
                "entry_date": t.entry_date.isoformat(),
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "risk_pct": t.risk_pct,
                "position_size": t.position_size,
                "shares": t.shares,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "fees_paid": t.fees_paid,
                "entry_rel_volume": t.entry_rel_volume,
                "volume_confirmed": t.volume_confirmed,
                "sizing_tier": t.sizing_tier,
                "signal_quality": t.signal_quality,
                "hold_days": t.hold_days,
                "stop_source": t.stop_source,
                "fractal_high": t.fractal_high,
                "fractal_low": t.fractal_low,
            }
            for t in result.trades
        ],
        "equity_curve": result.equity_curve,
    }
