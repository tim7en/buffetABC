from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from edgar.services.binance_data import get_local_binance_time_bounds, load_local_binance_klines
from edgar.services.intraday_strategy import _ema
from edgar.services.local_tiingo_data import get_local_tiingo_time_bounds, load_local_tiingo_klines
from edgar.services.sentiment_data import get_score_for_date
from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest


CORE_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("BTC-USD", "binance", "hong_kong_open"),
    ("BTC-USD", "binance", "new_york_equity_open"),
    ("ETH-USD", "binance", "hong_kong_open"),
    ("ETH-USD", "binance", "new_york_equity_open"),
    ("SOL-USD", "binance", "hong_kong_open"),
    ("SOL-USD", "binance", "new_york_equity_open"),
    ("PAXG-USD", "binance", "hong_kong_open"),
    ("PAXG-USD", "binance", "new_york_equity_open"),
    ("AMZN", "tiingo", "new_york_equity_open"),
    ("COIN", "tiingo", "new_york_equity_open"),
    ("COPPER", "tiingo", "new_york_equity_open"),
    ("CRCL", "tiingo", "new_york_equity_open"),
    ("GLD", "tiingo", "new_york_equity_open"),
    ("HOOD", "tiingo", "new_york_equity_open"),
    ("INTC", "tiingo", "new_york_equity_open"),
    ("MSTR", "tiingo", "new_york_equity_open"),
    ("PLTR", "tiingo", "new_york_equity_open"),
    ("PPLT", "tiingo", "new_york_equity_open"),
    ("SLV", "tiingo", "new_york_equity_open"),
    ("TSLA", "tiingo", "new_york_equity_open"),
)

INDEX_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("QQQ", "tiingo", "new_york_equity_open"),
    ("SPY", "tiingo", "new_york_equity_open"),
)

EXPANDED_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    *CORE_SESSION_TURTLE_UNIVERSE,
    *INDEX_SESSION_TURTLE_UNIVERSE,
)

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
GOLD_TICKERS = {"PAXG-USD", "GLD"}
EQUITY_TICKERS = {"AMZN", "COIN", "CRCL", "HOOD", "INTC", "MSTR", "PLTR", "QQQ", "SPY", "TSLA"}
METAL_TICKERS = {"COPPER", "PPLT", "SLV"}


def _resolve_universe(basket: str) -> tuple[tuple[str, str, str], ...]:
    key = (basket or "expanded").strip().lower()
    if key == "core":
        return CORE_SESSION_TURTLE_UNIVERSE
    if key == "index":
        return INDEX_SESSION_TURTLE_UNIVERSE
    if key == "expanded":
        return EXPANDED_SESSION_TURTLE_UNIVERSE
    raise ValueError("basket must be one of {'core', 'index', 'expanded'}")


def _normalize_asof_date(investable_universe_asof: date | datetime | str | None) -> date | None:
    if investable_universe_asof is None:
        return None
    if isinstance(investable_universe_asof, datetime):
        return investable_universe_asof.date()
    if isinstance(investable_universe_asof, date):
        return investable_universe_asof
    text = str(investable_universe_asof).strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _market_data_start_timestamp(
    *,
    ticker: str,
    source: str,
) -> datetime | None:
    source_key = (source or "").strip().lower()
    if source_key == "tiingo":
        start_ts, _, _, _ = get_local_tiingo_time_bounds(ticker=ticker)
        return start_ts
    if source_key == "binance":
        start_ts, _, _, _ = get_local_binance_time_bounds(ticker=ticker)
        return start_ts
    return None


def _resolve_investable_universe(
    *,
    basket: str,
    investable_universe_asof: date | datetime | str | None = None,
) -> tuple[tuple[str, str, str], ...]:
    universe = _resolve_universe(basket)
    asof_date = _normalize_asof_date(investable_universe_asof)
    if asof_date is None:
        return universe

    filtered: list[tuple[str, str, str]] = []
    for ticker, source, session_open in universe:
        start_ts = _market_data_start_timestamp(ticker=ticker, source=source)
        if start_ts is not None and start_ts.date() <= asof_date:
            filtered.append((ticker, source, session_open))
    if not filtered:
        raise ValueError(f"No investable universe members remain for as-of date {asof_date.isoformat()}")
    return tuple(filtered)


@dataclass
class _OpenTrade:
    combo_idx: int
    trade_idx: int
    ticker: str
    source: str
    session_open: str
    direction: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    shares: float
    position_size: float
    pnl: float
    risk_model: str
    entry_rel_volume: float
    asset_bucket: str
    entry_exposure_mult: float
    entry_sentiment_score: float | None
    direct_sentiment_score: float | None
    direct_sentiment_size_mult: float
    intraday_vol_proxy_regime: str | None
    intraday_vol_proxy_mult: float
    intraday_vol_proxy_signal_age_min: float | None
    intraday_vol_proxy_close: float | None
    ext_hours_proxy_regime: str | None
    ext_hours_proxy_mult: float
    ext_hours_proxy_source: str | None
    ext_hours_proxy_score: float | None
    volatility_persistence_regime: str | None
    volatility_persistence_mult: float
    volatility_persistence_ratio: float | None
    volatility_persistence_vix_rel: float | None
    volatility_persistence_vixy_rel: float | None
    volatility_persistence_signal_age_min: float | None
    scale: float
    scaled_position_size: float
    scaled_shares: float
    scaled_pnl: float
    performance_risk_mult: float
    performance_score: float | None
    performance_rank_pct: float | None
    performance_peer_count: int
    technical_ema_regime: str | None
    technical_ema_mult: float
    technical_adx_value: float | None
    technical_adx_mult: float


def _asset_bucket(ticker: str) -> str:
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in GOLD_TICKERS:
        return "gold"
    if ticker in EQUITY_TICKERS:
        return "equity"
    if ticker in METAL_TICKERS:
        return "metals"
    return "other"


# ── per-asset technical overlay helpers ──────────────────────────────────────

def _adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """Average Directional Index (Wilder, 1978). Returns one value per bar."""
    n = len(closes)
    result: list[float | None] = [None] * n
    if n < period * 3 + 1:
        return result

    def _wilder(vals: list[float]) -> list[float | None]:
        s: list[float | None] = [None] * len(vals)
        if len(vals) < period:
            return s
        s[period - 1] = sum(vals[:period])
        for i in range(period, len(vals)):
            prev = s[i - 1]
            if prev is not None:
                s[i] = prev - prev / period + vals[i]
        return s

    tr_v, pdm_v, mdm_v = [], [], []
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr_v.append(max(h - l, abs(h - pc), abs(l - pc)))
        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdm_v.append(up   if up > down and up > 0   else 0.0)
        mdm_v.append(down if down > up and down > 0 else 0.0)

    str_s  = _wilder(tr_v)
    spdm_s = _wilder(pdm_v)
    smdm_s = _wilder(mdm_v)

    dx_v: list[float | None] = []
    for i in range(len(tr_v)):
        if str_s[i] is None or str_s[i] < 1e-9:  # type: ignore[operator]
            dx_v.append(None)
            continue
        pdi = 100.0 * (spdm_s[i] or 0.0) / str_s[i]  # type: ignore[operator]
        mdi = 100.0 * (smdm_s[i] or 0.0) / str_s[i]  # type: ignore[operator]
        denom = pdi + mdi
        dx_v.append(100.0 * abs(pdi - mdi) / denom if denom > 1e-9 else 0.0)

    valid_dx = [v for v in dx_v if v is not None]
    adx_raw  = _wilder(valid_dx)
    vi = 0
    for i, dx in enumerate(dx_v):
        if dx is not None:
            result[i + 1] = adx_raw[vi]
            vi += 1
    return result


def build_per_asset_technical_state(
    universe: list[tuple[str, str, str]],
    lookback_years: float = 5.0,
    warmup_days: int = 300,
    ema_period: int = 200,
    adx_period: int = 14,
) -> dict:
    """Pre-compute per-asset daily EMA and 4H ADX for the universe.

    Returns:
        {
          "daily_ema": {ticker: {date_str: {"close": float, "ema": float|None}}},
          "4h_adx":    {ticker: [(bar_ts_isoformat, adx_value), ...]},
          "ema_period": int,
          "adx_period": int,
        }
    """
    seen: set[tuple[str, str]] = set()
    daily_ema_out: dict[str, dict[str, dict]] = {}
    adx_4h_out:   dict[str, list[tuple[str, float]]] = {}

    for ticker, source, _session in universe:
        key = (ticker, source)
        if key in seen:
            continue
        seen.add(key)

        try:
            if source == "binance":
                bars, _ = load_local_binance_klines(
                    ticker=ticker, interval="15m",
                    lookback_years=lookback_years, warmup_days=warmup_days,
                )
            else:
                bars, _, _ = load_local_tiingo_klines(
                    ticker=ticker, interval="5m",
                    lookback_years=lookback_years, warmup_days=warmup_days,
                )
        except Exception:
            continue

        if not bars:
            continue

        # ── aggregate to daily ────────────────────────────────────────────
        daily: dict[str, dict] = {}
        for b in bars:
            ts = b.get("timestamp") or b.get("ts")
            if ts is None:
                continue
            d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
            if d not in daily:
                daily[d] = {"open": float(b["open"]), "high": float(b["high"]),
                             "low": float(b["low"]), "close": float(b["close"])}
            else:
                daily[d]["high"]  = max(daily[d]["high"],  float(b["high"]))
                daily[d]["low"]   = min(daily[d]["low"],   float(b["low"]))
                daily[d]["close"] = float(b["close"])

        sorted_dates  = sorted(daily)
        daily_closes  = [daily[d]["close"] for d in sorted_dates]
        ema_vals      = _ema(daily_closes, ema_period)
        daily_ema_out[ticker] = {
            d: {"close": daily_closes[i], "ema": ema_vals[i]}
            for i, d in enumerate(sorted_dates)
        }

        # ── aggregate to 4H ───────────────────────────────────────────────
        buckets: dict[datetime, dict] = {}
        for b in bars:
            ts = b.get("timestamp") or b.get("ts")
            if ts is None:
                continue
            t4h = ts.replace(
                hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0
            )
            if t4h not in buckets:
                buckets[t4h] = {"open": float(b["open"]), "high": float(b["high"]),
                                 "low": float(b["low"]),  "close": float(b["close"])}
            else:
                buckets[t4h]["high"]  = max(buckets[t4h]["high"],  float(b["high"]))
                buckets[t4h]["low"]   = min(buckets[t4h]["low"],   float(b["low"]))
                buckets[t4h]["close"] = float(b["close"])

        sorted_4h = sorted(buckets)
        h4_h = [buckets[t]["high"]  for t in sorted_4h]
        h4_l = [buckets[t]["low"]   for t in sorted_4h]
        h4_c = [buckets[t]["close"] for t in sorted_4h]
        adx_vals = _adx(h4_h, h4_l, h4_c, adx_period)
        adx_4h_out[ticker] = [
            (t.isoformat(), adx_vals[i])
            for i, t in enumerate(sorted_4h)
            if adx_vals[i] is not None
        ]

    return {
        "daily_ema": daily_ema_out,
        "4h_adx":    adx_4h_out,
        "ema_period": ema_period,
        "adx_period": adx_period,
    }


def _lookup_per_asset_technical_signal(
    *,
    entry_ts: datetime,
    ticker: str,
    direction: str,
    technical_state: dict,
    ema_lag_days: int = 1,
    ema_above_long_mult: float = 1.0,
    ema_above_short_mult: float = 0.5,
    ema_below_long_mult: float = 0.5,
    ema_below_short_mult: float = 1.0,
    use_adx_gate: bool = False,
    adx_strong_threshold: float = 25.0,
    adx_weak_threshold: float = 20.0,
    adx_weak_mult: float = 0.5,
) -> tuple[str | None, float, float | None, float]:
    """Lookup daily EMA regime and 4H ADX gate for a candidate entry.

    Returns (ema_regime, ema_mult, adx_value, adx_mult).
    """
    daily_ema = technical_state.get("daily_ema", {})
    adx_4h    = technical_state.get("4h_adx",    {})

    # ── daily EMA ─────────────────────────────────────────────────────────
    ema_regime: str | None = None
    ema_mult = 1.0
    asset_daily = daily_ema.get(ticker, {})
    if asset_daily:
        lookup_d = (entry_ts.date() - timedelta(days=max(ema_lag_days, 0)))
        for lag in range(6):
            d_str = (lookup_d - timedelta(days=lag)).isoformat()
            if d_str in asset_daily:
                row = asset_daily[d_str]
                ema_val = row.get("ema")
                if ema_val is not None:
                    above = row["close"] >= ema_val
                    ema_regime = "above_ema" if above else "below_ema"
                    if direction == "long":
                        ema_mult = ema_above_long_mult if above else ema_below_long_mult
                    else:
                        ema_mult = ema_above_short_mult if above else ema_below_short_mult
                break

    # ── 4H ADX gate ───────────────────────────────────────────────────────
    adx_value: float | None = None
    adx_mult = 1.0
    if use_adx_gate:
        asset_adx = adx_4h.get(ticker, [])
        if asset_adx:
            # find last completed 4H bar strictly before entry_ts
            cutoff = entry_ts.isoformat()
            lo, hi = 0, len(asset_adx)
            while lo < hi:
                mid = (lo + hi) // 2
                if asset_adx[mid][0] < cutoff:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx >= 0:
                adx_value = asset_adx[idx][1]
                if adx_value is not None and adx_value < adx_weak_threshold:
                    adx_mult = adx_weak_mult

    return ema_regime, ema_mult, adx_value, adx_mult


def _realized_drawdown_pct(capital: float, peak_capital: float) -> float:
    if peak_capital <= 0:
        return 0.0
    return (peak_capital - capital) / peak_capital * 100.0


def _candidate_return_pct(candidate: dict) -> float:
    position_size = float(candidate.get("position_size", 0.0) or 0.0)
    if position_size <= 0:
        return 0.0
    return float(candidate.get("pnl", 0.0) or 0.0) / position_size


def _candidate_requested_position_size(
    *,
    candidate: dict,
    capital: float,
    max_position_pct: float,
) -> float:
    entry_price = float(candidate.get("entry_price", 0.0) or 0.0)
    if capital <= 0 or entry_price <= 0:
        return 0.0

    risk_pct = float(candidate.get("risk_pct", 0.0) or 0.0)
    stop_loss_raw = candidate.get("stop_loss")
    stop_loss = float(stop_loss_raw) if stop_loss_raw is not None else None
    sl_distance = abs(entry_price - stop_loss) if stop_loss is not None else 0.0
    if risk_pct > 0 and sl_distance > 1e-9:
        requested_risk_amount = capital * risk_pct
        requested_shares = requested_risk_amount / sl_distance
        requested_notional = requested_shares * entry_price
        return min(requested_notional, capital * max_position_pct)

    # Backward-compatible fallback for older precomputed candidate payloads.
    return float(candidate.get("position_size", 0.0) or 0.0)


def _decayed_trade_return_score(returns: list[float], lookback_trades: int, decay: float) -> float:
    if lookback_trades <= 0 or decay <= 0:
        return 0.0
    recent = returns[-lookback_trades:]
    if not recent:
        return 0.0
    total = 0.0
    weights = 0.0
    for age, value in enumerate(reversed(recent)):
        weight = decay**age
        total += float(value) * weight
        weights += weight
    return total / weights if weights > 0 else 0.0


def _performance_leadership_scale(
    *,
    ticker: str,
    closed_trade_returns_by_ticker: dict[str, list[float]],
    lookback_trades: int,
    decay: float,
    floor_mult: float,
    cap_mult: float,
    min_history: int,
) -> tuple[float, float | None, float | None, int]:
    if lookback_trades <= 0 or min_history <= 0:
        return 1.0, None, None, 0

    scores: dict[str, float] = {}
    for asset_ticker, returns in closed_trade_returns_by_ticker.items():
        if len(returns) < min_history:
            continue
        scores[str(asset_ticker)] = _decayed_trade_return_score(
            returns=returns,
            lookback_trades=lookback_trades,
            decay=decay,
        )
    if ticker not in scores or len(scores) < 2:
        return 1.0, scores.get(ticker), None, len(scores)

    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    ticker_idx = next(idx for idx, item in enumerate(ordered) if item[0] == ticker)
    rank_pct = ticker_idx / (len(ordered) - 1) if len(ordered) > 1 else 0.5
    mult = floor_mult + (rank_pct * (cap_mult - floor_mult))
    return mult, scores[ticker], rank_pct, len(scores)


def _active_exposure_mult(
    *,
    base_exposure_mult: float,
    capital: float,
    peak_capital: float,
    # Drawdown governor
    use_drawdown_governor: bool,
    drawdown_trigger_1_pct: float,
    drawdown_exposure_mult_1: float,
    drawdown_trigger_2_pct: float,
    drawdown_exposure_mult_2: float,
    # Sentiment governor
    use_sentiment_governor: bool = False,
    sentiment_score: float | None = None,
    sentiment_threshold_1: float = 45.0,
    sentiment_threshold_2: float = 25.0,
    sentiment_exposure_mult_1: float = 1.0,
    sentiment_exposure_mult_2: float = 0.5,
    sentiment_reversal_recent_low: float | None = None,
    sentiment_reversal_min_rise: float = 10.0,
    sentiment_reversal_mult: float = 1.0,
) -> tuple[float, float | None]:
    """
    Return ``(active_mult, sentiment_score_used)``.

    Drawdown governor and sentiment governor are independent layers.
    The final multiplier is the minimum of both; the more conservative limit wins.
    """
    if use_drawdown_governor:
        drawdown_pct = _realized_drawdown_pct(capital=capital, peak_capital=peak_capital)
        if drawdown_pct >= drawdown_trigger_2_pct:
            dd_mult = drawdown_exposure_mult_2
        elif drawdown_pct >= drawdown_trigger_1_pct:
            dd_mult = drawdown_exposure_mult_1
        else:
            dd_mult = base_exposure_mult
    else:
        dd_mult = base_exposure_mult

    if not use_sentiment_governor or sentiment_score is None:
        return dd_mult, sentiment_score

    s_mult = _sentiment_regime_mult(
        base_mult=base_exposure_mult,
        sentiment_score=sentiment_score,
        sentiment_threshold_1=sentiment_threshold_1,
        sentiment_threshold_2=sentiment_threshold_2,
        sentiment_mult_1=sentiment_exposure_mult_1,
        sentiment_mult_2=sentiment_exposure_mult_2,
        sentiment_reversal_recent_low=sentiment_reversal_recent_low,
        sentiment_reversal_min_rise=sentiment_reversal_min_rise,
        sentiment_reversal_mult=sentiment_reversal_mult,
    )

    return min(dd_mult, s_mult), sentiment_score


def _sentiment_regime_mult(
    *,
    base_mult: float,
    sentiment_score: float,
    sentiment_threshold_1: float,
    sentiment_threshold_2: float,
    sentiment_mult_1: float,
    sentiment_mult_2: float,
    sentiment_reversal_recent_low: float | None,
    sentiment_reversal_min_rise: float,
    sentiment_reversal_mult: float,
) -> float:
    if sentiment_score < sentiment_threshold_2:
        if (
            sentiment_reversal_recent_low is not None
            and (sentiment_score - sentiment_reversal_recent_low) >= sentiment_reversal_min_rise
        ):
            return sentiment_reversal_mult
        return sentiment_mult_2
    if sentiment_score < sentiment_threshold_1:
        return sentiment_mult_1
    return base_mult


def _lookup_sentiment_signal(
    *,
    entry_ts: datetime,
    sentiment_scores: dict[str, float] | None,
    sentiment_lag_days: int,
    sentiment_reversal_window: int,
) -> tuple[float | None, float | None]:
    if not sentiment_scores:
        return None, None

    lookup_ts = entry_ts - timedelta(days=sentiment_lag_days)
    sentiment_score = get_score_for_date(
        lookup_ts.strftime("%Y-%m-%d"),
        sentiment_scores,
    )
    if sentiment_score is None or sentiment_reversal_window <= 0:
        return sentiment_score, None

    window_scores = [
        get_score_for_date(
            (lookup_ts - timedelta(days=offset)).strftime("%Y-%m-%d"),
            sentiment_scores,
        )
        for offset in range(1, sentiment_reversal_window + 1)
    ]
    valid_window_scores = [score for score in window_scores if score is not None]
    recent_low = min(valid_window_scores) if valid_window_scores else None
    return sentiment_score, recent_low


def _simple_moving_average(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= window:
            total -= values[idx - window]
        if idx >= window - 1:
            out[idx] = total / window
    return out


def _exponential_moving_average(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("EMA period must be positive")
    out: list[float | None] = [None] * len(values)
    alpha = 2.0 / (period + 1.0)
    ema_value: float | None = None
    for idx, value in enumerate(values):
        ema_value = float(value) if ema_value is None else (alpha * float(value) + (1.0 - alpha) * ema_value)
        out[idx] = ema_value
    return out


def build_intraday_volatility_proxy_state(
    *,
    proxy_bars: list[dict],
    short_ma_bars: int = 78,
    long_ma_bars: int = 390,
    interval_minutes: int = 5,
) -> dict[str, list | int]:
    if short_ma_bars <= 0 or long_ma_bars <= 0:
        raise ValueError("intraday volatility proxy MA lengths must be positive")
    bars = sorted(proxy_bars, key=lambda row: row["timestamp"])
    timestamps = [row["timestamp"] for row in bars]
    closes = [float(row["close"]) for row in bars]
    return {
        "timestamps": timestamps,
        "closes": closes,
        "sma_short": _simple_moving_average(closes, short_ma_bars),
        "sma_long": _simple_moving_average(closes, long_ma_bars),
        "short_ma_bars": short_ma_bars,
        "long_ma_bars": long_ma_bars,
        "interval_minutes": interval_minutes,
    }


def build_daily_volatility_reference_state(
    *,
    daily_closes: dict[str, float],
    ema_period: int = 20,
) -> dict[str, list | int]:
    if ema_period <= 0:
        raise ValueError("daily volatility EMA period must be positive")
    ordered = sorted((date.fromisoformat(day), float(value)) for day, value in daily_closes.items())
    dates = [item[0] for item in ordered]
    closes = [item[1] for item in ordered]
    return {
        "dates": dates,
        "closes": closes,
        "ema": _exponential_moving_average(closes, ema_period),
        "ema_period": ema_period,
    }


def build_intraday_volatility_relative_state(
    *,
    proxy_bars: list[dict],
    ema_period: int = 78,
    interval_minutes: int = 5,
) -> dict[str, list | int]:
    if ema_period <= 0:
        raise ValueError("intraday volatility EMA period must be positive")
    bars = sorted(proxy_bars, key=lambda row: row["timestamp"])
    timestamps = [row["timestamp"] for row in bars]
    closes = [float(row["close"]) for row in bars]
    return {
        "timestamps": timestamps,
        "closes": closes,
        "ema": _exponential_moving_average(closes, ema_period),
        "ema_period": ema_period,
        "interval_minutes": interval_minutes,
    }


def build_extended_hours_proxy_state(
    *,
    daily_vix_closes: dict[str, float],
    crypto_fg_scores: dict[str, float],
) -> dict:
    """Pre-computed state for the extended-hours overlay.

    Used for sessions outside US equity hours (e.g. hong_kong_open):
      - non-crypto asset buckets  → daily VIX close threshold regime
      - crypto asset bucket       → Crypto Fear & Greed score regime
    """
    return {
        "daily_vix_closes": {str(k): float(v) for k, v in daily_vix_closes.items()},
        "crypto_fg_scores":  {str(k): float(v) for k, v in crypto_fg_scores.items()},
    }


def _score_with_fallback(scores: dict[str, float], date_str: str, max_lag: int = 5) -> float | None:
    """Return the score for date_str, falling back up to max_lag days earlier."""
    from datetime import date as _date
    d = _date.fromisoformat(date_str)
    for lag in range(max_lag + 1):
        key = (d - timedelta(days=lag)).isoformat()
        if key in scores:
            return float(scores[key])
    return None


def _lookup_extended_hours_signal(
    *,
    entry_ts: datetime,
    session_open: str,
    asset_bucket: str,
    direction: str,
    extended_hours_state: dict | None,
    lag_days: int = 1,
    vix_risk_on_threshold: float = 15.0,
    vix_risk_off_threshold: float = 25.0,
    fg_greed_threshold: float = 60.0,
    fg_fear_threshold: float = 30.0,
    long_risk_on_mult: float = 1.0,
    long_neutral_mult: float = 1.0,
    long_risk_off_mult: float = 0.5,
    short_risk_on_mult: float = 0.5,
    short_neutral_mult: float = 1.0,
    short_risk_off_mult: float = 1.0,
) -> tuple[str | None, float, str | None, float | None]:
    """Regime signal for trades outside the US equity session.

    Returns (regime, multiplier, source_label, raw_score).
    NY equity-open trades are left to the intraday VIXY proxy and return no-op.
    For all other sessions:
      - crypto bucket  → Crypto Fear & Greed (0-100; high = greed = risk-on)
      - other buckets  → daily VIX close (low = calm = risk-on)
    """
    if (
        not extended_hours_state
        or direction not in {"long", "short"}
    ):
        return None, 1.0, None, None

    lookup_date = (entry_ts.date() - timedelta(days=max(lag_days, 0))).isoformat()

    if asset_bucket == "crypto":
        scores = extended_hours_state.get("crypto_fg_scores", {})
        raw = _score_with_fallback(scores, lookup_date)
        if raw is None:
            return None, 1.0, "CryptoFG", None
        if raw >= fg_greed_threshold:
            regime = "risk_on_micro"
        elif raw <= fg_fear_threshold:
            regime = "risk_off_micro"
        else:
            regime = "neutral_micro"
        source = "CryptoFG"
    else:
        vix_closes = extended_hours_state.get("daily_vix_closes", {})
        raw = _score_with_fallback(vix_closes, lookup_date)
        if raw is None:
            return None, 1.0, "VIX_daily", None
        if raw <= vix_risk_on_threshold:
            regime = "risk_on_micro"
        elif raw >= vix_risk_off_threshold:
            regime = "risk_off_micro"
        else:
            regime = "neutral_micro"
        source = "VIX_daily"

    if direction == "long":
        mult_map = {
            "risk_on_micro":  long_risk_on_mult,
            "neutral_micro":  long_neutral_mult,
            "risk_off_micro": long_risk_off_mult,
        }
    else:
        mult_map = {
            "risk_on_micro":  short_risk_on_mult,
            "neutral_micro":  short_neutral_mult,
            "risk_off_micro": short_risk_off_mult,
        }
    return regime, float(mult_map.get(regime, 1.0)), source, raw


def _intraday_volatility_regime(
    *,
    close: float,
    sma_short: float | None,
    sma_long: float | None,
) -> str | None:
    if sma_short is None or sma_long is None:
        return None
    if close <= sma_short and sma_short <= sma_long:
        return "risk_on_micro"
    if close > sma_short and sma_short > sma_long:
        return "risk_off_micro"
    return "neutral_micro"


def _lookup_intraday_volatility_signal(
    *,
    entry_ts: datetime,
    session_open: str,
    asset_bucket: str,
    direction: str,
    proxy_state: dict[str, list | int] | None,
    max_age_minutes: int,
    lag_bars: int,
    allowed_buckets: frozenset[str] | None,
    long_risk_on_mult: float,
    long_neutral_mult: float,
    long_risk_off_mult: float,
    short_risk_on_mult: float,
    short_neutral_mult: float,
    short_risk_off_mult: float,
) -> tuple[str | None, float, float | None, float | None]:
    if (
        not proxy_state
        or session_open != "new_york_equity_open"
        or direction not in {"long", "short"}
        or (allowed_buckets is not None and asset_bucket not in allowed_buckets)
    ):
        return None, 1.0, None, None

    interval_minutes = int(proxy_state.get("interval_minutes", 5) or 5)
    lookup_ts = entry_ts - timedelta(minutes=max(lag_bars, 0) * interval_minutes)
    timestamps = proxy_state["timestamps"]
    idx = bisect.bisect_right(timestamps, lookup_ts) - 1
    if idx < 0:
        return None, 1.0, None, None

    matched_ts = timestamps[idx]
    signal_age_min = (entry_ts - matched_ts).total_seconds() / 60.0
    if signal_age_min > max_age_minutes:
        return None, 1.0, signal_age_min, None

    closes = proxy_state["closes"]
    sma_short = proxy_state["sma_short"]
    sma_long = proxy_state["sma_long"]
    regime = _intraday_volatility_regime(
        close=float(closes[idx]),
        sma_short=sma_short[idx],
        sma_long=sma_long[idx],
    )
    if regime is None:
        return None, 1.0, signal_age_min, float(closes[idx])

    if direction == "long":
        mult_map = {
            "risk_on_micro": long_risk_on_mult,
            "neutral_micro": long_neutral_mult,
            "risk_off_micro": long_risk_off_mult,
        }
    else:
        mult_map = {
            "risk_on_micro": short_risk_on_mult,
            "neutral_micro": short_neutral_mult,
            "risk_off_micro": short_risk_off_mult,
        }
    return regime, float(mult_map.get(regime, 1.0)), signal_age_min, float(closes[idx])


def _lookup_volatility_persistence_signal(
    *,
    entry_ts: datetime,
    session_open: str,
    direction: str,
    daily_vix_state: dict[str, list | int] | None,
    intraday_vixy_state: dict[str, list | int] | None,
    daily_lag_days: int,
    intraday_max_age_minutes: int,
    intraday_lag_bars: int,
    ratio_upper: float,
    ratio_lower: float,
    daily_stress_min_rel: float,
    long_persistent_stress_mult: float,
    long_neutral_mult: float,
    long_fading_stress_mult: float,
    short_persistent_stress_mult: float,
    short_neutral_mult: float,
    short_fading_stress_mult: float,
) -> tuple[str | None, float, float | None, float | None, float | None, float | None]:
    if (
        not daily_vix_state
        or not intraday_vixy_state
        or session_open != "new_york_equity_open"
        or direction not in {"long", "short"}
    ):
        return None, 1.0, None, None, None, None

    target_day = (entry_ts - timedelta(days=max(daily_lag_days, 0))).date()
    daily_dates = daily_vix_state["dates"]
    daily_idx = bisect.bisect_right(daily_dates, target_day) - 1
    if daily_idx < 0:
        return None, 1.0, None, None, None, None

    daily_closes = daily_vix_state["closes"]
    daily_ema = daily_vix_state["ema"]
    if daily_ema[daily_idx] in (None, 0):
        return None, 1.0, None, None, None, None
    vix_rel = float(daily_closes[daily_idx]) / float(daily_ema[daily_idx])

    interval_minutes = int(intraday_vixy_state.get("interval_minutes", 5) or 5)
    lookup_ts = entry_ts - timedelta(minutes=max(intraday_lag_bars, 0) * interval_minutes)
    intraday_timestamps = intraday_vixy_state["timestamps"]
    intraday_idx = bisect.bisect_right(intraday_timestamps, lookup_ts) - 1
    if intraday_idx < 0:
        return None, 1.0, None, vix_rel, None, None

    matched_ts = intraday_timestamps[intraday_idx]
    signal_age_min = (entry_ts - matched_ts).total_seconds() / 60.0
    if signal_age_min > intraday_max_age_minutes:
        return None, 1.0, signal_age_min, vix_rel, None, None

    intraday_closes = intraday_vixy_state["closes"]
    intraday_ema = intraday_vixy_state["ema"]
    if intraday_ema[intraday_idx] in (None, 0):
        return None, 1.0, signal_age_min, vix_rel, None, None
    vixy_rel = float(intraday_closes[intraday_idx]) / float(intraday_ema[intraday_idx])
    ratio = vixy_rel / vix_rel if vix_rel > 0 else None
    if ratio is None:
        return None, 1.0, signal_age_min, vix_rel, vixy_rel, None

    regime = "neutral_persistence"
    if vix_rel >= daily_stress_min_rel and ratio >= ratio_upper:
        regime = "persistent_stress"
    elif vix_rel >= daily_stress_min_rel and ratio <= ratio_lower:
        regime = "fading_stress"

    if direction == "long":
        mult_map = {
            "persistent_stress": long_persistent_stress_mult,
            "neutral_persistence": long_neutral_mult,
            "fading_stress": long_fading_stress_mult,
        }
    else:
        mult_map = {
            "persistent_stress": short_persistent_stress_mult,
            "neutral_persistence": short_neutral_mult,
            "fading_stress": short_fading_stress_mult,
        }

    return regime, float(mult_map.get(regime, 1.0)), signal_age_min, vix_rel, vixy_rel, ratio


def _build_yearly_rows(executed_trades: list[dict], initial_capital: float) -> list[dict]:
    grouped: dict[int, dict] = defaultdict(
        lambda: {
            "pnl": 0.0,
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "winning_trades": 0,
        }
    )
    for trade in executed_trades:
        year = datetime.fromisoformat(trade["exit_ts"]).year
        row = grouped[year]
        row["pnl"] += float(trade["net_pnl"])
        row["trades"] += 1
        if trade["direction"] == "long":
            row["long_trades"] += 1
        else:
            row["short_trades"] += 1
        if float(trade["net_pnl"]) > 0:
            row["winning_trades"] += 1

    start_equity = float(initial_capital)
    yearly_rows: list[dict] = []
    for year in sorted(grouped):
        row = grouped[year]
        pnl = float(row["pnl"])
        end_equity = start_equity + pnl
        yearly_rows.append(
            {
                "year": year,
                "start_equity": round(start_equity, 4),
                "end_equity": round(end_equity, 4),
                "pnl": round(pnl, 4),
                "return_pct": round((pnl / start_equity * 100.0) if start_equity > 0 else 0.0, 2),
                "trades": row["trades"],
                "long_trades": row["long_trades"],
                "short_trades": row["short_trades"],
                "win_rate_pct": round((row["winning_trades"] / row["trades"] * 100.0) if row["trades"] else 0.0, 2),
            }
        )
        start_equity = end_equity
    return yearly_rows


def _build_asset_rows(executed_trades: list[dict], total_pnl: float) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "source": "",
            "asset_bucket": "",
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "pnl": 0.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        }
    )
    for trade in executed_trades:
        ticker = trade["ticker"]
        row = grouped[ticker]
        row["source"] = trade["source"]
        row["asset_bucket"] = trade["asset_bucket"]
        row["trades"] += 1
        row["pnl"] += float(trade["net_pnl"])
        if trade["direction"] == "long":
            row["long_trades"] += 1
            row["long_pnl"] += float(trade["net_pnl"])
        else:
            row["short_trades"] += 1
            row["short_pnl"] += float(trade["net_pnl"])

    rows: list[dict] = []
    for ticker, row in sorted(grouped.items(), key=lambda item: item[1]["pnl"], reverse=True):
        rows.append(
            {
                "ticker": ticker,
                "source": row["source"],
                "asset_bucket": row["asset_bucket"],
                "trades": row["trades"],
                "long_trades": row["long_trades"],
                "short_trades": row["short_trades"],
                "pnl": round(row["pnl"], 4),
                "pnl_share_pct": round((row["pnl"] / total_pnl * 100.0) if abs(total_pnl) > 1e-9 else 0.0, 2),
                "long_pnl": round(row["long_pnl"], 4),
                "short_pnl": round(row["short_pnl"], 4),
            }
        )
    return rows


def build_session_turtle_shared_account_candidates(
    *,
    basket: str = "expanded",
    investable_universe_asof: date | datetime | str | None = None,
    initial_capital: float = 1_000.0,
    lookback_years: float = 4.1,
    channel_period: int = 20,
    base_risk_pct: float = 0.05,
    fixed_stop_pct: float = 0.10,
    directional_volume_risk_pct: float = 0.07,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    use_extended_hours_protective_exits: bool = False,
    extended_hours_core_session_minutes: int = 390,
) -> list[dict]:
    universe = _resolve_investable_universe(
        basket=basket,
        investable_universe_asof=investable_universe_asof,
    )
    candidates: list[dict] = []
    for combo_idx, (ticker, source, session_open) in enumerate(universe):
        use_extended_hours_mode = (
            use_extended_hours_protective_exits
            and source == "tiingo"
            and session_open == "new_york_equity_open"
        )
        payload = run_session_turtle_trend_backtest(
            ticker=ticker,
            initial_capital=initial_capital,
            interval="5m",
            lookback_years=lookback_years,
            market_data_source=source,
            session_open=session_open,
            channel_period=channel_period,
            base_risk_pct=base_risk_pct,
            max_position_pct=0.90,
            fixed_stop_pct=fixed_stop_pct,
            entry_window_minutes=extended_hours_core_session_minutes if use_extended_hours_mode else 480,
            core_session_minutes=extended_hours_core_session_minutes if use_extended_hours_mode else None,
            use_4h_trend_filter=True,
            trend_fast_period=trend_fast_period,
            trend_slow_period=trend_slow_period,
            use_extended_hours_protective_exits_only=use_extended_hours_mode,
            use_directional_volume_risk_boost=True,
            directional_volume_min_rel_volume=1.25,
            directional_volume_close_location_threshold=0.65,
            directional_volume_risk_pct=directional_volume_risk_pct,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        for trade_idx, trade in enumerate(payload["trades"]):
            entry_price = float(trade["entry_price"])
            stop_loss = trade.get("stop_loss")
            if stop_loss is None and fixed_stop_pct is not None:
                if str(trade["direction"]) == "long":
                    stop_loss = entry_price * (1.0 - fixed_stop_pct)
                else:
                    stop_loss = entry_price * (1.0 + fixed_stop_pct)
            candidates.append(
                {
                    "combo_idx": combo_idx,
                    "trade_idx": trade_idx,
                    "ticker": ticker,
                    "source": source,
                    "session_open": session_open,
                    "direction": trade["direction"],
                    "entry_ts": datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts": datetime.fromisoformat(trade["exit_date"]),
                    "entry_price": entry_price,
                    "exit_price": float(trade["exit_price"]),
                    "stop_loss": float(stop_loss) if stop_loss is not None else None,
                    "risk_pct": float(trade.get("risk_pct", 0.0) or 0.0),
                    "shares": float(trade["shares"]),
                    "position_size": float(trade["position_size"]),
                    "pnl": float(trade["pnl"]),
                    "risk_model": str(trade["risk_model"]),
                    "entry_rel_volume": float(trade["entry_rel_volume"]),
                    "asset_bucket": _asset_bucket(ticker),
                }
            )
    return candidates


def generate_session_turtle_shared_account_report(
    *,
    basket: str = "expanded",
    investable_universe_asof: date | datetime | str | None = None,
    exposure_mult: float = 2.0,
    use_drawdown_governor: bool = False,
    drawdown_trigger_1_pct: float = 10.0,
    drawdown_exposure_mult_1: float = 1.5,
    drawdown_trigger_2_pct: float = 20.0,
    drawdown_exposure_mult_2: float = 1.0,
    crypto_cap_mult: float | None = None,
    gold_cap_mult: float | None = None,
    metals_cap_mult: float | None = None,
    equity_cap_mult: float | None = None,
    initial_capital: float = 1_000.0,
    lookback_years: float = 4.1,
    channel_period: int = 20,
    base_risk_pct: float = 0.05,
    fixed_stop_pct: float = 0.10,
    directional_volume_risk_pct: float = 0.07,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    base_portfolio_cap_pct: float = 0.90,
    use_performance_leadership_overlay: bool = False,
    performance_lookback_trades: int = 6,
    performance_decay: float = 0.75,
    performance_floor_mult: float = 0.75,
    performance_cap_mult: float = 1.25,
    performance_min_history: int = 3,
    use_extended_hours_protective_exits: bool = False,
    extended_hours_core_session_minutes: int = 390,
    use_sentiment_governor: bool = False,
    sentiment_scores: dict[str, float] | None = None,
    sentiment_lag_days: int = 1,
    sentiment_threshold_1: float = 45.0,
    sentiment_threshold_2: float = 25.0,
    sentiment_exposure_mult_1: float = 1.0,
    sentiment_exposure_mult_2: float = 0.5,
    sentiment_reversal_window: int = 10,
    sentiment_reversal_min_rise: float = 10.0,
    sentiment_reversal_mult: float = 1.0,
    use_direct_bucket_sentiment_sizing: bool = False,
    bucket_sentiment_scores: dict[str, dict[str, float]] | None = None,
    bucket_sentiment_lag_days: int = 1,
    bucket_sentiment_threshold_1: float = 45.0,
    bucket_sentiment_threshold_2: float = 25.0,
    bucket_sentiment_size_mult_1: float = 0.75,
    bucket_sentiment_size_mult_2: float = 0.5,
    bucket_sentiment_reversal_window: int = 10,
    bucket_sentiment_reversal_min_rise: float = 10.0,
    bucket_sentiment_reversal_mult: float = 0.75,
    use_intraday_volatility_proxy: bool = False,
    intraday_volatility_proxy_state: dict[str, list | int] | None = None,
    intraday_volatility_proxy_label: str = "VIXY",
    intraday_volatility_proxy_max_age_minutes: int = 60,
    intraday_volatility_proxy_lag_bars: int = 1,
    intraday_volatility_proxy_short_ma_bars: int = 78,
    intraday_volatility_proxy_long_ma_bars: int = 390,
    intraday_volatility_long_risk_on_mult: float = 1.0,
    intraday_volatility_long_neutral_mult: float = 1.0,
    intraday_volatility_long_risk_off_mult: float = 0.5,
    intraday_volatility_short_risk_on_mult: float = 0.5,
    intraday_volatility_short_neutral_mult: float = 1.0,
    intraday_volatility_short_risk_off_mult: float = 1.0,
    intraday_volatility_proxy_buckets: frozenset[str] | None = None,
    use_extended_hours_proxy: bool = False,
    extended_hours_proxy_state: dict | None = None,
    extended_hours_proxy_lag_days: int = 1,
    extended_hours_vix_risk_on_threshold: float = 15.0,
    extended_hours_vix_risk_off_threshold: float = 25.0,
    extended_hours_fg_greed_threshold: float = 60.0,
    extended_hours_fg_fear_threshold: float = 30.0,
    extended_hours_long_risk_on_mult: float = 1.0,
    extended_hours_long_neutral_mult: float = 1.0,
    extended_hours_long_risk_off_mult: float = 0.5,
    extended_hours_short_risk_on_mult: float = 0.5,
    extended_hours_short_neutral_mult: float = 1.0,
    extended_hours_short_risk_off_mult: float = 1.0,
    use_volatility_persistence_overlay: bool = False,
    daily_vix_reference_state: dict[str, list | int] | None = None,
    intraday_vixy_relative_state: dict[str, list | int] | None = None,
    volatility_persistence_label: str = "VIX/VIXY persistence",
    volatility_persistence_daily_lag_days: int = 1,
    volatility_persistence_intraday_max_age_minutes: int = 60,
    volatility_persistence_intraday_lag_bars: int = 1,
    volatility_persistence_daily_ema_period: int = 20,
    volatility_persistence_intraday_ema_period: int = 78,
    volatility_persistence_ratio_upper: float = 1.05,
    volatility_persistence_ratio_lower: float = 0.95,
    volatility_persistence_daily_stress_min_rel: float = 1.0,
    volatility_persistence_long_persistent_stress_mult: float = 0.5,
    volatility_persistence_long_neutral_mult: float = 1.0,
    volatility_persistence_long_fading_stress_mult: float = 1.0,
    volatility_persistence_short_persistent_stress_mult: float = 1.0,
    volatility_persistence_short_neutral_mult: float = 1.0,
    volatility_persistence_short_fading_stress_mult: float = 0.5,
    use_per_asset_technical_overlay: bool = False,
    per_asset_technical_state: dict | None = None,
    per_asset_ema_lag_days: int = 1,
    per_asset_ema_above_long_mult: float = 1.0,
    per_asset_ema_above_short_mult: float = 0.5,
    per_asset_ema_below_long_mult: float = 0.5,
    per_asset_ema_below_short_mult: float = 1.0,
    per_asset_use_adx_gate: bool = False,
    per_asset_adx_strong_threshold: float = 25.0,
    per_asset_adx_weak_threshold: float = 20.0,
    per_asset_adx_weak_mult: float = 0.5,
    precomputed_candidates: list[dict] | None = None,
) -> dict:
    if exposure_mult <= 0:
        raise ValueError("exposure_mult must be positive")
    if drawdown_trigger_1_pct < 0 or drawdown_trigger_2_pct < 0:
        raise ValueError("drawdown triggers must be non-negative")
    if drawdown_trigger_2_pct <= drawdown_trigger_1_pct:
        raise ValueError("drawdown_trigger_2_pct must be greater than drawdown_trigger_1_pct")
    if drawdown_exposure_mult_1 <= 0 or drawdown_exposure_mult_2 <= 0:
        raise ValueError("drawdown exposure multipliers must be positive")
    if drawdown_exposure_mult_1 > exposure_mult:
        raise ValueError("drawdown_exposure_mult_1 must be <= exposure_mult")
    if drawdown_exposure_mult_2 > drawdown_exposure_mult_1:
        raise ValueError("drawdown_exposure_mult_2 must be <= drawdown_exposure_mult_1")
    for label, cap_mult in (
        ("crypto_cap_mult", crypto_cap_mult),
        ("gold_cap_mult", gold_cap_mult),
        ("metals_cap_mult", metals_cap_mult),
        ("equity_cap_mult", equity_cap_mult),
    ):
        if cap_mult is not None and cap_mult <= 0:
            raise ValueError(f"{label} must be positive when provided")
    if performance_lookback_trades <= 0:
        raise ValueError("performance_lookback_trades must be positive")
    if not 0 < performance_decay <= 1.0:
        raise ValueError("performance_decay must be between 0 and 1")
    if performance_floor_mult <= 0 or performance_cap_mult <= 0:
        raise ValueError("performance floor/cap multipliers must be positive")
    if performance_floor_mult > performance_cap_mult:
        raise ValueError("performance_floor_mult must be <= performance_cap_mult")
    if not performance_floor_mult <= 1.0 <= performance_cap_mult:
        raise ValueError("performance overlay must bracket neutral sizing at 1.0")
    if performance_min_history <= 0:
        raise ValueError("performance_min_history must be positive")
    if extended_hours_core_session_minutes <= 0:
        raise ValueError("extended_hours_core_session_minutes must be positive")
    if sentiment_lag_days < 0:
        raise ValueError("sentiment_lag_days must be non-negative")
    if sentiment_threshold_2 >= sentiment_threshold_1:
        raise ValueError("sentiment_threshold_2 must be less than sentiment_threshold_1")
    if sentiment_exposure_mult_1 <= 0 or sentiment_exposure_mult_2 <= 0 or sentiment_reversal_mult <= 0:
        raise ValueError("sentiment multipliers must be positive")
    if sentiment_exposure_mult_1 > exposure_mult:
        raise ValueError("sentiment_exposure_mult_1 must be <= exposure_mult")
    if sentiment_exposure_mult_2 > sentiment_exposure_mult_1:
        raise ValueError("sentiment_exposure_mult_2 must be <= sentiment_exposure_mult_1")
    if sentiment_reversal_mult > exposure_mult:
        raise ValueError("sentiment_reversal_mult must be <= exposure_mult")
    if sentiment_reversal_window < 0:
        raise ValueError("sentiment_reversal_window must be non-negative")
    if bucket_sentiment_lag_days < 0:
        raise ValueError("bucket_sentiment_lag_days must be non-negative")
    if bucket_sentiment_threshold_2 >= bucket_sentiment_threshold_1:
        raise ValueError("bucket_sentiment_threshold_2 must be less than bucket_sentiment_threshold_1")
    if (
        bucket_sentiment_size_mult_1 <= 0
        or bucket_sentiment_size_mult_2 <= 0
        or bucket_sentiment_reversal_mult <= 0
    ):
        raise ValueError("bucket sentiment multipliers must be positive")
    if bucket_sentiment_size_mult_1 > 1.0:
        raise ValueError("bucket_sentiment_size_mult_1 must be <= 1.0")
    if bucket_sentiment_size_mult_2 > bucket_sentiment_size_mult_1:
        raise ValueError("bucket_sentiment_size_mult_2 must be <= bucket_sentiment_size_mult_1")
    if bucket_sentiment_reversal_mult > 1.0:
        raise ValueError("bucket_sentiment_reversal_mult must be <= 1.0")
    if bucket_sentiment_reversal_window < 0:
        raise ValueError("bucket_sentiment_reversal_window must be non-negative")
    if use_intraday_volatility_proxy and not intraday_volatility_proxy_state:
        raise ValueError("intraday_volatility_proxy_state is required when use_intraday_volatility_proxy=True")
    if intraday_volatility_proxy_max_age_minutes < 0:
        raise ValueError("intraday_volatility_proxy_max_age_minutes must be non-negative")
    if intraday_volatility_proxy_lag_bars < 0:
        raise ValueError("intraday_volatility_proxy_lag_bars must be non-negative")
    if intraday_volatility_proxy_short_ma_bars <= 0 or intraday_volatility_proxy_long_ma_bars <= 0:
        raise ValueError("intraday volatility proxy MA lengths must be positive")
    for label, mult in (
        ("intraday_volatility_long_risk_on_mult", intraday_volatility_long_risk_on_mult),
        ("intraday_volatility_long_neutral_mult", intraday_volatility_long_neutral_mult),
        ("intraday_volatility_long_risk_off_mult", intraday_volatility_long_risk_off_mult),
        ("intraday_volatility_short_risk_on_mult", intraday_volatility_short_risk_on_mult),
        ("intraday_volatility_short_neutral_mult", intraday_volatility_short_neutral_mult),
        ("intraday_volatility_short_risk_off_mult", intraday_volatility_short_risk_off_mult),
    ):
        if mult <= 0 or mult > 1.0:
            raise ValueError(f"{label} must be > 0 and <= 1.0")
    if use_volatility_persistence_overlay and (not daily_vix_reference_state or not intraday_vixy_relative_state):
        raise ValueError(
            "daily_vix_reference_state and intraday_vixy_relative_state are required when "
            "use_volatility_persistence_overlay=True"
        )
    if volatility_persistence_daily_lag_days < 0:
        raise ValueError("volatility_persistence_daily_lag_days must be non-negative")
    if volatility_persistence_intraday_max_age_minutes < 0:
        raise ValueError("volatility_persistence_intraday_max_age_minutes must be non-negative")
    if volatility_persistence_intraday_lag_bars < 0:
        raise ValueError("volatility_persistence_intraday_lag_bars must be non-negative")
    if volatility_persistence_daily_ema_period <= 0 or volatility_persistence_intraday_ema_period <= 0:
        raise ValueError("volatility persistence EMA periods must be positive")
    if volatility_persistence_ratio_lower <= 0 or volatility_persistence_ratio_upper <= 0:
        raise ValueError("volatility persistence ratio thresholds must be positive")
    if volatility_persistence_ratio_lower >= volatility_persistence_ratio_upper:
        raise ValueError("volatility_persistence_ratio_lower must be < volatility_persistence_ratio_upper")
    if volatility_persistence_daily_stress_min_rel <= 0:
        raise ValueError("volatility_persistence_daily_stress_min_rel must be positive")
    for label, mult in (
        ("volatility_persistence_long_persistent_stress_mult", volatility_persistence_long_persistent_stress_mult),
        ("volatility_persistence_long_neutral_mult", volatility_persistence_long_neutral_mult),
        ("volatility_persistence_long_fading_stress_mult", volatility_persistence_long_fading_stress_mult),
        ("volatility_persistence_short_persistent_stress_mult", volatility_persistence_short_persistent_stress_mult),
        ("volatility_persistence_short_neutral_mult", volatility_persistence_short_neutral_mult),
        ("volatility_persistence_short_fading_stress_mult", volatility_persistence_short_fading_stress_mult),
    ):
        if mult <= 0 or mult > 1.0:
            raise ValueError(f"{label} must be > 0 and <= 1.0")
    asof_date = _normalize_asof_date(investable_universe_asof)
    if precomputed_candidates is None:
        candidates = build_session_turtle_shared_account_candidates(
            basket=basket,
            investable_universe_asof=investable_universe_asof,
            initial_capital=initial_capital,
            lookback_years=lookback_years,
            channel_period=channel_period,
            base_risk_pct=base_risk_pct,
            fixed_stop_pct=fixed_stop_pct,
            directional_volume_risk_pct=directional_volume_risk_pct,
            trend_fast_period=trend_fast_period,
            trend_slow_period=trend_slow_period,
            use_extended_hours_protective_exits=use_extended_hours_protective_exits,
            extended_hours_core_session_minutes=extended_hours_core_session_minutes,
        )
    else:
        candidates = list(precomputed_candidates)

    candidates.sort(key=lambda row: (row["entry_ts"], row["combo_idx"], row["trade_idx"]))
    closed_candidate_results = sorted(candidates, key=lambda row: (row["exit_ts"], row["combo_idx"], row["trade_idx"]))
    capital = float(initial_capital)
    peak_capital = capital
    max_drawdown = 0.0
    skipped_same_ticker = 0
    skipped_no_capacity = 0
    open_positions: list[_OpenTrade] = []
    executed_trades: list[dict] = []
    equity_curve: list[dict] = []
    asset_class_caps = {
        "crypto": crypto_cap_mult,
        "gold": gold_cap_mult,
        "metals": metals_cap_mult,
        "equity": equity_cap_mult,
    }
    closed_candidate_idx = 0
    closed_trade_returns_by_ticker: dict[str, list[float]] = defaultdict(list)

    def close_candidate_history_up_to(timestamp: datetime) -> None:
        nonlocal closed_candidate_idx
        while (
            closed_candidate_idx < len(closed_candidate_results)
            and closed_candidate_results[closed_candidate_idx]["exit_ts"] <= timestamp
        ):
            closed_candidate = closed_candidate_results[closed_candidate_idx]
            closed_trade_returns_by_ticker[str(closed_candidate["ticker"])].append(
                _candidate_return_pct(closed_candidate)
            )
            closed_candidate_idx += 1

    def close_positions_up_to(timestamp: datetime) -> None:
        nonlocal capital, peak_capital, max_drawdown, open_positions
        still_open: list[_OpenTrade] = []
        closing: list[_OpenTrade] = []
        for position in open_positions:
            if position.exit_ts <= timestamp:
                closing.append(position)
            else:
                still_open.append(position)
        open_positions = still_open

        for position in sorted(closing, key=lambda row: (row.exit_ts, row.combo_idx, row.trade_idx)):
            capital += position.scaled_pnl
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            equity_curve.append({"date": position.exit_ts.isoformat(), "equity": round(capital, 4)})
            executed_trades.append(
                {
                    "ticker": position.ticker,
                    "source": position.source,
                    "session_open": position.session_open,
                    "asset_bucket": position.asset_bucket,
                    "direction": position.direction,
                    "entry_ts": position.entry_ts.isoformat(),
                    "exit_ts": position.exit_ts.isoformat(),
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(position.exit_price, 4),
                    "shares": round(position.scaled_shares, 6),
                    "notional": round(position.scaled_position_size, 4),
                    "entry_exposure_mult": round(position.entry_exposure_mult, 4),
                    "entry_sentiment_score": (
                        round(position.entry_sentiment_score, 2)
                        if position.entry_sentiment_score is not None
                        else None
                    ),
                    "direct_sentiment_score": (
                        round(position.direct_sentiment_score, 2)
                        if position.direct_sentiment_score is not None
                        else None
                    ),
                    "direct_sentiment_size_mult": round(position.direct_sentiment_size_mult, 4),
                    "intraday_vol_proxy_regime": position.intraday_vol_proxy_regime,
                    "intraday_vol_proxy_mult": round(position.intraday_vol_proxy_mult, 4),
                    "intraday_vol_proxy_signal_age_min": (
                        round(position.intraday_vol_proxy_signal_age_min, 1)
                        if position.intraday_vol_proxy_signal_age_min is not None
                        else None
                    ),
                    "intraday_vol_proxy_close": (
                        round(position.intraday_vol_proxy_close, 4)
                        if position.intraday_vol_proxy_close is not None
                        else None
                    ),
                    "ext_hours_proxy_regime": position.ext_hours_proxy_regime,
                    "ext_hours_proxy_mult": round(position.ext_hours_proxy_mult, 4),
                    "ext_hours_proxy_source": position.ext_hours_proxy_source,
                    "ext_hours_proxy_score": (
                        round(position.ext_hours_proxy_score, 2)
                        if position.ext_hours_proxy_score is not None
                        else None
                    ),
                    "volatility_persistence_regime": position.volatility_persistence_regime,
                    "volatility_persistence_mult": round(position.volatility_persistence_mult, 4),
                    "volatility_persistence_ratio": (
                        round(position.volatility_persistence_ratio, 4)
                        if position.volatility_persistence_ratio is not None
                        else None
                    ),
                    "volatility_persistence_vix_rel": (
                        round(position.volatility_persistence_vix_rel, 4)
                        if position.volatility_persistence_vix_rel is not None
                        else None
                    ),
                    "volatility_persistence_vixy_rel": (
                        round(position.volatility_persistence_vixy_rel, 4)
                        if position.volatility_persistence_vixy_rel is not None
                        else None
                    ),
                    "volatility_persistence_signal_age_min": (
                        round(position.volatility_persistence_signal_age_min, 1)
                        if position.volatility_persistence_signal_age_min is not None
                        else None
                    ),
                    "scale": round(position.scale, 6),
                    "entry_rel_volume": round(position.entry_rel_volume, 4),
                    "risk_model": position.risk_model,
                    "performance_risk_mult": round(position.performance_risk_mult, 4),
                    "performance_score": (
                        round(position.performance_score, 6)
                        if position.performance_score is not None
                        else None
                    ),
                    "performance_rank_pct": (
                        round(position.performance_rank_pct, 4)
                        if position.performance_rank_pct is not None
                        else None
                    ),
                    "performance_peer_count": position.performance_peer_count,
                    "technical_ema_regime": position.technical_ema_regime,
                    "technical_ema_mult": round(position.technical_ema_mult, 4),
                    "technical_adx_value": (
                        round(position.technical_adx_value, 2)
                        if position.technical_adx_value is not None else None
                    ),
                    "technical_adx_mult": round(position.technical_adx_mult, 4),
                    "net_pnl": round(position.scaled_pnl, 4),
                    "equity_after_exit": round(capital, 4),
                }
            )

    candidate_idx = 0
    while candidate_idx < len(candidates):
        batch_entry_ts = candidates[candidate_idx]["entry_ts"]
        close_positions_up_to(batch_entry_ts)
        close_candidate_history_up_to(batch_entry_ts)

        batch_candidates: list[dict] = []
        while candidate_idx < len(candidates) and candidates[candidate_idx]["entry_ts"] == batch_entry_ts:
            candidate = candidates[candidate_idx]
            candidate_idx += 1
            if any(position.ticker == candidate["ticker"] for position in open_positions):
                skipped_same_ticker += 1
                continue
            batch_candidates.append(candidate)

        if not batch_candidates:
            continue

        sentiment_score, sentiment_reversal_recent_low = _lookup_sentiment_signal(
            entry_ts=batch_entry_ts,
            sentiment_scores=sentiment_scores if use_sentiment_governor else None,
            sentiment_lag_days=sentiment_lag_days,
            sentiment_reversal_window=sentiment_reversal_window,
        )

        active_exposure_mult, _ = _active_exposure_mult(
            base_exposure_mult=exposure_mult,
            capital=capital,
            peak_capital=peak_capital,
            use_drawdown_governor=use_drawdown_governor,
            drawdown_trigger_1_pct=drawdown_trigger_1_pct,
            drawdown_exposure_mult_1=drawdown_exposure_mult_1,
            drawdown_trigger_2_pct=drawdown_trigger_2_pct,
            drawdown_exposure_mult_2=drawdown_exposure_mult_2,
            use_sentiment_governor=use_sentiment_governor,
            sentiment_score=sentiment_score,
            sentiment_threshold_1=sentiment_threshold_1,
            sentiment_threshold_2=sentiment_threshold_2,
            sentiment_exposure_mult_1=sentiment_exposure_mult_1,
            sentiment_exposure_mult_2=sentiment_exposure_mult_2,
            sentiment_reversal_recent_low=sentiment_reversal_recent_low,
            sentiment_reversal_min_rise=sentiment_reversal_min_rise,
            sentiment_reversal_mult=sentiment_reversal_mult,
        )

        portfolio_cap = capital * base_portfolio_cap_pct * active_exposure_mult
        used_notional = sum(position.scaled_position_size for position in open_positions)
        available_portfolio_notional = max(portfolio_cap - used_notional, 0.0)
        if available_portfolio_notional <= 1e-9:
            skipped_no_capacity += len(batch_candidates)
            continue

        class_available_notional: dict[str, float | None] = {}
        for bucket_name, cap_mult in asset_class_caps.items():
            if cap_mult is None:
                class_available_notional[bucket_name] = None
                continue
            class_cap = capital * base_portfolio_cap_pct * cap_mult
            used_class_notional = sum(
                position.scaled_position_size
                for position in open_positions
                if position.asset_bucket == bucket_name
            )
            class_available_notional[bucket_name] = max(class_cap - used_class_notional, 0.0)

        request_records: list[dict] = []
        for candidate in batch_candidates:
            asset_bucket = str(candidate["asset_bucket"])
            performance_risk_mult = 1.0
            performance_score: float | None = None
            performance_rank_pct: float | None = None
            performance_peer_count = 0
            direct_sentiment_score: float | None = None
            direct_sentiment_size_mult = 1.0
            intraday_vol_proxy_regime: str | None = None
            intraday_vol_proxy_mult = 1.0
            intraday_vol_proxy_signal_age_min: float | None = None
            intraday_vol_proxy_close: float | None = None
            ext_hours_proxy_regime: str | None = None
            ext_hours_proxy_mult = 1.0
            ext_hours_proxy_source: str | None = None
            ext_hours_proxy_score: float | None = None
            volatility_persistence_regime: str | None = None
            volatility_persistence_mult = 1.0
            volatility_persistence_ratio: float | None = None
            volatility_persistence_vix_rel: float | None = None
            volatility_persistence_vixy_rel: float | None = None
            volatility_persistence_signal_age_min: float | None = None

            if use_performance_leadership_overlay:
                (
                    performance_risk_mult,
                    performance_score,
                    performance_rank_pct,
                    performance_peer_count,
                ) = _performance_leadership_scale(
                    ticker=str(candidate["ticker"]),
                    closed_trade_returns_by_ticker=closed_trade_returns_by_ticker,
                    lookback_trades=performance_lookback_trades,
                    decay=performance_decay,
                    floor_mult=performance_floor_mult,
                    cap_mult=performance_cap_mult,
                    min_history=performance_min_history,
                )

            bucket_scores = None
            if use_direct_bucket_sentiment_sizing and bucket_sentiment_scores:
                bucket_scores = bucket_sentiment_scores.get(asset_bucket)
            if bucket_scores:
                direct_sentiment_score, direct_sentiment_recent_low = _lookup_sentiment_signal(
                    entry_ts=batch_entry_ts,
                    sentiment_scores=bucket_scores,
                    sentiment_lag_days=bucket_sentiment_lag_days,
                    sentiment_reversal_window=bucket_sentiment_reversal_window,
                )
                if direct_sentiment_score is not None:
                    direct_sentiment_size_mult = _sentiment_regime_mult(
                        base_mult=1.0,
                        sentiment_score=direct_sentiment_score,
                        sentiment_threshold_1=bucket_sentiment_threshold_1,
                        sentiment_threshold_2=bucket_sentiment_threshold_2,
                        sentiment_mult_1=bucket_sentiment_size_mult_1,
                        sentiment_mult_2=bucket_sentiment_size_mult_2,
                        sentiment_reversal_recent_low=direct_sentiment_recent_low,
                        sentiment_reversal_min_rise=bucket_sentiment_reversal_min_rise,
                        sentiment_reversal_mult=bucket_sentiment_reversal_mult,
                    )

            if use_intraday_volatility_proxy:
                (
                    intraday_vol_proxy_regime,
                    intraday_vol_proxy_mult,
                    intraday_vol_proxy_signal_age_min,
                    intraday_vol_proxy_close,
                ) = _lookup_intraday_volatility_signal(
                    entry_ts=batch_entry_ts,
                    session_open=str(candidate["session_open"]),
                    asset_bucket=asset_bucket,
                    direction=str(candidate["direction"]),
                    proxy_state=intraday_volatility_proxy_state,
                    max_age_minutes=intraday_volatility_proxy_max_age_minutes,
                    lag_bars=intraday_volatility_proxy_lag_bars,
                    allowed_buckets=intraday_volatility_proxy_buckets,
                    long_risk_on_mult=intraday_volatility_long_risk_on_mult,
                    long_neutral_mult=intraday_volatility_long_neutral_mult,
                    long_risk_off_mult=intraday_volatility_long_risk_off_mult,
                    short_risk_on_mult=intraday_volatility_short_risk_on_mult,
                    short_neutral_mult=intraday_volatility_short_neutral_mult,
                    short_risk_off_mult=intraday_volatility_short_risk_off_mult,
                )

            if use_extended_hours_proxy:
                (
                    ext_hours_proxy_regime,
                    ext_hours_proxy_mult,
                    ext_hours_proxy_source,
                    ext_hours_proxy_score,
                ) = _lookup_extended_hours_signal(
                    entry_ts=batch_entry_ts,
                    session_open=str(candidate["session_open"]),
                    asset_bucket=asset_bucket,
                    direction=str(candidate["direction"]),
                    extended_hours_state=extended_hours_proxy_state,
                    lag_days=extended_hours_proxy_lag_days,
                    vix_risk_on_threshold=extended_hours_vix_risk_on_threshold,
                    vix_risk_off_threshold=extended_hours_vix_risk_off_threshold,
                    fg_greed_threshold=extended_hours_fg_greed_threshold,
                    fg_fear_threshold=extended_hours_fg_fear_threshold,
                    long_risk_on_mult=extended_hours_long_risk_on_mult,
                    long_neutral_mult=extended_hours_long_neutral_mult,
                    long_risk_off_mult=extended_hours_long_risk_off_mult,
                    short_risk_on_mult=extended_hours_short_risk_on_mult,
                    short_neutral_mult=extended_hours_short_neutral_mult,
                    short_risk_off_mult=extended_hours_short_risk_off_mult,
                )

            if use_volatility_persistence_overlay:
                (
                    volatility_persistence_regime,
                    volatility_persistence_mult,
                    volatility_persistence_signal_age_min,
                    volatility_persistence_vix_rel,
                    volatility_persistence_vixy_rel,
                    volatility_persistence_ratio,
                ) = _lookup_volatility_persistence_signal(
                    entry_ts=batch_entry_ts,
                    session_open=str(candidate["session_open"]),
                    direction=str(candidate["direction"]),
                    daily_vix_state=daily_vix_reference_state,
                    intraday_vixy_state=intraday_vixy_relative_state,
                    daily_lag_days=volatility_persistence_daily_lag_days,
                    intraday_max_age_minutes=volatility_persistence_intraday_max_age_minutes,
                    intraday_lag_bars=volatility_persistence_intraday_lag_bars,
                    ratio_upper=volatility_persistence_ratio_upper,
                    ratio_lower=volatility_persistence_ratio_lower,
                    daily_stress_min_rel=volatility_persistence_daily_stress_min_rel,
                    long_persistent_stress_mult=volatility_persistence_long_persistent_stress_mult,
                    long_neutral_mult=volatility_persistence_long_neutral_mult,
                    long_fading_stress_mult=volatility_persistence_long_fading_stress_mult,
                    short_persistent_stress_mult=volatility_persistence_short_persistent_stress_mult,
                    short_neutral_mult=volatility_persistence_short_neutral_mult,
                    short_fading_stress_mult=volatility_persistence_short_fading_stress_mult,
                )

            # ── per-asset technical overlay (daily EMA + 4H ADX) ─────────
            technical_ema_regime: str | None = None
            technical_ema_mult = 1.0
            technical_adx_value: float | None = None
            technical_adx_mult = 1.0
            if use_per_asset_technical_overlay and per_asset_technical_state:
                technical_ema_regime, technical_ema_mult, technical_adx_value, technical_adx_mult = (
                    _lookup_per_asset_technical_signal(
                        entry_ts=datetime.fromisoformat(str(candidate["entry_ts"])),
                        ticker=str(candidate["ticker"]),
                        direction=str(candidate["direction"]),
                        technical_state=per_asset_technical_state,
                        ema_lag_days=per_asset_ema_lag_days,
                        ema_above_long_mult=per_asset_ema_above_long_mult,
                        ema_above_short_mult=per_asset_ema_above_short_mult,
                        ema_below_long_mult=per_asset_ema_below_long_mult,
                        ema_below_short_mult=per_asset_ema_below_short_mult,
                        use_adx_gate=per_asset_use_adx_gate,
                        adx_strong_threshold=per_asset_adx_strong_threshold,
                        adx_weak_threshold=per_asset_adx_weak_threshold,
                        adx_weak_mult=per_asset_adx_weak_mult,
                    )
                )

            base_position_size = _candidate_requested_position_size(
                candidate=candidate,
                capital=capital,
                max_position_pct=base_portfolio_cap_pct,
            )
            target_position_size = (
                base_position_size
                * performance_risk_mult
                * direct_sentiment_size_mult
                * intraday_vol_proxy_mult
                * ext_hours_proxy_mult
                * volatility_persistence_mult
                * technical_ema_mult
                * technical_adx_mult
            )
            if target_position_size <= 1e-9:
                skipped_no_capacity += 1
                continue

            request_records.append(
                {
                    "candidate": candidate,
                    "asset_bucket": asset_bucket,
                    "target_position_size": target_position_size,
                    "performance_risk_mult": performance_risk_mult,
                    "performance_score": performance_score,
                    "performance_rank_pct": performance_rank_pct,
                    "performance_peer_count": performance_peer_count,
                    "direct_sentiment_score": direct_sentiment_score,
                    "direct_sentiment_size_mult": direct_sentiment_size_mult,
                    "intraday_vol_proxy_regime": intraday_vol_proxy_regime,
                    "intraday_vol_proxy_mult": intraday_vol_proxy_mult,
                    "intraday_vol_proxy_signal_age_min": intraday_vol_proxy_signal_age_min,
                    "intraday_vol_proxy_close": intraday_vol_proxy_close,
                    "ext_hours_proxy_regime": ext_hours_proxy_regime,
                    "ext_hours_proxy_mult": ext_hours_proxy_mult,
                    "ext_hours_proxy_source": ext_hours_proxy_source,
                    "ext_hours_proxy_score": ext_hours_proxy_score,
                    "volatility_persistence_regime": volatility_persistence_regime,
                    "volatility_persistence_mult": volatility_persistence_mult,
                    "volatility_persistence_ratio": volatility_persistence_ratio,
                    "volatility_persistence_vix_rel": volatility_persistence_vix_rel,
                    "volatility_persistence_vixy_rel": volatility_persistence_vixy_rel,
                    "volatility_persistence_signal_age_min": volatility_persistence_signal_age_min,
                    "technical_ema_regime": technical_ema_regime,
                    "technical_ema_mult": technical_ema_mult,
                    "technical_adx_value": technical_adx_value,
                    "technical_adx_mult": technical_adx_mult,
                    "entry_exposure_mult": active_exposure_mult,
                    "entry_sentiment_score": sentiment_score,
                    "stable_key": (
                        str(candidate["ticker"]),
                        str(candidate["source"]),
                        str(candidate["session_open"]),
                        str(candidate["direction"]),
                        int(candidate["trade_idx"]),
                    ),
                }
            )

        if not request_records:
            continue

        unique_by_ticker: dict[str, dict] = {}
        for record in request_records:
            ticker = str(record["candidate"]["ticker"])
            incumbent = unique_by_ticker.get(ticker)
            if incumbent is None:
                unique_by_ticker[ticker] = record
                continue
            better = record["target_position_size"] > incumbent["target_position_size"] + 1e-9 or (
                abs(record["target_position_size"] - incumbent["target_position_size"]) <= 1e-9
                and record["stable_key"] < incumbent["stable_key"]
            )
            if better:
                skipped_same_ticker += 1
                unique_by_ticker[ticker] = record
            else:
                skipped_same_ticker += 1
        batch_records = list(unique_by_ticker.values())
        if not batch_records:
            continue

        bucket_request_totals = Counter()
        for record in batch_records:
            bucket_request_totals[str(record["asset_bucket"])] += float(record["target_position_size"])

        bucket_scales: dict[str, float] = {}
        for bucket_name, requested_total in bucket_request_totals.items():
            available_bucket = class_available_notional.get(bucket_name)
            if available_bucket is None or requested_total <= 1e-9:
                bucket_scales[bucket_name] = 1.0
            else:
                bucket_scales[bucket_name] = min(1.0, max(float(available_bucket), 0.0) / requested_total)

        bucket_adjusted_total = sum(
            float(record["target_position_size"]) * bucket_scales.get(str(record["asset_bucket"]), 1.0)
            for record in batch_records
        )
        overall_scale = (
            min(1.0, available_portfolio_notional / bucket_adjusted_total)
            if bucket_adjusted_total > 1e-9
            else 0.0
        )

        for record in batch_records:
            candidate = record["candidate"]
            bucket_scale = bucket_scales.get(str(record["asset_bucket"]), 1.0)
            scaled_position_size = float(record["target_position_size"]) * bucket_scale * overall_scale
            if scaled_position_size <= 1e-9:
                skipped_no_capacity += 1
                continue

            candidate_position_size = float(candidate["position_size"])
            scale = scaled_position_size / candidate_position_size if candidate_position_size > 0 else 0.0
            open_positions.append(
                _OpenTrade(
                    combo_idx=int(candidate["combo_idx"]),
                    trade_idx=int(candidate["trade_idx"]),
                    ticker=str(candidate["ticker"]),
                    source=str(candidate["source"]),
                    session_open=str(candidate["session_open"]),
                    direction=str(candidate["direction"]),
                    entry_ts=candidate["entry_ts"],
                    exit_ts=candidate["exit_ts"],
                    entry_price=float(candidate["entry_price"]),
                    exit_price=float(candidate["exit_price"]),
                    shares=float(candidate["shares"]),
                    position_size=candidate_position_size,
                    pnl=float(candidate["pnl"]),
                    risk_model=str(candidate["risk_model"]),
                    entry_rel_volume=float(candidate["entry_rel_volume"]),
                    asset_bucket=str(record["asset_bucket"]),
                    entry_exposure_mult=float(record["entry_exposure_mult"]),
                    entry_sentiment_score=record["entry_sentiment_score"],
                    direct_sentiment_score=record["direct_sentiment_score"],
                    direct_sentiment_size_mult=float(record["direct_sentiment_size_mult"]),
                    intraday_vol_proxy_regime=record["intraday_vol_proxy_regime"],
                    intraday_vol_proxy_mult=float(record["intraday_vol_proxy_mult"]),
                    intraday_vol_proxy_signal_age_min=record["intraday_vol_proxy_signal_age_min"],
                    intraday_vol_proxy_close=record["intraday_vol_proxy_close"],
                    ext_hours_proxy_regime=record["ext_hours_proxy_regime"],
                    ext_hours_proxy_mult=float(record["ext_hours_proxy_mult"]),
                    ext_hours_proxy_source=record["ext_hours_proxy_source"],
                    ext_hours_proxy_score=record["ext_hours_proxy_score"],
                    volatility_persistence_regime=record["volatility_persistence_regime"],
                    volatility_persistence_mult=float(record["volatility_persistence_mult"]),
                    volatility_persistence_ratio=record["volatility_persistence_ratio"],
                    volatility_persistence_vix_rel=record["volatility_persistence_vix_rel"],
                    volatility_persistence_vixy_rel=record["volatility_persistence_vixy_rel"],
                    volatility_persistence_signal_age_min=record["volatility_persistence_signal_age_min"],
                    scale=scale,
                    scaled_position_size=scaled_position_size,
                    scaled_shares=float(candidate["shares"]) * scale,
                    scaled_pnl=float(candidate["pnl"]) * scale,
                    performance_risk_mult=float(record["performance_risk_mult"]),
                    performance_score=record["performance_score"],
                    performance_rank_pct=record["performance_rank_pct"],
                    performance_peer_count=int(record["performance_peer_count"]),
                    technical_ema_regime=record["technical_ema_regime"],
                    technical_ema_mult=float(record["technical_ema_mult"]),
                    technical_adx_value=record["technical_adx_value"],
                    technical_adx_mult=float(record["technical_adx_mult"]),
                )
            )

    close_positions_up_to(datetime.max)

    if executed_trades:
        start_date = executed_trades[0]["entry_ts"]
        end_date = executed_trades[-1]["exit_ts"]
        equity_curve.insert(0, {"date": start_date, "equity": round(initial_capital, 4)})
    else:
        start_date = None
        end_date = None

    total_pnl = capital - initial_capital
    years = (
        max((datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
        if start_date and end_date
        else 1 / 365.25
    )
    winning_trades = [trade for trade in executed_trades if float(trade["net_pnl"]) > 0]
    losing_trades = [trade for trade in executed_trades if float(trade["net_pnl"]) <= 0]
    gross_profit = sum(float(trade["net_pnl"]) for trade in winning_trades)
    gross_loss_abs = abs(sum(float(trade["net_pnl"]) for trade in losing_trades))
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    long_trades = [trade for trade in executed_trades if trade["direction"] == "long"]
    short_trades = [trade for trade in executed_trades if trade["direction"] == "short"]

    yearly_rows = _build_yearly_rows(executed_trades, initial_capital)
    asset_rows = _build_asset_rows(executed_trades, total_pnl)
    bucket_counter = Counter(trade["asset_bucket"] for trade in executed_trades)
    bucket_pnl = Counter()
    exposure_counter = Counter()
    for trade in executed_trades:
        bucket_pnl[trade["asset_bucket"]] += float(trade["net_pnl"])
        exposure_counter[float(trade["entry_exposure_mult"])] += 1
    performance_mults = [
        float(trade.get("performance_risk_mult")) if trade.get("performance_risk_mult") is not None else 1.0
        for trade in executed_trades
    ]
    direct_sentiment_mults = [
        float(trade.get("direct_sentiment_size_mult"))
        if trade.get("direct_sentiment_size_mult") is not None
        else 1.0
        for trade in executed_trades
    ]
    intraday_vol_proxy_mults = [
        float(trade.get("intraday_vol_proxy_mult")) if trade.get("intraday_vol_proxy_mult") is not None else 1.0
        for trade in executed_trades
    ]
    intraday_vol_proxy_regimes = Counter(
        str(trade.get("intraday_vol_proxy_regime"))
        for trade in executed_trades
        if trade.get("intraday_vol_proxy_regime")
    )
    ext_hours_proxy_mults = [
        float(trade.get("ext_hours_proxy_mult")) if trade.get("ext_hours_proxy_mult") is not None else 1.0
        for trade in executed_trades
    ]
    ext_hours_proxy_regimes = Counter(
        str(trade.get("ext_hours_proxy_regime"))
        for trade in executed_trades
        if trade.get("ext_hours_proxy_regime")
    )
    volatility_persistence_mults = [
        float(trade.get("volatility_persistence_mult"))
        if trade.get("volatility_persistence_mult") is not None
        else 1.0
        for trade in executed_trades
    ]
    volatility_persistence_regimes = Counter(
        str(trade.get("volatility_persistence_regime"))
        for trade in executed_trades
        if trade.get("volatility_persistence_regime")
    )

    label = f"Session Turtle Trend {basket.capitalize()} x{exposure_mult:g}"
    if asof_date is not None:
        label += f" Universe As Of {asof_date.isoformat()}"
    if use_drawdown_governor:
        label += " With DD Governor"
    if use_extended_hours_protective_exits:
        label += " With Extended Hours Protective Exits"
    if use_performance_leadership_overlay:
        label += " With Leadership Overlay"
    if use_sentiment_governor:
        label += " With Sentiment Governor"
    if use_direct_bucket_sentiment_sizing:
        label += " With Direct Bucket Sentiment Sizing"
    if use_intraday_volatility_proxy:
        label += f" With {intraday_volatility_proxy_label} Micro Overlay"
    if use_extended_hours_proxy:
        label += " With Extended Hours VIX/FG Proxy"
    if use_volatility_persistence_overlay:
        label += f" With {volatility_persistence_label}"
    if use_per_asset_technical_overlay:
        parts = ["daily EMA"]
        if per_asset_use_adx_gate:
            parts.append("4H ADX gate")
        label += f" With Per-Asset Technical ({', '.join(parts)})"
    if any(cap is not None for cap in asset_class_caps.values()):
        label += " With Asset Class Caps"

    summary = {
        "strategy_variant": "session_turtle_trend_shared_account",
        "label": label,
        "basket": basket,
        "investable_universe_asof": asof_date.isoformat() if asof_date is not None else None,
        "candidate_universe_size": len({(row["ticker"], row["source"], row["session_open"]) for row in candidates}),
        "start_date": start_date,
        "end_date": end_date,
        "candidate_trades": len(candidates),
        "executed_trades": len(executed_trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "skipped_same_ticker": skipped_same_ticker,
        "skipped_no_capacity": skipped_no_capacity,
        "initial_capital": round(initial_capital, 4),
        "final_equity": round(capital, 4),
        "total_return_pct": round((capital / initial_capital - 1.0) * 100.0, 2) if initial_capital > 0 else 0.0,
        "cagr_pct": round(((capital / initial_capital) ** (1 / years) - 1.0) * 100.0, 2) if initial_capital > 0 else 0.0,
        "max_realized_drawdown_pct": round(max_drawdown * 100.0, 2),
        "win_rate_pct": round((len(winning_trades) / len(executed_trades) * 100.0) if executed_trades else 0.0, 2),
        "profit_factor": round(profit_factor, 2),
        "exposure_mult": exposure_mult,
        "use_drawdown_governor": use_drawdown_governor,
        "drawdown_trigger_1_pct": round(drawdown_trigger_1_pct, 2),
        "drawdown_exposure_mult_1": round(drawdown_exposure_mult_1, 4),
        "drawdown_trigger_2_pct": round(drawdown_trigger_2_pct, 2),
        "drawdown_exposure_mult_2": round(drawdown_exposure_mult_2, 4),
        "crypto_cap_mult": crypto_cap_mult,
        "gold_cap_mult": gold_cap_mult,
        "metals_cap_mult": metals_cap_mult,
        "equity_cap_mult": equity_cap_mult,
        "entries_at_base_exposure": exposure_counter[float(exposure_mult)],
        "entries_at_drawdown_exposure_1": exposure_counter[float(drawdown_exposure_mult_1)],
        "entries_at_drawdown_exposure_2": exposure_counter[float(drawdown_exposure_mult_2)],
        "use_sentiment_governor": use_sentiment_governor,
        "sentiment_lag_days": sentiment_lag_days,
        "sentiment_threshold_1": sentiment_threshold_1,
        "sentiment_threshold_2": sentiment_threshold_2,
        "sentiment_exposure_mult_1": sentiment_exposure_mult_1,
        "sentiment_exposure_mult_2": sentiment_exposure_mult_2,
        "sentiment_reversal_window": sentiment_reversal_window,
        "sentiment_reversal_min_rise": sentiment_reversal_min_rise,
        "sentiment_reversal_mult": sentiment_reversal_mult,
        "use_direct_bucket_sentiment_sizing": use_direct_bucket_sentiment_sizing,
        "bucket_sentiment_lag_days": bucket_sentiment_lag_days,
        "bucket_sentiment_threshold_1": bucket_sentiment_threshold_1,
        "bucket_sentiment_threshold_2": bucket_sentiment_threshold_2,
        "bucket_sentiment_size_mult_1": bucket_sentiment_size_mult_1,
        "bucket_sentiment_size_mult_2": bucket_sentiment_size_mult_2,
        "bucket_sentiment_reversal_window": bucket_sentiment_reversal_window,
        "bucket_sentiment_reversal_min_rise": bucket_sentiment_reversal_min_rise,
        "bucket_sentiment_reversal_mult": bucket_sentiment_reversal_mult,
        "bucket_sentiment_bucket_count": len(bucket_sentiment_scores or {}),
        "avg_direct_sentiment_size_mult": (
            round(sum(direct_sentiment_mults) / len(direct_sentiment_mults), 4)
            if direct_sentiment_mults
            else 1.0
        ),
        "entries_direct_sentiment_downscaled": sum(1 for mult in direct_sentiment_mults if mult < 0.999999),
        "use_intraday_volatility_proxy": use_intraday_volatility_proxy,
        "intraday_volatility_proxy_label": intraday_volatility_proxy_label,
        "intraday_volatility_proxy_max_age_minutes": intraday_volatility_proxy_max_age_minutes,
        "intraday_volatility_proxy_lag_bars": intraday_volatility_proxy_lag_bars,
        "intraday_volatility_proxy_short_ma_bars": intraday_volatility_proxy_short_ma_bars,
        "intraday_volatility_proxy_long_ma_bars": intraday_volatility_proxy_long_ma_bars,
        "intraday_volatility_long_risk_on_mult": intraday_volatility_long_risk_on_mult,
        "intraday_volatility_long_neutral_mult": intraday_volatility_long_neutral_mult,
        "intraday_volatility_long_risk_off_mult": intraday_volatility_long_risk_off_mult,
        "intraday_volatility_short_risk_on_mult": intraday_volatility_short_risk_on_mult,
        "intraday_volatility_short_neutral_mult": intraday_volatility_short_neutral_mult,
        "intraday_volatility_short_risk_off_mult": intraday_volatility_short_risk_off_mult,
        "avg_intraday_volatility_proxy_mult": (
            round(sum(intraday_vol_proxy_mults) / len(intraday_vol_proxy_mults), 4)
            if intraday_vol_proxy_mults
            else 1.0
        ),
        "entries_intraday_volatility_proxy_scaled": sum(
            1 for mult in intraday_vol_proxy_mults if mult < 0.999999
        ),
        "entries_intraday_volatility_risk_on_micro": intraday_vol_proxy_regimes["risk_on_micro"],
        "entries_intraday_volatility_neutral_micro": intraday_vol_proxy_regimes["neutral_micro"],
        "entries_intraday_volatility_risk_off_micro": intraday_vol_proxy_regimes["risk_off_micro"],
        "use_extended_hours_proxy": use_extended_hours_proxy,
        "extended_hours_proxy_lag_days": extended_hours_proxy_lag_days,
        "extended_hours_vix_risk_on_threshold": extended_hours_vix_risk_on_threshold,
        "extended_hours_vix_risk_off_threshold": extended_hours_vix_risk_off_threshold,
        "extended_hours_fg_greed_threshold": extended_hours_fg_greed_threshold,
        "extended_hours_fg_fear_threshold": extended_hours_fg_fear_threshold,
        "extended_hours_long_risk_on_mult": extended_hours_long_risk_on_mult,
        "extended_hours_long_neutral_mult": extended_hours_long_neutral_mult,
        "extended_hours_long_risk_off_mult": extended_hours_long_risk_off_mult,
        "extended_hours_short_risk_on_mult": extended_hours_short_risk_on_mult,
        "extended_hours_short_neutral_mult": extended_hours_short_neutral_mult,
        "extended_hours_short_risk_off_mult": extended_hours_short_risk_off_mult,
        "avg_ext_hours_proxy_mult": (
            round(sum(ext_hours_proxy_mults) / len(ext_hours_proxy_mults), 4)
            if ext_hours_proxy_mults
            else 1.0
        ),
        "entries_ext_hours_proxy_scaled": sum(1 for mult in ext_hours_proxy_mults if mult < 0.999999),
        "entries_ext_hours_risk_on_micro": ext_hours_proxy_regimes["risk_on_micro"],
        "entries_ext_hours_neutral_micro": ext_hours_proxy_regimes["neutral_micro"],
        "entries_ext_hours_risk_off_micro": ext_hours_proxy_regimes["risk_off_micro"],
        "use_volatility_persistence_overlay": use_volatility_persistence_overlay,
        "volatility_persistence_label": volatility_persistence_label,
        "volatility_persistence_daily_lag_days": volatility_persistence_daily_lag_days,
        "volatility_persistence_intraday_max_age_minutes": volatility_persistence_intraday_max_age_minutes,
        "volatility_persistence_intraday_lag_bars": volatility_persistence_intraday_lag_bars,
        "volatility_persistence_daily_ema_period": volatility_persistence_daily_ema_period,
        "volatility_persistence_intraday_ema_period": volatility_persistence_intraday_ema_period,
        "volatility_persistence_ratio_upper": volatility_persistence_ratio_upper,
        "volatility_persistence_ratio_lower": volatility_persistence_ratio_lower,
        "volatility_persistence_daily_stress_min_rel": volatility_persistence_daily_stress_min_rel,
        "volatility_persistence_long_persistent_stress_mult": volatility_persistence_long_persistent_stress_mult,
        "volatility_persistence_long_neutral_mult": volatility_persistence_long_neutral_mult,
        "volatility_persistence_long_fading_stress_mult": volatility_persistence_long_fading_stress_mult,
        "volatility_persistence_short_persistent_stress_mult": volatility_persistence_short_persistent_stress_mult,
        "volatility_persistence_short_neutral_mult": volatility_persistence_short_neutral_mult,
        "volatility_persistence_short_fading_stress_mult": volatility_persistence_short_fading_stress_mult,
        "avg_volatility_persistence_mult": (
            round(sum(volatility_persistence_mults) / len(volatility_persistence_mults), 4)
            if volatility_persistence_mults
            else 1.0
        ),
        "entries_volatility_persistence_scaled": sum(
            1 for mult in volatility_persistence_mults if mult < 0.999999
        ),
        "entries_volatility_persistence_persistent_stress": volatility_persistence_regimes["persistent_stress"],
        "entries_volatility_persistence_neutral": volatility_persistence_regimes["neutral_persistence"],
        "entries_volatility_persistence_fading_stress": volatility_persistence_regimes["fading_stress"],
        "channel_period": channel_period,
        "lookback_years": lookback_years,
        "base_risk_pct": base_risk_pct,
        "directional_volume_risk_pct": directional_volume_risk_pct,
        "use_extended_hours_protective_exits": use_extended_hours_protective_exits,
        "extended_hours_core_session_minutes": extended_hours_core_session_minutes,
        "use_performance_leadership_overlay": use_performance_leadership_overlay,
        "performance_lookback_trades": performance_lookback_trades,
        "performance_decay": performance_decay,
        "performance_floor_mult": performance_floor_mult,
        "performance_cap_mult": performance_cap_mult,
        "performance_min_history": performance_min_history,
        "avg_performance_risk_mult": (
            round(sum(performance_mults) / len(performance_mults), 4) if performance_mults else 1.0
        ),
        "entries_performance_upscaled": sum(1 for mult in performance_mults if mult > 1.000001),
        "entries_performance_downscaled": sum(1 for mult in performance_mults if mult < 0.999999),
        "trend_fast_period": trend_fast_period,
        "trend_slow_period": trend_slow_period,
        "crypto_trades": bucket_counter["crypto"],
        "gold_trades": bucket_counter["gold"],
        "equity_trades": bucket_counter["equity"],
        "metals_trades": bucket_counter["metals"],
        "crypto_pnl": round(bucket_pnl["crypto"], 4),
        "gold_pnl": round(bucket_pnl["gold"], 4),
        "equity_pnl": round(bucket_pnl["equity"], 4),
        "metals_pnl": round(bucket_pnl["metals"], 4),
        "use_per_asset_technical_overlay": use_per_asset_technical_overlay,
        "per_asset_ema_period": per_asset_technical_state.get("ema_period") if per_asset_technical_state else None,
        "per_asset_adx_period": per_asset_technical_state.get("adx_period") if per_asset_technical_state else None,
        "per_asset_use_adx_gate": per_asset_use_adx_gate,
        "entries_above_ema": sum(1 for t in executed_trades if t.get("technical_ema_regime") == "above_ema"),
        "entries_below_ema": sum(1 for t in executed_trades if t.get("technical_ema_regime") == "below_ema"),
        "entries_adx_scaled_down": sum(1 for t in executed_trades if (t.get("technical_adx_mult") or 1.0) < 0.999),
        "avg_technical_ema_mult": round(
            sum(float(t.get("technical_ema_mult") or 1.0) for t in executed_trades) / len(executed_trades), 4
        ) if executed_trades else 1.0,
    }

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": executed_trades,
        "yearly_returns": yearly_rows,
        "asset_summary": asset_rows,
    }
