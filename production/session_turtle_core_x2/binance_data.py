"""Binance spot kline data fetch utilities (public, no API key required)."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import gzip
import math
from pathlib import Path
import time
from typing import Any

_BINANCE_BASE_URL = "https://api.binance.com"
_VALID_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
}
_INTERVAL_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
}


def _interval_to_minutes(interval: str) -> int:
    text = (interval or "").strip().lower()
    minutes = _INTERVAL_MINUTES.get(text)
    if minutes is None:
        raise ValueError(
            f"Unsupported Binance interval '{interval}'. Supported: {sorted(_INTERVAL_MINUTES.keys())}"
        )
    return minutes


def _cache_timeout_sec(interval_minutes: int) -> int:
    if interval_minutes <= 5:
        return 15 * 60
    if interval_minutes <= 15:
        return 30 * 60
    if interval_minutes <= 60:
        return 60 * 60
    return 2 * 60 * 60


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_cache_roots() -> list[Path]:
    root = _project_root()
    return [
        root / "cache" / "binance_asia_orb",
        root / "cache" / "cache" / "cache" / "binance_asia_orb",
    ]


def _locate_local_cache_file(symbol: str) -> Path | None:
    best_match: Path | None = None
    for base in _local_cache_roots():
        matches = sorted(base.glob(f"{symbol}_*_5m.csv.gz"))
        if not matches:
            continue
        candidate = max(matches, key=lambda path: path.stat().st_size)
        if best_match is None or candidate.stat().st_size > best_match.stat().st_size:
            best_match = candidate
    return best_match


def _load_local_cache_5m_bars(path: Path) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                open_time = int(row["open_time"])
                bars.append(
                    {
                        "timestamp": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
                        .astimezone(timezone.utc)
                        .replace(tzinfo=None),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue
    return bars


def _bucket_start(ts: datetime, interval_minutes: int) -> datetime:
    return ts.replace(
        minute=(ts.minute // interval_minutes) * interval_minutes,
        second=0,
        microsecond=0,
    )


def _resample_local_cache_bars(
    base_bars: list[dict[str, Any]],
    interval_key: str,
) -> list[dict[str, Any]]:
    if interval_key == "5m":
        return base_bars

    interval_minutes = _interval_to_minutes(interval_key)
    if interval_minutes % 5 != 0:
        return []

    aggregated: list[dict[str, Any]] = []
    current_bucket: datetime | None = None
    current_bar: dict[str, Any] | None = None

    for bar in base_bars:
        bucket = _bucket_start(bar["timestamp"], interval_minutes)
        if current_bucket != bucket:
            if current_bar is not None:
                aggregated.append(current_bar)
            current_bucket = bucket
            current_bar = {
                "timestamp": bucket,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", 0.0) or 0.0),
            }
            continue

        current_bar["high"] = max(float(current_bar["high"]), float(bar["high"]))
        current_bar["low"] = min(float(current_bar["low"]), float(bar["low"]))
        current_bar["close"] = float(bar["close"])
        current_bar["volume"] = float(current_bar["volume"]) + float(bar.get("volume", 0.0) or 0.0)

    if current_bar is not None:
        aggregated.append(current_bar)

    return aggregated


def _load_local_cache_payload(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_symbol: str | None = None,
) -> tuple[list[dict[str, Any]], str] | None:
    interval_key = (interval or "").strip().lower()
    if interval_key not in {"5m", "15m", "30m", "60m"}:
        return None

    symbol = resolve_binance_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
    cache_file = _locate_local_cache_file(symbol)
    if cache_file is None:
        return None

    base_bars = _load_local_cache_5m_bars(cache_file)
    if not base_bars:
        return None

    bars = _resample_local_cache_bars(base_bars=base_bars, interval_key=interval_key)
    if not bars:
        return None

    lookback_days = max(int(365.25 * lookback_years), 1)
    total_days = lookback_days + max(int(warmup_days), 1)
    cutoff = bars[-1]["timestamp"] - timedelta(days=total_days)
    return ([bar for bar in bars if bar["timestamp"] >= cutoff], symbol)


def load_local_binance_klines(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_symbol: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    payload = _load_local_cache_payload(
        ticker=ticker,
        interval=interval,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
        market_data_symbol=market_data_symbol,
    )
    if payload is None or not payload[0]:
        symbol = resolve_binance_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
        raise FileNotFoundError(
            f"No local Binance cache is available for {symbol} interval={interval}. "
            "Expected it under cache/binance_asia_orb/."
        )
    return payload


def resolve_binance_symbol(ticker: str, explicit_symbol: str | None = None) -> str:
    if explicit_symbol:
        return explicit_symbol.strip().upper()

    text = (ticker or "").strip().upper()
    if not text:
        raise ValueError("ticker required for Binance symbol resolution")

    direct_map = {
        "BTC-USD": "BTCUSDT",
        "BTCUSD": "BTCUSDT",
        "BTCUSDT": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "ETHUSD": "ETHUSDT",
        "ETHUSDT": "ETHUSDT",
        "PAXG-USD": "PAXGUSDT",
        "PAXGUSD": "PAXGUSDT",
        "PAXGUSDT": "PAXGUSDT",
    }
    if text in direct_map:
        return direct_map[text]

    # Generic fallback: XXX-USD -> XXXUSDT
    if text.endswith("-USD") and len(text) > 4:
        base = text[:-4]
        return f"{base}USDT"
    if text.endswith("USD") and len(text) > 3:
        base = text[:-3]
        return f"{base}USDT"

    return text


def fetch_binance_klines(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_symbol: str | None = None,
    request_timeout_sec: int = 20,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from django.core.cache import cache
    except Exception:  # pragma: no cover - fallback for non-Django execution
        cache = None

    interval_key = (interval or "").strip().lower()
    api_interval = _VALID_INTERVALS.get(interval_key)
    if api_interval is None:
        raise ValueError(
            f"Binance source supports intervals {sorted(_VALID_INTERVALS.keys())}. Received: {interval}"
        )

    symbol = resolve_binance_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
    interval_minutes = _interval_to_minutes(interval_key)
    step_ms = interval_minutes * 60_000

    lookback_days = max(int(365.25 * lookback_years), 1)
    total_days = lookback_days + max(int(warmup_days), 1)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=total_days)
    end_ms = int(end_dt.timestamp() * 1000)
    cursor_ms = int(start_dt.timestamp() * 1000)
    end_bucket = math.floor(end_dt.timestamp() / max(interval_minutes * 60, 1))
    cache_key = (
        f"binance:klines:v1:{symbol}:{interval_key}:"
        f"{lookback_days}:{int(warmup_days)}:{end_bucket}"
    )
    local_cache_key = f"{cache_key}:local"
    if use_cache and cache is not None:
        cached_payload = cache.get(local_cache_key)
        if cached_payload is not None:
            return cached_payload

    local_payload = _load_local_cache_payload(
        ticker=ticker,
        interval=interval_key,
        lookback_years=lookback_years,
        warmup_days=warmup_days,
        market_data_symbol=market_data_symbol,
    )
    if local_payload is not None and local_payload[0]:
        if use_cache and cache is not None:
            cache.set(local_cache_key, local_payload, timeout=_cache_timeout_sec(interval_minutes))
        return local_payload

    if use_cache and cache is not None:
        cached_payload = cache.get(cache_key)
        if cached_payload is not None:
            return cached_payload

    import requests

    rows: list[list[Any]] = []
    session = requests.Session()

    while cursor_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": api_interval,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = session.get(
            f"{_BINANCE_BASE_URL}/api/v3/klines",
            params=params,
            timeout=request_timeout_sec,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Binance kline request failed ({resp.status_code}) for symbol={symbol}: {resp.text[:240]}"
            )
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Binance response for symbol={symbol}: {data}")
        if not data:
            break

        rows.extend(data)
        last_open_time = int(data[-1][0])
        next_cursor = last_open_time + step_ms
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor

        # Public endpoint: keep low request cadence to avoid temporary 429 bans.
        time.sleep(0.03)

    if not rows:
        return [], symbol

    # Deduplicate by candle open timestamp.
    unique: dict[int, list[Any]] = {}
    for row in rows:
        try:
            unique[int(row[0])] = row
        except Exception:
            continue

    bars: list[dict[str, Any]] = []
    for open_time in sorted(unique.keys()):
        row = unique[open_time]
        try:
            bars.append(
                {
                    "timestamp": datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
                    .astimezone(timezone.utc)
                    .replace(tzinfo=None),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        except Exception:
            continue

    payload = (bars, symbol)
    if use_cache and cache is not None and bars:
        cache.set(cache_key, payload, timeout=_cache_timeout_sec(interval_minutes))
    return payload
