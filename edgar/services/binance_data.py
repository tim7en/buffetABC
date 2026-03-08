"""Binance spot kline data fetch utilities (public, no API key required)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any

import requests


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
) -> tuple[list[dict[str, Any]], str]:
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

    return bars, symbol
