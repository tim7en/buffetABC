"""Local Tiingo parquet cache loader for real intraday OHLCV."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    try:
        from django.conf import settings

        return Path(settings.BASE_DIR)
    except Exception:  # pragma: no cover
        return Path(__file__).resolve().parents[2]


def _tiingo_root() -> Path:
    return _project_root() / "cache" / "cache" / "tiingo"


def available_tiingo_symbols() -> set[str]:
    root = _tiingo_root()
    symbols: set[str] = set()
    for path in root.glob("*_5m.parquet"):
        stem = path.stem
        if stem.startswith("_"):
            continue
        if stem.endswith("_5m"):
            symbols.add(stem[:-3])
    return symbols


def resolve_tiingo_symbol(ticker: str, explicit_symbol: str | None = None) -> str:
    if explicit_symbol:
        return explicit_symbol.strip().upper()

    text = (ticker or "").strip().upper()
    if not text:
        raise ValueError("ticker required for Tiingo symbol resolution")
    if text.endswith("-USD") and len(text) > 4:
        return text[:-4]
    return text


def _locate_tiingo_file(symbol: str) -> Path:
    path = _tiingo_root() / f"{symbol}_5m.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No local Tiingo parquet cache found for {symbol}. Expected it under cache/cache/tiingo/."
        )
    return path


def _bucket_start(ts: datetime, interval_minutes: int) -> datetime:
    return ts.replace(
        minute=(ts.minute // interval_minutes) * interval_minutes,
        second=0,
        microsecond=0,
    )


def _resample_bars(base_bars: list[dict[str, Any]], interval_minutes: int) -> list[dict[str, Any]]:
    if interval_minutes == 5:
        return base_bars
    if interval_minutes % 5 != 0:
        return []

    aggregated: list[dict[str, Any]] = []
    current_bucket: datetime | None = None
    current_bar: dict[str, Any] | None = None
    for bar in base_bars:
        bucket = _bucket_start(bar["timestamp"], interval_minutes)
        if bucket != current_bucket:
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
        else:
            current_bar["high"] = max(float(current_bar["high"]), float(bar["high"]))
            current_bar["low"] = min(float(current_bar["low"]), float(bar["low"]))
            current_bar["close"] = float(bar["close"])
            current_bar["volume"] = float(current_bar["volume"]) + float(bar.get("volume", 0.0) or 0.0)
    if current_bar is not None:
        aggregated.append(current_bar)
    return aggregated


def _interval_minutes(interval: str) -> int:
    text = (interval or "").strip().lower()
    if text.endswith("m") and text[:-1].isdigit():
        return int(text[:-1])
    if text.endswith("h") and text[:-1].isdigit():
        return int(text[:-1]) * 60
    raise ValueError("Unsupported Tiingo interval")


def load_local_tiingo_klines(
    ticker: str,
    interval: str,
    lookback_years: float,
    warmup_days: int,
    market_data_symbol: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    symbol = resolve_tiingo_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
    path = _locate_tiingo_file(symbol)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the local Tiingo parquet cache") from exc

    table = pq.read_table(path, columns=["time", "o", "h", "l", "c", "v"])
    cols = {name: table[name].to_pylist() for name in table.column_names}
    base_bars: list[dict[str, Any]] = []
    for idx in range(len(cols["time"])):
        try:
            ts = cols["time"][idx]
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            base_bars.append(
                {
                    "timestamp": ts,
                    "open": float(cols["o"][idx]),
                    "high": float(cols["h"][idx]),
                    "low": float(cols["l"][idx]),
                    "close": float(cols["c"][idx]),
                    "volume": float(cols["v"][idx] or 0.0),
                }
            )
        except Exception:
            continue

    if not base_bars:
        return [], symbol, str(path)

    interval_minutes = _interval_minutes(interval)
    bars = _resample_bars(base_bars, interval_minutes=interval_minutes)
    if not bars:
        return [], symbol, str(path)

    lookback_days = max(int(365.25 * lookback_years), 1)
    total_days = lookback_days + max(int(warmup_days), 1)
    cutoff = bars[-1]["timestamp"] - timedelta(days=total_days)
    return ([bar for bar in bars if bar["timestamp"] >= cutoff], symbol, str(path))


def get_local_tiingo_time_bounds(
    ticker: str,
    market_data_symbol: str | None = None,
) -> tuple[datetime | None, datetime | None, str, str]:
    symbol = resolve_tiingo_symbol(ticker=ticker, explicit_symbol=market_data_symbol)
    path = _locate_tiingo_file(symbol)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the local Tiingo parquet cache") from exc

    table = pq.read_table(path, columns=["time"])
    values = table["time"].to_pylist()
    if not values:
        return None, None, symbol, str(path)

    def _normalize(ts: datetime) -> datetime:
        if getattr(ts, "tzinfo", None) is not None:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    return _normalize(values[0]), _normalize(values[-1]), symbol, str(path)
