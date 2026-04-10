#!/usr/bin/env python3
"""Download daily QQQ history into the local cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_OUTPUT = os.path.join(CACHE_DIR, "cache", "QQQ_daily.parquet")
DEFAULT_START = "1999-03-10"
QQQ_TICKER = "QQQ"
YAHOO_CHART_URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{QQQ_TICKER}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; qqq-daily-data-downloader/1.0)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Inclusive end date, YYYY-MM-DD",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output parquet path")
    return parser.parse_args()


def checked_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def request_text(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = 6,
    timeout: int = 45,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, headers=HEADERS, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_seconds = min(2.0 * (attempt + 1), 20.0)
            print(f"  Retry {attempt + 1}/{retries - 1} after {exc}; sleeping {sleep_seconds:.0f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def _event_map(
    events: dict[str, Any],
    event_key: str,
    value_builder: Callable[[dict[str, Any]], float],
) -> dict[int, float]:
    output: dict[int, float] = {}
    for item in (events.get(event_key) or {}).values():
        try:
            timestamp = int(item["date"])
            output[timestamp] = float(value_builder(item))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    return output


def fetch_qqq(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    yahoo_end = end_date + timedelta(days=1)
    end_dt = datetime(yahoo_end.year, yahoo_end.month, yahoo_end.day, tzinfo=timezone.utc)
    text = request_text(
        session,
        YAHOO_CHART_URL,
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "div,splits,capitalGains",
            "includeAdjustedClose": "true",
        },
    )
    payload = json.loads(text)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {QQQ_TICKER}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo chart returned no data for {QQQ_TICKER}")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    events = result.get("events") or {}
    dividends = _event_map(events, "dividends", lambda item: float(item.get("amount", 0.0)))
    stock_splits = _event_map(
        events,
        "splits",
        lambda item: float(item.get("numerator", 0.0)) / float(item.get("denominator", 1.0)),
    )
    capital_gains = _event_map(events, "capitalGains", lambda item: float(item.get("amount", 0.0)))

    df = pd.DataFrame({"timestamp": timestamps})
    for column, output in [
        ("open", "o"),
        ("high", "h"),
        ("low", "l"),
        ("close", "c"),
        ("volume", "v"),
    ]:
        values = quote.get(column)
        df[output] = values if values is not None else pd.NA
    df["adj_c"] = adjclose if adjclose is not None else pd.NA
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["date"] = df["time"].dt.date
    df["ts"] = df["timestamp"].astype("int64") * 1000
    df["dividends"] = df["timestamp"].map(dividends).fillna(0.0)
    df["stock_splits"] = df["timestamp"].map(stock_splits).fillna(0.0)
    df["capital_gains"] = df["timestamp"].map(capital_gains).fillna(0.0)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    for column in ["o", "h", "l", "c", "adj_c", "dividends", "stock_splits", "capital_gains"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0.0).astype("int64")
    df = df.dropna(subset=["c"]).reset_index(drop=True)
    return df[["date", "time", "ts", "o", "h", "l", "c", "adj_c", "v", "dividends", "stock_splits", "capital_gains"]]


def main() -> None:
    args = parse_args()
    start_date = checked_date(args.start)
    end_date = checked_date(args.end)
    if end_date < start_date:
        raise SystemExit(f"--end {end_date} is before --start {start_date}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print(f"Downloading {QQQ_TICKER} daily history: {start_date} -> {end_date}")
    qqq = fetch_qqq(session, start_date, end_date)
    qqq.to_parquet(output_path, index=False)

    metadata = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": QQQ_TICKER,
        "source": "Yahoo Finance chart API",
        "source_url": YAHOO_CHART_URL,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "output_parquet": str(output_path),
        "rows": int(len(qqq)),
        "first_date": qqq["date"].min().isoformat() if not qqq.empty else None,
        "last_date": qqq["date"].max().isoformat() if not qqq.empty else None,
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if qqq.empty:
        raise RuntimeError("Yahoo returned no usable QQQ rows.")
    print(f"  Saved {len(qqq):,} rows to {output_path}")
    print(f"  Range: {qqq['date'].min()} -> {qqq['date'].max()}")


if __name__ == "__main__":
    main()
