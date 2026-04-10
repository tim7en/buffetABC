#!/usr/bin/env python3
"""Download daily macro market data into the local cache.

Series:
- DXY: Yahoo Finance chart data for DX-Y.NYB (ICE U.S. Dollar Index)
- Gold: Yahoo Finance chart data for GC=F (COMEX front-month gold futures)
- Wilshire total-market proxy: Yahoo Finance chart data for ^FTW5000 or ^W5000
- US2Y/US10Y/US30Y: FRED Treasury constant maturity yields
- Nominal GDP: FRED quarterly U.S. gross domestic product, current dollars
- WTI: FRED WTI Cushing crude oil spot price
- CPI: FRED CPIAUCSL with derived month-over-month and year-over-year inflation
- UNRATE: FRED civilian unemployment rate
- Market Cap to GDP: FRED World Bank annual listed domestic companies market cap as % of GDP
- Shiller CAPE: Multpl monthly Shiller PE / CAPE table
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_START = "1999-01-01"

FRED_SERIES = {
    "US2Y": {
        "series_id": "DGS2",
        "name": "2-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_2y_yield",
    },
    "US10Y": {
        "series_id": "DGS10",
        "name": "10-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_10y_yield",
    },
    "US30Y": {
        "series_id": "DGS30",
        "name": "30-Year Treasury Constant Maturity Rate",
        "units": "percent",
        "combined_col": "us_30y_yield",
    },
    "WTI": {
        "series_id": "DCOILWTICO",
        "name": "WTI Crude Oil Spot Price, Cushing, OK",
        "units": "USD per barrel",
        "combined_col": "wti_usd_per_bbl",
    },
    "NOMINAL_GDP": {
        "series_id": "GDP",
        "name": "Gross Domestic Product",
        "units": "billions of current dollars, seasonally adjusted annual rate",
        "combined_col": "us_nominal_gdp_saar_bil",
    },
    "CPI": {
        "series_id": "CPIAUCSL",
        "name": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "units": "index 1982-1984=100, seasonally adjusted",
        "combined_col": "cpi_all_items_index",
        "derived": "cpi_inflation",
        "derived_combined_cols": {
            "cpi_mom_pct": "cpi_mom_pct",
            "cpi_yoy_pct": "cpi_yoy_pct",
        },
    },
    "UNRATE": {
        "series_id": "UNRATE",
        "name": "Unemployment Rate",
        "units": "percent, seasonally adjusted",
        "combined_col": "unemployment_rate_pct",
    },
    "MARKET_CAP_TO_GDP": {
        "series_id": "DDDM01USA156NWDB",
        "name": "Market Capitalization of Listed Domestic Companies for United States",
        "units": "percent of GDP",
        "combined_col": "market_cap_to_gdp_pct",
    },
}

DXY_TICKER = "DX-Y.NYB"
DXY_LABEL = "DXY"
GOLD_TICKER = "GC=F"
GOLD_LABEL = "GOLD"
WILSHIRE_TICKERS = ["^FTW5000", "^W5000"]
WILSHIRE_LABEL = "WILSHIRE_TOTAL_MARKET"

# --- Breadth, cross-asset, and vol-structure tickers (added for v2 features) ---
XLU_TICKER = "XLU"
XLU_LABEL = "XLU"
XLY_TICKER = "XLY"
XLY_LABEL = "XLY"
EEM_TICKER = "EEM"
EEM_LABEL = "EEM"
EFA_TICKER = "EFA"
EFA_LABEL = "EFA"
COPPER_TICKER = "HG=F"
COPPER_LABEL = "COPPER"
VIX3M_TICKER = "^VIX3M"
VIX3M_LABEL = "VIX3M"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
MULTPL_SHILLER_CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; daily-macro-data-downloader/1.0)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Inclusive end date, YYYY-MM-DD",
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Output cache directory")
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
        except Exception as exc:  # requests raises several transport-specific exception types.
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_seconds = min(2.0 * (attempt + 1), 20.0)
            print(f"  Retry {attempt + 1}/{retries - 1} after {exc}; sleeping {sleep_seconds:.0f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def fetch_fred_series(
    session: requests.Session,
    label: str,
    config: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    series_id = config["series_id"]
    text = request_text(session, FRED_CSV_URL, {"id": series_id})
    df = pd.read_csv(StringIO(text))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED CSV columns for {series_id}: {list(df.columns)}")

    df = df.rename(columns={"observation_date": "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    if config.get("derived") == "cpi_inflation":
        df["cpi_mom_pct"] = df["value"].pct_change(periods=1, fill_method=None) * 100.0
        df["cpi_yoy_pct"] = df["value"].pct_change(periods=12, fill_method=None) * 100.0
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df["series_id"] = series_id
    df["label"] = label
    df["name"] = config["name"]
    df["units"] = config["units"]
    output_cols = ["date", "value"]
    output_cols.extend((config.get("derived_combined_cols") or {}).keys())
    output_cols.extend(["series_id", "label", "name", "units"])
    return df[output_cols].reset_index(drop=True)


def fetch_yahoo_chart(
    session: requests.Session,
    ticker: str,
    label: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    # Yahoo's period2 is exclusive, so request the day after the inclusive end date.
    yahoo_end = end_date + timedelta(days=1)
    end_dt = datetime(yahoo_end.year, yahoo_end.month, yahoo_end.day, tzinfo=timezone.utc)
    text = request_text(
        session,
        YAHOO_CHART_URL.format(ticker=quote(ticker, safe="")),
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
    )
    payload = json.loads(text)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {ticker}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo chart returned no data for {ticker}")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_data = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    df = pd.DataFrame({"timestamp": timestamps})
    for col in ["open", "high", "low", "close", "volume"]:
        values = quote_data.get(col)
        df[col] = values if values is not None else pd.NA
    df["adj_close"] = adjclose if adjclose is not None else pd.NA
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    df["ticker"] = ticker
    df["label"] = label
    df["source"] = "Yahoo Finance chart API"
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    return df[
        ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker", "label", "source"]
    ].reset_index(drop=True)


def fetch_dxy(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    return fetch_yahoo_chart(session, DXY_TICKER, DXY_LABEL, start_date, end_date)


def fetch_gold(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    return fetch_yahoo_chart(session, GOLD_TICKER, GOLD_LABEL, start_date, end_date)


def fetch_wilshire_total_market(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    last_error: Exception | None = None
    for ticker in WILSHIRE_TICKERS:
        try:
            df = fetch_yahoo_chart(session, ticker, WILSHIRE_LABEL, start_date, end_date)
            df["ticker"] = ticker
            return df
        except Exception as exc:
            last_error = exc
            print(f"  Wilshire fallback `{ticker}` failed: {exc}")
    raise RuntimeError(f"Unable to download Wilshire total-market proxy from Yahoo: {last_error}") from last_error


def fetch_shiller_cape(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    html = request_text(session, MULTPL_SHILLER_CAPE_URL, {})
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("Multpl CAPE page returned no HTML tables.")
    table = tables[0].copy()
    if "Date" not in table.columns or "Value" not in table.columns:
        raise RuntimeError(f"Unexpected Multpl CAPE table columns: {list(table.columns)}")

    table = table.rename(columns={"Date": "date", "Value": "value"})
    table["date"] = pd.to_datetime(table["date"], errors="coerce").dt.date
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    table = table[(table["date"] >= start_date) & (table["date"] <= end_date)].copy()
    table["series_id"] = "SHILLER_CAPE_MULTPL"
    table["label"] = "SHILLER_CAPE"
    table["name"] = "Shiller CAPE Ratio"
    table["units"] = "ratio"
    return table[["date", "value", "series_id", "label", "name", "units"]].reset_index(drop=True)


def save_table(df: pd.DataFrame, path_without_ext: str) -> None:
    df.to_csv(f"{path_without_ext}.csv", index=False)
    df.to_parquet(f"{path_without_ext}.parquet", index=False)


def print_range(label: str, df: pd.DataFrame, value_col: str) -> None:
    first = df["date"].min()
    last = df["date"].max()
    non_null = int(df[value_col].notna().sum())
    print(f"  {label}: {len(df):,} rows, {non_null:,} non-null, {first} -> {last}")


def main() -> None:
    args = parse_args()
    start_date = checked_date(args.start)
    end_date = checked_date(args.end)
    if end_date < start_date:
        raise SystemExit(f"--end {end_date} is before --start {start_date}")

    os.makedirs(args.cache_dir, exist_ok=True)
    session = requests.Session()

    print(f"Downloading daily macro data: {start_date} -> {end_date}")
    dxy = fetch_dxy(session, start_date, end_date)
    save_table(dxy, os.path.join(args.cache_dir, "DXY_daily"))
    print_range("DXY", dxy, "close")
    gold = fetch_gold(session, start_date, end_date)
    save_table(gold, os.path.join(args.cache_dir, "GOLD_daily"))
    print_range("GOLD", gold, "close")
    wilshire = fetch_wilshire_total_market(session, start_date, end_date)
    save_table(wilshire, os.path.join(args.cache_dir, "WILSHIRE_TOTAL_MARKET_daily"))
    print_range("WILSHIRE_TOTAL_MARKET", wilshire, "close")
    shiller_cape = fetch_shiller_cape(session, start_date, end_date)
    save_table(shiller_cape, os.path.join(args.cache_dir, "SHILLER_CAPE_monthly"))
    print_range("SHILLER_CAPE", shiller_cape, "value")

    # --- v2 breadth / cross-asset / vol-structure downloads ---
    xlu = fetch_yahoo_chart(session, XLU_TICKER, XLU_LABEL, start_date, end_date)
    save_table(xlu, os.path.join(args.cache_dir, "XLU_daily"))
    print_range("XLU", xlu, "close")
    xly = fetch_yahoo_chart(session, XLY_TICKER, XLY_LABEL, start_date, end_date)
    save_table(xly, os.path.join(args.cache_dir, "XLY_daily"))
    print_range("XLY", xly, "close")
    eem = fetch_yahoo_chart(session, EEM_TICKER, EEM_LABEL, start_date, end_date)
    save_table(eem, os.path.join(args.cache_dir, "EEM_daily"))
    print_range("EEM", eem, "close")
    efa = fetch_yahoo_chart(session, EFA_TICKER, EFA_LABEL, start_date, end_date)
    save_table(efa, os.path.join(args.cache_dir, "EFA_daily"))
    print_range("EFA", efa, "close")
    copper = fetch_yahoo_chart(session, COPPER_TICKER, COPPER_LABEL, start_date, end_date)
    save_table(copper, os.path.join(args.cache_dir, "COPPER_daily"))
    print_range("COPPER", copper, "close")
    vix3m = fetch_yahoo_chart(session, VIX3M_TICKER, VIX3M_LABEL, start_date, end_date)
    save_table(vix3m, os.path.join(args.cache_dir, "VIX3M_daily"))
    print_range("VIX3M", vix3m, "close")

    combined = dxy[["date", "close"]].rename(columns={"close": "dxy_close"})
    combined = combined.merge(
        gold[["date", "close"]].rename(columns={"close": "gold_usd_per_oz"}),
        on="date",
        how="outer",
    )
    combined = combined.merge(
        wilshire[["date", "close"]].rename(columns={"close": "wilshire_total_market_index"}),
        on="date",
        how="outer",
    )
    combined = combined.merge(
        shiller_cape[["date", "value"]].rename(columns={"value": "shiller_cape_ratio"}),
        on="date",
        how="outer",
    )
    # --- v2 series into combined ---
    for label_df, col_name in [
        (xlu, "xlu_close"),
        (xly, "xly_close"),
        (eem, "eem_close"),
        (efa, "efa_close"),
        (copper, "copper_usd_per_lb"),
        (vix3m, "vix3m_level"),
    ]:
        combined = combined.merge(
            label_df[["date", "close"]].rename(columns={"close": col_name}),
            on="date",
            how="outer",
        )
    series_metadata: dict[str, Any] = {
        DXY_LABEL: {
            "ticker": DXY_TICKER,
            "source": "Yahoo Finance chart API",
            "source_url": YAHOO_CHART_URL.format(ticker=DXY_TICKER),
            "combined_col": "dxy_close",
            "units": "index level",
        },
        GOLD_LABEL: {
            "ticker": GOLD_TICKER,
            "source": "Yahoo Finance chart API",
            "source_url": YAHOO_CHART_URL.format(ticker=quote(GOLD_TICKER, safe="")),
            "combined_col": "gold_usd_per_oz",
            "units": "USD per ounce",
        },
        WILSHIRE_LABEL: {
            "ticker_candidates": WILSHIRE_TICKERS,
            "ticker_used": wilshire["ticker"].iloc[-1] if not wilshire.empty else None,
            "source": "Yahoo Finance chart API",
            "source_url": YAHOO_CHART_URL.format(
                ticker=quote((wilshire["ticker"].iloc[-1] if not wilshire.empty else WILSHIRE_TICKERS[0]), safe="")
            ),
            "combined_col": "wilshire_total_market_index",
            "units": "index level",
        },
        "SHILLER_CAPE": {
            "series_id": "SHILLER_CAPE_MULTPL",
            "source": "Multpl",
            "source_url": MULTPL_SHILLER_CAPE_URL,
            "combined_col": "shiller_cape_ratio",
            "units": "ratio",
        },
        XLU_LABEL: {
            "ticker": XLU_TICKER,
            "source": "Yahoo Finance chart API",
            "combined_col": "xlu_close",
            "units": "USD per share",
        },
        XLY_LABEL: {
            "ticker": XLY_TICKER,
            "source": "Yahoo Finance chart API",
            "combined_col": "xly_close",
            "units": "USD per share",
        },
        EEM_LABEL: {
            "ticker": EEM_TICKER,
            "source": "Yahoo Finance chart API",
            "combined_col": "eem_close",
            "units": "USD per share",
        },
        EFA_LABEL: {
            "ticker": EFA_TICKER,
            "source": "Yahoo Finance chart API",
            "combined_col": "efa_close",
            "units": "USD per share",
        },
        COPPER_LABEL: {
            "ticker": COPPER_TICKER,
            "source": "Yahoo Finance chart API",
            "combined_col": "copper_usd_per_lb",
            "units": "USD per pound",
        },
        VIX3M_LABEL: {
            "ticker": VIX3M_TICKER,
            "source": "Yahoo Finance chart API (CBOE 3-Month Volatility Index)",
            "combined_col": "vix3m_level",
            "units": "index level",
        },
    }

    for label, config in FRED_SERIES.items():
        fred_df = fetch_fred_series(session, label, config, start_date, end_date)
        save_table(fred_df, os.path.join(args.cache_dir, f"{label}_daily"))
        print_range(label, fred_df, "value")
        derived_cols = config.get("derived_combined_cols") or {}
        fred_combined_cols = ["date", "value", *derived_cols.keys()]
        fred_rename = {"value": config["combined_col"], **derived_cols}
        combined = combined.merge(
            fred_df[fred_combined_cols].rename(columns=fred_rename),
            on="date",
            how="outer",
        )
        series_metadata[label] = {
            **config,
            "source": "FRED",
            "source_url": f"https://fred.stlouisfed.org/series/{config['series_id']}",
        }

    combined = combined.sort_values("date").reset_index(drop=True)
    value_columns = [
        "dxy_close", "gold_usd_per_oz", "wilshire_total_market_index", "shiller_cape_ratio",
        "xlu_close", "xly_close", "eem_close", "efa_close", "copper_usd_per_lb", "vix3m_level",
    ]
    for config in FRED_SERIES.values():
        value_columns.append(config["combined_col"])
        value_columns.extend((config.get("derived_combined_cols") or {}).values())
    combined = combined.dropna(subset=value_columns, how="all").reset_index(drop=True)
    save_table(combined, os.path.join(args.cache_dir, "macro_daily_1999"))
    print_range("combined", combined, "dxy_close")

    metadata = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "files": {
            "combined_csv": os.path.join(args.cache_dir, "macro_daily_1999.csv"),
            "combined_parquet": os.path.join(args.cache_dir, "macro_daily_1999.parquet"),
        },
        "series": series_metadata,
        "notes": [
            "Combined table uses an outer join on date and does not forward-fill missing observations.",
            "Rows with no values in any combined data column are dropped from the combined table.",
            "Treasury yields, CPI inflation, unemployment, and market cap to GDP are percent; WTI is USD per barrel; gold is USD per ounce.",
            "CPI and unemployment are monthly FRED observations and are not forward-filled.",
            "Nominal GDP is quarterly FRED GDP in current-dollar SAAR terms and is not forward-filled in the raw combined table.",
            "Market cap to GDP comes from the World Bank via FRED and is annual, so it moves slowly and updates with a long publication lag.",
            "CPI month-over-month and year-over-year inflation percentages are computed from CPIAUCSL.",
            "Shiller CAPE is scraped from Multpl's monthly table and kept on its stated observation date.",
            "Gold uses Yahoo Finance GC=F front-month futures, which is a practical proxy rather than a point-in-time spot fix.",
            "Wilshire uses Yahoo Finance ^FTW5000 with ^W5000 fallback as a practical broad-U.S.-equity proxy.",
        ],
    }
    with open(os.path.join(args.cache_dir, "macro_daily_1999_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
