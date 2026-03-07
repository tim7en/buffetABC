"""Service for fetching and persisting stock prices.

Uses yfinance to pull latest market data and stores it in StockPrice model.
Supports both single-ticker and bulk refresh operations.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from edgar.models import EdgarCompany, StockPrice

logger = logging.getLogger(__name__)


def fetch_and_store_prices(
    company: EdgarCompany,
    period: str = "1y",
) -> int:
    """Fetch price history for a single company and upsert into DB.

    Returns the number of price rows saved.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed. Run: pip install yfinance")
        return 0

    try:
        ticker_obj = yf.Ticker(company.ticker)
        hist = ticker_obj.history(period=period)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", company.ticker, exc)
        return 0

    if hist.empty:
        return 0

    saved = 0
    for dt_index, row in hist.iterrows():
        price_date = dt_index.date() if hasattr(dt_index, "date") else dt_index
        _, created = StockPrice.objects.update_or_create(
            company=company,
            date=price_date,
            defaults={
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "close": row["Close"],
                "volume": int(row.get("Volume", 0)) if row.get("Volume") else None,
            },
        )
        saved += 1
    return saved


def get_latest_price(company: EdgarCompany) -> StockPrice | None:
    """Return the most recent price record for a company."""
    return StockPrice.objects.filter(company=company).order_by("-date").first()


def fetch_current_quote(ticker: str) -> dict | None:
    """Fetch a real-time quote snapshot (not persisted)."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "ticker": ticker,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as exc:
        logger.warning("Quote fetch failed for %s: %s", ticker, exc)
        return None


def bulk_refresh_prices(
    tickers: list[str] | None = None,
    period: str = "1y",
) -> dict:
    """Refresh prices for multiple companies. If tickers is None, refreshes all persisted companies."""
    if tickers:
        companies = EdgarCompany.objects.filter(ticker__in=[t.upper() for t in tickers])
    else:
        companies = EdgarCompany.objects.all()

    results = {}
    for company in companies:
        count = fetch_and_store_prices(company, period=period)
        results[company.ticker] = count
    return results
