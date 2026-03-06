"""Utilities for working with the S&P 500 company list.

The source is the community managed JSON or CSV, for example from
"https://datahub.io/core/s-and-p-500-companies" or similar.
This module provides helpers to load and cache the list and to map
symbols to CIKs or names.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# match the client user agent so SEC doesn't 403 us
USER_AGENT = "buffet-edgar/1.0 (contact: you@example.com)"

logger = logging.getLogger(__name__)

# default path inside project where we keep a snapshot, not committed
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SP500_CSV = DATA_DIR / "sp500.csv"


def load_sp500() -> List[Dict[str, str]]:
    """Load the S&P 500 list from a local CSV file. If it doesn't exist,
    attempt to download from a known URL and store it for future use.
    """
    if not SP500_CSV.exists():
        try:
            import requests

            url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            logger.info("downloading sp500 list from %s", url)
            resp = requests.get(url)
            resp.raise_for_status()
            SP500_CSV.write_bytes(resp.content)
        except Exception:
            logger.exception("failed to obtain sp500 list")
            return []

    rows = []
    with SP500_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def symbols() -> List[str]:
    return [r["Symbol"] for r in load_sp500()]


def by_symbol(symbol: str) -> Optional[Dict[str, str]]:
    """Return the raw row from the cached sp500 list (if any)."""
    for r in load_sp500():
        if r["Symbol"] == symbol.upper():
            return r
    # if not in sp500 list, try loading the universal ticker -> CIK mapping
    extra = _load_cik_map()
    if symbol.upper() in extra:
        return {"Symbol": symbol.upper(), "CIK": extra[symbol.upper()]}
    return None


def search_companies(query: str, limit: int = 25) -> List[Dict[str, str]]:
    """Search companies by ticker or display name.

    Results include S&P 500 records first, then extra SEC ticker map matches.
    """
    q = query.strip().upper()
    if not q:
        return []

    results: List[Dict[str, str]] = []
    seen = set()
    for row in load_sp500():
        symbol = row.get("Symbol", "").upper()
        name = row.get("Security", "")
        if q in symbol or q in name.upper():
            record = {
                "Symbol": symbol,
                "Security": name,
                "CIK": row.get("CIK", ""),
                "source": "sp500",
            }
            results.append(record)
            seen.add(symbol)
            if len(results) >= limit:
                return results

    for item in _load_cik_entries():
        symbol = item.get("ticker", "").upper()
        title = item.get("title", "")
        if not symbol or symbol in seen:
            continue
        if q in symbol or q in title.upper():
            results.append(
                {
                    "Symbol": symbol,
                    "Security": title,
                    "CIK": str(item.get("cik_str", "")).zfill(10),
                    "source": "sec_tickers",
                }
            )
            seen.add(symbol)
            if len(results) >= limit:
                break
    return results


def _load_cik_entries() -> List[Dict[str, str]]:
    path = DATA_DIR / "company_tickers.json"
    if not path.exists():
        _load_cik_map()
        if not path.exists():
            return []
    try:
        data = json.loads(path.read_text())
        return list(data.values())
    except Exception:
        logger.exception("failed parsing CIK entries")
        return []


def _load_cik_map() -> Dict[str, str]:
    """Load or download the SEC's master ticker-to-CIK JSON file.

    The file is fairly small (~20MB on disk); we cache it under data/.
    """
    path = DATA_DIR / "company_tickers.json"
    if not path.exists():
        try:
            import requests
            url = "https://www.sec.gov/files/company_tickers.json"
            logger.info("downloading ticker->CIK map from %s", url)
            resp = requests.get(url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except Exception:
            logger.exception("could not download CIK map")
            return {}
    try:
        data = json.loads(path.read_text())
        # structure is dict of entries with 'ticker' and numeric 'cik_str'
        result: Dict[str, str] = {}
        for item in data.values():
            ticker = item.get('ticker')
            cik_val = item.get('cik_str')
            if ticker and cik_val is not None:
                # ensure zero-padded string
                cik = str(cik_val).zfill(10)
                result[ticker.upper()] = cik
        return result
    except Exception:
        logger.exception("failed parsing CIK map")
        return {}
