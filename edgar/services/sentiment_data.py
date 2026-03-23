"""
Sentiment data fetchers for the exposure governor.

Each loader returns a ``dict[str, float]`` mapping ISO date strings (``YYYY-MM-DD``)
to a normalised **fear/greed score on a 0–100 scale**:

  * 0   = extreme fear  (reduce exposure)
  * 100 = extreme greed (full exposure)

Supported sources
-----------------
vix       CBOE VIX daily close (yfinance ``^VIX``).
          Inverted: low VIX → high score.
          Formula: score = clamp(100 − (vix − 10) × 2, 0, 100)
          VIX 10 → 100, VIX 20 → 80, VIX 30 → 60, VIX 40 → 40, VIX 60 → 0

crypto_fg Crypto Fear & Greed Index (alternative.me public API).
          Already 0–100; no transformation needed.

cnn_fg    CNN Fear & Greed Index (CNN dataviz semi-public API).
          Already 0–100; no transformation needed.

aaii      AAII Investor Sentiment Survey — bull% column.
          Weekly survey, forward-filled to daily.
          Requires a manual CSV download (see ``load_aaii_scores`` docstring).

All fetched data is cached under ``cache/sentiment/`` as JSON so subsequent
calls are instant.  Pass ``force_refresh=True`` to re-download.

Coverage verification
---------------------
Use ``coverage_report(scores, backtest_start, backtest_end)`` to check that
every trading date in the backtest window has a score.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache/sentiment")


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _load_json_cache(filename: str) -> dict[str, float] | None:
    path = _cache_path(filename)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read cache %s: %s", filename, exc)
        return None


def _save_json_cache(filename: str, data: dict[str, float]) -> None:
    path = _cache_path(filename)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Cached %d rows → %s", len(data), path)


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _iter_dates(start: str, end: str):
    """Yield YYYY-MM-DD strings from *start* to *end* inclusive."""
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= stop:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _forward_fill(weekly: dict[str, float], start: str, end: str) -> dict[str, float]:
    """
    Forward-fill a sparse dict (e.g. weekly survey) into a daily dict
    covering *start* → *end* inclusive.
    """
    if not weekly:
        return {}
    all_weekly = sorted(weekly)
    daily: dict[str, float] = {}
    last_score: float = weekly[all_weekly[0]]
    weekly_dates = {
        datetime.strptime(d, "%Y-%m-%d").date(): v for d, v in weekly.items()
    }
    cur = datetime.strptime(min(start, all_weekly[0]), "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= stop:
        if cur in weekly_dates:
            last_score = weekly_dates[cur]
        daily[cur.strftime("%Y-%m-%d")] = last_score
        cur += timedelta(days=1)
    return daily


# ---------------------------------------------------------------------------
# VIX
# ---------------------------------------------------------------------------

def load_vix_scores(
    start: str = "2018-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> dict[str, float]:
    """
    Fetch CBOE VIX daily closes via yfinance and convert to a 0–100 fear/greed
    score (100 = low volatility / greed, 0 = extreme volatility / fear).

    Score formula::

        score = clamp(100 − (vix_close − 10) × 2, 0, 100)

    VIX benchmarks → score:  12 → 96  |  20 → 80  |  30 → 60  |  40 → 40  |  60 → 0
    """
    cache_file = "vix_scores.json"
    if not force_refresh:
        cached = _load_json_cache(cache_file)
        if cached:
            logger.info("VIX: loaded %d rows from cache", len(cached))
            return cached

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    end_str = end or date.today().strftime("%Y-%m-%d")
    logger.info("VIX: downloading ^VIX from %s to %s …", start, end_str)
    raw = yf.download("^VIX", start=start, end=end_str, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned empty VIX data")

    # yfinance ≥1.x may return multi-level columns; flatten if needed
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)

    scores: dict[str, float] = {}
    for idx, row in raw.iterrows():
        d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        vix_val = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
        score = max(0.0, min(100.0, 100.0 - (vix_val - 10.0) * 2.0))
        scores[d] = round(score, 2)

    _save_json_cache(cache_file, scores)
    logger.info("VIX: %d rows saved", len(scores))
    return scores


# ---------------------------------------------------------------------------
# Crypto Fear & Greed
# ---------------------------------------------------------------------------

def load_crypto_fg_scores(force_refresh: bool = False) -> dict[str, float]:
    """
    Fetch the Crypto Fear & Greed Index from the alternative.me public API.
    Returns the full history (since 2018-02-01) as a daily 0–100 score.

    API endpoint: https://api.alternative.me/fng/?limit=0&format=json
    """
    cache_file = "crypto_fg_scores.json"
    if not force_refresh:
        cached = _load_json_cache(cache_file)
        if cached:
            logger.info("Crypto F&G: loaded %d rows from cache", len(cached))
            return cached

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    logger.info("Crypto F&G: fetching from %s …", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", [])
    if not data:
        raise RuntimeError("alternative.me returned no data")

    scores: dict[str, float] = {}
    for entry in data:
        # API returns Unix timestamps as strings
        ts = int(entry["timestamp"])
        d = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        scores[d] = float(entry["value"])

    _save_json_cache(cache_file, scores)
    logger.info("Crypto F&G: %d rows saved (earliest: %s)", len(scores), min(scores))
    return scores


# ---------------------------------------------------------------------------
# CNN Fear & Greed
# ---------------------------------------------------------------------------

def load_cnn_fg_scores(
    force_refresh: bool = False,
) -> dict[str, float]:
    """
    Fetch the CNN Fear & Greed Index from the CNN dataviz API.
    Returns daily 0–100 scores.

    API endpoint (semi-public, no key required):
    https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}

    CNN returns ~2–3 years of history per request.  We batch from 2022-01-01
    in ~1-year chunks and merge, so the full backtest window is covered.
    """
    cache_file = "cnn_fg_scores.json"
    if not force_refresh:
        cached = _load_json_cache(cache_file)
        if cached:
            logger.info("CNN F&G: loaded %d rows from cache", len(cached))
            return cached

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        "Origin": "https://edition.cnn.com",
    }

    # Batch start dates — CNN 500s on very old dates, so walk forward
    batch_starts = ["2022-01-01", "2023-06-01", "2025-01-01"]
    scores: dict[str, float] = {}

    for start in batch_starts:
        url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
        logger.info("CNN F&G: fetching batch from %s …", start)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            historical = payload.get("fear_and_greed_historical", {}).get("data", [])
            for entry in historical:
                ts_ms = int(entry["x"])
                d = datetime.utcfromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")
                scores[d] = round(float(entry["y"]), 2)
            logger.info("CNN F&G: batch %s → %d entries (total so far: %d)", start, len(historical), len(scores))
        except Exception as exc:
            logger.warning("CNN F&G: batch %s failed: %s", start, exc)

    if not scores:
        raise RuntimeError(
            "CNN F&G: all batches failed — the endpoint may have changed or be "
            "temporarily unavailable.  Try again later."
        )

    _save_json_cache(cache_file, scores)
    logger.info("CNN F&G: %d rows saved (earliest: %s)", len(scores), min(scores))
    return scores


# ---------------------------------------------------------------------------
# AAII Investor Sentiment
# ---------------------------------------------------------------------------

_AAII_CACHE_CSV = "aaii_sentiment.csv"
_AAII_INSTRUCTIONS = """
AAII data not found.

Download steps:
  1. Go to https://www.aaii.com/sentimentsurvey/sent_results
  2. Click the "Download" / "Export" button (free registration required).
  3. Save the CSV as:  cache/sentiment/aaii_sentiment.csv

Expected columns (any order):  Date, Bullish, Neutral, Bearish
Date format:  M/D/YYYY  (e.g. 1/5/2023)
Bullish column: percentage as decimal (0.3845) or integer (38.45)

Once the file is in place, re-run the fetch command.
"""


def load_aaii_scores(
    csv_path: str | Path | None = None,
    backtest_end: str = "2026-03-03",
    force_refresh: bool = False,
) -> dict[str, float]:
    """
    Load AAII Investor Sentiment Survey bull% as a daily 0–100 score.

    The AAII survey is weekly (every Thursday).  Missing days are forward-filled.

    *csv_path* defaults to ``cache/sentiment/aaii_sentiment.csv``.
    Download instructions are printed if the file is missing.
    """
    import csv as _csv

    cache_file = "aaii_scores.json"
    if not force_refresh:
        cached = _load_json_cache(cache_file)
        if cached:
            logger.info("AAII: loaded %d rows from cache", len(cached))
            return cached

    if csv_path is None:
        csv_path = _cache_path(_AAII_CACHE_CSV)
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(_AAII_INSTRUCTIONS)

    weekly: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8-sig") as fh:
        # Skip comment / header lines that don't look like data
        lines = [l for l in fh if l.strip() and not l.startswith(";")]

    reader = _csv.DictReader(lines)
    date_col = bull_col = None
    for row in reader:
        # Detect column names on first real row
        if date_col is None:
            for k in row:
                kl = k.strip().lower()
                if kl in ("date", "reported date"):
                    date_col = k
                elif kl == "bullish":
                    bull_col = k
            if date_col is None or bull_col is None:
                raise ValueError(
                    f"Could not find 'Date' and 'Bullish' columns in AAII CSV. "
                    f"Found: {list(row.keys())}"
                )

        date_raw = row[date_col].strip()
        bull_raw = row[bull_col].strip().rstrip("%")
        if not date_raw or not bull_raw:
            continue
        try:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%Y"):
                try:
                    d = datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                logger.debug("AAII: could not parse date %r, skipping", date_raw)
                continue

            score = float(bull_raw)
            if score <= 1.0:          # stored as 0.xx fraction
                score *= 100.0
            weekly[d] = round(min(100.0, max(0.0, score)), 2)
        except (ValueError, TypeError):
            continue

    if not weekly:
        raise ValueError("No valid rows parsed from AAII CSV")

    # Forward-fill weekly → daily up to backtest_end
    daily = _forward_fill(weekly, min(weekly), backtest_end)
    _save_json_cache(cache_file, daily)
    logger.info("AAII: %d weekly rows → %d daily rows", len(weekly), len(daily))
    return daily


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

def get_score_for_date(
    date_str: str,
    scores: dict[str, float],
    lookback_days: int = 5,
) -> float | None:
    """
    Return the sentiment score for *date_str* (``YYYY-MM-DD``).

    Falls back up to *lookback_days* prior calendar days if the exact date
    is missing (weekends, holidays, data gaps).  Returns ``None`` if no
    score is available within the window.
    """
    d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    for offset in range(lookback_days + 1):
        key = (d - timedelta(days=offset)).strftime("%Y-%m-%d")
        if key in scores:
            return scores[key]
    return None


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(
    scores: dict[str, float],
    source_label: str,
    backtest_start: str = "2022-02-09",
    backtest_end: str = "2026-03-02",
) -> dict:
    """
    Check how well *scores* covers the backtest window.

    Returns a summary dict with counts and lists of gap dates.
    Prints a human-readable table.
    """
    all_dates = list(_iter_dates(backtest_start, backtest_end))
    total = len(all_dates)
    covered = 0
    gap_dates: list[str] = []

    for d in all_dates:
        score = get_score_for_date(d, scores, lookback_days=3)
        if score is not None:
            covered += 1
        else:
            gap_dates.append(d)

    pct = covered / total * 100.0 if total else 0.0
    print(
        f"\n{'─'*60}\n"
        f"  Source      : {source_label}\n"
        f"  Window      : {backtest_start} → {backtest_end}\n"
        f"  Total days  : {total}\n"
        f"  Covered     : {covered}  ({pct:.1f}%)\n"
        f"  Gaps        : {len(gap_dates)}"
    )
    if gap_dates:
        # Show first and last 5 gaps only
        show = gap_dates[:5] + (["…"] if len(gap_dates) > 10 else []) + gap_dates[-5:]
        print(f"  Gap dates   : {show}")
    if scores:
        earliest = min(scores)
        latest = max(scores)
        score_min = min(scores.values())
        score_max = max(scores.values())
        score_mean = sum(scores.values()) / len(scores)
        print(
            f"  Data range  : {earliest} → {latest}\n"
            f"  Score range : {score_min:.1f} – {score_max:.1f}  "
            f"(mean {score_mean:.1f})\n"
            f"{'─'*60}"
        )

    return {
        "source": source_label,
        "total_days": total,
        "covered": covered,
        "coverage_pct": round(pct, 2),
        "gap_count": len(gap_dates),
        "gap_dates": gap_dates,
        "data_start": min(scores) if scores else None,
        "data_end": max(scores) if scores else None,
    }
