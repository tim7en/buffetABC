import requests
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

USER_AGENT = "buffet-edgar/1.0 (contact: you@example.com)"


class RateLimiter:
    """Simple rate limiter enforcing a maximum number of calls per interval."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._calls = []

    def acquire(self):
        """Block until a new call is allowed."""
        now = time.time()
        # remove timestamps older than period
        self._calls = [t for t in self._calls if now - t < self.period]
        if len(self._calls) >= self.max_calls:
            sleep_for = self.period - (now - self._calls[0])
            logger.debug("Rate limit reached, sleeping for %.2f seconds", sleep_for)
            time.sleep(sleep_for)
        self._calls.append(time.time())


class EdgarClient:
    """Client for fetching data from the SEC EDGAR API.

    Usage:
        client = EdgarClient()
        facts = client.company_facts("0000320193")
    """

    BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    CONCEPT_URL = (
        "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    )
    FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(
        self,
        rate_limit: float = 10,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 20.0,
    ):
        # rate_limit = requests per second
        self._limiter = RateLimiter(max_calls=rate_limit, period=1.0)
        self.retries = max(1, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_status_code: Optional[int] = None
        self._last_url: str = ""

    def _request(self, method: str, url: str, **kwargs) -> dict:
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                self._limiter.acquire()
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                self._last_status_code = resp.status_code
                self._last_url = url
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
                sleep_for = self.backoff_seconds * attempt
                logger.warning(
                    "request failed (%s/%s) for %s %s: %s; retrying in %.1fs",
                    attempt,
                    self.retries,
                    method,
                    url,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        if last_exc:
            raise last_exc
        return {}

    def _get(self, url: str) -> dict:
        return self._request("GET", url)

    def company_facts(self, cik: str) -> dict:
        """Return the raw JSON company facts for the given padded CIK (10 digits)."""
        padded = cik.zfill(10)
        url = self.BASE_URL.format(cik=padded)
        return self._get(url)

    def filings(self, cik: str, count: int = 40) -> dict:
        """Retrieve recent filings index for a company using another SEC endpoint.

        This endpoint returns the filing timeline with links to individual
        filings (10-K, 10-Q, etc). The method returns the JSON blob verbatim.
        """
        padded = cik.zfill(10)
        url = self.SUBMISSIONS_URL.format(cik=padded)
        return self._get(url)

    def company_concept(self, cik: str, taxonomy: str, tag: str) -> dict:
        padded = cik.zfill(10)
        url = self.CONCEPT_URL.format(cik=padded, taxonomy=taxonomy, tag=tag)
        return self._get(url)

    def full_text_search(self, query: str, start: int = 0, size: int = 25) -> dict:
        payload = {
            "q": query,
            "category": "custom",
            "startdt": "2001-01-01",
            "enddt": None,
            "from": str(start),
            "size": str(size),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        return self._request("POST", self.FULL_TEXT_SEARCH_URL, json=payload)

    # future helpers could parse specific metrics
def extract_metric(facts: Dict, metric: str, year: Optional[int] = None) -> Optional[float]:
    """Helper to pull a number from the us-gaap facts dictionary.

    Arguments:
        facts: the 'facts' subobject of the returned JSON
        metric: e.g. 'Assets', 'Revenues'
        year: if specified, restrict to a single year
    """
    inst = facts.get(metric.lower()) or facts.get(metric)
    if not inst:
        return None
    # look for 'units' -> 'USD' -> list of data points
    units = inst.get("units", {})
    usd = units.get("USD") or units.get("USD")
    if not usd:
        return None
    if year:
        for entry in usd:
            if entry.get("end").startswith(str(year)):
                return entry.get("val")
        return None
    # otherwise return the most recent
    last = usd[-1]
    return last.get("val")
