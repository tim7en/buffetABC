# buffetABC

This repository is a modular Django project focused on collecting and serving financial data from the U.S. SEC EDGAR system.

## Components

### `edgar` Django app

- `services/edgar_client.py` - SEC EDGAR client with rate limiting and retry/backoff.
- `sp500.py` - helpers for loading S&P 500 constituents and searching by symbol/company name.
- `management/commands/fetch_edgar.py` - fetch one or many EDGAR endpoints with optional persistence.
- `management/commands/sync_edgar_nightly.py` - cron/Celery-friendly nightly bulk sync.
- `models.py` - persistent company and payload storage.
- `urls.py` / `views.py` - JSON API for company/document access and full-text search.
- `admin.py` - admin screens for monitoring successful/failed downloads.

## Usage

1. **Activate environment**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run migrations**
   ```bash
   python manage.py migrate
   ```

4. **Fetch/persist data**
   ```bash
   # Facts
   python manage.py fetch_edgar AAPL --facts --persist

   # Filings for first 10 S&P 500 names
   python manage.py fetch_edgar --filings --persist --limit 10

   # Search by company name
   python manage.py fetch_edgar --search-name apple --facts --persist --limit 5

   # Company concept (XBRL endpoint)
   python manage.py fetch_edgar AAPL --concept --taxonomy us-gaap --tag Assets --persist

   # SEC full-text filing search
   python manage.py fetch_edgar --fulltext --query "10-K artificial intelligence" --persist
   ```

5. **Nightly bulk sync**
   ```bash
   python manage.py sync_edgar_nightly
   ```

   Example cron (02:30 daily):
   ```cron
   30 2 * * * cd /path/to/buffetABC && /path/to/python manage.py sync_edgar_nightly >> /var/log/edgar_sync.log 2>&1
   ```

6. **API endpoints**
   - `GET /api/edgar/companies/?q=AAP&page=1&page_size=20`
   - `GET /api/edgar/companies/search/?q=apple&limit=10`
   - `GET /api/edgar/companies/AAPL/documents/?kind=facts&include_payload=1`
   - `GET /api/edgar/filings/fulltext/?q=10-K+AI&start=0&size=25`
   - DRF API root: `GET /api/edgar/drf/`
   - DRF bulk ingestion: `POST /api/edgar/drf/ingestion/fetch/`
   - DRF single-company fetch: `POST /api/edgar/drf/companies/{id}/fetch/`
   - DRF fundamentals over time: `GET /api/edgar/drf/companies/{id}/fundamentals/?tag=Assets&period_start=2020-01-01&period_end=2024-12-31`
   - DRF fundamentals table: `GET /api/edgar/drf/fundamentals/?ticker=AAPL&taxonomy=us-gaap&tag=Assets&from=2020-01-01&to=2024-12-31`

   DRF bulk ingestion example:
   ```json
   {
     "symbols": ["AAPL", "MSFT"],
     "endpoint": "company_concept",
     "taxonomy": "us-gaap",
     "tag": "Assets",
     "persist": true,
     "period_start": "2020-01-01",
     "period_end": "2024-12-31"
   }
   ```

7. **Web dashboard**
   - Open `http://127.0.0.1:8000/`
   - Supports single/bulk company selection, timeline period filters, fetch/save actions, and auto-refresh.
   - No login is required for dashboard/API by default.
   - Login is required only for `http://127.0.0.1:8000/admin/`.

## Notes

- EDGAR API requests are rate-limited (10 req/s default) and retried with backoff.
- Failed downloads are stored with `success=False` and `error_message` for admin monitoring.
- Raw source payloads are stored as JSON for later analytics/parsing.
- Normalized point-in-time fundamentals are also stored in `EdgarFundamental`.
- Set `EDGAR_USER_AGENT` to your real app/contact (SEC fair-access requirement), for example:
  ```bash
  export EDGAR_USER_AGENT=\"my-edgar-app/1.0 (contact: you@company.com)\"
  ```
- SEC full-text search (`efts.sec.gov`) may return `403` for fair-access / network policy reasons; this is surfaced as a clear error in ingestion responses.
