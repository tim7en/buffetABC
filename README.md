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

## Notes

- EDGAR API requests are rate-limited (10 req/s default) and retried with backoff.
- Failed downloads are stored with `success=False` and `error_message` for admin monitoring.
- Raw source payloads are stored as JSON for later analytics/parsing.
