from django.core.management.base import BaseCommand, CommandError
from edgar.services.edgar_client import EdgarClient
from edgar.services.fundamentals import (
    save_fundamentals_from_concept,
    save_fundamentals_from_facts,
)
from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Fetch EDGAR data for one or many companies."

    def add_arguments(self, parser):
        parser.add_argument(
            "symbol",
            nargs="?",
            help="Ticker symbol to fetch (case-insensitive). If omitted, data for all S&P500 are retrieved.",
        )
        parser.add_argument(
            "--output",
            "-o",
            help="Directory to save JSON files (default=disabled)",
            default=None,
        )
        parser.add_argument(
            "--facts",
            action="store_true",
            help="Download company facts (default).",
        )
        parser.add_argument(
            "--filings",
            action="store_true",
            help="Download filing timeline JSON.",
        )
        parser.add_argument(
            "--concept",
            action="store_true",
            help="Download a specific company concept endpoint.",
        )
        parser.add_argument(
            "--fulltext",
            action="store_true",
            help="Run SEC full-text filing search.",
        )
        parser.add_argument(
            "--taxonomy",
            default="us-gaap",
            help="Taxonomy for --concept (default=us-gaap).",
        )
        parser.add_argument(
            "--tag",
            default="Assets",
            help="XBRL tag for --concept (default=Assets).",
        )
        parser.add_argument(
            "--query",
            default="",
            help="Query string for --fulltext.",
        )
        parser.add_argument(
            "--search-name",
            default="",
            help="Find symbols by company name substring and fetch matches.",
        )
        parser.add_argument(
            "--persist",
            action="store_true",
            help="Persist responses to Django models.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="When iterating over all companies, stop after this many symbols.",
            default=None,
        )
        parser.add_argument(
            "--retries",
            type=int,
            default=3,
            help="HTTP retries per request (default=3).",
        )
        parser.add_argument(
            "--backoff",
            type=float,
            default=1.0,
            help="Linear backoff seconds for retries (default=1.0).",
        )

    def _resolve_rows(self, symbol: str | None, search_name: str, limit: int | None):
        if symbol:
            row = sp500.by_symbol(symbol)
            if not row:
                raise CommandError(f"Unknown symbol {symbol}")
            return [row]

        if search_name:
            matches = sp500.search_companies(search_name, limit=limit or 25)
            rows = []
            for item in matches:
                rows.append(
                    {
                        "Symbol": item["Symbol"],
                        "Security": item.get("Security", ""),
                        "CIK": item.get("CIK", ""),
                    }
                )
            if not rows:
                raise CommandError(f"No matches for company name search: {search_name}")
            return rows

        return sp500.load_sp500()

    def _persist(self, row, kind, endpoint, payload, attempts=1):
        ticker = row.get("Symbol", "").upper()
        cik = str(row.get("CIK") or row.get("cik") or "").zfill(10)
        name = row.get("Security") or row.get("Name") or ""
        company, _ = EdgarCompany.objects.update_or_create(
            ticker=ticker,
            defaults={
                "cik": cik,
                "name": name,
                "is_sp500": bool(sp500.by_symbol(ticker)),
            },
        )
        doc = EdgarDocument.objects.create(
            company=company,
            kind=kind,
            endpoint=endpoint,
            payload=payload,
            source_url=endpoint,
            http_status=None,
            attempt_count=attempts,
            success=True,
            params={},
        )
        return company, doc

    def _persist_failure(self, row, kind, endpoint, error_message, attempts=1):
        ticker = row.get("Symbol", "").upper() or "UNKNOWN"
        cik = str(row.get("CIK") or row.get("cik") or "").zfill(10)
        name = row.get("Security") or row.get("Name") or ""
        company, _ = EdgarCompany.objects.update_or_create(
            ticker=ticker,
            defaults={
                "cik": cik or "0000000000",
                "name": name,
                "is_sp500": bool(sp500.by_symbol(ticker)),
            },
        )
        EdgarDocument.objects.create(
            company=company,
            kind=kind,
            endpoint=endpoint,
            payload={},
            source_url=endpoint,
            http_status=None,
            attempt_count=attempts,
            success=False,
            error_message=str(error_message),
            params={},
        )

    def handle(self, *args, **options):
        symbol = options.get("symbol")
        outdir = options.get("output")
        do_filings = options.get("filings")
        do_facts = options.get("facts")
        do_concept = options.get("concept")
        do_fulltext = options.get("fulltext")
        search_name = options.get("search_name")
        query = options.get("query")
        taxonomy = options.get("taxonomy")
        tag = options.get("tag")
        persist = options.get("persist")
        limit = options.get("limit")
        retries = options.get("retries")
        backoff = options.get("backoff")

        selected_count = sum([bool(do_facts), bool(do_filings), bool(do_concept), bool(do_fulltext)])
        if selected_count == 0:
            do_facts = True
        elif selected_count > 1:
            raise CommandError("Choose exactly one endpoint flag from --facts/--filings/--concept/--fulltext")

        if do_fulltext and not query:
            if symbol:
                query = symbol
            elif search_name:
                query = search_name
            else:
                raise CommandError("--fulltext requires --query or symbol/search context")

        rows = self._resolve_rows(symbol, search_name, limit)
        if do_fulltext and not symbol and not search_name:
            rows = [{"Symbol": "SEARCH", "CIK": ""}]

        client = EdgarClient(retries=retries, backoff_seconds=backoff)
        count = 0
        for row in rows:
            if limit and count >= limit:
                break
            count += 1

            sym = row.get("Symbol", "").upper()
            cik = str(row.get("CIK") or row.get("cik") or "")
            if not cik and not do_fulltext:
                logger.warning("no CIK for %s, skipping", sym)
                continue

            try:
                if do_facts:
                    kind = EdgarDocument.KIND_FACTS
                    endpoint = client.BASE_URL.format(cik=cik.zfill(10))
                    data = client.company_facts(cik)
                elif do_filings:
                    kind = EdgarDocument.KIND_FILINGS
                    endpoint = client.SUBMISSIONS_URL.format(cik=cik.zfill(10))
                    data = client.filings(cik)
                elif do_concept:
                    kind = EdgarDocument.KIND_CONCEPT
                    endpoint = client.CONCEPT_URL.format(
                        cik=cik.zfill(10), taxonomy=taxonomy, tag=tag
                    )
                    data = client.company_concept(cik, taxonomy=taxonomy, tag=tag)
                else:
                    kind = EdgarDocument.KIND_FULLTEXT
                    endpoint = client.FULL_TEXT_SEARCH_URL
                    scoped_query = query
                    if sym and query == symbol:
                        scoped_query = f"{query} OR {sym}"
                    data = client.full_text_search(scoped_query)

                if outdir:
                    import os
                    import json

                    os.makedirs(outdir, exist_ok=True)
                    suffix = kind
                    path = os.path.join(outdir, f"{sym or 'search'}-{suffix}.json")
                    with open(path, "w") as f:
                        json.dump(data, f)
                    self.stdout.write(f"saved {path}\n")
                else:
                    if kind == EdgarDocument.KIND_FULLTEXT:
                        total = data.get("hits", {}).get("total", {}).get("value", "?")
                    else:
                        total = len(data.get("facts", {})) if kind == EdgarDocument.KIND_FACTS else len(data)
                    self.stdout.write(f"{sym or 'SEARCH'}: {total} items\n")

                if persist:
                    company, doc = self._persist(
                        row, kind=kind, endpoint=endpoint, payload=data, attempts=retries
                    )
                    if kind == EdgarDocument.KIND_FACTS:
                        points = save_fundamentals_from_facts(
                            company=company,
                            payload=data,
                            source_document=doc,
                            taxonomy_filter=None,
                            tag_filter=None,
                        )
                        self.stdout.write(f"{sym}: persisted {points} fundamentals\n")
                    elif kind == EdgarDocument.KIND_CONCEPT:
                        points = save_fundamentals_from_concept(
                            company=company,
                            payload=data,
                            source_document=doc,
                        )
                        self.stdout.write(f"{sym}: persisted {points} fundamentals\n")

            except Exception as exc:
                err_msg = str(exc)
                if do_fulltext and "403" in err_msg:
                    err_msg = (
                        "SEC full-text search returned 403. This endpoint is often blocked "
                        "without approved SEC access profile or when fair-access limits are hit."
                    )
                logger.exception("failed fetching %s for %s", kind if 'kind' in locals() else 'unknown', sym)
                self.stderr.write(f"error {sym}: {err_msg}\n")
                if persist:
                    self._persist_failure(
                        row,
                        kind=kind if "kind" in locals() else EdgarDocument.KIND_FACTS,
                        endpoint=endpoint if "endpoint" in locals() else "",
                        error_message=err_msg,
                        attempts=retries,
                    )
                if symbol:
                    raise CommandError(err_msg)
