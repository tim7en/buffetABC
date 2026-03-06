from io import StringIO
import json
from unittest.mock import MagicMock, patch

import requests
from django.core.management import call_command
from django.test import TestCase

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental, EdgarMetricMapping
from edgar.services.edgar_client import EdgarClient, RateLimiter


class SP500Tests(TestCase):
    def test_load_returns_list(self):
        companies = sp500.load_sp500()
        self.assertIsInstance(companies, list)

    def test_symbols_upper(self):
        syms = sp500.symbols()
        for s in syms:
            self.assertEqual(s, s.upper())


class RateLimiterTests(TestCase):
    def test_simple_rate_limit(self):
        rl = RateLimiter(max_calls=2, period=1)
        import time

        start = time.time()
        rl.acquire()
        rl.acquire()
        rl.acquire()
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 1.0)


class EdgarClientTests(TestCase):
    def test_request_retries_then_success(self):
        client = EdgarClient(retries=3, backoff_seconds=0)

        bad = requests.RequestException("temporary")
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status.return_value = None
        ok_response.json.return_value = {"ok": True}

        with patch.object(
            client.session,
            "request",
            side_effect=[bad, ok_response],
        ):
            data = client._request("GET", "http://example.com")
        self.assertEqual(data, {"ok": True})


class CommandPersistenceTests(TestCase):
    @patch("edgar.management.commands.fetch_edgar.sp500.load_sp500")
    @patch("edgar.management.commands.fetch_edgar.EdgarClient.company_facts")
    def test_fetch_command_persists_success(self, mock_company_facts, mock_load_sp500):
        mock_load_sp500.return_value = [
            {"Symbol": "AAPL", "Security": "Apple Inc.", "CIK": "320193"}
        ]
        mock_company_facts.return_value = {"facts": {"us-gaap": {}}}

        out = StringIO()
        call_command("fetch_edgar", "--facts", "--persist", "--limit=1", stdout=out)

        self.assertEqual(EdgarCompany.objects.count(), 1)
        self.assertEqual(EdgarDocument.objects.count(), 1)
        doc = EdgarDocument.objects.first()
        self.assertTrue(doc.success)
        self.assertEqual(doc.kind, EdgarDocument.KIND_FACTS)

    @patch("edgar.management.commands.fetch_edgar.sp500.load_sp500")
    @patch("edgar.management.commands.fetch_edgar.EdgarClient.company_facts")
    def test_fetch_command_persists_failure(self, mock_company_facts, mock_load_sp500):
        mock_load_sp500.return_value = [
            {"Symbol": "AAPL", "Security": "Apple Inc.", "CIK": "320193"}
        ]
        mock_company_facts.side_effect = RuntimeError("boom")

        out = StringIO()
        call_command("fetch_edgar", "--facts", "--persist", "--limit=1", stdout=out)

        self.assertEqual(EdgarDocument.objects.count(), 1)
        doc = EdgarDocument.objects.first()
        self.assertFalse(doc.success)
        self.assertIn("boom", doc.error_message)


class ApiTests(TestCase):
    def setUp(self):
        company = EdgarCompany.objects.create(
            ticker="AAPL", name="Apple Inc.", cik="0000320193", is_sp500=True
        )
        EdgarDocument.objects.create(
            company=company,
            kind=EdgarDocument.KIND_FACTS,
            endpoint="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
            payload={"facts": {}},
            success=True,
        )

    def test_company_documents_endpoint(self):
        res = self.client.get("/api/edgar/companies/AAPL/documents/")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(body["results"][0]["kind"], EdgarDocument.KIND_FACTS)

    def test_company_universe_endpoint(self):
        res = self.client.get("/api/edgar/companies/universe/?q=AAPL&limit=10")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertGreaterEqual(body["count"], 1)
        first = body["results"][0]
        self.assertIn("ticker", first)
        self.assertIn("company_id", first)


class DrfApiTests(TestCase):
    @patch("edgar.drf_views.EdgarClient.company_facts")
    def test_bulk_ingestion_endpoint(self, mock_company_facts):
        mock_company_facts.return_value = {"facts": {"us-gaap": {"Assets": {"units": {"USD": []}}}}}
        body = {
            "symbols": ["AAPL", "MSFT"],
            "endpoint": "facts",
            "persist": True,
        }
        res = self.client.post(
            "/api/edgar/drf/ingestion/fetch/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["requested"], 2)
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("company_id", payload["results"][0])
        self.assertTrue(EdgarDocument.objects.filter(kind=EdgarDocument.KIND_FACTS).exists())

    @patch("edgar.drf_views.EdgarClient.company_concept")
    def test_single_company_fetch_and_fundamentals_period_filter(self, mock_company_concept):
        mock_company_concept.return_value = {
            "taxonomy": "us-gaap",
            "tag": "Assets",
            "units": {
                "USD": [
                    {"end": "2021-12-31", "val": 100},
                    {"end": "2022-12-31", "val": 150},
                    {"end": "2023-12-31", "val": 180},
                ]
            }
        }
        company = EdgarCompany.objects.create(
            ticker="AAPL", name="Apple Inc.", cik="0000320193", is_sp500=True
        )
        fetch_body = {"endpoint": "company_concept", "tag": "Assets", "persist": True}
        fetch_res = self.client.post(
            f"/api/edgar/drf/companies/{company.id}/fetch/",
            data=json.dumps(fetch_body),
            content_type="application/json",
        )
        self.assertEqual(fetch_res.status_code, 200)

        res = self.client.get(
            f"/api/edgar/drf/companies/{company.id}/fundamentals/?tag=Assets&period_start=2022-01-01&period_end=2023-12-31"
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["count"], 2)

    @patch("edgar.drf_views.EdgarClient.company_facts")
    def test_ingestion_saves_normalized_fundamentals(self, mock_company_facts):
        mock_company_facts.return_value = {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2022-12-31", "val": 100, "fy": 2022, "fp": "FY", "form": "10-K"}
                            ]
                        }
                    }
                }
            }
        }
        body = {"symbols": ["AAPL"], "endpoint": "facts", "persist": True}
        res = self.client.post(
            "/api/edgar/drf/ingestion/fetch/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(EdgarFundamental.objects.count(), 1)
        point = EdgarFundamental.objects.first()
        self.assertEqual(point.tag, "Assets")
        self.assertEqual(point.taxonomy, "us-gaap")

        list_res = self.client.get("/api/edgar/drf/fundamentals/?ticker=AAPL&tag=Assets")
        self.assertEqual(list_res.status_code, 200)
        self.assertGreaterEqual(list_res.json()["count"], 1)

    def test_fundamental_table_endpoint_returns_rows_and_mapping(self):
        company = EdgarCompany.objects.create(
            ticker="AAPL", name="Apple Inc.", cik="0000320193", is_sp500=True
        )
        EdgarFundamental.objects.create(
            company=company,
            taxonomy="us-gaap",
            tag="Revenues",
            unit="USD",
            end_date="2022-12-31",
            filed_date="2023-01-30",
            value=1000,
            form="10-K",
            fiscal_year=2022,
            fiscal_period="FY",
        )
        EdgarFundamental.objects.create(
            company=company,
            taxonomy="us-gaap",
            tag="NetIncomeLoss",
            unit="USD",
            end_date="2022-12-31",
            filed_date="2023-01-30",
            value=210,
            form="10-K",
            fiscal_year=2022,
            fiscal_period="FY",
        )
        res = self.client.get(
            f"/api/edgar/drf/companies/{company.id}/fundamental-table/?use_ai=0&frequency=annual&refresh_mapping=1"
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertGreaterEqual(payload["row_count"], 1)
        self.assertIn("revenue", payload["mapping"])
        self.assertIn("net_income", payload["mapping"])
        self.assertTrue(EdgarMetricMapping.objects.filter(company=company).exists())
