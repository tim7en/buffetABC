from io import StringIO
from unittest.mock import MagicMock, patch

import requests
from django.core.management import call_command
from django.test import TestCase

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument
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
