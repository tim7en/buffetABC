from datetime import date
from io import StringIO
import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental, EdgarMetricMapping
from edgar.services.edgar_client import EdgarClient, RateLimiter
from edgar.services.strategy import BacktestResult, Trade, backtest_to_dict, _williams_fractals


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

    @patch("edgar.drf_views.EdgarClient.company_facts")
    def test_ingestion_endpoint_not_blocked_for_logged_in_session(self, mock_company_facts):
        mock_company_facts.return_value = {"facts": {"us-gaap": {"Assets": {"units": {"USD": []}}}}}
        user_model = get_user_model()
        user = user_model.objects.create_user(username="u1", password="pass12345")
        self.client.force_login(user)
        body = {"symbols": ["AAPL"], "endpoint": "facts", "persist": True}
        res = self.client.post(
            "/api/edgar/drf/ingestion/fetch/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)

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


class StrategySerializationTests(TestCase):
    def test_backtest_payload_uses_volume_fields_not_buffett_fields(self):
        trade = Trade(
            direction="long",
            entry_date=date(2024, 1, 2),
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            risk_pct=0.01,
            position_size=1000.0,
            shares=10.0,
            exit_date=date(2024, 1, 10),
            exit_price=108.0,
            pnl=79.5,
            exit_reason="take_profit",
            fees_paid=2.5,
            entry_rel_volume=1.25,
            volume_confirmed=True,
            sizing_tier="standard",
            signal_quality="B",
            hold_days=8,
            stop_source="fractal",
            fractal_high=112.0,
            fractal_low=94.0,
        )
        result = BacktestResult(
            ticker="AOS",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=10_000.0,
            final_capital=10_750.0,
            total_return_pct=7.5,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
            max_drawdown_pct=2.1,
            profit_factor=3.2,
            cagr_pct=7.4,
            avg_trade_return_pct=7.95,
            exposure_pct=14.2,
            total_fees=2.5,
            long_trades=1,
            short_trades=0,
            trades=[trade],
            equity_curve=[{"date": "2024-01-02", "equity": 10010.0, "capital": 10000.0}],
        )
        payload = backtest_to_dict(result)
        self.assertIn("profit_factor", payload)
        self.assertIn("cagr_pct", payload)
        self.assertIn("long_trades", payload)
        self.assertIn("short_trades", payload)
        self.assertEqual(len(payload["trades"]), 1)
        t0 = payload["trades"][0]
        self.assertIn("entry_rel_volume", t0)
        self.assertIn("volume_confirmed", t0)
        self.assertIn("sizing_tier", t0)
        self.assertIn("signal_quality", t0)
        self.assertIn("stop_source", t0)
        self.assertIn("fractal_high", t0)
        self.assertIn("fractal_low", t0)
        self.assertNotIn("buffett_direction", t0)
        self.assertNotIn("confirmation", t0)


class StrategyIndicatorTests(TestCase):
    def test_williams_fractal_detection(self):
        highs = [10.0, 11.0, 15.0, 12.0, 11.0, 13.0, 12.0]
        lows = [9.0, 8.0, 7.0, 8.0, 9.0, 8.5, 9.5]
        frac_hi, frac_lo = _williams_fractals(highs, lows, period=2)
        self.assertEqual(len(frac_hi), len(highs))
        self.assertEqual(len(frac_lo), len(lows))
        self.assertEqual(frac_hi[2], 15.0)
        self.assertEqual(frac_lo[2], 7.0)
