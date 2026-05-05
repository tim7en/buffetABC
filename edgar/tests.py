from datetime import date, datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.management import call_command
from django.test import TestCase

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental, EdgarMetricMapping
from edgar.services.edgar_client import EdgarClient, RateLimiter
from edgar.services.session_turtle_portfolio import generate_session_turtle_shared_account_report
from edgar.services.session_turtle_trend_strategy import _breakout_conviction_multiplier
from edgar.services.strategy import (
    BacktestResult,
    Trade,
    _break_even_stop_candidate,
    _chandelier_stop_candidate,
    _resolve_bar_bracket_exit,
    _sweep_exhaustion_confirmed,
    _stop_is_break_even_or_better,
    _williams_fractals,
    backtest_to_dict,
)


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

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_writes_csv_and_png_outputs(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "strategy_variant": "session_turtle_trend_shared_account",
                "label": "Session Turtle Trend x2",
                "start_date": "2024-01-02T10:00:00",
                "end_date": "2024-01-10T10:00:00",
                "candidate_trades": 2,
                "executed_trades": 1,
                "long_trades": 1,
                "short_trades": 0,
                "winning_trades": 1,
                "losing_trades": 0,
                "skipped_same_ticker": 0,
                "skipped_no_capacity": 1,
                "initial_capital": 1000.0,
                "final_equity": 1125.0,
                "total_return_pct": 12.5,
                "cagr_pct": 12.5,
                "max_realized_drawdown_pct": 0.0,
                "win_rate_pct": 100.0,
                "profit_factor": 999.0,
                "exposure_mult": 2.0,
                "channel_period": 20,
                "lookback_years": 4.1,
                "base_risk_pct": 0.05,
                "directional_volume_risk_pct": 0.07,
                "trend_fast_period": 55,
                "trend_slow_period": 200,
                "gold_trades": 0,
                "crypto_trades": 1,
                "equity_trades": 0,
                "metals_trades": 0,
                "gold_pnl": 0.0,
                "crypto_pnl": 125.0,
                "equity_pnl": 0.0,
                "metals_pnl": 0.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [
                {
                    "ticker": "BTC-USD",
                    "source": "binance",
                    "session_open": "new_york_equity_open",
                    "asset_bucket": "crypto",
                    "direction": "long",
                    "entry_ts": "2024-01-02T10:00:00",
                    "exit_ts": "2024-01-10T10:00:00",
                    "entry_price": 100.0,
                    "exit_price": 112.5,
                    "shares": 10.0,
                    "notional": 1000.0,
                    "scale": 1.0,
                    "entry_rel_volume": 1.8,
                    "risk_model": "directional_volume_boost",
                    "net_pnl": 125.0,
                    "equity_after_exit": 1125.0,
                }
            ],
            "yearly_returns": [
                {
                    "year": 2024,
                    "start_equity": 1000.0,
                    "end_equity": 1125.0,
                    "pnl": 125.0,
                    "return_pct": 12.5,
                    "trades": 1,
                    "long_trades": 1,
                    "short_trades": 0,
                    "win_rate_pct": 100.0,
                }
            ],
            "asset_summary": [
                {
                    "ticker": "BTC-USD",
                    "source": "binance",
                    "asset_bucket": "crypto",
                    "trades": 1,
                    "long_trades": 1,
                    "short_trades": 0,
                    "pnl": 125.0,
                    "pnl_share_pct": 100.0,
                    "long_pnl": 125.0,
                    "short_pnl": 0.0,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            out = StringIO()
            call_command("generate_session_turtle_plots", f"--output-dir={temp_dir}", stdout=out)

            expected = [
                "shared_account_summary.csv",
                "shared_account_equity_curve.csv",
                "shared_account_trades.csv",
                "shared_account_yearly_returns.csv",
                "shared_account_asset_summary.csv",
                "shared_account_equity_drawdown.png",
                "shared_account_yearly_pnl.png",
                "shared_account_asset_pnl.png",
            ]
            for name in expected:
                self.assertTrue((Path(temp_dir) / name).exists(), name)
                self.assertGreater((Path(temp_dir) / name).stat().st_size, 0, name)

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_drawdown_governor_settings(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend x2 With DD Governor",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--exposure-mult=2",
                "--use-drawdown-governor",
                "--drawdown-trigger-1-pct=10",
                "--drawdown-exposure-mult-1=1.5",
                "--drawdown-trigger-2-pct=20",
                "--drawdown-exposure-mult-2=1.0",
            )

        self.assertEqual(mock_report.call_args.kwargs["exposure_mult"], 2.0)
        self.assertTrue(mock_report.call_args.kwargs["use_drawdown_governor"])
        self.assertEqual(mock_report.call_args.kwargs["drawdown_trigger_1_pct"], 10.0)
        self.assertEqual(mock_report.call_args.kwargs["drawdown_exposure_mult_1"], 1.5)
        self.assertEqual(mock_report.call_args.kwargs["drawdown_trigger_2_pct"], 20.0)
        self.assertEqual(mock_report.call_args.kwargs["drawdown_exposure_mult_2"], 1.0)

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_asset_class_caps(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend x2 With Asset Class Caps",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--crypto-cap-mult=1.0",
                "--gold-cap-mult=1.0",
                "--metals-cap-mult=1.0",
                "--equity-cap-mult=1.0",
            )

        self.assertEqual(mock_report.call_args.kwargs["crypto_cap_mult"], 1.0)
        self.assertEqual(mock_report.call_args.kwargs["gold_cap_mult"], 1.0)
        self.assertEqual(mock_report.call_args.kwargs["metals_cap_mult"], 1.0)
        self.assertEqual(mock_report.call_args.kwargs["equity_cap_mult"], 1.0)

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_risk_and_stop_settings(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend x10 Custom Risk",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--exposure-mult=10",
                "--base-risk-pct=1",
                "--fixed-stop-pct=5",
                "--directional-volume-risk-pct=1.4",
            )

        self.assertEqual(mock_report.call_args.kwargs["exposure_mult"], 10.0)
        self.assertEqual(mock_report.call_args.kwargs["base_risk_pct"], 0.01)
        self.assertEqual(mock_report.call_args.kwargs["fixed_stop_pct"], 0.05)
        self.assertAlmostEqual(mock_report.call_args.kwargs["directional_volume_risk_pct"], 0.014)

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_basket(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend Core x2",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--basket=core",
            )

        self.assertEqual(mock_report.call_args.kwargs["basket"], "core")

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_performance_leadership_overlay(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend Core x2 With Leadership Overlay",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--use-performance-leadership-overlay",
                "--performance-lookback-trades=8",
                "--performance-decay=0.8",
                "--performance-floor-mult=0.7",
                "--performance-cap-mult=1.3",
                "--performance-min-history=2",
            )

        self.assertTrue(mock_report.call_args.kwargs["use_performance_leadership_overlay"])
        self.assertEqual(mock_report.call_args.kwargs["performance_lookback_trades"], 8)
        self.assertEqual(mock_report.call_args.kwargs["performance_decay"], 0.8)
        self.assertEqual(mock_report.call_args.kwargs["performance_floor_mult"], 0.7)
        self.assertEqual(mock_report.call_args.kwargs["performance_cap_mult"], 1.3)
        self.assertEqual(mock_report.call_args.kwargs["performance_min_history"], 2)

    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_extended_hours_protective_exits(self, mock_report):
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend Core x2 With Extended Hours Protective Exits",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--use-extended-hours-protective-exits",
                "--extended-hours-core-session-minutes=390",
            )

        self.assertTrue(mock_report.call_args.kwargs["use_extended_hours_protective_exits"])
        self.assertEqual(mock_report.call_args.kwargs["extended_hours_core_session_minutes"], 390)

    @patch("edgar.management.commands.generate_session_turtle_plots.load_vix_scores")
    @patch("edgar.management.commands.generate_session_turtle_plots.generate_session_turtle_shared_account_report")
    def test_generate_session_turtle_plots_forwards_sentiment_governor_settings(
        self,
        mock_report,
        mock_load_vix_scores,
    ):
        mock_load_vix_scores.return_value = {"2024-01-02": 22.0}
        mock_report.return_value = {
            "summary": {
                "label": "Session Turtle Trend Core x2 With Sentiment Governor",
                "initial_capital": 1000.0,
            },
            "equity_curve": [
                {"date": "2024-01-02T10:00:00", "equity": 1000.0},
                {"date": "2024-01-10T10:00:00", "equity": 1125.0},
            ],
            "trades": [{"ticker": "BTC-USD"}],
            "yearly_returns": [{"year": 2024, "pnl": 125.0}],
            "asset_summary": [{"ticker": "BTC-USD", "pnl": 125.0}],
        }

        with TemporaryDirectory() as temp_dir:
            call_command(
                "generate_session_turtle_plots",
                f"--output-dir={temp_dir}",
                "--use-sentiment-governor",
                "--sentiment-source=vix",
                "--sentiment-lag-days=1",
                "--sentiment-threshold-1=50",
                "--sentiment-threshold-2=30",
                "--sentiment-mult-1=1.25",
                "--sentiment-mult-2=0.75",
                "--sentiment-reversal-window=7",
                "--sentiment-reversal-min-rise=8",
                "--sentiment-reversal-mult=1.0",
            )

        mock_load_vix_scores.assert_called_once_with()
        self.assertTrue(mock_report.call_args.kwargs["use_sentiment_governor"])
        self.assertEqual(mock_report.call_args.kwargs["sentiment_scores"], {"2024-01-02": 22.0})
        self.assertEqual(mock_report.call_args.kwargs["sentiment_lag_days"], 1)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_threshold_1"], 50.0)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_threshold_2"], 30.0)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_exposure_mult_1"], 1.25)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_exposure_mult_2"], 0.75)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_reversal_window"], 7)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_reversal_min_rise"], 8.0)
        self.assertEqual(mock_report.call_args.kwargs["sentiment_reversal_mult"], 1.0)


class SessionTurtlePortfolioTests(TestCase):
    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_supports_performance_leadership_overlay(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("AAA", "tiingo", "new_york_equity_open"),
            ("BBB", "tiingo", "new_york_equity_open"),
        )

        def _trade(entry_date: str, exit_date: str, pnl: float) -> dict:
            return {
                "direction": "long",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": pnl,
                "risk_model": "directional_volume_boost",
                "entry_rel_volume": 1.5,
            }

        payloads = {
            "AAA": {
                "trades": [
                    _trade("2024-01-01T10:00:00", "2024-01-02T10:00:00", 20.0),
                    _trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0),
                    _trade("2024-01-05T10:00:00", "2024-01-06T10:00:00", 10.0),
                ]
            },
            "BBB": {
                "trades": [
                    _trade("2024-01-01T10:00:00", "2024-01-02T10:00:00", -20.0),
                    _trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", -20.0),
                    _trade("2024-01-05T10:00:00", "2024-01-06T10:00:00", 10.0),
                ]
            },
        }
        mock_backtest.side_effect = lambda ticker, **kwargs: payloads[ticker]

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            base_portfolio_cap_pct=0.9,
            use_performance_leadership_overlay=True,
            performance_lookback_trades=2,
            performance_decay=1.0,
            performance_floor_mult=0.5,
            performance_cap_mult=1.5,
            performance_min_history=2,
        )

        trades = report["trades"]
        self.assertEqual(len(trades), 6)

        aaa_last = next(
            trade for trade in trades if trade["ticker"] == "AAA" and trade["entry_ts"] == "2024-01-05T10:00:00"
        )
        bbb_last = next(
            trade for trade in trades if trade["ticker"] == "BBB" and trade["entry_ts"] == "2024-01-05T10:00:00"
        )
        self.assertEqual(aaa_last["performance_risk_mult"], 1.5)
        self.assertEqual(bbb_last["performance_risk_mult"], 0.5)
        self.assertEqual(aaa_last["notional"], 150.0)
        self.assertEqual(bbb_last["notional"], 50.0)
        self.assertEqual(aaa_last["performance_rank_pct"], 1.0)
        self.assertEqual(bbb_last["performance_rank_pct"], 0.0)

        summary = report["summary"]
        self.assertTrue(summary["use_performance_leadership_overlay"])
        self.assertEqual(summary["entries_performance_upscaled"], 1)
        self.assertEqual(summary["entries_performance_downscaled"], 1)
        self.assertIsNone(summary["performance_bucket_scopes"])

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_supports_bucket_scoped_performance_leadership_overlay(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("BTC-USD", "binance", "new_york_equity_open"),
            ("ETH-USD", "binance", "new_york_equity_open"),
            ("AMZN", "tiingo", "new_york_equity_open"),
        )

        def _trade(entry_date: str, exit_date: str, pnl: float) -> dict:
            return {
                "direction": "long",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": pnl,
                "risk_model": "directional_volume_boost",
                "entry_rel_volume": 1.5,
            }

        payloads = {
            "BTC-USD": {
                "trades": [
                    _trade("2024-01-01T10:00:00", "2024-01-02T10:00:00", 20.0),
                    _trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0),
                    _trade("2024-01-05T10:00:00", "2024-01-06T10:00:00", 10.0),
                ]
            },
            "ETH-USD": {
                "trades": [
                    _trade("2024-01-01T10:00:00", "2024-01-02T10:00:00", -20.0),
                    _trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", -20.0),
                    _trade("2024-01-05T10:00:00", "2024-01-06T10:00:00", 10.0),
                ]
            },
            "AMZN": {
                "trades": [
                    _trade("2024-01-01T10:00:00", "2024-01-02T10:00:00", 15.0),
                    _trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 15.0),
                    _trade("2024-01-05T10:00:00", "2024-01-06T10:00:00", 15.0),
                ]
            },
        }
        mock_backtest.side_effect = lambda ticker, **kwargs: payloads[ticker]

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            base_portfolio_cap_pct=0.9,
            use_performance_leadership_overlay=True,
            performance_lookback_trades=2,
            performance_decay=1.0,
            performance_floor_mult=0.5,
            performance_cap_mult=1.5,
            performance_min_history=2,
            performance_bucket_scopes=frozenset({"crypto"}),
        )

        trades = report["trades"]
        btc_last = next(
            trade for trade in trades if trade["ticker"] == "BTC-USD" and trade["entry_ts"] == "2024-01-05T10:00:00"
        )
        eth_last = next(
            trade for trade in trades if trade["ticker"] == "ETH-USD" and trade["entry_ts"] == "2024-01-05T10:00:00"
        )
        amzn_last = next(
            trade for trade in trades if trade["ticker"] == "AMZN" and trade["entry_ts"] == "2024-01-05T10:00:00"
        )

        self.assertEqual(btc_last["performance_risk_mult"], 1.5)
        self.assertEqual(eth_last["performance_risk_mult"], 0.5)
        self.assertEqual(amzn_last["performance_risk_mult"], 1.0)
        self.assertEqual(amzn_last["performance_rank_pct"], None)
        self.assertEqual(amzn_last["performance_peer_count"], 0)

        summary = report["summary"]
        self.assertEqual(summary["performance_bucket_scopes"], ["crypto"])

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_forwards_extended_hours_protective_exits_to_tiingo_assets(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("AAA", "tiingo", "new_york_equity_open"),
            ("BTC-USD", "binance", "new_york_equity_open"),
        )
        mock_backtest.return_value = {"trades": []}

        generate_session_turtle_shared_account_report(
            basket="core",
            use_extended_hours_protective_exits=True,
            extended_hours_core_session_minutes=390,
        )

        first_call = mock_backtest.call_args_list[0].kwargs
        second_call = mock_backtest.call_args_list[1].kwargs
        self.assertTrue(first_call["use_extended_hours_protective_exits_only"])
        self.assertEqual(first_call["entry_window_minutes"], 390)
        self.assertEqual(first_call["core_session_minutes"], 390)
        self.assertFalse(second_call["use_extended_hours_protective_exits_only"])
        self.assertEqual(second_call["entry_window_minutes"], 480)
        self.assertIsNone(second_call["core_session_minutes"])

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_lags_sentiment_to_avoid_forward_bias(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (("AAA", "tiingo", "new_york_equity_open"),)
        mock_backtest.return_value = {
            "trades": [
                {
                    "direction": "long",
                    "entry_date": "2024-01-03T10:00:00",
                    "exit_date": "2024-01-04T10:00:00",
                    "entry_price": 100.0,
                    "exit_price": 120.0,
                    "shares": 1.0,
                    "position_size": 100.0,
                    "pnl": 20.0,
                    "risk_model": "directional_volume_boost",
                    "entry_rel_volume": 1.5,
                }
            ]
        }

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            use_sentiment_governor=True,
            sentiment_scores={
                "2024-01-02": 20.0,
                "2024-01-03": 80.0,
            },
            sentiment_lag_days=1,
            sentiment_threshold_1=45.0,
            sentiment_threshold_2=25.0,
            sentiment_exposure_mult_1=1.0,
            sentiment_exposure_mult_2=0.5,
            sentiment_reversal_window=0,
        )

        trade = report["trades"][0]
        self.assertEqual(trade["entry_sentiment_score"], 20.0)
        self.assertEqual(trade["entry_exposure_mult"], 0.5)
        self.assertEqual(report["summary"]["sentiment_lag_days"], 1)

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_supports_direct_bucket_sentiment_sizing_for_crypto(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("BTC-USD", "binance", "new_york_equity_open"),
            ("AMZN", "tiingo", "new_york_equity_open"),
        )

        def _trade(entry_date: str, exit_date: str, pnl: float) -> dict:
            return {
                "direction": "long",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": pnl,
                "risk_model": "directional_volume_boost",
                "entry_rel_volume": 1.5,
            }

        payloads = {
            "BTC-USD": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0)]},
            "AMZN": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0)]},
        }
        mock_backtest.side_effect = lambda ticker, **kwargs: payloads[ticker]

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            use_sentiment_governor=True,
            sentiment_scores={"2024-01-02": 80.0},
            sentiment_lag_days=1,
            use_direct_bucket_sentiment_sizing=True,
            bucket_sentiment_scores={
                "crypto": {
                    "2024-01-02": 20.0,
                    "2024-01-03": 80.0,
                }
            },
            bucket_sentiment_lag_days=1,
            bucket_sentiment_threshold_1=45.0,
            bucket_sentiment_threshold_2=25.0,
            bucket_sentiment_size_mult_1=0.75,
            bucket_sentiment_size_mult_2=0.5,
            bucket_sentiment_reversal_window=0,
        )

        btc_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BTC-USD")
        amzn_trade = next(trade for trade in report["trades"] if trade["ticker"] == "AMZN")

        self.assertEqual(btc_trade["direct_sentiment_score"], 20.0)
        self.assertEqual(btc_trade["direct_sentiment_size_mult"], 0.5)
        self.assertEqual(btc_trade["notional"], 50.0)

        self.assertIsNone(amzn_trade["direct_sentiment_score"])
        self.assertEqual(amzn_trade["direct_sentiment_size_mult"], 1.0)
        self.assertEqual(amzn_trade["notional"], 100.0)

        summary = report["summary"]
        self.assertTrue(summary["use_direct_bucket_sentiment_sizing"])
        self.assertEqual(summary["entries_direct_sentiment_downscaled"], 1)

    def test_intraday_volatility_proxy_lookup_uses_prior_completed_bar(self):
        from edgar.services.session_turtle_portfolio import _lookup_intraday_volatility_signal

        proxy_state = {
            "timestamps": [
                datetime(2024, 1, 3, 9, 55, 0),
                datetime(2024, 1, 3, 10, 0, 0),
            ],
            "closes": [31.0, 28.0],
            "sma_short": [30.0, 29.0],
            "sma_long": [29.0, 30.0],
            "interval_minutes": 5,
        }

        regime, mult, age_min, close_value = _lookup_intraday_volatility_signal(
            entry_ts=datetime(2024, 1, 3, 10, 0, 0),
            session_open="new_york_equity_open",
            asset_bucket="equity",
            direction="long",
            proxy_state=proxy_state,
            max_age_minutes=60,
            lag_bars=1,
            allowed_buckets=None,
            long_risk_on_mult=1.0,
            long_neutral_mult=1.0,
            long_risk_off_mult=0.5,
            short_risk_on_mult=0.5,
            short_neutral_mult=1.0,
            short_risk_off_mult=1.0,
        )

        self.assertEqual(regime, "risk_off_micro")
        self.assertEqual(mult, 0.5)
        self.assertEqual(age_min, 5.0)
        self.assertEqual(close_value, 31.0)

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_supports_intraday_volatility_proxy_overlay(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("AAA", "tiingo", "new_york_equity_open"),
            ("BBB", "tiingo", "new_york_equity_open"),
            ("BTC-USD", "binance", "hong_kong_open"),
        )

        def _trade(entry_date: str, exit_date: str, pnl: float, direction: str) -> dict:
            return {
                "direction": direction,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": pnl,
                "risk_model": "directional_volume_boost",
                "entry_rel_volume": 1.5,
            }

        payloads = {
            "AAA": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0, "long")]},
            "BBB": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0, "short")]},
            "BTC-USD": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0, "short")]},
        }
        mock_backtest.side_effect = lambda ticker, **kwargs: payloads[ticker]

        proxy_state = {
            "timestamps": [datetime(2024, 1, 3, 9, 55, 0)],
            "closes": [31.0],
            "sma_short": [30.0],
            "sma_long": [29.0],
            "interval_minutes": 5,
        }

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            use_intraday_volatility_proxy=True,
            intraday_volatility_proxy_state=proxy_state,
            intraday_volatility_proxy_label="VIXY",
            intraday_volatility_proxy_max_age_minutes=60,
            intraday_volatility_proxy_lag_bars=1,
            intraday_volatility_long_risk_on_mult=1.0,
            intraday_volatility_long_neutral_mult=1.0,
            intraday_volatility_long_risk_off_mult=0.5,
            intraday_volatility_short_risk_on_mult=0.5,
            intraday_volatility_short_neutral_mult=1.0,
            intraday_volatility_short_risk_off_mult=1.0,
        )

        aaa_trade = next(trade for trade in report["trades"] if trade["ticker"] == "AAA")
        bbb_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BBB")
        btc_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BTC-USD")

        self.assertEqual(aaa_trade["intraday_vol_proxy_regime"], "risk_off_micro")
        self.assertEqual(aaa_trade["intraday_vol_proxy_mult"], 0.5)
        self.assertEqual(aaa_trade["notional"], 50.0)

        self.assertEqual(bbb_trade["intraday_vol_proxy_regime"], "risk_off_micro")
        self.assertEqual(bbb_trade["intraday_vol_proxy_mult"], 1.0)
        self.assertEqual(bbb_trade["notional"], 100.0)

        self.assertIsNone(btc_trade["intraday_vol_proxy_regime"])
        self.assertEqual(btc_trade["intraday_vol_proxy_mult"], 1.0)
        self.assertEqual(btc_trade["notional"], 100.0)

        summary = report["summary"]
        self.assertTrue(summary["use_intraday_volatility_proxy"])
        self.assertEqual(summary["intraday_volatility_proxy_label"], "VIXY")
        self.assertEqual(summary["entries_intraday_volatility_proxy_scaled"], 1)
        self.assertEqual(summary["entries_intraday_volatility_risk_off_micro"], 2)

    def test_lookup_volatility_persistence_signal_uses_previous_day_vix_and_prior_bar(self):
        from edgar.services.session_turtle_portfolio import _lookup_volatility_persistence_signal

        regime, mult, age_min, vix_rel, vixy_rel, ratio = _lookup_volatility_persistence_signal(
            entry_ts=datetime(2024, 1, 3, 10, 0, 0),
            session_open="new_york_equity_open",
            direction="long",
            daily_vix_state={
                "dates": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "closes": [18.0, 24.0, 12.0],
                "ema": [18.0, 20.0, 16.0],
            },
            intraday_vixy_state={
                "timestamps": [datetime(2024, 1, 3, 9, 55, 0), datetime(2024, 1, 3, 10, 0, 0)],
                "closes": [26.0, 10.0],
                "ema": [20.0, 18.0],
                "interval_minutes": 5,
            },
            daily_lag_days=1,
            intraday_max_age_minutes=60,
            intraday_lag_bars=1,
            ratio_upper=1.05,
            ratio_lower=0.95,
            daily_stress_min_rel=1.0,
            long_persistent_stress_mult=0.5,
            long_neutral_mult=1.0,
            long_fading_stress_mult=1.0,
            short_persistent_stress_mult=1.0,
            short_neutral_mult=1.0,
            short_fading_stress_mult=0.5,
        )

        self.assertEqual(regime, "persistent_stress")
        self.assertEqual(mult, 0.5)
        self.assertEqual(age_min, 5.0)
        self.assertAlmostEqual(vix_rel, 1.2)
        self.assertAlmostEqual(vixy_rel, 1.3)
        self.assertAlmostEqual(ratio, 1.3 / 1.2)

    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_session_turtle_portfolio_supports_volatility_persistence_overlay(
        self,
        mock_backtest,
        mock_resolve_universe,
    ):
        mock_resolve_universe.return_value = (
            ("AAA", "tiingo", "new_york_equity_open"),
            ("BBB", "tiingo", "new_york_equity_open"),
            ("CCC", "tiingo", "new_york_equity_open"),
        )

        def _trade(entry_date: str, exit_date: str, pnl: float, direction: str) -> dict:
            return {
                "direction": direction,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": pnl,
                "risk_model": "directional_volume_boost",
                "entry_rel_volume": 1.5,
            }

        payloads = {
            "AAA": {"trades": [_trade("2024-01-03T10:00:00", "2024-01-04T10:00:00", 20.0, "long")]},
            "BBB": {"trades": [_trade("2024-01-03T11:00:00", "2024-01-04T11:00:00", 20.0, "short")]},
            "CCC": {"trades": [_trade("2024-01-03T12:00:00", "2024-01-04T12:00:00", 20.0, "long")]},
        }
        mock_backtest.side_effect = lambda ticker, **kwargs: payloads[ticker]

        daily_state = {
            "dates": [date(2024, 1, 1), date(2024, 1, 2)],
            "closes": [18.0, 24.0],
            "ema": [18.0, 20.0],
        }
        intraday_state = {
            "timestamps": [
                datetime(2024, 1, 3, 9, 55, 0),
                datetime(2024, 1, 3, 10, 55, 0),
                datetime(2024, 1, 3, 11, 55, 0),
            ],
            "closes": [26.0, 18.0, 24.6],
            "ema": [20.0, 20.0, 20.0],
            "interval_minutes": 5,
        }

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=2.0,
            use_volatility_persistence_overlay=True,
            daily_vix_reference_state=daily_state,
            intraday_vixy_relative_state=intraday_state,
            volatility_persistence_daily_lag_days=1,
            volatility_persistence_intraday_max_age_minutes=60,
            volatility_persistence_intraday_lag_bars=1,
            volatility_persistence_ratio_upper=1.05,
            volatility_persistence_ratio_lower=0.95,
            volatility_persistence_daily_stress_min_rel=1.0,
            volatility_persistence_long_persistent_stress_mult=0.5,
            volatility_persistence_long_neutral_mult=1.0,
            volatility_persistence_long_fading_stress_mult=1.0,
            volatility_persistence_short_persistent_stress_mult=1.0,
            volatility_persistence_short_neutral_mult=1.0,
            volatility_persistence_short_fading_stress_mult=0.5,
        )

        aaa_trade = next(trade for trade in report["trades"] if trade["ticker"] == "AAA")
        bbb_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BBB")
        ccc_trade = next(trade for trade in report["trades"] if trade["ticker"] == "CCC")

        self.assertEqual(aaa_trade["volatility_persistence_regime"], "persistent_stress")
        self.assertEqual(aaa_trade["volatility_persistence_mult"], 0.5)
        self.assertEqual(aaa_trade["notional"], 50.0)

        self.assertEqual(bbb_trade["volatility_persistence_regime"], "fading_stress")
        self.assertEqual(bbb_trade["volatility_persistence_mult"], 0.5)
        self.assertEqual(bbb_trade["notional"], 50.0)

        self.assertEqual(ccc_trade["volatility_persistence_regime"], "neutral_persistence")
        self.assertEqual(ccc_trade["volatility_persistence_mult"], 1.0)
        self.assertEqual(ccc_trade["notional"], 100.0)

        summary = report["summary"]
        self.assertTrue(summary["use_volatility_persistence_overlay"])
        self.assertEqual(summary["entries_volatility_persistence_scaled"], 2)
        self.assertEqual(summary["entries_volatility_persistence_persistent_stress"], 1)
        self.assertEqual(summary["entries_volatility_persistence_fading_stress"], 1)

    def test_tightening_liquidity_gate_uses_slow_macro_state_for_long_sizing(self):
        from edgar.services.tightening_liquidity_gate import (
            build_tightening_liquidity_state,
            lookup_tightening_liquidity_signal,
        )

        state = build_tightening_liquidity_state(
            curve_series={
                "2024-01-01": 1.0,
                "2024-01-02": 0.4,
                "2024-01-03": -0.1,
            },
            credit_spread_series={
                "2024-01-01": 4.0,
                "2024-01-02": 4.1,
                "2024-01-03": 5.5,
            },
            nfci_series={
                "2024-01-01": -0.5,
                "2024-01-02": -0.4,
                "2024-01-03": 0.3,
            },
            vix_series={
                "2024-01-01": 15.0,
                "2024-01-02": 15.5,
                "2024-01-03": 25.0,
            },
            ma_days=2,
            tight_score_threshold=2,
        )

        regime, mult, score, blocked = lookup_tightening_liquidity_signal(
            entry_ts=datetime(2024, 1, 4, 10, 0, 0),
            asset_bucket="equity",
            direction="long",
            state=state,
            lag_days=1,
            gated_buckets=frozenset({"equity"}),
            mode="size",
            tight_long_mult=0.5,
            neutral_long_mult=1.0,
            tight_short_mult=1.0,
            neutral_short_mult=1.0,
        )

        self.assertEqual(regime, "tightening_liquidity_tight")
        self.assertEqual(mult, 0.5)
        self.assertEqual(score, 4)
        self.assertFalse(blocked)

    def test_session_turtle_portfolio_supports_tightening_liquidity_size_gate(self):
        from edgar.services.tightening_liquidity_gate import build_tightening_liquidity_state

        candidates = [
            {
                "combo_idx": 0,
                "trade_idx": 0,
                "ticker": "AAA",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 4, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 5, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 120.0,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": 20.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
            {
                "combo_idx": 1,
                "trade_idx": 0,
                "ticker": "BBB",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "short",
                "entry_ts": datetime(2024, 1, 4, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 5, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 90.0,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": 10.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
            {
                "combo_idx": 2,
                "trade_idx": 0,
                "ticker": "GLD",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 4, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 5, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 110.0,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": 10.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "gold",
            },
        ]
        gate_state = build_tightening_liquidity_state(
            curve_series={"2024-01-01": 1.0, "2024-01-02": 0.4, "2024-01-03": -0.1},
            credit_spread_series={"2024-01-01": 4.0, "2024-01-02": 4.1, "2024-01-03": 5.5},
            nfci_series={"2024-01-01": -0.5, "2024-01-02": -0.4, "2024-01-03": 0.3},
            vix_series={"2024-01-01": 15.0, "2024-01-02": 15.5, "2024-01-03": 25.0},
            ma_days=2,
            tight_score_threshold=2,
        )

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=1.0,
            drawdown_exposure_mult_1=1.0,
            drawdown_exposure_mult_2=1.0,
            use_tightening_liquidity_gate=True,
            tightening_liquidity_state=gate_state,
            tightening_liquidity_mode="size",
            tightening_liquidity_lag_days=1,
            tightening_liquidity_buckets=frozenset({"equity"}),
            tightening_liquidity_long_tight_mult=0.5,
            tightening_liquidity_short_tight_mult=1.0,
            precomputed_candidates=candidates,
        )

        aaa_trade = next(trade for trade in report["trades"] if trade["ticker"] == "AAA")
        bbb_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BBB")
        gld_trade = next(trade for trade in report["trades"] if trade["ticker"] == "GLD")

        self.assertEqual(aaa_trade["tightening_liquidity_regime"], "tightening_liquidity_tight")
        self.assertEqual(aaa_trade["tightening_liquidity_mult"], 0.5)
        self.assertEqual(aaa_trade["notional"], 50.0)
        self.assertEqual(bbb_trade["tightening_liquidity_mult"], 1.0)
        self.assertEqual(bbb_trade["notional"], 100.0)
        self.assertIsNone(gld_trade["tightening_liquidity_regime"])
        self.assertEqual(gld_trade["notional"], 100.0)

        summary = report["summary"]
        self.assertTrue(summary["use_tightening_liquidity_gate"])
        self.assertEqual(summary["entries_tightening_liquidity_scaled"], 1)
        self.assertEqual(summary["entries_tightening_liquidity_tight"], 2)
        self.assertEqual(summary["skipped_tightening_liquidity_gate"], 0)

    def test_session_turtle_portfolio_supports_tightening_liquidity_entry_gate(self):
        from edgar.services.tightening_liquidity_gate import build_tightening_liquidity_state

        candidates = [
            {
                "combo_idx": 0,
                "trade_idx": 0,
                "ticker": "AAA",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 4, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 5, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 120.0,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": 20.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
            {
                "combo_idx": 1,
                "trade_idx": 0,
                "ticker": "BBB",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "short",
                "entry_ts": datetime(2024, 1, 4, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 5, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 90.0,
                "shares": 1.0,
                "position_size": 100.0,
                "pnl": 10.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
        ]
        gate_state = build_tightening_liquidity_state(
            curve_series={"2024-01-01": 1.0, "2024-01-02": 0.4, "2024-01-03": -0.1},
            credit_spread_series={"2024-01-01": 4.0, "2024-01-02": 4.1, "2024-01-03": 5.5},
            nfci_series={"2024-01-01": -0.5, "2024-01-02": -0.4, "2024-01-03": 0.3},
            vix_series={"2024-01-01": 15.0, "2024-01-02": 15.5, "2024-01-03": 25.0},
            ma_days=2,
            tight_score_threshold=2,
        )

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=1.0,
            drawdown_exposure_mult_1=1.0,
            drawdown_exposure_mult_2=1.0,
            use_tightening_liquidity_gate=True,
            tightening_liquidity_state=gate_state,
            tightening_liquidity_mode="entry",
            tightening_liquidity_lag_days=1,
            tightening_liquidity_buckets=frozenset({"equity"}),
            tightening_liquidity_short_tight_mult=1.0,
            precomputed_candidates=candidates,
        )

        tickers = {trade["ticker"] for trade in report["trades"]}
        self.assertEqual(tickers, {"BBB"})
        self.assertEqual(report["trades"][0]["tightening_liquidity_regime"], "tightening_liquidity_tight")

        summary = report["summary"]
        self.assertEqual(summary["executed_trades"], 1)
        self.assertEqual(summary["skipped_tightening_liquidity_gate"], 1)
        self.assertEqual(summary["entries_tightening_liquidity_tight"], 1)

    def test_macro_regime_score_blocks_risk_asset_longs_when_macro_is_negative(self):
        from edgar.services.macro_regime_score import lookup_macro_regime_signal

        state = {
            "dates": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "scores": [None, 0, -3],
            "raw_scores": [None, 0, -4],
            "labels": [None, "macro_neutral", "macro_headwind"],
            "component_scores": {
                "dollar": [None, 0, -1],
                "rates": [None, 0, -1],
                "stress": [None, 0, -1],
                "liquidity": [None, 0, -1],
            },
        }

        long_signal = lookup_macro_regime_signal(
            entry_ts=datetime(2024, 1, 4, 10, 0, 0),
            asset_bucket="equity",
            direction="long",
            state=state,
            lag_days=1,
            gated_buckets=frozenset({"equity"}),
            long_half_mult=0.5,
            negative_long_mult=0.0,
            short_half_mult=0.5,
            positive_short_mult=0.0,
        )
        short_signal = lookup_macro_regime_signal(
            entry_ts=datetime(2024, 1, 4, 10, 0, 0),
            asset_bucket="equity",
            direction="short",
            state=state,
            lag_days=1,
            gated_buckets=frozenset({"equity"}),
            long_half_mult=0.5,
            negative_long_mult=0.0,
            short_half_mult=0.5,
            positive_short_mult=0.0,
        )

        self.assertEqual(long_signal["score"], -3)
        self.assertTrue(long_signal["blocked"])
        self.assertEqual(long_signal["action"], "blocked_long")
        self.assertEqual(short_signal["score"], -3)
        self.assertFalse(short_signal["blocked"])
        self.assertEqual(short_signal["action"], "full_short")

    def test_macro_regime_score_v2_front_end_uses_two_year_curve_and_vix_term(self):
        from edgar.services.macro_regime_score import build_macro_regime_score_state

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            combined_path = base_dir / "macro.csv"
            fred_dir = base_dir / "fred"
            fred_dir.mkdir()

            combined_path.write_text(
                "\n".join(
                    [
                        "date,dxy_close,us_2y_yield,us_10y_yield,vix3m_level",
                        "2024-01-01,100,5.0,4.7,20",
                        "2024-01-02,99,4.9,4.7,21",
                        "2024-01-03,98,4.6,4.8,23",
                    ]
                ),
                encoding="utf-8",
            )
            (fred_dir / "T10Y3M.csv").write_text(
                "\n".join(
                    [
                        "observation_date,T10Y3M",
                        "2024-01-01,-0.3",
                        "2024-01-02,0.0",
                        "2024-01-03,0.2",
                    ]
                ),
                encoding="utf-8",
            )
            (fred_dir / "BAMLH0A0HYM2.csv").write_text(
                "\n".join(
                    [
                        "observation_date,BAMLH0A0HYM2",
                        "2024-01-01,4.0",
                        "2024-01-02,3.9",
                        "2024-01-03,3.8",
                    ]
                ),
                encoding="utf-8",
            )
            (fred_dir / "NFCI.csv").write_text(
                "\n".join(
                    [
                        "observation_date,NFCI",
                        "2024-01-01,0.5",
                        "2024-01-02,0.4",
                        "2024-01-03,0.2",
                    ]
                ),
                encoding="utf-8",
            )
            (fred_dir / "VIXCLS.csv").write_text(
                "\n".join(
                    [
                        "observation_date,VIXCLS",
                        "2024-01-01,18",
                        "2024-01-02,17",
                        "2024-01-03,16",
                    ]
                ),
                encoding="utf-8",
            )

            state = build_macro_regime_score_state(
                combined_macro_path=combined_path,
                fred_dir=fred_dir,
                version="v2_front_end",
                dxy_fast_ma_days=2,
                dxy_slow_ma_days=3,
                rates_ma_days=2,
                stress_ma_days=2,
                lookback_days=1,
                rates_change_threshold_bps=5.0,
                front_end_rates_change_threshold_bps=10.0,
                score_cap=3,
            )

        self.assertEqual(state["version"], "v2_front_end")
        self.assertEqual(state["component_scores"]["dollar"][-1], 1)
        self.assertEqual(state["component_scores"]["rates"][-1], 1)
        self.assertEqual(state["component_scores"]["stress"][-1], 1)
        self.assertEqual(state["component_scores"]["liquidity"][-1], 1)
        self.assertEqual(state["raw_scores"][-1], 4)
        self.assertEqual(state["scores"][-1], 3)
        self.assertEqual(state["labels"][-1], "macro_tailwind")

    def test_session_turtle_portfolio_recomputes_position_size_from_live_capital(self):
        candidates = [
            {
                "combo_idx": 0,
                "trade_idx": 0,
                "ticker": "AAA",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 1, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 2, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 80.0,
                "stop_loss": 90.0,
                "risk_pct": 0.05,
                "shares": 5.0,
                "position_size": 500.0,
                "pnl": -100.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
            {
                "combo_idx": 1,
                "trade_idx": 0,
                "ticker": "BBB",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 3, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 4, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 120.0,
                "stop_loss": 90.0,
                "risk_pct": 0.05,
                "shares": 7.0,
                "position_size": 700.0,
                "pnl": 140.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
        ]

        report = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=1.0,
            drawdown_exposure_mult_1=1.0,
            drawdown_exposure_mult_2=1.0,
            precomputed_candidates=candidates,
        )

        aaa_trade = next(trade for trade in report["trades"] if trade["ticker"] == "AAA")
        bbb_trade = next(trade for trade in report["trades"] if trade["ticker"] == "BBB")
        self.assertEqual(aaa_trade["notional"], 500.0)
        self.assertEqual(bbb_trade["notional"], 450.0)
        self.assertEqual(report["summary"]["final_equity"], 990.0)

    def test_session_turtle_portfolio_is_order_invariant_for_same_timestamp_capacity(self):
        base_candidates = [
            {
                "combo_idx": 0,
                "trade_idx": 0,
                "ticker": "AAA",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 1, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 2, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 120.0,
                "stop_loss": 90.0,
                "risk_pct": 0.06,
                "shares": 6.0,
                "position_size": 600.0,
                "pnl": 120.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
            {
                "combo_idx": 1,
                "trade_idx": 0,
                "ticker": "BBB",
                "source": "tiingo",
                "session_open": "new_york_equity_open",
                "direction": "long",
                "entry_ts": datetime(2024, 1, 1, 10, 0, 0),
                "exit_ts": datetime(2024, 1, 2, 10, 0, 0),
                "entry_price": 100.0,
                "exit_price": 100.0,
                "stop_loss": 90.0,
                "risk_pct": 0.06,
                "shares": 6.0,
                "position_size": 600.0,
                "pnl": 0.0,
                "risk_model": "base",
                "entry_rel_volume": 1.0,
                "asset_bucket": "equity",
            },
        ]
        reversed_candidates = [dict(candidate, combo_idx=1 - int(candidate["combo_idx"])) for candidate in base_candidates]

        report_a = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=1.0,
            drawdown_exposure_mult_1=1.0,
            drawdown_exposure_mult_2=1.0,
            precomputed_candidates=base_candidates,
        )
        report_b = generate_session_turtle_shared_account_report(
            basket="core",
            initial_capital=1000.0,
            exposure_mult=1.0,
            drawdown_exposure_mult_1=1.0,
            drawdown_exposure_mult_2=1.0,
            precomputed_candidates=reversed_candidates,
        )

        notionals_a = sorted(float(trade["notional"]) for trade in report_a["trades"])
        notionals_b = sorted(float(trade["notional"]) for trade in report_b["trades"])
        self.assertEqual(notionals_a, [450.0, 450.0])
        self.assertEqual(notionals_b, [450.0, 450.0])
        self.assertEqual(report_a["summary"]["final_equity"], 1090.0)
        self.assertEqual(report_a["summary"]["final_equity"], report_b["summary"]["final_equity"])

    @patch("edgar.services.session_turtle_portfolio._market_data_start_timestamp")
    @patch("edgar.services.session_turtle_portfolio._resolve_universe")
    @patch("edgar.services.session_turtle_portfolio.run_session_turtle_trend_backtest")
    def test_build_candidates_filters_investable_universe_asof_date(
        self,
        mock_backtest,
        mock_resolve_universe,
        mock_market_data_start_timestamp,
    ):
        from edgar.services.session_turtle_portfolio import build_session_turtle_shared_account_candidates

        mock_resolve_universe.return_value = (
            ("AAA", "tiingo", "new_york_equity_open"),
            ("BBB", "tiingo", "new_york_equity_open"),
        )
        mock_market_data_start_timestamp.side_effect = lambda ticker, source: {
            "AAA": datetime(2023, 1, 3, 14, 30, 0),
            "BBB": datetime(2025, 1, 2, 14, 30, 0),
        }[ticker]
        mock_backtest.return_value = {
            "trades": [
                {
                    "direction": "long",
                    "entry_date": "2024-01-03T10:00:00",
                    "exit_date": "2024-01-04T10:00:00",
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "stop_loss": 90.0,
                    "risk_pct": 0.05,
                    "shares": 5.0,
                    "position_size": 500.0,
                    "pnl": 50.0,
                    "risk_model": "base",
                    "entry_rel_volume": 1.0,
                }
            ]
        }

        candidates = build_session_turtle_shared_account_candidates(
            basket="core",
            investable_universe_asof=date(2024, 1, 1),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["ticker"], "AAA")
        self.assertEqual(mock_backtest.call_count, 1)

    @patch("edgar.services.session_turtle_portfolio.load_local_tiingo_klines")
    @patch("edgar.services.session_turtle_portfolio.load_local_binance_klines")
    def test_build_per_asset_technical_state_includes_binance_assets(
        self,
        mock_load_binance,
        mock_load_tiingo,
    ):
        from edgar.services.session_turtle_portfolio import build_per_asset_technical_state

        start = datetime(2024, 1, 1, 0, 0, 0)

        def _bars(base_price: float) -> list[dict]:
            rows: list[dict] = []
            for idx in range(160):
                price = base_price + idx * 0.25
                rows.append(
                    {
                        "timestamp": start + timedelta(minutes=15 * idx),
                        "open": price,
                        "high": price + 0.5,
                        "low": price - 0.5,
                        "close": price + 0.2,
                        "volume": 1000.0 + idx,
                    }
                )
            return rows

        mock_load_binance.return_value = (_bars(10.0), "ATOMUSDT")
        mock_load_tiingo.return_value = (_bars(100.0), "AMZN", "cache/cache/tiingo/AMZN_5m.parquet")

        state = build_per_asset_technical_state(
            universe=[
                ("ATOM-USD", "binance", "hong_kong_open"),
                ("AMZN", "tiingo", "new_york_equity_open"),
            ],
            lookback_years=0.2,
            warmup_days=5,
            ema_period=3,
            adx_period=2,
        )

        self.assertIn("ATOM-USD", state["daily_ema"])
        self.assertIn("ATOM-USD", state["4h_adx"])
        self.assertTrue(state["4h_adx"]["ATOM-USD"])
        self.assertIn("AMZN", state["daily_ema"])
        self.assertIn("AMZN", state["4h_adx"])


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
    @patch("edgar.services.strategy.run_backtest")
    def test_strategy_daily_endpoint_passes_chandelier_options(self, mock_backtest):
        mock_backtest.return_value = BacktestResult(
            ticker="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=10000.0,
            final_capital=10100.0,
            total_return_pct=1.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
        )
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "fetch_period": "5y",
            "use_chandelier_exit": True,
            "chandelier_period": 18,
            "chandelier_atr_period": 14,
            "chandelier_atr_mult": 2.5,
            "exit_fill_policy": "target_first",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_backtest.assert_called_once()
        self.assertTrue(mock_backtest.call_args.kwargs["use_chandelier_exit"])
        self.assertEqual(mock_backtest.call_args.kwargs["chandelier_period"], 18)
        self.assertEqual(mock_backtest.call_args.kwargs["chandelier_atr_period"], 14)
        self.assertEqual(mock_backtest.call_args.kwargs["chandelier_atr_mult"], 2.5)
        self.assertEqual(mock_backtest.call_args.kwargs["exit_fill_policy"], "target_first")

    @patch("edgar.services.strategy.run_backtest")
    def test_strategy_daily_endpoint_passes_break_even_options(self, mock_backtest):
        mock_backtest.return_value = BacktestResult(
            ticker="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=10000.0,
            final_capital=10100.0,
            total_return_pct=1.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
        )
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "fetch_period": "5y",
            "use_break_even_stop": True,
            "break_even_trigger_r": 1.25,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_backtest.assert_called_once()
        self.assertTrue(mock_backtest.call_args.kwargs["use_break_even_stop"])
        self.assertEqual(mock_backtest.call_args.kwargs["break_even_trigger_r"], 1.25)

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

    @patch("edgar.services.intraday_strategy.run_intraday_backtest")
    def test_strategy_intraday_endpoint(self, mock_intraday):
        mock_intraday.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "15m",
            "strategy_variant": "fractal_breakout_ema200",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10100,
            "total_return_pct": 1.0,
            "total_trades": 2,
            "long_trades": 1,
            "short_trades": 1,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 50.0,
            "max_drawdown_pct": 2.0,
            "profit_factor": 1.2,
            "cagr_pct": 0.5,
            "avg_trade_return_pct": 0.2,
            "exposure_pct": 10.0,
            "total_fees": 3.5,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "15m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "fractal_breakout_ema200",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["data_mode"], "intraday")
        self.assertEqual(payload["interval"], "15m")
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["strategy_variant"], "fractal_breakout_ema200")
        self.assertIn("total_trades", payload)
        mock_intraday.assert_called_once()
        self.assertEqual(
            mock_intraday.call_args.kwargs["strategy_variant"],
            "fractal_breakout_ema200",
        )
        self.assertEqual(mock_intraday.call_args.kwargs["market_data_source"], "auto")
        self.assertTrue(mock_intraday.call_args.kwargs["auto_adjust_for_yf_limits"])
        self.assertFalse(mock_intraday.call_args.kwargs["use_chandelier_exit"])
        self.assertEqual(mock_intraday.call_args.kwargs["exit_fill_policy"], "stop_first")

    @patch("edgar.services.intraday_strategy.run_intraday_backtest")
    def test_strategy_intraday_endpoint_passes_chandelier_options(self, mock_intraday):
        mock_intraday.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "15m",
            "strategy_variant": "fractal_breakout_ema200",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10100,
            "total_return_pct": 1.0,
            "total_trades": 2,
            "long_trades": 1,
            "short_trades": 1,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 50.0,
            "max_drawdown_pct": 2.0,
            "profit_factor": 1.2,
            "cagr_pct": 0.5,
            "avg_trade_return_pct": 0.2,
            "exposure_pct": 10.0,
            "total_fees": 3.5,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "15m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "fractal_breakout_ema200",
            "use_chandelier_exit": True,
            "chandelier_period": 20,
            "chandelier_atr_period": 10,
            "chandelier_atr_mult": 2.7,
            "exit_fill_policy": "target_first",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_intraday.assert_called_once()
        self.assertTrue(mock_intraday.call_args.kwargs["use_chandelier_exit"])
        self.assertEqual(mock_intraday.call_args.kwargs["chandelier_period"], 20)
        self.assertEqual(mock_intraday.call_args.kwargs["chandelier_atr_period"], 10)
        self.assertEqual(mock_intraday.call_args.kwargs["chandelier_atr_mult"], 2.7)
        self.assertEqual(mock_intraday.call_args.kwargs["exit_fill_policy"], "target_first")

    @patch("edgar.services.intraday_strategy.run_intraday_backtest")
    def test_strategy_intraday_endpoint_passes_break_even_options(self, mock_intraday):
        mock_intraday.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "15m",
            "strategy_variant": "fractal_breakout_ema200",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10100,
            "total_return_pct": 1.0,
            "total_trades": 2,
            "long_trades": 1,
            "short_trades": 1,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 50.0,
            "max_drawdown_pct": 2.0,
            "profit_factor": 1.2,
            "cagr_pct": 0.5,
            "avg_trade_return_pct": 0.2,
            "exposure_pct": 10.0,
            "total_fees": 3.5,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "15m",
            "lookback_years": 2,
            "strategy_variant": "fractal_breakout_ema200",
            "use_break_even_stop": True,
            "break_even_trigger_r": 1.5,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_intraday.assert_called_once()
        self.assertTrue(mock_intraday.call_args.kwargs["use_break_even_stop"])
        self.assertEqual(mock_intraday.call_args.kwargs["break_even_trigger_r"], 1.5)

    @patch("edgar.services.intraday_strategy.run_intraday_backtest")
    def test_strategy_intraday_endpoint_passes_stop_buffer(self, mock_intraday):
        mock_intraday.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "15m",
            "strategy_variant": "fractal_breakout_ema200",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10100,
            "total_return_pct": 1.0,
            "total_trades": 2,
            "long_trades": 1,
            "short_trades": 1,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_rate": 50.0,
            "max_drawdown_pct": 2.0,
            "profit_factor": 1.2,
            "cagr_pct": 0.5,
            "avg_trade_return_pct": 0.2,
            "exposure_pct": 10.0,
            "total_fees": 3.5,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "15m",
            "lookback_years": 2,
            "strategy_variant": "fractal_breakout_ema200",
            "stop_buffer_bps": 9.0,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_intraday.assert_called_once()
        self.assertEqual(mock_intraday.call_args.kwargs["stop_buffer_bps"], 9.0)

    def test_strategy_intraday_endpoint_missing_ticker(self):
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps({"interval": "15m", "lookback_years": 2}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    @patch("edgar.services.manipulation_strategy.run_manipulation_backtest")
    def test_strategy_intraday_endpoint_manipulation_variant(self, mock_manipulation):
        mock_manipulation.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "manipulation_ifvg",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10400,
            "total_return_pct": 4.0,
            "total_trades": 6,
            "long_trades": 3,
            "short_trades": 3,
            "winning_trades": 4,
            "losing_trades": 2,
            "win_rate": 66.7,
            "max_drawdown_pct": 3.2,
            "profit_factor": 1.8,
            "cagr_pct": 2.1,
            "avg_trade_return_pct": 0.5,
            "exposure_pct": 8.4,
            "total_fees": 8.1,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "manipulation_ifvg",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["interval"], "60m")
        self.assertEqual(payload["strategy_variant"], "manipulation_ifvg")
        mock_manipulation.assert_called_once()
        self.assertEqual(mock_manipulation.call_args.kwargs["lookback_years"], 2.0)
        self.assertEqual(mock_manipulation.call_args.kwargs["market_data_source"], "auto")
        self.assertTrue(mock_manipulation.call_args.kwargs["auto_adjust_for_yf_limits"])

    @patch("edgar.services.manipulation_strategy.run_manipulation_backtest")
    def test_strategy_intraday_endpoint_manipulation_passes_stop_buffer(self, mock_manipulation):
        mock_manipulation.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "manipulation_ifvg",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10400,
            "total_return_pct": 4.0,
            "total_trades": 6,
            "long_trades": 3,
            "short_trades": 3,
            "winning_trades": 4,
            "losing_trades": 2,
            "win_rate": 66.7,
            "max_drawdown_pct": 3.2,
            "profit_factor": 1.8,
            "cagr_pct": 2.1,
            "avg_trade_return_pct": 0.5,
            "exposure_pct": 8.4,
            "total_fees": 8.1,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "strategy_variant": "manipulation_ifvg",
            "stop_buffer_bps": 11.0,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_manipulation.assert_called_once()
        self.assertEqual(mock_manipulation.call_args.kwargs["stop_buffer_bps"], 11.0)

    @patch("edgar.services.manipulation_strategy.run_manipulation_backtest")
    def test_strategy_intraday_endpoint_manipulation_passes_volume_exhaustion_options(self, mock_manipulation):
        mock_manipulation.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "manipulation_ifvg",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10400,
            "total_return_pct": 4.0,
            "total_trades": 6,
            "long_trades": 3,
            "short_trades": 3,
            "winning_trades": 4,
            "losing_trades": 2,
            "win_rate": 66.7,
            "max_drawdown_pct": 3.2,
            "profit_factor": 1.8,
            "cagr_pct": 2.1,
            "avg_trade_return_pct": 0.5,
            "exposure_pct": 8.4,
            "total_fees": 8.1,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "strategy_variant": "manipulation_ifvg",
            "use_volume_exhaustion_filter": True,
            "max_sweep_rel_volume": 2.4,
            "min_reversal_pressure_ratio": 0.4,
            "min_rejection_wick_ratio": 0.3,
            "exhaustion_lookback_bars": 18,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_manipulation.assert_called_once()
        self.assertTrue(mock_manipulation.call_args.kwargs["use_volume_exhaustion_filter"])
        self.assertEqual(mock_manipulation.call_args.kwargs["max_sweep_rel_volume"], 2.4)
        self.assertEqual(mock_manipulation.call_args.kwargs["min_reversal_pressure_ratio"], 0.4)
        self.assertEqual(mock_manipulation.call_args.kwargs["min_rejection_wick_ratio"], 0.3)
        self.assertEqual(mock_manipulation.call_args.kwargs["exhaustion_lookback_bars"], 18)

    @patch("edgar.services.market_mechanics_strategy.run_market_mechanics_backtest")
    def test_strategy_intraday_endpoint_price_action_variant(self, mock_market_mechanics):
        mock_market_mechanics.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "price_action_3step",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10325,
            "total_return_pct": 3.25,
            "total_trades": 5,
            "long_trades": 3,
            "short_trades": 2,
            "winning_trades": 3,
            "losing_trades": 2,
            "win_rate": 60.0,
            "max_drawdown_pct": 2.8,
            "profit_factor": 1.7,
            "cagr_pct": 1.8,
            "avg_trade_return_pct": 0.6,
            "exposure_pct": 7.5,
            "total_fees": 6.2,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "price_action_3step",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["interval"], "60m")
        self.assertEqual(payload["strategy_variant"], "price_action_3step")
        mock_market_mechanics.assert_called_once()
        self.assertEqual(mock_market_mechanics.call_args.kwargs["lookback_years"], 2.0)
        self.assertEqual(mock_market_mechanics.call_args.kwargs["market_data_source"], "auto")
        self.assertTrue(mock_market_mechanics.call_args.kwargs["auto_adjust_for_yf_limits"])

    @patch("edgar.services.session_sfp_fvg_strategy.run_session_sfp_fvg_backtest")
    def test_strategy_intraday_endpoint_hourly_sfp_fvg_variant(self, mock_session_sfp):
        mock_session_sfp.return_value = {
            "ticker": "ETH-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "hourly_sfp_fvg",
            "entry_session": "new_york_equity_open",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10310,
            "total_return_pct": 3.1,
            "total_trades": 5,
            "long_trades": 3,
            "short_trades": 2,
            "winning_trades": 3,
            "losing_trades": 2,
            "win_rate": 60.0,
            "max_drawdown_pct": 2.2,
            "profit_factor": 1.8,
            "cagr_pct": 1.5,
            "avg_trade_return_pct": 0.5,
            "exposure_pct": 4.0,
            "total_fees": 5.0,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "ETH-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "hourly_sfp_fvg",
            "market_data_source": "binance",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "ETH-USD")
        self.assertEqual(payload["interval"], "5m")
        self.assertEqual(payload["strategy_variant"], "hourly_sfp_fvg")
        mock_session_sfp.assert_called_once()
        self.assertEqual(mock_session_sfp.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_session_sfp.call_args.kwargs["rr_multiple"], 2.0)
        self.assertEqual(mock_session_sfp.call_args.kwargs["session_trigger_window_minutes"], 60)
        self.assertTrue(mock_session_sfp.call_args.kwargs["use_target_room_filter"])
        self.assertEqual(mock_session_sfp.call_args.kwargs["min_target_room_ratio"], 1.0)

    @patch("edgar.services.session_range_breakout_strategy.run_session_range_breakout_backtest")
    def test_strategy_intraday_endpoint_session_range_breakout_variant(self, mock_session_breakout):
        mock_session_breakout.return_value = {
            "ticker": "BTC-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "session_range_breakout",
            "entry_session": "new_york_equity_open_breakout",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10150,
            "total_return_pct": 1.5,
            "total_trades": 8,
            "long_trades": 5,
            "short_trades": 3,
            "winning_trades": 4,
            "losing_trades": 4,
            "win_rate": 50.0,
            "max_drawdown_pct": 1.1,
            "profit_factor": 1.2,
            "cagr_pct": 0.7,
            "avg_trade_return_pct": 0.3,
            "exposure_pct": 3.5,
            "total_fees": 4.2,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "BTC-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "session_range_breakout",
            "market_data_source": "binance",
            "session_open": "asia_open",
            "range_lookback_minutes": 180,
            "breakout_window_minutes": 90,
            "breakout_buffer_bps": 2.0,
            "breakout_close_buffer_bps": 5.0,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "BTC-USD")
        self.assertEqual(payload["strategy_variant"], "session_range_breakout")
        mock_session_breakout.assert_called_once()
        self.assertEqual(mock_session_breakout.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_session_breakout.call_args.kwargs["session_open"], "asia_open")
        self.assertEqual(mock_session_breakout.call_args.kwargs["range_lookback_minutes"], 180)
        self.assertEqual(mock_session_breakout.call_args.kwargs["breakout_window_minutes"], 90)
        self.assertEqual(mock_session_breakout.call_args.kwargs["breakout_buffer_bps"], 2.0)
        self.assertEqual(mock_session_breakout.call_args.kwargs["breakout_close_buffer_bps"], 5.0)

    @patch("edgar.services.opening_shock_fade_strategy.run_opening_shock_fade_backtest")
    def test_strategy_intraday_endpoint_opening_shock_fade_variant(self, mock_opening_shock):
        mock_opening_shock.return_value = {
            "ticker": "BTC-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "opening_shock_fade",
            "entry_session": "hong_kong_open",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10080,
            "total_return_pct": 0.8,
            "total_trades": 6,
            "long_trades": 0,
            "short_trades": 6,
            "winning_trades": 4,
            "losing_trades": 2,
            "win_rate": 66.7,
            "max_drawdown_pct": 1.3,
            "profit_factor": 1.4,
            "cagr_pct": 0.4,
            "avg_trade_return_pct": 0.2,
            "exposure_pct": 1.8,
            "total_fees": 2.4,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "BTC-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "opening_shock_fade",
            "market_data_source": "binance",
            "session_open": "hong_kong_open",
            "opening_range_minutes": 20,
            "shock_window_minutes": 35,
            "entry_window_minutes": 65,
            "min_shock_bps": 40.0,
            "min_shock_atr_mult": 0.9,
            "reclaim_buffer_bps": 3.0,
            "stop_buffer_bps": 6.0,
            "max_hold_minutes": 150,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "BTC-USD")
        self.assertEqual(payload["strategy_variant"], "opening_shock_fade")
        mock_opening_shock.assert_called_once()
        self.assertEqual(mock_opening_shock.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_opening_shock.call_args.kwargs["session_open"], "hong_kong_open")
        self.assertEqual(mock_opening_shock.call_args.kwargs["opening_range_minutes"], 20)
        self.assertEqual(mock_opening_shock.call_args.kwargs["shock_window_minutes"], 35)
        self.assertEqual(mock_opening_shock.call_args.kwargs["entry_window_minutes"], 65)
        self.assertEqual(mock_opening_shock.call_args.kwargs["min_shock_bps"], 40.0)
        self.assertEqual(mock_opening_shock.call_args.kwargs["min_shock_atr_mult"], 0.9)
        self.assertEqual(mock_opening_shock.call_args.kwargs["reclaim_buffer_bps"], 3.0)
        self.assertEqual(mock_opening_shock.call_args.kwargs["stop_buffer_bps"], 6.0)
        self.assertEqual(mock_opening_shock.call_args.kwargs["max_hold_minutes"], 150)
        self.assertEqual(mock_opening_shock.call_args.kwargs["slippage_bps"], 2.0)

    @patch("edgar.services.opening_range_breakdown_strategy.run_opening_range_breakdown_backtest")
    def test_strategy_intraday_endpoint_opening_range_breakdown_short_variant(self, mock_orb_breakdown):
        mock_orb_breakdown.return_value = {
            "ticker": "BTC-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "opening_range_breakdown_short",
            "entry_session": "tokyo_open",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10040,
            "total_return_pct": 0.4,
            "total_trades": 7,
            "long_trades": 0,
            "short_trades": 7,
            "winning_trades": 3,
            "losing_trades": 4,
            "win_rate": 42.9,
            "max_drawdown_pct": 1.8,
            "profit_factor": 1.1,
            "cagr_pct": 0.2,
            "avg_trade_return_pct": 0.1,
            "exposure_pct": 2.0,
            "total_fees": 3.1,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "BTC-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "opening_range_breakdown_short",
            "market_data_source": "binance",
            "session_open": "hong_kong_open",
            "opening_range_minutes": 25,
            "entry_window_minutes": 80,
            "breakdown_buffer_bps": 4.0,
            "breakdown_close_buffer_bps": 6.0,
            "require_wick_retest": True,
            "retest_tolerance_bps": 7.0,
            "trend_filter_mode": "below_20d_low_and_lower_highs",
            "daily_ema_period": 18,
            "lookback_low_period": 20,
            "max_hold_minutes": 210,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "BTC-USD")
        self.assertEqual(payload["strategy_variant"], "opening_range_breakdown_short")
        mock_orb_breakdown.assert_called_once()
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["session_open"], "hong_kong_open")
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["opening_range_minutes"], 25)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["entry_window_minutes"], 80)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["breakdown_buffer_bps"], 4.0)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["breakdown_close_buffer_bps"], 6.0)
        self.assertTrue(mock_orb_breakdown.call_args.kwargs["require_wick_retest"])
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["retest_tolerance_bps"], 7.0)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["trend_filter_mode"], "below_20d_low_and_lower_highs")
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["daily_ema_period"], 18)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["lookback_low_period"], 20)
        self.assertEqual(mock_orb_breakdown.call_args.kwargs["max_hold_minutes"], 210)

    @patch("edgar.services.asia_turtle_short_strategy.run_asia_turtle_short_backtest")
    def test_strategy_intraday_endpoint_asia_turtle_short_variant(self, mock_turtle_short):
        mock_turtle_short.return_value = {
            "ticker": "ETH-USD",
            "data_mode": "intraday",
            "interval": "15m",
            "strategy_variant": "asia_turtle_short",
            "entry_session": "tokyo_open",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10120,
            "total_return_pct": 1.2,
            "total_trades": 5,
            "long_trades": 0,
            "short_trades": 5,
            "winning_trades": 2,
            "losing_trades": 3,
            "win_rate": 40.0,
            "max_drawdown_pct": 2.4,
            "profit_factor": 1.3,
            "cagr_pct": 0.6,
            "avg_trade_return_pct": 0.4,
            "exposure_pct": 14.0,
            "total_fees": 2.5,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "ETH-USD",
            "initial_capital": 10000,
            "interval": "15m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "asia_turtle_short",
            "market_data_source": "binance",
            "session_open": "tokyo_open",
            "channel_period": 55,
            "exit_channel_period": 20,
            "atr_period": 21,
            "atr_stop_mult": 2.2,
            "entry_window_minutes": 360,
            "enable_pyramiding": True,
            "pyramid_add_atr": 0.6,
            "max_units": 3,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "ETH-USD")
        self.assertEqual(payload["strategy_variant"], "asia_turtle_short")
        mock_turtle_short.assert_called_once()
        self.assertEqual(mock_turtle_short.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_turtle_short.call_args.kwargs["session_open"], "tokyo_open")
        self.assertEqual(mock_turtle_short.call_args.kwargs["channel_period"], 55)
        self.assertEqual(mock_turtle_short.call_args.kwargs["exit_channel_period"], 20)
        self.assertEqual(mock_turtle_short.call_args.kwargs["atr_period"], 21)
        self.assertEqual(mock_turtle_short.call_args.kwargs["atr_stop_mult"], 2.2)
        self.assertEqual(mock_turtle_short.call_args.kwargs["entry_window_minutes"], 360)
        self.assertTrue(mock_turtle_short.call_args.kwargs["enable_pyramiding"])
        self.assertEqual(mock_turtle_short.call_args.kwargs["pyramid_add_atr"], 0.6)
        self.assertEqual(mock_turtle_short.call_args.kwargs["max_units"], 3)

    @patch("edgar.services.orb_turtle_hybrid_strategy.run_orb_turtle_hybrid_backtest")
    def test_strategy_intraday_endpoint_orb_turtle_variant(self, mock_orb_turtle):
        mock_orb_turtle.return_value = {
            "ticker": "ETH-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "orb_turtle_hybrid",
            "strategy_name": "Opening Pressure + Turtle",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10320,
            "total_return_pct": 3.2,
            "total_trades": 11,
            "long_trades": 6,
            "short_trades": 5,
            "winning_trades": 6,
            "losing_trades": 5,
            "win_rate": 54.5,
            "max_drawdown_pct": 1.7,
            "profit_factor": 1.4,
            "cagr_pct": 1.7,
            "avg_trade_return_pct": 0.5,
            "exposure_pct": 5.0,
            "total_fees": 6.0,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "ETH-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "orb_turtle_hybrid",
            "market_data_source": "binance",
            "orb_window_minutes": 15,
            "donchian_period": 200,
            "min_rel_volume": 1.1,
            "short_risk_pct": 0.03,
            "long_risk_pct": 0.015,
            "short_time_stop_minutes": 120,
            "portfolio_gate_threshold_pct": -1.0,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "ETH-USD")
        self.assertEqual(payload["strategy_variant"], "orb_turtle_hybrid")
        mock_orb_turtle.assert_called_once()
        self.assertEqual(mock_orb_turtle.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_orb_turtle.call_args.kwargs["orb_window_minutes"], 15)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["donchian_period"], 200)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["min_rel_volume"], 1.1)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["turtle_initial_stop_atr_mult"], 2.5)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["turtle_weak_breakout_minutes"], 120)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["turtle_weak_breakout_atr"], 0.5)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["chandelier_period"], 14)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["chandelier_atr_period"], 14)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["chandelier_atr_mult"], 3.5)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["short_time_stop_minutes"], 120)
        self.assertEqual(mock_orb_turtle.call_args.kwargs["portfolio_gate_threshold_pct"], -1.0)

    @patch("edgar.services.mtf_liquidity_flow_strategy.run_mtf_liquidity_flow_backtest")
    def test_strategy_intraday_endpoint_mtf_liquidity_flow_variant(self, mock_mtf_flow):
        mock_mtf_flow.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "mtf_liquidity_flow",
            "entry_model": "hybrid",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10210,
            "total_return_pct": 2.1,
            "total_trades": 4,
            "long_trades": 1,
            "short_trades": 3,
            "winning_trades": 2,
            "losing_trades": 2,
            "win_rate": 50.0,
            "max_drawdown_pct": 1.9,
            "profit_factor": 1.5,
            "cagr_pct": 1.2,
            "avg_trade_return_pct": 0.4,
            "exposure_pct": 6.1,
            "total_fees": 5.4,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "allow_shorts": True,
            "strategy_variant": "mtf_liquidity_flow",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["ticker"], "AAPL")
        self.assertEqual(payload["interval"], "60m")
        self.assertEqual(payload["strategy_variant"], "mtf_liquidity_flow")
        mock_mtf_flow.assert_called_once()
        self.assertEqual(mock_mtf_flow.call_args.kwargs["lookback_years"], 2.0)
        self.assertEqual(mock_mtf_flow.call_args.kwargs["market_data_source"], "auto")
        self.assertTrue(mock_mtf_flow.call_args.kwargs["auto_adjust_for_yf_limits"])

    @patch("edgar.services.mtf_liquidity_flow_strategy.run_mtf_liquidity_flow_backtest")
    def test_strategy_intraday_endpoint_mtf_passes_alignment_and_exhaustion_options(self, mock_mtf_flow):
        mock_mtf_flow.return_value = {
            "ticker": "AAPL",
            "data_mode": "intraday",
            "interval": "60m",
            "strategy_variant": "mtf_liquidity_flow",
            "entry_model": "hybrid",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10210,
            "total_return_pct": 2.1,
            "total_trades": 4,
            "long_trades": 1,
            "short_trades": 3,
            "winning_trades": 2,
            "losing_trades": 2,
            "win_rate": 50.0,
            "max_drawdown_pct": 1.9,
            "profit_factor": 1.5,
            "cagr_pct": 1.2,
            "avg_trade_return_pct": 0.4,
            "exposure_pct": 6.1,
            "total_fees": 5.4,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "AAPL",
            "initial_capital": 10000,
            "interval": "60m",
            "lookback_years": 2,
            "strategy_variant": "mtf_liquidity_flow",
            "trend_alignment_mode": "aligned",
            "entry_session": "london",
            "rr_multiple": 2.0,
            "use_volume_exhaustion_filter": True,
            "max_sweep_rel_volume": 2.5,
            "min_reversal_pressure_ratio": 0.45,
            "min_rejection_wick_ratio": 0.3,
            "exhaustion_lookback_bars": 20,
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_mtf_flow.assert_called_once()
        self.assertEqual(mock_mtf_flow.call_args.kwargs["trend_alignment_mode"], "aligned")
        self.assertEqual(mock_mtf_flow.call_args.kwargs["entry_session"], "london")
        self.assertEqual(mock_mtf_flow.call_args.kwargs["rr_multiple"], 2.0)
        self.assertTrue(mock_mtf_flow.call_args.kwargs["use_volume_exhaustion_filter"])
        self.assertEqual(mock_mtf_flow.call_args.kwargs["max_sweep_rel_volume"], 2.5)
        self.assertEqual(mock_mtf_flow.call_args.kwargs["min_reversal_pressure_ratio"], 0.45)
        self.assertEqual(mock_mtf_flow.call_args.kwargs["min_rejection_wick_ratio"], 0.3)
        self.assertEqual(mock_mtf_flow.call_args.kwargs["exhaustion_lookback_bars"], 20)

    @patch("edgar.services.market_mechanics_strategy.run_market_mechanics_backtest")
    def test_strategy_intraday_endpoint_price_action_explicit_binance_source(self, mock_market_mechanics):
        mock_market_mechanics.return_value = {
            "ticker": "BTC-USD",
            "data_mode": "intraday",
            "interval": "5m",
            "strategy_variant": "price_action_3step",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2026-01-01T00:00:00",
            "initial_capital": 10000,
            "final_capital": 10325,
            "total_return_pct": 3.25,
            "total_trades": 5,
            "long_trades": 3,
            "short_trades": 2,
            "winning_trades": 3,
            "losing_trades": 2,
            "win_rate": 60.0,
            "max_drawdown_pct": 2.8,
            "profit_factor": 1.7,
            "cagr_pct": 1.8,
            "avg_trade_return_pct": 0.6,
            "exposure_pct": 7.5,
            "total_fees": 6.2,
            "trades": [],
            "equity_curve": [],
        }
        body = {
            "ticker": "BTC-USD",
            "initial_capital": 10000,
            "interval": "5m",
            "lookback_years": 0.2,
            "allow_shorts": True,
            "strategy_variant": "price_action_3step",
            "market_data_source": "binance",
            "market_data_symbol": "BTCUSDT",
        }
        res = self.client.post(
            "/api/edgar/drf/strategy/backtest-intraday/",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        mock_market_mechanics.assert_called_once()
        self.assertEqual(mock_market_mechanics.call_args.kwargs["market_data_source"], "binance")
        self.assertEqual(mock_market_mechanics.call_args.kwargs["market_data_symbol"], "BTCUSDT")


class MtfIntervalPolicyTests(TestCase):
    def test_resolve_effective_interval_auto_adjusts_for_long_lookback(self):
        from edgar.services.mtf_liquidity_flow_strategy import _resolve_effective_interval

        effective, note = _resolve_effective_interval(
            requested_interval="5m",
            lookback_years=2.0,
            auto_adjust_for_yf_limits=True,
        )
        self.assertEqual(effective, "60m")
        self.assertIn("Adjusted interval", note)

    def test_resolve_effective_interval_strict_mode_raises(self):
        from edgar.services.mtf_liquidity_flow_strategy import _resolve_effective_interval

        with self.assertRaises(ValueError):
            _resolve_effective_interval(
                requested_interval="5m",
                lookback_years=2.0,
                auto_adjust_for_yf_limits=False,
            )

    def test_resolve_effective_interval_intraday_auto_adjusts(self):
        from edgar.services.intraday_strategy import _resolve_effective_interval

        effective, note = _resolve_effective_interval(
            requested_interval="5m",
            lookback_years=2.0,
            auto_adjust_for_yf_limits=True,
        )
        self.assertEqual(effective, "60m")
        self.assertIn("Adjusted interval", note)

    def test_resolve_effective_interval_market_mechanics_strict_raises(self):
        from edgar.services.market_mechanics_strategy import _resolve_effective_interval

        with self.assertRaises(ValueError):
            _resolve_effective_interval(
                requested_interval="5m",
                lookback_years=2.0,
                auto_adjust_for_yf_limits=False,
            )

    def test_resolve_effective_interval_manipulation_auto_adjusts(self):
        from edgar.services.manipulation_strategy import _resolve_effective_interval

        effective, note = _resolve_effective_interval(
            requested_interval="15m",
            lookback_years=1.5,
            auto_adjust_for_yf_limits=True,
        )
        self.assertEqual(effective, "60m")
        self.assertIn("Adjusted interval", note)


class BinanceDataTests(TestCase):
    def test_resolve_binance_symbol_for_btc_and_paxg(self):
        from edgar.services.binance_data import resolve_binance_symbol

        self.assertEqual(resolve_binance_symbol("BTC-USD"), "BTCUSDT")
        self.assertEqual(resolve_binance_symbol("ETH-USD"), "ETHUSDT")
        self.assertEqual(resolve_binance_symbol("PAXG-USD"), "PAXGUSDT")
        self.assertEqual(resolve_binance_symbol("SOL-USD"), "SOLUSDT")

    def _real_local_cache_file(self, symbol: str) -> Path | None:
        from django.conf import settings

        roots = [
            Path(settings.BASE_DIR) / "cache" / "binance_asia_orb",
            Path(settings.BASE_DIR) / "cache" / "cache" / "cache" / "binance_asia_orb",
        ]
        best_match: Path | None = None
        for root in roots:
            matches = sorted(root.glob(f"{symbol}_*_5m.csv.gz"))
            if not matches:
                continue
            candidate = max(matches, key=lambda path: path.stat().st_size)
            if best_match is None or candidate.stat().st_size > best_match.stat().st_size:
                best_match = candidate
        return best_match

    def _require_real_local_cache(self, symbol: str) -> Path:
        cache_file = self._real_local_cache_file(symbol)
        if cache_file is None:
            self.skipTest(f"real {symbol} local cache file is not available")
        return cache_file

    def _real_tiingo_cache_file(self, symbol: str) -> Path | None:
        from django.conf import settings

        path = Path(settings.BASE_DIR) / "cache" / "cache" / "tiingo" / f"{symbol}_5m.parquet"
        return path if path.exists() else None

    def _require_real_tiingo_cache(self, symbol: str) -> Path:
        cache_file = self._real_tiingo_cache_file(symbol)
        if cache_file is None:
            self.skipTest(f"real {symbol} Tiingo cache file is not available")
        return cache_file

    def _assert_real_binance_strategy_smoke(
        self,
        *,
        runner,
        ticker: str,
        symbol: str,
        yfinance_patch: str,
        interval: str = "5m",
        lookback_years: float = 0.1,
        **kwargs,
    ) -> dict:
        self._require_real_local_cache(symbol)

        with patch(yfinance_patch, side_effect=AssertionError("yfinance path should not be used")), patch(
            "requests.sessions.Session.get",
            side_effect=AssertionError("live Binance API should not be used when local cache exists"),
        ):
            payload = runner(
                ticker=ticker,
                interval=interval,
                lookback_years=lookback_years,
                market_data_source="binance",
                auto_adjust_for_yf_limits=False,
                **kwargs,
            )

        self.assertEqual(payload["market_data_source"], "binance")
        self.assertEqual(payload["market_data_symbol"], symbol)
        self.assertEqual(payload["effective_interval"], interval)
        self.assertGreater(payload["bar_count"], 0)
        return payload

    @patch("edgar.services.binance_data.fetch_binance_klines")
    def test_binance_klines_chart_endpoint(self, mock_fetch):
        start = datetime(2025, 1, 1)
        mock_fetch.return_value = (
            [
                {
                    "timestamp": start,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1200.0,
                },
                {
                    "timestamp": start + timedelta(minutes=5),
                    "open": 100.5,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.8,
                    "volume": 1500.0,
                },
            ],
            "ETHUSDT",
        )
        res = self.client.get("/api/edgar/drf/charts/binance-klines/?symbol=ETHUSDT&interval=5m&days=3")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["symbol"], "ETHUSDT")
        self.assertEqual(payload["bar_count"], 2)
        self.assertEqual(len(payload["price_series"]), 2)

    @patch("edgar.services.binance_data.time.sleep")
    @patch("edgar.services.binance_data.requests.Session.get")
    def test_fetch_binance_klines_uses_cache_for_repeat_5m_calls(self, mock_get, mock_sleep):
        from edgar.services.binance_data import fetch_binance_klines

        cache.clear()
        row = [
            int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "1200.0",
        ]
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.text = ""
        first_response.json.return_value = [row]
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.text = ""
        second_response.json.return_value = []
        mock_get.side_effect = [first_response, second_response]

        bars1, symbol1 = fetch_binance_klines(
            ticker="BNB-USD",
            interval="5m",
            lookback_years=0.1,
            warmup_days=5,
        )
        bars2, symbol2 = fetch_binance_klines(
            ticker="BNB-USD",
            interval="5m",
            lookback_years=0.1,
            warmup_days=5,
        )

        self.assertEqual(symbol1, "BNBUSDT")
        self.assertEqual(symbol2, "BNBUSDT")
        self.assertEqual(bars1, bars2)
        self.assertEqual(mock_get.call_count, 1)

    @patch("edgar.services.binance_data.requests.Session.get")
    def test_fetch_binance_klines_prefers_real_local_cache_when_available(self, mock_get):
        from edgar.services.binance_data import fetch_binance_klines

        self._require_real_local_cache("BTCUSDT")
        mock_get.side_effect = AssertionError("live Binance API should not be used when local cache exists")

        bars, symbol = fetch_binance_klines(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=0.05,
            warmup_days=5,
            use_cache=False,
        )

        self.assertEqual(symbol, "BTCUSDT")
        self.assertGreater(len(bars), 0)
        self.assertLess(bars[0]["timestamp"], bars[-1]["timestamp"])
        self.assertEqual(mock_get.call_count, 0)

    def test_mtf_strategy_binance_source_path(self):
        from edgar.services.mtf_liquidity_flow_strategy import run_mtf_liquidity_flow_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_mtf_liquidity_flow_backtest,
            ticker="BTC-USD",
            symbol="BTCUSDT",
            yfinance_patch="edgar.services.mtf_liquidity_flow_strategy._fetch_intraday_bars",
        )

        self.assertEqual(payload["strategy_variant"], "mtf_liquidity_flow")

    def test_intraday_strategy_binance_source_path(self):
        from edgar.services.intraday_strategy import run_intraday_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_intraday_backtest,
            ticker="BTC-USD",
            symbol="BTCUSDT",
            yfinance_patch="edgar.services.intraday_strategy._fetch_intraday_bars",
        )

        self.assertEqual(payload["strategy_variant"], "fractal_breakout_ema200")

    def test_session_range_breakout_binance_source_path(self):
        from edgar.services.session_range_breakout_strategy import run_session_range_breakout_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_session_range_breakout_backtest,
            ticker="BTC-USD",
            symbol="BTCUSDT",
            yfinance_patch="edgar.services.session_range_breakout_strategy._fetch_intraday_bars",
            lookback_years=0.2,
        )

        self.assertEqual(payload["strategy_variant"], "session_range_breakout")

    def test_orb_turtle_hybrid_binance_source_path(self):
        from edgar.services.orb_turtle_hybrid_strategy import run_orb_turtle_hybrid_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_orb_turtle_hybrid_backtest,
            ticker="ETH-USD",
            symbol="ETHUSDT",
            yfinance_patch="edgar.services.orb_turtle_hybrid_strategy._fetch_intraday_bars",
            lookback_years=0.2,
        )

        self.assertEqual(payload["strategy_variant"], "orb_turtle_hybrid")

    def test_orb_turtle_long_gate_snapshot_requires_all_five_gates(self):
        from edgar.services.orb_turtle_hybrid_strategy import _turtle_long_gate_snapshot

        passed = _turtle_long_gate_snapshot(
            ts=datetime(2025, 1, 6, 15, 5, tzinfo=timezone.utc),
            close_tf=101.5,
            donchian_upper=100.0,
            ema_now=99.0,
            daily_slope_positive=True,
            rel_volume=1.3,
            min_rel_volume=1.0,
            breakout_buffer=0.0,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["count"], 5)
        self.assertEqual(passed["failed_names"], [])

        failed = _turtle_long_gate_snapshot(
            ts=datetime(2025, 1, 6, 14, 20, tzinfo=timezone.utc),
            close_tf=99.8,
            donchian_upper=100.0,
            ema_now=100.5,
            daily_slope_positive=False,
            rel_volume=0.8,
            min_rel_volume=1.0,
            breakout_buffer=0.0,
        )
        self.assertFalse(failed["passed"])
        self.assertLess(failed["count"], failed["required"])
        self.assertIn("session_window", failed["failed_names"])
        self.assertIn("donchian_breakout", failed["failed_names"])
        self.assertIn("ema_filter", failed["failed_names"])
        self.assertIn("daily_slope", failed["failed_names"])
        self.assertIn("rvol_filter", failed["failed_names"])

    def test_session_range_breakout_pre_open_range_handles_asia_midnight_crossover(self):
        from edgar.services.session_range_breakout_strategy import _pre_open_range

        session_times = [
            datetime(2025, 1, 1, 23, 45, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 23, 50, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 23, 55, tzinfo=timezone.utc),
            datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 2, 0, 5, tzinfo=timezone.utc),
        ]
        highs = [101.0, 102.0, 103.0, 104.0, 105.0]
        lows = [99.0, 98.5, 98.0, 97.5, 97.0]

        low, high, count = _pre_open_range(
            session_times=session_times,
            highs=highs,
            lows=lows,
            idx=4,
            range_lookback_minutes=15,
            session_open="asia_open",
        )

        self.assertEqual(count, 3)
        self.assertEqual(low, 98.0)
        self.assertEqual(high, 103.0)

    def test_opening_shock_session_anchor_handles_hk_pre_open_crossover(self):
        from edgar.services.opening_shock_fade_strategy import _session_anchor_for_ts

        before_open = _session_anchor_for_ts(datetime(2025, 1, 2, 0, 55), "hong_kong_open")
        after_open = _session_anchor_for_ts(datetime(2025, 1, 2, 1, 35), "hong_kong_open")

        self.assertEqual(before_open, datetime(2025, 1, 1, 1, 30))
        self.assertEqual(after_open, datetime(2025, 1, 2, 1, 30))

    def test_opening_shock_fade_uses_real_local_cache_data(self):
        from edgar.services.opening_shock_fade_strategy import run_opening_shock_fade_backtest

        cache_file = self._require_real_local_cache("BTCUSDT")

        payload = run_opening_shock_fade_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=0.25,
            market_data_source="binance",
            session_open="tokyo_open",
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertEqual(payload["market_data_symbol"], "BTCUSDT")
        self.assertEqual(payload["strategy_variant"], "opening_shock_fade")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_opening_range_breakdown_uses_real_local_cache_data(self):
        from edgar.services.opening_range_breakdown_strategy import run_opening_range_breakdown_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_opening_range_breakdown_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=0.5,
            market_data_source="binance",
            session_open="tokyo_open",
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertEqual(payload["market_data_symbol"], "BTCUSDT")
        self.assertEqual(payload["strategy_variant"], "opening_range_breakdown_short")
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_asia_turtle_short_uses_real_local_cache_data(self):
        from edgar.services.asia_turtle_short_strategy import run_asia_turtle_short_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_asia_turtle_short_backtest(
            ticker="BTC-USD",
            interval="15m",
            lookback_years=1.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertEqual(payload["market_data_symbol"], "BTCUSDT")
        self.assertEqual(payload["strategy_variant"], "asia_turtle_short")
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_opening_shock_fade_uses_real_tiingo_cache_data(self):
        from edgar.services.opening_shock_fade_strategy import run_opening_shock_fade_backtest

        cache_file = self._require_real_tiingo_cache("COIN")
        payload = run_opening_shock_fade_backtest(
            ticker="COIN",
            interval="5m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_symbol"], "COIN")
        self.assertEqual(payload["strategy_variant"], "opening_shock_fade")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_opening_range_breakdown_uses_real_tiingo_cache_data(self):
        from edgar.services.opening_range_breakdown_strategy import run_opening_range_breakdown_backtest

        cache_file = self._require_real_tiingo_cache("MSTR")
        payload = run_opening_range_breakdown_backtest(
            ticker="MSTR",
            interval="5m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_symbol"], "MSTR")
        self.assertEqual(payload["strategy_variant"], "opening_range_breakdown_short")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_asia_turtle_short_uses_real_tiingo_cache_data(self):
        from edgar.services.asia_turtle_short_strategy import run_asia_turtle_short_backtest

        cache_file = self._require_real_tiingo_cache("MSTR")
        payload = run_asia_turtle_short_backtest(
            ticker="MSTR",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_symbol"], "MSTR")
        self.assertEqual(payload["strategy_variant"], "asia_turtle_short")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["short_trades"], payload["total_trades"])
        self.assertEqual(payload["trades"][0]["direction"], "short")

    def test_session_turtle_trend_uses_real_local_cache_data(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="15m",
            lookback_years=1.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertEqual(payload["market_data_symbol"], "BTCUSDT")
        self.assertEqual(payload["strategy_variant"], "session_turtle_trend")
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_turtle_trend_uses_real_tiingo_cache_data(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        cache_file = self._require_real_tiingo_cache("COIN")
        payload = run_session_turtle_trend_backtest(
            ticker="COIN",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=55,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_symbol"], "COIN")
        self.assertEqual(payload["strategy_variant"], "session_turtle_trend")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertGreater(payload["bar_count"], 0)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_turtle_trend_5m_uses_real_local_cache_with_4h_filter(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertTrue(payload["use_4h_trend_filter"])
        self.assertEqual(payload["trend_filter_interval"], "4h")
        self.assertEqual(payload["trend_fast_period"], 55)
        self.assertEqual(payload["trend_slow_period"], 200)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_turtle_trend_5m_uses_real_tiingo_cache_with_4h_filter(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        cache_file = self._require_real_tiingo_cache("COIN")
        payload = run_session_turtle_trend_backtest(
            ticker="COIN",
            interval="5m",
            lookback_years=2.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertTrue(payload["use_4h_trend_filter"])
        self.assertEqual(payload["trend_filter_interval"], "4h")
        self.assertEqual(payload["trend_fast_period"], 55)
        self.assertEqual(payload["trend_slow_period"], 200)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_turtle_trend_supports_fixed_stop_pct(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            base_risk_pct=0.05,
            max_position_pct=0.90,
            fixed_stop_pct=0.10,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertEqual(payload["fixed_stop_pct"], 0.10)
        self.assertGreater(payload["total_trades"], 0)
        self.assertEqual(payload["trades"][0]["stop_source"], "fixed_pct_stop")

    @patch("edgar.services.session_turtle_trend_strategy._load_local_bars")
    def test_session_turtle_trend_supports_extended_hours_protective_exits_only(self, mock_load_bars):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        start = datetime(2024, 1, 1, 14, 30)
        bars: list[dict] = []
        total_sessions = 22
        bars_per_session = 96
        breakout_session = 21

        for session_idx in range(total_sessions):
            session_start = start + timedelta(days=session_idx)
            for step in range(bars_per_session):
                ts = session_start + timedelta(minutes=15 * step)
                open_price = 100.0
                high_price = 100.2
                low_price = 99.8
                close_price = 100.0
                if session_idx == breakout_session:
                    if step == 2:
                        low_price = 99.9
                        high_price = 101.0
                        close_price = 101.0
                    elif 3 <= step < 26:
                        open_price = 101.0
                        high_price = 101.2
                        low_price = 100.8
                        close_price = 101.0
                    elif step == 34:
                        open_price = 101.0
                        high_price = 101.0
                        low_price = 99.5
                        close_price = 100.5
                    elif step > 34:
                        open_price = 101.0
                        high_price = 101.2
                        low_price = 100.4
                        close_price = 101.0
                bars.append(
                    {
                        "timestamp": ts,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": 1000.0,
                    }
                )

        mock_load_bars.return_value = (bars, "TOKEQ", "local_tiingo_cache", "synthetic")

        baseline = run_session_turtle_trend_backtest(
            ticker="TOKEQ",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            fixed_stop_pct=0.05,
            entry_window_minutes=390,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        extended = run_session_turtle_trend_backtest(
            ticker="TOKEQ",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            fixed_stop_pct=0.05,
            entry_window_minutes=390,
            core_session_minutes=390,
            use_extended_hours_protective_exits_only=True,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertGreater(baseline["total_trades"], 0)
        self.assertGreater(extended["total_trades"], 0)
        baseline_trade = baseline["trades"][0]
        extended_trade = extended["trades"][0]
        self.assertEqual(baseline_trade["exit_reason"], "exit_channel")
        self.assertEqual(extended_trade["exit_reason"], "end_of_data")
        self.assertEqual(extended["core_session_minutes"], 390)
        self.assertTrue(extended["use_extended_hours_protective_exits_only"])
        self.assertGreater(
            datetime.fromisoformat(extended_trade["exit_date"]),
            datetime.fromisoformat(baseline_trade["exit_date"]),
        )

    @patch("edgar.services.session_turtle_trend_strategy._load_local_bars")
    def test_session_turtle_trend_can_flatten_before_weekend(self, mock_load_bars):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        start = datetime(2024, 1, 1, 14, 30)
        session_days: list[datetime] = []
        cursor = start
        while len(session_days) < 27:
            if cursor.weekday() < 5:
                session_days.append(cursor)
            cursor += timedelta(days=1)

        bars: list[dict] = []
        bars_per_session = 96
        breakout_session = 24  # Friday with enough completed lookback sessions
        for session_idx, session_start in enumerate(session_days):
            for step in range(bars_per_session):
                ts = session_start + timedelta(minutes=15 * step)
                open_price = 100.0
                high_price = 100.2
                low_price = 99.8
                close_price = 100.0
                if session_idx == breakout_session:
                    if step == 2:
                        high_price = 101.0
                        close_price = 101.0
                    elif step > 2:
                        open_price = 101.0
                        high_price = 101.2
                        low_price = 100.8
                        close_price = 101.0
                elif session_idx > breakout_session:
                    open_price = 101.0
                    high_price = 101.2
                    low_price = 100.8
                    close_price = 101.0
                bars.append(
                    {
                        "timestamp": ts,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": 1000.0,
                    }
                )

        mock_load_bars.return_value = (bars, "TOKEQ", "local_tiingo_cache", "synthetic")

        baseline = run_session_turtle_trend_backtest(
            ticker="TOKEQ",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            fixed_stop_pct=0.05,
            entry_window_minutes=390,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        weekend_flat = run_session_turtle_trend_backtest(
            ticker="TOKEQ",
            interval="15m",
            lookback_years=1.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            channel_period=20,
            fixed_stop_pct=0.05,
            entry_window_minutes=390,
            close_positions_before_weekend=True,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertGreater(baseline["total_trades"], 0)
        self.assertGreater(weekend_flat["total_trades"], 0)
        baseline_trade = baseline["trades"][0]
        weekend_trade = weekend_flat["trades"][0]
        self.assertEqual(baseline_trade["exit_reason"], "end_of_data")
        self.assertEqual(weekend_trade["exit_reason"], "weekend_flat")
        self.assertTrue(weekend_flat["close_positions_before_weekend"])
        self.assertLess(
            datetime.fromisoformat(weekend_trade["exit_date"]),
            datetime.fromisoformat(baseline_trade["exit_date"]),
        )

    def test_session_turtle_trend_supports_volume_risk_scaling(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        self._require_real_local_cache("BTCUSDT")
        baseline = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            base_risk_pct=0.05,
            max_position_pct=0.90,
            fixed_stop_pct=0.10,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        scaled = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            base_risk_pct=0.05,
            max_position_pct=0.90,
            fixed_stop_pct=0.10,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            use_volume_risk_scaling=True,
            volume_period=40,
            volume_risk_floor=0.5,
            volume_risk_cap=1.5,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertTrue(scaled["use_volume_risk_scaling"])
        self.assertEqual(scaled["volume_period"], 40)
        self.assertEqual(scaled["total_trades"], baseline["total_trades"])
        self.assertTrue(any(abs(t["volume_risk_scale"] - 1.0) > 1e-6 for t in scaled["trades"]))

    def test_session_turtle_trend_supports_directional_volume_boost_and_pyramiding(self):
        from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_session_turtle_trend_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            channel_period=20,
            base_risk_pct=0.05,
            max_position_pct=0.90,
            fixed_stop_pct=0.10,
            use_4h_trend_filter=True,
            trend_fast_period=55,
            trend_slow_period=200,
            use_directional_volume_risk_boost=True,
            directional_volume_min_rel_volume=1.25,
            directional_volume_close_location_threshold=0.65,
            directional_volume_risk_pct=0.07,
            enable_pyramiding=True,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )

        self.assertTrue(payload["use_directional_volume_risk_boost"])
        self.assertEqual(payload["directional_volume_risk_pct"], 0.07)
        self.assertGreater(payload["total_trades"], 0)
        self.assertTrue(any(t["directional_volume_confirmed"] for t in payload["trades"]))
        self.assertTrue(any(t["risk_model"] == "directional_volume_boost" for t in payload["trades"]))
        self.assertTrue(any(t["add_count"] > 0 for t in payload["trades"]))
        self.assertTrue(any(len(t["add_events"]) > 0 for t in payload["trades"]))

    def test_session_ma_crossover_uses_real_local_cache_data(self):
        from edgar.services.session_ma_crossover_strategy import run_session_ma_crossover_backtest

        self._require_real_local_cache("BTCUSDT")
        payload = run_session_ma_crossover_backtest(
            ticker="BTC-USD",
            interval="5m",
            lookback_years=2.0,
            market_data_source="binance",
            session_open="tokyo_open",
            trend_fast_period=55,
            trend_slow_period=200,
            fixed_stop_pct=0.10,
            base_risk_pct=0.05,
            max_position_pct=0.90,
        )

        self.assertEqual(payload["market_data_source"], "local_binance_cache")
        self.assertEqual(payload["market_data_symbol"], "BTCUSDT")
        self.assertEqual(payload["strategy_variant"], "session_ma_crossover")
        self.assertEqual(payload["trend_fast_period"], 55)
        self.assertEqual(payload["trend_slow_period"], 200)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_ma_crossover_uses_real_tiingo_cache_data(self):
        from edgar.services.session_ma_crossover_strategy import run_session_ma_crossover_backtest

        cache_file = self._require_real_tiingo_cache("COIN")
        payload = run_session_ma_crossover_backtest(
            ticker="COIN",
            interval="5m",
            lookback_years=2.0,
            market_data_source="tiingo",
            session_open="new_york_equity_open",
            trend_fast_period=55,
            trend_slow_period=200,
            fixed_stop_pct=0.10,
            base_risk_pct=0.05,
            max_position_pct=0.90,
        )

        self.assertEqual(payload["market_data_source"], "local_tiingo_cache")
        self.assertEqual(payload["market_data_path"], str(cache_file))
        self.assertEqual(payload["strategy_variant"], "session_ma_crossover")
        self.assertEqual(payload["trend_fast_period"], 55)
        self.assertEqual(payload["trend_slow_period"], 200)
        self.assertGreater(payload["total_trades"], 0)
        self.assertGreater(payload["long_trades"], 0)
        self.assertGreater(payload["short_trades"], 0)

    def test_session_sfp_strategy_binance_source_path(self):
        from edgar.services.session_sfp_fvg_strategy import run_session_sfp_fvg_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_session_sfp_fvg_backtest,
            ticker="ETH-USD",
            symbol="ETHUSDT",
            yfinance_patch="edgar.services.session_sfp_fvg_strategy._fetch_intraday_bars",
        )

        self.assertEqual(payload["strategy_variant"], "hourly_sfp_fvg")

    def test_market_mechanics_strategy_binance_source_path(self):
        from edgar.services.market_mechanics_strategy import run_market_mechanics_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_market_mechanics_backtest,
            ticker="BTC-USD",
            symbol="BTCUSDT",
            yfinance_patch="edgar.services.market_mechanics_strategy._fetch_intraday_bars",
        )

        self.assertEqual(payload["strategy_variant"], "price_action_3step")

    def test_manipulation_strategy_binance_source_path(self):
        from edgar.services.manipulation_strategy import run_manipulation_backtest

        payload = self._assert_real_binance_strategy_smoke(
            runner=run_manipulation_backtest,
            ticker="BTC-USD",
            symbol="BTCUSDT",
            yfinance_patch="edgar.services.manipulation_strategy._fetch_intraday_bars",
        )

        self.assertEqual(payload["strategy_variant"], "manipulation_ifvg")


class ManipulationStopLogicTests(TestCase):
    def test_initial_long_stop_uses_sweep_wick_not_ifvg_edge(self):
        from edgar.services.manipulation_strategy import _FVG, _initial_stop_from_event

        stop_loss, stop_source = _initial_stop_from_event(
            direction="long",
            ifvg=_FVG(kind="bearish", idx=10, zone_low=98.0, zone_high=99.0),
            event={"idx": 12, "level": 100.0, "sweep_low": 96.0},
            stop_buffer=0.001,
        )

        self.assertEqual(stop_source, "sweep_wick")
        self.assertAlmostEqual(stop_loss, 95.904, places=6)

    def test_initial_short_stop_uses_sweep_wick_not_ifvg_edge(self):
        from edgar.services.manipulation_strategy import _FVG, _initial_stop_from_event

        stop_loss, stop_source = _initial_stop_from_event(
            direction="short",
            ifvg=_FVG(kind="bullish", idx=10, zone_low=101.0, zone_high=102.0),
            event={"idx": 12, "level": 100.0, "sweep_high": 104.0},
            stop_buffer=0.001,
        )

        self.assertEqual(stop_source, "sweep_wick")
        self.assertAlmostEqual(stop_loss, 104.104, places=6)


class MtfLiquidityFlowValidationTests(TestCase):
    def test_invalid_entry_session_raises(self):
        from edgar.services.mtf_liquidity_flow_strategy import run_mtf_liquidity_flow_backtest

        with self.assertRaises(ValueError):
            run_mtf_liquidity_flow_backtest(
                ticker="BTC-USD",
                market_data_source="binance",
                interval="5m",
                lookback_years=2.0,
                entry_session="tokyo_open",
            )

    def test_session_sfp_variant_rejects_hourly_execution_interval(self):
        from edgar.services.session_sfp_fvg_strategy import run_session_sfp_fvg_backtest

        with self.assertRaises(ValueError):
            run_session_sfp_fvg_backtest(
                ticker="ETH-USD",
                market_data_source="binance",
                interval="60m",
                lookback_years=0.5,
            )

    def test_session_sfp_previous_two_sessions_bias_helper(self):
        from edgar.services.session_sfp_fvg_strategy import _SessionBar, _session_bias_from_previous_two

        sessions = [
            _SessionBar(date(2025, 1, 1), 100.0, 105.0, 95.0, 102.0, 1000.0),
            _SessionBar(date(2025, 1, 2), 102.0, 109.0, 99.0, 108.0, 1100.0),
            _SessionBar(date(2025, 1, 3), 108.0, 110.0, 104.0, 106.0, 900.0),
        ]

        bias, older, recent = _session_bias_from_previous_two(
            sessions=sessions,
            current_session_idx=2,
            buffer=0.0,
        )

        self.assertEqual(bias, "long")
        self.assertEqual(older.session_date, date(2025, 1, 1))
        self.assertEqual(recent.session_date, date(2025, 1, 2))

    def test_session_sfp_prefers_nearest_liquidity_target(self):
        from edgar.services.session_sfp_fvg_strategy import _resolve_next_liquidity_target

        target, source = _resolve_next_liquidity_target(
            direction="long",
            reference_price=100.0,
            hourly_targets=(106.0, "hourly_pivot"),
            session_targets=(104.0, "prior_session_high"),
        )

        self.assertEqual(target, 104.0)
        self.assertEqual(source, "prior_session_high")


class StrategySerializationTests(TestCase):
    def test_break_even_helpers(self):
        self.assertEqual(
            _break_even_stop_candidate(
                direction="long",
                entry_price=100.0,
                initial_stop=95.0,
                bar_high=105.0,
                bar_low=99.0,
                trigger_r=1.0,
            ),
            100.0,
        )
        self.assertTrue(
            _stop_is_break_even_or_better(
                direction="long",
                entry_price=100.0,
                active_stop=100.0,
            )
        )

    def test_sweep_exhaustion_helper_detects_rejection_with_controlled_volume(self):
        confirmed, reference_volume, reference_pressure = _sweep_exhaustion_confirmed(
            direction="short",
            volumes=[100.0, 110.0, 120.0, 130.0],
            opens=[10.0, 10.5, 11.0, 11.4],
            highs=[10.5, 11.0, 11.6, 12.3],
            lows=[9.9, 10.4, 10.9, 11.0],
            closes=[10.4, 10.9, 11.3, 11.1],
            idx=3,
            lookback=3,
            max_sweep_rel_volume=2.0,
            min_reversal_pressure_ratio=0.25,
            min_rejection_wick_ratio=0.25,
            vol_sma=[None, None, 100.0, 100.0],
        )

        self.assertTrue(confirmed)
        self.assertAlmostEqual(reference_volume, 110.0, places=6)
        self.assertGreater(reference_pressure, 0.0)

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
            use_break_even_stop=True,
            break_even_trigger_r=1.0,
            trades=[trade],
            equity_curve=[{"date": "2024-01-02", "equity": 10010.0, "capital": 10000.0}],
        )
        payload = backtest_to_dict(result)
        self.assertIn("profit_factor", payload)
        self.assertIn("cagr_pct", payload)
        self.assertIn("use_break_even_stop", payload)
        self.assertIn("break_even_trigger_r", payload)
        self.assertIn("long_trades", payload)
        self.assertIn("short_trades", payload)
        self.assertIn("use_chandelier_exit", payload)
        self.assertIn("exit_fill_policy", payload)
        self.assertEqual(len(payload["trades"]), 1)
        t0 = payload["trades"][0]
        self.assertIn("active_stop_loss", t0)
        self.assertIn("intrabar_conflict", t0)
        self.assertIn("exit_fill_policy", t0)
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
    def test_breakout_conviction_multiplier_rewards_stronger_breakouts(self):
        conviction_mult, breakout_penetration, directional_close_score = _breakout_conviction_multiplier(
            direction="long",
            rel_volume=2.4,
            rel_volume_ratio=1.8,
            close_location=0.96,
            close_price=104.0,
            breakout_level=100.0,
            channel_high=101.0,
            channel_low=95.0,
            max_mult=1.35,
        )
        self.assertEqual(conviction_mult, 1.35)
        self.assertAlmostEqual(breakout_penetration, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(directional_close_score, 0.92, places=4)

    def test_williams_fractal_detection(self):
        highs = [10.0, 11.0, 15.0, 12.0, 11.0, 13.0, 12.0]
        lows = [9.0, 8.0, 7.0, 8.0, 9.0, 8.5, 9.5]
        frac_hi, frac_lo = _williams_fractals(highs, lows, period=2)
        self.assertEqual(len(frac_hi), len(highs))
        self.assertEqual(len(frac_lo), len(lows))
        self.assertEqual(frac_hi[2], 15.0)
        self.assertEqual(frac_lo[2], 7.0)

    def test_same_bar_exit_policy_marks_conflict(self):
        raw_exit, hit_sl, hit_tp, intrabar_conflict = _resolve_bar_bracket_exit(
            direction="long",
            bar_high=110.0,
            bar_low=94.0,
            stop_loss=95.0,
            take_profit=108.0,
            fill_policy="stop_first",
        )
        self.assertEqual(raw_exit, 95.0)
        self.assertTrue(hit_sl)
        self.assertFalse(hit_tp)
        self.assertTrue(intrabar_conflict)

    def test_chandelier_candidate_uses_recent_extreme(self):
        rolling_highs = [None, None, 105.0, 107.0]
        rolling_lows = [None, None, 95.0, 96.0]
        atr_vals = [None, None, 2.0, 2.5]
        candidate = _chandelier_stop_candidate(
            direction="long",
            idx=3,
            rolling_highs=rolling_highs,
            rolling_lows=rolling_lows,
            atr_values=atr_vals,
            atr_mult=3.0,
        )
        self.assertEqual(candidate, 99.5)
