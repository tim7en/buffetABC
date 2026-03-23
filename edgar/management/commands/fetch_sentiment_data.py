"""
Django management command: fetch_sentiment_data

Downloads and caches all four sentiment data sources used by the
sentiment exposure governor, then prints a coverage report showing
how well each source covers the production backtest window
(2022-02-09 → 2026-03-02).

Usage
-----
    python manage.py fetch_sentiment_data

    # Force re-download (ignore cache):
    python manage.py fetch_sentiment_data --force-refresh

    # Only fetch specific sources:
    python manage.py fetch_sentiment_data --sources vix crypto_fg

    # Use a custom AAII CSV path:
    python manage.py fetch_sentiment_data --aaii-csv /path/to/aaii.csv

Sources
-------
  vix        CBOE VIX (yfinance) — auto-downloaded
  crypto_fg  Crypto Fear & Greed (alternative.me) — auto-downloaded
  cnn_fg     CNN Fear & Greed (CNN dataviz API) — auto-downloaded
  aaii       AAII Sentiment (manual CSV required — see instructions)

Cache location: cache/sentiment/
"""

from __future__ import annotations

import logging
from pathlib import Path

from django.core.management.base import BaseCommand

from edgar.services.sentiment_data import (
    coverage_report,
    load_aaii_scores,
    load_cnn_fg_scores,
    load_crypto_fg_scores,
    load_vix_scores,
)

logger = logging.getLogger(__name__)

_BACKTEST_START = "2022-02-09"
_BACKTEST_END = "2026-03-02"

_ALL_SOURCES = ["vix", "crypto_fg", "cnn_fg", "aaii"]


class Command(BaseCommand):
    help = (
        "Download and cache VIX, Crypto F&G, CNN F&G, and AAII sentiment data. "
        "Prints a coverage report against the production backtest window."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--sources",
            nargs="+",
            choices=_ALL_SOURCES,
            default=_ALL_SOURCES,
            metavar="SOURCE",
            help=(
                "Which sources to fetch. "
                f"Choices: {_ALL_SOURCES}. Default: all."
            ),
        )
        parser.add_argument(
            "--force-refresh",
            action="store_true",
            help="Ignore existing cache and re-download from remote.",
        )
        parser.add_argument(
            "--aaii-csv",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Path to manually downloaded AAII sentiment CSV. "
                "Defaults to cache/sentiment/aaii_sentiment.csv. "
                "Download from: https://www.aaii.com/sentimentsurvey/sent_results"
            ),
        )
        parser.add_argument(
            "--backtest-start",
            default=_BACKTEST_START,
            help=f"Backtest window start for coverage report (default: {_BACKTEST_START}).",
        )
        parser.add_argument(
            "--backtest-end",
            default=_BACKTEST_END,
            help=f"Backtest window end for coverage report (default: {_BACKTEST_END}).",
        )

    def handle(self, *args, **options):
        sources = options["sources"]
        force = options["force_refresh"]
        aaii_csv = options["aaii_csv"]
        bt_start = options["backtest_start"]
        bt_end = options["backtest_end"]

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"\nFetching sentiment data — backtest window {bt_start} → {bt_end}\n"
            )
        )

        results: list[dict] = []

        # ── VIX ──────────────────────────────────────────────────────────────
        if "vix" in sources:
            self.stdout.write("► VIX (yfinance ^VIX) …")
            try:
                scores = load_vix_scores(start="2018-01-01", force_refresh=force)
                report = coverage_report(scores, "VIX", bt_start, bt_end)
                results.append(report)
                self.stdout.write(self.style.SUCCESS("  ✓ VIX OK"))
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"  ✗ VIX FAILED: {exc}"))
                results.append({"source": "VIX", "error": str(exc)})

        # ── Crypto Fear & Greed ───────────────────────────────────────────────
        if "crypto_fg" in sources:
            self.stdout.write("► Crypto Fear & Greed (alternative.me) …")
            try:
                scores = load_crypto_fg_scores(force_refresh=force)
                report = coverage_report(scores, "Crypto F&G", bt_start, bt_end)
                results.append(report)
                self.stdout.write(self.style.SUCCESS("  ✓ Crypto F&G OK"))
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"  ✗ Crypto F&G FAILED: {exc}"))
                results.append({"source": "Crypto F&G", "error": str(exc)})

        # ── CNN Fear & Greed ──────────────────────────────────────────────────
        if "cnn_fg" in sources:
            self.stdout.write("► CNN Fear & Greed (CNN dataviz API) …")
            try:
                scores = load_cnn_fg_scores(force_refresh=force)
                report = coverage_report(scores, "CNN F&G", bt_start, bt_end)
                results.append(report)
                self.stdout.write(self.style.SUCCESS("  ✓ CNN F&G OK"))
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"  ✗ CNN F&G FAILED: {exc}"))
                results.append({"source": "CNN F&G", "error": str(exc)})

        # ── AAII ──────────────────────────────────────────────────────────────
        if "aaii" in sources:
            self.stdout.write("► AAII Investor Sentiment (local CSV) …")
            try:
                scores = load_aaii_scores(
                    csv_path=aaii_csv,
                    backtest_end=bt_end,
                    force_refresh=force,
                )
                report = coverage_report(scores, "AAII", bt_start, bt_end)
                results.append(report)
                self.stdout.write(self.style.SUCCESS("  ✓ AAII OK"))
            except FileNotFoundError as exc:
                self.stdout.write(self.style.WARNING(str(exc)))
                results.append({"source": "AAII", "error": "CSV not found — manual download required"})
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"  ✗ AAII FAILED: {exc}"))
                results.append({"source": "AAII", "error": str(exc)})

        # ── Summary table ─────────────────────────────────────────────────────
        self.stdout.write(
            self.style.MIGRATE_HEADING("\n═══ Coverage Summary ═══\n")
        )
        header = f"{'Source':<14} {'Covered':>8} {'Total':>7} {'Coverage':>10} {'Gaps':>6}  {'Data start':>12}  {'Data end':>12}"
        self.stdout.write(header)
        self.stdout.write("─" * len(header))
        for r in results:
            if "error" in r:
                self.stdout.write(
                    f"  {r['source']:<12}  {'— ERROR':>8}  {r['error']}"
                )
            else:
                ok = r["coverage_pct"] >= 95.0
                style = self.style.SUCCESS if ok else self.style.WARNING
                line = (
                    f"  {r['source']:<12}  "
                    f"{r['covered']:>8}  "
                    f"{r['total_days']:>7}  "
                    f"{r['coverage_pct']:>9.1f}%  "
                    f"{r['gap_count']:>6}  "
                    f"  {r.get('data_start', 'n/a'):>12}  "
                    f"{r.get('data_end', 'n/a'):>12}"
                )
                self.stdout.write(style(line))

        self.stdout.write("")
        # Highlight any sources with <95% coverage
        low = [r for r in results if "error" not in r and r.get("coverage_pct", 0) < 95.0]
        if low:
            self.stdout.write(
                self.style.WARNING(
                    "⚠  The following sources have gaps in the backtest window — "
                    "missing dates will fall back to the nearest prior available score:\n"
                    + "\n".join(f"   • {r['source']}: {r['gap_count']} gaps" for r in low)
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS("✓ All fetched sources cover ≥95% of the backtest window.")
            )

        self.stdout.write(
            f"\nCache location: {Path('cache/sentiment').resolve()}\n"
        )
