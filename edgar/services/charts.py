"""Data exploration and chart data generation module.

Produces JSON-ready data structures for frontend visualization:
- Sector distribution (pie/bar charts)
- Score distribution across companies
- Financial metric trends over time
- Price history charts
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from django.db.models import Avg, Count, Max, Min, Q

from edgar.models import (
    BuffettScore,
    EdgarCompany,
    EdgarFundamental,
    StockPrice,
)

logger = logging.getLogger(__name__)


def sector_distribution(sp500_only: bool = True) -> dict:
    """Return sector distribution data for charting."""
    qs = EdgarCompany.objects.exclude(sector="")
    if sp500_only:
        qs = qs.filter(is_sp500=True)

    counts = (
        qs.values("sector")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    labels = [c["sector"] for c in counts]
    values = [c["count"] for c in counts]

    return {
        "chart_type": "sector_distribution",
        "labels": labels,
        "values": values,
        "total": sum(values),
    }


def score_distribution(bin_size: int = 10) -> dict:
    """Return distribution of Buffett scores in bins."""
    scores = BuffettScore.objects.values_list("overall_score", flat=True)

    bins = {}
    for i in range(0, 101, bin_size):
        label = f"{i}-{i + bin_size - 1}"
        bins[label] = 0

    for s in scores:
        idx = min(int(s // bin_size) * bin_size, 100 - bin_size)
        label = f"{idx}-{idx + bin_size - 1}"
        bins[label] = bins.get(label, 0) + 1

    return {
        "chart_type": "score_distribution",
        "labels": list(bins.keys()),
        "values": list(bins.values()),
    }


def sector_score_comparison() -> dict:
    """Average Buffett score by sector."""
    data = (
        BuffettScore.objects.filter(company__sector__gt="")
        .values("company__sector")
        .annotate(avg_score=Avg("overall_score"), count=Count("id"))
        .order_by("-avg_score")
    )

    return {
        "chart_type": "sector_scores",
        "labels": [d["company__sector"] for d in data],
        "values": [round(d["avg_score"], 1) for d in data],
        "counts": [d["count"] for d in data],
    }


def metric_trend(company_id: int, metric_key: str, mapping: dict | None = None) -> dict:
    """Return time-series data for a given metric for charting."""
    from edgar.services.metric_mapping import resolve_and_store_metric_mapping

    company = EdgarCompany.objects.get(pk=company_id)

    if mapping is None:
        mapping = resolve_and_store_metric_mapping(company=company, use_ai=False, persist=False)

    info = mapping.get(metric_key)
    if not info:
        return {"chart_type": "metric_trend", "metric": metric_key, "labels": [], "values": []}

    points = (
        EdgarFundamental.objects.filter(
            company=company,
            taxonomy=info["taxonomy"],
            tag=info["tag"],
            form__in=["10-K", "20-F", "40-F"],
        )
        .order_by("end_date")
        .values_list("end_date", "value")
    )

    seen_years = set()
    labels = []
    values = []
    for end_date, value in points:
        year = end_date.year
        if year not in seen_years:
            seen_years.add(year)
            labels.append(end_date.isoformat())
            values.append(value)

    return {
        "chart_type": "metric_trend",
        "metric": metric_key,
        "ticker": company.ticker,
        "labels": labels,
        "values": values,
    }


def price_history(company_id: int, period_days: int = 365) -> dict:
    """Return price history data for charting."""
    from datetime import date, timedelta

    company = EdgarCompany.objects.get(pk=company_id)
    cutoff = date.today() - timedelta(days=period_days)

    prices = (
        StockPrice.objects.filter(company=company, date__gte=cutoff)
        .order_by("date")
        .values_list("date", "close")
    )

    return {
        "chart_type": "price_history",
        "ticker": company.ticker,
        "labels": [d.isoformat() for d, _ in prices],
        "values": [v for _, v in prices],
    }


def company_score_breakdown(company_id: int) -> dict:
    """Return radar chart data for a single company's Buffett score components."""
    try:
        score = BuffettScore.objects.filter(company_id=company_id).latest()
    except BuffettScore.DoesNotExist:
        return {"chart_type": "score_breakdown", "labels": [], "values": []}

    labels = ["ROE", "Debt", "Margin", "Earnings Growth", "FCF", "Valuation"]
    values = [
        score.roe_score,
        score.debt_score,
        score.margin_score,
        score.earnings_growth_score,
        score.fcf_score,
        score.valuation_score,
    ]

    return {
        "chart_type": "score_breakdown",
        "ticker": score.company.ticker,
        "overall_score": score.overall_score,
        "labels": labels,
        "values": values,
        "current_price": score.current_price,
        "intrinsic_value": score.intrinsic_value,
        "margin_of_safety": score.margin_of_safety,
    }


def top_scored_companies(limit: int = 20) -> dict:
    """Return top Buffett-scored companies."""
    scores = (
        BuffettScore.objects.select_related("company")
        .order_by("-overall_score")[:limit]
    )

    return {
        "chart_type": "top_scores",
        "companies": [
            {
                "ticker": s.company.ticker,
                "name": s.company.name,
                "sector": s.company.sector,
                "overall_score": s.overall_score,
                "current_price": s.current_price,
                "intrinsic_value": s.intrinsic_value,
                "margin_of_safety": s.margin_of_safety,
            }
            for s in scores
        ],
    }
