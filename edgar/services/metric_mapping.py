from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import requests

from edgar.models import EdgarCompany, EdgarFundamental, EdgarMetricMapping

logger = logging.getLogger(__name__)

# Canonical keys -> likely US-GAAP tags.
STANDARD_METRICS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpenditure",
    ],
    "free_cash_flow": ["FreeCashFlow"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "EntityCommonStockSharesOutstanding",
    ],
}


def _available_tags(company: EdgarCompany, taxonomy: str = "us-gaap"):
    return list(
        EdgarFundamental.objects.filter(company=company, taxonomy=taxonomy)
        .values_list("tag", flat=True)
        .distinct()
    )


def _heuristic_mapping(company: EdgarCompany, taxonomy: str = "us-gaap") -> dict[str, dict]:
    tags = set(_available_tags(company=company, taxonomy=taxonomy))
    mapping: dict[str, dict] = {}
    for metric_key, candidates in STANDARD_METRICS.items():
        for candidate in candidates:
            if candidate in tags:
                mapping[metric_key] = {
                    "taxonomy": taxonomy,
                    "tag": candidate,
                    "source": EdgarMetricMapping.SOURCE_HEURISTIC,
                    "confidence": 0.7,
                    "rationale": "matched standard candidate tag",
                }
                break
    return mapping


def _openai_mapping(company: EdgarCompany, taxonomy: str = "us-gaap") -> dict[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}

    tags = _available_tags(company=company, taxonomy=taxonomy)
    if not tags:
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You map company XBRL tags to canonical metrics. Return JSON object with key 'mapping'. "
                    "Each mapping value must contain: tag, confidence (0-1), rationale. "
                    "Use only tags from provided list."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "taxonomy": taxonomy,
                        "canonical_metrics": list(STANDARD_METRICS.keys()),
                        "available_tags": tags,
                    }
                ),
            },
        ],
        "temperature": 0,
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        out = {}
        for metric_key, info in (data.get("mapping") or {}).items():
            tag = (info or {}).get("tag")
            if metric_key in STANDARD_METRICS and tag in tags:
                out[metric_key] = {
                    "taxonomy": taxonomy,
                    "tag": tag,
                    "source": EdgarMetricMapping.SOURCE_OPENAI,
                    "confidence": float((info or {}).get("confidence", 0.6)),
                    "rationale": (info or {}).get("rationale", "openai mapped"),
                }
        return out
    except Exception as exc:
        logger.warning("OpenAI mapping failed for %s: %s", company.ticker, exc)
        return {}


def resolve_and_store_metric_mapping(
    company: EdgarCompany,
    taxonomy: str = "us-gaap",
    use_ai: bool = True,
    force_refresh: bool = False,
    persist: bool = True,
) -> dict[str, dict]:
    """Build metric mapping for a company and persist mappings."""
    if not force_refresh:
        existing = EdgarMetricMapping.objects.filter(company=company)
        if existing.exists():
            return {
                m.metric_key: {
                    "taxonomy": m.taxonomy,
                    "tag": m.tag,
                    "source": m.source,
                    "confidence": m.confidence,
                    "rationale": m.rationale,
                }
                for m in existing
            }

    mapping = _heuristic_mapping(company=company, taxonomy=taxonomy)
    if use_ai:
        ai = _openai_mapping(company=company, taxonomy=taxonomy)
        # AI suggestions can fill gaps and override weak heuristic matches.
        for metric_key, info in ai.items():
            existing = mapping.get(metric_key)
            if not existing or float(info.get("confidence", 0)) >= float(existing.get("confidence", 0)):
                mapping[metric_key] = info

    if persist:
        for metric_key, info in mapping.items():
            EdgarMetricMapping.objects.update_or_create(
                company=company,
                metric_key=metric_key,
                defaults={
                    "taxonomy": info["taxonomy"],
                    "tag": info["tag"],
                    "source": info["source"],
                    "confidence": info.get("confidence", 0.0),
                    "rationale": info.get("rationale", ""),
                },
            )
    return mapping


def build_fundamental_table(
    company: EdgarCompany,
    mapping: dict[str, dict],
    period_start=None,
    period_end=None,
    frequency: str = "annual",
):
    forms = None
    if frequency == "annual":
        forms = ["10-K", "20-F", "40-F"]
    elif frequency == "quarterly":
        forms = ["10-Q"]

    metric_points = defaultdict(dict)
    metric_meta = {}

    for metric_key, info in mapping.items():
        qs = EdgarFundamental.objects.filter(
            company=company,
            taxonomy=info["taxonomy"],
            tag=info["tag"],
        )
        if period_start:
            qs = qs.filter(end_date__gte=period_start)
        if period_end:
            qs = qs.filter(end_date__lte=period_end)
        if forms is not None:
            qs = qs.filter(form__in=forms)
        qs = qs.order_by("end_date", "-filed_date")

        metric_meta[metric_key] = {
            "taxonomy": info["taxonomy"],
            "tag": info["tag"],
            "source": info.get("source", "heuristic"),
            "confidence": info.get("confidence", 0.0),
        }

        for p in qs:
            period_key = p.end_date.isoformat()
            # latest filed_date wins per period+metric
            prev = metric_points[period_key].get(metric_key)
            if prev and prev.get("filed_date") and p.filed_date and prev["filed_date"] >= p.filed_date:
                continue
            metric_points[period_key][metric_key] = {
                "value": p.value,
                "form": p.form,
                "fiscal_year": p.fiscal_year,
                "fiscal_period": p.fiscal_period,
                "filed_date": p.filed_date,
            }

    rows = []
    for period_key in sorted(metric_points.keys()):
        row = {
            "period_end": period_key,
            "form": "",
            "fiscal_year": None,
            "fiscal_period": "",
        }
        for metric_key in STANDARD_METRICS.keys():
            row[metric_key] = None

        # Fill row fields from first available metric point.
        for metric_key, info in metric_points[period_key].items():
            row[metric_key] = info["value"]
            if not row["form"] and info.get("form"):
                row["form"] = info["form"]
            if row["fiscal_year"] is None and info.get("fiscal_year") is not None:
                row["fiscal_year"] = info["fiscal_year"]
            if not row["fiscal_period"] and info.get("fiscal_period"):
                row["fiscal_period"] = info["fiscal_period"]
        rows.append(row)

    return {
        "metrics": list(STANDARD_METRICS.keys()),
        "mapping": metric_meta,
        "rows": rows,
    }
