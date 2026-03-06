from __future__ import annotations

from datetime import date
from typing import Iterable

from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental


def _parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _upsert_point(company, source_document, taxonomy, tag, unit, point):
    end_date = _parse_date(point.get("end"))
    if not end_date:
        return False

    filed_date = _parse_date(point.get("filed"))
    defaults = {
        "source_document": source_document,
        "value": float(point.get("val")) if point.get("val") is not None else 0.0,
        "fiscal_year": point.get("fy"),
        "fiscal_period": point.get("fp") or "",
        "accession": point.get("accn") or "",
        "form": point.get("form") or "",
        "frame": point.get("frame") or "",
    }

    EdgarFundamental.objects.update_or_create(
        company=company,
        taxonomy=taxonomy,
        tag=tag,
        unit=unit,
        end_date=end_date,
        filed_date=filed_date,
        fiscal_year=point.get("fy"),
        fiscal_period=point.get("fp") or "",
        accession=point.get("accn") or "",
        form=point.get("form") or "",
        frame=point.get("frame") or "",
        defaults=defaults,
    )
    return True


def save_fundamentals_from_facts(
    company: EdgarCompany,
    payload: dict,
    source_document: EdgarDocument | None = None,
    taxonomy_filter: str | None = None,
    tag_filter: str | None = None,
) -> int:
    """Extract facts payload into normalized time-series rows."""
    count = 0
    facts = (payload or {}).get("facts", {})
    for taxonomy, tags in facts.items():
        if taxonomy_filter and taxonomy != taxonomy_filter:
            continue
        if not isinstance(tags, dict):
            continue
        for tag, tag_obj in tags.items():
            if tag_filter and tag != tag_filter:
                continue
            units = (tag_obj or {}).get("units", {})
            if not isinstance(units, dict):
                continue
            for unit, points in units.items():
                if not isinstance(points, Iterable):
                    continue
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    if _upsert_point(
                        company=company,
                        source_document=source_document,
                        taxonomy=taxonomy,
                        tag=tag,
                        unit=unit,
                        point=point,
                    ):
                        count += 1
    return count


def save_fundamentals_from_concept(
    company: EdgarCompany,
    payload: dict,
    source_document: EdgarDocument | None = None,
) -> int:
    """Extract company_concept payload into normalized rows."""
    count = 0
    taxonomy = (payload or {}).get("taxonomy") or ""
    tag = (payload or {}).get("tag") or ""
    units = (payload or {}).get("units", {})
    if not taxonomy or not tag or not isinstance(units, dict):
        return 0

    for unit, points in units.items():
        if not isinstance(points, Iterable):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            if _upsert_point(
                company=company,
                source_document=source_document,
                taxonomy=taxonomy,
                tag=tag,
                unit=unit,
                point=point,
            ):
                count += 1
    return count
