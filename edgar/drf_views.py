from datetime import date

from django.db.models import Q
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument
from edgar.serializers import (
    EdgarCompanySerializer,
    EdgarDocumentSerializer,
    IngestionRequestSerializer,
)
from edgar.services.edgar_client import EdgarClient


def _row_for_symbol(symbol: str):
    row = sp500.by_symbol(symbol)
    if row:
        return row
    return {"Symbol": symbol.upper(), "CIK": "", "Security": ""}


def _upsert_company(row):
    ticker = (row.get("Symbol") or "").upper()
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
    return company


def _save_document(company, kind, endpoint, payload, params, success=True, error_message="", attempts=1):
    return EdgarDocument.objects.create(
        company=company,
        kind=kind,
        endpoint=endpoint,
        params=params,
        payload=payload,
        source_url=endpoint,
        http_status=None,
        attempt_count=attempts,
        success=success,
        error_message=error_message,
    )


def _filter_points_by_period(points, period_start: date | None, period_end: date | None):
    if not period_start and not period_end:
        return points

    filtered = []
    for p in points:
        end = p.get("end")
        if not end:
            continue
        try:
            d = date.fromisoformat(str(end)[:10])
        except ValueError:
            continue
        if period_start and d < period_start:
            continue
        if period_end and d > period_end:
            continue
        filtered.append(p)
    return filtered


def _extract_concept_points(payload, unit="USD"):
    units = (payload or {}).get("units", {})
    return list(units.get(unit) or [])


class EdgarCompanyViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EdgarCompany.objects.all().order_by("ticker")
    serializer_class = EdgarCompanySerializer

    def get_queryset(self):
        qs = super().get_queryset()
        q = self.request.query_params.get("q", "").strip()
        if q:
            qs = qs.filter(Q(ticker__icontains=q) | Q(name__icontains=q))
        return qs

    @action(detail=True, methods=["post"], url_path="fetch")
    def fetch_one(self, request, pk=None):
        company = self.get_object()
        payload = {
            "symbols": [company.ticker],
            "endpoint": request.data.get("endpoint", EdgarDocument.KIND_FACTS),
            "taxonomy": request.data.get("taxonomy", "us-gaap"),
            "tag": request.data.get("tag", "Assets"),
            "query": request.data.get("query", ""),
            "persist": request.data.get("persist", True),
            "include_payload": request.data.get("include_payload", False),
            "retries": request.data.get("retries", 3),
            "backoff": request.data.get("backoff", 1.0),
            "limit": 1,
        }
        if request.data.get("period_start"):
            payload["period_start"] = request.data["period_start"]
        if request.data.get("period_end"):
            payload["period_end"] = request.data["period_end"]
        serializer = IngestionRequestSerializer(data=payload)
        serializer.is_valid(raise_exception=True)
        result = run_ingestion(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)

    @action(detail=True, methods=["get"], url_path="fundamentals")
    def fundamentals(self, request, pk=None):
        company = self.get_object()
        taxonomy = request.query_params.get("taxonomy", "us-gaap")
        tag = request.query_params.get("tag", "Assets")
        unit = request.query_params.get("unit", "USD")
        period_start = request.query_params.get("period_start")
        period_end = request.query_params.get("period_end")

        start_date = date.fromisoformat(period_start) if period_start else None
        end_date = date.fromisoformat(period_end) if period_end else None

        docs = EdgarDocument.objects.filter(
            company=company,
            kind=EdgarDocument.KIND_CONCEPT,
            success=True,
            params__taxonomy=taxonomy,
            params__tag=tag,
        ).order_by("-fetched_at")

        points = []
        if docs.exists():
            # Use most recent concept snapshot; it already contains historical values.
            points = _extract_concept_points(docs.first().payload, unit=unit)
        else:
            facts_docs = EdgarDocument.objects.filter(
                company=company,
                kind=EdgarDocument.KIND_FACTS,
                success=True,
            ).order_by("-fetched_at")
            if facts_docs.exists():
                payload = facts_docs.first().payload
                points = (
                    payload.get("facts", {})
                    .get(taxonomy, {})
                    .get(tag, {})
                    .get("units", {})
                    .get(unit, [])
                )

        points = _filter_points_by_period(points, start_date, end_date)
        points = sorted(points, key=lambda x: x.get("end", ""))

        return Response(
            {
                "ticker": company.ticker,
                "taxonomy": taxonomy,
                "tag": tag,
                "unit": unit,
                "count": len(points),
                "points": points,
            }
        )


class EdgarDocumentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EdgarDocument.objects.select_related("company").all().order_by("-fetched_at")
    serializer_class = EdgarDocumentSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        ticker = self.request.query_params.get("ticker", "").strip().upper()
        kind = self.request.query_params.get("kind", "").strip()
        success = self.request.query_params.get("success", "").strip().lower()
        from_date = self.request.query_params.get("from", "").strip()
        to_date = self.request.query_params.get("to", "").strip()

        if ticker:
            qs = qs.filter(company__ticker=ticker)
        if kind:
            qs = qs.filter(kind=kind)
        if success in {"true", "false"}:
            qs = qs.filter(success=(success == "true"))
        if from_date:
            qs = qs.filter(fetched_at__date__gte=from_date)
        if to_date:
            qs = qs.filter(fetched_at__date__lte=to_date)
        return qs


class EdgarIngestionViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["post"], url_path="fetch")
    def fetch(self, request):
        serializer = IngestionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = run_ingestion(serializer.validated_data)
        return Response(result, status=status.HTTP_200_OK)


def run_ingestion(config):
    endpoint = config.get("endpoint", EdgarDocument.KIND_FACTS)
    symbols = [s.upper() for s in (config.get("symbols") or [])]
    search_name = (config.get("search_name") or "").strip()
    taxonomy = config.get("taxonomy", "us-gaap")
    tag = config.get("tag", "Assets")
    query = (config.get("query") or "").strip()
    limit = config.get("limit")
    retries = config.get("retries", 3)
    backoff = config.get("backoff", 1.0)
    persist = config.get("persist", True)
    include_payload = config.get("include_payload", False)
    period_start = config.get("period_start")
    period_end = config.get("period_end")

    if search_name:
        matches = sp500.search_companies(search_name, limit=limit or 25)
        symbols.extend([m["Symbol"].upper() for m in matches])

    if endpoint == EdgarDocument.KIND_FULLTEXT and not symbols:
        symbols = ["SEARCH"]

    if limit:
        symbols = symbols[:limit]

    client = EdgarClient(retries=retries, backoff_seconds=backoff)
    saved_documents = []
    items = []

    for symbol in symbols:
        row = _row_for_symbol(symbol)
        cik = str(row.get("CIK") or row.get("cik") or "")
        if endpoint != EdgarDocument.KIND_FULLTEXT and not cik:
            items.append({"symbol": symbol, "success": False, "error": "missing CIK"})
            continue

        params = {
            "taxonomy": taxonomy,
            "tag": tag,
            "query": query,
            "period_start": str(period_start) if period_start else "",
            "period_end": str(period_end) if period_end else "",
        }

        try:
            if endpoint == EdgarDocument.KIND_FACTS:
                source_url = client.BASE_URL.format(cik=cik.zfill(10))
                payload = client.company_facts(cik)
            elif endpoint == EdgarDocument.KIND_FILINGS:
                source_url = client.SUBMISSIONS_URL.format(cik=cik.zfill(10))
                payload = client.filings(cik)
            elif endpoint == EdgarDocument.KIND_CONCEPT:
                source_url = client.CONCEPT_URL.format(
                    cik=cik.zfill(10), taxonomy=taxonomy, tag=tag
                )
                payload = client.company_concept(cik, taxonomy=taxonomy, tag=tag)
                if period_start or period_end:
                    points = _extract_concept_points(payload, unit="USD")
                    payload = {
                        **payload,
                        "units": {
                            **payload.get("units", {}),
                            "USD": _filter_points_by_period(points, period_start, period_end),
                        },
                    }
            else:
                effective_query = query or symbol
                source_url = client.FULL_TEXT_SEARCH_URL
                payload = client.full_text_search(effective_query)

            doc = None
            if persist:
                company = _upsert_company(row)
                doc = _save_document(
                    company,
                    kind=endpoint,
                    endpoint=source_url,
                    payload=payload,
                    params=params,
                    attempts=retries,
                    success=True,
                )
                saved_documents.append(doc.id)

            item = {"symbol": symbol, "success": True, "saved": bool(doc)}
            if doc:
                item["document_id"] = doc.id
            if include_payload:
                item["payload"] = payload
            items.append(item)

        except Exception as exc:
            doc = None
            if persist:
                company = _upsert_company(row)
                doc = _save_document(
                    company,
                    kind=endpoint,
                    endpoint="",
                    payload={},
                    params=params,
                    attempts=retries,
                    success=False,
                    error_message=str(exc),
                )
                saved_documents.append(doc.id)
            items.append(
                {
                    "symbol": symbol,
                    "success": False,
                    "error": str(exc),
                    "saved": bool(doc),
                    "document_id": doc.id if doc else None,
                }
            )

    return {
        "endpoint": endpoint,
        "requested": len(symbols),
        "saved_document_ids": saved_documents,
        "results": items,
    }
