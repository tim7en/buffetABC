from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_GET
from django.core.paginator import Paginator
from django.db.models import Q

from edgar import sp500
from edgar.models import EdgarCompany, EdgarDocument
from edgar.services.edgar_client import EdgarClient


@require_GET
def company_search(request):
    query = request.GET.get("q", "").strip()
    limit = int(request.GET.get("limit", "25"))
    if not query:
        return HttpResponseBadRequest("missing query parameter: q")
    results = sp500.search_companies(query, limit=max(1, min(limit, 100)))
    return JsonResponse({"results": results})


@require_GET
def companies(request):
    query = request.GET.get("q", "").strip()
    page = int(request.GET.get("page", "1"))
    page_size = min(int(request.GET.get("page_size", "20")), 100)

    qs = EdgarCompany.objects.all().order_by("ticker")
    if query:
        qs = qs.filter(Q(ticker__icontains=query) | Q(name__icontains=query))

    paginator = Paginator(qs, page_size)
    obj_page = paginator.get_page(page)
    data = [
        {
            "ticker": c.ticker,
            "name": c.name,
            "cik": c.cik,
            "is_sp500": c.is_sp500,
            "updated_at": c.updated_at.isoformat(),
        }
        for c in obj_page.object_list
    ]

    return JsonResponse(
        {
            "count": paginator.count,
            "num_pages": paginator.num_pages,
            "page": obj_page.number,
            "results": data,
        }
    )


@require_GET
def company_documents(request, ticker):
    include_payload = request.GET.get("include_payload") == "1"
    kind = request.GET.get("kind", "").strip()
    limit = min(int(request.GET.get("limit", "10")), 100)

    qs = EdgarDocument.objects.filter(company__ticker=ticker.upper())
    if kind:
        qs = qs.filter(kind=kind)
    docs = qs.order_by("-fetched_at")[:limit]

    results = []
    for d in docs:
        item = {
            "id": d.id,
            "ticker": d.company.ticker,
            "kind": d.kind,
            "endpoint": d.endpoint,
            "success": d.success,
            "error_message": d.error_message,
            "fetched_at": d.fetched_at.isoformat(),
            "attempt_count": d.attempt_count,
        }
        if include_payload:
            item["payload"] = d.payload
        results.append(item)

    return JsonResponse({"results": results})


@require_GET
def fulltext_search(request):
    query = request.GET.get("q", "").strip()
    if not query:
        return HttpResponseBadRequest("missing query parameter: q")
    start = int(request.GET.get("start", "0"))
    size = min(int(request.GET.get("size", "25")), 100)

    client = EdgarClient()
    data = client.full_text_search(query, start=start, size=size)
    return JsonResponse(data)
