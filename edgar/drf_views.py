from datetime import date

from django.db.models import Q
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from edgar import sp500
from edgar.models import BuffettScore, EdgarCompany, EdgarDocument, EdgarFundamental, StockPrice
from edgar.serializers import (
    BuffettScoreSerializer,
    EdgarCompanySerializer,
    EdgarDocumentSerializer,
    EdgarFundamentalSerializer,
    IngestionRequestSerializer,
    StockPriceSerializer,
)
from edgar.services.fundamentals import (
    save_fundamentals_from_concept,
    save_fundamentals_from_facts,
)
from edgar.services.metric_mapping import build_fundamental_table, resolve_and_store_metric_mapping
from edgar.services.edgar_client import EdgarClient


def _row_for_symbol(symbol: str):
    row = sp500.by_symbol(symbol)
    if row:
        return row
    return {"Symbol": symbol.upper(), "CIK": "", "Security": "", "GICS Sector": "", "GICS Sub-Industry": ""}


def _upsert_company(row):
    ticker = (row.get("Symbol") or "").upper()
    cik = str(row.get("CIK") or row.get("cik") or "").zfill(10)
    name = row.get("Security") or row.get("Name") or ""
    sector = row.get("GICS Sector") or row.get("sector") or ""
    sub_industry = row.get("GICS Sub-Industry") or row.get("sub_industry") or ""
    company, _ = EdgarCompany.objects.update_or_create(
        ticker=ticker,
        defaults={
            "cik": cik or "0000000000",
            "name": name,
            "sector": sector,
            "sub_industry": sub_industry,
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

        qs = EdgarFundamental.objects.filter(
            company=company,
            taxonomy=taxonomy,
            tag=tag,
            unit=unit,
        ).order_by("end_date", "filed_date")
        if start_date:
            qs = qs.filter(end_date__gte=start_date)
        if end_date:
            qs = qs.filter(end_date__lte=end_date)
        points = [
            {
                "end": p.end_date.isoformat(),
                "filed": p.filed_date.isoformat() if p.filed_date else "",
                "val": p.value,
                "form": p.form,
                "fy": p.fiscal_year,
                "fp": p.fiscal_period,
                "accn": p.accession,
                "frame": p.frame,
            }
            for p in qs
        ]

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

    @action(detail=True, methods=["get"], url_path="fundamental-table")
    def fundamental_table(self, request, pk=None):
        company = self.get_object()
        period_start = request.query_params.get("period_start")
        period_end = request.query_params.get("period_end")
        frequency = request.query_params.get("frequency", "annual")
        use_ai = request.query_params.get("use_ai", "1").strip() != "0"
        refresh_mapping = request.query_params.get("refresh_mapping", "0").strip() == "1"

        start_date = date.fromisoformat(period_start) if period_start else None
        end_date = date.fromisoformat(period_end) if period_end else None

        mapping = resolve_and_store_metric_mapping(
            company=company,
            taxonomy="us-gaap",
            use_ai=(use_ai and refresh_mapping),
            force_refresh=refresh_mapping,
            persist=refresh_mapping,
        )
        table = build_fundamental_table(
            company=company,
            mapping=mapping,
            period_start=start_date,
            period_end=end_date,
            frequency=frequency,
        )
        return Response(
            {
                "ticker": company.ticker,
                "frequency": frequency,
                "row_count": len(table["rows"]),
                "metrics": table["metrics"],
                "mapping": table["mapping"],
                "rows": table["rows"],
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


class EdgarFundamentalViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EdgarFundamental.objects.select_related("company").all().order_by("-end_date")
    serializer_class = EdgarFundamentalSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        ticker = self.request.query_params.get("ticker", "").strip().upper()
        taxonomy = self.request.query_params.get("taxonomy", "").strip()
        tag = self.request.query_params.get("tag", "").strip()
        unit = self.request.query_params.get("unit", "").strip()
        from_date = self.request.query_params.get("from", "").strip()
        to_date = self.request.query_params.get("to", "").strip()
        if ticker:
            qs = qs.filter(company__ticker=ticker)
        if taxonomy:
            qs = qs.filter(taxonomy=taxonomy)
        if tag:
            qs = qs.filter(tag=tag)
        if unit:
            qs = qs.filter(unit=unit)
        if from_date:
            qs = qs.filter(end_date__gte=from_date)
        if to_date:
            qs = qs.filter(end_date__lte=to_date)
        return qs


class EdgarIngestionViewSet(viewsets.ViewSet):
    authentication_classes = []
    permission_classes = [AllowAny]

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
    fundamentals_saved = 0

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
            company = None
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
                if endpoint == EdgarDocument.KIND_FACTS:
                    fundamentals_saved += save_fundamentals_from_facts(
                        company=company,
                        payload=payload,
                        source_document=doc,
                        taxonomy_filter=None,
                        tag_filter=None,
                    )
                elif endpoint == EdgarDocument.KIND_CONCEPT:
                    fundamentals_saved += save_fundamentals_from_concept(
                        company=company,
                        payload=payload,
                        source_document=doc,
                    )

            item = {"symbol": symbol, "success": True, "saved": bool(doc)}
            if doc:
                item["document_id"] = doc.id
                item["company_id"] = company.id
            if include_payload:
                item["payload"] = payload
            items.append(item)

        except Exception as exc:
            err_msg = str(exc)
            if endpoint == EdgarDocument.KIND_FULLTEXT and "403" in err_msg:
                err_msg = (
                    "SEC full-text search returned 403. This endpoint is often blocked "
                    "without approved SEC access profile or when fair-access limits are hit."
                )
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
                    error_message=err_msg,
                )
                saved_documents.append(doc.id)
            items.append(
                {
                    "symbol": symbol,
                    "success": False,
                    "error": err_msg,
                    "saved": bool(doc),
                    "document_id": doc.id if doc else None,
                    "company_id": company.id if doc and company else None,
                }
            )

    return {
        "endpoint": endpoint,
        "requested": len(symbols),
        "saved_document_ids": saved_documents,
        "fundamentals_saved": fundamentals_saved,
        "results": items,
    }


# ---------------------------------------------------------------------------
# Stock Price ViewSet
# ---------------------------------------------------------------------------

class StockPriceViewSet(viewsets.ReadOnlyModelViewSet):
    authentication_classes = []
    permission_classes = [AllowAny]
    queryset = StockPrice.objects.select_related("company").all().order_by("-date")
    serializer_class = StockPriceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        ticker = self.request.query_params.get("ticker", "").strip().upper()
        if ticker:
            qs = qs.filter(company__ticker=ticker)
        return qs

    @action(detail=False, methods=["post"], url_path="refresh")
    def refresh(self, request):
        from edgar.services.stock_price import bulk_refresh_prices
        tickers = request.data.get("tickers")
        period = request.data.get("period", "1y")
        results = bulk_refresh_prices(tickers=tickers, period=period)
        return Response({"refreshed": results})

    @action(detail=False, methods=["get"], url_path="quote")
    def quote(self, request):
        from edgar.services.stock_price import fetch_current_quote
        ticker = request.query_params.get("ticker", "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=status.HTTP_400_BAD_REQUEST)
        data = fetch_current_quote(ticker)
        if data is None:
            return Response({"error": "failed to fetch quote"}, status=status.HTTP_502_BAD_GATEWAY)
        return Response(data)


# ---------------------------------------------------------------------------
# Buffett Score ViewSet
# ---------------------------------------------------------------------------

class BuffettScoreViewSet(viewsets.ReadOnlyModelViewSet):
    authentication_classes = []
    permission_classes = [AllowAny]
    queryset = BuffettScore.objects.select_related("company").all().order_by("-overall_score")
    serializer_class = BuffettScoreSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        ticker = self.request.query_params.get("ticker", "").strip().upper()
        sector = self.request.query_params.get("sector", "").strip()
        if ticker:
            qs = qs.filter(company__ticker=ticker)
        if sector:
            qs = qs.filter(company__sector__icontains=sector)
        return qs

    @action(detail=False, methods=["post"], url_path="compute")
    def compute(self, request):
        from edgar.services.buffett_score import bulk_compute_scores
        tickers = request.data.get("tickers")
        force = request.data.get("force_refresh", False)
        results = bulk_compute_scores(tickers=tickers, force_refresh=force)
        return Response({"scores": results})

    @action(detail=False, methods=["post"], url_path="compute-single")
    def compute_single(self, request):
        from edgar.services.buffett_score import compute_buffett_score
        ticker = (request.data.get("ticker") or "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            company = EdgarCompany.objects.get(ticker=ticker)
        except EdgarCompany.DoesNotExist:
            return Response({"error": f"company {ticker} not found"}, status=status.HTTP_404_NOT_FOUND)
        force = request.data.get("force_refresh", False)
        score = compute_buffett_score(company, force_refresh=force)
        return Response(BuffettScoreSerializer(score).data)


# ---------------------------------------------------------------------------
# Charts / Exploration ViewSet
# ---------------------------------------------------------------------------

class StrategyViewSet(viewsets.ViewSet):
    authentication_classes = []
    permission_classes = [AllowAny]

    @action(detail=False, methods=["post"], url_path="backtest")
    def backtest(self, request):
        from edgar.services.strategy import backtest_to_dict, run_backtest

        def _as_bool(value, default=True):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            return text not in {"0", "false", "no", "off", ""}

        ticker = (request.data.get("ticker") or "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=status.HTTP_400_BAD_REQUEST)
        capital = float(request.data.get("initial_capital", 100))
        force_fetch = request.data.get("force_fetch", False)
        fetch_period = request.data.get("fetch_period", "5y")
        allow_shorts = _as_bool(request.data.get("allow_shorts"), True)
        allow_longs = _as_bool(request.data.get("allow_longs"), True)
        require_fractal_confirmation = _as_bool(request.data.get("require_fractal_confirmation"), True)
        require_fractal_breakout = _as_bool(request.data.get("require_fractal_breakout"), False)
        lookback_years = int(request.data.get("lookback_years", 5))
        if isinstance(fetch_period, str) and fetch_period.endswith("y") and fetch_period[:-1].isdigit():
            lookback_years = int(fetch_period[:-1])
        try:
            result = run_backtest(
                ticker=ticker,
                initial_capital=capital,
                allow_longs=allow_longs,
                allow_shorts=allow_shorts,
                require_fractal_confirmation=require_fractal_confirmation,
                require_fractal_breakout=require_fractal_breakout,
                lookback_years=lookback_years,
                force_fetch=force_fetch,
                fetch_period=fetch_period,
            )
            return Response(backtest_to_dict(result))
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"], url_path="backtest-intraday")
    def backtest_intraday(self, request):
        from edgar.services.intraday_strategy import run_intraday_backtest
        from edgar.services.manipulation_strategy import run_manipulation_backtest

        def _as_bool(value, default=True):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            return text not in {"0", "false", "no", "off", ""}

        ticker = (request.data.get("ticker") or "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=status.HTTP_400_BAD_REQUEST)

        capital = float(request.data.get("initial_capital", 10_000))
        interval = (request.data.get("interval") or "15m").strip()
        lookback_years = float(request.data.get("lookback_years", 2))
        strategy_variant = (request.data.get("strategy_variant") or "fractal_breakout_ema200").strip()
        allow_shorts = _as_bool(request.data.get("allow_shorts"), True)
        allow_longs = _as_bool(request.data.get("allow_longs"), True)

        try:
            if strategy_variant == "manipulation_ifvg":
                payload = run_manipulation_backtest(
                    ticker=ticker,
                    initial_capital=capital,
                    interval=interval,
                    lookback_years=lookback_years,
                    allow_longs=allow_longs,
                    allow_shorts=allow_shorts,
                    pivot_window=int(request.data.get("pivot_window", 3)),
                    liquidity_search_window=int(request.data.get("liquidity_search_window", 240)),
                    manipulation_max_age_bars=int(request.data.get("manipulation_max_age_bars", 14)),
                    ifvg_proximity_bars=int(request.data.get("ifvg_proximity_bars", 16)),
                    sweep_buffer_bps=float(request.data.get("sweep_buffer_bps", 0.0)),
                    recovery_buffer_bps=float(request.data.get("recovery_buffer_bps", 0.0)),
                    ifvg_break_buffer_bps=float(
                        request.data.get(
                            "ifvg_break_buffer_bps",
                            request.data.get("breakout_buffer_bps", 0.0),
                        )
                    ),
                    stop_buffer_bps=float(request.data.get("stop_buffer_bps", 3.0)),
                    rr_multiple=float(request.data.get("rr_multiple", 2.0)),
                    volume_period=int(request.data.get("volume_period", 40)),
                    use_volume_filter=_as_bool(request.data.get("use_volume_filter"), False),
                    min_rel_volume=float(request.data.get("min_rel_volume", 1.0)),
                    base_risk_pct=float(request.data.get("base_risk_pct", 0.01)),
                    max_risk_pct=float(request.data.get("max_risk_pct", 0.02)),
                    max_position_pct=float(request.data.get("max_position_pct", 0.30)),
                    slippage_bps=float(request.data.get("slippage_bps", 4.0)),
                    commission_bps=float(request.data.get("commission_bps", 1.0)),
                )
            else:
                payload = run_intraday_backtest(
                    ticker=ticker,
                    initial_capital=capital,
                    interval=interval,
                    lookback_years=lookback_years,
                    strategy_variant=strategy_variant,
                    allow_longs=allow_longs,
                    allow_shorts=allow_shorts,
                    use_volume_filter=_as_bool(request.data.get("use_volume_filter"), False),
                    min_rel_volume=float(request.data.get("min_rel_volume", 1.0)),
                    oversold=float(request.data.get("oversold", 20.0)),
                    overbought=float(request.data.get("overbought", 80.0)),
                    fractal_window=int(request.data.get("fractal_window", 9)),
                    rr_multiple=float(request.data.get("rr_multiple", 1.5)),
                    breakout_buffer_bps=float(request.data.get("breakout_buffer_bps", 0.0)),
                )
            return Response(payload)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class ChartsViewSet(viewsets.ViewSet):
    authentication_classes = []
    permission_classes = [AllowAny]

    @action(detail=False, methods=["get"], url_path="sector-distribution")
    def sector_distribution(self, request):
        from edgar.services.charts import sector_distribution
        sp500_only = request.query_params.get("sp500_only", "1").strip() != "0"
        return Response(sector_distribution(sp500_only=sp500_only))

    @action(detail=False, methods=["get"], url_path="score-distribution")
    def score_distribution(self, request):
        from edgar.services.charts import score_distribution
        return Response(score_distribution())

    @action(detail=False, methods=["get"], url_path="sector-scores")
    def sector_scores(self, request):
        from edgar.services.charts import sector_score_comparison
        return Response(sector_score_comparison())

    @action(detail=False, methods=["get"], url_path="metric-trend")
    def metric_trend(self, request):
        from edgar.services.charts import metric_trend
        company_id = request.query_params.get("company_id")
        metric_key = request.query_params.get("metric", "revenue")
        if not company_id:
            return Response({"error": "company_id required"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(metric_trend(int(company_id), metric_key))

    @action(detail=False, methods=["get"], url_path="price-history")
    def price_history(self, request):
        from edgar.services.charts import price_history
        company_id = request.query_params.get("company_id")
        days = int(request.query_params.get("days", "365"))
        if not company_id:
            return Response({"error": "company_id required"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(price_history(int(company_id), period_days=days))

    @action(detail=False, methods=["get"], url_path="score-breakdown")
    def score_breakdown(self, request):
        from edgar.services.charts import company_score_breakdown
        company_id = request.query_params.get("company_id")
        if not company_id:
            return Response({"error": "company_id required"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(company_score_breakdown(int(company_id)))

    @action(detail=False, methods=["get"], url_path="top-scores")
    def top_scores(self, request):
        from edgar.services.charts import top_scored_companies
        limit = int(request.query_params.get("limit", "20"))
        return Response(top_scored_companies(limit=limit))
