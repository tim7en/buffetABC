from rest_framework.routers import DefaultRouter

from edgar.drf_views import (
    BuffettScoreViewSet,
    ChartsViewSet,
    EdgarCompanyViewSet,
    EdgarDocumentViewSet,
    EdgarFundamentalViewSet,
    EdgarIngestionViewSet,
    StockPriceViewSet,
    StrategyViewSet,
)

router = DefaultRouter()
router.register(r"companies", EdgarCompanyViewSet, basename="drf-edgar-company")
router.register(r"documents", EdgarDocumentViewSet, basename="drf-edgar-document")
router.register(r"fundamentals", EdgarFundamentalViewSet, basename="drf-edgar-fundamental")
router.register(r"ingestion", EdgarIngestionViewSet, basename="drf-edgar-ingestion")
router.register(r"prices", StockPriceViewSet, basename="drf-stock-price")
router.register(r"scores", BuffettScoreViewSet, basename="drf-buffett-score")
router.register(r"charts", ChartsViewSet, basename="drf-charts")
router.register(r"strategy", StrategyViewSet, basename="drf-strategy")

urlpatterns = router.urls
