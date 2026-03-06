from rest_framework.routers import DefaultRouter

from edgar.drf_views import (
    EdgarCompanyViewSet,
    EdgarDocumentViewSet,
    EdgarFundamentalViewSet,
    EdgarIngestionViewSet,
)

router = DefaultRouter()
router.register(r"companies", EdgarCompanyViewSet, basename="drf-edgar-company")
router.register(r"documents", EdgarDocumentViewSet, basename="drf-edgar-document")
router.register(r"fundamentals", EdgarFundamentalViewSet, basename="drf-edgar-fundamental")
router.register(r"ingestion", EdgarIngestionViewSet, basename="drf-edgar-ingestion")

urlpatterns = router.urls
