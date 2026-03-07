from django.contrib import admin

from edgar.models import (
    BuffettScore,
    EdgarCompany,
    EdgarDocument,
    EdgarFundamental,
    EdgarMetricMapping,
    StockPrice,
)


@admin.register(EdgarCompany)
class EdgarCompanyAdmin(admin.ModelAdmin):
    list_display = ("ticker", "name", "cik", "sector", "is_sp500", "updated_at")
    list_filter = ("is_sp500", "sector")
    search_fields = ("ticker", "name", "cik", "sector")


@admin.register(EdgarDocument)
class EdgarDocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "company",
        "kind",
        "success",
        "attempt_count",
        "http_status",
        "fetched_at",
    )
    list_filter = ("kind", "success", "fetched_at")
    search_fields = ("company__ticker", "company__name", "error_message", "endpoint")
    readonly_fields = ("fetched_at",)


@admin.register(EdgarFundamental)
class EdgarFundamentalAdmin(admin.ModelAdmin):
    list_display = (
        "company",
        "taxonomy",
        "tag",
        "unit",
        "end_date",
        "value",
        "form",
    )
    list_filter = ("taxonomy", "tag", "unit", "form")
    search_fields = ("company__ticker", "company__name", "taxonomy", "tag", "accession")


@admin.register(EdgarMetricMapping)
class EdgarMetricMappingAdmin(admin.ModelAdmin):
    list_display = ("company", "metric_key", "taxonomy", "tag", "source", "confidence", "updated_at")
    list_filter = ("source", "taxonomy")
    search_fields = ("company__ticker", "metric_key", "tag")


@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    list_display = ("company", "date", "close", "volume", "fetched_at")
    list_filter = ("date",)
    search_fields = ("company__ticker",)


@admin.register(BuffettScore)
class BuffettScoreAdmin(admin.ModelAdmin):
    list_display = (
        "company", "overall_score", "roe_score", "debt_score",
        "margin_score", "earnings_growth_score", "fcf_score",
        "valuation_score", "computed_at",
    )
    list_filter = ("computed_at",)
    search_fields = ("company__ticker", "company__name")
