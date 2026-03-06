from django.contrib import admin

from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental, EdgarMetricMapping


@admin.register(EdgarCompany)
class EdgarCompanyAdmin(admin.ModelAdmin):
    list_display = ("ticker", "name", "cik", "is_sp500", "updated_at")
    list_filter = ("is_sp500",)
    search_fields = ("ticker", "name", "cik")


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
