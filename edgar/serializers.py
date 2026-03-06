from rest_framework import serializers

from edgar.models import EdgarCompany, EdgarDocument, EdgarFundamental


class EdgarDocumentSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source="company.ticker", read_only=True)

    class Meta:
        model = EdgarDocument
        fields = [
            "id",
            "ticker",
            "kind",
            "endpoint",
            "params",
            "payload",
            "source_url",
            "fetched_at",
            "http_status",
            "attempt_count",
            "success",
            "error_message",
        ]


class EdgarCompanySerializer(serializers.ModelSerializer):
    documents_count = serializers.IntegerField(source="documents.count", read_only=True)

    class Meta:
        model = EdgarCompany
        fields = [
            "id",
            "ticker",
            "name",
            "cik",
            "is_sp500",
            "created_at",
            "updated_at",
            "documents_count",
        ]


class IngestionRequestSerializer(serializers.Serializer):
    symbols = serializers.ListField(
        child=serializers.CharField(max_length=16), required=False, allow_empty=False
    )
    search_name = serializers.CharField(required=False, allow_blank=True)
    endpoint = serializers.ChoiceField(
        choices=[
            EdgarDocument.KIND_FACTS,
            EdgarDocument.KIND_FILINGS,
            EdgarDocument.KIND_CONCEPT,
            EdgarDocument.KIND_FULLTEXT,
        ],
        default=EdgarDocument.KIND_FACTS,
    )
    taxonomy = serializers.CharField(required=False, default="us-gaap")
    tag = serializers.CharField(required=False, default="Assets")
    query = serializers.CharField(required=False, allow_blank=True)
    limit = serializers.IntegerField(required=False, min_value=1, max_value=500)
    retries = serializers.IntegerField(required=False, min_value=1, max_value=10, default=3)
    backoff = serializers.FloatField(required=False, min_value=0.0, max_value=30.0, default=1.0)
    persist = serializers.BooleanField(required=False, default=True)
    include_payload = serializers.BooleanField(required=False, default=False)
    period_start = serializers.DateField(required=False)
    period_end = serializers.DateField(required=False)

    def validate(self, attrs):
        symbols = attrs.get("symbols") or []
        search_name = attrs.get("search_name", "").strip()
        endpoint = attrs.get("endpoint", EdgarDocument.KIND_FACTS)
        query = attrs.get("query", "").strip()

        if endpoint == EdgarDocument.KIND_FULLTEXT:
            if not query and not search_name and not symbols:
                raise serializers.ValidationError(
                    "For full_text_search provide query, symbols, or search_name."
                )
        else:
            if not symbols and not search_name:
                raise serializers.ValidationError("Provide symbols or search_name.")

        period_start = attrs.get("period_start")
        period_end = attrs.get("period_end")
        if period_start and period_end and period_start > period_end:
            raise serializers.ValidationError("period_start must be <= period_end")

        return attrs


class EdgarFundamentalSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source="company.ticker", read_only=True)

    class Meta:
        model = EdgarFundamental
        fields = [
            "id",
            "ticker",
            "taxonomy",
            "tag",
            "unit",
            "end_date",
            "filed_date",
            "value",
            "form",
            "fiscal_year",
            "fiscal_period",
            "accession",
            "frame",
            "source_document",
            "updated_at",
        ]
