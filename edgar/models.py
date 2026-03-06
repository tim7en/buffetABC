from django.db import models


class EdgarCompany(models.Model):
    ticker = models.CharField(max_length=16, unique=True, db_index=True)
    name = models.CharField(max_length=255, blank=True)
    cik = models.CharField(max_length=10, db_index=True)
    is_sp500 = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["ticker"]

    def __str__(self) -> str:
        return f"{self.ticker} ({self.cik})"


class EdgarDocument(models.Model):
    KIND_FACTS = "facts"
    KIND_FILINGS = "filings"
    KIND_CONCEPT = "company_concept"
    KIND_FULLTEXT = "full_text_search"
    KIND_CHOICES = (
        (KIND_FACTS, "Company Facts"),
        (KIND_FILINGS, "Filings"),
        (KIND_CONCEPT, "Company Concept"),
        (KIND_FULLTEXT, "Full Text Search"),
    )

    company = models.ForeignKey(
        EdgarCompany,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    kind = models.CharField(max_length=32, choices=KIND_CHOICES, db_index=True)
    endpoint = models.CharField(max_length=255)
    params = models.JSONField(default=dict, blank=True)
    payload = models.JSONField(default=dict, blank=True)
    source_url = models.URLField(blank=True)
    fetched_at = models.DateTimeField(auto_now_add=True, db_index=True)
    http_status = models.PositiveSmallIntegerField(null=True, blank=True)
    attempt_count = models.PositiveIntegerField(default=1)
    success = models.BooleanField(default=True, db_index=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-fetched_at"]
        indexes = [
            models.Index(fields=["company", "kind", "-fetched_at"]),
            models.Index(fields=["kind", "success"]),
        ]

    def __str__(self) -> str:
        state = "ok" if self.success else "failed"
        return f"{self.company.ticker}:{self.kind}:{state}@{self.fetched_at.isoformat()}"


class EdgarFundamental(models.Model):
    company = models.ForeignKey(
        EdgarCompany,
        on_delete=models.CASCADE,
        related_name="fundamentals",
    )
    source_document = models.ForeignKey(
        EdgarDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fundamental_points",
    )
    taxonomy = models.CharField(max_length=64, db_index=True)
    tag = models.CharField(max_length=128, db_index=True)
    unit = models.CharField(max_length=32, db_index=True)
    end_date = models.DateField(db_index=True)
    filed_date = models.DateField(null=True, blank=True, db_index=True)
    value = models.FloatField()
    form = models.CharField(max_length=32, blank=True, db_index=True)
    fiscal_year = models.IntegerField(null=True, blank=True)
    fiscal_period = models.CharField(max_length=16, blank=True)
    accession = models.CharField(max_length=24, blank=True, db_index=True)
    frame = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["company__ticker", "taxonomy", "tag", "end_date"]
        indexes = [
            models.Index(fields=["company", "end_date"]),
            models.Index(fields=["company", "taxonomy", "tag", "end_date"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "company",
                    "taxonomy",
                    "tag",
                    "unit",
                    "end_date",
                    "filed_date",
                    "form",
                    "fiscal_year",
                    "fiscal_period",
                    "accession",
                    "frame",
                ],
                name="edgar_unique_fundamental_point",
            )
        ]

    def __str__(self) -> str:
        return f"{self.company.ticker}:{self.taxonomy}:{self.tag}:{self.end_date}={self.value}"
