from django.core.management.base import BaseCommand

from edgar.models import EdgarDocument
from edgar.services.fundamentals import (
    save_fundamentals_from_concept,
    save_fundamentals_from_facts,
)


class Command(BaseCommand):
    help = "Backfill normalized fundamentals from already-saved EDGAR documents."

    def add_arguments(self, parser):
        parser.add_argument("--ticker", default="", help="Optional ticker filter")

    def handle(self, *args, **options):
        ticker = options.get("ticker", "").strip().upper()
        qs = EdgarDocument.objects.filter(success=True, kind__in=["facts", "company_concept"]).select_related("company")
        if ticker:
            qs = qs.filter(company__ticker=ticker)

        total_points = 0
        for doc in qs.iterator():
            if doc.kind == EdgarDocument.KIND_FACTS:
                total_points += save_fundamentals_from_facts(
                    company=doc.company,
                    payload=doc.payload,
                    source_document=doc,
                    taxonomy_filter=None,
                    tag_filter=None,
                )
            elif doc.kind == EdgarDocument.KIND_CONCEPT:
                total_points += save_fundamentals_from_concept(
                    company=doc.company,
                    payload=doc.payload,
                    source_document=doc,
                )

        self.stdout.write(self.style.SUCCESS(f"normalized points upserted: {total_points}"))
