from django.core.management import BaseCommand, call_command


class Command(BaseCommand):
    help = "Nightly sync job for EDGAR facts/filings. Suitable for cron or Celery beat wrappers."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional limit for partial sync runs.",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Optional output directory for raw snapshots.",
        )

    def handle(self, *args, **options):
        common_args = ["--persist", f"--retries=3", "--backoff=1.5"]
        if options.get("limit"):
            common_args.append(f"--limit={options['limit']}")
        if options.get("output"):
            common_args.append(f"--output={options['output']}")

        self.stdout.write("starting nightly EDGAR sync (facts)")
        call_command("fetch_edgar", *common_args, "--facts")

        self.stdout.write("starting nightly EDGAR sync (filings)")
        call_command("fetch_edgar", *common_args, "--filings")

        self.stdout.write(self.style.SUCCESS("nightly EDGAR sync complete"))
