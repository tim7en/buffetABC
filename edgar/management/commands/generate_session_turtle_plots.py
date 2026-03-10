import csv
from pathlib import Path

from django.core.management.base import BaseCommand

from edgar.services.backtest_plotting import (
    plot_asset_pnl,
    plot_equity_and_drawdown,
    plot_yearly_pnl,
)
from edgar.services.session_turtle_portfolio import generate_session_turtle_shared_account_report


class Command(BaseCommand):
    help = "Generate CSV reports and PNG plots for the shared-account session turtle trend basket."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            default="reports/session_turtle_trend_x2",
            help="Directory for CSV and PNG outputs.",
        )
        parser.add_argument(
            "--exposure-mult",
            type=float,
            default=2.0,
            help="Shared-account gross exposure multiplier (default: 2.0).",
        )
        parser.add_argument(
            "--use-drawdown-governor",
            action="store_true",
            help="Reduce exposure after realized drawdown thresholds are hit.",
        )
        parser.add_argument(
            "--drawdown-trigger-1-pct",
            type=float,
            default=10.0,
            help="First realized drawdown threshold in percent.",
        )
        parser.add_argument(
            "--drawdown-exposure-mult-1",
            type=float,
            default=1.5,
            help="Exposure multiplier to use after the first drawdown threshold.",
        )
        parser.add_argument(
            "--drawdown-trigger-2-pct",
            type=float,
            default=20.0,
            help="Second realized drawdown threshold in percent.",
        )
        parser.add_argument(
            "--drawdown-exposure-mult-2",
            type=float,
            default=1.0,
            help="Exposure multiplier to use after the second drawdown threshold.",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        exposure_mult = float(options["exposure_mult"])
        report = generate_session_turtle_shared_account_report(
            exposure_mult=exposure_mult,
            use_drawdown_governor=bool(options["use_drawdown_governor"]),
            drawdown_trigger_1_pct=float(options["drawdown_trigger_1_pct"]),
            drawdown_exposure_mult_1=float(options["drawdown_exposure_mult_1"]),
            drawdown_trigger_2_pct=float(options["drawdown_trigger_2_pct"]),
            drawdown_exposure_mult_2=float(options["drawdown_exposure_mult_2"]),
        )
        summary = report["summary"]

        self._write_csv(output_dir / "shared_account_summary.csv", [summary])
        self._write_csv(output_dir / "shared_account_equity_curve.csv", report["equity_curve"])
        self._write_csv(output_dir / "shared_account_trades.csv", report["trades"])
        self._write_csv(output_dir / "shared_account_yearly_returns.csv", report["yearly_returns"])
        self._write_csv(output_dir / "shared_account_asset_summary.csv", report["asset_summary"])

        label = str(summary["label"])
        plot_equity_and_drawdown(
            report["equity_curve"],
            output_dir / "shared_account_equity_drawdown.png",
            title=f"{label} Equity And Drawdown",
            initial_capital=float(summary["initial_capital"]),
        )
        plot_yearly_pnl(
            report["yearly_returns"],
            output_dir / "shared_account_yearly_pnl.png",
            title=f"{label} Yearly Realized PnL",
        )
        plot_asset_pnl(
            report["asset_summary"],
            output_dir / "shared_account_asset_pnl.png",
            title=f"{label} Asset PnL Contribution",
        )

        self.stdout.write(self.style.SUCCESS(f"session turtle plots written to {output_dir}"))

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
