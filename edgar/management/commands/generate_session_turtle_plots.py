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
            "--basket",
            choices=["core", "index", "expanded"],
            default="expanded",
            help="Asset basket to replay (default: expanded).",
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
        parser.add_argument(
            "--crypto-cap-mult",
            type=float,
            default=None,
            help="Optional cap for the crypto asset class, expressed as an exposure multiplier.",
        )
        parser.add_argument(
            "--gold-cap-mult",
            type=float,
            default=None,
            help="Optional cap for the gold asset class, expressed as an exposure multiplier.",
        )
        parser.add_argument(
            "--metals-cap-mult",
            type=float,
            default=None,
            help="Optional cap for the metals asset class, expressed as an exposure multiplier.",
        )
        parser.add_argument(
            "--equity-cap-mult",
            type=float,
            default=None,
            help="Optional cap for the equity asset class, expressed as an exposure multiplier.",
        )
        parser.add_argument(
            "--base-risk-pct",
            type=float,
            default=5.0,
            help="Base per-trade risk as a percent of account equity (default: 5.0).",
        )
        parser.add_argument(
            "--fixed-stop-pct",
            type=float,
            default=10.0,
            help="Fixed stop distance as a percent of entry price (default: 10.0).",
        )
        parser.add_argument(
            "--directional-volume-risk-pct",
            type=float,
            default=7.0,
            help="Risk percent to use for directional volume-boosted entries (default: 7.0).",
        )
        parser.add_argument(
            "--use-performance-leadership-overlay",
            action="store_true",
            help="Scale each asset's position size using decayed recent paper-performance versus peers.",
        )
        parser.add_argument(
            "--performance-lookback-trades",
            type=int,
            default=6,
            help="Closed trades per asset to score for the leadership overlay (default: 6).",
        )
        parser.add_argument(
            "--performance-decay",
            type=float,
            default=0.75,
            help="Exponential decay for older asset scores in the leadership overlay (default: 0.75).",
        )
        parser.add_argument(
            "--performance-floor-mult",
            type=float,
            default=0.75,
            help="Minimum position-size multiplier for lagging assets (default: 0.75).",
        )
        parser.add_argument(
            "--performance-cap-mult",
            type=float,
            default=1.25,
            help="Maximum position-size multiplier for leading assets (default: 1.25).",
        )
        parser.add_argument(
            "--performance-min-history",
            type=int,
            default=3,
            help="Closed trades required before the leadership overlay sizes an asset away from neutral (default: 3).",
        )
        parser.add_argument(
            "--use-extended-hours-protective-exits",
            action="store_true",
            help="For New York session Tiingo assets, keep entries in core hours but allow protective stops outside them.",
        )
        parser.add_argument(
            "--extended-hours-core-session-minutes",
            type=int,
            default=390,
            help="Core-session length used before outside-hours protective exits take over (default: 390).",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        exposure_mult = float(options["exposure_mult"])
        report = generate_session_turtle_shared_account_report(
            basket=str(options["basket"]),
            exposure_mult=exposure_mult,
            use_drawdown_governor=bool(options["use_drawdown_governor"]),
            drawdown_trigger_1_pct=float(options["drawdown_trigger_1_pct"]),
            drawdown_exposure_mult_1=float(options["drawdown_exposure_mult_1"]),
            drawdown_trigger_2_pct=float(options["drawdown_trigger_2_pct"]),
            drawdown_exposure_mult_2=float(options["drawdown_exposure_mult_2"]),
            crypto_cap_mult=float(options["crypto_cap_mult"]) if options["crypto_cap_mult"] is not None else None,
            gold_cap_mult=float(options["gold_cap_mult"]) if options["gold_cap_mult"] is not None else None,
            metals_cap_mult=float(options["metals_cap_mult"]) if options["metals_cap_mult"] is not None else None,
            equity_cap_mult=float(options["equity_cap_mult"]) if options["equity_cap_mult"] is not None else None,
            base_risk_pct=float(options["base_risk_pct"]) / 100.0,
            fixed_stop_pct=float(options["fixed_stop_pct"]) / 100.0,
            directional_volume_risk_pct=float(options["directional_volume_risk_pct"]) / 100.0,
            use_performance_leadership_overlay=bool(options["use_performance_leadership_overlay"]),
            performance_lookback_trades=int(options["performance_lookback_trades"]),
            performance_decay=float(options["performance_decay"]),
            performance_floor_mult=float(options["performance_floor_mult"]),
            performance_cap_mult=float(options["performance_cap_mult"]),
            performance_min_history=int(options["performance_min_history"]),
            use_extended_hours_protective_exits=bool(options["use_extended_hours_protective_exits"]),
            extended_hours_core_session_minutes=int(options["extended_hours_core_session_minutes"]),
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
