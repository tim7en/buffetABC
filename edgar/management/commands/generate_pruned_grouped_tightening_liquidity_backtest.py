import csv
import json
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand

from edgar.services.backtest_plotting import (
    plot_asset_pnl,
    plot_equity_and_drawdown,
    plot_yearly_pnl,
)
from edgar.services.sentiment_data import load_crypto_fg_scores, load_vix_closes
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    generate_session_turtle_shared_account_report,
)
from edgar.services.session_turtle_portfolio_pruned import (
    PRUNED_SESSION_TURTLE_UNIVERSE,
    build_pruned_grouped_candidates,
    grouped_channel_summary,
    removed_ticker_summary,
)
from edgar.services.tightening_liquidity_gate import build_tightening_liquidity_state


class Command(BaseCommand):
    help = "Run the pruned/grouped 10/20 Session Turtle backtest with optional tightening/liquidity gate variants."

    def add_arguments(self, parser):
        default_dir = f"generated_reports/pruned_grouped_tightening_liquidity_{date.today().strftime('%Y%m%d')}"
        parser.add_argument(
            "--output-dir",
            default=default_dir,
            help="Directory for baseline and gate-variant outputs.",
        )
        parser.add_argument(
            "--gate-modes",
            choices=["size", "entry", "both"],
            default="both",
            help="Which tightening/liquidity gate variants to run in addition to baseline.",
        )
        parser.add_argument(
            "--gate-ma-days",
            type=int,
            default=60,
            help="Moving-average window for the macro gate (default: 60).",
        )
        parser.add_argument(
            "--gate-lag-days",
            type=int,
            default=1,
            help="Calendar-day lag for the macro gate lookup (default: 1).",
        )
        parser.add_argument(
            "--gate-tight-score-threshold",
            type=int,
            default=2,
            help="Components needed to mark the gate as tight (default: 2).",
        )
        parser.add_argument(
            "--gate-buckets",
            default="crypto,equity,etf",
            help="Comma-separated asset buckets affected by the macro gate.",
        )
        parser.add_argument(
            "--gate-tight-long-mult",
            type=float,
            default=0.5,
            help="Long multiplier in tight regimes for size mode (default: 0.5).",
        )
        parser.add_argument(
            "--gate-tight-short-mult",
            type=float,
            default=1.0,
            help="Short multiplier in tight regimes (default: 1.0).",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Building grouped 10/20 candidates ...")
        candidates = build_pruned_grouped_candidates(
            basket="core",
            initial_capital=1000.0,
            lookback_years=4.1,
            base_risk_pct=0.05,
            fixed_stop_pct=0.10,
            directional_volume_risk_pct=0.07,
            use_breakout_conviction_boost=True,
            conviction_rel_volume_ratio_period=5,
            conviction_max_mult=1.25,
            conviction_rel_volume_weight=0.10,
            conviction_rel_volume_ratio_weight=0.15,
            conviction_breakout_weight=0.10,
            conviction_close_location_weight=0.05,
            trend_fast_period=55,
            trend_slow_period=200,
        )
        self.stdout.write(self.style.SUCCESS(f"  built {len(candidates)} grouped trade candidates"))

        self.stdout.write("Loading overlay state ...")
        extended_hours_state = build_extended_hours_proxy_state(
            daily_vix_closes=load_vix_closes(),
            crypto_fg_scores=load_crypto_fg_scores(),
        )
        per_asset_state = build_per_asset_technical_state(
            universe=list(PRUNED_SESSION_TURTLE_UNIVERSE),
            lookback_years=5.0,
            warmup_days=300,
            ema_period=200,
            adx_period=14,
        )
        gate_state = build_tightening_liquidity_state(
            ma_days=int(options["gate_ma_days"]),
            tight_score_threshold=int(options["gate_tight_score_threshold"]),
        )
        self.stdout.write(self.style.SUCCESS("  overlay state ready"))

        base_kwargs = {
            "basket": "core",
            "exposure_mult": 3.0,
            "use_drawdown_governor": True,
            "drawdown_trigger_1_pct": 15.0,
            "drawdown_exposure_mult_1": 1.5,
            "drawdown_trigger_2_pct": 25.0,
            "drawdown_exposure_mult_2": 0.5,
            "crypto_cap_mult": 1.0,
            "gold_cap_mult": 1.0,
            "metals_cap_mult": 1.0,
            "energy_cap_mult": 1.0,
            "equity_cap_mult": None,
            "initial_capital": 1000.0,
            "lookback_years": 4.1,
            "channel_period": 10,
            "base_risk_pct": 0.05,
            "fixed_stop_pct": 0.10,
            "directional_volume_risk_pct": 0.07,
            "use_breakout_conviction_boost": True,
            "conviction_rel_volume_ratio_period": 5,
            "conviction_max_mult": 1.25,
            "conviction_rel_volume_weight": 0.10,
            "conviction_rel_volume_ratio_weight": 0.15,
            "conviction_breakout_weight": 0.10,
            "conviction_close_location_weight": 0.05,
            "trend_fast_period": 55,
            "trend_slow_period": 200,
            "use_extended_hours_proxy": True,
            "extended_hours_proxy_state": extended_hours_state,
            "extended_hours_proxy_lag_days": 1,
            "extended_hours_vix_risk_on_threshold": 15.0,
            "extended_hours_vix_risk_off_threshold": 25.0,
            "extended_hours_fg_greed_threshold": 60.0,
            "extended_hours_fg_fear_threshold": 30.0,
            "extended_hours_long_risk_on_mult": 1.0,
            "extended_hours_long_neutral_mult": 1.0,
            "extended_hours_long_risk_off_mult": 0.5,
            "extended_hours_short_risk_on_mult": 1e-9,
            "extended_hours_short_neutral_mult": 1.0,
            "extended_hours_short_risk_off_mult": 1.0,
            "use_per_asset_technical_overlay": True,
            "per_asset_technical_state": per_asset_state,
            "per_asset_ema_lag_days": 1,
            "per_asset_ema_above_long_mult": 1.0,
            "per_asset_ema_above_short_mult": 0.5,
            "per_asset_ema_below_long_mult": 0.5,
            "per_asset_ema_below_short_mult": 1.0,
            "per_asset_use_adx_gate": False,
            "precomputed_candidates": candidates,
        }
        gate_buckets = frozenset(
            bucket.strip().lower()
            for bucket in str(options["gate_buckets"]).split(",")
            if bucket.strip()
        ) or None

        variants = [("baseline", {})]
        gate_modes = str(options["gate_modes"])
        if gate_modes in {"size", "both"}:
            variants.append(
                (
                    "tightening_liquidity_size",
                    {
                        "use_tightening_liquidity_gate": True,
                        "tightening_liquidity_state": gate_state,
                        "tightening_liquidity_label": "Monthly tightening/liquidity gate",
                        "tightening_liquidity_mode": "size",
                        "tightening_liquidity_lag_days": int(options["gate_lag_days"]),
                        "tightening_liquidity_buckets": gate_buckets,
                        "tightening_liquidity_long_tight_mult": float(options["gate_tight_long_mult"]),
                        "tightening_liquidity_long_neutral_mult": 1.0,
                        "tightening_liquidity_short_tight_mult": float(options["gate_tight_short_mult"]),
                        "tightening_liquidity_short_neutral_mult": 1.0,
                    },
                )
            )
        if gate_modes in {"entry", "both"}:
            variants.append(
                (
                    "tightening_liquidity_entry",
                    {
                        "use_tightening_liquidity_gate": True,
                        "tightening_liquidity_state": gate_state,
                        "tightening_liquidity_label": "Monthly tightening/liquidity gate",
                        "tightening_liquidity_mode": "entry",
                        "tightening_liquidity_lag_days": int(options["gate_lag_days"]),
                        "tightening_liquidity_buckets": gate_buckets,
                        "tightening_liquidity_long_tight_mult": 0.0,
                        "tightening_liquidity_long_neutral_mult": 1.0,
                        "tightening_liquidity_short_tight_mult": float(options["gate_tight_short_mult"]),
                        "tightening_liquidity_short_neutral_mult": 1.0,
                    },
                )
            )

        comparison_rows: list[dict[str, object]] = []
        for variant_name, extra_kwargs in variants:
            self.stdout.write(f"Running {variant_name} ...")
            report = generate_session_turtle_shared_account_report(
                **base_kwargs,
                **extra_kwargs,
            )
            variant_dir = output_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            self._write_report_bundle(
                variant_dir=variant_dir,
                report=report,
            )
            summary = report["summary"]
            comparison_rows.append(
                {
                    "variant": variant_name,
                    "final_equity": summary["final_equity"],
                    "total_return_pct": summary["total_return_pct"],
                    "cagr_pct": summary["cagr_pct"],
                    "max_realized_drawdown_pct": summary["max_realized_drawdown_pct"],
                    "profit_factor": summary["profit_factor"],
                    "executed_trades": summary["executed_trades"],
                    "avg_tightening_liquidity_mult": summary.get("avg_tightening_liquidity_mult"),
                    "entries_tightening_liquidity_scaled": summary.get("entries_tightening_liquidity_scaled"),
                    "skipped_tightening_liquidity_gate": summary.get("skipped_tightening_liquidity_gate"),
                }
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"  {variant_name}: CAGR {summary['cagr_pct']}%, "
                    f"max DD {summary['max_realized_drawdown_pct']}%, PF {summary['profit_factor']}"
                )
            )

        self._write_csv(output_dir / "comparison.csv", comparison_rows)
        self._write_comparison_md(output_dir / "comparison.md", comparison_rows)
        self.stdout.write(self.style.SUCCESS(f"pruned grouped macro-gate backtests written to {output_dir}"))

    def _write_report_bundle(self, *, variant_dir: Path, report: dict) -> None:
        summary_payload = {
            "groups": grouped_channel_summary(),
            "removed": removed_ticker_summary(),
            "summary": report["summary"],
        }
        (variant_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, default=self._json_default),
            encoding="utf-8",
        )
        self._write_csv(variant_dir / "trades.csv", report["trades"])
        self._write_csv(variant_dir / "equity_curve.csv", report["equity_curve"])
        self._write_csv(variant_dir / "yearly_returns.csv", report["yearly_returns"])
        self._write_csv(variant_dir / "asset_summary.csv", report["asset_summary"])

        label = str(report["summary"]["label"])
        plot_equity_and_drawdown(
            report["equity_curve"],
            variant_dir / "equity_drawdown.png",
            title=f"{label} Equity And Drawdown",
            initial_capital=float(report["summary"]["initial_capital"]),
        )
        plot_yearly_pnl(
            report["yearly_returns"],
            variant_dir / "yearly_pnl.png",
            title=f"{label} Yearly Realized PnL",
        )
        plot_asset_pnl(
            report["asset_summary"],
            variant_dir / "asset_pnl.png",
            title=f"{label} Asset PnL Contribution",
        )

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_comparison_md(self, path: Path, rows: list[dict[str, object]]) -> None:
        lines = [
            "# Pruned Grouped Tightening/Liquidity Comparison",
            "",
            "| Variant | Final Equity | Return % | CAGR % | Max DD % | PF | Trades | Gate-Scaled Entries | Gate-Blocked |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {variant} | {final_equity} | {total_return_pct} | {cagr_pct} | {max_realized_drawdown_pct} | "
                "{profit_factor} | {executed_trades} | {entries_tightening_liquidity_scaled} | "
                "{skipped_tightening_liquidity_gate} |".format(**row)
            )
        path.write_text("\n".join(lines), encoding="utf-8")

    def _json_default(self, value):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
