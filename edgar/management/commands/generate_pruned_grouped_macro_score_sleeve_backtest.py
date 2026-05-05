import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from django.core.management.base import BaseCommand

from edgar.services.backtest_plotting import (
    plot_asset_pnl,
    plot_equity_and_drawdown,
    plot_yearly_pnl,
)
from edgar.services.macro_regime_score import (
    build_macro_regime_score_state,
    lookup_macro_regime_signal,
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


class Command(BaseCommand):
    help = "Run a separate grouped/timed macro-score sleeve backtest with optional weekend flat handling."

    def add_arguments(self, parser):
        default_dir = f"generated_reports/pruned_grouped_macro_score_sleeve_{date.today().strftime('%Y%m%d')}"
        parser.add_argument(
            "--output-dir",
            default=default_dir,
            help="Directory for macro sleeve comparison outputs.",
        )
        parser.add_argument(
            "--macro-lag-days",
            type=int,
            default=1,
            help="Calendar-day lag for macro score lookup (default: 1).",
        )
        parser.add_argument(
            "--macro-version",
            default="v1",
            help="Macro regime model version: v1 or v2_front_end (default: v1).",
        )
        parser.add_argument(
            "--macro-gated-buckets",
            default="crypto,equity,etf",
            help="Comma-separated buckets gated by the macro sleeve.",
        )
        parser.add_argument(
            "--long-half-mult",
            type=float,
            default=0.5,
            help="Long multiplier when macro score is 0 to +1 (default: 0.5).",
        )
        parser.add_argument(
            "--short-half-mult",
            type=float,
            default=0.5,
            help="Short multiplier when macro score is -1 to 0 (default: 0.5).",
        )
        parser.add_argument(
            "--negative-long-mult",
            type=float,
            default=0.0,
            help="Long multiplier when macro score is negative (default: 0.0).",
        )
        parser.add_argument(
            "--positive-short-mult",
            type=float,
            default=0.0,
            help="Short multiplier when macro score is positive (default: 0.0).",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading shared overlay state ...")
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
        macro_version = str(options["macro_version"]).strip() or "v1"
        macro_state = build_macro_regime_score_state(version=macro_version)
        self.stdout.write(self.style.SUCCESS("  overlay state ready"))

        self.stdout.write("Building grouped/timed baseline candidates ...")
        baseline_candidates = build_pruned_grouped_candidates(
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
            close_positions_before_weekend=False,
        )
        self.stdout.write(self.style.SUCCESS(f"  built {len(baseline_candidates)} baseline candidates"))

        self.stdout.write("Building weekend-flat candidates ...")
        weekend_candidates = build_pruned_grouped_candidates(
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
            close_positions_before_weekend=True,
        )
        self.stdout.write(self.style.SUCCESS(f"  built {len(weekend_candidates)} weekend-flat candidates"))

        gated_buckets = frozenset(
            bucket.strip().lower()
            for bucket in str(options["macro_gated_buckets"]).split(",")
            if bucket.strip()
        ) or None
        macro_kwargs = {
            "version": macro_version,
            "lag_days": int(options["macro_lag_days"]),
            "gated_buckets": gated_buckets,
            "long_half_mult": float(options["long_half_mult"]),
            "negative_long_mult": float(options["negative_long_mult"]),
            "short_half_mult": float(options["short_half_mult"]),
            "positive_short_mult": float(options["positive_short_mult"]),
        }

        variants = [
            ("baseline", baseline_candidates, False),
            ("macro_score_sleeve", baseline_candidates, True),
            ("weekend_flat", weekend_candidates, False),
            ("macro_score_weekend_flat", weekend_candidates, True),
        ]

        base_report_kwargs = {
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
        }

        comparison_rows: list[dict[str, object]] = []
        for variant_name, source_candidates, use_macro in variants:
            if use_macro:
                transformed_candidates, macro_lookup, candidate_stats = self._apply_macro_sleeve(
                    candidates=source_candidates,
                    macro_state=macro_state,
                    macro_kwargs=macro_kwargs,
                )
            else:
                transformed_candidates = [dict(candidate) for candidate in source_candidates]
                macro_lookup = self._baseline_annotation_lookup(source_candidates)
                candidate_stats = {
                    "macro_candidate_total": len(source_candidates),
                    "macro_candidate_blocked": 0,
                    "macro_candidate_scaled": 0,
                    "macro_action_counts": {},
                }

            self.stdout.write(f"Running {variant_name} ...")
            report = generate_session_turtle_shared_account_report(
                **base_report_kwargs,
                precomputed_candidates=transformed_candidates,
            )
            enriched_trades, summary_extras = self._enrich_trades(
                report["trades"],
                macro_lookup=macro_lookup,
                use_macro=use_macro,
            )
            report["trades"] = enriched_trades

            variant_dir = output_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            self._write_report_bundle(
                variant_dir=variant_dir,
                report=report,
                summary_extras={
                    **candidate_stats,
                    **summary_extras,
                    "uses_macro_score_sleeve": use_macro,
                    "uses_weekend_flat": variant_name in {"weekend_flat", "macro_score_weekend_flat"},
                },
                macro_config=macro_kwargs,
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
                    "macro_candidate_blocked": candidate_stats["macro_candidate_blocked"],
                    "macro_candidate_scaled": candidate_stats["macro_candidate_scaled"],
                    "weekend_spanning_trades": summary_extras["weekend_spanning_trades"],
                    "hong_kong_trades": summary_extras["session_counts"].get("hong_kong_open", 0),
                    "new_york_trades": summary_extras["session_counts"].get("new_york_equity_open", 0),
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
        self.stdout.write(self.style.SUCCESS(f"macro sleeve backtests written to {output_dir}"))

    def _apply_macro_sleeve(
        self,
        *,
        candidates: list[dict],
        macro_state: dict,
        macro_kwargs: dict[str, object],
    ) -> tuple[list[dict], dict[tuple[str, str, str, str, str, str], dict[str, object]], dict[str, object]]:
        transformed: list[dict] = []
        macro_lookup: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
        action_counts = Counter()
        scaled_count = 0
        blocked_count = 0

        for candidate in candidates:
            key = self._trade_key(candidate)
            signal = lookup_macro_regime_signal(
                entry_ts=candidate["entry_ts"],
                asset_bucket=str(candidate["asset_bucket"]),
                direction=str(candidate["direction"]),
                state=macro_state,
                lag_days=int(macro_kwargs["lag_days"]),
                gated_buckets=macro_kwargs["gated_buckets"],
                long_half_mult=float(macro_kwargs["long_half_mult"]),
                negative_long_mult=float(macro_kwargs["negative_long_mult"]),
                short_half_mult=float(macro_kwargs["short_half_mult"]),
                positive_short_mult=float(macro_kwargs["positive_short_mult"]),
            )
            signal = dict(signal)
            signal["channel_period"] = candidate.get("channel_period")
            signal["exit_channel_period"] = candidate.get("exit_channel_period")
            macro_lookup[key] = signal
            action_counts[str(signal["action"])] += 1

            if bool(signal["blocked"]):
                blocked_count += 1
                continue

            transformed_candidate = dict(candidate)
            mult = float(signal["mult"])
            if abs(mult - 1.0) > 1e-9:
                scaled_count += 1
                for field in ("risk_pct", "position_size", "shares", "pnl"):
                    if transformed_candidate.get(field) is not None:
                        transformed_candidate[field] = float(transformed_candidate[field]) * mult
            transformed.append(transformed_candidate)

        return (
            transformed,
            macro_lookup,
            {
                "macro_candidate_total": len(candidates),
                "macro_candidate_blocked": blocked_count,
                "macro_candidate_scaled": scaled_count,
                "macro_action_counts": dict(action_counts),
            },
        )

    def _baseline_annotation_lookup(
        self,
        candidates: list[dict],
    ) -> dict[tuple[str, str, str, str, str, str], dict[str, object]]:
        lookup: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
        for candidate in candidates:
            lookup[self._trade_key(candidate)] = {
                "score_date": None,
                "score": None,
                "raw_score": None,
                "label": "baseline",
                "mult": 1.0,
                "blocked": False,
                "action": "baseline",
                "components": {},
                "channel_period": candidate.get("channel_period"),
                "exit_channel_period": candidate.get("exit_channel_period"),
            }
        return lookup

    def _enrich_trades(
        self,
        trades: list[dict],
        *,
        macro_lookup: dict[tuple[str, str, str, str, str, str], dict[str, object]],
        use_macro: bool,
    ) -> tuple[list[dict], dict[str, object]]:
        enriched: list[dict] = []
        label_counts = Counter()
        action_counts = Counter()
        session_counts = Counter()
        session_pnl = Counter()
        weekend_spanning = 0

        for trade in trades:
            key = (
                str(trade["ticker"]),
                str(trade["source"]),
                str(trade["session_open"]),
                str(trade["direction"]),
                str(trade["entry_ts"]),
                str(trade["exit_ts"]),
            )
            signal = macro_lookup.get(
                key,
                {
                    "score_date": None,
                    "score": None,
                    "raw_score": None,
                    "label": "missing",
                    "mult": 1.0,
                    "blocked": False,
                    "action": "missing",
                    "components": {},
                    "channel_period": None,
                    "exit_channel_period": None,
                },
            )
            row = dict(trade)
            row["macro_score_date"] = signal.get("score_date")
            row["macro_regime_score"] = signal.get("score")
            row["macro_regime_raw_score"] = signal.get("raw_score")
            row["macro_regime_label"] = signal.get("label")
            row["macro_regime_action"] = signal.get("action")
            row["macro_regime_mult"] = signal.get("mult")
            components = dict(signal.get("components") or {})
            row["macro_component_dollar"] = components.get("dollar")
            row["macro_component_rates"] = components.get("rates")
            row["macro_component_stress"] = components.get("stress")
            row["macro_component_liquidity"] = components.get("liquidity")
            row["candidate_channel_period"] = signal.get("channel_period")
            row["candidate_exit_channel_period"] = signal.get("exit_channel_period")
            spans_weekend = self._trade_spans_weekend(
                datetime.fromisoformat(str(trade["entry_ts"])),
                datetime.fromisoformat(str(trade["exit_ts"])),
            )
            row["spans_weekend"] = spans_weekend
            if spans_weekend:
                weekend_spanning += 1
            session_open = str(trade["session_open"])
            session_counts[session_open] += 1
            session_pnl[session_open] += float(trade["net_pnl"])
            label_counts[str(signal.get("label"))] += 1
            action_counts[str(signal.get("action"))] += 1
            enriched.append(row)

        return (
            enriched,
            {
                "macro_trade_label_counts": dict(label_counts) if use_macro else {},
                "macro_trade_action_counts": dict(action_counts) if use_macro else {},
                "session_counts": dict(session_counts),
                "session_pnl": {key: round(value, 4) for key, value in session_pnl.items()},
                "weekend_spanning_trades": weekend_spanning,
            },
        )

    def _trade_spans_weekend(self, entry_ts: datetime, exit_ts: datetime) -> bool:
        current = entry_ts.date()
        stop = exit_ts.date()
        while current <= stop:
            if current.weekday() >= 5:
                return True
            current += timedelta(days=1)
        return False

    def _trade_key(self, candidate: dict) -> tuple[str, str, str, str, str, str]:
        return (
            str(candidate["ticker"]),
            str(candidate["source"]),
            str(candidate["session_open"]),
            str(candidate["direction"]),
            candidate["entry_ts"].isoformat(),
            candidate["exit_ts"].isoformat(),
        )

    def _write_report_bundle(
        self,
        *,
        variant_dir: Path,
        report: dict,
        summary_extras: dict[str, object],
        macro_config: dict[str, object],
    ) -> None:
        summary_payload = {
            "groups": grouped_channel_summary(),
            "removed": removed_ticker_summary(),
            "summary": report["summary"],
            "macro_sleeve": summary_extras,
            "macro_config": macro_config,
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
            "# Pruned Grouped Macro Score Sleeve Comparison",
            "",
            "| Variant | Final Equity | Return % | CAGR % | Max DD % | PF | Trades | Macro Blocked | Macro Scaled | Weekend Spans | HK Trades | NY Trades |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {variant} | {final_equity} | {total_return_pct} | {cagr_pct} | {max_realized_drawdown_pct} | "
                "{profit_factor} | {executed_trades} | {macro_candidate_blocked} | {macro_candidate_scaled} | "
                "{weekend_spanning_trades} | {hong_kong_trades} | {new_york_trades} |".format(**row)
            )
        path.write_text("\n".join(lines), encoding="utf-8")

    def _json_default(self, value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
