from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from backtest_plotting import plot_asset_pnl, plot_equity_and_drawdown, plot_yearly_pnl
from session_turtle_portfolio import generate_session_turtle_shared_account_report


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "production_baseline"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone Session Turtle Trend Core x2 production backtest."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--basket",
        choices=["core", "index", "expanded"],
        default="core",
        help="Asset basket to replay.",
    )
    parser.add_argument(
        "--exposure-mult",
        type=float,
        default=2.0,
        help="Shared-account gross exposure multiplier.",
    )
    parser.add_argument(
        "--crypto-cap-mult",
        type=float,
        default=1.0,
        help="Crypto asset-class cap multiplier.",
    )
    parser.add_argument(
        "--gold-cap-mult",
        type=float,
        default=0.8,
        help="Gold asset-class cap multiplier.",
    )
    parser.add_argument(
        "--metals-cap-mult",
        type=float,
        default=0.8,
        help="Metals asset-class cap multiplier.",
    )
    parser.add_argument(
        "--equity-cap-mult",
        type=float,
        default=None,
        help="Optional equity asset-class cap multiplier.",
    )
    parser.add_argument(
        "--base-risk-pct",
        type=float,
        default=5.0,
        help="Base per-trade risk percent.",
    )
    parser.add_argument(
        "--fixed-stop-pct",
        type=float,
        default=10.0,
        help="Fixed stop distance percent.",
    )
    parser.add_argument(
        "--directional-volume-risk-pct",
        type=float,
        default=7.0,
        help="Risk percent for directional volume-confirmed entries.",
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
        help="Exposure multiplier after the first drawdown threshold.",
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
        help="Exposure multiplier after the second drawdown threshold.",
    )
    parser.add_argument(
        "--use-performance-leadership-overlay",
        action="store_true",
        help="Scale each asset using the research-only leadership overlay.",
    )
    parser.add_argument(
        "--performance-lookback-trades",
        type=int,
        default=6,
        help="Closed trades per asset to score for the leadership overlay.",
    )
    parser.add_argument(
        "--performance-decay",
        type=float,
        default=0.75,
        help="Exponential decay applied to older asset scores.",
    )
    parser.add_argument(
        "--performance-floor-mult",
        type=float,
        default=0.75,
        help="Minimum multiplier for lagging assets under the leadership overlay.",
    )
    parser.add_argument(
        "--performance-cap-mult",
        type=float,
        default=1.25,
        help="Maximum multiplier for leading assets under the leadership overlay.",
    )
    parser.add_argument(
        "--performance-min-history",
        type=int,
        default=3,
        help="Closed trades required before non-neutral leadership sizing is allowed.",
    )
    parser.add_argument(
        "--use-extended-hours-protective-exits",
        action="store_true",
        help="Allow protective exits outside core hours for New York-session Tiingo assets.",
    )
    parser.add_argument(
        "--extended-hours-core-session-minutes",
        type=int,
        default=390,
        help="Core-session length used before outside-hours protective exits take over.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = generate_session_turtle_shared_account_report(
        basket=args.basket,
        exposure_mult=float(args.exposure_mult),
        use_drawdown_governor=bool(args.use_drawdown_governor),
        drawdown_trigger_1_pct=float(args.drawdown_trigger_1_pct),
        drawdown_exposure_mult_1=float(args.drawdown_exposure_mult_1),
        drawdown_trigger_2_pct=float(args.drawdown_trigger_2_pct),
        drawdown_exposure_mult_2=float(args.drawdown_exposure_mult_2),
        crypto_cap_mult=float(args.crypto_cap_mult) if args.crypto_cap_mult is not None else None,
        gold_cap_mult=float(args.gold_cap_mult) if args.gold_cap_mult is not None else None,
        metals_cap_mult=float(args.metals_cap_mult) if args.metals_cap_mult is not None else None,
        equity_cap_mult=float(args.equity_cap_mult) if args.equity_cap_mult is not None else None,
        base_risk_pct=float(args.base_risk_pct) / 100.0,
        fixed_stop_pct=float(args.fixed_stop_pct) / 100.0,
        directional_volume_risk_pct=float(args.directional_volume_risk_pct) / 100.0,
        use_performance_leadership_overlay=bool(args.use_performance_leadership_overlay),
        performance_lookback_trades=int(args.performance_lookback_trades),
        performance_decay=float(args.performance_decay),
        performance_floor_mult=float(args.performance_floor_mult),
        performance_cap_mult=float(args.performance_cap_mult),
        performance_min_history=int(args.performance_min_history),
        use_extended_hours_protective_exits=bool(args.use_extended_hours_protective_exits),
        extended_hours_core_session_minutes=int(args.extended_hours_core_session_minutes),
    )
    summary = report["summary"]

    _write_csv(output_dir / "shared_account_summary.csv", [summary])
    _write_csv(output_dir / "shared_account_equity_curve.csv", report["equity_curve"])
    _write_csv(output_dir / "shared_account_trades.csv", report["trades"])
    _write_csv(output_dir / "shared_account_yearly_returns.csv", report["yearly_returns"])
    _write_csv(output_dir / "shared_account_asset_summary.csv", report["asset_summary"])

    plot_equity_and_drawdown(
        report["equity_curve"],
        output_dir / "shared_account_equity_drawdown.png",
        title=f"{summary['label']} Equity And Drawdown",
        initial_capital=float(summary["initial_capital"]),
    )
    plot_yearly_pnl(
        report["yearly_returns"],
        output_dir / "shared_account_yearly_pnl.png",
        title=f"{summary['label']} Yearly Realized PnL",
    )
    plot_asset_pnl(
        report["asset_summary"],
        output_dir / "shared_account_asset_pnl.png",
        title=f"{summary['label']} Asset PnL Contribution",
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
