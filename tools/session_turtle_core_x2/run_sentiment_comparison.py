"""
Sentiment Governor Comparison Runner
===================================

Replays the current best saved production baseline in this repo:
    Session Turtle Trend Core x2 With Asset Class Caps

By default the script compares:
1. Baseline (no sentiment governor)
2. VIX composite
3. VIX + Crypto Fear & Greed composite
4. VIX + Crypto Fear & Greed + CNN Fear & Greed composite

Daily sentiment data is lagged by 1 calendar day by default so intraday
entries do not consume same-day values that may only be known after the close.

Usage
-----
    python tools/session_turtle_core_x2/run_sentiment_comparison.py

    python tools/session_turtle_core_x2/run_sentiment_comparison.py \
        --sources vix crypto_fg cnn_fg aaii \
        --aaii-csv cache/sentiment/aaii_sentiment.csv \
        --combine-mode average \
        --output-dir reports/sentiment_comparison_core_x2
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

from edgar.services.sentiment_data import (
    coverage_report,
    load_aaii_scores,
    load_cnn_fg_scores,
    load_crypto_fg_scores,
    load_vix_scores,
)
from edgar.services.session_turtle_portfolio import (
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)


_BASE_KWARGS = {
    "basket": "core",
    "exposure_mult": 2.0,
    "crypto_cap_mult": 1.0,
    "gold_cap_mult": 0.8,
    "metals_cap_mult": 0.8,
    "base_risk_pct": 0.05,
    "fixed_stop_pct": 0.10,
    "directional_volume_risk_pct": 0.07,
}

_DEFAULT_SOURCES = ["vix", "crypto_fg", "cnn_fg"]
_SOURCE_LABELS = {
    "vix": "VIX",
    "crypto_fg": "Crypto F&G",
    "cnn_fg": "CNN F&G",
    "aaii": "AAII",
}


def _load_source(name: str, *, aaii_csv: str | None) -> dict[str, float]:
    if name == "vix":
        return load_vix_scores()
    if name == "crypto_fg":
        return load_crypto_fg_scores()
    if name == "cnn_fg":
        return load_cnn_fg_scores()
    if name == "aaii":
        return load_aaii_scores(csv_path=aaii_csv)
    raise ValueError(f"Unsupported source: {name}")


def _combine_scores(score_sets: list[dict[str, float]], mode: str) -> dict[str, float]:
    combined: dict[str, float] = {}
    all_dates = sorted({date_key for scores in score_sets for date_key in scores})
    for date_key in all_dates:
        values = [scores[date_key] for scores in score_sets if date_key in scores]
        if not values:
            continue
        if mode == "average":
            value = sum(values) / len(values)
        elif mode == "min":
            value = min(values)
        elif mode == "max":
            value = max(values)
        else:
            raise ValueError(f"Unsupported combine mode: {mode}")
        combined[date_key] = round(value, 2)
    return combined


def _run(
    *,
    label: str,
    source_labels: list[str],
    scores: dict[str, float] | None,
    governor_kwargs: dict,
    precomputed_candidates: list[dict],
) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_BASE_KWARGS,
        precomputed_candidates=precomputed_candidates,
        use_sentiment_governor=scores is not None,
        sentiment_scores=scores,
        **governor_kwargs,
    )
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["sentiment_inputs"] = " + ".join(source_labels) if source_labels else "Baseline"
    summary["sentiment_source_count"] = len(source_labels)
    summary["_trades"] = result["trades"]
    return summary


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("Variant", "variant_label", "<42"),
        ("Inputs", "sentiment_inputs", "<28"),
        ("Return %", "total_return_pct", ">9"),
        ("CAGR %", "cagr_pct", ">8"),
        ("Max DD %", "max_realized_drawdown_pct", ">9"),
        ("PF", "profit_factor", ">6"),
        ("Trades", "executed_trades", ">7"),
    ]
    header = "  ".join(f"{title:{spec}}" for title, _, spec in cols)
    separator = "  ".join("-" * int(spec.strip("<>")) for _, _, spec in cols)
    print()
    print(header)
    print(separator)
    for row in rows:
        rendered: list[str] = []
        for _, key, spec in cols:
            raw = row.get(key, "-")
            text = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
            rendered.append(f"{text:{spec}}")
        print("  ".join(rendered))
    print()


def _write_csvs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sentiment_comparison_summary.csv"
    summary_rows = [{key: value for key, value in row.items() if key != "_trades"} for row in rows]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV -> {summary_path}")

    for row in rows:
        safe_name = re.sub(r"[^a-z0-9_-]+", "_", row["variant_label"].lower()).strip("_")
        trades_path = output_dir / f"trades_{safe_name}.csv"
        trades = row.get("_trades", [])
        if trades:
            with trades_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trades[0].keys()))
                writer.writeheader()
                writer.writerows(trades)
        print(f"Trades CSV  -> {trades_path} ({len(trades)} rows)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["vix", "crypto_fg", "cnn_fg", "aaii"],
        default=_DEFAULT_SOURCES,
        help="Ordered sentiment sources to add cumulatively (default: vix crypto_fg cnn_fg).",
    )
    parser.add_argument(
        "--combine-mode",
        choices=["average", "min", "max"],
        default="average",
        help="How to combine multiple sentiment sources into one composite (default: average).",
    )
    parser.add_argument(
        "--aaii-csv",
        default=None,
        help="Path to manually downloaded AAII CSV (required only when using the aaii source).",
    )
    parser.add_argument("--threshold1", type=float, default=45.0, help="Upper fear threshold (default: 45).")
    parser.add_argument("--threshold2", type=float, default=25.0, help="Extreme fear threshold (default: 25).")
    parser.add_argument("--mult1", type=float, default=1.0, help="Exposure multiplier between thresholds.")
    parser.add_argument("--mult2", type=float, default=0.5, help="Exposure multiplier below threshold2.")
    parser.add_argument("--reversal-window", type=int, default=10, help="Look-back window for reversal detection.")
    parser.add_argument("--reversal-min-rise", type=float, default=10.0, help="Minimum rise from the recent low.")
    parser.add_argument("--reversal-mult", type=float, default=1.0, help="Exposure multiplier on reversal.")
    parser.add_argument(
        "--sentiment-lag-days",
        type=int,
        default=1,
        help="Calendar-day lag applied to sentiment data to avoid same-day look-ahead bias (default: 1).",
    )
    parser.add_argument("--output-dir", default=None, help="If set, write summary and trade CSVs here.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip printing raw source coverage reports.")
    args = parser.parse_args()

    governor_kwargs = {
        "sentiment_lag_days": args.sentiment_lag_days,
        "sentiment_threshold_1": args.threshold1,
        "sentiment_threshold_2": args.threshold2,
        "sentiment_exposure_mult_1": args.mult1,
        "sentiment_exposure_mult_2": args.mult2,
        "sentiment_reversal_window": args.reversal_window,
        "sentiment_reversal_min_rise": args.reversal_min_rise,
        "sentiment_reversal_mult": args.reversal_mult,
    }

    print("\nLoading sentiment data sources...")
    loaded_sources: list[tuple[str, str, dict[str, float]]] = []
    for source_name in args.sources:
        label = _SOURCE_LABELS[source_name]
        scores = _load_source(source_name, aaii_csv=args.aaii_csv)
        loaded_sources.append((source_name, label, scores))
        print(f"  loaded {label}: {len(scores)} dates")

    if not args.skip_coverage:
        for _, label, scores in loaded_sources:
            coverage_report(scores, label)

    print("\nBuilding baseline candidate trade set once...")
    precomputed_candidates = build_session_turtle_shared_account_candidates(
        basket=_BASE_KWARGS["basket"],
        initial_capital=1_000.0,
        lookback_years=4.1,
        channel_period=20,
        base_risk_pct=_BASE_KWARGS["base_risk_pct"],
        fixed_stop_pct=_BASE_KWARGS["fixed_stop_pct"],
        directional_volume_risk_pct=_BASE_KWARGS["directional_volume_risk_pct"],
        trend_fast_period=55,
        trend_slow_period=200,
    )
    print(f"  built {len(precomputed_candidates)} candidate trades")

    print("\nRunning backtests against the current best production baseline...")
    results = [
        _run(
            label="Baseline (no sentiment)",
            source_labels=[],
            scores=None,
            governor_kwargs=governor_kwargs,
            precomputed_candidates=precomputed_candidates,
        )
    ]

    cumulative_labels: list[str] = []
    cumulative_scores: list[dict[str, float]] = []
    total_runs = len(loaded_sources) + 1
    for run_index, (_, label, scores) in enumerate(loaded_sources, start=2):
        cumulative_labels.append(label)
        cumulative_scores.append(scores)
        composite_scores = _combine_scores(cumulative_scores, args.combine_mode)
        variant_label = f"Cumulative {args.combine_mode}: {' + '.join(cumulative_labels)}"
        print(f"  [{run_index}/{total_runs}] {variant_label}")
        results.append(
            _run(
                label=variant_label,
                source_labels=list(cumulative_labels),
                scores=composite_scores,
                governor_kwargs=governor_kwargs,
                precomputed_candidates=precomputed_candidates,
            )
        )

    print("\nSentiment Governor Comparison")
    print(f"Baseline basket: {_BASE_KWARGS['basket']}")
    print(
        "Production caps: "
        f"crypto={_BASE_KWARGS['crypto_cap_mult']}  "
        f"gold={_BASE_KWARGS['gold_cap_mult']}  "
        f"metals={_BASE_KWARGS['metals_cap_mult']}"
    )
    print(
        "Sentiment lag: "
        f"{args.sentiment_lag_days} day(s); thresholds={args.threshold1}/{args.threshold2}; "
        f"combine={args.combine_mode}"
    )
    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline")
    for row in results[1:]:
        d_return = float(row["total_return_pct"]) - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"]) - float(baseline["cagr_pct"])
        d_drawdown = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf = float(row["profit_factor"]) - float(baseline["profit_factor"])
        print(
            f"  {row['variant_label']:<42} "
            f"return {d_return:+.2f}%  "
            f"cagr {d_cagr:+.2f}%  "
            f"dd {d_drawdown:+.2f}%  "
            f"pf {d_pf:+.3f}"
        )
    print()

    if args.output_dir:
        _write_csvs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
