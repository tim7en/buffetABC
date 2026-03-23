"""Walk-forward VIXY micro-regime validation with a frozen investable universe."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

from edgar.services.local_tiingo_data import load_local_tiingo_klines
from edgar.services.session_turtle_portfolio import (
    build_intraday_volatility_proxy_state,
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


def _score(summary: dict) -> tuple[float, float, float]:
    return (
        float(summary.get("profit_factor", 0.0) or 0.0),
        float(summary.get("total_return_pct", 0.0) or 0.0),
        -float(summary.get("max_realized_drawdown_pct", 0.0) or 0.0),
    )


def _filter_candidates(candidates: list[dict], windows: list[tuple[date, date]]) -> list[dict]:
    if not windows:
        return []
    selected: list[dict] = []
    for candidate in candidates:
        candidate_day = candidate["entry_ts"].date()
        if any(start_day <= candidate_day <= end_day for start_day, end_day in windows):
            selected.append(candidate)
    return selected


def _coverage_blocks(proxy_bars: list[dict], max_gap_days: int = 7) -> list[tuple[date, date]]:
    unique_days = sorted({bar["timestamp"].date() for bar in proxy_bars})
    if not unique_days:
        return []

    blocks: list[tuple[date, date]] = []
    start_day = unique_days[0]
    prev_day = unique_days[0]
    for current_day in unique_days[1:]:
        if (current_day - prev_day).days > max_gap_days:
            blocks.append((start_day, prev_day))
            start_day = current_day
        prev_day = current_day
    blocks.append((start_day, prev_day))
    return blocks


def _variant_definitions(common_kwargs: dict) -> list[tuple[str, dict]]:
    return [
        ("Baseline", {"investable_universe_asof": common_kwargs["investable_universe_asof"]}),
        (
            "VIXY conservative",
            {
                **common_kwargs,
                "intraday_volatility_long_risk_off_mult": 0.5,
                "intraday_volatility_short_risk_on_mult": 0.5,
            },
        ),
        (
            "VIXY asymmetric",
            {
                **common_kwargs,
                "intraday_volatility_long_risk_off_mult": 0.5,
                "intraday_volatility_short_risk_on_mult": 1e-9,
            },
        ),
        (
            "VIXY strict filter",
            {
                **common_kwargs,
                "intraday_volatility_long_risk_off_mult": 1e-9,
                "intraday_volatility_short_risk_on_mult": 1e-9,
            },
        ),
    ]


def _run_variant(
    *,
    label: str,
    variant_kwargs: dict,
    candidates: list[dict],
) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_BASE_KWARGS,
        precomputed_candidates=candidates,
        **variant_kwargs,
    )
    summary = dict(result["summary"])
    summary["variant_label"] = label
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    *,
    output_dir: Path,
    investable_universe_asof: str | None,
    blocks: list[tuple[date, date]],
    selected_rows: list[dict],
) -> None:
    lines = [
        "# VIXY Micro Walk-Forward",
        "",
        "## Setup",
        "",
        f"- investable universe as of: `{investable_universe_asof or 'none'}`",
        f"- folds: `{len(selected_rows)}`",
        "- selection metric: profit factor, then total return, then lower max drawdown",
        "",
        "## Coverage Blocks",
        "",
    ]
    for idx, (start_day, end_day) in enumerate(blocks, start=1):
        lines.append(f"- block {idx}: `{start_day.isoformat()}` -> `{end_day.isoformat()}`")
    lines.extend(["", "## Out-Of-Sample Results", ""])
    for row in selected_rows:
        lines.append(
            f"- fold {row['fold']}: selected `{row['selected_variant']}` on train, "
            f"test return `{row['selected_test_return_pct']}%` vs baseline `{row['baseline_test_return_pct']}%`, "
            f"delta `{row['test_return_delta_pct']:+.2f}%`, "
            f"test PF `{row['selected_test_profit_factor']}` vs baseline `{row['baseline_test_profit_factor']}`"
        )
    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "This report reduces tuning bias by choosing the VIXY variant only on prior coverage blocks, then judging it on the next unseen block.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-symbol", default="VIX", help="Local Tiingo cache symbol for VIXY proxy bars.")
    parser.add_argument("--proxy-label", default="VIXY", help="Human-readable proxy label.")
    parser.add_argument("--short-ma-bars", type=int, default=78, help="Short MA bars for regime state.")
    parser.add_argument("--long-ma-bars", type=int, default=390, help="Long MA bars for regime state.")
    parser.add_argument("--max-age-minutes", type=int, default=60, help="Freshness limit in minutes.")
    parser.add_argument("--lag-bars", type=int, default=1, help="Completed-bar lag to avoid look-ahead.")
    parser.add_argument(
        "--investable-universe-asof",
        default="2023-01-03",
        help="Optional YYYY-MM-DD date used to freeze the investable universe.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/vixy_micro_walkforward_core_x2",
        help="Output directory for walk-forward artifacts.",
    )
    args = parser.parse_args()

    print("\nLoading local VIXY proxy bars...")
    proxy_bars, proxy_symbol, proxy_path = load_local_tiingo_klines(
        ticker=args.proxy_label,
        interval="5m",
        lookback_years=10.0,
        warmup_days=30,
        market_data_symbol=args.proxy_symbol,
    )
    print(f"  loaded {len(proxy_bars)} bars from {proxy_path}")

    blocks = _coverage_blocks(proxy_bars)
    if len(blocks) < 2:
        raise ValueError("Need at least two VIXY coverage blocks for walk-forward validation")

    proxy_state = build_intraday_volatility_proxy_state(
        proxy_bars=proxy_bars,
        short_ma_bars=args.short_ma_bars,
        long_ma_bars=args.long_ma_bars,
        interval_minutes=5,
    )

    print("\nBuilding frozen-universe candidate trade set once...")
    precomputed_candidates = build_session_turtle_shared_account_candidates(
        basket=_BASE_KWARGS["basket"],
        investable_universe_asof=args.investable_universe_asof,
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

    common_variant_kwargs = {
        "investable_universe_asof": args.investable_universe_asof,
        "use_intraday_volatility_proxy": True,
        "intraday_volatility_proxy_state": proxy_state,
        "intraday_volatility_proxy_label": args.proxy_label,
        "intraday_volatility_proxy_max_age_minutes": args.max_age_minutes,
        "intraday_volatility_proxy_lag_bars": args.lag_bars,
        "intraday_volatility_proxy_short_ma_bars": args.short_ma_bars,
        "intraday_volatility_proxy_long_ma_bars": args.long_ma_bars,
        "intraday_volatility_long_risk_on_mult": 1.0,
        "intraday_volatility_long_neutral_mult": 1.0,
        "intraday_volatility_short_neutral_mult": 1.0,
        "intraday_volatility_short_risk_off_mult": 1.0,
    }
    variants = _variant_definitions(common_variant_kwargs)

    all_rows: list[dict] = []
    selected_rows: list[dict] = []
    print("\nRunning anchored walk-forward folds...")
    for fold_idx in range(1, len(blocks)):
        train_windows = blocks[:fold_idx]
        test_window = blocks[fold_idx]
        train_candidates = _filter_candidates(precomputed_candidates, train_windows)
        test_candidates = _filter_candidates(precomputed_candidates, [test_window])
        if not train_candidates or not test_candidates:
            continue

        train_results = []
        for label, variant_kwargs in variants:
            summary = _run_variant(label=label, variant_kwargs=variant_kwargs, candidates=train_candidates)
            summary.update(
                {
                    "fold": fold_idx,
                    "split": "train",
                    "window_start": train_windows[0][0].isoformat(),
                    "window_end": train_windows[-1][1].isoformat(),
                }
            )
            train_results.append(summary)
            all_rows.append(summary)

        best_train = max(train_results, key=_score)

        test_results = {}
        for label, variant_kwargs in variants:
            summary = _run_variant(label=label, variant_kwargs=variant_kwargs, candidates=test_candidates)
            summary.update(
                {
                    "fold": fold_idx,
                    "split": "test",
                    "window_start": test_window[0].isoformat(),
                    "window_end": test_window[1].isoformat(),
                }
            )
            test_results[label] = summary
            all_rows.append(summary)

        baseline_test = test_results["Baseline"]
        selected_test = test_results[str(best_train["variant_label"])]
        selected_rows.append(
            {
                "fold": fold_idx,
                "train_start": train_windows[0][0].isoformat(),
                "train_end": train_windows[-1][1].isoformat(),
                "test_start": test_window[0].isoformat(),
                "test_end": test_window[1].isoformat(),
                "selected_variant": best_train["variant_label"],
                "selected_train_return_pct": best_train["total_return_pct"],
                "selected_train_profit_factor": best_train["profit_factor"],
                "selected_test_return_pct": selected_test["total_return_pct"],
                "selected_test_profit_factor": selected_test["profit_factor"],
                "baseline_test_return_pct": baseline_test["total_return_pct"],
                "baseline_test_profit_factor": baseline_test["profit_factor"],
                "test_return_delta_pct": round(
                    float(selected_test["total_return_pct"]) - float(baseline_test["total_return_pct"]),
                    2,
                ),
                "test_pf_delta": round(
                    float(selected_test["profit_factor"]) - float(baseline_test["profit_factor"]),
                    4,
                ),
            }
        )
        print(
            f"  fold {fold_idx}: train winner={best_train['variant_label']}, "
            f"test return {selected_test['total_return_pct']}% vs baseline {baseline_test['total_return_pct']}%"
        )

    if not all_rows or not selected_rows:
        raise ValueError("No walk-forward folds produced results")

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "walkforward_variant_results.csv", all_rows)
    _write_csv(output_dir / "walkforward_selected_results.csv", selected_rows)
    _write_readme(
        output_dir=output_dir,
        investable_universe_asof=args.investable_universe_asof,
        blocks=blocks,
        selected_rows=selected_rows,
    )

    print(f"\nVariant results -> {output_dir / 'walkforward_variant_results.csv'}")
    print(f"Selected folds  -> {output_dir / 'walkforward_selected_results.csv'}")
    print(f"README          -> {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
