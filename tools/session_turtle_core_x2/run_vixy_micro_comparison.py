"""VIXY micro-regime backtest comparison for the core x2 baseline."""

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


def _run(
    *,
    label: str,
    overlay_kwargs: dict,
    precomputed_candidates: list[dict],
) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_BASE_KWARGS,
        precomputed_candidates=precomputed_candidates,
        **overlay_kwargs,
    )
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["_trades"] = result["trades"]
    return summary


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("Variant", "variant_label", "<36"),
        ("Return %", "total_return_pct", ">9"),
        ("CAGR %", "cagr_pct", ">8"),
        ("Max DD %", "max_realized_drawdown_pct", ">9"),
        ("PF", "profit_factor", ">6"),
        ("Trades", "executed_trades", ">7"),
        ("Proxy x", "avg_intraday_volatility_proxy_mult", ">7"),
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
    summary_path = output_dir / "vixy_micro_comparison_summary.csv"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-symbol", default="VIX", help="Local tiingo cache symbol to load (default: VIX).")
    parser.add_argument("--proxy-label", default="VIXY", help="Human-readable label for the proxy (default: VIXY).")
    parser.add_argument("--short-ma-bars", type=int, default=78, help="Short MA bars (default: 78).")
    parser.add_argument("--long-ma-bars", type=int, default=390, help="Long MA bars (default: 390).")
    parser.add_argument("--max-age-minutes", type=int, default=60, help="Freshness limit in minutes (default: 60).")
    parser.add_argument("--lag-bars", type=int, default=1, help="Completed-bar lag to avoid look-ahead (default: 1).")
    parser.add_argument(
        "--investable-universe-asof",
        default=None,
        help="Optional YYYY-MM-DD date used to freeze the investable universe.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/vixy_micro_backtest_core_x2",
        help="Output directory for CSV artifacts.",
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

    proxy_state = build_intraday_volatility_proxy_state(
        proxy_bars=proxy_bars,
        short_ma_bars=args.short_ma_bars,
        long_ma_bars=args.long_ma_bars,
        interval_minutes=5,
    )

    print("\nBuilding baseline candidate trade set once...")
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

    overlay_common = {
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

    print("\nRunning VIXY micro overlay variants...")
    results = [
        _run(
            label="Baseline",
            overlay_kwargs={"investable_universe_asof": args.investable_universe_asof},
            precomputed_candidates=precomputed_candidates,
        ),
        _run(
            label="VIXY conservative",
            overlay_kwargs={
                **overlay_common,
                "intraday_volatility_long_risk_off_mult": 0.5,
                "intraday_volatility_short_risk_on_mult": 0.5,
            },
            precomputed_candidates=precomputed_candidates,
        ),
        _run(
            label="VIXY asymmetric",
            overlay_kwargs={
                **overlay_common,
                "intraday_volatility_long_risk_off_mult": 0.5,
                "intraday_volatility_short_risk_on_mult": 0.0 + 1e-9,
            },
            precomputed_candidates=precomputed_candidates,
        ),
        _run(
            label="VIXY strict filter",
            overlay_kwargs={
                **overlay_common,
                "intraday_volatility_long_risk_off_mult": 1e-9,
                "intraday_volatility_short_risk_on_mult": 1e-9,
            },
            precomputed_candidates=precomputed_candidates,
        ),
    ]

    print("\nVIXY Micro Comparison")
    print(f"Proxy symbol: {proxy_symbol} ({args.proxy_label})")
    print(f"Freshness: <= {args.max_age_minutes} minutes; lag bars: {args.lag_bars}")
    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline")
    for row in results[1:]:
        d_return = float(row["total_return_pct"]) - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"]) - float(baseline["cagr_pct"])
        d_drawdown = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf = float(row["profit_factor"]) - float(baseline["profit_factor"])
        print(
            f"  {row['variant_label']:<24} "
            f"return {d_return:+.2f}%  "
            f"cagr {d_cagr:+.2f}%  "
            f"dd {d_drawdown:+.2f}%  "
            f"pf {d_pf:+.3f}"
        )

    _write_csvs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
