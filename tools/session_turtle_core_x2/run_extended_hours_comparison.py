"""
Extended-hours proxy comparison for the core x2 baseline.

Outside US equity hours:
  - non-crypto asset buckets (gold, equity, metals)  -> daily VIX close regime
  - crypto asset bucket                              -> Crypto Fear & Greed regime

US equity-open trades continue to use the existing intraday VIXY SMA overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    build_extended_hours_proxy_state,
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

# VIXY intraday overlay (NY equity open) — same params as the existing baseline
_VIXY_KWARGS = {
    "use_intraday_volatility_proxy": True,
    "intraday_volatility_proxy_label": "VIXY",
    "intraday_volatility_proxy_max_age_minutes": 60,
    "intraday_volatility_proxy_lag_bars": 1,
    "intraday_volatility_proxy_short_ma_bars": 78,
    "intraday_volatility_proxy_long_ma_bars": 390,
    "intraday_volatility_long_risk_on_mult": 1.0,
    "intraday_volatility_long_neutral_mult": 1.0,
    "intraday_volatility_long_risk_off_mult": 0.5,
    "intraday_volatility_short_risk_on_mult": 1e-9,   # asymmetric: suppress shorts in risk-on
    "intraday_volatility_short_neutral_mult": 1.0,
    "intraday_volatility_short_risk_off_mult": 1.0,
}


def _run(*, label: str, overlay_kwargs: dict, precomputed_candidates: list[dict]) -> dict:
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
        ("Variant",           "variant_label",                    "<44"),
        ("Return %",          "total_return_pct",                 ">9"),
        ("CAGR %",            "cagr_pct",                         ">8"),
        ("Max DD %",          "max_realized_drawdown_pct",         ">9"),
        ("PF",                "profit_factor",                    ">6"),
        ("Trades",            "executed_trades",                  ">7"),
        ("EH scaled",         "entries_ext_hours_proxy_scaled",   ">9"),
        ("EH avg x",          "avg_ext_hours_proxy_mult",         ">8"),
    ]
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(); print(header); print(sep)
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
    summary_rows = [{k: v for k, v in row.items() if k != "_trades"} for row in rows]
    summary_path = output_dir / "extended_hours_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV  -> {summary_path}")

    for row in rows:
        safe = re.sub(r"[^a-z0-9_-]+", "_", row["variant_label"].lower()).strip("_")
        trades_path = output_dir / f"trades_{safe}.csv"
        trades = row.get("_trades", [])
        if trades:
            with trades_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                writer.writeheader()
                writer.writerows(trades)
        print(f"Trades CSV   -> {trades_path} ({len(trades)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/extended_hours_proxy_core_x2")
    parser.add_argument("--vix-risk-on",  type=float, default=15.0, help="VIX <= X → risk_on")
    parser.add_argument("--vix-risk-off", type=float, default=25.0, help="VIX >= X → risk_off")
    parser.add_argument("--fg-greed",     type=float, default=60.0, help="F&G >= X → risk_on (greed)")
    parser.add_argument("--fg-fear",      type=float, default=30.0, help="F&G <= X → risk_off (fear)")
    args = parser.parse_args()

    # ── load data ────────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parents[2]

    print("\nLoading intraday VIXY proxy bars (for NY equity open)...")
    proxy_bars, proxy_symbol, proxy_path = load_local_tiingo_klines(
        ticker="VIXY", interval="5m", lookback_years=10.0, warmup_days=30,
        market_data_symbol="VIX",
    )
    print(f"  {len(proxy_bars)} bars from {proxy_path}")
    vixy_state = build_intraday_volatility_proxy_state(
        proxy_bars=proxy_bars, short_ma_bars=78, long_ma_bars=390, interval_minutes=5,
    )

    print("\nLoading extended-hours proxy data (VIX daily + Crypto F&G)...")
    vix_closes_path = root / "cache" / "sentiment" / "vix_closes.json"
    cfg_path        = root / "cache" / "sentiment" / "crypto_fg_scores.json"
    vix_closes  = json.loads(vix_closes_path.read_text())
    crypto_fg   = json.loads(cfg_path.read_text())
    print(f"  VIX closes : {len(vix_closes)} days")
    print(f"  Crypto F&G : {len(crypto_fg)} days")
    ext_hours_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes,
        crypto_fg_scores=crypto_fg,
    )

    # ── build candidates once ────────────────────────────────────────────────
    print("\nBuilding candidate trade set...")
    candidates = build_session_turtle_shared_account_candidates(
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
    print(f"  {len(candidates)} candidate trades")

    _ext_common = {
        "use_extended_hours_proxy": True,
        "extended_hours_proxy_state": ext_hours_state,
        "extended_hours_proxy_lag_days": 1,
        "extended_hours_vix_risk_on_threshold":  args.vix_risk_on,
        "extended_hours_vix_risk_off_threshold": args.vix_risk_off,
        "extended_hours_fg_greed_threshold": args.fg_greed,
        "extended_hours_fg_fear_threshold":  args.fg_fear,
        "extended_hours_long_risk_on_mult":   1.0,
        "extended_hours_long_neutral_mult":   1.0,
        "extended_hours_short_neutral_mult":  1.0,
        "extended_hours_short_risk_off_mult": 1.0,
    }

    print("\nRunning variants...")
    results = [
        # 1. Baseline – no overlays
        _run(
            label="Baseline",
            overlay_kwargs={},
            precomputed_candidates=candidates,
        ),
        # 2. VIXY asymmetric only (NY equity open)
        _run(
            label="VIXY asymmetric (NY only)",
            overlay_kwargs={
                **_VIXY_KWARGS,
                "intraday_volatility_proxy_state": vixy_state,
            },
            precomputed_candidates=candidates,
        ),
        # 3. Extended hours only (HK open: VIX + F&G)
        _run(
            label="Ext-hours conservative",
            overlay_kwargs={
                **_ext_common,
                "extended_hours_long_risk_off_mult":  0.5,
                "extended_hours_short_risk_on_mult":  0.5,
            },
            precomputed_candidates=candidates,
        ),
        # 4. Extended hours asymmetric (mirror VIXY asymmetric logic)
        _run(
            label="Ext-hours asymmetric",
            overlay_kwargs={
                **_ext_common,
                "extended_hours_long_risk_off_mult":  0.5,
                "extended_hours_short_risk_on_mult":  1e-9,
            },
            precomputed_candidates=candidates,
        ),
        # 5. VIXY asymmetric + extended hours asymmetric (full coverage)
        _run(
            label="VIXY + Ext-hours (full coverage)",
            overlay_kwargs={
                **_VIXY_KWARGS,
                "intraday_volatility_proxy_state": vixy_state,
                **_ext_common,
                "extended_hours_long_risk_off_mult":  0.5,
                "extended_hours_short_risk_on_mult":  1e-9,
            },
            precomputed_candidates=candidates,
        ),
        # 6. Strict: suppress all misaligned trades in both sessions
        _run(
            label="Full coverage strict",
            overlay_kwargs={
                **_VIXY_KWARGS,
                "intraday_volatility_proxy_state": vixy_state,
                "intraday_volatility_long_risk_off_mult":  1e-9,
                "intraday_volatility_short_risk_on_mult":  1e-9,
                **_ext_common,
                "extended_hours_long_risk_off_mult":  1e-9,
                "extended_hours_short_risk_on_mult":  1e-9,
            },
            precomputed_candidates=candidates,
        ),
    ]

    print("\nExtended-hours Proxy Comparison")
    print(f"VIX thresholds: risk_on <= {args.vix_risk_on:.0f}, risk_off >= {args.vix_risk_off:.0f}")
    print(f"F&G thresholds: greed  >= {args.fg_greed:.0f}, fear   <= {args.fg_fear:.0f}")
    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline:")
    for row in results[1:]:
        d_ret = float(row["total_return_pct"])  - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"])         - float(baseline["cagr_pct"])
        d_dd   = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf   = float(row["profit_factor"])    - float(baseline["profit_factor"])
        print(
            f"  {row['variant_label']:<44} "
            f"return {d_ret:+.2f}%  cagr {d_cagr:+.2f}%  "
            f"dd {d_dd:+.2f}%  pf {d_pf:+.3f}"
        )

    _write_csvs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
