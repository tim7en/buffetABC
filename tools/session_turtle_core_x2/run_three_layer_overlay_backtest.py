"""
Three-layer overlay backtest — asymmetric variants.

Layer design
------------
1. VIXY intraday (5m SMA, NY equity open, equity bucket only)
   low VIXY  = risk_on  → longs 1.0x, suppress shorts
   high VIXY = risk_off → halve longs

2. VIX daily macro (all sessions, all non-crypto buckets)
   VIX <= 15 = risk_on  → longs 1.0x, halve shorts
   VIX >= 25 = risk_off → halve longs, shorts 1.0x

3. Crypto Fear & Greed (all sessions, crypto bucket only)
   F&G >= 60 = greed/risk_on  → longs 1.0x, suppress shorts
   F&G <= 30 = fear/risk_off  → halve longs, shorts 1.0x
"""

from __future__ import annotations

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

OUTPUT_DIR = Path("reports/three_layer_overlay_core_x2")


def _run(*, label: str, overlay_kwargs: dict, candidates: list[dict]) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_BASE_KWARGS,
        precomputed_candidates=candidates,
        **overlay_kwargs,
    )
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["_trades"] = result["trades"]
    return summary


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("Variant",           "variant_label",                   "<46"),
        ("Return %",          "total_return_pct",                ">9"),
        ("CAGR %",            "cagr_pct",                        ">7"),
        ("Max DD %",          "max_realized_drawdown_pct",        ">8"),
        ("PF",                "profit_factor",                   ">5"),
        ("Trades",            "executed_trades",                 ">7"),
        ("VIXY x",            "avg_intraday_volatility_proxy_mult", ">7"),
        ("EH x",              "avg_ext_hours_proxy_mult",         ">6"),
    ]
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(); print(header); print(sep)
    for row in rows:
        parts = []
        for _, key, spec in cols:
            raw = row.get(key, "-")
            text = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
            parts.append(f"{text:{spec}}")
        print("  ".join(parts))
    print()


def _write_csvs(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = [{k: v for k, v in r.items() if k != "_trades"} for r in rows]
    sp = out / "three_layer_comparison_summary.csv"
    with sp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"Summary -> {sp}")
    for row in rows:
        safe = re.sub(r"[^a-z0-9_-]+", "_", row["variant_label"].lower()).strip("_")
        tp = out / f"trades_{safe}.csv"
        trades = row.get("_trades", [])
        if trades:
            with tp.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                w.writeheader(); w.writerows(trades)
        print(f"Trades  -> {tp} ({len(trades)} rows)")


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # ── Layer 1: VIXY intraday (full coverage now) ────────────────────────
    print("\nLoading VIXY bars (VIX_5m, full coverage)...")
    vixy_bars, _, vixy_path = load_local_tiingo_klines(
        ticker="VIXY", interval="5m", lookback_years=5.0, warmup_days=60,
        market_data_symbol="VIX",
    )
    print(f"  {len(vixy_bars)} bars from {vixy_path}")
    vixy_state = build_intraday_volatility_proxy_state(
        proxy_bars=vixy_bars, short_ma_bars=78, long_ma_bars=390, interval_minutes=5,
    )

    # ── Layer 2 + 3: VIX daily macro + Crypto F&G ────────────────────────
    print("Loading VIX daily closes + Crypto Fear & Greed...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg  = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    print(f"  VIX closes: {len(vix_closes)} days")
    print(f"  Crypto F&G: {len(crypto_fg)} days")
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes,
        crypto_fg_scores=crypto_fg,
    )

    # ── Build candidates once ─────────────────────────────────────────────
    print("\nBuilding candidate trades...")
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
    print(f"  {len(candidates)} candidates")

    # ── Shared asymmetric mult settings ──────────────────────────────────
    # VIXY (equity only, NY equity open)
    _vixy = {
        "use_intraday_volatility_proxy": True,
        "intraday_volatility_proxy_state": vixy_state,
        "intraday_volatility_proxy_label": "VIXY",
        "intraday_volatility_proxy_max_age_minutes": 60,
        "intraday_volatility_proxy_lag_bars": 1,
        "intraday_volatility_proxy_short_ma_bars": 78,
        "intraday_volatility_proxy_long_ma_bars": 390,
        "intraday_volatility_proxy_buckets": frozenset({"equity"}),
        # asymmetric: halve longs in risk_off, suppress shorts in risk_on
        "intraday_volatility_long_risk_on_mult":   1.0,
        "intraday_volatility_long_neutral_mult":   1.0,
        "intraday_volatility_long_risk_off_mult":  0.5,
        "intraday_volatility_short_risk_on_mult":  1e-9,
        "intraday_volatility_short_neutral_mult":  1.0,
        "intraday_volatility_short_risk_off_mult": 1.0,
    }

    # VIX daily macro (non-crypto, all sessions) + Crypto F&G (all sessions)
    _macro = {
        "use_extended_hours_proxy": True,
        "extended_hours_proxy_state": macro_state,
        "extended_hours_proxy_lag_days": 1,
        # VIX thresholds for non-crypto
        "extended_hours_vix_risk_on_threshold":  15.0,
        "extended_hours_vix_risk_off_threshold": 25.0,
        # F&G thresholds for crypto
        "extended_hours_fg_greed_threshold": 60.0,
        "extended_hours_fg_fear_threshold":  30.0,
        # asymmetric: halve longs in risk_off, suppress shorts in risk_on
        "extended_hours_long_risk_on_mult":   1.0,
        "extended_hours_long_neutral_mult":   1.0,
        "extended_hours_long_risk_off_mult":  0.5,
        "extended_hours_short_risk_on_mult":  1e-9,
        "extended_hours_short_neutral_mult":  1.0,
        "extended_hours_short_risk_off_mult": 1.0,
    }

    # ── Variants ──────────────────────────────────────────────────────────
    print("\nRunning variants...\n")
    results = [
        _run(label="Baseline", overlay_kwargs={}, candidates=candidates),
        _run(label="VIXY only (equity, NY, asymmetric)",  overlay_kwargs={**_vixy}, candidates=candidates),
        _run(label="VIX macro + F&G only (asymmetric)",  overlay_kwargs={**_macro}, candidates=candidates),
        _run(label="All three layers (asymmetric)",       overlay_kwargs={**_vixy, **_macro}, candidates=candidates),
    ]

    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline:")
    for row in results[1:]:
        d_ret  = float(row["total_return_pct"])          - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"])                  - float(baseline["cagr_pct"])
        d_dd   = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf   = float(row["profit_factor"])             - float(baseline["profit_factor"])
        print(
            f"  {row['variant_label']:<46} "
            f"return {d_ret:+.2f}%  cagr {d_cagr:+.2f}%  "
            f"dd {d_dd:+.2f}%  pf {d_pf:+.3f}"
        )

    print()
    _write_csvs(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
