"""
Per-asset technical overlay comparison.

Tests four variants on top of the existing VIX/F&G two-layer overlay (x3):
  Baseline       — no technical overlay
  Daily EMA(200) — scale longs/shorts based on price vs daily EMA
  4H ADX(14)     — scale down in ranging markets (ADX < 20)
  EMA + ADX      — both combined

EMA scaling rule (standard, not data-mined):
  price > EMA(200): long 1.0x, short 0.5x   (with-trend full, counter-trend cautious)
  price < EMA(200): long 0.5x, short 1.0x

ADX gate (Wilder original period = 14):
  ADX < 20: scale everything to 0.5x (ranging, no conviction)
  ADX >= 20: no change
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

from edgar.services.session_turtle_portfolio import (
    CORE_SESSION_TURTLE_UNIVERSE,
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)

OUTPUT_DIR = Path("reports/technical_overlay_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_BASE_KWARGS = dict(
    basket="core",
    exposure_mult=3.0,
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    base_risk_pct=0.05,
    fixed_stop_pct=0.10,
    directional_volume_risk_pct=0.07,
)

_MACRO_OVERLAY = dict(
    use_extended_hours_proxy=True,
    extended_hours_proxy_lag_days=1,
    extended_hours_vix_risk_on_threshold=15.0,
    extended_hours_vix_risk_off_threshold=25.0,
    extended_hours_fg_greed_threshold=60.0,
    extended_hours_fg_fear_threshold=30.0,
    extended_hours_long_risk_on_mult=1.0,
    extended_hours_long_neutral_mult=1.0,
    extended_hours_long_risk_off_mult=0.5,
    extended_hours_short_risk_on_mult=1e-9,   # reverted — suppressed in risk-on
    extended_hours_short_neutral_mult=1.0,
    extended_hours_short_risk_off_mult=1.0,
)


def _run(label, extra_kwargs, candidates, macro_state, tech_state=None):
    kw = dict(**_BASE_KWARGS, **_MACRO_OVERLAY)
    kw["extended_hours_proxy_state"] = macro_state
    kw.update(extra_kwargs)
    if tech_state is not None:
        kw["per_asset_technical_state"] = tech_state
    result = generate_session_turtle_shared_account_report(
        **kw, precomputed_candidates=candidates,
    )
    s = dict(result["summary"])
    s["variant_label"] = label
    s["_trades"] = result["trades"]
    return s


def _print_table(rows):
    cols = [
        ("Variant",          "variant_label",                "<46"),
        ("Return %",         "total_return_pct",             ">9"),
        ("CAGR %",           "cagr_pct",                     ">7"),
        ("Max DD %",         "max_realized_drawdown_pct",    ">8"),
        ("PF",               "profit_factor",                ">5"),
        ("Trades",           "executed_trades",              ">7"),
        ("Above EMA",        "entries_above_ema",            ">10"),
        ("Below EMA",        "entries_below_ema",            ">10"),
        ("ADX scaled",       "entries_adx_scaled_down",      ">10"),
        ("Avg EMA x",        "avg_technical_ema_mult",       ">9"),
    ]
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(); print(header); print(sep)
    for row in rows:
        parts = []
        for _, key, spec in cols:
            raw  = row.get(key, "-")
            text = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
            parts.append(f"{text:{spec}}")
        print("  ".join(parts))
    print()


def _write_csvs(rows, out):
    summary_rows = [{k: v for k, v in r.items() if k != "_trades"} for r in rows]
    sp = out / "technical_overlay_summary.csv"
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


def main():
    root = Path(__file__).resolve().parents[2]

    print("\nLoading VIX daily + Crypto F&G...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg  = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg,
    )

    print("Building per-asset technical state (daily EMA200 + 4H ADX14)...")
    universe = list(dict.fromkeys((t, s, ses) for t, s, ses in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe,
        lookback_years=5.0,
        warmup_days=300,
        ema_period=200,
        adx_period=14,
    )
    print(f"  EMA coverage: {len(tech_state['daily_ema'])} tickers")
    print(f"  ADX coverage: {len(tech_state['4h_adx'])} tickers")

    print("\nBuilding candidates...")
    candidates = build_session_turtle_shared_account_candidates(
        basket="core",
        initial_capital=1_000.0,
        lookback_years=4.1,
        channel_period=20,
        base_risk_pct=0.05,
        fixed_stop_pct=0.10,
        directional_volume_risk_pct=0.07,
        trend_fast_period=55,
        trend_slow_period=200,
    )
    print(f"  {len(candidates)} candidates")

    print("\nRunning variants...")
    results = [
        _run(
            "Baseline (VIX+FG only, no tech)",
            {}, candidates, macro_state,
        ),
        _run(
            "Daily EMA(200) scaling",
            dict(
                use_per_asset_technical_overlay=True,
                per_asset_ema_lag_days=1,
                per_asset_ema_above_long_mult=1.0,
                per_asset_ema_above_short_mult=0.5,
                per_asset_ema_below_long_mult=0.5,
                per_asset_ema_below_short_mult=1.0,
                per_asset_use_adx_gate=False,
            ),
            candidates, macro_state, tech_state,
        ),
        _run(
            "4H ADX(14) gate only",
            dict(
                use_per_asset_technical_overlay=True,
                per_asset_ema_above_long_mult=1.0,
                per_asset_ema_above_short_mult=1.0,
                per_asset_ema_below_long_mult=1.0,
                per_asset_ema_below_short_mult=1.0,
                per_asset_use_adx_gate=True,
                per_asset_adx_strong_threshold=25.0,
                per_asset_adx_weak_threshold=20.0,
                per_asset_adx_weak_mult=0.5,
            ),
            candidates, macro_state, tech_state,
        ),
        _run(
            "EMA(200) scaling + 4H ADX(14) gate",
            dict(
                use_per_asset_technical_overlay=True,
                per_asset_ema_lag_days=1,
                per_asset_ema_above_long_mult=1.0,
                per_asset_ema_above_short_mult=0.5,
                per_asset_ema_below_long_mult=0.5,
                per_asset_ema_below_short_mult=1.0,
                per_asset_use_adx_gate=True,
                per_asset_adx_strong_threshold=25.0,
                per_asset_adx_weak_threshold=20.0,
                per_asset_adx_weak_mult=0.5,
            ),
            candidates, macro_state, tech_state,
        ),
    ]

    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline:")
    for row in results[1:]:
        d_ret  = float(row["total_return_pct"])          - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"])                  - float(baseline["cagr_pct"])
        d_dd   = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf   = float(row["profit_factor"])             - float(baseline["profit_factor"])
        d_tr   = int(row["executed_trades"])             - int(baseline["executed_trades"])
        print(
            f"  {row['variant_label']:<46} "
            f"return {d_ret:+.2f}%  cagr {d_cagr:+.2f}%  "
            f"dd {d_dd:+.2f}%  pf {d_pf:+.3f}  trades {d_tr:+d}"
        )

    print()
    _write_csvs(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
