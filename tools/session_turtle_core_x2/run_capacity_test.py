"""
Capacity test: high base_risk_pct vs high exposure_mult.

Hypothesis:
  - base_risk_pct=0.20  → each trade 4× bigger → FEWER entries (more capacity skips)
  - exposure_mult=3.0   → total capacity 1.5× larger → MORE entries
  - exposure_mult=4.0   → total capacity 2× larger → even more entries

All variants use the full three-layer overlay (VIXY + VIX macro + F&G).
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

OUTPUT_DIR = Path("reports/capacity_test_core_x2")

# ── shared candidate params (same as three-layer baseline) ─────────────────
_CANDIDATE_PARAMS = dict(
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

# ── shared fixed kwargs (no exposure/risk variant) ─────────────────────────
_FIXED_KWARGS = dict(
    basket="core",
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    fixed_stop_pct=0.10,
)


def _run(*, label: str, base_risk_pct: float, exposure_mult: float,
         overlay_kwargs: dict, candidates: list[dict]) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_FIXED_KWARGS,
        base_risk_pct=base_risk_pct,
        directional_volume_risk_pct=base_risk_pct * 1.4,  # keep same ratio as baseline
        exposure_mult=exposure_mult,
        precomputed_candidates=candidates,
        **overlay_kwargs,
    )
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["_trades"] = result["trades"]
    return summary


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("Variant",          "variant_label",                   "<44"),
        ("Exp x",            "exposure_mult_used",              ">6"),
        ("Risk %",           "base_risk_pct_used",              ">7"),
        ("Return %",         "total_return_pct",                ">9"),
        ("CAGR %",           "cagr_pct",                        ">7"),
        ("Max DD %",         "max_realized_drawdown_pct",       ">8"),
        ("PF",               "profit_factor",                   ">5"),
        ("Trades",           "executed_trades",                 ">7"),
        ("Skipped cap",      "skipped_no_capacity",             ">11"),
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
    sp = out / "capacity_test_summary.csv"
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

    # ── Load overlays ─────────────────────────────────────────────────────
    print("\nLoading VIXY bars (VIX_5m)...")
    vixy_bars, _, vixy_path = load_local_tiingo_klines(
        ticker="VIXY", interval="5m", lookback_years=5.0, warmup_days=60,
        market_data_symbol="VIX",
    )
    print(f"  {len(vixy_bars)} bars from {vixy_path}")
    vixy_state = build_intraday_volatility_proxy_state(
        proxy_bars=vixy_bars, short_ma_bars=78, long_ma_bars=390, interval_minutes=5,
    )

    print("Loading VIX daily closes + Crypto F&G...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg  = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    print(f"  VIX closes: {len(vix_closes)} days,  Crypto F&G: {len(crypto_fg)} days")
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes,
        crypto_fg_scores=crypto_fg,
    )

    # ── Build candidates once ─────────────────────────────────────────────
    print("\nBuilding candidate trades...")
    candidates = build_session_turtle_shared_account_candidates(**_CANDIDATE_PARAMS)
    print(f"  {len(candidates)} candidates")

    # ── Full three-layer overlay kwargs ───────────────────────────────────
    _overlay = {
        # VIXY — equity only, NY session
        "use_intraday_volatility_proxy": True,
        "intraday_volatility_proxy_state": vixy_state,
        "intraday_volatility_proxy_label": "VIXY",
        "intraday_volatility_proxy_max_age_minutes": 60,
        "intraday_volatility_proxy_lag_bars": 1,
        "intraday_volatility_proxy_short_ma_bars": 78,
        "intraday_volatility_proxy_long_ma_bars": 390,
        "intraday_volatility_proxy_buckets": frozenset({"equity"}),
        "intraday_volatility_long_risk_on_mult":   1.0,
        "intraday_volatility_long_neutral_mult":   1.0,
        "intraday_volatility_long_risk_off_mult":  0.5,
        "intraday_volatility_short_risk_on_mult":  1e-9,
        "intraday_volatility_short_neutral_mult":  1.0,
        "intraday_volatility_short_risk_off_mult": 1.0,
        # VIX macro + F&G — all sessions
        "use_extended_hours_proxy": True,
        "extended_hours_proxy_state": macro_state,
        "extended_hours_proxy_lag_days": 1,
        "extended_hours_vix_risk_on_threshold":  15.0,
        "extended_hours_vix_risk_off_threshold": 25.0,
        "extended_hours_fg_greed_threshold": 60.0,
        "extended_hours_fg_fear_threshold":  30.0,
        "extended_hours_long_risk_on_mult":   1.0,
        "extended_hours_long_neutral_mult":   1.0,
        "extended_hours_long_risk_off_mult":  0.5,
        "extended_hours_short_risk_on_mult":  1e-9,
        "extended_hours_short_neutral_mult":  1.0,
        "extended_hours_short_risk_off_mult": 1.0,
    }

    # ── Variants ──────────────────────────────────────────────────────────
    variants = [
        # label,                          base_risk_pct, exposure_mult
        ("Baseline  (risk=5%,  exp=2x)",  0.05,          2.0),
        ("High risk (risk=20%, exp=2x)",  0.20,          2.0),
        ("High exp  (risk=5%,  exp=3x)",  0.05,          3.0),
        ("High exp  (risk=5%,  exp=4x)",  0.05,          4.0),
        ("Both high (risk=20%, exp=4x)",  0.20,          4.0),
    ]

    print("\nRunning variants...\n")
    results = []
    for label, risk, exp in variants:
        print(f"  {label} ...")
        row = _run(
            label=label,
            base_risk_pct=risk,
            exposure_mult=exp,
            overlay_kwargs=_overlay,
            candidates=candidates,
        )
        # stash params for table display
        row["exposure_mult_used"] = exp
        row["base_risk_pct_used"] = risk * 100  # show as %
        results.append(row)

    _print_table(results)

    baseline = results[0]
    print("Delta vs baseline:")
    for row in results[1:]:
        d_ret  = float(row["total_return_pct"])          - float(baseline["total_return_pct"])
        d_cagr = float(row["cagr_pct"])                  - float(baseline["cagr_pct"])
        d_dd   = float(row["max_realized_drawdown_pct"]) - float(baseline["max_realized_drawdown_pct"])
        d_pf   = float(row["profit_factor"])             - float(baseline["profit_factor"])
        d_tr   = int(row["executed_trades"])             - int(baseline["executed_trades"])
        d_sk   = int(row.get("skipped_no_capacity", 0)) - int(baseline.get("skipped_no_capacity", 0))
        print(
            f"  {row['variant_label']:<44} "
            f"return {d_ret:+.2f}%  cagr {d_cagr:+.2f}%  "
            f"dd {d_dd:+.2f}%  pf {d_pf:+.3f}  "
            f"trades {d_tr:+d}  skipped {d_sk:+d}"
        )

    print()
    _write_csvs(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
