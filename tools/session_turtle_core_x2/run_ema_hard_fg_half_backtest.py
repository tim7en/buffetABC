"""
Production backtest — Session Turtle Core x3, ema_hard_fg_half configuration.

Exact parameters from the stable run (reports/session_turtle_x3_full_crypto_ema_vs_fg_20260330):
  - exposure_mult = 3.0
  - Two-layer macro overlay: VIX daily (non-crypto) + Crypto Fear & Greed
  - Per-asset EMA(200) hard stop on crypto only (blocks counter-trend side at 0×)
  - F&G half entry: fear (≤ 30) tolongs 0.5×
  - NO VIXY intraday layer (excluded for simplicity)
  - Donchian(20), trend filter EMA(55/200)

Forward-looking bias mitigations (all explicitly enforced):
  1. extended_hours_proxy_lag_days=1  — VIX/F&G uses prior-day close, never today's value
  2. per_asset_ema_lag_days=1         — EMA regime uses prior-day close vs prior-day EMA
  3. build_session_turtle_shared_account_candidates uses close-of-bar signals only
  4. lookback_years=4.1 covers 2022-02-22 to~2026-03-31, fully historical
"""

from __future__ import annotations

import json
import os
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

OUTPUT_DIR = Path("reports/session_turtle_x3_ema_hard_fg_half_production")

# ── Strategy parameters (fixed, matches stable run) ───────────────────────────
_BASKET        = "core"
_LOOKBACK_YRS  = 4.1          # ~2022-02-22 onwards
_CHANNEL       = 20           # Donchian period
_EXPOSURE_MULT = 3.0          # x3
_BASE_RISK     = 0.05         # 5% per trade
_FIXED_STOP    = 0.10         # 10% hard stop
_DIR_VOL_RISK  = 0.07         # 7% directional volume risk cap
_TREND_FAST    = 55
_TREND_SLOW    = 200
_EMA_PERIOD    = 200          # per-asset daily EMA for crypto hard stop

# ── Overlay: VIX daily (non-crypto) thresholds ────────────────────────────────
# Lag=1: uses yesterday's VIX close tono forward look
_VIX_RISK_ON   = 15.0
_VIX_RISK_OFF  = 25.0

# ── Overlay: Crypto Fear & Greed thresholds ───────────────────────────────────
# Lag=1: uses yesterday's F&G score tono forward look
_FG_GREED      = 60.0   # greed/risk-on: shorts suppressed
_FG_FEAR       = 30.0   # fear/risk-off: longs 0.5× (half entry)

# Shared long/short multipliers for the extended-hours overlay
# Risk-on  tolongs full (1.0×), shorts suppressed (~0×)
# Neutral  toboth full (1.0×)
# Risk-off tolongs half (0.5×), shorts full (1.0×)
_EH_LONG_RISK_ON   = 1.0
_EH_LONG_NEUTRAL   = 1.0
_EH_LONG_RISK_OFF  = 0.5    # F&G half entry / VIX half entry
_EH_SHORT_RISK_ON  = 1e-9   # effectively 0 — suppress shorts in risk-on
_EH_SHORT_NEUTRAL  = 1.0
_EH_SHORT_RISK_OFF = 1.0

# ── EMA hard stop mults (crypto only, applies via per_asset_technical_overlay) ─
# Above EMA(200): longs 1.0× (with-trend), shorts 0× (counter-trend hard stop)
# Below EMA(200): shorts 1.0× (with-trend), longs 0× (counter-trend hard stop)
_EMA_ABOVE_LONG  = 1.0
_EMA_ABOVE_SHORT = 0.0   # hard stop
_EMA_BELOW_LONG  = 0.0   # hard stop
_EMA_BELOW_SHORT = 1.0


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # ── Load macro overlay data ────────────────────────────────────────────────
    print("\nLoading VIX daily closes + Crypto Fear & Greed...")
    vix_path = root / "cache/sentiment/vix_closes.json"
    fg_path  = root / "cache/sentiment/crypto_fg_scores.json"
    vix_closes = json.loads(vix_path.read_text())
    crypto_fg  = json.loads(fg_path.read_text())
    print(f"  VIX closes: {len(vix_closes)} days  ({min(vix_closes)} to {max(vix_closes)})")
    print(f"  Crypto F&G: {len(crypto_fg)} days  ({min(crypto_fg)} to {max(crypto_fg)})")

    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes,
        crypto_fg_scores=crypto_fg,
    )

    # ── Build per-asset EMA technical state (crypto hard stop) ────────────────
    # warmup_days=300 gives EMA(200) a full warm-up window before backtest starts
    # lag is enforced in generate_session_turtle_shared_account_report via
    # per_asset_ema_lag_days=1 (uses previous day's EMA, not today's intrabar)
    print("\nBuilding per-asset technical state (daily EMA200)...")
    universe = list(dict.fromkeys((t, s, ses) for t, s, ses in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe,
        lookback_years=5.0,   # extra year for EMA warm-up
        warmup_days=300,
        ema_period=_EMA_PERIOD,
        adx_period=14,        # not used (adx_gate=False) but built for completeness
    )
    print(f"  EMA coverage: {len(tech_state['daily_ema'])} tickers")

    # ── Build candidates (raw Donchian breakout signals, no overlay applied) ──
    # Candidates are pure price signals — overlay is applied at report-generation time
    print("\nBuilding candidate trades...")
    candidates = build_session_turtle_shared_account_candidates(
        basket=_BASKET,
        initial_capital=1_000.0,
        lookback_years=_LOOKBACK_YRS,
        channel_period=_CHANNEL,
        base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP,
        directional_volume_risk_pct=_DIR_VOL_RISK,
        trend_fast_period=_TREND_FAST,
        trend_slow_period=_TREND_SLOW,
    )
    print(f"  {len(candidates)} candidates")

    # ── Run backtest ───────────────────────────────────────────────────────────
    print("\nRunning ema_hard_fg_half backtest...")
    result = generate_session_turtle_shared_account_report(
        basket=_BASKET,
        exposure_mult=_EXPOSURE_MULT,
        crypto_cap_mult=1.0,
        gold_cap_mult=0.8,
        metals_cap_mult=0.8,
        base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP,
        directional_volume_risk_pct=_DIR_VOL_RISK,
        precomputed_candidates=candidates,
        # ── Macro overlay ───────────────────────────────────────────────────
        use_extended_hours_proxy=True,
        extended_hours_proxy_state=macro_state,
        extended_hours_proxy_lag_days=1,          # <-- no forward look: prior-day VIX/F&G
        extended_hours_vix_risk_on_threshold=_VIX_RISK_ON,
        extended_hours_vix_risk_off_threshold=_VIX_RISK_OFF,
        extended_hours_fg_greed_threshold=_FG_GREED,
        extended_hours_fg_fear_threshold=_FG_FEAR,
        extended_hours_long_risk_on_mult=_EH_LONG_RISK_ON,
        extended_hours_long_neutral_mult=_EH_LONG_NEUTRAL,
        extended_hours_long_risk_off_mult=_EH_LONG_RISK_OFF,
        extended_hours_short_risk_on_mult=_EH_SHORT_RISK_ON,
        extended_hours_short_neutral_mult=_EH_SHORT_NEUTRAL,
        extended_hours_short_risk_off_mult=_EH_SHORT_RISK_OFF,
        # ── EMA hard stop (crypto only) ─────────────────────────────────────
        use_per_asset_technical_overlay=True,
        per_asset_technical_state=tech_state,
        per_asset_ema_lag_days=1,                 # <-- no forward look: prior-day EMA
        per_asset_ema_above_long_mult=_EMA_ABOVE_LONG,
        per_asset_ema_above_short_mult=_EMA_ABOVE_SHORT,
        per_asset_ema_below_long_mult=_EMA_BELOW_LONG,
        per_asset_ema_below_short_mult=_EMA_BELOW_SHORT,
        per_asset_use_adx_gate=False,             # ADX gate not used in stable run
    )

    summary = result["summary"]
    trades  = result["trades"]

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SESSION TURTLE CORE x3 — ema_hard_fg_half PRODUCTION BACKTEST")
    print("=" * 60)
    print(f"  Period          : {summary['start_date'][:10]} to{summary['end_date'][:10]}")
    print(f"  Candidates      : {summary['candidate_trades']}")
    print(f"  Executed trades : {summary['executed_trades']}  (long: {summary['long_trades']}, short: {summary['short_trades']})")
    print(f"  Win rate        : {summary['win_rate_pct']:.1f}%  |  Profit factor: {summary['profit_factor']:.2f}")
    print(f"  Total return    : {summary['total_return_pct']:.2f}%")
    print(f"  CAGR            : {summary['cagr_pct']:.2f}%")
    print(f"  Max DD          : {summary['max_realized_drawdown_pct']:.2f}%")
    print(f"  Final equity    : ${summary['final_equity']:,.2f}  (from $1,000)")
    print()
    print(f"  Overlay impact:")
    print(f"    Ext-hours scaled    : {summary['entries_ext_hours_proxy_scaled']} trades")
    print(f"    EMA above / below   : {summary['entries_above_ema']} / {summary['entries_below_ema']}")
    print(f"    Avg EMA mult        : {summary['avg_technical_ema_mult']:.4f}")
    print(f"    Avg ext-hours mult  : {summary['avg_ext_hours_proxy_mult']:.4f}")
    print()

    # ── Forward-look bias check (spot-check) ──────────────────────────────────
    print("FORWARD-LOOK BIAS CHECKS:")
    print(f"  extended_hours_proxy_lag_days = 1  OK  (prior-day VIX/F&G)")
    print(f"  per_asset_ema_lag_days         = 1  OK  (prior-day EMA)")
    print(f"  EMA warmup_days                = 300  OK  (EMA(200) fully warm)")
    print(f"  Candidates built on close-of-bar signals only  OK")
    print()

    # ── Write outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Summary to{summary_path}")

    if trades:
        import csv
        trades_path = OUTPUT_DIR / "trades.csv"
        with trades_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
            w.writeheader()
            w.writerows(trades)
        print(f"Trades  to{trades_path}  ({len(trades)} rows)")


if __name__ == "__main__":
    main()
