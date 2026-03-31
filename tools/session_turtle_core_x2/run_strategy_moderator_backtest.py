"""
Strategy Moderator Backtest.

Simulates launching the strategy moderator after the first 3 months of live running.

Flow
----
Phase 1  (months 0-3)  Warm-up: run with ALL candidates, no filtering.
                        Collect executed trades per asset.

Scoring  (at month 3)  Score each asset using only the warm-up OOS results.
                        Apply the strategy moderator's tier thresholds.

Phase 2  (months 3-end) Moderated run: full candidate list is filtered —
                         Tier A/B kept at full weight
                         Tier C  kept but capped at 50% notional (every 2nd candidate)
                         Tier D  kept but capped at 25% notional (every 4th candidate)
                         Tier F  removed entirely
                         Assets with < MIN_TRADES_TO_SCORE get the benefit of the doubt (keep)

Two variants are compared:
  baseline   — no moderation, full 4-year run
  moderated  — moderation kicks in after 3-month warm-up

Both use identical overlay config (ema_hard_fg_half).

Forward-looking bias: scoring uses ONLY warm-up trades (entry_ts < cutoff).
                      Candidates after the cutoff are never seen during scoring.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
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

# ── Config ─────────────────────────────────────────────────────────────────────
WARM_UP_MONTHS   = 3        # observation-only period before moderation begins
MIN_TRADES_TO_SCORE = 3     # assets with fewer warm-up trades get benefit of doubt

# Tier score thresholds (matches strategy-moderator.jsx)
TIER_A = 75
TIER_B = 55
TIER_C = 40
TIER_D = 25
# below TIER_D = Tier F (remove)

OUTPUT_DIR = Path("reports/strategy_moderator_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy params (identical to ema_hard_fg_half production run) ─────────────
_BASKET        = "core"
_LOOKBACK_YRS  = 4.1
_CHANNEL       = 20
_EXPOSURE_MULT = 3.0
_BASE_RISK     = 0.05
_FIXED_STOP    = 0.10
_DIR_VOL_RISK  = 0.07
_TREND_FAST    = 55
_TREND_SLOW    = 200
_EMA_PERIOD    = 200

_OVERLAY_KWARGS = dict(
    use_extended_hours_proxy=True,
    extended_hours_proxy_lag_days=1,
    extended_hours_vix_risk_on_threshold=15.0,
    extended_hours_vix_risk_off_threshold=25.0,
    extended_hours_fg_greed_threshold=60.0,
    extended_hours_fg_fear_threshold=30.0,
    extended_hours_long_risk_on_mult=1.0,
    extended_hours_long_neutral_mult=1.0,
    extended_hours_long_risk_off_mult=0.5,
    extended_hours_short_risk_on_mult=1e-9,
    extended_hours_short_neutral_mult=1.0,
    extended_hours_short_risk_off_mult=1.0,
    use_per_asset_technical_overlay=True,
    per_asset_ema_lag_days=1,
    per_asset_ema_above_long_mult=1.0,
    per_asset_ema_above_short_mult=0.0,
    per_asset_ema_below_long_mult=0.0,
    per_asset_ema_below_short_mult=1.0,
    per_asset_use_adx_gate=False,
)

_BASE_KWARGS = dict(
    basket=_BASKET,
    exposure_mult=_EXPOSURE_MULT,
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    base_risk_pct=_BASE_RISK,
    fixed_stop_pct=_FIXED_STOP,
    directional_volume_risk_pct=_DIR_VOL_RISK,
)


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_asset(trades: list[dict]) -> tuple[int, str]:
    """
    Compute a 0-100 composite score for one asset using warm-up OOS trades.
    Returns (score, tier_label).

    Pillars (each 0-25):
      net_return    — total P&L sign and magnitude relative to max loss
      win_rate      — % winning trades
      profit_factor — gross wins / gross losses
      expectancy    — mean P&L per trade
    """
    if len(trades) < MIN_TRADES_TO_SCORE:
        return -1, "INSUFFICIENT"   # benefit of doubt

    pnls   = [t["net_pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr     = len(wins) / len(pnls)
    total  = sum(pnls)
    avg    = total / len(pnls)
    gross_win  = sum(wins)   if wins   else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf     = gross_win / gross_loss

    # Pillar 1: net return (25 pts)
    if total > 0:
        ret_score = 25
    elif total > -50:
        ret_score = 15
    elif total > -150:
        ret_score = 5
    else:
        ret_score = 0

    # Pillar 2: win rate (25 pts)
    if wr >= 0.45:
        wr_score = 25
    elif wr >= 0.35:
        wr_score = 15
    elif wr >= 0.25:
        wr_score = 8
    else:
        wr_score = 0

    # Pillar 3: profit factor (25 pts)
    if pf >= 1.5:
        pf_score = 25
    elif pf >= 1.0:
        pf_score = 15
    elif pf >= 0.5:
        pf_score = 5
    else:
        pf_score = 0

    # Pillar 4: expectancy per trade (25 pts)
    if avg > 20:
        exp_score = 25
    elif avg > 0:
        exp_score = 15
    elif avg > -20:
        exp_score = 5
    else:
        exp_score = 0

    score = ret_score + wr_score + pf_score + exp_score

    if score >= TIER_A:
        tier = "A"
    elif score >= TIER_B:
        tier = "B"
    elif score >= TIER_C:
        tier = "C"
    elif score >= TIER_D:
        tier = "D"
    else:
        tier = "F"

    return score, tier


def _tier_to_keep_ratio(tier: str) -> float:
    """What fraction of an asset's post-cutoff candidates to keep."""
    return {"A": 1.0, "B": 1.0, "C": 0.5, "D": 0.25, "F": 0.0,
            "INSUFFICIENT": 1.0}[tier]


# ── Run one variant ────────────────────────────────────────────────────────────

def _run(*, label: str, candidates: list[dict], macro_state: dict,
         tech_state: dict) -> dict:
    result = generate_session_turtle_shared_account_report(
        **_BASE_KWARGS,
        **_OVERLAY_KWARGS,
        extended_hours_proxy_state=macro_state,
        per_asset_technical_state=tech_state,
        precomputed_candidates=candidates,
    )
    s = dict(result["summary"])
    s["variant_label"] = label
    s["_trades"] = result["trades"]
    return s


# ── Filter candidates based on tier ───────────────────────────────────────────

def _apply_moderation(
    all_candidates: list[dict],
    warm_trades: list[dict],
    cutoff: datetime,
) -> tuple[list[dict], dict[str, tuple[int, str]]]:
    """
    Returns (filtered_candidates, scores_by_ticker).

    Before cutoff: all candidates kept (observation phase).
    After cutoff:  candidates filtered by tier ratio.
    """
    # Group warm-up trades by ticker
    warm_by_ticker: dict[str, list] = defaultdict(list)
    for t in warm_trades:
        warm_by_ticker[t["ticker"]].append(t)

    # Score every ticker that appears anywhere in the candidate pool
    all_tickers = set(c["ticker"] for c in all_candidates)
    scores: dict[str, tuple[int, str]] = {}
    for ticker in all_tickers:
        score, tier = _score_asset(warm_by_ticker.get(ticker, []))
        scores[ticker] = (score, tier)

    # Filter: candidates before cutoff → all pass; after cutoff → tier ratio
    filtered: list[dict] = []
    # Track per-ticker counter for ratio-based thinning
    ticker_post_counter: dict[str, int] = defaultdict(int)

    for c in all_candidates:
        if c["entry_ts"] < cutoff:
            filtered.append(c)
            continue
        _, tier = scores[c["ticker"]]
        ratio = _tier_to_keep_ratio(tier)
        if ratio == 0.0:
            continue
        ticker_post_counter[c["ticker"]] += 1
        # keep every N-th candidate to approximate the weight reduction
        # counter starts at 1; % n == 0 keeps: 1st (n=1), every 2nd (n=2), every 4th (n=4)
        n = round(1.0 / ratio)
        if ticker_post_counter[c["ticker"]] % n == 0:
            filtered.append(c)

    return filtered, scores


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # ── Overlay state ────────────────────────────────────────────────────────
    print("\nLoading VIX daily closes + Crypto Fear & Greed...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg  = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg,
    )

    print("Building per-asset EMA200 technical state...")
    universe   = list(dict.fromkeys((t, s, ses) for t, s, ses in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe, lookback_years=5.0, warmup_days=300,
        ema_period=_EMA_PERIOD, adx_period=14,
    )

    # ── Build candidates ──────────────────────────────────────────────────────
    print("\nBuilding candidate trades...")
    all_candidates = build_session_turtle_shared_account_candidates(
        basket=_BASKET, initial_capital=1_000.0, lookback_years=_LOOKBACK_YRS,
        channel_period=_CHANNEL, base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP, directional_volume_risk_pct=_DIR_VOL_RISK,
        trend_fast_period=_TREND_FAST, trend_slow_period=_TREND_SLOW,
    )
    print(f"  {len(all_candidates)} candidates")

    start_ts = min(c["entry_ts"] for c in all_candidates)
    cutoff   = datetime(start_ts.year + (start_ts.month + WARM_UP_MONTHS - 1) // 12,
                        (start_ts.month + WARM_UP_MONTHS - 1) % 12 + 1,
                        start_ts.day)
    print(f"  Backtest start : {start_ts.date()}")
    print(f"  Warm-up cutoff : {cutoff.date()}  ({WARM_UP_MONTHS} months)")

    # ── Baseline: run unchanged ───────────────────────────────────────────────
    print("\nRunning BASELINE (no moderation)...")
    baseline = _run(
        label="Baseline (no moderation)",
        candidates=all_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
    )

    # ── Warm-up pass: execute only warm-up candidates to get scoring data ─────
    print(f"\nRunning WARM-UP phase (first {WARM_UP_MONTHS} months)...")
    warmup_candidates = [c for c in all_candidates if c["entry_ts"] < cutoff]
    print(f"  {len(warmup_candidates)} warm-up candidates")
    warmup_result = _run(
        label="Warm-up phase",
        candidates=warmup_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
    )
    warmup_trades = warmup_result["_trades"]
    print(f"  {len(warmup_trades)} trades executed in warm-up")

    # ── Score each asset ──────────────────────────────────────────────────────
    print("\n  ASSET SCORES (based on warm-up OOS trades):")
    print(f"  {'Ticker':<14} {'Trades':>6} {'WR%':>5} {'PF':>5} {'P&L':>9} {'Score':>6} {'Tier'}")
    print("  " + "-" * 60)

    # compute scores
    warm_by_ticker: dict[str, list] = defaultdict(list)
    for t in warmup_trades:
        warm_by_ticker[t["ticker"]].append({"net_pnl": float(t["net_pnl"])})

    all_tickers = sorted(set(c["ticker"] for c in all_candidates))
    scores_display: dict[str, tuple[int, str]] = {}
    for ticker in all_tickers:
        wt = warm_by_ticker.get(ticker, [])
        score, tier = _score_asset(wt)
        scores_display[ticker] = (score, tier)
        n     = len(wt)
        pnls  = [t["net_pnl"] for t in wt]
        wr    = (sum(1 for p in pnls if p > 0) / n * 100) if n > 0 else 0.0
        gross_win  = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 1e-9
        pf    = gross_win / gross_loss if n > 0 else 0.0
        total = sum(pnls) if pnls else 0.0
        score_str = f"{score}" if score >= 0 else "n/a"
        print(f"  {ticker:<14} {n:>6} {wr:>5.0f}% {pf:>5.2f} {total:>9.0f} {score_str:>6} {tier}")

    # ── Apply moderation ──────────────────────────────────────────────────────
    moderated_candidates, _ = _apply_moderation(all_candidates, warmup_trades, cutoff)
    removed_tickers  = [tk for tk, (sc, ti) in scores_display.items() if ti == "F"]
    reduced_tickers  = [tk for tk, (sc, ti) in scores_display.items() if ti in ("C", "D")]
    kept_tickers     = [tk for tk, (sc, ti) in scores_display.items() if ti in ("A", "B", "INSUFFICIENT")]
    print(f"\n  Tier F (removed after cutoff): {removed_tickers or 'none'}")
    print(f"  Tier C/D (reduced after cutoff): {reduced_tickers or 'none'}")
    print(f"  Benefit of doubt (kept, <{MIN_TRADES_TO_SCORE} trades): {[tk for tk, (_, ti) in scores_display.items() if ti=='INSUFFICIENT']}")
    print(f"\n  Moderated candidates: {len(moderated_candidates)}  (was {len(all_candidates)})")

    # ── Moderated run ─────────────────────────────────────────────────────────
    print("\nRunning MODERATED backtest...")
    moderated = _run(
        label=f"Moderated (SM active after {WARM_UP_MONTHS}mo)",
        candidates=moderated_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
    )

    # ── Results comparison ────────────────────────────────────────────────────
    rows = [baseline, moderated]
    cols = [
        ("Variant",     "variant_label",                "<46"),
        ("Return %",    "total_return_pct",             ">9"),
        ("CAGR %",      "cagr_pct",                     ">7"),
        ("Max DD %",    "max_realized_drawdown_pct",    ">8"),
        ("PF",          "profit_factor",                ">5"),
        ("WR %",        "win_rate_pct",                 ">6"),
        ("Trades",      "executed_trades",              ">7"),
    ]
    print("\n" + "=" * 100)
    print("  STRATEGY MODERATOR BACKTEST — RESULTS")
    print("=" * 100)
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(header); print(sep)
    for row in rows:
        parts = []
        for _, key, spec in cols:
            raw  = row.get(key, "-")
            text = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
            parts.append(f"{text:{spec}}")
        print("  ".join(parts))

    b, m = baseline, moderated
    print("\n  Delta (moderated vs baseline):")
    print(f"    Return   : {float(m['total_return_pct']) - float(b['total_return_pct']):+.2f}%")
    print(f"    CAGR     : {float(m['cagr_pct']) - float(b['cagr_pct']):+.2f}%")
    print(f"    Max DD   : {float(m['max_realized_drawdown_pct']) - float(b['max_realized_drawdown_pct']):+.2f}%")
    print(f"    PF       : {float(m['profit_factor']) - float(b['profit_factor']):+.3f}")
    print(f"    Trades   : {int(m['executed_trades']) - int(b['executed_trades']):+d}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    for row in rows:
        safe = row["variant_label"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in safe)[:50]
        sub = OUTPUT_DIR / safe
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps({k: v for k, v in row.items() if k != "_trades"}, indent=2, default=str),
            encoding="utf-8",
        )
        trades = row.get("_trades", [])
        if trades:
            with (sub / "trades.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                w.writeheader(); w.writerows(trades)
        print(f"  {row['variant_label']} -> {sub}/")

    # Save scores
    scores_out = {
        "cutoff_date": str(cutoff.date()),
        "warm_up_months": WARM_UP_MONTHS,
        "min_trades_to_score": MIN_TRADES_TO_SCORE,
        "asset_scores": {tk: {"score": sc, "tier": ti} for tk, (sc, ti) in scores_display.items()},
    }
    (OUTPUT_DIR / "asset_scores.json").write_text(
        json.dumps(scores_out, indent=2), encoding="utf-8"
    )
    print(f"  Asset scores -> {OUTPUT_DIR}/asset_scores.json")


if __name__ == "__main__":
    main()
