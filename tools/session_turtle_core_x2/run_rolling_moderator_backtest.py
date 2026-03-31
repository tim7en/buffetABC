"""
Rolling Strategy Moderator Backtest.

Architecture
------------
The core engine runs in a single pass over precomputed_candidates.  To simulate
a rolling moderator that re-scores assets every month we pre-filter the candidate
list BEFORE handing it to the engine:

  1. Run the BASELINE (no moderation) to collect the full trade history.
  2. For each monthly review checkpoint (starting at warm-up cutoff, every month):
       - Score each asset on the trailing SCORE_LOOKBACK_MONTHS of baseline trades.
       - Assign tier (A/B/C/D/F) → derive keep-ratio (1.0 / 1.0 / 0.5 / 0.25 / 0.0).
       - Assets with < MIN_TRADES_TO_SCORE in the window → INSUFFICIENT → keep 1.0.
  3. For each candidate in the full pool:
       - Before warm-up cutoff                     → always keep (observation phase).
       - At or after cutoff                         → look up the most recent review
                                                       decision for that asset, apply
                                                       keep-ratio thinning.
  4. Run the MODERATED backtest with the filtered candidate list.
  5. DD circuit breaker: use_drawdown_governor active, halves exposure at 15% DD,
     near-halts at 25% DD.
  6. Print a decision timeline and comparison table.

Scoring proxy note
------------------
Scoring uses BASELINE trades as the signal the moderator would have observed.
Using the moderated trades would create a circular dependency (can't know moderated
execution before the moderation decisions are made).  This is a standard
walk-forward approximation.  The capital curve diverges between baseline and
moderated, but the TIER SIGNALS are clean and lag-safe.

Forward-look bias guards
------------------------
  * Scoring window uses only trades with exit_ts < review_date.
  * Review decisions apply to candidates with entry_ts in [review, next_review).
  * Candidates before warm-up cutoff are never scored (observation phase).
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
WARM_UP_MONTHS        = 3     # observation-only window before first review
REVIEW_FREQ_MONTHS    = 1     # how often moderator re-scores (monthly)
SCORE_LOOKBACK_MONTHS = 3     # trailing window of trades used to score each asset
MIN_TRADES_TO_SCORE   = 3     # fewer → INSUFFICIENT → benefit of the doubt (keep)

# Tier score thresholds (matches strategy-moderator.jsx)
TIER_A, TIER_B, TIER_C, TIER_D = 75, 55, 40, 25
# below TIER_D → Tier F (remove)

# DD circuit breaker thresholds (as absolute % drawdown)
# 50% of backtest max DD ≈ 15%;  80% ≈ 25%  (max DD in baseline ≈ 30%)
DD_TRIGGER_1   = 15.0   # halve exposure
DD_TRIGGER_2   = 25.0   # near-halt (0.5x exposure_mult)
DD_MULT_1      = 1.5    # exposure at trigger 1  (3.0 → 1.5)
DD_MULT_2      = 0.5    # exposure at trigger 2  (3.0 → 0.5)

OUTPUT_DIR = Path("reports/rolling_moderator_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy params ────────────────────────────────────────────────────────────
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


# ── Date helpers ───────────────────────────────────────────────────────────────

def _add_months(dt: datetime, months: int) -> datetime:
    """Add N months to a datetime, clamping to month-end where needed."""
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    import calendar
    day = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=day)


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_asset(trades: list[dict]) -> tuple[int, str]:
    """0-100 composite score.  Returns (score, tier).  -1/'INSUFFICIENT' if sparse."""
    if len(trades) < MIN_TRADES_TO_SCORE:
        return -1, "INSUFFICIENT"

    pnls       = [float(t["net_pnl"]) for t in trades]
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p <= 0]
    wr         = len(wins) / len(pnls)
    total      = sum(pnls)
    avg        = total / len(pnls)
    gross_win  = sum(wins)           if wins   else 0.0
    gross_loss = abs(sum(losses))    if losses else 1e-9
    pf         = gross_win / gross_loss

    ret_score = 25 if total > 0 else 15 if total > -50 else 5 if total > -150 else 0
    wr_score  = 25 if wr >= 0.45 else 15 if wr >= 0.35 else 8 if wr >= 0.25 else 0
    pf_score  = 25 if pf >= 1.5  else 15 if pf >= 1.0  else 5 if pf >= 0.5  else 0
    exp_score = 25 if avg > 20   else 15 if avg > 0     else 5 if avg > -20  else 0

    score = ret_score + wr_score + pf_score + exp_score
    tier  = ("A" if score >= TIER_A else "B" if score >= TIER_B else
             "C" if score >= TIER_C else "D" if score >= TIER_D else "F")
    return score, tier


def _tier_to_keep_ratio(tier: str) -> float:
    return {"A": 1.0, "B": 1.0, "C": 0.5, "D": 0.25, "F": 0.0,
            "INSUFFICIENT": 1.0}[tier]


# ── Rolling decision map ───────────────────────────────────────────────────────

def _build_rolling_decisions(
    baseline_trades: list[dict],
    all_tickers: set[str],
    first_review: datetime,
    last_entry: datetime,
) -> dict[datetime, dict[str, tuple[int, str]]]:
    """
    For each monthly review date, compute tier for every ticker using the
    trailing SCORE_LOOKBACK_MONTHS of baseline trades (exits before review date).

    Returns: {review_dt: {ticker: (score, tier)}}
    """
    # Index baseline trades by ticker, storing (exit_ts_dt, trade_dict)
    by_ticker: dict[str, list] = defaultdict(list)
    for t in baseline_trades:
        by_ticker[t["ticker"]].append(
            (datetime.fromisoformat(t["exit_ts"]), t)
        )

    decisions: dict[datetime, dict[str, tuple[int, str]]] = {}
    review_dt = first_review
    while review_dt <= last_entry + timedelta(days=32):
        window_start = _add_months(review_dt, -SCORE_LOOKBACK_MONTHS)
        tick_decisions: dict[str, tuple[int, str]] = {}
        for ticker in all_tickers:
            window_trades = [
                td for (exit_dt, td) in by_ticker.get(ticker, [])
                if window_start <= exit_dt < review_dt   # strict: no look-ahead
            ]
            tick_decisions[ticker] = _score_asset(window_trades)
        decisions[review_dt] = tick_decisions
        review_dt = _add_months(review_dt, REVIEW_FREQ_MONTHS)

    return decisions


# ── Candidate filter ───────────────────────────────────────────────────────────

def _apply_rolling_moderation(
    all_candidates: list[dict],
    decisions: dict[datetime, dict[str, tuple[int, str]]],
    cutoff: datetime,
) -> tuple[list[dict], list[datetime]]:
    """
    Filter the candidate list.
    Before cutoff              → all pass (observation phase).
    After cutoff               → apply the most recent review decision.

    Returns (filtered_candidates, sorted_review_dates).
    """
    sorted_reviews = sorted(decisions.keys())
    ticker_post_counter: dict[tuple[str, datetime], int] = defaultdict(int)

    filtered: list[dict] = []
    for c in all_candidates:
        if c["entry_ts"] < cutoff:
            filtered.append(c)
            continue

        # Most recent review date on or before this candidate's entry
        applicable = None
        for rd in sorted_reviews:
            if rd <= c["entry_ts"]:
                applicable = rd
            else:
                break
        if applicable is None:
            filtered.append(c)
            continue

        ticker = c["ticker"]
        _, tier = decisions[applicable][ticker]
        ratio   = _tier_to_keep_ratio(tier)
        if ratio == 0.0:
            continue

        key = (ticker, applicable)
        ticker_post_counter[key] += 1
        n = round(1.0 / ratio)
        if ticker_post_counter[key] % n == 0:
            filtered.append(c)

    return filtered, sorted_reviews


# ── Run engine ─────────────────────────────────────────────────────────────────

def _run(*, label: str, candidates: list[dict], macro_state: dict,
         tech_state: dict, use_dd_governor: bool = False) -> dict:
    kw = dict(**_BASE_KWARGS, **_OVERLAY_KWARGS)
    kw["extended_hours_proxy_state"] = macro_state
    kw["per_asset_technical_state"]  = tech_state
    kw["precomputed_candidates"]     = candidates
    if use_dd_governor:
        kw["use_drawdown_governor"]        = True
        kw["drawdown_trigger_1_pct"]       = DD_TRIGGER_1
        kw["drawdown_exposure_mult_1"]     = DD_MULT_1
        kw["drawdown_trigger_2_pct"]       = DD_TRIGGER_2
        kw["drawdown_exposure_mult_2"]     = DD_MULT_2
    result = generate_session_turtle_shared_account_report(**kw)
    s = dict(result["summary"])
    s["variant_label"] = label
    s["_trades"]       = result["trades"]
    return s


# ── Decision timeline printer ──────────────────────────────────────────────────

def _print_timeline(
    decisions: dict[datetime, dict[str, tuple[int, str]]],
    sorted_reviews: list[datetime],
    all_tickers: list[str],
) -> None:
    """Print a compact monthly decision table."""
    TIER_ICON = {"A": "A", "B": "B", "C": "C", "D": "D", "F": "X",
                 "INSUFFICIENT": "."}
    print("\n  ROLLING MODERATOR DECISIONS  (. = insufficient data, X = removed)")
    header = f"  {'Review':<12}" + "".join(f"{tk[:6]:>7}" for tk in all_tickers)
    print(header)
    print("  " + "-" * (12 + 7 * len(all_tickers)))
    for rd in sorted_reviews:
        row = f"  {str(rd.date()):<12}"
        for tk in all_tickers:
            _, tier = decisions[rd][tk]
            row += f"{TIER_ICON[tier]:>7}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # ── Overlay state ────────────────────────────────────────────────────────
    print("\nLoading VIX daily closes + Crypto Fear & Greed...")
    vix_closes  = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg   = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg,
    )

    print("Building per-asset EMA200 technical state...")
    universe   = list(dict.fromkeys((t, s, ses) for t, s, ses in CORE_SESSION_TURTLE_UNIVERSE))
    tech_state = build_per_asset_technical_state(
        universe=universe, lookback_years=5.0, warmup_days=300,
        ema_period=_EMA_PERIOD, adx_period=14,
    )

    # ── Candidates ───────────────────────────────────────────────────────────
    print("\nBuilding candidate trades...")
    all_candidates = build_session_turtle_shared_account_candidates(
        basket=_BASKET, initial_capital=1_000.0, lookback_years=_LOOKBACK_YRS,
        channel_period=_CHANNEL, base_risk_pct=_BASE_RISK,
        fixed_stop_pct=_FIXED_STOP, directional_volume_risk_pct=_DIR_VOL_RISK,
        trend_fast_period=_TREND_FAST, trend_slow_period=_TREND_SLOW,
    )
    print(f"  {len(all_candidates)} candidates")

    start_ts     = min(c["entry_ts"] for c in all_candidates)
    last_entry   = max(c["entry_ts"] for c in all_candidates)
    cutoff       = _add_months(start_ts, WARM_UP_MONTHS)
    first_review = cutoff
    all_tickers  = sorted(set(c["ticker"] for c in all_candidates))

    print(f"  Backtest start   : {start_ts.date()}")
    print(f"  Warm-up cutoff   : {cutoff.date()}  ({WARM_UP_MONTHS} months)")
    print(f"  Review frequency : monthly")
    print(f"  Score lookback   : {SCORE_LOOKBACK_MONTHS} months trailing")
    print(f"  Assets           : {len(all_tickers)}")

    # ── Step 1: Baseline ─────────────────────────────────────────────────────
    print("\nStep 1 — Running BASELINE (no moderation, no DD governor)...")
    baseline = _run(
        label="Baseline (no moderation)",
        candidates=all_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        use_dd_governor=False,
    )
    print(f"  Return {baseline['total_return_pct']:.1f}%  CAGR {baseline['cagr_pct']:.1f}%  "
          f"MaxDD {baseline['max_realized_drawdown_pct']:.1f}%  "
          f"PF {baseline['profit_factor']:.2f}  Trades {baseline['executed_trades']}")

    # ── Step 2: Build rolling decision map from baseline trade history ────────
    print("\nStep 2 — Building rolling monthly decision map...")
    decisions = _build_rolling_decisions(
        baseline_trades=baseline["_trades"],
        all_tickers=set(all_tickers),
        first_review=first_review,
        last_entry=last_entry,
    )
    print(f"  {len(decisions)} monthly review checkpoints")

    # ── Step 3: Filter candidates ─────────────────────────────────────────────
    print("\nStep 3 — Filtering candidates via rolling decisions...")
    moderated_candidates, sorted_reviews = _apply_rolling_moderation(
        all_candidates=all_candidates,
        decisions=decisions,
        cutoff=cutoff,
    )
    removed = len(all_candidates) - len(moderated_candidates)
    print(f"  {len(moderated_candidates)} candidates kept  ({removed} filtered out  "
          f"{removed/len(all_candidates)*100:.1f}%)")

    # ── Step 4: Moderated run (rolling filter + DD governor) ──────────────────
    print("\nStep 4 — Running MODERATED backtest (rolling filter + DD governor)...")
    moderated = _run(
        label=f"Rolling moderated ({WARM_UP_MONTHS}mo warmup, monthly review, DD governor)",
        candidates=moderated_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        use_dd_governor=True,
    )
    print(f"  Return {moderated['total_return_pct']:.1f}%  CAGR {moderated['cagr_pct']:.1f}%  "
          f"MaxDD {moderated['max_realized_drawdown_pct']:.1f}%  "
          f"PF {moderated['profit_factor']:.2f}  Trades {moderated['executed_trades']}")

    # Also run moderated WITHOUT DD governor to isolate effects
    print("\nStep 4b — Running MODERATED (rolling filter only, no DD governor)...")
    moderated_no_dd = _run(
        label="Rolling moderated (filter only, no DD governor)",
        candidates=moderated_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        use_dd_governor=False,
    )

    # ── Print timeline ────────────────────────────────────────────────────────
    _print_timeline(decisions, sorted_reviews, all_tickers)

    # Per-asset summary across all reviews
    print(f"\n  ASSET TIER SUMMARY  (across {len(sorted_reviews)} monthly reviews)")
    print(f"  {'Ticker':<14} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'F':>4} {'Insuf':>6}  Final tier")
    print("  " + "-" * 55)
    for ticker in all_tickers:
        counts = defaultdict(int)
        for rd in sorted_reviews:
            _, tier = decisions[rd][ticker]
            counts[tier] += 1
        _, final_tier = decisions[sorted_reviews[-1]][ticker]
        print(f"  {ticker:<14} {counts['A']:>4} {counts['B']:>4} {counts['C']:>4} "
              f"{counts['D']:>4} {counts['F']:>4} {counts['INSUFFICIENT']:>6}   {final_tier}")

    # ── Comparison table ──────────────────────────────────────────────────────
    rows = [baseline, moderated_no_dd, moderated]
    cols = [
        ("Variant",     "variant_label",                "<52"),
        ("Return %",    "total_return_pct",             ">9"),
        ("CAGR %",      "cagr_pct",                     ">7"),
        ("Max DD %",    "max_realized_drawdown_pct",    ">8"),
        ("PF",          "profit_factor",                ">5"),
        ("WR %",        "win_rate_pct",                 ">6"),
        ("Trades",      "executed_trades",              ">7"),
    ]
    print("\n" + "=" * 110)
    print("  ROLLING MODERATOR BACKTEST — RESULTS COMPARISON")
    print("=" * 110)
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print(header); print(sep)
    for row in rows:
        parts = [f"{(f'{v:.2f}' if isinstance(v := row.get(k, '-'), float) else str(v)):{s}}"
                 for _, k, s in cols]
        print("  ".join(parts))

    b = baseline
    print("\n  Delta vs baseline:")
    for row in [moderated_no_dd, moderated]:
        print(f"  {row['variant_label']}")
        print(f"    Return {float(row['total_return_pct'])-float(b['total_return_pct']):+.1f}%  "
              f"CAGR {float(row['cagr_pct'])-float(b['cagr_pct']):+.1f}%  "
              f"MaxDD {float(row['max_realized_drawdown_pct'])-float(b['max_realized_drawdown_pct']):+.1f}%  "
              f"PF {float(row['profit_factor'])-float(b['profit_factor']):+.3f}  "
              f"Trades {int(row['executed_trades'])-int(b['executed_trades']):+d}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    for row in rows:
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in
                       row["variant_label"].lower().replace(" ", "_"))[:50]
        sub = OUTPUT_DIR / safe
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps({k: v for k, v in row.items() if k != "_trades"},
                       indent=2, default=str), encoding="utf-8",
        )
        trades = row.get("_trades", [])
        if trades:
            with (sub / "trades.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
                w.writeheader(); w.writerows(trades)

    # Save rolling decisions as JSON for charting
    decisions_out = {
        str(rd.date()): {
            tk: {"score": sc, "tier": ti}
            for tk, (sc, ti) in tick_d.items()
        }
        for rd, tick_d in decisions.items()
    }
    (OUTPUT_DIR / "rolling_decisions.json").write_text(
        json.dumps(decisions_out, indent=2), encoding="utf-8",
    )
    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  Rolling decisions: {OUTPUT_DIR}/rolling_decisions.json")


if __name__ == "__main__":
    main()
