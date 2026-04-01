"""
Run a bucket-level theme allocator on the big-tech modified core universe.

Universe change:
  remove: LTC-USD, ENJ-USD, BCH-USD, ATOM-USD
  add   : GOOGL, META, NVDA

Variants:
1. EMA cutoff base with explicit theme caps
2. EMA cutoff base + bucket/theme allocator

The theme allocator is intentionally mild:
  - monthly review cadence
  - 13-week trailing lookback window
  - bucket multipliers in the range 0.75x -> 1.25x
  - no per-asset leadership overlay inside buckets
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

import edgar.services.session_turtle_portfolio as stp
from edgar.services.session_turtle_portfolio import (
    build_extended_hours_proxy_state,
    build_per_asset_technical_state,
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
)


REMOVED_TICKERS = ("LTC-USD", "ENJ-USD", "BCH-USD", "ATOM-USD")
ADDED_EQUITIES = ("GOOGL", "META", "NVDA")

OUTPUT_DIR = Path("reports/strategy_health_audit/bigtech_theme_allocator")
FALLBACK_TIINGO_DIR = Path("cache/cache/cache")
PRIMARY_TIINGO_DIR = Path("cache/cache/tiingo")

_BASKET = "core"
_LOOKBACK_YRS = 4.1
_CHANNEL = 20
_EXPOSURE_MULT = 3.0
_BASE_RISK = 0.05
_FIXED_STOP = 0.10
_DIR_VOL_RISK = 0.07
_TREND_FAST = 55
_TREND_SLOW = 200
_EMA_PERIOD = 200

_BASE_KWARGS = dict(
    basket=_BASKET,
    exposure_mult=_EXPOSURE_MULT,
    crypto_cap_mult=1.0,
    gold_cap_mult=0.8,
    metals_cap_mult=0.8,
    equity_cap_mult=1.0,
    base_risk_pct=_BASE_RISK,
    fixed_stop_pct=_FIXED_STOP,
    directional_volume_risk_pct=_DIR_VOL_RISK,
)

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

THEME_FLOOR_MULT = 0.75
THEME_CAP_MULT = 1.25
THEME_MIN_HISTORY = 3
THEME_REVIEW_FREQ_DAYS = 30
THEME_LOOKBACK_DAYS = 91
THEME_DECAY = 0.75
THEME_LOOKBACK_TRADES = 6
THEME_BUCKETS = ("crypto", "gold", "metals", "equity")


def _ensure_tiingo_cache_symbols() -> None:
    PRIMARY_TIINGO_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in ADDED_EQUITIES:
        dst = PRIMARY_TIINGO_DIR / f"{symbol}_5m.parquet"
        if dst.exists():
            continue
        src = FALLBACK_TIINGO_DIR / f"{symbol}_5m.parquet"
        if not src.exists():
            raise FileNotFoundError(
                f"Missing local Tiingo parquet for {symbol}. "
                f"Checked {src} and {dst}."
            )
        shutil.copy2(src, dst)


def _extend_universe() -> tuple[tuple[str, str, str], ...]:
    extra = [(ticker, "tiingo", "new_york_equity_open") for ticker in ADDED_EQUITIES]
    custom_universe = tuple(list(stp.CORE_SESSION_TURTLE_UNIVERSE) + extra)
    stp.CORE_SESSION_TURTLE_UNIVERSE = custom_universe
    stp.EXPANDED_SESSION_TURTLE_UNIVERSE = tuple(
        list(custom_universe) + list(stp.INDEX_SESSION_TURTLE_UNIVERSE)
    )
    stp.EQUITY_TICKERS = set(stp.EQUITY_TICKERS) | set(ADDED_EQUITIES)
    return custom_universe


def _candidate_return_pct(candidate: dict) -> float:
    position_size = float(candidate.get("position_size", 0.0) or 0.0)
    if position_size <= 0:
        return 0.0
    return float(candidate.get("pnl", 0.0) or 0.0) / position_size


def _decayed_trade_return_score(returns: list[float]) -> float:
    recent = returns[-THEME_LOOKBACK_TRADES:]
    if not recent:
        return 0.0
    total = 0.0
    weights = 0.0
    for age, value in enumerate(reversed(recent)):
        weight = THEME_DECAY**age
        total += float(value) * weight
        weights += weight
    return total / weights if weights > 0 else 0.0


def _build_theme_decisions(
    *,
    candidates: list[dict],
    strategy_start: datetime,
    last_entry: datetime,
) -> dict[datetime, dict[str, dict]]:
    by_bucket_exit: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for candidate in candidates:
        bucket = str(candidate["asset_bucket"])
        if bucket not in THEME_BUCKETS:
            continue
        by_bucket_exit[bucket].append(
            (datetime.fromisoformat(str(candidate["exit_ts"])), _candidate_return_pct(candidate))
        )
    for bucket in by_bucket_exit:
        by_bucket_exit[bucket].sort(key=lambda item: item[0])

    decisions: dict[datetime, dict[str, dict]] = {}
    review_dt = strategy_start + timedelta(days=THEME_LOOKBACK_DAYS)
    while review_dt <= last_entry + timedelta(days=THEME_REVIEW_FREQ_DAYS):
        window_start = review_dt - timedelta(days=THEME_LOOKBACK_DAYS)
        bucket_scores: dict[str, tuple[float, int]] = {}
        for bucket in THEME_BUCKETS:
            returns = [
                ret for exit_dt, ret in by_bucket_exit.get(bucket, [])
                if window_start <= exit_dt < review_dt
            ]
            if len(returns) < THEME_MIN_HISTORY:
                continue
            bucket_scores[bucket] = (_decayed_trade_return_score(returns), len(returns))

        review_payload: dict[str, dict] = {}
        if len(bucket_scores) >= 2:
            ordered = sorted(bucket_scores.items(), key=lambda item: (item[1][0], item[0]))
            rank_map = {
                bucket: (idx / (len(ordered) - 1)) if len(ordered) > 1 else 0.5
                for idx, (bucket, _) in enumerate(ordered)
            }
            for bucket in THEME_BUCKETS:
                if bucket in bucket_scores:
                    score, trade_count = bucket_scores[bucket]
                    rank_pct = rank_map[bucket]
                    mult = THEME_FLOOR_MULT + rank_pct * (THEME_CAP_MULT - THEME_FLOOR_MULT)
                    review_payload[bucket] = {
                        "score": round(score, 6),
                        "trade_count": trade_count,
                        "rank_pct": round(rank_pct, 4),
                        "mult": round(mult, 4),
                    }
                else:
                    review_payload[bucket] = {
                        "score": None,
                        "trade_count": 0,
                        "rank_pct": None,
                        "mult": 1.0,
                    }
        else:
            for bucket in THEME_BUCKETS:
                trade_count = bucket_scores.get(bucket, (None, 0))[1]
                score = bucket_scores.get(bucket, (None, 0))[0]
                review_payload[bucket] = {
                    "score": round(score, 6) if score is not None else None,
                    "trade_count": trade_count,
                    "rank_pct": None,
                    "mult": 1.0,
                }
        decisions[review_dt] = review_payload
        review_dt += timedelta(days=THEME_REVIEW_FREQ_DAYS)
    return decisions


def _apply_theme_allocator(
    *,
    candidates: list[dict],
    decisions: dict[datetime, dict[str, dict]],
) -> tuple[list[dict], dict]:
    sorted_reviews = sorted(decisions)
    scaled_candidates: list[dict] = []
    mults: list[float] = []
    upscaled = 0
    downscaled = 0
    bucket_mult_totals: dict[str, list[float]] = defaultdict(list)

    for candidate in candidates:
        entry_ts = datetime.fromisoformat(str(candidate["entry_ts"]))
        bucket = str(candidate["asset_bucket"])
        mult = 1.0
        applicable = None
        for review_dt in sorted_reviews:
            if review_dt <= entry_ts:
                applicable = review_dt
            else:
                break
        if applicable is not None and bucket in decisions[applicable]:
            mult = float(decisions[applicable][bucket]["mult"])

        scaled = deepcopy(candidate)
        scaled["risk_pct"] = float(candidate.get("risk_pct", 0.0) or 0.0) * mult
        scaled["position_size"] = float(candidate.get("position_size", 0.0) or 0.0) * mult
        scaled["theme_alloc_mult"] = mult
        scaled_candidates.append(scaled)
        mults.append(mult)
        bucket_mult_totals[bucket].append(mult)
        if mult > 1.000001:
            upscaled += 1
        elif mult < 0.999999:
            downscaled += 1

    stats = {
        "avg_theme_alloc_mult": round(sum(mults) / len(mults), 4) if mults else 1.0,
        "entries_theme_upscaled": upscaled,
        "entries_theme_downscaled": downscaled,
        "bucket_avg_mults": {
            bucket: round(sum(values) / len(values), 4)
            for bucket, values in sorted(bucket_mult_totals.items())
            if values
        },
    }
    return scaled_candidates, stats


def _run_variant(
    *,
    label: str,
    candidates: list[dict],
    macro_state: dict,
    tech_state: dict,
    extra_summary: dict | None = None,
) -> dict:
    kwargs = dict(**_BASE_KWARGS, **_OVERLAY_KWARGS)
    kwargs["extended_hours_proxy_state"] = macro_state
    kwargs["per_asset_technical_state"] = tech_state
    kwargs["precomputed_candidates"] = candidates
    result = generate_session_turtle_shared_account_report(**kwargs)
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["removed_tickers"] = list(REMOVED_TICKERS)
    summary["added_equities"] = list(ADDED_EQUITIES)
    if extra_summary:
        summary.update(extra_summary)
    summary["_trades"] = result["trades"]
    return summary


def _save_variant(row: dict) -> None:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in row["variant_label"].lower().replace(" ", "_"))
    subdir = OUTPUT_DIR / safe
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "summary.json").write_text(
        json.dumps({key: value for key, value in row.items() if key != "_trades"}, indent=2, default=str),
        encoding="utf-8",
    )
    trades = row.get("_trades", [])
    if not trades:
        return
    with (subdir / "trades.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)


def _build_trade_counts(trades: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        ticker = str(trade["ticker"])
        counts[ticker] = counts.get(ticker, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]

    print("\nPreparing Tiingo cache for added equities...")
    _ensure_tiingo_cache_symbols()

    print("\nExtending core universe with large-cap equities...")
    custom_universe = _extend_universe()
    unique_tickers = sorted({ticker for ticker, _, _ in custom_universe})
    print(f"  Tickers: {len(unique_tickers)} ({', '.join(unique_tickers)})")

    print("\nLoading macro overlay state...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = build_extended_hours_proxy_state(daily_vix_closes=vix_closes, crypto_fg_scores=crypto_fg)

    print("Building per-asset technical state...")
    tech_state = build_per_asset_technical_state(
        universe=list(dict.fromkeys(custom_universe)),
        lookback_years=5.0,
        warmup_days=300,
        ema_period=_EMA_PERIOD,
        adx_period=14,
    )

    print("Building candidate trades...")
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
    print(f"  Candidates: {len(candidates)}")

    print("\nStep 1 - Base run with explicit equity cap...")
    base = _run_variant(
        label="Base + equity cap + GOOGL/META/NVDA",
        candidates=candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_summary={
            "theme_allocator_review_freq_days": None,
            "theme_allocator_lookback_days": None,
            "theme_allocator_floor_mult": None,
            "theme_allocator_cap_mult": None,
            "avg_theme_alloc_mult": 1.0,
            "entries_theme_upscaled": 0,
            "entries_theme_downscaled": 0,
            "bucket_avg_mults": {bucket: 1.0 for bucket in THEME_BUCKETS},
        },
    )
    print(
        f"  Return {base['total_return_pct']:.1f}%  CAGR {base['cagr_pct']:.1f}%  "
        f"MaxDD {base['max_realized_drawdown_pct']:.1f}%  PF {base['profit_factor']:.2f}  "
        f"WR {base['win_rate_pct']:.1f}%  Trades {base['executed_trades']}"
    )

    strategy_start = min(datetime.fromisoformat(str(candidate["entry_ts"])) for candidate in candidates)
    last_entry = max(datetime.fromisoformat(str(candidate["entry_ts"])) for candidate in candidates)

    print("\nStep 2 - Building theme allocation decisions...")
    decisions = _build_theme_decisions(
        candidates=candidates,
        strategy_start=strategy_start,
        last_entry=last_entry,
    )
    scaled_candidates, theme_stats = _apply_theme_allocator(
        candidates=candidates,
        decisions=decisions,
    )

    print("\nStep 3 - Theme allocator run...")
    themed = _run_variant(
        label="Theme allocator + GOOGL/META/NVDA",
        candidates=scaled_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_summary={
            "theme_allocator_review_freq_days": THEME_REVIEW_FREQ_DAYS,
            "theme_allocator_lookback_days": THEME_LOOKBACK_DAYS,
            "theme_allocator_floor_mult": THEME_FLOOR_MULT,
            "theme_allocator_cap_mult": THEME_CAP_MULT,
            **theme_stats,
        },
    )
    print(
        f"  Return {themed['total_return_pct']:.1f}%  CAGR {themed['cagr_pct']:.1f}%  "
        f"MaxDD {themed['max_realized_drawdown_pct']:.1f}%  PF {themed['profit_factor']:.2f}  "
        f"WR {themed['win_rate_pct']:.1f}%  Trades {themed['executed_trades']}"
    )

    rows = [base, themed]
    for row in rows:
        _save_variant(row)

    (OUTPUT_DIR / "theme_decisions.json").write_text(
        json.dumps(
            {
                str(review_dt.date()): payload
                for review_dt, payload in decisions.items()
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    comparison = {
        row["variant_label"]: {
            "candidate_universe_size": row["candidate_universe_size"],
            "candidate_trades": row["candidate_trades"],
            "executed_trades": row["executed_trades"],
            "total_return_pct": row["total_return_pct"],
            "cagr_pct": row["cagr_pct"],
            "max_realized_drawdown_pct": row["max_realized_drawdown_pct"],
            "profit_factor": row["profit_factor"],
            "win_rate_pct": row["win_rate_pct"],
            "final_equity": row["final_equity"],
            "crypto_trades": row["crypto_trades"],
            "gold_trades": row["gold_trades"],
            "equity_trades": row["equity_trades"],
            "metals_trades": row["metals_trades"],
            "crypto_pnl": row["crypto_pnl"],
            "gold_pnl": row["gold_pnl"],
            "equity_pnl": row["equity_pnl"],
            "metals_pnl": row["metals_pnl"],
            "avg_theme_alloc_mult": row["avg_theme_alloc_mult"],
            "entries_theme_upscaled": row["entries_theme_upscaled"],
            "entries_theme_downscaled": row["entries_theme_downscaled"],
            "bucket_avg_mults": row["bucket_avg_mults"],
            "trades_by_ticker": _build_trade_counts(row["_trades"]),
            "added_equities": list(ADDED_EQUITIES),
            "removed_tickers": list(REMOVED_TICKERS),
        }
        for row in rows
    }
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "=" * 94)
    print("BIG TECH THEME ALLOCATOR COMPARISON")
    print("=" * 94)
    print(f"{'Variant':<42} {'Return %':>10} {'CAGR %':>9} {'MaxDD %':>9} {'PF':>6} {'WR %':>7} {'Trades':>8}")
    print("-" * 94)
    for row in rows:
        print(
            f"{row['variant_label']:<42} {row['total_return_pct']:>10.2f} {row['cagr_pct']:>9.2f} "
            f"{row['max_realized_drawdown_pct']:>9.2f} {row['profit_factor']:>6.2f} "
            f"{row['win_rate_pct']:>7.2f} {row['executed_trades']:>8}"
        )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
