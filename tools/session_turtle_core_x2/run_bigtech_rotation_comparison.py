"""
Compare base vs all-asset rotation after swapping the alt-crypto sleeve for
large-cap equities.

Universe change relative to the prior alt-rotation experiment:
  remove: LTC-USD, ENJ-USD, BCH-USD, ATOM-USD
  add   : GOOGL, META, NVDA

Both variants keep the same per-asset EMA cutoff used in the recent rotation
tests. The comparison focuses only on:
  1. Base universe with EMA cutoff
  2. All-asset rotation with EMA cutoff
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
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

OUTPUT_DIR = Path("reports/strategy_health_audit/bigtech_rotation_comparison")
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

_ROTATION_KWARGS = dict(
    use_performance_leadership_overlay=True,
    performance_lookback_trades=6,
    performance_decay=0.75,
    performance_floor_mult=0.50,
    performance_cap_mult=1.50,
    performance_min_history=3,
)


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


def _run_variant(*, label: str, candidates: list[dict], macro_state: dict, tech_state: dict, extra_kwargs: dict | None = None) -> dict:
    kwargs = dict(**_BASE_KWARGS, **_OVERLAY_KWARGS)
    kwargs["extended_hours_proxy_state"] = macro_state
    kwargs["per_asset_technical_state"] = tech_state
    kwargs["precomputed_candidates"] = candidates
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    result = generate_session_turtle_shared_account_report(**kwargs)
    summary = dict(result["summary"])
    summary["variant_label"] = label
    summary["removed_tickers"] = list(REMOVED_TICKERS)
    summary["added_equities"] = list(ADDED_EQUITIES)
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

    print("\nStep 1 - Base run...")
    base = _run_variant(
        label="Base + GOOGL/META/NVDA",
        candidates=candidates,
        macro_state=macro_state,
        tech_state=tech_state,
    )
    print(
        f"  Return {base['total_return_pct']:.1f}%  CAGR {base['cagr_pct']:.1f}%  "
        f"MaxDD {base['max_realized_drawdown_pct']:.1f}%  PF {base['profit_factor']:.2f}  "
        f"WR {base['win_rate_pct']:.1f}%  Trades {base['executed_trades']}"
    )

    print("\nStep 2 - All-asset rotation run...")
    rotation = _run_variant(
        label="All asset rotation + GOOGL/META/NVDA",
        candidates=candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_kwargs=dict(_ROTATION_KWARGS),
    )
    print(
        f"  Return {rotation['total_return_pct']:.1f}%  CAGR {rotation['cagr_pct']:.1f}%  "
        f"MaxDD {rotation['max_realized_drawdown_pct']:.1f}%  PF {rotation['profit_factor']:.2f}  "
        f"WR {rotation['win_rate_pct']:.1f}%  Trades {rotation['executed_trades']}"
    )

    rows = [base, rotation]
    for row in rows:
        _save_variant(row)

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
            "trades_by_ticker": _build_trade_counts(row["_trades"]),
            "added_equities": list(ADDED_EQUITIES),
            "removed_tickers": list(REMOVED_TICKERS),
        }
        for row in rows
    }
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("BIG TECH ROTATION COMPARISON")
    print("=" * 92)
    print(f"{'Variant':<40} {'Return %':>10} {'CAGR %':>9} {'MaxDD %':>9} {'PF':>6} {'WR %':>7} {'Trades':>8}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['variant_label']:<40} {row['total_return_pct']:>10.2f} {row['cagr_pct']:>9.2f} "
            f"{row['max_realized_drawdown_pct']:>9.2f} {row['profit_factor']:>6.2f} "
            f"{row['win_rate_pct']:>7.2f} {row['executed_trades']:>8}"
        )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
