"""
Run the big-tech + energy theme allocator using a slower quarterly review cadence.

This reuses the same universe and sleeve-cap setup as the monthly energy run, but
refreshes theme allocation decisions every 90 days instead of every 30.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django

django.setup()

from tools.session_turtle_core_x2 import (
    run_bigtech_theme_allocator_with_energy_backtest as monthly_run,
)


OUTPUT_DIR = Path("reports/strategy_health_audit/bigtech_theme_allocator_energy_quarterly")
THEME_REVIEW_FREQ_DAYS = 90


def main() -> None:
    monthly_run.OUTPUT_DIR = OUTPUT_DIR
    monthly_run.THEME_REVIEW_FREQ_DAYS = THEME_REVIEW_FREQ_DAYS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]

    print("\nPreparing Tiingo cache for added symbols...")
    monthly_run._ensure_tiingo_cache_symbols()

    print("\nExtending core universe with big tech and energy...")
    custom_universe = monthly_run._extend_universe()
    unique_tickers = sorted({ticker for ticker, _, _ in custom_universe})
    print(f"  Tickers: {len(unique_tickers)} ({', '.join(unique_tickers)})")

    print("\nLoading macro overlay state...")
    vix_closes = json.loads((root / "cache/sentiment/vix_closes.json").read_text())
    crypto_fg = json.loads((root / "cache/sentiment/crypto_fg_scores.json").read_text())
    macro_state = monthly_run.build_extended_hours_proxy_state(
        daily_vix_closes=vix_closes,
        crypto_fg_scores=crypto_fg,
    )

    print("Building per-asset technical state...")
    tech_state = monthly_run.build_per_asset_technical_state(
        universe=list(dict.fromkeys(custom_universe)),
        lookback_years=5.0,
        warmup_days=300,
        ema_period=monthly_run._EMA_PERIOD,
        adx_period=14,
    )

    print("Building candidate trades...")
    candidates = monthly_run.build_session_turtle_shared_account_candidates(
        basket=monthly_run._BASKET,
        initial_capital=1_000.0,
        lookback_years=monthly_run._LOOKBACK_YRS,
        channel_period=monthly_run._CHANNEL,
        base_risk_pct=monthly_run._BASE_RISK,
        fixed_stop_pct=monthly_run._FIXED_STOP,
        directional_volume_risk_pct=monthly_run._DIR_VOL_RISK,
        trend_fast_period=monthly_run._TREND_FAST,
        trend_slow_period=monthly_run._TREND_SLOW,
    )
    print(f"  Candidates: {len(candidates)}")

    print("\nStep 1 - Base run with explicit sleeve caps...")
    base = monthly_run._run_variant(
        label="Base + energy sleeve + GOOGL/META/NVDA",
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
            "bucket_avg_mults": {bucket: 1.0 for bucket in monthly_run.THEME_BUCKETS},
        },
    )
    print(
        f"  Return {base['total_return_pct']:.1f}%  CAGR {base['cagr_pct']:.1f}%  "
        f"MaxDD {base['max_realized_drawdown_pct']:.1f}%  PF {base['profit_factor']:.2f}  "
        f"WR {base['win_rate_pct']:.1f}%  Trades {base['executed_trades']}"
    )

    strategy_start = min(datetime.fromisoformat(str(candidate["entry_ts"])) for candidate in candidates)
    last_entry = max(datetime.fromisoformat(str(candidate["entry_ts"])) for candidate in candidates)

    print("\nStep 2 - Building quarterly theme allocation decisions...")
    decisions = monthly_run._build_theme_decisions(
        candidates=candidates,
        strategy_start=strategy_start,
        last_entry=last_entry,
    )
    scaled_candidates, theme_stats = monthly_run._apply_theme_allocator(
        candidates=candidates,
        decisions=decisions,
    )

    print("\nStep 3 - Quarterly theme allocator run...")
    themed = monthly_run._run_variant(
        label="Quarterly theme allocator + energy sleeve + GOOGL/META/NVDA",
        candidates=scaled_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_summary={
            "theme_allocator_review_freq_days": monthly_run.THEME_REVIEW_FREQ_DAYS,
            "theme_allocator_lookback_days": monthly_run.THEME_LOOKBACK_DAYS,
            "theme_allocator_floor_mult": monthly_run.THEME_FLOOR_MULT,
            "theme_allocator_cap_mult": monthly_run.THEME_CAP_MULT,
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
        monthly_run._save_variant(row)

    (OUTPUT_DIR / "theme_decisions.json").write_text(
        json.dumps(
            {str(review_dt.date()): payload for review_dt, payload in decisions.items()},
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
            "metals_trades": row["metals_trades"],
            "energy_trades": row["energy_trades"],
            "equity_trades": row["equity_trades"],
            "crypto_pnl": row["crypto_pnl"],
            "gold_pnl": row["gold_pnl"],
            "metals_pnl": row["metals_pnl"],
            "energy_pnl": row["energy_pnl"],
            "equity_pnl": row["equity_pnl"],
            "avg_theme_alloc_mult": row["avg_theme_alloc_mult"],
            "entries_theme_upscaled": row["entries_theme_upscaled"],
            "entries_theme_downscaled": row["entries_theme_downscaled"],
            "bucket_avg_mults": row["bucket_avg_mults"],
            "trades_by_ticker": monthly_run._build_trade_counts(row["_trades"]),
            "added_equities": list(monthly_run.ADDED_EQUITIES),
            "added_energy": list(monthly_run.ADDED_ENERGY),
            "removed_tickers": list(monthly_run.REMOVED_TICKERS),
        }
        for row in rows
    }
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("BIG TECH + ENERGY THEME ALLOCATOR COMPARISON (QUARTERLY REVIEW)")
    print("=" * 112)
    print(f"{'Variant':<60} {'Return %':>10} {'CAGR %':>9} {'MaxDD %':>9} {'PF':>6} {'WR %':>7} {'Trades':>8}")
    print("-" * 112)
    for row in rows:
        print(
            f"{row['variant_label']:<60} {row['total_return_pct']:>10.2f} {row['cagr_pct']:>9.2f} "
            f"{row['max_realized_drawdown_pct']:>9.2f} {row['profit_factor']:>6.2f} "
            f"{row['win_rate_pct']:>7.2f} {row['executed_trades']:>8}"
        )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
