"""
Compare base vs rotation-focused performance on an alt-expanded core universe.

Variants:
1. Base alt-expanded
2. All-asset rotation (leadership overlay across all assets)
3. Rotation-focused (all-asset rotation + moderator prefilter)

Run:
  python tools/session_turtle_core_x2/run_alt_crypto_rotation_comparison.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
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


ALT_CRYPTO = ("LTC-USD", "ENJ-USD", "BCH-USD", "ATOM-USD")
OUTPUT_DIR = Path("reports/strategy_health_audit/alt_rotation_comparison")

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

_LIGHT_REVIEW_DAYS = 14
_LIGHT_LOOKBACK_DAYS = 91
_LIGHT_BUCKET_MIN_TRADES = 4


def _load_fund_manager_module():
    module_path = Path(__file__).with_name("run_fund_manager_backtest.py")
    spec = importlib.util.spec_from_file_location("run_fund_manager_backtest_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _extend_universe() -> tuple[tuple[str, str, str], ...]:
    extra = []
    for ticker in ALT_CRYPTO:
        extra.append((ticker, "binance", "hong_kong_open"))
        extra.append((ticker, "binance", "new_york_equity_open"))
    custom_universe = tuple(list(stp.CORE_SESSION_TURTLE_UNIVERSE) + extra)
    stp.CORE_SESSION_TURTLE_UNIVERSE = custom_universe
    stp.EXPANDED_SESSION_TURTLE_UNIVERSE = tuple(list(custom_universe) + list(stp.INDEX_SESSION_TURTLE_UNIVERSE))
    stp.CRYPTO_TICKERS = set(stp.CRYPTO_TICKERS) | set(ALT_CRYPTO)
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


def _serialize_decisions(decisions: dict) -> dict:
    out = {}
    for review_dt, ticker_map in decisions.items():
        out[str(review_dt.date())] = {
            ticker: {"score": score, "tier": tier, "keep_ratio": ratio, "metrics": metrics}
            for ticker, (score, tier, ratio, metrics) in ticker_map.items()
        }
    return out


def _safe_pf_from_pnls(pnls: list[float]) -> float:
    gross_win = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    if gross_loss <= 1e-9:
        return 999.0 if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _build_light_bucket_decisions(
    *,
    baseline_trades: list[dict],
    candidates: list[dict],
    fm,
    strategy_start: datetime,
    last_entry: datetime,
) -> dict[datetime, dict[str, dict]]:
    ticker_to_bucket = {str(candidate["ticker"]): str(candidate["asset_bucket"]) for candidate in candidates}
    all_tickers = sorted(ticker_to_bucket)
    decisions: dict[datetime, dict[str, dict]] = {}
    review_dt = strategy_start + timedelta(days=_LIGHT_REVIEW_DAYS)

    while review_dt <= last_entry:
        window_start = review_dt - timedelta(days=_LIGHT_LOOKBACK_DAYS)
        bucket_trades: dict[str, list[dict]] = {}
        for bucket in {"crypto", "gold", "metals", "equity"}:
            bucket_trades[bucket] = [
                trade for trade in baseline_trades
                if window_start <= datetime.fromisoformat(trade["exit_ts"]) < review_dt
                and str(trade.get("asset_bucket")) == bucket
            ]

        bucket_state: dict[str, tuple[float, str, dict]] = {}
        for bucket, trades_in_bucket in bucket_trades.items():
            pnls = [float(trade["net_pnl"]) for trade in trades_in_bucket]
            total_pnl = sum(pnls)
            pf = _safe_pf_from_pnls(pnls) if pnls else 0.0
            if len(trades_in_bucket) < _LIGHT_BUCKET_MIN_TRADES:
                ratio, state = 1.0, "insufficient"
            elif pf < 0.85 and total_pnl < 0:
                ratio, state = 0.50, "weak_hard"
            elif pf < 1.0 or total_pnl < 0:
                ratio, state = 0.75, "weak"
            else:
                ratio, state = 1.0, "ok"
            bucket_state[bucket] = (
                ratio,
                state,
                {"trades": len(trades_in_bucket), "profit_factor": round(pf, 3), "net_pnl": round(total_pnl, 2)},
            )

        ticker_map: dict[str, dict] = {}
        for ticker in all_tickers:
            window_trades = [
                trade for trade in baseline_trades
                if window_start <= datetime.fromisoformat(trade["exit_ts"]) < review_dt and trade["ticker"] == ticker
            ]
            score, tier, metrics = fm._score_asset(window_trades)
            bucket = ticker_to_bucket[ticker]
            bucket_ratio, bucket_label, bucket_metrics = bucket_state[bucket]
            if tier in {"INSUFFICIENT", "A", "B"}:
                asset_ratio = 1.0
            elif tier == "C":
                asset_ratio = 0.85
            else:
                asset_ratio = 0.70 if bucket_ratio >= 0.75 else 0.50
            final_ratio = max(0.25, min(1.0, bucket_ratio * asset_ratio))
            ticker_map[ticker] = {
                "score": score,
                "tier": tier,
                "asset_ratio": round(asset_ratio, 4),
                "bucket_ratio": round(bucket_ratio, 4),
                "keep_ratio": round(final_ratio, 4),
                "bucket_state": bucket_label,
                "bucket_metrics": bucket_metrics,
                "metrics": metrics,
            }
        decisions[review_dt] = ticker_map
        review_dt += timedelta(days=_LIGHT_REVIEW_DAYS)

    return decisions


def _apply_light_bucket_filter(
    *,
    all_candidates: list[dict],
    decisions: dict[datetime, dict[str, dict]],
    strategy_start: datetime,
) -> list[dict]:
    observe_cutoff = strategy_start + timedelta(days=28)
    sorted_reviews = sorted(decisions)
    ticker_counters: dict[tuple[str, datetime], int] = {}
    filtered: list[dict] = []

    for candidate in all_candidates:
        entry_ts = candidate["entry_ts"]
        if entry_ts < observe_cutoff:
            filtered.append(candidate)
            continue

        applicable = None
        for review_dt in sorted_reviews:
            if review_dt <= entry_ts:
                applicable = review_dt
            else:
                break
        if applicable is None:
            filtered.append(candidate)
            continue

        ticker = str(candidate["ticker"])
        ratio = float(decisions[applicable][ticker]["keep_ratio"])
        if ratio >= 0.999:
            filtered.append(candidate)
            continue

        key = (ticker, applicable)
        ticker_counters[key] = ticker_counters.get(key, 0) + 1
        n = max(1, round(1.0 / max(ratio, 1e-9)))
        if ticker_counters[key] % n == 0:
            filtered.append(candidate)

    return filtered


def _safe_direction_stats(trades: list[dict], direction: str, total_pnl: float) -> dict:
    side_trades = [trade for trade in trades if str(trade.get("direction")) == direction]
    pnls = [float(trade["net_pnl"]) for trade in side_trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    pf = None if gross_loss <= 1e-9 else gross_profit / gross_loss
    pnl = sum(pnls)
    return {
        "trades": len(side_trades),
        "pnl": round(pnl, 4),
        "avg_pnl": round((pnl / len(side_trades)) if side_trades else 0.0, 4),
        "win_rate_pct": round((len(wins) / len(side_trades) * 100.0) if side_trades else 0.0, 2),
        "profit_factor": None if pf is None else round(pf, 3),
        "share_of_total_pnl_pct": round((pnl / total_pnl * 100.0) if abs(total_pnl) > 1e-9 else 0.0, 2),
    }


def _build_underwater_stats(summary: dict, trades: list[dict]) -> dict:
    start_day = datetime.fromisoformat(str(summary["start_date"])).date()
    end_day = datetime.fromisoformat(str(summary["end_date"])).date()
    initial = float(summary["initial_capital"])

    last_equity_by_day: dict = {}
    for trade in sorted(trades, key=lambda item: datetime.fromisoformat(item["exit_ts"])):
        if trade.get("equity_after_exit") is None:
            continue
        last_equity_by_day[datetime.fromisoformat(trade["exit_ts"]).date()] = float(trade["equity_after_exit"])

    daily_equity: list[tuple] = []
    current_day = start_day
    equity = initial
    while current_day <= end_day:
        if current_day in last_equity_by_day:
            equity = last_equity_by_day[current_day]
        daily_equity.append((current_day, equity))
        current_day += timedelta(days=1)

    peak = initial
    underwater_days = 0
    current_episode = 0
    episode_lengths: list[int] = []
    max_dd_pct = 0.0

    for _day, equity in daily_equity:
        peak = max(peak, equity)
        dd_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd_pct)
        is_underwater = dd_pct > 0.001
        if is_underwater:
            underwater_days += 1
            current_episode += 1
        elif current_episode:
            episode_lengths.append(current_episode)
            current_episode = 0

    if current_episode:
        episode_lengths.append(current_episode)

    return {
        "time_underwater_pct": round((underwater_days / len(daily_equity) * 100.0) if daily_equity else 0.0, 2),
        "avg_underwater_days": round((sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0, 1),
        "max_underwater_days": max(episode_lengths) if episode_lengths else 0,
        "ongoing_underwater_days": current_episode,
        "episodes": len(episode_lengths),
        "max_drawdown_pct_from_daily_curve": round(max_dd_pct, 2),
    }


def _build_yearly_rows(trades: list[dict], initial_capital: float) -> list[dict]:
    grouped: dict[int, dict] = {}
    for trade in trades:
        year = datetime.fromisoformat(trade["exit_ts"]).year
        row = grouped.setdefault(
            year,
            {"pnl": 0.0, "trades": 0, "long_trades": 0, "short_trades": 0, "winning_trades": 0},
        )
        pnl = float(trade["net_pnl"])
        row["pnl"] += pnl
        row["trades"] += 1
        if str(trade.get("direction")) == "long":
            row["long_trades"] += 1
        else:
            row["short_trades"] += 1
        if pnl > 0:
            row["winning_trades"] += 1

    start_equity = float(initial_capital)
    yearly_rows: list[dict] = []
    for year in sorted(grouped):
        row = grouped[year]
        pnl = float(row["pnl"])
        end_equity = start_equity + pnl
        yearly_rows.append(
            {
                "year": year,
                "start_equity": round(start_equity, 4),
                "end_equity": round(end_equity, 4),
                "pnl": round(pnl, 4),
                "return_pct": round((pnl / start_equity * 100.0) if start_equity > 0 else 0.0, 2),
                "trades": row["trades"],
                "long_trades": row["long_trades"],
                "short_trades": row["short_trades"],
                "win_rate_pct": round((row["winning_trades"] / row["trades"] * 100.0) if row["trades"] else 0.0, 2),
            }
        )
        start_equity = end_equity
    return yearly_rows


def _build_variant_diagnostics(summary: dict) -> dict:
    trades = summary.get("_trades", [])
    total_pnl = sum(float(trade["net_pnl"]) for trade in trades)
    yearly = _build_yearly_rows(trades, float(summary["initial_capital"]))
    negative_years = [row for row in yearly if float(row.get("return_pct", 0.0)) < 0]
    return {
        "directional_breakdown": {
            "long": _safe_direction_stats(trades, "long", total_pnl),
            "short": _safe_direction_stats(trades, "short", total_pnl),
        },
        "yearly_returns": yearly,
        "yearly_health": {
            "negative_year_count": len(negative_years),
            "worst_year": min(
                (
                    {
                        "year": int(row["year"]),
                        "return_pct": float(row["return_pct"]),
                        "pnl": float(row["pnl"]),
                    }
                    for row in yearly
                ),
                key=lambda row: row["return_pct"],
                default=None,
            ),
            "best_year": max(
                (
                    {
                        "year": int(row["year"]),
                        "return_pct": float(row["return_pct"]),
                        "pnl": float(row["pnl"]),
                    }
                    for row in yearly
                ),
                key=lambda row: row["return_pct"],
                default=None,
            ),
        },
        "underwater": _build_underwater_stats(summary, trades),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]

    print("\nExtending core universe with alt crypto...")
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

    print("\nStep 1 - Base alt-expanded run...")
    base = _run_variant(label="Base alt expanded", candidates=candidates, macro_state=macro_state, tech_state=tech_state)
    print(
        f"  Return {base['total_return_pct']:.1f}%  CAGR {base['cagr_pct']:.1f}%  "
        f"MaxDD {base['max_realized_drawdown_pct']:.1f}%  PF {base['profit_factor']:.2f}  "
        f"WR {base['win_rate_pct']:.1f}%  Trades {base['executed_trades']}"
    )

    print("\nStep 2 - All-asset rotation run...")
    rotation = _run_variant(
        label="All asset rotation",
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

    fm = _load_fund_manager_module()
    strategy_start = min(candidate["entry_ts"] for candidate in candidates)
    last_entry = max(candidate["entry_ts"] for candidate in candidates)
    all_tickers = sorted({candidate["ticker"] for candidate in candidates})

    print("\nStep 3 - Light bucket-aware moderator run...")
    light_decisions = _build_light_bucket_decisions(
        baseline_trades=rotation["_trades"],
        candidates=candidates,
        fm=fm,
        strategy_start=strategy_start,
        last_entry=last_entry,
    )
    light_filtered = _apply_light_bucket_filter(
        all_candidates=candidates,
        decisions=light_decisions,
        strategy_start=strategy_start,
    )
    light_rotation = _run_variant(
        label="Light bucket-aware moderator",
        candidates=light_filtered,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_kwargs=dict(_ROTATION_KWARGS),
    )
    print(
        f"  Return {light_rotation['total_return_pct']:.1f}%  CAGR {light_rotation['cagr_pct']:.1f}%  "
        f"MaxDD {light_rotation['max_realized_drawdown_pct']:.1f}%  PF {light_rotation['profit_factor']:.2f}  "
        f"WR {light_rotation['win_rate_pct']:.1f}%  Trades {light_rotation['executed_trades']}"
    )

    print("\nStep 4 - Legacy heavier moderator run...")
    ref_dd = float(rotation["max_realized_drawdown_pct"])
    fm.REF_MAX_DD_PCT = ref_dd
    fm.CB_REDUCE_DD_PCT = ref_dd * 0.60
    fm.CB_HALT_DD_PCT = ref_dd * 0.80
    fm.CB_RESUME_DD_PCT = ref_dd * 0.40

    equity_curve = fm._build_equity_curve(rotation["_trades"])
    cb_state = fm._build_circuit_breaker_state(equity_curve)
    daily_return_state = fm._build_daily_return_state(tech_state)
    asset_start_ts = {
        ticker: min(candidate["entry_ts"] for candidate in candidates if candidate["ticker"] == ticker)
        for ticker in all_tickers
    }
    asset_states = {ticker: fm.AssetState(ticker, asset_start_ts[ticker]) for ticker in all_tickers}
    decisions = fm._build_rolling_decisions(
        baseline_trades=rotation["_trades"],
        asset_states=asset_states,
        all_tickers=set(all_tickers),
        strategy_start=strategy_start,
        last_entry=last_entry,
        daily_return_state=daily_return_state,
    )
    filtered_candidates, _ = fm._apply_fund_manager_filter(
        all_candidates=candidates,
        decisions=decisions,
        strategy_start=strategy_start,
        cb_state=cb_state,
    )
    rotation_focused = _run_variant(
        label="Rotation focused (leadership + moderator)",
        candidates=filtered_candidates,
        macro_state=macro_state,
        tech_state=tech_state,
        extra_kwargs=dict(_ROTATION_KWARGS),
    )
    print(
        f"  Return {rotation_focused['total_return_pct']:.1f}%  CAGR {rotation_focused['cagr_pct']:.1f}%  "
        f"MaxDD {rotation_focused['max_realized_drawdown_pct']:.1f}%  PF {rotation_focused['profit_factor']:.2f}  "
        f"WR {rotation_focused['win_rate_pct']:.1f}%  Trades {rotation_focused['executed_trades']}"
    )

    rows = [base, rotation, light_rotation, rotation_focused]
    for row in rows:
        row["added_alt_crypto"] = list(ALT_CRYPTO)
        _save_variant(row)

    (OUTPUT_DIR / "light_bucket_decisions.json").write_text(
        json.dumps({str(review_dt.date()): ticker_map for review_dt, ticker_map in light_decisions.items()}, indent=2, default=str),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "moderator_decisions.json").write_text(
        json.dumps(_serialize_decisions(decisions), indent=2, default=str),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "circuit_breaker_events.json").write_text(
        json.dumps(
            [{"ts": str(ts.date()), "state": state} for ts, state in sorted(cb_state.items()) if state != "normal"],
            indent=2,
        ),
        encoding="utf-8",
    )

    diagnostics = {row["variant_label"]: _build_variant_diagnostics(row) for row in rows}
    (OUTPUT_DIR / "variant_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str),
        encoding="utf-8",
    )

    comparison = {
        row["variant_label"]: {
            "candidate_trades": row["candidate_trades"],
            "executed_trades": row["executed_trades"],
            "total_return_pct": row["total_return_pct"],
            "cagr_pct": row["cagr_pct"],
            "max_realized_drawdown_pct": row["max_realized_drawdown_pct"],
            "profit_factor": row["profit_factor"],
            "win_rate_pct": row["win_rate_pct"],
            "final_equity": row["final_equity"],
            "crypto_trades": row["crypto_trades"],
            "crypto_pnl": row["crypto_pnl"],
            "long_trades": row["long_trades"],
            "short_trades": row["short_trades"],
            "yearly_returns": diagnostics[row["variant_label"]]["yearly_returns"],
            "underwater": diagnostics[row["variant_label"]]["underwater"],
        }
        for row in rows
    }
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "=" * 106)
    print("ALT-EXPANDED ROTATION COMPARISON")
    print("=" * 106)
    print(f"{'Variant':<46} {'Return %':>10} {'CAGR %':>9} {'MaxDD %':>9} {'PF':>6} {'WR %':>7} {'Trades':>8}")
    print("-" * 106)
    for row in rows:
        print(
            f"{row['variant_label']:<46} {row['total_return_pct']:>10.2f} {row['cagr_pct']:>9.2f} "
            f"{row['max_realized_drawdown_pct']:>9.2f} {row['profit_factor']:>6.2f} "
            f"{row['win_rate_pct']:>7.2f} {row['executed_trades']:>8}"
        )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
