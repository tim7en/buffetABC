"""Quick what-if: run the Document Review backtest at multiple exposure multiples."""

import sys, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.session_turtle_core_x2.run_document_strategy_review_backtest import (
    REQUESTED_SYMBOLS,
    _resolve_runnable_symbols,
    _build_custom_universe,
    _load_macro_state,
    _patched_document_universe,
    _patched_document_macro_scope,
)
from edgar.services.session_turtle_portfolio import (
    build_session_turtle_shared_account_candidates,
    generate_session_turtle_shared_account_report,
    build_per_asset_technical_state,
)

EXPOSURE_MULTS = [3.0, 4.0, 5.0]


def main():
    runnable, missing = _resolve_runnable_symbols()

    print(f"Symbols: {len(runnable)} runnable, {len(missing)} missing")
    print(f"Testing exposure multiples: {EXPOSURE_MULTS}\n")

    macro_state = _load_macro_state(ROOT)

    with _patched_document_universe(runnable) as runnable_universe, _patched_document_macro_scope():
        tech_state = build_per_asset_technical_state(
            universe=list(dict.fromkeys(runnable_universe)),
            lookback_years=5.0,
            warmup_days=300,
            ema_period=200,
            adx_period=14,
        )

        candidates = build_session_turtle_shared_account_candidates(
            basket="core",
            initial_capital=1_000.0,
            lookback_years=4.1,
            channel_period=10,
            base_risk_pct=0.05,
            fixed_stop_pct=0.10,
            directional_volume_risk_pct=0.07,
            use_breakout_conviction_boost=True,
            conviction_max_mult=1.25,
            trend_fast_period=55,
            trend_slow_period=200,
        )

        print(f"{'Exp':>4s}  {'Trades':>6s}  {'Final $':>12s}  {'Return%':>10s}  {'CAGR%':>8s}  {'MaxDD%':>7s}  {'PF':>5s}  {'WR%':>5s}  {'Zero$':>5s}")
        print("-" * 80)

        for mult in EXPOSURE_MULTS:
            result = generate_session_turtle_shared_account_report(
                basket="core",
                exposure_mult=mult,
                use_drawdown_governor=True,
                drawdown_trigger_1_pct=15.0,
                drawdown_exposure_mult_1=1.5,
                drawdown_trigger_2_pct=25.0,
                drawdown_exposure_mult_2=0.5,
                crypto_cap_mult=1.0,
                gold_cap_mult=1.0,
                metals_cap_mult=1.0,
                energy_cap_mult=1.0,
                equity_cap_mult=None,
                base_risk_pct=0.05,
                fixed_stop_pct=0.10,
                directional_volume_risk_pct=0.07,
                lookback_years=4.1,
                channel_period=10,
                use_breakout_conviction_boost=True,
                conviction_max_mult=1.25,
                trend_fast_period=55,
                trend_slow_period=200,
                precomputed_candidates=candidates,
                use_extended_hours_proxy=True,
                extended_hours_proxy_state=macro_state,
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
                per_asset_technical_state=tech_state,
                per_asset_ema_lag_days=1,
                per_asset_ema_above_long_mult=1.0,
                per_asset_ema_above_short_mult=0.25,
                per_asset_ema_below_long_mult=0.25,
                per_asset_ema_below_short_mult=1.0,
                per_asset_use_adx_gate=False,
            )

            s = result["summary"]
            trades = result["trades"]
            zero_notional = sum(1 for t in trades if abs(t.get("notional", 0)) < 1e-9)
            wins = sum(1 for t in trades if t["net_pnl"] > 0)
            wr = wins / len(trades) * 100 if trades else 0

            print(f" x{mult:<3.0f}  {s['executed_trades']:>6d}  ${s['final_equity']:>10,.2f}  "
                  f"{s['total_return_pct']:>+9.2f}%  {s['cagr_pct']:>7.2f}%  {s['max_realized_drawdown_pct']:>6.2f}%  "
                  f"{s['profit_factor']:>5.2f}  {wr:>4.1f}%  {zero_notional:>5d}")


if __name__ == "__main__":
    main()
