"""Test aggressive SPY-gated QQQ execution variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import qqq_macro_walkforward_leverage as leverage
import spy_gate_qqq_ensemble_backtest as combo


ROOT = combo.ROOT
DEFAULT_OUT_DIR = ROOT / "reports" / "spy_gate_qqq_aggressive_variants_20260410"
WINDOWS = combo.WINDOWS
TRADING_DAYS_PER_YEAR = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qqq-analysis-dir", type=Path, default=combo.DEFAULT_QQQ_ANALYSIS_DIR)
    parser.add_argument("--qqq-compare-dir", type=Path, default=combo.DEFAULT_QQQ_COMPARE_DIR)
    parser.add_argument("--spy-compare-dir", type=Path, default=combo.DEFAULT_SPY_COMPARE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=100.0)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument("--risk-on-leverage", type=float, default=3.0)
    parser.add_argument("--neutral-leverage", type=float, default=1.0)
    parser.add_argument("--accel-multiplier", type=float, default=2.0)
    parser.add_argument("--qqq-drop-threshold", type=float, default=-0.03)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    dataset_path = path / "aligned_daily_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing aligned dataset: {dataset_path}")
    dataset = combo.load_csv(dataset_path, index_col="date")
    required = {"qqq_close", "qqq_21d_return", "vix_level", "vix_21d_change"}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"Missing required dataset columns: {sorted(missing)}")
    return dataset


def leverage_for_regime(regime: str, risk_on_leverage: float, neutral_leverage: float) -> float:
    if regime == "risk_on":
        return float(risk_on_leverage)
    if regime == "neutral":
        return float(neutral_leverage)
    return 0.0


def build_acceleration_mask(dataset: pd.DataFrame, policy_signal: pd.Series, qqq_drop_threshold: float) -> pd.Series:
    vix_spike = (
        dataset["volatility_shock"].astype(bool)
        if "volatility_shock" in dataset.columns
        else (
            pd.to_numeric(dataset["vix_level"], errors="coerce").ge(25.0)
            | pd.to_numeric(dataset["vix_21d_change"], errors="coerce").ge(5.0)
        )
    )
    qqq_drop = pd.to_numeric(dataset["qqq_21d_return"], errors="coerce").le(float(qqq_drop_threshold))
    accel = vix_spike & qqq_drop
    accel = accel.reindex(policy_signal.index).fillna(False)
    return accel.shift(1).fillna(False).rename("accelerated_dca_signal")


def simulate_full_cash_regime(
    close: pd.Series,
    regime_signal: pd.Series,
    *,
    strategy: str,
    risk_on_leverage: float,
    neutral_leverage: float,
    initial_capital: float,
    periodic_contribution: float,
    contribution_frequency: str,
    trading_day_interval: int,
    trading_cost_bps: float,
    borrow_rate: float,
    accel_mask: pd.Series | None = None,
    accel_multiplier: float = 1.0,
) -> leverage.StrategyResult:
    signal = regime_signal.reindex(close.index)
    if signal.dropna().empty:
        raise RuntimeError(f"No regime signal available for {strategy}.")
    price = close.loc[signal.dropna().index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
    accel = (
        accel_mask.reindex(price.index).fillna(False).astype(bool)
        if accel_mask is not None
        else pd.Series(False, index=price.index)
    )

    cost_rate = trading_cost_bps / 10_000.0
    exposure_equity = float(initial_capital)
    reserve_cash = 0.0
    transaction_costs = 0.0
    borrow_costs = 0.0
    reserve_contributions = 0.0
    reserve_deployments = 0.0
    reserve_deploy_count = 0
    risk_on_entries = 0
    risk_on_leverage_cuts = 0
    risk_off_entries = 0
    accelerated_contribution_events = 0
    accelerated_extra_contribution = 0.0
    events: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(price.index[0], -float(initial_capital))]

    first_regime = str(signal.iloc[0])
    current_leverage = leverage_for_regime(first_regime, risk_on_leverage, neutral_leverage)
    if current_leverage > 0.0:
        initial_cost = exposure_equity * current_leverage * cost_rate
        exposure_equity = max(0.0, exposure_equity - initial_cost)
        transaction_costs += initial_cost
    else:
        reserve_cash = exposure_equity
        exposure_equity = 0.0

    previous_price = float(price.iloc[0])
    previous_regime = first_regime
    previous_period = leverage.contribution_bucket(0, price.index[0], contribution_frequency, trading_day_interval)

    for i, (date, px) in enumerate(price.items()):
        px = float(px)
        regime = str(signal.loc[date])

        if i > 0 and exposure_equity > 0.0 and current_leverage > 0.0:
            daily_borrow = max(current_leverage - 1.0, 0.0) * borrow_rate / TRADING_DAYS_PER_YEAR
            borrow_costs += exposure_equity * daily_borrow
            day_return = px / previous_price - 1.0
            exposure_equity = max(0.0, exposure_equity * (1.0 + current_leverage * day_return - daily_borrow))

        account_before_actions = exposure_equity + reserve_cash
        period = leverage.contribution_bucket(i, date, contribution_frequency, trading_day_interval)
        new_leverage = leverage_for_regime(regime, risk_on_leverage, neutral_leverage)

        if i > 0 and period != previous_period:
            contribution = float(periodic_contribution)
            if regime == "risk_on" and bool(accel.loc[date]):
                accelerated_extra_contribution += contribution * max(accel_multiplier - 1.0, 0.0)
                contribution *= float(accel_multiplier)
                accelerated_contribution_events += 1
                leverage.record_event(
                    events,
                    date=date,
                    strategy=strategy,
                    event="accelerated_contribution",
                    regime=regime,
                    value_before=account_before_actions,
                    reserve_cash=reserve_cash,
                    amount=contribution,
                )
            cashflows.append((date, -contribution))
            if regime == "risk_off":
                reserve_cash += contribution
                reserve_contributions += contribution
            else:
                contribution_cost = contribution * new_leverage * cost_rate
                transaction_costs += contribution_cost
                exposure_equity += max(0.0, contribution - contribution_cost)
            previous_period = period

        if regime != previous_regime:
            if regime == "risk_off":
                risk_off_entries += 1
                leverage.record_event(
                    events,
                    date=date,
                    strategy=strategy,
                    event="enter_risk_off",
                    regime=regime,
                    value_before=account_before_actions,
                    reserve_cash=reserve_cash,
                    amount=None,
                )
            if previous_regime == "risk_on" and regime != "risk_on":
                risk_on_leverage_cuts += 1
                leverage.record_event(
                    events,
                    date=date,
                    strategy=strategy,
                    event="cut_risk_on_leverage",
                    regime=regime,
                    value_before=account_before_actions,
                    reserve_cash=reserve_cash,
                    amount=None,
                )
            if regime == "risk_on" and previous_regime != "risk_on":
                risk_on_entries += 1
                leverage.record_event(
                    events,
                    date=date,
                    strategy=strategy,
                    event=f"enter_risk_on_{int(risk_on_leverage)}x",
                    regime=regime,
                    value_before=account_before_actions,
                    reserve_cash=reserve_cash,
                    amount=None,
                )

        if new_leverage == 0.0 and exposure_equity > 0.0:
            liquidation_cost = current_leverage * exposure_equity * cost_rate
            transaction_costs += liquidation_cost
            exposure_equity = max(0.0, exposure_equity - liquidation_cost)
            reserve_cash += exposure_equity
            exposure_equity = 0.0
            leverage.record_event(
                events,
                date=date,
                strategy=strategy,
                event="liquidate_to_cash",
                regime=regime,
                value_before=account_before_actions,
                reserve_cash=reserve_cash,
                amount=None,
            )
        elif current_leverage == 0.0 and new_leverage > 0.0 and reserve_cash > 0.0:
            deploy_amount = reserve_cash
            deploy_cost = deploy_amount * new_leverage * cost_rate
            transaction_costs += deploy_cost
            exposure_equity += max(0.0, deploy_amount - deploy_cost)
            reserve_cash = 0.0
            reserve_deployments += deploy_amount
            reserve_deploy_count += 1
            leverage.record_event(
                events,
                date=date,
                strategy=strategy,
                event="deploy_reserve",
                regime=regime,
                value_before=account_before_actions,
                reserve_cash=reserve_cash,
                amount=deploy_amount,
            )
        elif i > 0 and exposure_equity > 0.0 and current_leverage > 0.0 and not np.isclose(new_leverage, current_leverage):
            rebalance_cost = abs(new_leverage - current_leverage) * exposure_equity * cost_rate
            transaction_costs += rebalance_cost
            exposure_equity = max(0.0, exposure_equity - rebalance_cost)

        total_value = exposure_equity + reserve_cash
        curves.append(
            {
                "date": date,
                "strategy": strategy,
                "total_value": total_value,
                "exposure_equity": exposure_equity,
                "reserve_cash": reserve_cash,
                "target_leverage": new_leverage,
                "regime_signal": regime,
                "accelerated_dca": bool(accel.loc[date]) and regime == "risk_on",
            }
        )

        previous_price = px
        previous_regime = regime
        current_leverage = new_leverage

    curve_df = pd.DataFrame(curves).set_index("date")
    month_signals = curve_df["regime_signal"].groupby(curve_df.index.to_period("M")).first()
    if accelerated_contribution_events > 0:
        curve_df.attrs["accelerated_contribution_events"] = accelerated_contribution_events
        curve_df.attrs["accelerated_extra_contribution"] = accelerated_extra_contribution
    return leverage.StrategyResult(
        strategy=strategy,
        curves=curve_df,
        events=pd.DataFrame(events),
        cashflows=cashflows,
        final_exposure_equity=float(curve_df["exposure_equity"].iloc[-1]),
        final_reserve_cash=float(curve_df["reserve_cash"].iloc[-1]),
        avg_target_leverage=float(curve_df["target_leverage"].mean()),
        total_external_contributed=leverage.total_external_from_cashflows(cashflows),
        reserve_contributions=float(reserve_contributions),
        reserve_deployments=float(reserve_deployments),
        reserve_deploy_count=int(reserve_deploy_count),
        risk_on_entries=int(risk_on_entries),
        risk_on_leverage_cuts=int(risk_on_leverage_cuts),
        risk_off_entries=int(risk_off_entries),
        transaction_costs=float(transaction_costs),
        borrow_costs=float(borrow_costs),
        risk_on_days=int((curve_df["regime_signal"] == "risk_on").sum()),
        neutral_days=int((curve_df["regime_signal"] == "neutral").sum()),
        risk_off_days=int((curve_df["regime_signal"] == "risk_off").sum()),
        risk_on_months=int((month_signals == "risk_on").sum()),
        neutral_months=int((month_signals == "neutral").sum()),
        risk_off_months=int((month_signals == "risk_off").sum()),
    )


def metrics_with_accel(window: str, result: leverage.StrategyResult, plain_final: float | None) -> dict[str, object]:
    row = leverage.metrics_row(window, result, plain_final)
    row["accelerated_contribution_events"] = int(result.curves.attrs.get("accelerated_contribution_events", 0))
    row["accelerated_extra_contribution"] = float(result.curves.attrs.get("accelerated_extra_contribution", 0.0))
    return row


def append_window(
    metric_frames: list[pd.DataFrame],
    curve_frames: list[pd.DataFrame],
    event_frames: list[pd.DataFrame],
    *,
    window: str,
    results: list[leverage.StrategyResult],
    plain_final: float,
) -> None:
    for result in results:
        metric_frames.append(pd.DataFrame([metrics_with_accel(window, result, plain_final)]))
        curve = result.curves.reset_index().copy()
        curve["window"] = window
        curve_frames.append(curve)
        if not result.events.empty:
            events = result.events.copy()
            events["window"] = window
            event_frames.append(events)


def plot_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    color_map = {
        "plain_dca_1x": "#1f77b4",
        "spy_gate_qqq_ensemble_blend_2x": "#0f766e",
        "spy_gate_full_cash_3x": "#d97706",
        "spy_gate_full_cash_3x_vix_spike_dca2x": "#b91c1c",
    }
    for window in sorted(curves["window"].unique()):
        subset = curves[curves["window"] == window].copy().sort_values("date")
        if subset.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for strategy, group in subset.groupby("strategy", sort=False):
            equity = group["total_value"].astype(float)
            dates = pd.to_datetime(group["date"])
            drawdown = equity / equity.cummax() - 1.0
            color = color_map.get(strategy, "#444444")
            axes[0].plot(dates, equity, label=strategy, color=color, linewidth=1.8)
            axes[1].plot(dates, drawdown, label=strategy, color=color, linewidth=1.8)
        axes[0].set_yscale("log")
        axes[0].set_title(f"Aggressive SPY-gated QQQ variants: {window}")
        axes[0].set_ylabel("Account value")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="upper left")
        axes[1].set_title(f"Drawdowns: {window}")
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / f"aggressive_variants_equity_drawdown_{window}.png", dpi=160)
        plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    qqq_drop_threshold: float,
    accel_multiplier: float,
) -> None:
    full = metrics[metrics["window"] == "full_common_window"].copy().sort_values("final_value", ascending=False)
    sub = metrics[metrics["window"] != "full_common_window"].copy()
    lines = [
        "# Aggressive SPY-Gated QQQ Variants",
        "",
        "## Variants",
        "",
        "- `spy_gate_qqq_ensemble_blend_2x`: current combined policy benchmark.",
        "- `spy_gate_full_cash_3x`: `risk_off -> full cash`, `neutral -> 1x QQQ`, `risk_on -> 3x QQQ`.",
        (
            "- `spy_gate_full_cash_3x_vix_spike_dca2x`: same as above, but on monthly contribution dates "
            f"inside `risk_on` plus lagged VIX spike plus lagged QQQ 21-day return <= `{qqq_drop_threshold:.0%}`, "
            f"the contribution is multiplied by `{accel_multiplier:.1f}x`."
        ),
        "- VIX spike uses the existing lagged volatility-shock definition: `VIX >= 25` or `VIX 21d change >= 5`.",
        "",
        "## Full-Window Metrics",
        "",
        full[
            [
                "strategy",
                "final_value",
                "xirr",
                "time_weighted_cagr",
                "max_drawdown",
                "avg_target_leverage",
                "risk_on_months",
                "neutral_months",
                "risk_off_months",
                "accelerated_contribution_events",
                "accelerated_extra_contribution",
                "final_delta_vs_plain_dca",
            ]
        ].to_markdown(index=False),
        "",
        "## Subwindow Metrics",
        "",
        sub[
            [
                "window",
                "strategy",
                "final_value",
                "xirr",
                "time_weighted_cagr",
                "max_drawdown",
                "avg_target_leverage",
                "accelerated_contribution_events",
                "accelerated_extra_contribution",
                "final_delta_vs_plain_dca",
            ]
        ].to_markdown(index=False),
    ]
    (out_dir / "aggressive_variant_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.qqq_analysis_dir)
    qqq_close = pd.to_numeric(dataset["qqq_close"], errors="coerce").dropna().rename("qqq_close")
    qqq_signal = combo.load_daily_signal(args.qqq_compare_dir, "ensemble_blend_signal_lag1")
    spy_signal = combo.load_daily_signal(args.spy_compare_dir, "ensemble_blend_signal_lag1")
    policy_signals = combo.build_policy_signal(spy_signal, qqq_signal)

    signal_start = max(qqq_close.index.min(), policy_signals.index.min())
    qqq_close = qqq_close.loc[qqq_close.index >= signal_start].copy()
    policy_signals = policy_signals.loc[policy_signals.index >= signal_start].copy()
    accel_mask = build_acceleration_mask(dataset.loc[qqq_close.index], policy_signals["policy_signal"], args.qqq_drop_threshold)

    metric_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for window_name, start, end in WINDOWS:
        start_ts = pd.Timestamp(start) if start else qqq_close.index.min()
        end_ts = pd.Timestamp(end) if end else qqq_close.index.max()
        price = qqq_close.loc[(qqq_close.index >= start_ts) & (qqq_close.index <= end_ts)].copy()
        if price.empty:
            continue
        signal_slice = policy_signals["policy_signal"].reindex(price.index)
        accel_slice = accel_mask.reindex(price.index).fillna(False)

        plain = leverage.simulate_plain_dca(
            price,
            strategy="plain_dca_1x",
            initial_capital=args.initial_capital,
            periodic_contribution=args.monthly_contribution,
            contribution_frequency="monthly",
            trading_day_interval=3,
            trading_cost_bps=args.trading_cost_bps,
        )
        combined_2x = leverage.simulate_regime_leverage(
            price,
            signal_slice,
            strategy="spy_gate_qqq_ensemble_blend_2x",
            risk_on_leverage=2.0,
            initial_capital=args.initial_capital,
            periodic_contribution=args.monthly_contribution,
            contribution_frequency="monthly",
            trading_day_interval=3,
            trading_cost_bps=args.trading_cost_bps,
            borrow_rate=args.borrow_rate,
        )
        full_cash_3x = simulate_full_cash_regime(
            price,
            signal_slice,
            strategy="spy_gate_full_cash_3x",
            risk_on_leverage=args.risk_on_leverage,
            neutral_leverage=args.neutral_leverage,
            initial_capital=args.initial_capital,
            periodic_contribution=args.monthly_contribution,
            contribution_frequency="monthly",
            trading_day_interval=3,
            trading_cost_bps=args.trading_cost_bps,
            borrow_rate=args.borrow_rate,
        )
        full_cash_3x_accel = simulate_full_cash_regime(
            price,
            signal_slice,
            strategy="spy_gate_full_cash_3x_vix_spike_dca2x",
            risk_on_leverage=args.risk_on_leverage,
            neutral_leverage=args.neutral_leverage,
            initial_capital=args.initial_capital,
            periodic_contribution=args.monthly_contribution,
            contribution_frequency="monthly",
            trading_day_interval=3,
            trading_cost_bps=args.trading_cost_bps,
            borrow_rate=args.borrow_rate,
            accel_mask=accel_slice,
            accel_multiplier=args.accel_multiplier,
        )

        plain_final = float(plain.curves["total_value"].iloc[-1])
        append_window(
            metric_frames,
            curve_frames,
            event_frames,
            window=window_name,
            results=[plain, combined_2x, full_cash_3x, full_cash_3x_accel],
            plain_final=plain_final,
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    metrics.to_csv(args.out_dir / "aggressive_variant_metrics.csv", index=False)
    curves.to_csv(args.out_dir / "aggressive_variant_curves.csv", index=False)
    if not events.empty:
        events.to_csv(args.out_dir / "aggressive_variant_events.csv", index=False)

    plot_curves(curves, args.out_dir)
    write_report(
        args.out_dir,
        metrics=metrics,
        qqq_drop_threshold=args.qqq_drop_threshold,
        accel_multiplier=args.accel_multiplier,
    )

    summary = metrics[metrics["window"] == "full_common_window"][
        [
            "strategy",
            "final_value",
            "xirr",
            "time_weighted_cagr",
            "max_drawdown",
            "avg_target_leverage",
            "accelerated_contribution_events",
            "accelerated_extra_contribution",
        ]
    ].copy()
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
