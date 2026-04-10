"""Backtest a SPY-gated QQQ ensemble leverage policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import qqq_macro_ml_regime_analysis as regime_analysis
import qqq_macro_walkforward_leverage as leverage


ROOT = regime_analysis.ROOT
DEFAULT_QQQ_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis_20260409_feature_refresh"
DEFAULT_QQQ_COMPARE_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260410_ensemble_refresh"
DEFAULT_SPY_COMPARE_DIR = ROOT / "reports" / "sp500_macro_regime_with_qqq_20260410" / "compare"
DEFAULT_OUT_DIR = ROOT / "reports" / "spy_gate_qqq_ensemble_backtest_20260410"
COMPARABLE_START = pd.Timestamp("2007-05-31")
WINDOWS = [
    ("full_common_window", None, None),
    ("pre_covid_2014_2019", "2014-11-03", "2019-12-31"),
    ("covid_recovery_2020_2021", "2020-01-02", "2021-12-31"),
    ("inflation_ai_2022_2026", "2022-01-03", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qqq-analysis-dir", type=Path, default=DEFAULT_QQQ_ANALYSIS_DIR)
    parser.add_argument("--qqq-compare-dir", type=Path, default=DEFAULT_QQQ_COMPARE_DIR)
    parser.add_argument("--spy-compare-dir", type=Path, default=DEFAULT_SPY_COMPARE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=100.0)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument("--risk-on-leverage", type=float, default=2.0)
    return parser.parse_args()


def load_csv(path: Path, *, index_col: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[index_col] if index_col else None)
    if index_col is not None:
        frame = frame.set_index(index_col).sort_index()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def load_qqq_close(analysis_dir: Path) -> pd.Series:
    dataset_path = analysis_dir / "aligned_daily_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing QQQ analysis dataset: {dataset_path}")
    dataset = load_csv(dataset_path, index_col="date")
    if "qqq_close" not in dataset.columns:
        raise KeyError(f"`qqq_close` column not found in {dataset_path}")
    close = pd.to_numeric(dataset["qqq_close"], errors="coerce").dropna().rename("qqq_close")
    return close


def load_indicator_frame(analysis_dir: Path) -> pd.DataFrame:
    dataset_path = analysis_dir / "aligned_daily_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing QQQ analysis dataset: {dataset_path}")
    dataset = load_csv(dataset_path, index_col="date")
    required = ["vix_level", "vix_21d_change", "qqq_21d_return"]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise KeyError(f"Missing required indicator columns in {dataset_path}: {missing}")
    return dataset[required].apply(pd.to_numeric, errors="coerce")


def load_daily_signal(compare_dir: Path, column: str) -> pd.Series:
    signal_path = compare_dir / "walkforward_model_signals_daily.csv"
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing signal export: {signal_path}")
    signals = load_csv(signal_path, index_col="date")
    if column not in signals.columns:
        raise KeyError(f"Missing `{column}` in {signal_path}")
    out = signals[column].astype("object")
    out.name = column
    return out


def build_policy_signal(spy_signal: pd.Series, qqq_signal: pd.Series) -> pd.DataFrame:
    joined = pd.concat(
        [
            spy_signal.rename("spy_gate_signal"),
            qqq_signal.rename("qqq_expression_signal"),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    joined = joined.dropna(subset=["spy_gate_signal", "qqq_expression_signal"]).copy()
    policy = pd.Series(index=joined.index, dtype=object, name="policy_signal")

    policy.loc[joined["spy_gate_signal"].eq("risk_off")] = "risk_off"
    allow_risk_on = joined["spy_gate_signal"].eq("risk_on") & joined["qqq_expression_signal"].ne("risk_off")
    policy.loc[allow_risk_on] = "risk_on"
    policy.loc[policy.isna()] = "neutral"
    joined["policy_signal"] = policy
    return joined


def append_window(
    metric_frames: list[pd.DataFrame],
    curve_frames: list[pd.DataFrame],
    event_frames: list[pd.DataFrame],
    *,
    window: str,
    results: list[leverage.StrategyResult],
    plain_final: float | None,
) -> None:
    for result in results:
        metric_row = leverage.metrics_row(window, result, plain_final)
        metric_frames.append(pd.DataFrame([metric_row]))

        curve = result.curves.reset_index().copy()
        curve["window"] = window
        curve_frames.append(curve)

        events = result.events.copy()
        if not events.empty:
            events["window"] = window
            event_frames.append(events)


def build_vix_accel_multiplier(
    indicators: pd.DataFrame,
    policy_signal: pd.Series,
    *,
    vix_level_threshold: float = 20.0,
) -> pd.Series:
    aligned = indicators.reindex(policy_signal.index).copy()
    lagged = aligned.shift(1)
    multiplier = pd.Series(1.0, index=policy_signal.index, name="contribution_multiplier")
    trigger = (
        policy_signal.eq("risk_on")
        & lagged["vix_level"].ge(vix_level_threshold)
        & lagged["vix_21d_change"].gt(0.0)
        & lagged["qqq_21d_return"].lt(0.0)
    )
    multiplier.loc[trigger] = 2.0
    return multiplier


def simulate_regime_leverage_dynamic_contribution(
    close: pd.Series,
    regime_signal: pd.Series,
    contribution_multiplier: pd.Series,
    *,
    strategy: str,
    risk_on_leverage: float,
    initial_capital: float,
    periodic_contribution: float,
    contribution_frequency: str,
    trading_day_interval: int,
    trading_cost_bps: float,
    borrow_rate: float,
) -> leverage.StrategyResult:
    signal = regime_signal.reindex(close.index)
    if signal.dropna().empty:
        raise RuntimeError("No lagged walk-forward regime signal available for leverage simulation.")
    price = close.loc[signal.dropna().index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
    multiplier = contribution_multiplier.reindex(price.index).fillna(1.0).astype(float)
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
    events: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(price.index[0], -float(initial_capital))]

    first_regime = str(signal.iloc[0])
    current_leverage = leverage.target_leverage_for_regime(first_regime, risk_on_leverage)
    initial_cost = exposure_equity * current_leverage * cost_rate
    exposure_equity = max(0.0, exposure_equity - initial_cost)
    transaction_costs += initial_cost

    previous_price = float(price.iloc[0])
    previous_regime = first_regime
    previous_period = leverage.contribution_bucket(0, price.index[0], contribution_frequency, trading_day_interval)

    for i, (date, px) in enumerate(price.items()):
        px = float(px)
        regime = str(signal.loc[date])
        if i > 0 and exposure_equity > 0.0:
            equity_before = exposure_equity
            daily_borrow = max(current_leverage - 1.0, 0.0) * borrow_rate / 252.0
            borrow_costs += equity_before * daily_borrow
            day_return = px / previous_price - 1.0
            exposure_equity = max(0.0, equity_before * (1.0 + current_leverage * day_return - daily_borrow))

        account_before_actions = exposure_equity + reserve_cash
        period = leverage.contribution_bucket(i, date, contribution_frequency, trading_day_interval)
        if i > 0 and period != previous_period:
            contribution = float(periodic_contribution)
            if regime == "risk_on":
                contribution *= float(multiplier.loc[date])
            cashflows.append((date, -contribution))
            if regime == "risk_on" and multiplier.loc[date] > 1.0:
                leverage.record_event(
                    events,
                    date=date,
                    strategy=strategy,
                    event=f"accelerated_dca_{multiplier.loc[date]:.1f}x",
                    regime=regime,
                    value_before=account_before_actions,
                    reserve_cash=reserve_cash,
                    amount=contribution,
                )
            if regime == "risk_off":
                reserve_cash += contribution
                reserve_contributions += contribution
            else:
                contribution_cost = contribution * leverage.target_leverage_for_regime(regime, risk_on_leverage) * cost_rate
                transaction_costs += contribution_cost
                exposure_equity += max(0.0, contribution - contribution_cost)
            previous_period = period

        new_leverage = leverage.target_leverage_for_regime(regime, risk_on_leverage)
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

        if regime != "risk_off" and reserve_cash > 0.0:
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

        if i > 0 and exposure_equity > 0.0 and not np.isclose(new_leverage, current_leverage):
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
            }
        )
        previous_price = px
        previous_regime = regime
        current_leverage = new_leverage

    curve_df = pd.DataFrame(curves).set_index("date")
    month_signals = curve_df["regime_signal"].groupby(curve_df.index.to_period("M")).first()
    total_external = leverage.total_external_from_cashflows(cashflows)
    return leverage.StrategyResult(
        strategy=strategy,
        curves=curve_df,
        events=pd.DataFrame(events),
        cashflows=cashflows,
        final_exposure_equity=float(curve_df["exposure_equity"].iloc[-1]),
        final_reserve_cash=float(curve_df["reserve_cash"].iloc[-1]),
        avg_target_leverage=float(curve_df["target_leverage"].mean()),
        total_external_contributed=total_external,
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


def simulate_window(
    window: str,
    close: pd.Series,
    qqq_signal: pd.Series,
    policy_signal: pd.Series,
    accel_multiplier: pd.Series,
    *,
    start: str | None,
    end: str | None,
    initial_capital: float,
    monthly_contribution: float,
    risk_on_leverage: float,
    trading_cost_bps: float,
    borrow_rate: float,
) -> list[leverage.StrategyResult]:
    start_ts = pd.Timestamp(start) if start else close.index.min()
    end_ts = pd.Timestamp(end) if end else close.index.max()
    price = close.loc[(close.index >= start_ts) & (close.index <= end_ts)].copy()
    if price.empty:
        return []

    qqq_slice = qqq_signal.reindex(price.index)
    policy_slice = policy_signal.reindex(price.index)
    accel_slice = accel_multiplier.reindex(price.index)

    plain = leverage.simulate_plain_dca(
        price,
        strategy="plain_dca_1x",
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
    )
    qqq_blend = leverage.simulate_regime_leverage(
        price,
        qqq_slice,
        strategy="qqq_ensemble_blend_2x",
        risk_on_leverage=2.0,
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
        borrow_rate=borrow_rate,
    )
    gated = leverage.simulate_regime_leverage(
        price,
        policy_slice,
        strategy="spy_gate_qqq_ensemble_blend_2x",
        risk_on_leverage=2.0,
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
        borrow_rate=borrow_rate,
    )
    qqq_blend_3x = leverage.simulate_regime_leverage(
        price,
        qqq_slice,
        strategy="qqq_ensemble_blend_3x",
        risk_on_leverage=3.0,
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
        borrow_rate=borrow_rate,
    )
    gated_3x = leverage.simulate_regime_leverage(
        price,
        policy_slice,
        strategy="spy_gate_qqq_ensemble_blend_3x",
        risk_on_leverage=3.0,
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
        borrow_rate=borrow_rate,
    )
    gated_3x_accel = simulate_regime_leverage_dynamic_contribution(
        price,
        policy_slice,
        accel_slice,
        strategy="spy_gate_qqq_ensemble_blend_3x_vix_accel",
        risk_on_leverage=3.0,
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
        borrow_rate=borrow_rate,
    )
    return [plain, qqq_blend, gated, qqq_blend_3x, gated_3x, gated_3x_accel]


def plot_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    color_map = {
        "plain_dca_1x": "#1f77b4",
        "qqq_ensemble_blend_2x": "#0f766e",
        "spy_gate_qqq_ensemble_blend_2x": "#d97706",
        "qqq_ensemble_blend_3x": "#115e59",
        "spy_gate_qqq_ensemble_blend_3x": "#b45309",
        "spy_gate_qqq_ensemble_blend_3x_vix_accel": "#7c3aed",
    }

    for window in sorted(curves["window"].unique()):
        subset = curves[curves["window"] == window].copy().sort_values("date")
        if subset.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for strategy, group in subset.groupby("strategy", sort=False):
            series = group["total_value"].astype(float)
            dates = pd.to_datetime(group["date"])
            drawdown = series / series.cummax() - 1.0
            color = color_map.get(strategy, "#444444")
            axes[0].plot(dates, series, label=strategy, color=color, linewidth=1.8)
            axes[1].plot(dates, drawdown, label=strategy, color=color, linewidth=1.8)

        axes[0].set_yscale("log")
        axes[0].set_title(f"SPY-gated QQQ ensemble policy: {window}")
        axes[0].set_ylabel("Account value")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="upper left")

        axes[1].set_title(f"Drawdowns: {window}")
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="lower left")

        fig.tight_layout()
        fig.savefig(out_dir / f"spy_gate_qqq_equity_drawdown_{window}.png", dpi=160)
        plt.close(fig)


def monthly_state_summary(curves: pd.DataFrame, policy_signals: pd.DataFrame) -> pd.DataFrame:
    policy_monthly = (
        policy_signals["policy_signal"].groupby(policy_signals.index.to_period("M")).last().rename("policy_signal")
    )
    spy_monthly = (
        policy_signals["spy_gate_signal"].groupby(policy_signals.index.to_period("M")).last().rename("spy_gate_signal")
    )
    qqq_monthly = (
        policy_signals["qqq_expression_signal"]
        .groupby(policy_signals.index.to_period("M"))
        .last()
        .rename("qqq_expression_signal")
    )

    leverage_monthly = curves[curves["strategy"] == "spy_gate_qqq_ensemble_blend_2x"].copy()
    leverage_monthly["date"] = pd.to_datetime(leverage_monthly["date"])
    leverage_monthly = leverage_monthly.groupby(leverage_monthly["date"].dt.to_period("M")).tail(1)
    leverage_monthly = leverage_monthly.set_index(pd.to_datetime(leverage_monthly["date"]).dt.to_period("M"))

    summary = pd.concat([spy_monthly, qqq_monthly, policy_monthly], axis=1)
    if not leverage_monthly.empty:
        summary["target_leverage"] = pd.to_numeric(leverage_monthly["target_leverage"], errors="coerce")
        summary["total_value"] = pd.to_numeric(leverage_monthly["total_value"], errors="coerce")
        summary["reserve_cash"] = pd.to_numeric(leverage_monthly["reserve_cash"], errors="coerce")
    summary.index = summary.index.astype(str)
    summary.index.name = "month"
    return summary.reset_index()


def write_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    policy_signals: pd.DataFrame,
    accel_multiplier: pd.Series,
    start_date: pd.Timestamp,
    qqq_signal_start: pd.Timestamp,
    spy_signal_start: pd.Timestamp,
    qqq_analysis_dir: Path,
    qqq_compare_dir: Path,
    spy_compare_dir: Path,
) -> None:
    full = metrics[metrics["window"] == "full_common_window"].copy().sort_values("final_value", ascending=False)
    subwindows = metrics[metrics["window"] != "full_common_window"].copy()
    policy_row = full[full["strategy"] == "spy_gate_qqq_ensemble_blend_2x"].iloc[0]
    qqq_row = full[full["strategy"] == "qqq_ensemble_blend_2x"].iloc[0]
    plain_row = full[full["strategy"] == "plain_dca_1x"].iloc[0]
    policy_row_3x = full[full["strategy"] == "spy_gate_qqq_ensemble_blend_3x"].iloc[0]
    accel_row_3x = full[full["strategy"] == "spy_gate_qqq_ensemble_blend_3x_vix_accel"].iloc[0]

    policy_counts = policy_signals["policy_signal"].value_counts(dropna=False)
    accel_months = int((accel_multiplier > 1.0).groupby(accel_multiplier.index.to_period("M")).any().sum())
    accel_days = int((accel_multiplier > 1.0).sum())
    lines = [
        "# SPY Gate / QQQ Ensemble Backtest",
        "",
        "## Policy",
        "",
        "- Broad market gate: `SPY ensemble_blend` lagged daily signal.",
        "- Beta expression: `QQQ ensemble_blend` lagged daily signal.",
        "- Combined rule:",
        "  - `risk_off` when SPY gate is `risk_off`.",
        "  - `risk_on` only when SPY gate is `risk_on` and QQQ signal is not `risk_off`.",
        "  - `neutral` otherwise.",
        "- Leverage rule: `risk_on => 2x`, `neutral => 1x`, `risk_off => no leverage + reserve cash for new contributions`.",
        "- Additional tested variant: `risk_on => 3x`, `neutral => 1x`, `risk_off => 1x + reserve cash`.",
        "- Accelerated DCA trigger for the 3x variant: on the contribution date, double the monthly DCA if the lagged policy is `risk_on`, lagged `VIX >= 20`, lagged `VIX 21d change > 0`, and lagged `QQQ 21d return < 0`.",
        "",
        "## Scope",
        "",
        f"- QQQ price source: `{qqq_analysis_dir / 'aligned_daily_dataset.csv'}`",
        f"- QQQ signal source: `{qqq_compare_dir / 'walkforward_model_signals_daily.csv'}`",
        f"- SPY signal source: `{spy_compare_dir / 'walkforward_model_signals_daily.csv'}`",
        f"- QQQ signal first valid date: `{qqq_signal_start.date()}`",
        f"- SPY signal first valid date: `{spy_signal_start.date()}`",
        f"- Common backtest start: `{start_date.date()}`",
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
                "final_delta_vs_plain_dca",
            ]
        ].to_markdown(index=False),
        "",
        "## Subwindow Metrics",
        "",
        subwindows[
            [
                "window",
                "strategy",
                "final_value",
                "xirr",
                "time_weighted_cagr",
                "max_drawdown",
                "avg_target_leverage",
                "risk_on_months",
                "neutral_months",
                "risk_off_months",
                "final_delta_vs_plain_dca",
            ]
        ].to_markdown(index=False),
        "",
        "## Read",
        "",
        (
            f"- Combined policy finished at `${policy_row['final_value']:,.0f}` vs "
            f"`$ {plain_row['final_value']:,.0f}` plain DCA and `${qqq_row['final_value']:,.0f}` standalone QQQ blend 2x."
        ).replace("$ ", "$"),
        (
            f"- Combined policy XIRR was `{policy_row['xirr']:.2%}` with max drawdown `{policy_row['max_drawdown']:.2%}`."
        ),
        (
            f"- 3x gated policy finished at `${policy_row_3x['final_value']:,.0f}` with XIRR `{policy_row_3x['xirr']:.2%}` and max drawdown `{policy_row_3x['max_drawdown']:.2%}`."
        ),
        (
            f"- 3x gated policy with accelerated DCA finished at `${accel_row_3x['final_value']:,.0f}` with XIRR `{accel_row_3x['xirr']:.2%}` and max drawdown `{accel_row_3x['max_drawdown']:.2%}`."
        ),
        (
            f"- The accelerated-DCA trigger was active on `{accel_days}` daily rows spanning `{accel_months}` contribution months."
        ),
        (
            f"- Compared with standalone QQQ blend 2x, the gate changed the path by "
            f"`{policy_row['final_value'] - qqq_row['final_value']:+,.0f}` of final value."
        ),
        (
            f"- Policy daily state counts: risk_on `{int(policy_counts.get('risk_on', 0))}`, "
            f"neutral `{int(policy_counts.get('neutral', 0))}`, risk_off `{int(policy_counts.get('risk_off', 0))}`."
        ),
        "",
        "## Assumptions",
        "",
        "- Neutral means standard 1x long exposure, not partial cash.",
        "- Risk-off keeps existing exposure unlevered and parks new contributions in reserve until the next non-risk-off regime.",
        "- The backtest uses the existing lagged-signal engine, trading costs, and borrow cost assumptions from the QQQ walk-forward scripts.",
    ]

    (out_dir / "spy_gate_qqq_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qqq_close = load_qqq_close(args.qqq_analysis_dir)
    indicators = load_indicator_frame(args.qqq_analysis_dir)
    qqq_signal = load_daily_signal(args.qqq_compare_dir, "ensemble_blend_signal_lag1")
    spy_signal = load_daily_signal(args.spy_compare_dir, "ensemble_blend_signal_lag1")

    policy_signals = build_policy_signal(spy_signal, qqq_signal)
    if policy_signals.empty:
        raise RuntimeError("No overlapping SPY and QQQ daily signals were found.")

    signal_start = max(
        qqq_close.index.min(),
        policy_signals.index.min(),
    )
    qqq_close = qqq_close.loc[qqq_close.index >= signal_start].copy()
    indicators = indicators.loc[indicators.index >= signal_start].copy()
    policy_signals = policy_signals.loc[policy_signals.index >= signal_start].copy()
    qqq_signal = qqq_signal.loc[qqq_signal.index >= signal_start].copy()
    accel_multiplier = build_vix_accel_multiplier(indicators, policy_signals["policy_signal"])

    metric_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for window_name, start, end in WINDOWS:
        results = simulate_window(
            window_name,
            qqq_close,
            qqq_signal,
            policy_signals["policy_signal"],
            accel_multiplier,
            start=start,
            end=end,
            initial_capital=args.initial_capital,
            monthly_contribution=args.monthly_contribution,
            risk_on_leverage=args.risk_on_leverage,
            trading_cost_bps=args.trading_cost_bps,
            borrow_rate=args.borrow_rate,
        )
        if not results:
            continue
        plain_final = float(results[0].curves["total_value"].iloc[-1])
        append_window(
            metric_frames,
            curve_frames,
            event_frames,
            window=window_name,
            results=results,
            plain_final=plain_final,
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    metrics.to_csv(args.out_dir / "combined_policy_metrics.csv", index=False)
    curves.to_csv(args.out_dir / "combined_policy_curves.csv", index=False)
    if not events.empty:
        events.to_csv(args.out_dir / "combined_policy_events.csv", index=False)

    policy_signals.to_csv(args.out_dir / "combined_policy_daily_signals.csv", index_label="date")
    accel_multiplier.rename("contribution_multiplier").to_csv(
        args.out_dir / "combined_policy_accelerated_dca_multiplier.csv",
        index_label="date",
    )
    monthly_state_summary(curves, policy_signals).to_csv(args.out_dir / "combined_policy_monthly_states.csv", index=False)

    plot_curves(curves, args.out_dir)
    write_report(
        args.out_dir,
        metrics=metrics,
        policy_signals=policy_signals,
        accel_multiplier=accel_multiplier,
        start_date=signal_start,
        qqq_signal_start=qqq_signal.dropna().index.min(),
        spy_signal_start=spy_signal.dropna().index.min(),
        qqq_analysis_dir=args.qqq_analysis_dir,
        qqq_compare_dir=args.qqq_compare_dir,
        spy_compare_dir=args.spy_compare_dir,
    )

    summary = metrics[
        metrics["window"].eq("full_common_window")
    ][["strategy", "final_value", "xirr", "time_weighted_cagr", "max_drawdown", "avg_target_leverage"]].copy()
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
