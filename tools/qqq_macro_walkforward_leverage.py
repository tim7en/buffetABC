"""Rebuild walk-forward GMM daily regimes and risk-on leverage vs DCA tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import qqq_macro_ml_regime_analysis as regime_analysis


warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
)


ROOT = regime_analysis.ROOT
DEFAULT_ANALYSIS_SCRIPT = ROOT / "tools" / "qqq_macro_ml_regime_analysis.py"
DEFAULT_ANALYSIS_OUT_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_DATASET_PATH = DEFAULT_ANALYSIS_OUT_DIR / "aligned_daily_dataset.csv"
DEFAULT_OUT_DIR = DEFAULT_ANALYSIS_OUT_DIR
COMPARABLE_START = pd.Timestamp("2007-05-31")
PLOT_STYLE = {
    "plain_dca": {"label": "Plain DCA", "color": "#1f77b4"},
    "walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca": {"label": "WF GMM 2x", "color": "#2ca02c"},
    "walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca": {"label": "WF GMM 3x", "color": "#d62728"},
}


@dataclass
class StrategyResult:
    strategy: str
    curves: pd.DataFrame
    events: pd.DataFrame
    cashflows: list[tuple[pd.Timestamp, float]]
    final_exposure_equity: float
    final_reserve_cash: float
    avg_target_leverage: float
    total_external_contributed: float
    reserve_contributions: float | None
    reserve_deployments: float | None
    reserve_deploy_count: int | None
    risk_on_entries: int | None
    risk_on_leverage_cuts: int | None
    risk_off_entries: int | None
    transaction_costs: float | None
    borrow_costs: float | None
    risk_on_days: int | None
    neutral_days: int | None
    risk_off_days: int | None
    risk_on_months: int | None
    neutral_months: int | None
    risk_off_months: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-script", type=Path, default=DEFAULT_ANALYSIS_SCRIPT)
    parser.add_argument("--analysis-out-dir", type=Path, default=DEFAULT_ANALYSIS_OUT_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default="1999-03-10")
    parser.add_argument("--end", default=None)
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument("--refresh-qqq", action="store_true")
    parser.add_argument("--refresh-macro", action="store_true")
    parser.add_argument("--refresh-fred", action="store_true")
    parser.add_argument("--qqq-refresh-start", default=regime_analysis.DEFAULT_QQQ_REFRESH_START)
    parser.add_argument("--macro-refresh-start", default=regime_analysis.DEFAULT_MACRO_REFRESH_START)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=1_000.0)
    parser.add_argument("--weekly-contribution", type=float, default=None)
    parser.add_argument("--contribution-frequency", choices=["monthly", "weekly", "trading_days"], default="monthly")
    parser.add_argument("--trading-day-interval", type=int, default=3)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    return parser.parse_args()


def maybe_run_analysis(args: argparse.Namespace) -> None:
    if not (args.run_analysis or args.refresh_all or args.refresh_qqq or args.refresh_macro or args.refresh_fred):
        return
    if not args.analysis_script.exists():
        raise FileNotFoundError(f"Missing analysis script: {args.analysis_script}")
    command = [
        sys.executable,
        str(args.analysis_script),
        "--out-dir",
        str(args.analysis_out_dir),
        "--start",
        args.start,
        "--qqq-refresh-start",
        args.qqq_refresh_start,
        "--macro-refresh-start",
        args.macro_refresh_start,
    ]
    if args.end:
        command.extend(["--end", args.end])
    if args.refresh_all:
        command.append("--refresh-all")
    if args.refresh_qqq:
        command.append("--refresh-qqq")
    if args.refresh_macro:
        command.append("--refresh-macro")
    if args.refresh_fred:
        command.append("--refresh-fred")
    print("Refreshing / rebuilding aligned dataset before leverage run")
    subprocess.run(command, check=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned dataset: {path}")
    dataset = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    dataset.index = dataset.index.tz_localize(None)
    return dataset


def first_trading_day_per_month(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    order = pd.Series(index, index=index)
    return [pd.Timestamp(day) for day in order.groupby(index.to_period("M")).first().tolist()]


def contribution_bucket(index_position: int, date: pd.Timestamp, frequency: str, trading_day_interval: int) -> Any:
    if frequency == "monthly":
        return date.to_period("M")
    if frequency == "weekly":
        return date.to_period("W-FRI")
    if frequency == "trading_days":
        if trading_day_interval <= 0:
            raise ValueError(f"Trading-day interval must be positive, got {trading_day_interval}")
        return index_position // trading_day_interval
    raise ValueError(f"Unsupported contribution frequency: {frequency}")


def resolve_periodic_contribution(args: argparse.Namespace) -> float:
    if args.contribution_frequency == "weekly":
        if args.weekly_contribution is not None:
            return float(args.weekly_contribution)
        return float(args.monthly_contribution) * 12.0 / 52.0
    if args.contribution_frequency == "trading_days":
        events_per_year = 252.0 / float(args.trading_day_interval)
        return float(args.monthly_contribution) * 12.0 / events_per_year
    return float(args.monthly_contribution)


def total_external_from_cashflows(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    return float(-sum(amount for _, amount in cashflows if amount < 0.0))


def contribution_frequency_label(args: argparse.Namespace) -> str:
    if args.contribution_frequency == "trading_days":
        return f"every_{args.trading_day_interval}_trading_days"
    return args.contribution_frequency


def cluster_mapping(train: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    score_frame = pd.DataFrame({"cluster": labels}, index=train.index)
    score_frame["latent_sentiment_index"] = train["latent_sentiment_index"]
    score_frame["external_shock_score"] = train["external_shock_score"]
    score_frame["qqq_63d_return"] = train["qqq_63d_return"]
    score_frame["vix_level"] = train["vix_level"]
    grouped = score_frame.groupby("cluster")
    cluster_score = (
        grouped["latent_sentiment_index"].mean()
        + grouped["qqq_63d_return"].mean().fillna(0.0)
        - grouped["external_shock_score"].mean().fillna(0.0)
        - regime_analysis._safe_zscore(grouped["vix_level"].mean()).fillna(0.0)
    )
    ordered = list(cluster_score.sort_values().index)
    return {ordered[0]: "risk_off", ordered[1]: "neutral", ordered[2]: "risk_on"}


def build_walkforward_daily_regimes(dataset: pd.DataFrame, min_train_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [
        feature
        for feature in regime_analysis.GMM_FEATURES
        if feature in dataset.columns and dataset[feature].notna().sum() >= min_train_days
    ]
    if len(features) < 4:
        raise RuntimeError(f"Not enough walk-forward GMM features. Found: {features}")

    valid = dataset[features].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(valid) < min_train_days:
        raise RuntimeError(f"Only {len(valid)} fully valid daily rows for walk-forward GMM.")

    walkforward = pd.DataFrame(index=dataset.index)
    refit_rows: list[dict[str, Any]] = []
    refit_dates = first_trading_day_per_month(valid.index)

    for refit_date in refit_dates:
        train = valid.loc[valid.index < refit_date].copy()
        if len(train) < min_train_days:
            continue
        prediction_index = valid.index[(valid.index >= refit_date) & (valid.index.to_period("M") == refit_date.to_period("M"))]
        if len(prediction_index) == 0:
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[features])
        gmm = GaussianMixture(
            n_components=3,
            covariance_type="full",
            init_params="random",
            n_init=5,
            random_state=regime_analysis.RANDOM_STATE,
        )
        train_labels = gmm.fit_predict(x_train)
        mapping = cluster_mapping(dataset.loc[train.index], train_labels)

        x_pred = scaler.transform(valid.loc[prediction_index, features])
        pred_labels = gmm.predict(x_pred)
        pred_probabilities = gmm.predict_proba(x_pred)

        walkforward.loc[prediction_index, "wf_gmm_regime"] = pd.Series(pred_labels, index=prediction_index).map(mapping)
        for cluster, regime in mapping.items():
            walkforward.loc[prediction_index, f"wf_gmm_prob_{regime}"] = pred_probabilities[:, cluster]
        walkforward.loc[prediction_index, "wf_refit_date"] = refit_date
        walkforward.loc[prediction_index, "wf_train_start"] = train.index.min()
        walkforward.loc[prediction_index, "wf_train_end"] = train.index.max()
        walkforward.loc[prediction_index, "wf_train_n"] = len(train)
        refit_rows.append(
            {
                "wf_refit_date": refit_date,
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "train_n": int(len(train)),
                "pred_start": prediction_index.min(),
                "pred_end": prediction_index.max(),
                "pred_n": int(len(prediction_index)),
            }
        )

    out = pd.DataFrame(index=dataset.index)
    out["gmm_regime"] = dataset.get("gmm_regime")
    out = out.join(walkforward)
    out["wf_gmm_regime_signal_lag1"] = out["wf_gmm_regime"].shift(1)
    out.index.name = "date"

    summary_rows = []
    scored = dataset.join(out[["wf_gmm_regime"]], how="left").dropna(subset=["wf_gmm_regime"])
    for regime, group in scored.groupby("wf_gmm_regime"):
        summary_rows.append(
            {
                "regime": regime,
                "count": int(len(group)),
                "avg_forward_63d_return": float(group["qqq_fwd_63d_return"].mean()),
                "median_forward_63d_return": float(group["qqq_fwd_63d_return"].median()),
                "positive_63d_rate": float((group["qqq_fwd_63d_return"] > 0.0).mean()),
                "avg_latent_sentiment": float(group["latent_sentiment_index"].mean()),
                "avg_external_shock_score": float(group["external_shock_score"].mean()),
                "avg_vix": float(group["vix_level"].mean()),
                "avg_qqq_vs_sma200": float(group["qqq_vs_sma200"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("regime")
    refits = pd.DataFrame(refit_rows)
    return out, refits, summary


def target_leverage_for_regime(regime_signal: str | float | None, risk_on_leverage: float) -> float:
    if isinstance(regime_signal, str) and regime_signal == "risk_on":
        return float(risk_on_leverage)
    return 1.0


def record_event(
    rows: list[dict[str, Any]],
    *,
    date: pd.Timestamp,
    strategy: str,
    event: str,
    regime: str,
    value_before: float,
    reserve_cash: float,
    amount: float | None,
) -> None:
    rows.append(
        {
            "date": date,
            "strategy": strategy,
            "event": event,
            "regime": regime,
            "value_before": value_before,
            "reserve_cash": reserve_cash,
            "amount": amount,
        }
    )


def simulate_plain_dca(
    close: pd.Series,
    *,
    strategy: str,
    initial_capital: float,
    periodic_contribution: float,
    contribution_frequency: str,
    trading_day_interval: int,
    trading_cost_bps: float,
) -> StrategyResult:
    cost_rate = trading_cost_bps / 10_000.0
    shares = 0.0
    cash = float(initial_capital)
    curves: list[dict[str, Any]] = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(close.index[0], -float(initial_capital))]
    previous_period = contribution_bucket(0, close.index[0], contribution_frequency, trading_day_interval)

    initial_trade = cash / (1.0 + cost_rate)
    cash -= initial_trade * (1.0 + cost_rate)
    shares += initial_trade / float(close.iloc[0])

    for i, (date, price) in enumerate(close.items()):
        if i > 0:
            period = contribution_bucket(i, date, contribution_frequency, trading_day_interval)
            if period != previous_period:
                contribution = float(periodic_contribution)
                cashflows.append((date, -contribution))
                buy_value = contribution / (1.0 + cost_rate)
                shares += buy_value / float(price)
                previous_period = period
        total_value = cash + shares * float(price)
        curves.append(
            {
                "date": date,
                "strategy": strategy,
                "total_value": total_value,
                "exposure_equity": total_value,
                "reserve_cash": 0.0,
                "target_leverage": 1.0,
                "regime_signal": np.nan,
            }
        )

    curve_df = pd.DataFrame(curves).set_index("date")
    total_external = total_external_from_cashflows(cashflows)
    return StrategyResult(
        strategy=strategy,
        curves=curve_df,
        events=pd.DataFrame(columns=["date", "strategy", "event", "regime", "value_before", "reserve_cash", "amount"]),
        cashflows=cashflows,
        final_exposure_equity=float(curve_df["exposure_equity"].iloc[-1]),
        final_reserve_cash=0.0,
        avg_target_leverage=1.0,
        total_external_contributed=total_external,
        reserve_contributions=None,
        reserve_deployments=None,
        reserve_deploy_count=None,
        risk_on_entries=None,
        risk_on_leverage_cuts=None,
        risk_off_entries=None,
        transaction_costs=None,
        borrow_costs=None,
        risk_on_days=None,
        neutral_days=None,
        risk_off_days=None,
        risk_on_months=None,
        neutral_months=None,
        risk_off_months=None,
    )


def simulate_regime_leverage(
    close: pd.Series,
    regime_signal: pd.Series,
    *,
    strategy: str,
    risk_on_leverage: float,
    initial_capital: float,
    periodic_contribution: float,
    contribution_frequency: str,
    trading_day_interval: int,
    trading_cost_bps: float,
    borrow_rate: float,
) -> StrategyResult:
    signal = regime_signal.reindex(close.index)
    if signal.dropna().empty:
        raise RuntimeError("No lagged walk-forward regime signal available for leverage simulation.")
    price = close.loc[signal.dropna().index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
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
    total_external = float(initial_capital)

    first_regime = str(signal.iloc[0])
    current_leverage = target_leverage_for_regime(first_regime, risk_on_leverage)
    initial_cost = exposure_equity * current_leverage * cost_rate
    exposure_equity = max(0.0, exposure_equity - initial_cost)
    transaction_costs += initial_cost

    previous_price = float(price.iloc[0])
    previous_regime = first_regime
    previous_period = contribution_bucket(0, price.index[0], contribution_frequency, trading_day_interval)

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
        period = contribution_bucket(i, date, contribution_frequency, trading_day_interval)
        if i > 0 and period != previous_period:
            contribution = float(periodic_contribution)
            cashflows.append((date, -contribution))
            if regime == "risk_off":
                reserve_cash += contribution
                reserve_contributions += contribution
            else:
                contribution_cost = contribution * target_leverage_for_regime(regime, risk_on_leverage) * cost_rate
                transaction_costs += contribution_cost
                exposure_equity += max(0.0, contribution - contribution_cost)
            previous_period = period

        new_leverage = target_leverage_for_regime(regime, risk_on_leverage)
        if regime != previous_regime:
            if regime == "risk_off":
                risk_off_entries += 1
                record_event(
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
                record_event(
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
                record_event(
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
            record_event(
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
    total_external = total_external_from_cashflows(cashflows)
    return StrategyResult(
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


def metrics_row(window: str, result: StrategyResult, plain_final: float | None) -> dict[str, Any]:
    total_value = result.curves["total_value"]
    drawdown = total_value / total_value.cummax() - 1.0
    final_value = float(total_value.iloc[-1])
    time_weighted_total_return, time_weighted_cagr = time_weighted_return_metrics(total_value, result.cashflows)
    row = {
        "window": window,
        "strategy": result.strategy,
        "start_date": total_value.index[0].date().isoformat(),
        "end_date": total_value.index[-1].date().isoformat(),
        "final_value": final_value,
        "xirr": regime_analysis._xirr(result.cashflows + [(total_value.index[-1], final_value)]),
        "time_weighted_total_return": time_weighted_total_return,
        "time_weighted_cagr": time_weighted_cagr,
        "max_drawdown": float(drawdown.min()),
        "final_reserve_cash": result.final_reserve_cash,
        "final_exposure_equity": result.final_exposure_equity,
        "avg_target_leverage": result.avg_target_leverage,
        "total_external_contributed": result.total_external_contributed,
        "final_multiple_on_contributed": (final_value / result.total_external_contributed)
        if result.total_external_contributed > 0.0
        else np.nan,
        "reserve_contributions": result.reserve_contributions,
        "reserve_deployments": result.reserve_deployments,
        "reserve_deploy_count": result.reserve_deploy_count,
        "risk_on_entries": result.risk_on_entries,
        "risk_on_leverage_cuts": result.risk_on_leverage_cuts,
        "risk_off_entries": result.risk_off_entries,
        "transaction_costs": result.transaction_costs,
        "borrow_costs": result.borrow_costs,
        "risk_on_days": result.risk_on_days,
        "neutral_days": result.neutral_days,
        "risk_off_days": result.risk_off_days,
        "risk_on_months": result.risk_on_months,
        "neutral_months": result.neutral_months,
        "risk_off_months": result.risk_off_months,
        "final_delta_vs_plain_dca": (final_value - plain_final) if plain_final is not None else np.nan,
    }
    return row


def append_window(frames: list[pd.DataFrame], df: pd.DataFrame, window: str) -> None:
    if df.empty:
        return
    out = df.copy()
    out["window"] = window
    frames.append(out.reset_index())


def contribution_series(index: pd.DatetimeIndex, cashflows: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    flows = pd.Series(0.0, index=index, dtype=float)
    skipped_initial = False
    for date, amount in cashflows:
        amount = float(amount)
        if amount >= 0.0:
            continue
        if not skipped_initial:
            skipped_initial = True
            continue
        timestamp = pd.Timestamp(date)
        if timestamp in flows.index:
            flows.loc[timestamp] += -amount
    return flows


def time_weighted_return_metrics(
    total_value: pd.Series,
    cashflows: list[tuple[pd.Timestamp, float]],
) -> tuple[float, float]:
    equity = total_value.astype(float).dropna()
    if len(equity) < 2:
        return np.nan, np.nan

    contributions = contribution_series(equity.index, cashflows)
    daily_returns: list[float] = []
    previous_value = float(equity.iloc[0])
    for date, current_value in equity.iloc[1:].items():
        if previous_value <= 0.0 or not np.isfinite(previous_value):
            previous_value = float(current_value)
            continue
        flow = float(contributions.loc[date]) if date in contributions.index else 0.0
        daily_returns.append((float(current_value) - flow) / previous_value - 1.0)
        previous_value = float(current_value)

    if not daily_returns:
        return np.nan, np.nan
    twr_total = float(np.prod(1.0 + np.array(daily_returns, dtype=float)) - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.0)
    if years <= 0.0 or twr_total <= -1.0:
        return twr_total, np.nan
    twr_cagr = float((1.0 + twr_total) ** (1.0 / years) - 1.0)
    return twr_total, twr_cagr


def strategy_label(strategy: str) -> str:
    return PLOT_STYLE.get(strategy, {}).get("label", strategy)


def strategy_color(strategy: str) -> str:
    return PLOT_STYLE.get(strategy, {}).get("color", "#444444")


def plot_equity_curves(curves: pd.DataFrame, out_path: Path, window: str) -> None:
    data = curves[curves["window"] == window].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy, group in data.groupby("strategy", sort=False):
        group = group.sort_values("date")
        ax.plot(
            pd.to_datetime(group["date"]),
            group["total_value"].astype(float),
            linewidth=1.6,
            label=strategy_label(strategy),
            color=strategy_color(strategy),
        )
    ax.set_title(f"Walk-forward GMM leverage vs DCA: {window}")
    ax.set_ylabel("Account value, USD")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_drawdowns(curves: pd.DataFrame, out_path: Path, window: str) -> None:
    data = curves[curves["window"] == window].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy, group in data.groupby("strategy", sort=False):
        group = group.sort_values("date")
        equity = group["total_value"].astype(float)
        drawdown = equity / equity.cummax() - 1.0
        ax.plot(
            pd.to_datetime(group["date"]),
            drawdown,
            linewidth=1.6,
            label=strategy_label(strategy),
            color=strategy_color(strategy),
        )
    ax.set_title(f"Walk-forward GMM leverage drawdowns: {window}")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_audit_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    refits: pd.DataFrame,
    dataset_path: Path,
    contribution_frequency: str,
    periodic_contribution: float,
) -> None:
    black_box_cols = [
        "sentiment_black_box_pc1",
        "sentiment_black_box_pc2",
        "sentiment_black_box_pc1_explained_var",
    ]
    traded_overlap = sorted(set(regime_analysis.GMM_FEATURES).intersection(black_box_cols))
    comparable = metrics[metrics["window"] == "comparable_2007_05_31"].copy()
    comparable = comparable.sort_values("final_value", ascending=False)

    lines = [
        "# Walk-forward GMM Backtest Audit",
        "",
        "## Scope",
        "",
        f"- Dataset used: `{dataset_path}`",
        f"- Contribution cadence: `{contribution_frequency}` at `${periodic_contribution:,.2f}` per event.",
        "- Backtest under audit: dedicated daily walk-forward GMM leverage script.",
        "",
        "## Findings",
        "",
        "- PASS: walk-forward GMM refits train only on rows strictly before each refit date.",
        "- PASS: predictions are limited to the current month of each refit and then traded with a one-day lag.",
        "- PASS: leverage backtest uses `wf_gmm_regime_signal_lag1`, not the full-sample descriptive GMM labels.",
        "- NOTE: the analysis dataset contains full-sample descriptive GMM fields for reporting only.",
        "- NOTE: black-box PCA fields were audited separately; overlap with traded GMM features is "
        + (", ".join(traded_overlap) if traded_overlap else "`none`."),
        "",
        "## Refit Coverage",
        "",
    ]

    if refits.empty:
        lines.append("- No refits were generated.")
    else:
        first_refit = refits.iloc[0]
        last_refit = refits.iloc[-1]
        lines.extend(
            [
                f"- First refit: `{pd.Timestamp(first_refit['wf_refit_date']).date()}` trained through `{pd.Timestamp(first_refit['train_end']).date()}`.",
                f"- Last refit: `{pd.Timestamp(last_refit['wf_refit_date']).date()}` trained through `{pd.Timestamp(last_refit['train_end']).date()}`.",
                f"- Refit count: `{len(refits)}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Comparable Window Metrics",
            "",
            "| Strategy | Final Value | TWR | TWR CAGR | Max DD | Final / Contributed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparable.iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['final_value']:,.0f} | {row['time_weighted_total_return']:.1%} | "
            f"{row['time_weighted_cagr']:.1%} | {row['max_drawdown']:.1%} | {row['final_multiple_on_contributed']:.2f}x |"
        )
    lines.append("")
    out_dir.joinpath("walkforward_gmm_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    maybe_run_analysis(args)
    periodic_contribution = resolve_periodic_contribution(args)

    dataset = load_dataset(args.dataset_path)
    if args.end:
        dataset = dataset.loc[dataset.index <= pd.Timestamp(args.end)]
    walkforward_daily, refits, summary = build_walkforward_daily_regimes(dataset, args.min_train_days)
    walkforward_daily.to_csv(args.out_dir / "walkforward_gmm_daily_regimes.csv", index_label="date")
    refits.to_csv(args.out_dir / "walkforward_gmm_refits.csv", index=False)
    summary.to_csv(args.out_dir / "walkforward_gmm_regime_summary.csv", index=False)

    close = dataset["qqq_close"].dropna().astype(float)
    lagged_signal = walkforward_daily["wf_gmm_regime_signal_lag1"]
    first_signal_date = lagged_signal.dropna().index.min()
    if first_signal_date is None:
        raise RuntimeError("Walk-forward daily GMM did not produce any lagged signals.")

    strategy_defs = [
        ("plain_dca", None),
        ("walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca", 2.0),
        ("walkforward_gmm_riskon_3x_keep_long_riskoff_reserve_dca", 3.0),
    ]
    window_starts = {
        "full_walkforward_window": first_signal_date,
        "comparable_2007_05_31": max(first_signal_date, COMPARABLE_START),
    }

    metrics_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for window, start_date in window_starts.items():
        window_close = close.loc[close.index >= start_date]
        window_signal = lagged_signal.loc[lagged_signal.index >= start_date]
        plain = simulate_plain_dca(
            window_close,
            strategy="plain_dca",
            initial_capital=args.initial_capital,
            periodic_contribution=periodic_contribution,
            contribution_frequency=args.contribution_frequency,
            trading_day_interval=args.trading_day_interval,
            trading_cost_bps=args.trading_cost_bps,
        )
        append_window(curve_frames, plain.curves, window)
        plain_final = float(plain.curves["total_value"].iloc[-1])
        metrics_rows.append(metrics_row(window, plain, None))

        for strategy, leverage in strategy_defs[1:]:
            result = simulate_regime_leverage(
                window_close,
                window_signal,
                strategy=strategy,
                risk_on_leverage=float(leverage),
                initial_capital=args.initial_capital,
                periodic_contribution=periodic_contribution,
                contribution_frequency=args.contribution_frequency,
                trading_day_interval=args.trading_day_interval,
                trading_cost_bps=args.trading_cost_bps,
                borrow_rate=args.borrow_rate,
            )
            append_window(curve_frames, result.curves, window)
            if not result.events.empty:
                events = result.events.copy()
                events["window"] = window
                event_frames.append(events)
            metrics_rows.append(metrics_row(window, result, plain_final))

    metrics = pd.DataFrame(metrics_rows)
    metrics["contribution_frequency"] = contribution_frequency_label(args)
    metrics["periodic_contribution"] = periodic_contribution
    metrics["trading_day_interval"] = args.trading_day_interval if args.contribution_frequency == "trading_days" else np.nan
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    metrics.to_csv(args.out_dir / "walkforward_gmm_riskon_leverage_metrics.csv", index=False)
    curves.to_csv(args.out_dir / "walkforward_gmm_riskon_leverage_curves.csv", index=False)
    events.to_csv(args.out_dir / "walkforward_gmm_riskon_leverage_events.csv", index=False)
    for window in window_starts:
        plot_equity_curves(curves, args.out_dir / f"walkforward_gmm_equity_{window}.png", window)
        plot_drawdowns(curves, args.out_dir / f"walkforward_gmm_drawdown_{window}.png", window)
    write_audit_report(
        args.out_dir,
        metrics=metrics,
        refits=refits,
        dataset_path=args.dataset_path,
        contribution_frequency=contribution_frequency_label(args),
        periodic_contribution=periodic_contribution,
    )

    print(f"Wrote walk-forward daily GMM files to {args.out_dir}")
    print(
        "Contribution cadence: "
        f"{contribution_frequency_label(args)} at ${periodic_contribution:,.2f} per contribution event"
    )
    latest = walkforward_daily.dropna(subset=["wf_gmm_regime"]).iloc[-1]
    print(
        "Latest walk-forward daily regime: "
        f"{latest.name.date()} -> {latest['wf_gmm_regime']} "
        f"(lagged signal for next day: {latest['wf_gmm_regime']})"
    )
    latest_window = metrics[metrics["window"] == "comparable_2007_05_31"].copy()
    if not latest_window.empty:
        latest_window = latest_window.sort_values("final_value", ascending=False)
        print("Comparable window final values:")
        for _, row in latest_window.iterrows():
            print(f"  {row['strategy']}: ${row['final_value']:,.0f}")


if __name__ == "__main__":
    main()
