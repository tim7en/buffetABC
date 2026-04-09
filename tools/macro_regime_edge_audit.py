"""Professional macro regime audit package for QQQ DCA edge research."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qqq_macro_ml_regime_analysis as regime_analysis
import qqq_macro_walkforward_leverage as leverage
import qqq_macro_walkforward_model_compare as model_compare


ROOT = regime_analysis.ROOT
DEFAULT_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_COMPARE_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260409_monthly_equal"
DEFAULT_OUT_DIR = ROOT / "reports" / "macro_regime_edge_audit_20260409"

BASE_ALLOCATION_MAP = {
    "risk_on": 1.00,
    "neutral": 0.70,
    "risk_off": 0.25,
}

SEASONAL_MONTH_BUCKETS = {
    "quarter_end": {3, 6, 9, 12},
    "turn_of_quarter": {1, 4, 7, 10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=100.0)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    return parser.parse_args()


def load_csv(path: Path, *, index_col: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[index_col] if index_col else None)
    if index_col:
        frame = frame.set_index(index_col).sort_index()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def realized_regime(sample: pd.DataFrame) -> pd.Series:
    regime = pd.Series("neutral", index=sample.index, dtype=object)
    regime.loc[sample["risk_off_target"].eq(1.0)] = "risk_off"
    regime.loc[sample["jump_in_target"].eq(1.0)] = "risk_on"
    regime.loc[sample["risk_off_target"].isna() | sample["jump_in_target"].isna()] = np.nan
    return regime.rename("realized_regime")


def regime_month_end_from_daily(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    valid = frame.dropna(subset=[column]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["model_name", "regime"])
    out = valid.groupby(valid.index.to_period("M"), sort=True).tail(1)[[column]].rename(columns={column: "regime"})
    out["model_name"] = label
    return out[["model_name", "regime"]]


def build_consensus_regime(monthly_predictions: dict[str, pd.Series]) -> pd.Series:
    joined = pd.DataFrame(monthly_predictions)
    out = pd.Series(index=joined.index, dtype=object)
    risk_off = joined.eq("risk_off").any(axis=1)
    risk_on = joined.eq("risk_on").all(axis=1)
    out.loc[risk_off] = "risk_off"
    out.loc[~risk_off & risk_on] = "risk_on"
    out.loc[out.isna() & joined.notna().all(axis=1)] = "neutral"
    return out.rename("consensus")


def quarterly_flag(index: pd.DatetimeIndex, bucket: str) -> pd.Series:
    months = SEASONAL_MONTH_BUCKETS[bucket]
    return pd.Series(index.month.isin(months), index=index, name=bucket)


def forward_return_stats(sample: pd.DataFrame, predicted: pd.Series, model_name: str) -> pd.DataFrame:
    merged = sample.join(predicted.rename("predicted_regime"), how="left").dropna(subset=["predicted_regime"])
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for regime_name, group in merged.groupby("predicted_regime", sort=False):
        rows.append(
            {
                "model_name": model_name,
                "predicted_regime": regime_name,
                "n_months": int(len(group)),
                "avg_fwd_21d_return": float(group["qqq_fwd_21d_return"].mean()),
                "avg_fwd_63d_return": float(group["qqq_fwd_63d_return"].mean()),
                "avg_fwd_126d_return": float(group["qqq_fwd_126d_return"].mean()),
                "median_fwd_63d_return": float(group["qqq_fwd_63d_return"].median()),
                "positive_63d_rate": float((group["qqq_fwd_63d_return"] > 0.0).mean()),
                "risk_off_event_rate": float(group["risk_off_target"].mean()),
                "jump_in_event_rate": float(group["jump_in_target"].mean()),
                "avg_max_drawdown_next_63d": float(group["qqq_fwd_63d_min_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def one_vs_rest_metrics(predicted: pd.Series, actual: pd.Series, label: str) -> dict[str, Any]:
    pred_pos = predicted.eq(label)
    actual_pos = actual.eq(label)
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    balanced_accuracy = np.nanmean([recall, specificity])
    false_positive_rate = fp / (fp + tn) if fp + tn > 0 else np.nan
    return {
        "label": label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_positive_rate,
    }


def regime_accuracy_report(sample: pd.DataFrame, predictions: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    actual = realized_regime(sample)
    metric_rows: list[dict[str, Any]] = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    for model_name, predicted in predictions.items():
        merged = pd.concat([actual, predicted.rename("predicted_regime")], axis=1).dropna()
        if merged.empty:
            continue
        confusion = pd.crosstab(
            merged["realized_regime"],
            merged["predicted_regime"],
            dropna=False,
        ).reindex(index=["risk_off", "neutral", "risk_on"], columns=["risk_off", "neutral", "risk_on"], fill_value=0)
        confusion_tables[model_name] = confusion
        accuracy = float((merged["realized_regime"] == merged["predicted_regime"]).mean())
        for label in ["risk_off", "neutral", "risk_on"]:
            row = one_vs_rest_metrics(merged["predicted_regime"], merged["realized_regime"], label)
            row["model_name"] = model_name
            row["overall_accuracy"] = accuracy
            row["n_months"] = int(len(merged))
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    return metrics, confusion_tables


def monthly_signal_to_allocation(predicted: pd.Series, allocation_map: dict[str, float]) -> pd.Series:
    out = predicted.map(allocation_map).astype(float)
    return out.rename("target_equity_allocation")


def strategy_metrics_from_dca(window: str, result: regime_analysis.DcaResult, plain_final: float) -> dict[str, Any]:
    equity = result.equity.astype(float)
    drawdown = equity / equity.cummax() - 1.0
    final_value = float(equity.iloc[-1])
    total_contributed = -sum(amount for _, amount in result.cashflows[:-1] if amount < 0.0)
    return {
        "window": window,
        "strategy": result.name,
        "start_date": equity.index[0].date().isoformat(),
        "end_date": equity.index[-1].date().isoformat(),
        "final_value": final_value,
        "xirr": regime_analysis._xirr(result.cashflows),
        "time_weighted_total_return": final_value / total_contributed - 1.0 if total_contributed > 0 else np.nan,
        "time_weighted_cagr": leverage.time_weighted_return_metrics(equity, result.cashflows)[1],
        "max_drawdown": float(drawdown.min()),
        "avg_target_leverage": float(result.allocation.mean()),
        "total_external_contributed": float(total_contributed),
        "final_multiple_on_contributed": (final_value / total_contributed) if total_contributed > 0 else np.nan,
        "final_delta_vs_plain_dca": final_value - plain_final,
    }


def simulate_allocation_strategies(
    close: pd.Series,
    monthly_predictions: dict[str, pd.Series],
    *,
    start_date: pd.Timestamp,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[regime_analysis.DcaResult] = []
    plain_target = pd.Series(1.0, index=pd.DatetimeIndex([start_date]), name="plain")
    plain = regime_analysis.simulate_dca(
        close,
        plain_target,
        name="plain_dca_1x",
        start_date=start_date,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        trading_cost_bps=trading_cost_bps,
    )
    results.append(plain)
    for model_name, signal in monthly_predictions.items():
        allocation = monthly_signal_to_allocation(signal, BASE_ALLOCATION_MAP)
        result = regime_analysis.simulate_dca(
            close,
            allocation,
            name=f"{model_name}_macro_aware_1x_dca",
            start_date=start_date,
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
            trading_cost_bps=trading_cost_bps,
        )
        results.append(result)
    metrics_rows = [strategy_metrics_from_dca("full_common_window", result, float(plain.equity.iloc[-1])) for result in results]
    curves = pd.concat([result.equity for result in results], axis=1)
    return pd.DataFrame(metrics_rows).sort_values("final_value", ascending=False), curves


def simulate_leverage_strategies(
    close: pd.Series,
    daily_signals: dict[str, pd.Series],
    *,
    start_date: pd.Timestamp,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
    borrow_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = close.loc[close.index >= start_date].copy()
    plain = leverage.simulate_plain_dca(
        price,
        strategy="plain_dca_1x",
        initial_capital=initial_capital,
        periodic_contribution=monthly_contribution,
        contribution_frequency="monthly",
        trading_day_interval=3,
        trading_cost_bps=trading_cost_bps,
    )
    metrics_rows = [leverage.metrics_row("full_common_window", plain, None)]
    curve_frames: list[pd.DataFrame] = []
    leverage.append_window(curve_frames, plain.curves, "full_common_window")
    plain_final = float(plain.curves["total_value"].iloc[-1])
    for model_name, signal in daily_signals.items():
        result = leverage.simulate_regime_leverage(
            price,
            signal.loc[signal.index >= start_date],
            strategy=f"{model_name}_macro_aware_2x_dca",
            risk_on_leverage=2.0,
            initial_capital=initial_capital,
            periodic_contribution=monthly_contribution,
            contribution_frequency="monthly",
            trading_day_interval=3,
            trading_cost_bps=trading_cost_bps,
            borrow_rate=borrow_rate,
        )
        metrics_rows.append(leverage.metrics_row("full_common_window", result, plain_final))
        leverage.append_window(curve_frames, result.curves, "full_common_window")
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    return pd.DataFrame(metrics_rows).sort_values("final_value", ascending=False), curves


def logistic_threshold_sensitivity_1x_2x(
    close: pd.Series,
    logistic_monthly: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
    borrow_rate: float,
) -> pd.DataFrame:
    if logistic_monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    risk_off_thresholds = [0.40, 0.45, 0.50]
    jump_in_thresholds = [0.50, 0.55, 0.60]
    for risk_off_threshold in risk_off_thresholds:
        for jump_in_threshold in jump_in_thresholds:
            regime = logistic_monthly.apply(
                lambda row: model_compare.regime_from_probabilities(
                    float(row["risk_off_probability"]),
                    float(row["jump_in_probability"]),
                    risk_off_threshold,
                    jump_in_threshold,
                ),
                axis=1,
            )
            allocation = monthly_signal_to_allocation(regime, BASE_ALLOCATION_MAP)
            result_1x = regime_analysis.simulate_dca(
                close,
                allocation,
                name="logistic_threshold_1x",
                start_date=start_date,
                initial_capital=initial_capital,
                monthly_contribution=monthly_contribution,
                trading_cost_bps=trading_cost_bps,
            )
            row_1x = strategy_metrics_from_dca("full_common_window", result_1x, np.nan)
            row_1x.update(
                {
                    "risk_off_threshold": risk_off_threshold,
                    "jump_in_threshold": jump_in_threshold,
                    "strategy_mode": "allocation_1x",
                }
            )
            rows.append(row_1x)

            daily_signal = model_compare.monthly_signal_to_daily(regime, close.index)
            result_2x = leverage.simulate_regime_leverage(
                close.loc[close.index >= start_date],
                daily_signal.loc[daily_signal.index >= start_date],
                strategy="logistic_threshold_2x",
                risk_on_leverage=2.0,
                initial_capital=initial_capital,
                periodic_contribution=monthly_contribution,
                contribution_frequency="monthly",
                trading_day_interval=3,
                trading_cost_bps=trading_cost_bps,
                borrow_rate=borrow_rate,
            )
            row_2x = leverage.metrics_row("full_common_window", result_2x, np.nan)
            row_2x.update(
                {
                    "risk_off_threshold": risk_off_threshold,
                    "jump_in_threshold": jump_in_threshold,
                    "strategy_mode": "leverage_2x",
                }
            )
            rows.append(row_2x)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["strategy_mode", "final_value"], ascending=[True, False])


def allocation_sensitivity_1x(
    close: pd.Series,
    monthly_regime: pd.Series,
    *,
    start_date: pd.Timestamp,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for risk_off_allocation in [0.0, 0.25, 0.40, 0.50]:
        for neutral_allocation in [0.60, 0.70, 0.85, 1.00]:
            if neutral_allocation < risk_off_allocation:
                continue
            allocation_map = {
                "risk_on": 1.0,
                "neutral": neutral_allocation,
                "risk_off": risk_off_allocation,
            }
            allocation = monthly_signal_to_allocation(monthly_regime, allocation_map)
            result = regime_analysis.simulate_dca(
                close,
                allocation,
                name=f"{prefix}_allocation_sensitivity_1x",
                start_date=start_date,
                initial_capital=initial_capital,
                monthly_contribution=monthly_contribution,
                trading_cost_bps=trading_cost_bps,
            )
            row = strategy_metrics_from_dca("full_common_window", result, np.nan)
            row["risk_off_allocation"] = risk_off_allocation
            row["neutral_allocation"] = neutral_allocation
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("final_value", ascending=False)


def seasonality_table(sample: pd.DataFrame, prediction: pd.Series, model_name: str) -> pd.DataFrame:
    merged = sample.join(prediction.rename("predicted_regime"), how="left").dropna(subset=["predicted_regime"])
    if merged.empty:
        return pd.DataFrame()
    merged["is_quarter_end"] = quarterly_flag(merged.index, "quarter_end").astype(bool)
    merged["is_turn_of_quarter"] = quarterly_flag(merged.index, "turn_of_quarter").astype(bool)
    rows: list[dict[str, Any]] = []
    for bucket in ["is_quarter_end", "is_turn_of_quarter"]:
        for (regime_name, flag), group in merged.groupby(["predicted_regime", bucket], sort=False):
            rows.append(
                {
                    "model_name": model_name,
                    "predicted_regime": regime_name,
                    "seasonality_bucket": bucket,
                    "flag": bool(flag),
                    "n_months": int(len(group)),
                    "avg_fwd_21d_return": float(group["qqq_fwd_21d_return"].mean()),
                    "avg_fwd_63d_return": float(group["qqq_fwd_63d_return"].mean()),
                    "positive_21d_rate": float((group["qqq_fwd_21d_return"] > 0.0).mean()),
                    "risk_off_event_rate": float(group["risk_off_target"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_feature_scorecard(impact: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = sorted(set(impact["term"]).union(set(importance["feature"])))
    for feature in features:
        if feature == "intercept":
            continue
        row: dict[str, Any] = {"feature": feature, "label": regime_analysis.feature_label(feature)}
        for horizon in [63, 126, 252]:
            subset = impact[(impact["term"] == feature) & (impact["horizon_days"] == horizon)]
            if subset.empty:
                row[f"ols_{horizon}d_coef_pp_per_1sd"] = np.nan
                row[f"ols_{horizon}d_q_value"] = np.nan
            else:
                best = subset.iloc[0]
                row[f"ols_{horizon}d_coef_pp_per_1sd"] = float(best["coef_pct_points_per_1sd"])
                row[f"ols_{horizon}d_q_value"] = float(best["q_value_bh_fdr"])
        for target, model, prefix in [
            ("qqq_fwd_63d_return", "ridge", "ridge_return"),
            ("qqq_fwd_63d_return", "random_forest", "rf_return"),
            ("risk_off_target", "logistic", "logistic_risk_off"),
            ("risk_off_target", "random_forest", "rf_risk_off"),
            ("jump_in_target", "logistic", "logistic_jump_in"),
            ("jump_in_target", "random_forest", "rf_jump_in"),
        ]:
            subset = importance[(importance["target"] == target) & (importance["model"] == model) & (importance["feature"] == feature)]
            row[f"{prefix}_importance"] = float(subset.iloc[0]["importance_mean"]) if not subset.empty else np.nan
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard
    scorecard["valuation_flag"] = scorecard["feature"].str.contains("cape|buffett_indicator", case=False, regex=True)
    scorecard["stress_flag"] = scorecard["feature"].str.contains("vix|hy_oas|nfci|unemployment|cpi|shock", case=False, regex=True)
    scorecard["trend_flag"] = scorecard["feature"].str.contains("sma|drawdown|qqq_", case=False, regex=True)
    return scorecard.sort_values(["ols_252d_q_value", "ols_126d_q_value", "feature"], ascending=[True, True, True])


def copy_inputs(out_dir: Path, analysis_dir: Path, compare_dir: Path) -> None:
    data_dir = out_dir / "data"
    scripts_dir = out_dir / "scripts"
    data_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for src in [
        analysis_dir / "aligned_daily_dataset.csv",
        analysis_dir / "month_end_model_sample.csv",
        analysis_dir / "analysis_variable_inventory.csv",
        analysis_dir / "current_market_environment.csv",
        analysis_dir / "ols_newey_west_impact.csv",
        analysis_dir / "model_feature_importance.csv",
        analysis_dir / "model_validation_metrics.csv",
        compare_dir / "walkforward_model_regimes_monthly.csv",
        compare_dir / "walkforward_model_signals_daily.csv",
        compare_dir / "walkforward_gmm_daily_regimes.csv",
        compare_dir / "walkforward_model_compare_leverage_metrics.csv",
        compare_dir / "walkforward_model_validation_metrics.csv",
        compare_dir / "walkforward_model_feature_importance.csv",
        compare_dir / "walkforward_logistic_threshold_sensitivity.csv",
    ]:
        if src.exists():
            shutil.copy2(src, data_dir / src.name)
    for src in [
        ROOT / "cache" / "download_daily_macro_data.py",
        ROOT / "tools" / "qqq_macro_ml_regime_analysis.py",
        ROOT / "tools" / "qqq_macro_walkforward_model_compare.py",
        Path(__file__),
    ]:
        if src.exists():
            shutil.copy2(src, scripts_dir / src.name)


def plot_equity(curves: pd.DataFrame, out_path: Path, title: str) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in curves.columns:
        ax.plot(curves.index, curves[column].astype(float), linewidth=1.5, label=column)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylabel("Account value, USD")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    current_environment: pd.DataFrame,
    accuracy: pd.DataFrame,
    expected_returns: pd.DataFrame,
    strategies_1x: pd.DataFrame,
    strategies_2x: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    allocation_sensitivity: pd.DataFrame,
    seasonality: pd.DataFrame,
    feature_scorecard: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Macro Regime Edge Audit",
        "",
        "## Anti-Leakage Discipline",
        "",
        "- QQQ prices are aligned on actual trading dates only.",
        "- Monthly supervised models train only on earlier month-end observations.",
        "- Any row whose forward target window overlaps the prediction date is purged from training.",
        "- Allocation and leverage simulations trade lagged regime signals only.",
        "- Quarterly GDP and annual market-cap-to-GDP anchor data are lagged before forward-fill, so the proxy is not using future macro releases.",
        "",
        "## Current Read",
        "",
    ]
    if not current_environment.empty:
        row = current_environment.iloc[0]
        lines.extend(
            [
                f"- As of `{pd.Timestamp(row['as_of']).date()}` the macro cycle is `{row['macro_cycle']}` with `{row['macro_cycle_confidence']}` confidence.",
                f"- Expansion / late-cycle / contraction scores are `{row['expansion_score']:.0f}` / `{row['late_cycle_score']:.0f}` / `{row['contraction_score']:.0f}`.",
                f"- The Wilshire / GDP proxy is `{row['buffett_indicator_proxy_level']:.4f}` with rolling z-score `{row['buffett_indicator_proxy_rolling_z']:.2f}`.",
                f"- The traded regime is `{row['combined_market_regime']}` with target allocation `{row['target_equity_allocation'] * 100:.0f}%`.",
            ]
        )
    lines.extend(["", "## Regime Accuracy", ""])
    if accuracy.empty:
        lines.append("- No regime accuracy table could be computed.")
    else:
        lines.append("| Model | Label | Precision | Recall | Balanced Accuracy | False Positive Rate | Overall Accuracy |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in accuracy.sort_values(["model_name", "label"]).iterrows():
            lines.append(
                f"| {row['model_name']} | {row['label']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['balanced_accuracy']:.3f} | {row['false_positive_rate']:.3f} | {row['overall_accuracy']:.3f} |"
            )
    lines.extend(["", "## Expected Returns By Predicted Regime", ""])
    if expected_returns.empty:
        lines.append("- No expected-return table could be computed.")
    else:
        lines.append("| Model | Predicted Regime | N | Avg 21D | Avg 63D | Avg 126D | Positive 63D Rate | Risk-off Event Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in expected_returns.sort_values(["model_name", "predicted_regime"]).iterrows():
            lines.append(
                f"| {row['model_name']} | {row['predicted_regime']} | {int(row['n_months'])} | "
                f"{row['avg_fwd_21d_return'] * 100:.1f}% | {row['avg_fwd_63d_return'] * 100:.1f}% | "
                f"{row['avg_fwd_126d_return'] * 100:.1f}% | {row['positive_63d_rate'] * 100:.1f}% | "
                f"{row['risk_off_event_rate'] * 100:.1f}% |"
            )
    lines.extend(["", "## Plain DCA vs Macro-Aware 1x", ""])
    lines.append("| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in strategies_1x.sort_values("final_value", ascending=False).iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['final_value']:,.0f} | {row['xirr'] * 100:.1f}% | "
            f"{row['time_weighted_cagr'] * 100:.1f}% | {row['max_drawdown'] * 100:.1f}% | "
            f"{row['final_multiple_on_contributed']:.2f}x |"
        )
    lines.extend(["", "## Plain DCA vs Macro-Aware 2x", ""])
    lines.append("| Strategy | Final Value | XIRR | CAGR | Max DD | Final / Contributed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in strategies_2x.sort_values("final_value", ascending=False).iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['final_value']:,.0f} | {row['xirr'] * 100:.1f}% | "
            f"{row['time_weighted_cagr'] * 100:.1f}% | {row['max_drawdown'] * 100:.1f}% | "
            f"{row['final_multiple_on_contributed']:.2f}x |"
        )
    lines.extend(["", "## Sensitivity", ""])
    if not threshold_sensitivity.empty:
        best_1x = threshold_sensitivity[threshold_sensitivity["strategy_mode"] == "allocation_1x"].head(1)
        best_2x = threshold_sensitivity[threshold_sensitivity["strategy_mode"] == "leverage_2x"].head(1)
        if not best_1x.empty:
            row = best_1x.iloc[0]
            lines.append(
                f"- Best ex-post logistic threshold in the audited 1x grid: risk-off `{row['risk_off_threshold']:.2f}`, "
                f"jump-in `{row['jump_in_threshold']:.2f}`, final value `${row['final_value']:,.0f}`, max DD `{row['max_drawdown'] * 100:.1f}%`."
            )
        if not best_2x.empty:
            row = best_2x.iloc[0]
            lines.append(
                f"- Best ex-post logistic threshold in the audited 2x grid: risk-off `{row['risk_off_threshold']:.2f}`, "
                f"jump-in `{row['jump_in_threshold']:.2f}`, final value `${row['final_value']:,.0f}`, max DD `{row['max_drawdown'] * 100:.1f}%`."
            )
    if not allocation_sensitivity.empty:
        row = allocation_sensitivity.iloc[0]
        lines.append(
            f"- Best ex-post 1x allocation grid for logistic used risk-off `{row['risk_off_allocation'] * 100:.0f}%`, "
            f"neutral `{row['neutral_allocation'] * 100:.0f}%`, finishing at `${row['final_value']:,.0f}`."
        )
    lines.extend(["", "## Quarter-End / Window-Dressing Diagnostics", ""])
    if seasonality.empty:
        lines.append("- No seasonality table could be computed.")
    else:
        for bucket in ["is_quarter_end", "is_turn_of_quarter"]:
            subset = seasonality[seasonality["seasonality_bucket"] == bucket].copy()
            if subset.empty:
                continue
            lines.append("")
            lines.append(f"### {bucket}")
            lines.append("")
            lines.append("| Model | Regime | Flag | N | Avg 21D | Avg 63D | Positive 21D Rate |")
            lines.append("|---|---|---|---:|---:|---:|---:|")
            for _, row in subset.sort_values(["model_name", "predicted_regime", "flag"]).iterrows():
                lines.append(
                    f"| {row['model_name']} | {row['predicted_regime']} | {row['flag']} | {int(row['n_months'])} | "
                    f"{row['avg_fwd_21d_return'] * 100:.1f}% | {row['avg_fwd_63d_return'] * 100:.1f}% | "
                    f"{row['positive_21d_rate'] * 100:.1f}% |"
                )
    lines.extend(["", "## Feature Takeaways", ""])
    if not feature_scorecard.empty:
        focus = feature_scorecard[
            feature_scorecard["feature"].isin(
                [
                    "buffett_indicator_proxy_level",
                    "buffett_indicator_proxy_252d_drift",
                    "cape_level",
                    "cpi_yoy_pct",
                    "unemployment_rate_pct",
                    "vix_level",
                    "hy_oas_level",
                    "qqq_sma65",
                    "qqq_sma222",
                    "latent_sentiment_index",
                ]
            )
        ].copy()
        lines.append("| Feature | 252D OLS Coef | 252D q-value | Ridge Return Coef | Logistic Risk-off Coef | Logistic Jump-in Coef |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in focus.sort_values("feature").iterrows():
            lines.append(
                f"| {row['label']} | {row['ols_252d_coef_pp_per_1sd']:.2f} | {row['ols_252d_q_value']:.3f} | "
                f"{row['ridge_return_importance']:.3f} | {row['logistic_risk_off_importance']:.3f} | "
                f"{row['logistic_jump_in_importance']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## View",
            "",
            "- Macro awareness does appear to add information, but most of the edge comes from combining valuation, stress, inflation, and trend rather than any single magic series.",
            "- The Wilshire / GDP proxy behaves like a medium-horizon valuation headwind, not a fast crash alarm.",
            "- Stress and liquidity proxies such as VIX, credit spreads, and financial conditions remain more useful for tactical regime shifts.",
            "- If the objective is to beat plain DCA without leverage, the bar is high and the evidence must come from stable out-of-sample allocation rules, not from the single best ex-post parameter cell.",
            "- If the objective allows 2x leverage, the evidence currently supports only measured use during confirmed risk-on states. A human macro-aware investor likely gets more benefit from cutting exposure in bad regimes than from maximizing leverage in good ones.",
            "- The professional playbook here is to use monthly walk-forward classification, daily lagged execution, disciplined reserve deployment, and explicit review of false positives around risk-off calls.",
        ]
    )
    out_dir.joinpath("macro_regime_edge_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = leverage.load_dataset(args.analysis_dir / "aligned_daily_dataset.csv")
    sample = load_csv(args.analysis_dir / "month_end_model_sample.csv", index_col="date")
    impact = pd.read_csv(args.analysis_dir / "ols_newey_west_impact.csv")
    feature_importance = pd.read_csv(args.analysis_dir / "model_feature_importance.csv")
    current_environment = pd.read_csv(args.analysis_dir / "current_market_environment.csv", parse_dates=["as_of"])

    compare_metrics = pd.read_csv(args.compare_dir / "walkforward_model_compare_leverage_metrics.csv")
    compare_validation = pd.read_csv(args.compare_dir / "walkforward_model_validation_metrics.csv")
    compare_feature_importance = pd.read_csv(args.compare_dir / "walkforward_model_feature_importance.csv")
    daily_signals = load_csv(args.compare_dir / "walkforward_model_signals_daily.csv", index_col="date")
    monthly_regimes = load_csv(args.compare_dir / "walkforward_model_regimes_monthly.csv", index_col="date")
    gmm_daily = load_csv(args.compare_dir / "walkforward_gmm_daily_regimes.csv", index_col="date")
    threshold_file = args.compare_dir / "walkforward_logistic_threshold_sensitivity.csv"
    base_threshold_sensitivity = pd.read_csv(threshold_file) if threshold_file.exists() else pd.DataFrame()

    plain_row = compare_metrics[
        (compare_metrics["window"] == "full_common_window") & (compare_metrics["strategy"] == "plain_dca")
    ].iloc[0]
    common_start = pd.Timestamp(plain_row["start_date"])
    close = dataset["qqq_close"].astype(float)

    logistic_monthly = monthly_regimes[monthly_regimes["model_name"] == "logistic"].copy()
    logistic_prediction = logistic_monthly["regime"].rename("logistic")
    gmm_monthly = regime_month_end_from_daily(gmm_daily, "wf_gmm_regime", "gmm")
    gmm_prediction = gmm_monthly["regime"].rename("gmm")
    consensus_prediction = build_consensus_regime({"logistic": logistic_prediction, "gmm": gmm_prediction})

    prediction_map = {
        "logistic": logistic_prediction,
        "gmm": gmm_prediction,
        "consensus": consensus_prediction,
    }

    accuracy, confusion_tables = regime_accuracy_report(sample, prediction_map)
    expected_returns = pd.concat(
        [forward_return_stats(sample, prediction, model_name) for model_name, prediction in prediction_map.items()],
        ignore_index=True,
    )
    seasonality = pd.concat(
        [seasonality_table(sample, prediction, model_name) for model_name, prediction in prediction_map.items()],
        ignore_index=True,
    )

    one_x_metrics, one_x_curves = simulate_allocation_strategies(
        close,
        prediction_map,
        start_date=common_start,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        trading_cost_bps=args.trading_cost_bps,
    )

    daily_consensus = model_compare.monthly_signal_to_daily(consensus_prediction, close.index)
    leverage_signal_map = {
        "logistic": daily_signals["logistic_signal_lag1"],
        "gmm": daily_signals["gmm_signal_lag1"],
        "consensus": daily_consensus,
    }
    two_x_metrics, two_x_curves = simulate_leverage_strategies(
        close,
        leverage_signal_map,
        start_date=common_start,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        trading_cost_bps=args.trading_cost_bps,
        borrow_rate=args.borrow_rate,
    )

    logistic_threshold_sensitivity = logistic_threshold_sensitivity_1x_2x(
        close,
        logistic_monthly,
        start_date=common_start,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        trading_cost_bps=args.trading_cost_bps,
        borrow_rate=args.borrow_rate,
    )
    allocation_sensitivity = allocation_sensitivity_1x(
        close,
        logistic_prediction,
        start_date=common_start,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        trading_cost_bps=args.trading_cost_bps,
        prefix="logistic",
    )
    feature_scorecard = build_feature_scorecard(impact, feature_importance)

    copy_inputs(args.out_dir, args.analysis_dir, args.compare_dir)
    expected_returns.to_csv(args.out_dir / "regime_expected_returns.csv", index=False)
    accuracy.to_csv(args.out_dir / "regime_accuracy_metrics.csv", index=False)
    seasonality.to_csv(args.out_dir / "quarter_end_seasonality_by_regime.csv", index=False)
    one_x_metrics.to_csv(args.out_dir / "macro_aware_strategy_metrics_1x.csv", index=False)
    two_x_metrics.to_csv(args.out_dir / "macro_aware_strategy_metrics_2x.csv", index=False)
    logistic_threshold_sensitivity.to_csv(args.out_dir / "logistic_threshold_sensitivity_1x_2x.csv", index=False)
    allocation_sensitivity.to_csv(args.out_dir / "logistic_allocation_sensitivity_1x.csv", index=False)
    feature_scorecard.to_csv(args.out_dir / "feature_signal_scorecard.csv", index=False)
    compare_validation.to_csv(args.out_dir / "walkforward_model_validation_metrics_snapshot.csv", index=False)
    compare_feature_importance.to_csv(args.out_dir / "walkforward_model_feature_importance_snapshot.csv", index=False)
    if not base_threshold_sensitivity.empty:
        base_threshold_sensitivity.to_csv(args.out_dir / "walkforward_logistic_threshold_sensitivity_snapshot.csv", index=False)

    for model_name, confusion in confusion_tables.items():
        confusion.to_csv(args.out_dir / f"regime_confusion_{model_name}.csv")

    one_x_curves.to_csv(args.out_dir / "macro_aware_equity_curves_1x.csv", index_label="date")
    if not two_x_curves.empty:
        two_x_curves.to_csv(args.out_dir / "macro_aware_equity_curves_2x.csv", index=False)

    plot_equity(one_x_curves, plots_dir / "macro_aware_equity_1x.png", "Plain DCA vs macro-aware 1x overlays")
    if not two_x_curves.empty:
        pivot = two_x_curves.pivot(index="date", columns="strategy", values="total_value")
        pivot.index = pd.to_datetime(pivot.index)
        plot_equity(pivot, plots_dir / "macro_aware_equity_2x.png", "Plain DCA vs macro-aware 2x overlays")

    metadata = {
        "analysis_dir": str(args.analysis_dir),
        "compare_dir": str(args.compare_dir),
        "common_start": common_start.date().isoformat(),
        "initial_capital": args.initial_capital,
        "monthly_contribution": args.monthly_contribution,
        "trading_cost_bps": args.trading_cost_bps,
        "borrow_rate": args.borrow_rate,
        "allocation_map": BASE_ALLOCATION_MAP,
        "notes": [
            "1x macro-aware strategies use fixed 25/70/100 target allocations for risk-off, neutral, and risk-on.",
            "2x strategies use existing reserve-and-deploy leverage logic with daily lagged signals.",
            "Consensus regime is risk-off if any core model is risk-off, risk-on only if logistic and GMM both agree on risk-on, otherwise neutral.",
            "Quarter-end diagnostics are descriptive and not used as a trading input in this package.",
        ],
    }
    (args.out_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        args.out_dir,
        current_environment=current_environment,
        accuracy=accuracy,
        expected_returns=expected_returns,
        strategies_1x=one_x_metrics,
        strategies_2x=two_x_metrics,
        threshold_sensitivity=logistic_threshold_sensitivity,
        allocation_sensitivity=allocation_sensitivity,
        seasonality=seasonality,
        feature_scorecard=feature_scorecard,
    )

    print(f"Wrote macro regime edge audit package to {args.out_dir}")
    print(f"Common start: {common_start.date()}")
    print("Top 1x strategies:")
    for _, row in one_x_metrics.head(4).iterrows():
        print(f"  {row['strategy']}: ${row['final_value']:,.0f}")
    print("Top 2x strategies:")
    for _, row in two_x_metrics.head(4).iterrows():
        print(f"  {row['strategy']}: ${row['final_value']:,.0f}")


if __name__ == "__main__":
    main()
