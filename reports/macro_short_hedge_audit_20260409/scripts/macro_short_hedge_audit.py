"""Short-horizon macro hedge audit with drawdown / CVaR targets.

This package builds a separate hedge overlay stack that only reduces net beta
to defensive levels rather than flipping fully net short. It follows the same
anti-leakage discipline as the long-side regime work:

- month-end model samples only
- purged chronological training
- lagged daily execution of monthly hedge states
- lagged macro releases preserved from the aligned dataset
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import macro_regime_edge_audit as long_audit
import qqq_macro_ml_regime_analysis as regime_analysis
import qqq_macro_walkforward_leverage as leverage
import qqq_macro_walkforward_model_compare as model_compare


ROOT = regime_analysis.ROOT
DEFAULT_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_COMPARE_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260409_monthly_equal"
DEFAULT_OUT_DIR = ROOT / "reports" / f"macro_short_hedge_audit_{pd.Timestamp.now().strftime('%Y%m%d')}"
REGIME_COLORS = {"unhedged": "#dff3e4", "hedge_0_6": "#f8f3d9", "hedge_0_3": "#f8d7da"}
HEDGE_BASE_BETAS = [1.0, 2.0, 3.0, 5.0]
HEDGE_FEATURES = regime_analysis.MODEL_FEATURES + [
    "is_quarter_end_month",
    "is_turn_of_quarter_month",
    "month_sin",
    "month_cos",
]


@dataclass
class HedgeResult:
    strategy: str
    curves: pd.DataFrame
    cashflows: list[tuple[pd.Timestamp, float]]
    final_value: float
    avg_target_beta: float
    transaction_costs: float
    borrow_costs: float
    unhedged_days: int
    hedge_0_6_days: int
    hedge_0_3_days: int
    unhedged_months: int
    hedge_0_6_months: int
    hedge_0_3_months: int
    total_external_contributed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=100.0)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument("--min-train-months", type=int, default=84)
    parser.add_argument("--random-state", type=int, default=regime_analysis.RANDOM_STATE)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--hedge-light-threshold", type=float, default=0.45)
    parser.add_argument("--hedge-strong-threshold", type=float, default=0.50)
    parser.add_argument("--hedge-light-beta", type=float, default=0.60)
    parser.add_argument("--hedge-strong-beta", type=float, default=0.30)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    return parser.parse_args()


def load_csv(path: Path, index_col: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if index_col is None:
        return pd.read_csv(path)
    return pd.read_csv(path, parse_dates=[index_col]).set_index(index_col).sort_index()


def forward_path_cvar(close: pd.Series, horizon: int, tail_fraction: float) -> pd.Series:
    values = close.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    tail_count = max(int(math.ceil(horizon * tail_fraction)), 1)
    for i in range(len(values) - horizon):
        path = values[i + 1 : i + horizon + 1] / values[i] - 1.0
        if len(path) == 0 or not np.isfinite(path).all():
            continue
        worst = np.sort(path)[:tail_count]
        out[i] = float(np.mean(worst))
    return pd.Series(out, index=close.index, name=f"qqq_fwd_{horizon}d_path_cvar_{int(tail_fraction * 100)}")


def add_hedge_targets(dataset: pd.DataFrame, tail_fraction: float) -> pd.DataFrame:
    df = dataset.copy()
    close = df["qqq_close"].astype(float)
    if "qqq_fwd_42d_return" not in df.columns:
        df["qqq_fwd_42d_return"] = close.shift(-42) / close - 1.0
        df["qqq_fwd_42d_min_return"] = regime_analysis._forward_min_return(close, 42)
        df["qqq_fwd_42d_end_date"] = pd.Series(close.index, index=close.index).shift(-42)
    df["qqq_fwd_21d_path_cvar20"] = forward_path_cvar(close, 21, tail_fraction)
    df["qqq_fwd_42d_path_cvar20"] = forward_path_cvar(close, 42, tail_fraction)
    df["hedge_strong_target"] = (
        (df["qqq_fwd_21d_min_return"] <= -0.10)
        | (df["qqq_fwd_21d_path_cvar20"] <= -0.075)
    ).astype(float)
    df.loc[df["qqq_fwd_21d_min_return"].isna(), "hedge_strong_target"] = np.nan
    df["hedge_light_target"] = (
        (df["hedge_strong_target"] == 1.0)
        | (df["qqq_fwd_42d_min_return"] <= -0.08)
        | (df["qqq_fwd_42d_path_cvar20"] <= -0.055)
    ).astype(float)
    df.loc[df["qqq_fwd_42d_min_return"].isna(), "hedge_light_target"] = np.nan
    df["is_quarter_end_month"] = df.index.month.isin([3, 6, 9, 12]).astype(float)
    df["is_turn_of_quarter_month"] = df.index.month.isin([1, 4, 7, 10]).astype(float)
    month_angle = 2.0 * math.pi * (df.index.month - 1) / 12.0
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    return df


def generalized_purged_train_test(
    sample: pd.DataFrame,
    target: str,
    horizon: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = sample.dropna(subset=[target]).copy()
    if valid.empty:
        return valid, valid
    split = max(int(len(valid) * (1.0 - test_size)), 1)
    test = valid.iloc[split:].copy()
    if test.empty:
        return valid.iloc[:split].copy(), test
    end_col = f"qqq_fwd_{horizon}d_end_date"
    if end_col in valid.columns:
        train = valid.iloc[:split].loc[valid.iloc[:split][end_col] < test.index[0]].copy()
    else:
        train = valid.iloc[: max(split - max(horizon // 21, 1), 1)].copy()
    return train, test


def classifier_pipeline(model_name: str, random_state: int, rf_estimators: int):
    if model_name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
        )
    if model_name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=rf_estimators,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unsupported classifier model: {model_name}")


def regressor_pipeline(model_name: str, random_state: int):
    if model_name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if model_name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unsupported regressor model: {model_name}")


def evaluate_hedge_models(
    sample: pd.DataFrame,
    features: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    regression_targets = {
        "qqq_fwd_21d_min_return": 21,
        "qqq_fwd_42d_min_return": 42,
        "qqq_fwd_21d_path_cvar20": 21,
        "qqq_fwd_42d_path_cvar20": 42,
    }
    for target, horizon in regression_targets.items():
        train, test = generalized_purged_train_test(sample, target, horizon, test_size)
        train = train.dropna(subset=[target])
        test = test.dropna(subset=[target])
        if len(train) < 60 or len(test) < 24:
            continue
        x_train, y_train = train[features], train[target]
        x_test, y_test = test[features], test[target]
        for model_name in ["ridge", "random_forest"]:
            model = regressor_pipeline(model_name, random_state)
            model.fit(x_train, y_train)
            pred = pd.Series(model.predict(x_test), index=x_test.index)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "train_start": train.index.min(),
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "train_n": len(train),
                    "test_n": len(test),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "r2": float(r2_score(y_test, pred)),
                    "spearman_pred_actual": float(pred.corr(y_test, method="spearman")) if len(pred) > 2 else np.nan,
                }
            )
            if model_name == "ridge":
                reg = model.named_steps["ridgecv"]
                for feature, coef in zip(features, reg.coef_):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(coef),
                            "importance_std": np.nan,
                        }
                    )
            else:
                perm = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring="neg_mean_absolute_error",
                )
                for feature, mean_imp, std_imp in zip(features, perm.importances_mean, perm.importances_std):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(mean_imp),
                            "importance_std": float(std_imp),
                        }
                    )

    classification_targets = {"hedge_light_target": 42, "hedge_strong_target": 21}
    for target, horizon in classification_targets.items():
        train, test = generalized_purged_train_test(sample, target, horizon, test_size)
        train = train.dropna(subset=[target])
        test = test.dropna(subset=[target])
        if len(train) < 60 or len(test) < 24 or train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        x_train, y_train = train[features], train[target].astype(int)
        x_test, y_test = test[features], test[target].astype(int)
        for model_name in ["logistic", "random_forest"]:
            model = classifier_pipeline(model_name, random_state, rf_estimators=300)
            model.fit(x_train, y_train)
            proba = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index)
            pred = (proba >= 0.50).astype(int)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "train_start": train.index.min(),
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "train_n": len(train),
                    "test_n": len(test),
                    "event_rate_train": float(y_train.mean()),
                    "event_rate_test": float(y_test.mean()),
                    "auc": float(roc_auc_score(y_test, proba)),
                    "average_precision": float(average_precision_score(y_test, proba)),
                    "brier": float(brier_score_loss(y_test, proba)),
                    "balanced_accuracy_at_50pct": float(balanced_accuracy_score(y_test, pred)),
                    "precision_at_50pct": float(precision_score(y_test, pred, zero_division=0)),
                    "recall_at_50pct": float(recall_score(y_test, pred, zero_division=0)),
                }
            )
            if model_name == "logistic":
                clf = model.named_steps["logisticregression"]
                for feature, coef in zip(features, clf.coef_[0]):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(coef),
                            "importance_std": np.nan,
                        }
                    )
            else:
                perm = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring="roc_auc",
                )
                for feature, mean_imp, std_imp in zip(features, perm.importances_mean, perm.importances_std):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(mean_imp),
                            "importance_std": float(std_imp),
                        }
                    )

        latest = sample.iloc[[-1]]
        all_train = sample.dropna(subset=[target]).copy()
        if len(all_train) >= 60 and all_train[target].nunique() >= 2:
            model = classifier_pipeline("logistic", random_state, rf_estimators=300)
            model.fit(all_train[features], all_train[target].astype(int))
            current[f"current_{target}_probability"] = float(model.predict_proba(latest[features])[:, 1][0])

    return pd.DataFrame(rows), pd.DataFrame(importance_rows), current


def hedge_state_from_probabilities(
    light_probability: float,
    strong_probability: float,
    light_threshold: float,
    strong_threshold: float,
) -> str:
    if strong_probability >= strong_threshold:
        return "hedge_0_3"
    if light_probability >= light_threshold:
        return "hedge_0_6"
    return "unhedged"


def build_walkforward_monthly_hedge_states(
    sample: pd.DataFrame,
    features: list[str],
    *,
    model_name: str,
    min_train_months: int,
    light_threshold: float,
    strong_threshold: float,
    random_state: int,
    rf_estimators: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=sample.index)
    out["model_name"] = model_name
    out["hedge_state"] = pd.Series(index=sample.index, dtype=object)
    out["hedge_light_probability"] = np.nan
    out["hedge_strong_probability"] = np.nan
    out["train_n_light"] = np.nan
    out["train_n_strong"] = np.nan
    out["train_start"] = pd.NaT
    out["train_end"] = pd.NaT

    use_features = regime_analysis.available_features(sample, features, min_non_na=min_train_months)
    if len(use_features) < 6:
        return out

    for i, date in enumerate(sample.index):
        current = sample.loc[[date], use_features].replace([np.inf, -np.inf], np.nan)
        if current.isna().any(axis=None):
            continue
        train = sample.iloc[:i].copy().replace([np.inf, -np.inf], np.nan)

        light_train = train.dropna(subset=use_features + ["hedge_light_target"]).copy()
        strong_train = train.dropna(subset=use_features + ["hedge_strong_target"]).copy()
        if "qqq_fwd_42d_end_date" in light_train.columns:
            light_train = light_train.loc[pd.to_datetime(light_train["qqq_fwd_42d_end_date"], errors="coerce") < date].copy()
        if "qqq_fwd_21d_end_date" in strong_train.columns:
            strong_train = strong_train.loc[pd.to_datetime(strong_train["qqq_fwd_21d_end_date"], errors="coerce") < date].copy()

        if (
            len(light_train) < min_train_months
            or len(strong_train) < min_train_months
            or light_train["hedge_light_target"].nunique() < 2
            or strong_train["hedge_strong_target"].nunique() < 2
        ):
            continue

        light_model = classifier_pipeline(model_name, random_state, rf_estimators)
        light_model.fit(light_train[use_features], light_train["hedge_light_target"].astype(int))
        light_probability = float(light_model.predict_proba(current[use_features])[:, 1][0])

        strong_model = classifier_pipeline(model_name, random_state, rf_estimators)
        strong_model.fit(strong_train[use_features], strong_train["hedge_strong_target"].astype(int))
        strong_probability = float(strong_model.predict_proba(current[use_features])[:, 1][0])

        out.loc[date, "hedge_light_probability"] = light_probability
        out.loc[date, "hedge_strong_probability"] = strong_probability
        out.loc[date, "hedge_state"] = hedge_state_from_probabilities(
            light_probability, strong_probability, light_threshold, strong_threshold
        )
        out.loc[date, "train_n_light"] = int(len(light_train))
        out.loc[date, "train_n_strong"] = int(len(strong_train))
        out.loc[date, "train_start"] = min(light_train.index.min(), strong_train.index.min())
        out.loc[date, "train_end"] = max(light_train.index.max(), strong_train.index.max())
    return out


def build_consensus_hedge_state(monthly_states: dict[str, pd.Series]) -> pd.Series:
    joined = pd.DataFrame(monthly_states)
    out = pd.Series(index=joined.index, dtype=object)
    strong = joined.eq("hedge_0_3").all(axis=1)
    light = joined.isin(["hedge_0_3", "hedge_0_6"]).any(axis=1)
    out.loc[strong] = "hedge_0_3"
    out.loc[~strong & light] = "hedge_0_6"
    out.loc[out.isna() & joined.notna().all(axis=1)] = "unhedged"
    return out.rename("consensus")


def hedge_state_to_beta(
    signal: pd.Series,
    *,
    base_beta: float,
    light_beta: float,
    strong_beta: float,
) -> pd.Series:
    out = pd.Series(np.nan, index=signal.index, dtype=float)
    out.loc[signal.eq("unhedged")] = base_beta
    out.loc[signal.eq("hedge_0_6")] = light_beta
    out.loc[signal.eq("hedge_0_3")] = strong_beta
    return out


def monthly_signal_to_daily(monthly_signal: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    return monthly_signal.reindex(daily_index).ffill().shift(1)


def contribution_bucket(date: pd.Timestamp) -> pd.Period:
    return pd.Timestamp(date).to_period("M")


def simulate_beta_overlay(
    close: pd.Series,
    state_signal: pd.Series,
    *,
    strategy: str,
    base_beta: float,
    light_beta: float,
    strong_beta: float,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
    borrow_rate: float,
) -> HedgeResult:
    signal = state_signal.reindex(close.index)
    if signal.dropna().empty:
        raise RuntimeError("No lagged hedge signal available for simulation.")
    price = close.loc[signal.dropna().index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
    target_beta = hedge_state_to_beta(signal, base_beta=base_beta, light_beta=light_beta, strong_beta=strong_beta)
    cost_rate = trading_cost_bps / 10_000.0

    equity = float(initial_capital)
    transaction_costs = equity * abs(float(target_beta.iloc[0])) * cost_rate
    equity -= transaction_costs
    borrow_costs = 0.0
    curves: list[dict[str, Any]] = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(price.index[0], -float(initial_capital))]
    total_external = float(initial_capital)
    current_beta = float(target_beta.iloc[0])
    prev_price = float(price.iloc[0])
    prev_bucket = contribution_bucket(price.index[0])

    for i, (date, px) in enumerate(price.items()):
        px = float(px)
        state = str(signal.loc[date])
        beta = float(target_beta.loc[date])
        if i > 0:
            daily_borrow = max(abs(current_beta) - 1.0, 0.0) * borrow_rate / 252.0
            borrow_costs += equity * daily_borrow
            day_return = px / prev_price - 1.0
            equity = max(0.0, equity * (1.0 + current_beta * day_return - daily_borrow))

        bucket = contribution_bucket(date)
        if i > 0 and bucket != prev_bucket:
            contribution = float(monthly_contribution)
            total_external += contribution
            cashflows.append((date, -contribution))
            contribution_cost = contribution * abs(beta) * cost_rate
            transaction_costs += contribution_cost
            equity += max(0.0, contribution - contribution_cost)
            prev_bucket = bucket

        if i > 0 and not np.isclose(beta, current_beta):
            rebalance_cost = abs(beta - current_beta) * equity * cost_rate
            transaction_costs += rebalance_cost
            equity = max(0.0, equity - rebalance_cost)

        curves.append(
            {
                "date": date,
                "strategy": strategy,
                "total_value": equity,
                "target_beta": beta,
                "hedge_state": state,
                "base_beta": base_beta,
            }
        )
        current_beta = beta
        prev_price = px

    curve_df = pd.DataFrame(curves).set_index("date")
    state_counts = curve_df["hedge_state"].value_counts()
    month_counts = curve_df.groupby(curve_df.index.to_period("M"))["hedge_state"].last().value_counts()
    cashflows.append((curve_df.index[-1], float(curve_df["total_value"].iloc[-1])))
    return HedgeResult(
        strategy=strategy,
        curves=curve_df,
        cashflows=cashflows,
        final_value=float(curve_df["total_value"].iloc[-1]),
        avg_target_beta=float(curve_df["target_beta"].mean()),
        transaction_costs=float(transaction_costs),
        borrow_costs=float(borrow_costs),
        unhedged_days=int(state_counts.get("unhedged", 0)),
        hedge_0_6_days=int(state_counts.get("hedge_0_6", 0)),
        hedge_0_3_days=int(state_counts.get("hedge_0_3", 0)),
        unhedged_months=int(month_counts.get("unhedged", 0)),
        hedge_0_6_months=int(month_counts.get("hedge_0_6", 0)),
        hedge_0_3_months=int(month_counts.get("hedge_0_3", 0)),
        total_external_contributed=float(total_external),
    )


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    return regime_analysis._xirr(cashflows)


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    return float((series / peak - 1.0).min())


def hedge_metrics_row(result: HedgeResult, base_beta: float, baseline_final: float | None = None) -> dict[str, Any]:
    equity = result.curves["total_value"].astype(float)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    row = {
        "strategy": result.strategy,
        "base_beta": base_beta,
        "start_date": equity.index[0].date().isoformat(),
        "end_date": equity.index[-1].date().isoformat(),
        "final_value": float(equity.iloc[-1]),
        "xirr": float(xirr(result.cashflows)),
        "time_weighted_total_return": total_return,
        "time_weighted_cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "avg_target_beta": result.avg_target_beta,
        "total_external_contributed": result.total_external_contributed,
        "final_multiple_on_contributed": float(equity.iloc[-1] / result.total_external_contributed),
        "transaction_costs": result.transaction_costs,
        "borrow_costs": result.borrow_costs,
        "unhedged_days": result.unhedged_days,
        "hedge_0_6_days": result.hedge_0_6_days,
        "hedge_0_3_days": result.hedge_0_3_days,
        "unhedged_months": result.unhedged_months,
        "hedge_0_6_months": result.hedge_0_6_months,
        "hedge_0_3_months": result.hedge_0_3_months,
    }
    row["final_delta_vs_same_beta_baseline"] = np.nan if baseline_final is None else float(row["final_value"] - baseline_final)
    return row


def forward_hedge_stats(sample: pd.DataFrame, prediction: pd.Series, model_name: str) -> pd.DataFrame:
    merged = sample.join(prediction.rename("predicted_state"), how="left").dropna(subset=["predicted_state"])
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for state, group in merged.groupby("predicted_state", sort=False):
        rows.append(
            {
                "model_name": model_name,
                "predicted_state": state,
                "n_months": int(len(group)),
                "avg_fwd_21d_return": float(group["qqq_fwd_21d_return"].mean()),
                "avg_fwd_42d_return": float(group["qqq_fwd_42d_return"].mean()),
                "avg_fwd_21d_min_return": float(group["qqq_fwd_21d_min_return"].mean()),
                "avg_fwd_42d_min_return": float(group["qqq_fwd_42d_min_return"].mean()),
                "avg_fwd_21d_path_cvar20": float(group["qqq_fwd_21d_path_cvar20"].mean()),
                "avg_fwd_42d_path_cvar20": float(group["qqq_fwd_42d_path_cvar20"].mean()),
                "strong_event_rate": float(group["hedge_strong_target"].mean()),
                "light_event_rate": float(group["hedge_light_target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def hedge_state_confusion(sample: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    realized = pd.Series("unhedged", index=sample.index, dtype=object)
    realized.loc[sample["hedge_light_target"].eq(1.0)] = "hedge_0_6"
    realized.loc[sample["hedge_strong_target"].eq(1.0)] = "hedge_0_3"
    aligned = pd.DataFrame({"actual": realized, "predicted": prediction}).dropna()
    if aligned.empty:
        return pd.DataFrame()
    labels = ["unhedged", "hedge_0_6", "hedge_0_3"]
    return pd.crosstab(
        pd.Categorical(aligned["actual"], categories=labels),
        pd.Categorical(aligned["predicted"], categories=labels),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )


def seasonality_table(sample: pd.DataFrame, prediction: pd.Series, model_name: str) -> pd.DataFrame:
    merged = sample.join(prediction.rename("predicted_state"), how="left").dropna(subset=["predicted_state"])
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    merged["is_quarter_end"] = merged["is_quarter_end_month"].astype(bool)
    merged["is_turn_of_quarter"] = merged["is_turn_of_quarter_month"].astype(bool)
    for bucket in ["is_quarter_end", "is_turn_of_quarter"]:
        for (state, flag), group in merged.groupby(["predicted_state", bucket], sort=False):
            rows.append(
                {
                    "model_name": model_name,
                    "predicted_state": state,
                    "seasonality_bucket": bucket,
                    "flag": bool(flag),
                    "n_months": int(len(group)),
                    "avg_fwd_21d_return": float(group["qqq_fwd_21d_return"].mean()),
                    "avg_fwd_42d_min_return": float(group["qqq_fwd_42d_min_return"].mean()),
                    "avg_fwd_42d_path_cvar20": float(group["qqq_fwd_42d_path_cvar20"].mean()),
                    "strong_event_rate": float(group["hedge_strong_target"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_hedge_impact_tests(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    outcomes = [
        ("qqq_fwd_21d_min_return", 21),
        ("qqq_fwd_42d_min_return", 42),
        ("qqq_fwd_21d_path_cvar20", 21),
        ("qqq_fwd_42d_path_cvar20", 42),
    ]
    for outcome, horizon in outcomes:
        table = regime_analysis.newey_west_ols(sample[outcome], sample[features], lags=max(1, int(math.ceil(horizon / 21))))
        if table.empty:
            continue
        table = table[table["term"] != "intercept"].copy()
        table["horizon_days"] = horizon
        table["outcome"] = outcome
        table["coef_pct_points_per_1sd"] = table["coef"] * 100.0
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["q_value_bh_fdr"] = out.groupby("outcome", group_keys=False)["p_value"].apply(regime_analysis._bh_fdr)
    return out.sort_values(["outcome", "p_value", "term"])


def build_feature_scorecard(impact: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = sorted(set(impact["term"]).union(set(importance["feature"])))
    targets = [
        ("qqq_fwd_21d_min_return", "ridge", "ridge_min_21"),
        ("qqq_fwd_42d_min_return", "ridge", "ridge_min_42"),
        ("qqq_fwd_21d_path_cvar20", "ridge", "ridge_cvar_21"),
        ("qqq_fwd_42d_path_cvar20", "ridge", "ridge_cvar_42"),
        ("hedge_light_target", "logistic", "logistic_light"),
        ("hedge_strong_target", "logistic", "logistic_strong"),
        ("hedge_light_target", "random_forest", "rf_light"),
        ("hedge_strong_target", "random_forest", "rf_strong"),
    ]
    for feature in features:
        if feature == "intercept":
            continue
        row: dict[str, Any] = {"feature": feature, "label": regime_analysis.feature_label(feature)}
        for outcome in ["qqq_fwd_21d_min_return", "qqq_fwd_42d_min_return", "qqq_fwd_21d_path_cvar20", "qqq_fwd_42d_path_cvar20"]:
            subset = impact[(impact["term"] == feature) & (impact["outcome"] == outcome)]
            row[f"{outcome}_coef_pp_per_1sd"] = float(subset.iloc[0]["coef_pct_points_per_1sd"]) if not subset.empty else np.nan
            row[f"{outcome}_q_value"] = float(subset.iloc[0]["q_value_bh_fdr"]) if not subset.empty else np.nan
        for target, model, prefix in targets:
            subset = importance[(importance["target"] == target) & (importance["model"] == model) & (importance["feature"] == feature)]
            row[f"{prefix}_importance"] = float(subset.iloc[0]["importance_mean"]) if not subset.empty else np.nan
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard
    scorecard["valuation_flag"] = scorecard["feature"].str.contains("cape|buffett_indicator", case=False, regex=True)
    scorecard["stress_flag"] = scorecard["feature"].str.contains("vix|hy_oas|nfci|unemployment|cpi|shock", case=False, regex=True)
    scorecard["trend_flag"] = scorecard["feature"].str.contains("sma|drawdown|qqq_", case=False, regex=True)
    return scorecard.sort_values(["qqq_fwd_42d_path_cvar20_q_value", "qqq_fwd_42d_min_return_q_value", "feature"], ascending=[True, True, True])


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
        compare_dir / "walkforward_model_regimes_monthly.csv",
        compare_dir / "walkforward_model_signals_daily.csv",
        compare_dir / "walkforward_model_compare_leverage_metrics.csv",
    ]:
        if src.exists():
            shutil.copy2(src, data_dir / src.name)
    for src in [
        ROOT / "cache" / "download_daily_macro_data.py",
        ROOT / "tools" / "qqq_macro_ml_regime_analysis.py",
        ROOT / "tools" / "qqq_macro_walkforward_model_compare.py",
        ROOT / "tools" / "qqq_macro_walkforward_leverage.py",
        Path(__file__),
    ]:
        if src.exists():
            shutil.copy2(src, scripts_dir / src.name)


def plot_equity(curves: pd.DataFrame, out_path: Path, title: str) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in curves.columns:
        ax.plot(curves.index, curves[column].astype(float), linewidth=1.4, label=column)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylabel("Account value, USD")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hedge_state_chart(close: pd.Series, state_signal: pd.Series, out_path: Path, title: str) -> None:
    signal = state_signal.reindex(close.index)
    signal = signal.dropna()
    if signal.empty:
        return
    price = close.loc[signal.index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
    level = signal.map({"hedge_0_3": 0.0, "hedge_0_6": 1.0, "unhedged": 2.0})
    fig, (ax_price, ax_state) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15]},
    )
    current_state = None
    start = None
    prev = None
    for date, state in signal.items():
        if state != current_state:
            if current_state is not None and start is not None and prev is not None:
                ax_price.axvspan(start, prev, color=REGIME_COLORS[current_state], alpha=0.55, linewidth=0)
                ax_state.axvspan(start, prev, color=REGIME_COLORS[current_state], alpha=0.55, linewidth=0)
            current_state = state
            start = pd.Timestamp(date)
        prev = pd.Timestamp(date)
    if current_state is not None and start is not None and prev is not None:
        ax_price.axvspan(start, prev, color=REGIME_COLORS[current_state], alpha=0.55, linewidth=0)
        ax_state.axvspan(start, prev, color=REGIME_COLORS[current_state], alpha=0.55, linewidth=0)
    ax_price.plot(price.index, price.astype(float), color="#0f172a", linewidth=1.7, label="QQQ close")
    ax_price.set_title(title)
    ax_price.set_ylabel("QQQ close")
    ax_price.set_yscale("log")
    ax_price.grid(alpha=0.25)
    ax_price.legend(
        handles=[
            Patch(facecolor=REGIME_COLORS["unhedged"], edgecolor="none", label="Unhedged"),
            Patch(facecolor=REGIME_COLORS["hedge_0_6"], edgecolor="none", label="Hedge to 0.6 beta"),
            Patch(facecolor=REGIME_COLORS["hedge_0_3"], edgecolor="none", label="Hedge to 0.3 beta"),
        ],
        loc="upper left",
        ncol=3,
    )
    ax_state.step(level.index, level.astype(float), where="post", color="#111827", linewidth=1.4)
    ax_state.set_ylim(-0.5, 2.5)
    ax_state.set_yticks([0.0, 1.0, 2.0], labels=["0.3", "0.6", "1.0+"])
    ax_state.set_ylabel("Net beta")
    ax_state.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    metadata: dict[str, Any],
    current_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    expected_stats: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    feature_scorecard: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Short-Horizon Hedge Audit",
        "",
        "## Anti-Leakage Discipline",
        "",
        "- Hedge targets are built from future 21-day and 42-day path pain, but model features use only information available at each month-end.",
        "- Walk-forward hedge models train on earlier month-end rows only.",
        "- Any row whose forward path window overlaps the prediction date is purged from training.",
        "- Daily hedge execution uses one-day-lagged monthly states.",
        "- Slow macro releases remain lagged because the source dataset already enforces release discipline.",
        "",
        "## Target Design",
        "",
    ]
    if not target_summary.empty:
        row = target_summary.iloc[0]
        lines.extend(
            [
                f"- `hedge_strong_target`: next 21d min return <= -10% or 21d path-CVaR20 <= -7.5%; event rate `{row['hedge_strong_rate'] * 100:.1f}%`.",
                f"- `hedge_light_target`: strong target or next 42d min return <= -8% or 42d path-CVaR20 <= -5.5%; event rate `{row['hedge_light_rate'] * 100:.1f}%`.",
                f"- Average next 21d min return = `{row['avg_21d_min_return'] * 100:.1f}%`; average next 42d path-CVaR20 = `{row['avg_42d_path_cvar20'] * 100:.1f}%`.",
            ]
        )
    lines.extend(["", "## Current Hedge Read", ""])
    if not current_summary.empty:
        row = current_summary.iloc[0]
        lines.extend(
            [
                f"- As of `{pd.Timestamp(row['as_of']).date()}`, logistic strong-hedge probability = `{row['logistic_hedge_strong_probability'] * 100:.1f}%` and light-hedge probability = `{row['logistic_hedge_light_probability'] * 100:.1f}%`.",
                f"- Random forest strong-hedge probability = `{row['random_forest_hedge_strong_probability'] * 100:.1f}%` and light-hedge probability = `{row['random_forest_hedge_light_probability'] * 100:.1f}%`.",
                f"- Consensus hedge state = `{row['consensus_hedge_state']}`.",
            ]
        )
    lines.extend(["", "## Validation", ""])
    if not validation_metrics.empty:
        lines.append("| Target | Model | Train N | Test N | AUC | Avg Precision | Brier | Precision@50 | Recall@50 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in validation_metrics[validation_metrics["target"].isin(["hedge_light_target", "hedge_strong_target"])].iterrows():
            lines.append(
                f"| {row['target']} | {row['model']} | {int(row['train_n'])} | {int(row['test_n'])} | {row.get('auc', np.nan):.3f} | "
                f"{row.get('average_precision', np.nan):.3f} | {row.get('brier', np.nan):.3f} | "
                f"{row.get('precision_at_50pct', np.nan):.3f} | {row.get('recall_at_50pct', np.nan):.3f} |"
            )
    lines.extend(["", "## Expected Path Pain By Predicted State", ""])
    if not expected_stats.empty:
        lines.append("| Model | State | N | Avg 21D Return | Avg 42D Return | Avg 21D Min | Avg 42D Min | 42D CVaR20 | Strong Event Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in expected_stats.sort_values(["model_name", "predicted_state"]).iterrows():
            lines.append(
                f"| {row['model_name']} | {row['predicted_state']} | {int(row['n_months'])} | {row['avg_fwd_21d_return'] * 100:.1f}% | "
                f"{row['avg_fwd_42d_return'] * 100:.1f}% | {row['avg_fwd_21d_min_return'] * 100:.1f}% | "
                f"{row['avg_fwd_42d_min_return'] * 100:.1f}% | {row['avg_fwd_42d_path_cvar20'] * 100:.1f}% | "
                f"{row['strong_event_rate'] * 100:.1f}% |"
            )
    lines.extend(["", "## Strategy Results", ""])
    if not strategy_metrics.empty:
        lines.append("| Strategy | Base Beta | Final Value | XIRR | CAGR | Max DD | Avg Beta | Delta vs Same-Beta Baseline |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in strategy_metrics.sort_values(["base_beta", "final_value"], ascending=[True, False]).iterrows():
            lines.append(
                f"| {row['strategy']} | {row['base_beta']:.1f} | ${row['final_value']:,.0f} | {row['xirr'] * 100:.1f}% | "
                f"{row['time_weighted_cagr'] * 100:.1f}% | {row['max_drawdown'] * 100:.1f}% | {row['avg_target_beta']:.2f} | "
                f"${row['final_delta_vs_same_beta_baseline']:,.0f} |"
            )
    lines.extend(["", "## Sensitivity", ""])
    if not threshold_sensitivity.empty:
        best = threshold_sensitivity.sort_values("final_value", ascending=False).iloc[0]
        lines.append(
            f"- Best ex-post hedge configuration in the audited grid: base beta `{best['base_beta']:.1f}`, light threshold `{best['hedge_light_threshold']:.2f}`, strong threshold `{best['hedge_strong_threshold']:.2f}`, final value `${best['final_value']:,.0f}`, max DD `{best['max_drawdown'] * 100:.1f}%`."
        )
    lines.extend(["", "## Feature Takeaways", ""])
    focus = feature_scorecard[
        feature_scorecard["feature"].isin(
            [
                "cape_level",
                "buffett_indicator_proxy_level",
                "vix_level",
                "hy_oas_level",
                "nfci_level",
                "qqq_sma65",
                "qqq_sma222",
                "unemployment_rate_pct",
                "cpi_yoy_pct",
                "latent_sentiment_index",
            ]
        )
    ].copy()
    if not focus.empty:
        lines.append("| Feature | 42D Min q-value | 42D CVaR q-value | Logistic Light | Logistic Strong |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in focus.sort_values("feature").iterrows():
            lines.append(
                f"| {row['label']} | {row['qqq_fwd_42d_min_return_q_value']:.3f} | {row['qqq_fwd_42d_path_cvar20_q_value']:.3f} | "
                f"{row['logistic_light_importance']:.3f} | {row['logistic_strong_importance']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## View",
            "",
            "- A hedge model should be judged on path pain, not just forward return. Predicting drawdown and path-CVaR is more aligned with actual hedging decisions.",
            "- The defensive edge will likely come from avoiding the worst path episodes while keeping most of the market beta in ordinary environments.",
            "- Quarter-end and turn-of-quarter effects are worth monitoring, but they should remain context features rather than dominant hedge triggers.",
            "- A hedge overlay that only reduces net beta to 0.6 or 0.3 is structurally safer than a fully short tactical model, because false positives remain expensive.",
            "- The honest benchmark is the same-beta unhedged baseline, not just plain 1x DCA. A useful hedge should improve path risk without giving away too much terminal wealth.",
        ]
    )
    out_dir.joinpath("macro_short_hedge_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = leverage.load_dataset(args.analysis_dir / "aligned_daily_dataset.csv")
    dataset = add_hedge_targets(dataset, args.tail_fraction)
    sample = regime_analysis.month_end_sample(dataset)
    feature_candidates = regime_analysis.available_features(sample, HEDGE_FEATURES, min_non_na=args.min_train_months)
    ols_features, vif_drop = regime_analysis.select_vif_filtered_features(sample, feature_candidates, max_vif=20.0)

    impact = run_hedge_impact_tests(sample, ols_features)
    validation_metrics, feature_importance, current_probs = evaluate_hedge_models(
        sample,
        feature_candidates,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    monthly_predictions: dict[str, pd.DataFrame] = {}
    for model_name in ["logistic", "random_forest"]:
        monthly_predictions[model_name] = build_walkforward_monthly_hedge_states(
            sample,
            feature_candidates,
            model_name=model_name,
            min_train_months=args.min_train_months,
            light_threshold=args.hedge_light_threshold,
            strong_threshold=args.hedge_strong_threshold,
            random_state=args.random_state,
            rf_estimators=args.rf_estimators,
        )

    monthly_state_series = {name: frame["hedge_state"].rename(name) for name, frame in monthly_predictions.items()}
    consensus_state = build_consensus_hedge_state(monthly_state_series)

    daily_state_map = {
        name: monthly_signal_to_daily(state, dataset.index) for name, state in monthly_state_series.items()
    }
    daily_state_map["consensus"] = monthly_signal_to_daily(consensus_state, dataset.index)

    common_start = min(signal.dropna().index.min() for signal in daily_state_map.values() if not signal.dropna().empty)
    close = dataset.loc[dataset.index >= common_start, "qqq_close"].astype(float)

    expected_stats = pd.concat(
        [
            forward_hedge_stats(sample, monthly_state_series["logistic"], "logistic"),
            forward_hedge_stats(sample, monthly_state_series["random_forest"], "random_forest"),
            forward_hedge_stats(sample, consensus_state, "consensus"),
        ],
        ignore_index=True,
    )

    seasonality = pd.concat(
        [
            seasonality_table(sample, monthly_state_series["logistic"], "logistic"),
            seasonality_table(sample, monthly_state_series["random_forest"], "random_forest"),
            seasonality_table(sample, consensus_state, "consensus"),
        ],
        ignore_index=True,
    )

    strategy_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    baseline_curves: dict[float, pd.Series] = {}
    baseline_finals: dict[float, float] = {}
    for base_beta in HEDGE_BASE_BETAS:
        baseline = simulate_beta_overlay(
            close,
            pd.Series("unhedged", index=close.index),
            strategy=f"baseline_beta_{base_beta:.0f}x",
            base_beta=base_beta,
            light_beta=args.hedge_light_beta,
            strong_beta=args.hedge_strong_beta,
            initial_capital=args.initial_capital,
            monthly_contribution=args.monthly_contribution,
            trading_cost_bps=args.trading_cost_bps,
            borrow_rate=args.borrow_rate,
        )
        baseline_curves[base_beta] = baseline.curves["total_value"].rename(baseline.strategy)
        baseline_finals[base_beta] = baseline.final_value
        strategy_rows.append(hedge_metrics_row(baseline, base_beta, baseline.final_value))
        curve_frames.append(baseline.curves.reset_index())
        for model_name, signal in daily_state_map.items():
            result = simulate_beta_overlay(
                close,
                signal.loc[signal.index >= common_start],
                strategy=f"{model_name}_hedge_base_{base_beta:.0f}x",
                base_beta=base_beta,
                light_beta=args.hedge_light_beta,
                strong_beta=args.hedge_strong_beta,
                initial_capital=args.initial_capital,
                monthly_contribution=args.monthly_contribution,
                trading_cost_bps=args.trading_cost_bps,
                borrow_rate=args.borrow_rate,
            )
            strategy_rows.append(hedge_metrics_row(result, base_beta, baseline_finals[base_beta]))
            curve_frames.append(result.curves.reset_index())

    strategy_metrics = pd.DataFrame(strategy_rows).sort_values(["base_beta", "final_value"], ascending=[True, False])
    strategy_curves = pd.concat(curve_frames, ignore_index=True)

    threshold_rows: list[dict[str, Any]] = []
    for light_threshold in [0.40, 0.45, 0.50]:
        for strong_threshold in [0.45, 0.50, 0.55]:
            if strong_threshold < light_threshold:
                continue
            threshold_states = build_walkforward_monthly_hedge_states(
                sample,
                feature_candidates,
                model_name="logistic",
                min_train_months=args.min_train_months,
                light_threshold=light_threshold,
                strong_threshold=strong_threshold,
                random_state=args.random_state,
                rf_estimators=args.rf_estimators,
            )["hedge_state"]
            daily_signal = monthly_signal_to_daily(threshold_states, dataset.index).loc[dataset.index >= common_start]
            for base_beta in [1.0, 2.0]:
                result = simulate_beta_overlay(
                    close,
                    daily_signal,
                    strategy="logistic_threshold_sensitivity",
                    base_beta=base_beta,
                    light_beta=args.hedge_light_beta,
                    strong_beta=args.hedge_strong_beta,
                    initial_capital=args.initial_capital,
                    monthly_contribution=args.monthly_contribution,
                    trading_cost_bps=args.trading_cost_bps,
                    borrow_rate=args.borrow_rate,
                )
                row = hedge_metrics_row(result, base_beta, baseline_finals[base_beta])
                row["hedge_light_threshold"] = light_threshold
                row["hedge_strong_threshold"] = strong_threshold
                threshold_rows.append(row)
    threshold_sensitivity = pd.DataFrame(threshold_rows).sort_values("final_value", ascending=False)

    feature_scorecard = build_feature_scorecard(impact, feature_importance)
    target_summary = pd.DataFrame(
        [
            {
                "hedge_strong_rate": float(sample["hedge_strong_target"].mean()),
                "hedge_light_rate": float(sample["hedge_light_target"].mean()),
                "avg_21d_min_return": float(sample["qqq_fwd_21d_min_return"].mean()),
                "avg_42d_path_cvar20": float(sample["qqq_fwd_42d_path_cvar20"].mean()),
            }
        ]
    )
    current_summary = pd.DataFrame(
        [
            {
                "as_of": dataset.index.max(),
                "logistic_hedge_light_probability": float(monthly_predictions["logistic"]["hedge_light_probability"].dropna().iloc[-1]) if monthly_predictions["logistic"]["hedge_light_probability"].notna().any() else np.nan,
                "logistic_hedge_strong_probability": float(monthly_predictions["logistic"]["hedge_strong_probability"].dropna().iloc[-1]) if monthly_predictions["logistic"]["hedge_strong_probability"].notna().any() else np.nan,
                "random_forest_hedge_light_probability": float(monthly_predictions["random_forest"]["hedge_light_probability"].dropna().iloc[-1]) if monthly_predictions["random_forest"]["hedge_light_probability"].notna().any() else np.nan,
                "random_forest_hedge_strong_probability": float(monthly_predictions["random_forest"]["hedge_strong_probability"].dropna().iloc[-1]) if monthly_predictions["random_forest"]["hedge_strong_probability"].notna().any() else np.nan,
                "consensus_hedge_state": str(consensus_state.dropna().iloc[-1]) if consensus_state.notna().any() else "unknown",
            }
        ]
    )

    copy_inputs(args.out_dir, args.analysis_dir, args.compare_dir)
    sample.to_csv(args.out_dir / "hedge_month_end_sample.csv", index_label="date")
    dataset.to_csv(args.out_dir / "hedge_aligned_daily_dataset.csv", index_label="date")
    impact.to_csv(args.out_dir / "hedge_ols_newey_west_impact.csv", index=False)
    validation_metrics.to_csv(args.out_dir / "hedge_model_validation_metrics.csv", index=False)
    feature_importance.to_csv(args.out_dir / "hedge_model_feature_importance.csv", index=False)
    feature_scorecard.to_csv(args.out_dir / "hedge_feature_scorecard.csv", index=False)
    expected_stats.to_csv(args.out_dir / "hedge_expected_path_stats.csv", index=False)
    seasonality.to_csv(args.out_dir / "hedge_seasonality.csv", index=False)
    strategy_metrics.to_csv(args.out_dir / "hedge_strategy_metrics.csv", index=False)
    strategy_curves.to_csv(args.out_dir / "hedge_strategy_curves.csv", index=False)
    threshold_sensitivity.to_csv(args.out_dir / "hedge_threshold_sensitivity.csv", index=False)
    current_summary.to_csv(args.out_dir / "hedge_current_summary.csv", index=False)
    target_summary.to_csv(args.out_dir / "hedge_target_summary.csv", index=False)
    pd.DataFrame({"feature": feature_candidates}).to_csv(args.out_dir / "hedge_model_features_used.csv", index=False)
    vif_drop.to_csv(args.out_dir / "hedge_vif_drops.csv", index=False)

    monthly_export = []
    for model_name, frame in monthly_predictions.items():
        export = frame.reset_index().copy()
        monthly_export.append(export)
        export.to_csv(args.out_dir / f"hedge_{model_name}_monthly_predictions.csv", index=False)
        confusion = hedge_state_confusion(sample, frame["hedge_state"])
        confusion.to_csv(args.out_dir / f"hedge_{model_name}_confusion.csv")
    consensus_state.rename("hedge_state").to_frame().reset_index().to_csv(
        args.out_dir / "hedge_consensus_monthly_predictions.csv", index=False
    )
    hedge_state_confusion(sample, consensus_state).to_csv(args.out_dir / "hedge_consensus_confusion.csv")

    daily_signal_export = pd.DataFrame(index=dataset.index)
    for model_name, signal in daily_state_map.items():
        daily_signal_export[f"{model_name}_hedge_state_lag1"] = signal
    daily_signal_export.reset_index().to_csv(args.out_dir / "hedge_daily_signals.csv", index=False)

    if monthly_export:
        pd.concat(monthly_export, ignore_index=True).to_csv(args.out_dir / "hedge_monthly_predictions.csv", index=False)

    for base_beta in [1.0, 2.0]:
        subset = strategy_curves[strategy_curves["base_beta"].eq(base_beta)].copy()
        if subset.empty:
            continue
        pivot = subset.pivot(index="date", columns="strategy", values="total_value")
        pivot.index = pd.to_datetime(pivot.index)
        plot_equity(pivot, plots_dir / f"hedge_equity_base_{base_beta:.0f}x.png", f"Hedge overlays vs baseline at base beta {base_beta:.0f}x")
    plot_hedge_state_chart(
        close,
        daily_state_map["consensus"].loc[daily_state_map["consensus"].index >= common_start],
        plots_dir / "consensus_hedge_states_full_common_window.png",
        "Consensus hedge states on QQQ",
    )

    metadata = {
        "analysis_dir": str(args.analysis_dir),
        "compare_dir": str(args.compare_dir),
        "common_start": common_start.date().isoformat(),
        "initial_capital": args.initial_capital,
        "monthly_contribution": args.monthly_contribution,
        "trading_cost_bps": args.trading_cost_bps,
        "borrow_rate": args.borrow_rate,
        "hedge_light_threshold": args.hedge_light_threshold,
        "hedge_strong_threshold": args.hedge_strong_threshold,
        "hedge_light_beta": args.hedge_light_beta,
        "hedge_strong_beta": args.hedge_strong_beta,
        "notes": [
            "Hedge model never flips fully net short; it reduces net beta to 0.6 or 0.3 only.",
            "Base-beta baselines are evaluated at 1x, 2x, 3x, and 5x to show what the hedge adds relative to the same risk appetite.",
            "Consensus hedge state requires agreement for the strongest hedge and uses a lighter hedge when any core model signals elevated risk.",
        ],
    }
    (args.out_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        args.out_dir,
        metadata=metadata,
        current_summary=current_summary,
        target_summary=target_summary,
        expected_stats=expected_stats,
        strategy_metrics=strategy_metrics,
        threshold_sensitivity=threshold_sensitivity,
        feature_scorecard=feature_scorecard,
        validation_metrics=validation_metrics,
    )

    print(f"Wrote short-horizon hedge audit package to {args.out_dir}")
    print(f"Common start: {common_start.date()}")
    print("Top strategy rows:")
    for _, row in strategy_metrics.head(8).iterrows():
        print(f"  {row['strategy']}: ${row['final_value']:,.0f}, DD {row['max_drawdown'] * 100:.1f}%")


if __name__ == "__main__":
    main()
