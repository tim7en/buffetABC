"""Compare fixed walk-forward GMM leverage against other no-lookahead ML models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import qqq_macro_ml_regime_analysis as regime_analysis
import qqq_macro_walkforward_leverage as leverage


ROOT = regime_analysis.ROOT
DEFAULT_ANALYSIS_SCRIPT = leverage.DEFAULT_ANALYSIS_SCRIPT
DEFAULT_ANALYSIS_OUT_DIR = leverage.DEFAULT_ANALYSIS_OUT_DIR
DEFAULT_DATASET_PATH = leverage.DEFAULT_DATASET_PATH
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare"

MODEL_LABELS = {
    "gmm": "GMM",
    "logistic": "Logistic",
    "random_forest": "Random Forest",
    "ensemble_blend": "Ensemble Blend",
    "ensemble_majority": "Ensemble Majority",
}

MODEL_COLORS = {
    ("gmm", 2): "#2ca02c",
    ("gmm", 3): "#d62728",
    ("logistic", 2): "#9467bd",
    ("logistic", 3): "#8c564b",
    ("random_forest", 2): "#17becf",
    ("random_forest", 3): "#ff7f0e",
    ("ensemble_blend", 2): "#0f766e",
    ("ensemble_blend", 3): "#115e59",
    ("ensemble_majority", 2): "#2563eb",
    ("ensemble_majority", 3): "#1d4ed8",
}

REGIME_FILL_COLORS = {
    "risk_on": "#dff3e4",
    "neutral": "#f8f3d9",
    "risk_off": "#f8d7da",
}

FEATURE_LABELS = {
    "latent_sentiment_index": "Latent sentiment",
    "external_shock_score": "External shock score",
    "qqq_feedback_score": "QQQ feedback",
    "qqq_21d_return": "QQQ 1-month return",
    "qqq_63d_return": "QQQ 3-month return",
    "qqq_sma65": "QQQ 65-day trend level",
    "qqq_sma222": "QQQ 222-day trend level",
    "qqq_vs_sma200": "QQQ vs 200-day trend",
    "qqq_realized_vol_21d": "QQQ 1-month realized volatility",
    "qqq_drawdown_252d": "QQQ 1-year drawdown",
    "dxy_63d_return": "US dollar 3-month return",
    "gold_63d_return": "Gold 3-month return",
    "gold_252d_return": "Gold 1-year return",
    "us10y_level": "10Y Treasury yield level",
    "us10y_63d_change_pp": "10Y yield 3-month change",
    "curve_10y2y_level": "Yield curve 10Y-2Y",
    "wti_63d_return": "Oil 3-month return",
    "cape_level": "Shiller CAPE",
    "cape_63d_change": "Shiller CAPE 3-month change",
    "buffett_indicator_proxy_level": "Wilshire / GDP valuation proxy",
    "buffett_indicator_proxy_252d_drift": "Wilshire / GDP 1-year drift",
    "buffett_indicator_proxy_rolling_z": "Wilshire / GDP rolling z-score",
    "wilshire_level": "Wilshire total-market index",
    "nominal_gdp_level": "Nominal GDP",
    "market_cap_to_gdp_anchor_level": "Official market cap to GDP anchor",
    "cpi_yoy_pct": "Inflation YoY",
    "cpi_yoy_3m_change_pp": "Inflation 3-month change",
    "unemployment_rate_pct": "Unemployment rate",
    "unemployment_6m_change_pp": "Unemployment 6-month change",
    "vix_level": "VIX level",
    "vix_21d_change": "VIX 1-month change",
    "hy_oas_level": "High-yield spread",
    "hy_oas_63d_change_pp": "High-yield spread 3-month change",
    "nfci_level": "Financial conditions level",
    "nfci_63d_change": "Financial conditions 3-month change",
    "t10y3m_level": "10Y-3M curve",
}

TARGET_LABELS = {
    "qqq_fwd_63d_return": "63-day QQQ return",
    "risk_off_target": "Risk-off",
    "jump_in_target": "Jump-in",
}


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
    parser.add_argument("--min-train-months", type=int, default=96)
    parser.add_argument("--risk-off-threshold", type=float, default=0.45)
    parser.add_argument("--jump-in-threshold", type=float, default=0.55)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--monthly-contribution", type=float, default=100.0)
    parser.add_argument("--weekly-contribution", type=float, default=None)
    parser.add_argument("--contribution-frequency", choices=["monthly", "weekly", "trading_days"], default="monthly")
    parser.add_argument("--trading-day-interval", type=int, default=3)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument(
        "--regime-models",
        nargs="+",
        choices=["gmm", "logistic", "random_forest"],
        default=["gmm", "logistic", "random_forest"],
    )
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=regime_analysis.RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=0.30)
    return parser.parse_args()


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


def regime_from_probabilities(
    risk_off_probability: float,
    jump_in_probability: float,
    risk_off_threshold: float,
    jump_in_threshold: float,
) -> str:
    if risk_off_probability >= risk_off_threshold:
        return "risk_off"
    if jump_in_probability >= jump_in_threshold and risk_off_probability < risk_off_threshold:
        return "risk_on"
    return "neutral"


def build_walkforward_monthly_classifier_regimes(
    sample: pd.DataFrame,
    features: list[str],
    *,
    model_name: str,
    target_horizon: int,
    min_train_months: int,
    risk_off_threshold: float,
    jump_in_threshold: float,
    random_state: int,
    rf_estimators: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=sample.index)
    out["model_name"] = model_name
    out["regime"] = pd.Series(index=sample.index, dtype=object)
    out["risk_off_probability"] = np.nan
    out["jump_in_probability"] = np.nan
    out["train_n_risk_off"] = np.nan
    out["train_n_jump_in"] = np.nan
    out["train_start"] = pd.NaT
    out["train_end"] = pd.NaT

    use_features = regime_analysis.available_features(sample, features, min_non_na=min_train_months)
    if len(use_features) < 6:
        return out

    end_col = f"qqq_fwd_{target_horizon}d_end_date"
    for i, date in enumerate(sample.index):
        current = sample.loc[[date], use_features].replace([np.inf, -np.inf], np.nan)
        if current.isna().any(axis=None):
            continue

        train = sample.iloc[:i].copy()
        if end_col in train.columns:
            train_end_dates = pd.to_datetime(train[end_col], errors="coerce")
            train = train.loc[train_end_dates < date].copy()
        train = train.replace([np.inf, -np.inf], np.nan)

        risk_off_train = train.dropna(subset=use_features + ["risk_off_target"]).copy()
        jump_in_train = train.dropna(subset=use_features + ["jump_in_target"]).copy()
        if (
            len(risk_off_train) < min_train_months
            or len(jump_in_train) < min_train_months
            or risk_off_train["risk_off_target"].nunique() < 2
            or jump_in_train["jump_in_target"].nunique() < 2
        ):
            continue

        risk_off_model = classifier_pipeline(model_name, random_state, rf_estimators)
        risk_off_model.fit(risk_off_train[use_features], risk_off_train["risk_off_target"].astype(int))
        risk_off_probability = float(risk_off_model.predict_proba(current[use_features])[:, 1][0])

        jump_in_model = classifier_pipeline(model_name, random_state, rf_estimators)
        jump_in_model.fit(jump_in_train[use_features], jump_in_train["jump_in_target"].astype(int))
        jump_in_probability = float(jump_in_model.predict_proba(current[use_features])[:, 1][0])

        out.loc[date, "risk_off_probability"] = risk_off_probability
        out.loc[date, "jump_in_probability"] = jump_in_probability
        out.loc[date, "regime"] = regime_from_probabilities(
            risk_off_probability,
            jump_in_probability,
            risk_off_threshold,
            jump_in_threshold,
        )
        out.loc[date, "train_n_risk_off"] = int(len(risk_off_train))
        out.loc[date, "train_n_jump_in"] = int(len(jump_in_train))
        out.loc[date, "train_start"] = min(risk_off_train.index.min(), jump_in_train.index.min())
        out.loc[date, "train_end"] = max(risk_off_train.index.max(), jump_in_train.index.max())

    return out


def monthly_signal_to_daily(monthly_signal: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    daily = monthly_signal.reindex(daily_index).ffill()
    return daily.shift(1)


def regime_month_end_from_daily(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    valid = frame.dropna(subset=[column]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["model_name", "regime"])
    out = valid.groupby(valid.index.to_period("M"), sort=True).tail(1)[[column]].rename(columns={column: "regime"})
    out["model_name"] = label
    return out[["model_name", "regime"]]


def build_probability_blend_regimes(
    logistic_monthly: pd.DataFrame,
    random_forest_monthly: pd.DataFrame,
    *,
    risk_off_threshold: float,
    jump_in_threshold: float,
) -> pd.DataFrame:
    logistic = logistic_monthly[
        ["risk_off_probability", "jump_in_probability", "train_start", "train_end"]
    ].rename(
        columns={
            "risk_off_probability": "logistic_risk_off_probability",
            "jump_in_probability": "logistic_jump_in_probability",
            "train_start": "logistic_train_start",
            "train_end": "logistic_train_end",
        }
    )
    random_forest = random_forest_monthly[
        ["risk_off_probability", "jump_in_probability", "train_start", "train_end"]
    ].rename(
        columns={
            "risk_off_probability": "random_forest_risk_off_probability",
            "jump_in_probability": "random_forest_jump_in_probability",
            "train_start": "random_forest_train_start",
            "train_end": "random_forest_train_end",
        }
    )
    probability_columns = [
        "logistic_risk_off_probability",
        "logistic_jump_in_probability",
        "random_forest_risk_off_probability",
        "random_forest_jump_in_probability",
    ]
    joined = logistic.join(random_forest, how="inner").dropna(subset=probability_columns)
    out = pd.DataFrame(index=joined.index)
    out["model_name"] = "ensemble_blend"
    out["risk_off_probability"] = joined[
        ["logistic_risk_off_probability", "random_forest_risk_off_probability"]
    ].mean(axis=1)
    out["jump_in_probability"] = joined[
        ["logistic_jump_in_probability", "random_forest_jump_in_probability"]
    ].mean(axis=1)
    out["regime"] = out.apply(
        lambda row: regime_from_probabilities(
            float(row["risk_off_probability"]),
            float(row["jump_in_probability"]),
            risk_off_threshold,
            jump_in_threshold,
        ),
        axis=1,
    )
    out["train_start"] = joined[["logistic_train_start", "random_forest_train_start"]].min(axis=1)
    out["train_end"] = joined[["logistic_train_end", "random_forest_train_end"]].max(axis=1)
    out["train_n_risk_off"] = np.nan
    out["train_n_jump_in"] = np.nan
    return out[
        [
            "model_name",
            "regime",
            "risk_off_probability",
            "jump_in_probability",
            "train_n_risk_off",
            "train_n_jump_in",
            "train_start",
            "train_end",
        ]
    ]


def build_majority_vote_regimes(monthly_predictions: dict[str, pd.Series]) -> pd.DataFrame:
    joined = pd.DataFrame(monthly_predictions)
    out = pd.DataFrame(index=joined.index)
    out["model_name"] = "ensemble_majority"
    out["regime"] = pd.Series(index=joined.index, dtype=object)
    valid = joined.notna().all(axis=1)
    for date, row in joined.loc[valid].iterrows():
        counts = row.astype(str).value_counts()
        top_count = int(counts.max())
        winners = counts[counts.eq(top_count)].index.tolist()
        if top_count >= 2 and len(winners) == 1:
            out.loc[date, "regime"] = winners[0]
        else:
            out.loc[date, "regime"] = "neutral"
    out["risk_off_probability"] = np.nan
    out["jump_in_probability"] = np.nan
    out["train_n_risk_off"] = np.nan
    out["train_n_jump_in"] = np.nan
    out["train_start"] = pd.NaT
    out["train_end"] = pd.NaT
    return out[
        [
            "model_name",
            "regime",
            "risk_off_probability",
            "jump_in_probability",
            "train_n_risk_off",
            "train_n_jump_in",
            "train_start",
            "train_end",
        ]
    ]


def update_plot_styles(model_names: list[str]) -> None:
    for model_name in model_names:
        if model_name == "gmm":
            continue
        for leverage_level in [2, 3]:
            strategy = f"walkforward_{model_name}_riskon_{leverage_level}x_prob_regime_dca"
            leverage.PLOT_STYLE[strategy] = {
                "label": f"WF {MODEL_LABELS[model_name]} {leverage_level}x",
                "color": MODEL_COLORS[(model_name, leverage_level)],
            }


def write_model_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    gmm_refits: pd.DataFrame,
    monthly_regimes: pd.DataFrame,
    dataset_path: Path,
    contribution_frequency: str,
    periodic_contribution: float,
    regime_models: list[str],
    common_start: pd.Timestamp,
) -> None:
    comparable = metrics[metrics["window"] == "comparable_2007_05_31"].copy().sort_values("final_value", ascending=False)
    reported_models = regime_models
    if not monthly_regimes.empty:
        reported_models = [
            MODEL_LABELS.get(model_name, model_name)
            for model_name in monthly_regimes["model_name"].dropna().drop_duplicates().tolist()
        ]
    lines = [
        "# Walk-forward Model Comparison",
        "",
        "## Scope",
        "",
        f"- Dataset used: `{dataset_path}`",
        f"- Regime models compared: `{', '.join(reported_models)}`",
        f"- Common strategy start: `{common_start.date()}`",
        f"- Contribution cadence: `{contribution_frequency}` at `${periodic_contribution:,.2f}` per event.",
        "",
        "## No-Lookahead Safeguards",
        "",
        "- Daily GMM refits train only on rows strictly before each refit date.",
        "- Supervised monthly models train only on month-end rows strictly before the prediction month.",
        "- Supervised monthly models purge any training rows whose forward target window overlaps the prediction date.",
        "- All daily leverage backtests trade lagged signals only; no same-day prediction is traded on the same bar.",
        "",
        "## Coverage",
        "",
    ]
    if not gmm_refits.empty:
        lines.extend(
            [
                f"- GMM first refit: `{pd.Timestamp(gmm_refits.iloc[0]['wf_refit_date']).date()}`",
                f"- GMM last refit: `{pd.Timestamp(gmm_refits.iloc[-1]['wf_refit_date']).date()}`",
                f"- GMM refit count: `{len(gmm_refits)}`",
            ]
        )
    if not monthly_regimes.empty:
        for model_name, group in monthly_regimes.dropna(subset=["regime"]).groupby("model_name", sort=False):
            lines.extend(
                [
                    f"- {MODEL_LABELS.get(model_name, model_name)} first monthly prediction: `{group.index.min().date()}`",
                    f"- {MODEL_LABELS.get(model_name, model_name)} last monthly prediction: `{group.index.max().date()}`",
                    f"- {MODEL_LABELS.get(model_name, model_name)} prediction count: `{len(group)}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Comparable Window Metrics",
            "",
            "| Strategy | Final Value | XIRR | TWR CAGR | Max DD | Final / Contributed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparable.iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['final_value']:,.0f} | {row['xirr'] * 100:.1f}% | "
            f"{row['time_weighted_cagr'] * 100:.1f}% | {row['max_drawdown'] * 100:.1f}% | "
            f"{row['final_multiple_on_contributed']:.2f}x |"
        )
    lines.append("")
    out_dir.joinpath("walkforward_model_compare_report.md").write_text("\n".join(lines), encoding="utf-8")


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def build_feature_importance_table(feature_importance: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "qqq_fwd_63d_return":
        signed_model = "ridge"
        unsigned_model = "random_forest"
        signed_col = "ridge_coef"
        unsigned_col = "random_forest_permutation_importance"
    else:
        signed_model = "logistic"
        unsigned_model = "random_forest"
        signed_col = "logistic_coef"
        unsigned_col = "random_forest_permutation_importance"

    data = feature_importance[
        feature_importance["target"].eq(target)
        & feature_importance["model"].isin([signed_model, unsigned_model])
    ].copy()
    if data.empty:
        return pd.DataFrame()

    logistic = (
        data[data["model"] == signed_model][["feature", "importance_mean"]]
        .rename(columns={"importance_mean": signed_col})
        .copy()
    )
    logistic["signed_abs_coef"] = logistic[signed_col].abs()

    random_forest = (
        data[data["model"] == unsigned_model][["feature", "importance_mean", "importance_std"]]
        .rename(
            columns={
                "importance_mean": unsigned_col,
                "importance_std": "random_forest_importance_std",
            }
        )
        .copy()
    )

    selected = set(logistic.nlargest(10, "signed_abs_coef")["feature"])
    selected.update(
        random_forest.nlargest(10, unsigned_col)["feature"].tolist()
    )

    table = pd.merge(logistic, random_forest, on="feature", how="outer")
    table = table[table["feature"].isin(selected)].copy()
    table["feature_label"] = table["feature"].map(feature_label)
    table["signed_driver"] = np.where(
        table[signed_col] > 0.0,
        "Higher reading raises event odds",
        np.where(
            table[signed_col] < 0.0,
            "Higher reading lowers outcome",
            "Neutral",
        ),
    )
    if target == "qqq_fwd_63d_return":
        table["signed_driver"] = np.where(
            table[signed_col] > 0.0,
            "Higher reading lifts forward return",
            np.where(table[signed_col] < 0.0, "Higher reading hurts forward return", "Neutral"),
        )
    else:
        table["signed_driver"] = np.where(
            table[signed_col] > 0.0,
            "Higher reading raises event odds",
            np.where(table[signed_col] < 0.0, "Higher reading lowers event odds", "Neutral"),
        )
    table["rf_rank_score"] = table[unsigned_col].fillna(-np.inf)
    table["combined_rank"] = (
        table["signed_abs_coef"].fillna(0.0).rank(ascending=False, method="dense")
        + table["rf_rank_score"].rank(ascending=False, method="dense")
    )
    return table.sort_values(
        ["combined_rank", "signed_abs_coef", unsigned_col],
        ascending=[True, False, False],
    )[
        [
            "feature",
            "feature_label",
            signed_col,
            "signed_abs_coef",
            "signed_driver",
            unsigned_col,
            "random_forest_importance_std",
        ]
    ]


def plot_feature_importance_compare(feature_importance: pd.DataFrame, target: str, out_path: Path) -> None:
    if target == "qqq_fwd_63d_return":
        signed_model = "ridge"
        left_title = "ridge coefficients"
    else:
        signed_model = "logistic"
        left_title = "logistic coefficients"
    data = feature_importance[
        feature_importance["target"].eq(target)
        & feature_importance["model"].isin([signed_model, "random_forest"])
    ].copy()
    if data.empty:
        return

    logistic = data[data["model"] == signed_model].copy()
    logistic["abs_importance"] = logistic["importance_mean"].abs()
    logistic = logistic.nlargest(10, "abs_importance").sort_values("importance_mean")

    random_forest = (
        data[data["model"] == "random_forest"]
        .nlargest(10, "importance_mean")
        .sort_values("importance_mean")
        .copy()
    )

    fig, (ax_left, ax_right) = leverage.plt.subplots(1, 2, figsize=(15, 7))

    left_colors = np.where(logistic["importance_mean"] >= 0.0, "#b91c1c", "#15803d")
    ax_left.barh(
        logistic["feature"].map(feature_label),
        logistic["importance_mean"],
        color=left_colors,
    )
    ax_left.axvline(0.0, color="#111827", linewidth=1.0)
    ax_left.set_title(f"{TARGET_LABELS.get(target, target)}: {left_title}")
    ax_left.set_xlabel("Signed coefficient")
    ax_left.grid(alpha=0.2, axis="x")

    ax_right.barh(
        random_forest["feature"].map(feature_label),
        random_forest["importance_mean"],
        xerr=random_forest["importance_std"],
        color="#2563eb",
        alpha=0.85,
    )
    ax_right.set_title(f"{TARGET_LABELS.get(target, target)}: random forest importance")
    ax_right.set_xlabel("Permutation importance")
    ax_right.grid(alpha=0.2, axis="x")

    fig.suptitle(f"Feature comparison for {TARGET_LABELS.get(target, target).lower()} model", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    leverage.plt.close(fig)


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def write_investor_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
) -> None:
    comparable = metrics[metrics["window"] == "comparable_2007_05_31"].copy()
    comparable = comparable.sort_values("final_value", ascending=False)

    class_metrics = validation_metrics[
        validation_metrics["target"].isin(["risk_off_target", "jump_in_target"])
        & validation_metrics["model"].isin(["logistic", "random_forest"])
    ].copy()

    lines = [
        "# Investor-Friendly ML Backtest Report",
        "",
        "## Bottom Line",
        "",
        "- The walk-forward logistic strategy still shows the strongest backtest result in this repo, but it earns that by staying risk-on most of the time and accepting much deeper drawdowns than plain DCA.",
        "- I do not see a direct look-ahead bug in the current logistic path. Training is chronological, overlapping forward windows are purged, and trades use lagged signals only.",
        "- Ensemble variants in this report are simple combinations of already-generated walk-forward signals; they do not add a second fitting stage.",
        "- I would still treat the result as promising rather than proven because the predictive validation is only modest and the model was selected after comparing several approaches.",
        "",
        "## Backtest Snapshot",
        "",
        "| Strategy | Final Value | XIRR | TWR CAGR | Max DD |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in comparable.iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['final_value']:,.0f} | {fmt_pct(row['xirr'])} | "
            f"{fmt_pct(row['time_weighted_cagr'])} | {fmt_pct(row['max_drawdown'])} |"
        )

    lines.extend(
        [
            "",
            "## Model Validation",
            "",
            "These scores come from a purged chronological train/test split on month-end samples. Higher AUC and average precision are better; lower Brier score is better.",
            "",
            "| Target | Model | AUC | Average Precision | Brier | Balanced Accuracy @ 50% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in class_metrics.sort_values(["target", "model"]).iterrows():
        lines.append(
            f"| {TARGET_LABELS.get(row['target'], row['target'])} | {MODEL_LABELS.get(row['model'], row['model'])} | "
            f"{float(row['auc']):.3f} | {float(row['average_precision']):.3f} | "
            f"{float(row['brier']):.3f} | {float(row['balanced_accuracy_at_50pct']):.3f} |"
        )

    if not threshold_sensitivity.empty:
        base_row = threshold_sensitivity[
            threshold_sensitivity["risk_off_threshold"].eq(0.45)
            & threshold_sensitivity["jump_in_threshold"].eq(0.55)
            & threshold_sensitivity["risk_on_leverage"].eq(3)
        ]
        best_row = threshold_sensitivity[threshold_sensitivity["risk_on_leverage"].eq(3)].head(1)
        if not base_row.empty and not best_row.empty:
            base = base_row.iloc[0]
            best = best_row.iloc[0]
            lines.extend(
                [
                    "",
                    "## Threshold Check",
                    "",
                    f"- The current logistic 3x setting (`risk_off={base['risk_off_threshold']:.2f}`, `jump_in={base['jump_in_threshold']:.2f}`) finished at `${base['final_value']:,.0f}` with `{fmt_pct(base['max_drawdown'])}` max drawdown.",
                    f"- The best nearby 3x threshold in the local grid finished at `${best['final_value']:,.0f}` with `{fmt_pct(best['max_drawdown'])}` max drawdown.",
                    "- That does not eliminate model-selection risk, but it suggests the current result is not coming from an obviously fragile one-cell threshold choice.",
                ]
            )

    lines.extend(
        [
            "",
            "## What Drives The Logistic Model",
            "",
            "The logistic model gives direction: a positive coefficient means a higher reading increases the odds of the event, while a negative coefficient means it lowers the odds. Random forest importance only tells us how useful a feature was, not which direction it pushes.",
            "",
        ]
    )

    return_table = build_feature_importance_table(feature_importance, "qqq_fwd_63d_return")
    if not return_table.empty:
        top_positive = return_table[return_table["ridge_coef"] > 0.0].nlargest(5, "ridge_coef")
        top_negative = return_table[return_table["ridge_coef"] < 0.0].nsmallest(5, "ridge_coef")
        top_rf = return_table.nlargest(5, "random_forest_permutation_importance")
        lines.extend(
            [
                "## What Drives Forward QQQ Returns",
                "",
                "| Ridge features linked to stronger returns | Coef |",
                "|---|---:|",
            ]
        )
        for _, row in top_positive.iterrows():
            lines.append(f"| {row['feature_label']} | {row['ridge_coef']:.3f} |")
        lines.extend(["", "| Ridge features linked to weaker returns | Coef |", "|---|---:|"])
        for _, row in top_negative.iterrows():
            lines.append(f"| {row['feature_label']} | {row['ridge_coef']:.3f} |")
        lines.extend(["", "| Strongest random forest return features | Importance |", "|---|---:|"])
        for _, row in top_rf.iterrows():
            lines.append(
                f"| {row['feature_label']} | {row['random_forest_permutation_importance']:.4f} |"
            )
        lines.append("")

    for target in ["risk_off_target", "jump_in_target"]:
        table = build_feature_importance_table(feature_importance, target)
        if table.empty:
            continue
        top_positive = table[table["logistic_coef"] > 0.0].nlargest(5, "logistic_coef")
        top_negative = table[table["logistic_coef"] < 0.0].nsmallest(5, "logistic_coef")
        top_rf = table.nlargest(5, "random_forest_permutation_importance")
        lines.extend(
            [
                f"### {TARGET_LABELS.get(target, target)}",
                "",
                "| Logistic coefficients pushing odds higher | Coef |",
                "|---|---:|",
            ]
        )
        for _, row in top_positive.iterrows():
            lines.append(f"| {row['feature_label']} | {row['logistic_coef']:.3f} |")
        lines.extend(["", "| Logistic coefficients pushing odds lower | Coef |", "|---|---:|"])
        for _, row in top_negative.iterrows():
            lines.append(f"| {row['feature_label']} | {row['logistic_coef']:.3f} |")
        lines.extend(["", "| Strongest random forest features | Importance |", "|---|---:|"])
        for _, row in top_rf.iterrows():
            lines.append(
                f"| {row['feature_label']} | {row['random_forest_permutation_importance']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Practical Take",
            "",
            "- The logistic strategy looks more like an aggressive risk-on timing overlay than a defensive capital-preservation model.",
            "- Plain DCA still has the cleaner risk story. Logistic 3x may be interesting for research, but its drawdown profile is severe enough that I would not present it as a conservative investor solution.",
            "- GMM remains the more intuitive macro-regime framework, but in the current corrected run it does not match logistic on return.",
            "",
            "## Files",
            "",
            "- `walkforward_model_validation_metrics.csv`: purged validation scores",
            "- `walkforward_model_feature_importance.csv`: raw model feature weights and permutation importance",
            "- `walkforward_feature_importance_compare_returns.png`: forward-return feature comparison chart",
            "- `walkforward_feature_importance_compare_risk_off.png`: risk-off feature comparison chart",
            "- `walkforward_feature_importance_compare_jump_in.png`: jump-in feature comparison chart",
            "- `walkforward_logistic_regimes_full_common_window.png`: logistic regime chart on QQQ",
            "- `walkforward_ensemble_blend_regimes_full_common_window.png`: probability-blend ensemble chart on QQQ",
            "- `walkforward_ensemble_majority_regimes_full_common_window.png`: majority-vote ensemble chart on QQQ",
        ]
    )

    out_dir.joinpath("walkforward_model_investor_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_model_compare_equity(curves: pd.DataFrame, out_path: Path, window: str) -> None:
    data = curves[curves["window"] == window].copy()
    if data.empty:
        return
    fig, ax = leverage.plt.subplots(figsize=(14, 7))
    for strategy, group in data.groupby("strategy", sort=False):
        group = group.sort_values("date")
        ax.plot(
            pd.to_datetime(group["date"]),
            group["total_value"].astype(float),
            linewidth=1.6,
            label=leverage.strategy_label(strategy),
            color=leverage.strategy_color(strategy),
        )
    ax.set_title(f"Walk-forward model comparison: {window}")
    ax.set_ylabel("Account value, USD")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    leverage.plt.close(fig)


def plot_model_compare_drawdowns(curves: pd.DataFrame, out_path: Path, window: str) -> None:
    data = curves[curves["window"] == window].copy()
    if data.empty:
        return
    fig, ax = leverage.plt.subplots(figsize=(14, 7))
    for strategy, group in data.groupby("strategy", sort=False):
        group = group.sort_values("date")
        equity = group["total_value"].astype(float)
        drawdown = equity / equity.cummax() - 1.0
        ax.plot(
            pd.to_datetime(group["date"]),
            drawdown,
            linewidth=1.6,
            label=leverage.strategy_label(strategy),
            color=leverage.strategy_color(strategy),
        )
    ax.set_title(f"Walk-forward model comparison drawdowns: {window}")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(leverage.mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    leverage.plt.close(fig)


def regime_spans(signal: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current_regime: str | None = None
    start_date: pd.Timestamp | None = None
    previous_date: pd.Timestamp | None = None
    for date, regime in signal.dropna().astype(str).items():
        if regime != current_regime:
            if current_regime is not None and start_date is not None and previous_date is not None:
                spans.append((start_date, previous_date, current_regime))
            current_regime = regime
            start_date = pd.Timestamp(date)
        previous_date = pd.Timestamp(date)
    if current_regime is not None and start_date is not None and previous_date is not None:
        spans.append((start_date, previous_date, current_regime))
    return spans


def plot_regime_chart(close: pd.Series, regime_signal: pd.Series, out_path: Path, title: str) -> None:
    signal = regime_signal.reindex(close.index)
    if signal.dropna().empty:
        return

    start_date = signal.dropna().index.min()
    price = close.loc[close.index >= start_date].copy()
    signal = signal.reindex(price.index).ffill()
    regime_level = signal.map({"risk_off": 0.0, "neutral": 1.0, "risk_on": 2.0})

    fig, (ax_price, ax_regime) = leverage.plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15]},
    )

    for span_start, span_end, regime in regime_spans(signal):
        color = REGIME_FILL_COLORS.get(regime, "#e5e7eb")
        ax_price.axvspan(span_start, span_end, color=color, alpha=0.55, linewidth=0)
        ax_regime.axvspan(span_start, span_end, color=color, alpha=0.55, linewidth=0)

    ax_price.plot(price.index, price.astype(float), color="#0f172a", linewidth=1.7, label="QQQ close")
    ax_price.set_title(title)
    ax_price.set_ylabel("QQQ close")
    ax_price.set_yscale("log")
    ax_price.grid(alpha=0.25)
    ax_price.legend(
        handles=[
            Patch(facecolor=REGIME_FILL_COLORS["risk_on"], edgecolor="none", label="Risk on"),
            Patch(facecolor=REGIME_FILL_COLORS["neutral"], edgecolor="none", label="Neutral"),
            Patch(facecolor=REGIME_FILL_COLORS["risk_off"], edgecolor="none", label="Risk off"),
        ],
        loc="upper left",
        ncol=3,
    )

    ax_regime.step(regime_level.index, regime_level.astype(float), where="post", color="#111827", linewidth=1.4)
    ax_regime.set_ylim(-0.5, 2.5)
    ax_regime.set_yticks([0.0, 1.0, 2.0], labels=["Risk off", "Neutral", "Risk on"])
    ax_regime.set_ylabel("Regime")
    ax_regime.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    leverage.plt.close(fig)


def logistic_threshold_sensitivity(
    close: pd.Series,
    monthly_regimes: pd.DataFrame,
    *,
    common_start: pd.Timestamp,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if monthly_regimes.empty:
        return pd.DataFrame()

    base = monthly_regimes.dropna(subset=["risk_off_probability", "jump_in_probability"]).copy()
    if base.empty:
        return pd.DataFrame()

    risk_off_thresholds = sorted(
        {
            round(float(np.clip(args.risk_off_threshold + delta, 0.05, 0.95)), 2)
            for delta in (-0.05, 0.00, 0.05)
        }
    )
    jump_in_thresholds = sorted(
        {
            round(float(np.clip(args.jump_in_threshold + delta, 0.05, 0.95)), 2)
            for delta in (-0.05, 0.00, 0.05)
        }
    )

    rows: list[dict[str, Any]] = []
    window_close = close.loc[close.index >= common_start]
    periodic_contribution = leverage.resolve_periodic_contribution(args)
    for risk_off_threshold in risk_off_thresholds:
        for jump_in_threshold in jump_in_thresholds:
            monthly_signal = base.apply(
                lambda row: regime_from_probabilities(
                    float(row["risk_off_probability"]),
                    float(row["jump_in_probability"]),
                    risk_off_threshold,
                    jump_in_threshold,
                ),
                axis=1,
            )
            daily_signal = monthly_signal_to_daily(monthly_signal, close.index).loc[lambda x: x.index >= common_start]
            if daily_signal.dropna().empty:
                continue
            for leverage_level in [2.0, 3.0]:
                result = leverage.simulate_regime_leverage(
                    window_close,
                    daily_signal,
                    strategy="logistic_threshold_sensitivity",
                    risk_on_leverage=leverage_level,
                    initial_capital=args.initial_capital,
                    periodic_contribution=periodic_contribution,
                    contribution_frequency=args.contribution_frequency,
                    trading_day_interval=args.trading_day_interval,
                    trading_cost_bps=args.trading_cost_bps,
                    borrow_rate=args.borrow_rate,
                )
                metrics_row = leverage.metrics_row("full_common_window", result, None)
                rows.append(
                    {
                        "risk_off_threshold": risk_off_threshold,
                        "jump_in_threshold": jump_in_threshold,
                        "risk_on_leverage": int(leverage_level),
                        "start_date": metrics_row["start_date"],
                        "end_date": metrics_row["end_date"],
                        "final_value": metrics_row["final_value"],
                        "xirr": metrics_row["xirr"],
                        "time_weighted_cagr": metrics_row["time_weighted_cagr"],
                        "max_drawdown": metrics_row["max_drawdown"],
                        "avg_target_leverage": metrics_row["avg_target_leverage"],
                        "risk_on_months": result.risk_on_months,
                        "neutral_months": result.neutral_months,
                        "risk_off_months": result.risk_off_months,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["risk_on_leverage", "final_value", "jump_in_threshold", "risk_off_threshold"],
        ascending=[True, False, True, True],
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    leverage.maybe_run_analysis(args)
    update_plot_styles(args.regime_models)
    periodic_contribution = leverage.resolve_periodic_contribution(args)

    dataset = leverage.load_dataset(args.dataset_path)
    if args.end:
        dataset = dataset.loc[dataset.index <= pd.Timestamp(args.end)]
    close = dataset["qqq_close"].dropna().astype(float)

    model_signals: dict[str, pd.Series] = {}
    gmm_refits = pd.DataFrame()
    gmm_summary = pd.DataFrame()
    walkforward_daily = pd.DataFrame(index=dataset.index)
    monthly_regime_tables: list[pd.DataFrame] = []
    monthly_model_tables: dict[str, pd.DataFrame] = {}

    if "gmm" in args.regime_models:
        walkforward_daily, gmm_refits, gmm_summary = leverage.build_walkforward_daily_regimes(dataset, args.min_train_days)
        model_signals["gmm"] = walkforward_daily["wf_gmm_regime_signal_lag1"]
        walkforward_daily.to_csv(args.out_dir / "walkforward_gmm_daily_regimes.csv", index_label="date")
        gmm_refits.to_csv(args.out_dir / "walkforward_gmm_refits.csv", index=False)
        gmm_summary.to_csv(args.out_dir / "walkforward_gmm_regime_summary.csv", index=False)

    sample = regime_analysis.month_end_sample(dataset)
    sample_features = regime_analysis.available_features(sample, regime_analysis.MODEL_FEATURES, min_non_na=args.min_train_months)
    if len(sample_features) < 6:
        raise RuntimeError(f"Not enough usable month-end model features. Found: {sample_features}")
    validation_metrics, feature_importance, _ = regime_analysis.evaluate_models(
        sample,
        sample_features,
        63,
        args.test_size,
        args.random_state,
    )

    for model_name in args.regime_models:
        if model_name == "gmm":
            continue
        monthly_regimes = build_walkforward_monthly_classifier_regimes(
            sample,
            sample_features,
            model_name=model_name,
            target_horizon=63,
            min_train_months=args.min_train_months,
            risk_off_threshold=args.risk_off_threshold,
            jump_in_threshold=args.jump_in_threshold,
            random_state=args.random_state,
            rf_estimators=args.rf_estimators,
        )
        monthly_regime_tables.append(monthly_regimes)
        monthly_model_tables[model_name] = monthly_regimes
        model_signals[model_name] = monthly_signal_to_daily(monthly_regimes["regime"], close.index)

    if {"logistic", "random_forest"}.issubset(monthly_model_tables):
        blend_monthly = build_probability_blend_regimes(
            monthly_model_tables["logistic"],
            monthly_model_tables["random_forest"],
            risk_off_threshold=args.risk_off_threshold,
            jump_in_threshold=args.jump_in_threshold,
        )
        monthly_regime_tables.append(blend_monthly)
        monthly_model_tables["ensemble_blend"] = blend_monthly
        model_signals["ensemble_blend"] = monthly_signal_to_daily(blend_monthly["regime"], close.index)
        for leverage_level in [2, 3]:
            strategy = f"walkforward_ensemble_blend_riskon_{leverage_level}x_prob_regime_dca"
            leverage.PLOT_STYLE[strategy] = {
                "label": f"WF {MODEL_LABELS['ensemble_blend']} {leverage_level}x",
                "color": MODEL_COLORS[('ensemble_blend', leverage_level)],
            }

    if {"logistic", "random_forest"}.issubset(monthly_model_tables) and "gmm" in model_signals:
        gmm_monthly = regime_month_end_from_daily(walkforward_daily, "wf_gmm_regime", "gmm")
        majority_monthly = build_majority_vote_regimes(
            {
                "logistic": monthly_model_tables["logistic"]["regime"],
                "random_forest": monthly_model_tables["random_forest"]["regime"],
                "gmm": gmm_monthly["regime"],
            }
        )
        monthly_regime_tables.append(majority_monthly)
        monthly_model_tables["ensemble_majority"] = majority_monthly
        model_signals["ensemble_majority"] = monthly_signal_to_daily(majority_monthly["regime"], close.index)
        for leverage_level in [2, 3]:
            strategy = f"walkforward_ensemble_majority_riskon_{leverage_level}x_prob_regime_dca"
            leverage.PLOT_STYLE[strategy] = {
                "label": f"WF {MODEL_LABELS['ensemble_majority']} {leverage_level}x",
                "color": MODEL_COLORS[('ensemble_majority', leverage_level)],
            }

    first_valid_dates = [signal.dropna().index.min() for signal in model_signals.values() if not signal.dropna().empty]
    if not first_valid_dates:
        raise RuntimeError("No valid walk-forward model signals were generated.")
    common_start = max(first_valid_dates)

    daily_signal_export = pd.DataFrame(index=close.index)
    for model_name, signal in model_signals.items():
        daily_signal_export[f"{model_name}_signal_lag1"] = signal.reindex(close.index)
    daily_signal_export.to_csv(args.out_dir / "walkforward_model_signals_daily.csv", index_label="date")

    monthly_regime_export = (
        pd.concat(monthly_regime_tables).sort_index()
        if monthly_regime_tables
        else pd.DataFrame(columns=["model_name", "regime"])
    )
    if not monthly_regime_export.empty:
        monthly_regime_export.to_csv(args.out_dir / "walkforward_model_regimes_monthly.csv", index_label="date")

    if "logistic" in model_signals:
        logistic_daily = pd.DataFrame(index=close.index)
        logistic_daily["qqq_close"] = close
        logistic_daily["logistic_signal_lag1"] = model_signals["logistic"].reindex(close.index)
        logistic_daily.to_csv(args.out_dir / "walkforward_logistic_traded_regimes_daily.csv", index_label="date")
    if "logistic" in monthly_model_tables:
        monthly_model_tables["logistic"].to_csv(
            args.out_dir / "walkforward_logistic_traded_regimes_monthly.csv",
            index_label="date",
        )

    window_starts = {
        "full_common_window": common_start,
        "comparable_2007_05_31": max(common_start, leverage.COMPARABLE_START),
    }
    strategy_defs = [("plain_dca", None, None)]
    strategy_model_names = list(model_signals.keys())
    for model_name in strategy_model_names:
        for leverage_level in [2.0, 3.0]:
            if model_name == "gmm":
                strategy_name = f"walkforward_gmm_riskon_{int(leverage_level)}x_keep_long_riskoff_reserve_dca"
            else:
                strategy_name = f"walkforward_{model_name}_riskon_{int(leverage_level)}x_prob_regime_dca"
            strategy_defs.append((strategy_name, model_name, leverage_level))

    metrics_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for window, start_date in window_starts.items():
        window_close = close.loc[close.index >= start_date]
        plain = leverage.simulate_plain_dca(
            window_close,
            strategy="plain_dca",
            initial_capital=args.initial_capital,
            periodic_contribution=periodic_contribution,
            contribution_frequency=args.contribution_frequency,
            trading_day_interval=args.trading_day_interval,
            trading_cost_bps=args.trading_cost_bps,
        )
        leverage.append_window(curve_frames, plain.curves, window)
        plain_final = float(plain.curves["total_value"].iloc[-1])
        metrics_rows.append(leverage.metrics_row(window, plain, None))

        for strategy_name, model_name, leverage_level in strategy_defs[1:]:
            signal = model_signals[model_name].loc[model_signals[model_name].index >= start_date]
            result = leverage.simulate_regime_leverage(
                window_close,
                signal,
                strategy=strategy_name,
                risk_on_leverage=float(leverage_level),
                initial_capital=args.initial_capital,
                periodic_contribution=periodic_contribution,
                contribution_frequency=args.contribution_frequency,
                trading_day_interval=args.trading_day_interval,
                trading_cost_bps=args.trading_cost_bps,
                borrow_rate=args.borrow_rate,
            )
            leverage.append_window(curve_frames, result.curves, window)
            if not result.events.empty:
                events = result.events.copy()
                events["window"] = window
                event_frames.append(events)
            metrics_rows.append(leverage.metrics_row(window, result, plain_final))

    metrics = pd.DataFrame(metrics_rows)
    metrics["contribution_frequency"] = leverage.contribution_frequency_label(args)
    metrics["periodic_contribution"] = periodic_contribution
    metrics["trading_day_interval"] = args.trading_day_interval if args.contribution_frequency == "trading_days" else np.nan
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    threshold_sensitivity = pd.DataFrame()

    metrics.to_csv(args.out_dir / "walkforward_model_compare_leverage_metrics.csv", index=False)
    curves.to_csv(args.out_dir / "walkforward_model_compare_leverage_curves.csv", index=False)
    events.to_csv(args.out_dir / "walkforward_model_compare_leverage_events.csv", index=False)
    validation_metrics.to_csv(args.out_dir / "walkforward_model_validation_metrics.csv", index=False)
    feature_importance.to_csv(args.out_dir / "walkforward_model_feature_importance.csv", index=False)
    for window in window_starts:
        plot_model_compare_equity(curves, args.out_dir / f"walkforward_model_compare_equity_{window}.png", window)
        plot_model_compare_drawdowns(curves, args.out_dir / f"walkforward_model_compare_drawdown_{window}.png", window)
    for target in ["qqq_fwd_63d_return", "risk_off_target", "jump_in_target"]:
        table = build_feature_importance_table(feature_importance, target)
        if not table.empty:
            table.to_csv(
                args.out_dir / f"walkforward_model_feature_importance_compare_{target}.csv",
                index=False,
            )
            plot_name = (
                "walkforward_feature_importance_compare_returns.png"
                if target == "qqq_fwd_63d_return"
                else f"walkforward_feature_importance_compare_{target.replace('_target', '')}.png"
            )
            plot_feature_importance_compare(
                feature_importance,
                target,
                args.out_dir / plot_name,
            )

    if "logistic" in model_signals:
        plot_regime_chart(
            close.loc[close.index >= common_start],
            model_signals["logistic"].loc[model_signals["logistic"].index >= common_start],
            args.out_dir / "walkforward_logistic_regimes_full_common_window.png",
            "Walk-forward logistic regimes on QQQ",
        )
    if "ensemble_blend" in model_signals:
        plot_regime_chart(
            close.loc[close.index >= common_start],
            model_signals["ensemble_blend"].loc[model_signals["ensemble_blend"].index >= common_start],
            args.out_dir / "walkforward_ensemble_blend_regimes_full_common_window.png",
            "Walk-forward ensemble-blend regimes on QQQ",
        )
    if "ensemble_majority" in model_signals:
        plot_regime_chart(
            close.loc[close.index >= common_start],
            model_signals["ensemble_majority"].loc[model_signals["ensemble_majority"].index >= common_start],
            args.out_dir / "walkforward_ensemble_majority_regimes_full_common_window.png",
            "Walk-forward ensemble-majority regimes on QQQ",
        )
    if "logistic" in args.regime_models:
        logistic_monthly = (
            monthly_regime_export[monthly_regime_export["model_name"] == "logistic"].copy()
            if not monthly_regime_export.empty
            else pd.DataFrame()
        )
        threshold_sensitivity = logistic_threshold_sensitivity(
            close,
            logistic_monthly,
            common_start=common_start,
            args=args,
        )
        if not threshold_sensitivity.empty:
            threshold_sensitivity.to_csv(args.out_dir / "walkforward_logistic_threshold_sensitivity.csv", index=False)

    write_model_report(
        args.out_dir,
        metrics=metrics,
        gmm_refits=gmm_refits,
        monthly_regimes=monthly_regime_export,
        dataset_path=args.dataset_path,
        contribution_frequency=leverage.contribution_frequency_label(args),
        periodic_contribution=periodic_contribution,
        regime_models=args.regime_models,
        common_start=common_start,
    )
    write_investor_report(
        args.out_dir,
        metrics=metrics,
        validation_metrics=validation_metrics,
        feature_importance=feature_importance,
        threshold_sensitivity=threshold_sensitivity if "logistic" in args.regime_models else pd.DataFrame(),
    )

    print(f"Wrote walk-forward model comparison files to {args.out_dir}")
    print(
        "Contribution cadence: "
        f"{leverage.contribution_frequency_label(args)} at ${periodic_contribution:,.2f} per contribution event"
    )
    for window in ["comparable_2007_05_31", "full_common_window"]:
        window_metrics = metrics[metrics["window"] == window].sort_values("final_value", ascending=False)
        if window_metrics.empty:
            continue
        print(f"{window} final values:")
        for _, row in window_metrics.iterrows():
            print(f"  {row['strategy']}: ${row['final_value']:,.0f}")


if __name__ == "__main__":
    main()
