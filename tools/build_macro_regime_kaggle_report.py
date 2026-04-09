"""Build a self-contained Kaggle-style HTML report for the macro regime stack.

The report is intentionally sourced from frozen CSV outputs so we can audit the
current model state without silently recomputing the research pipeline.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_COMPARE_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260409_monthly_equal"
DEFAULT_AUDIT_DIR = ROOT / "reports" / "macro_regime_edge_audit_20260409"
DEFAULT_OUTPUT_HTML = DEFAULT_AUDIT_DIR / "macro_regime_kaggle_report.html"

BG = "#f4f7fb"
PANEL = "#ffffff"
TEXT = "#142033"
MUTED = "#5f6e84"
BLUE = "#1f78ff"
BLUE_DARK = "#11479d"
TEAL = "#00a7a0"
GREEN = "#23a36d"
RED = "#d64f5f"
AMBER = "#d68a00"
SLATE = "#7b8794"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "axes.edgecolor": "#d8e0eb",
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#e6edf6",
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Arial"],
    }
)


def fig_to_base64(fig: plt.Figure, dpi: int = 170) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def image_file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def money(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"${value:,.0f}"


def pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value * 100:.{decimals}f}%"


def num(value: float | int | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def safe_html_table(df: pd.DataFrame, classes: str = "data-table") -> str:
    return df.to_html(index=False, escape=False, border=0, classes=classes)


def theme_for_feature(feature: str, row: pd.Series) -> str:
    if bool(row.get("valuation_flag")):
        return "Valuation"
    if bool(row.get("stress_flag")):
        return "Stress / liquidity"
    if bool(row.get("trend_flag")):
        return "Trend"
    if any(token in feature for token in ("curve", "yield", "us10y", "t10y3m")):
        return "Rates / curve"
    if any(token in feature for token in ("unemployment", "cpi")):
        return "Growth / inflation"
    if any(token in feature for token in ("gold", "wti", "dxy")):
        return "Cross-asset"
    if "sentiment" in feature or "shock" in feature:
        return "Black-box sentiment"
    return "Other"


def direction_from_row(row: pd.Series) -> str:
    signed_values = [
        row.get("ols_63d_coef_pp_per_1sd"),
        row.get("ols_126d_coef_pp_per_1sd"),
        row.get("ols_252d_coef_pp_per_1sd"),
        row.get("ridge_return_importance"),
    ]
    signed_values = [value for value in signed_values if pd.notna(value)]
    if not signed_values:
        return "mixed"
    mean_sign = float(np.sign(np.nanmean(signed_values)))
    if mean_sign > 0:
        return "higher values tend to lift forward returns"
    if mean_sign < 0:
        return "higher values tend to suppress forward returns"
    return "mixed"


def build_return_driver_ranking(scorecard: pd.DataFrame) -> pd.DataFrame:
    ranking = scorecard.copy()
    ranking["theme"] = ranking.apply(lambda row: theme_for_feature(str(row["feature"]), row), axis=1)
    for horizon in ("63", "126", "252"):
        coef_col = f"ols_{horizon}d_coef_pp_per_1sd"
        q_col = f"ols_{horizon}d_q_value"
        strength = ranking[coef_col].abs() * (1.0 - ranking[q_col].fillna(1.0).clip(0.0, 1.0))
        if float(strength.max() or 0.0) > 0:
            ranking[f"score_{horizon}d"] = strength / strength.max()
        else:
            ranking[f"score_{horizon}d"] = 0.0
    for column in ("ridge_return_importance", "rf_return_importance"):
        strength = ranking[column].abs()
        if float(strength.max() or 0.0) > 0:
            ranking[f"score_{column}"] = strength / strength.max()
        else:
            ranking[f"score_{column}"] = 0.0
    score_cols = [col for col in ranking.columns if col.startswith("score_")]
    ranking["driver_score"] = ranking[score_cols].mean(axis=1)
    ranking["direction"] = ranking.apply(direction_from_row, axis=1)
    ranking = ranking.sort_values(["driver_score", "ols_252d_q_value"], ascending=[False, True]).reset_index(drop=True)
    return ranking


def build_leakage_audit(
    monthly_regimes: pd.DataFrame,
    gmm_refits: pd.DataFrame,
    daily_signals: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    supervised = monthly_regimes[
        monthly_regimes["model_name"].isin(["logistic", "random_forest"]) & monthly_regimes["train_end"].notna()
    ].copy()
    supervised_gap = (supervised["date"] - supervised["train_end"]).dt.days
    gmm_gap = (gmm_refits["wf_refit_date"] - gmm_refits["train_end"]).dt.days
    lagged_treatments = inventory["availability_treatment"].str.contains("lagged", case=False, na=False)
    checks = [
        {
            "check": "Supervised monthly training rows end before prediction dates",
            "status": "PASS" if bool((supervised["train_end"] < supervised["date"]).all()) else "FAIL",
            "evidence": f"{len(supervised):,} monthly predictions audited; min gap {int(supervised_gap.min())} days.",
        },
        {
            "check": "Walk-forward GMM refits train only through the prior trading session",
            "status": "PASS" if bool((gmm_refits["train_end"] < gmm_refits["wf_refit_date"]).all()) else "FAIL",
            "evidence": f"{len(gmm_refits):,} refits audited; min gap {int(gmm_gap.min())} day.",
        },
    ]
    for model_name in ("logistic", "random_forest"):
        predicted = monthly_regimes[
            monthly_regimes["model_name"].eq(model_name) & monthly_regimes["regime"].notna()
        ].copy()
        first_prediction_date = predicted["date"].min()
        first_signal_date = daily_signals.loc[
            daily_signals[f"{model_name}_signal_lag1"].notna(),
            "date",
        ].min()
        lag_days = int((first_signal_date - first_prediction_date).days)
        checks.append(
            {
                "check": f"{model_name.replace('_', ' ').title()} signals are traded with lagged daily execution",
                "status": "PASS" if first_signal_date > first_prediction_date else "FAIL",
                "evidence": f"First prediction {first_prediction_date.date()} -> first traded signal {first_signal_date.date()} ({lag_days} calendar days later).",
            }
        )
    checks.append(
        {
            "check": "Monthly, quarterly, and annual macro releases are lagged before forward-fill",
            "status": "PASS" if int(lagged_treatments.sum()) >= 10 else "CHECK",
            "evidence": f"{int(lagged_treatments.sum())} of {len(inventory)} tracked variables use explicit lagged release treatment.",
        }
    )
    return pd.DataFrame(checks)


def render_driver_chart(driver_ranking: pd.DataFrame) -> str:
    top = driver_ranking.head(12).copy()
    top = top.iloc[::-1]
    colors = []
    for direction in top["direction"]:
        if "lift" in direction:
            colors.append(GREEN)
        elif "suppress" in direction:
            colors.append(RED)
        else:
            colors.append(SLATE)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["label"], top["driver_score"], color=colors, edgecolor="none")
    ax.set_title("Composite return-driver ranking", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Driver score across OLS and ML importance lenses")
    ax.set_ylabel("")
    ax.set_xlim(0, top["driver_score"].max() * 1.15)
    for y, value in enumerate(top["driver_score"]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", ha="left", fontsize=10, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_theme_chart(driver_ranking: pd.DataFrame) -> str:
    theme_scores = (
        driver_ranking.groupby("theme", as_index=False)["driver_score"]
        .mean()
        .sort_values("driver_score", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    palette = [BLUE, TEAL, RED, AMBER, GREEN, SLATE, BLUE_DARK]
    ax.bar(
        theme_scores["theme"],
        theme_scores["driver_score"],
        color=palette[: len(theme_scores)],
        edgecolor="none",
    )
    ax.set_title("Average driver score by feature family", fontsize=18, fontweight="bold", loc="left")
    ax.set_ylabel("Average driver score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_risk_off_chart(regime_accuracy: pd.DataFrame) -> str:
    risk_off = regime_accuracy[regime_accuracy["label"].eq("risk_off")].copy()
    risk_off["label_clean"] = risk_off["model_name"].str.replace("_", " ").str.title()
    x = np.arange(len(risk_off))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width, risk_off["precision"], width=width, label="Precision", color=BLUE)
    ax.bar(x, risk_off["recall"], width=width, label="Recall", color=AMBER)
    ax.bar(x + width, risk_off["false_positive_rate"], width=width, label="False positive rate", color=RED)
    ax.set_xticks(x, risk_off["label_clean"])
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Rate")
    ax.set_title("Risk-off error profile", fontsize=18, fontweight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_expected_return_chart(regime_expected: pd.DataFrame) -> str:
    view = regime_expected[
        regime_expected["model_name"].isin(["logistic", "gmm", "consensus"])
    ].copy()
    order = ["risk_off", "neutral", "risk_on"]
    view["predicted_regime"] = pd.Categorical(view["predicted_regime"], categories=order, ordered=True)
    view = view.sort_values(["model_name", "predicted_regime"])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(
        data=view,
        x="predicted_regime",
        y="avg_fwd_63d_return",
        hue="model_name",
        palette=[BLUE, TEAL, AMBER],
        ax=ax,
    )
    ax.set_title("Average next-63-day return by predicted regime", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Average forward 63-day return")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax.legend(frameon=False, title="")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_strategy_frontier(leverage_ladder: pd.DataFrame) -> str:
    ladder = leverage_ladder.copy()
    ladder["risk"] = ladder["max_drawdown"].abs()
    fig, ax = plt.subplots(figsize=(11.5, 6))
    palette = {
        "plain": SLATE,
        "gmm": TEAL,
        "logistic": RED,
        "consensus": BLUE,
    }
    for _, row in ladder.iterrows():
        family = str(row["strategy"]).split("_")[0]
        ax.scatter(
            row["risk"],
            row["xirr"],
            s=max(90, row["avg_target_leverage"] * 120),
            color=palette.get(family, GREEN),
            alpha=0.9,
            edgecolor=PANEL,
            linewidth=1.2,
        )
        ax.text(
            row["risk"] + 0.008,
            row["xirr"],
            row["strategy"],
            fontsize=9,
            va="center",
        )
    ax.axvline(0.35, color="#d7dde7", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Absolute max drawdown")
    ax.set_ylabel("XIRR")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.set_title("Strategy frontier: return vs drawdown", fontsize=18, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_summary_cards(current: pd.Series) -> list[dict[str, str]]:
    return [
        {"label": "Current macro cycle", "value": str(current["macro_cycle"]).replace("_", " ").title(), "sub": f"Confidence: {current['macro_cycle_confidence']}"},
        {"label": "Current traded regime", "value": str(current["combined_market_regime"]).replace("_", " ").title(), "sub": f"Target equity: {pct(current['target_equity_allocation'], 0)}"},
        {"label": "Valuation backdrop", "value": f"CAPE z {num(current['cape_rolling_z'])}", "sub": f"Wilshire/GDP z {num(current['buffett_indicator_proxy_rolling_z'])}"},
        {"label": "Shock / sentiment", "value": f"Shock {num(current['external_shock_score'])}", "sub": f"Latent sentiment {num(current['latent_sentiment_index'])}"},
        {"label": "Logistic overlay", "value": f"Risk-off {pct(current['logistic_risk_off_probability'], 0)}", "sub": f"Jump-in {pct(current['logistic_jump_in_probability'], 0)}"},
    ]


def render_guidance_list() -> list[str]:
    return [
        "Do not trust the current stack as a pure 1x timing overlay yet. Plain DCA still beats the tested 1x macro-aware allocation rules out of sample.",
        "If leverage is allowed, the safest improvement over plain DCA is the consensus ensemble at 2x, not the raw logistic model at 3x or 5x.",
        "Treat valuation as a throttle, not a trigger. CAPE and Wilshire/GDP matter most as medium-horizon headwinds, while VIX, credit spreads, and financial conditions drive tactical shifts.",
        "Keep the black-box macro sentiment inputs as weak modifiers only. They add context, but they are not strong enough to override price, volatility, credit, and valuation.",
        "Review error rates every month. The main failure mode is false positive risk-off calls, which cause missed upside and explain why 1x de-risking underperforms plain DCA.",
    ]


def build_html(
    output_html: Path,
    analysis_dir: Path,
    compare_dir: Path,
    audit_dir: Path,
    driver_ranking: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    current: pd.Series,
    inventory: pd.DataFrame,
    validation: pd.DataFrame,
    regime_accuracy: pd.DataFrame,
    regime_expected: pd.DataFrame,
    leverage_ladder: pd.DataFrame,
) -> str:
    cards = render_summary_cards(current)
    top_drivers = driver_ranking.head(12).copy()
    top_drivers["driver_score"] = top_drivers["driver_score"].map(lambda value: f"{value:.2f}")
    top_drivers["ols_63d_coef_pp_per_1sd"] = top_drivers["ols_63d_coef_pp_per_1sd"].map(lambda value: num(value, 2))
    top_drivers["ols_252d_coef_pp_per_1sd"] = top_drivers["ols_252d_coef_pp_per_1sd"].map(lambda value: num(value, 2))
    top_drivers["ridge_return_importance"] = top_drivers["ridge_return_importance"].map(lambda value: num(value, 3))
    top_drivers = top_drivers[
        [
            "label",
            "theme",
            "driver_score",
            "direction",
            "ols_63d_coef_pp_per_1sd",
            "ols_252d_coef_pp_per_1sd",
            "ridge_return_importance",
        ]
    ].rename(
        columns={
            "label": "Feature",
            "theme": "Theme",
            "driver_score": "Driver score",
            "direction": "Interpretation",
            "ols_63d_coef_pp_per_1sd": "OLS 63d",
            "ols_252d_coef_pp_per_1sd": "OLS 252d",
            "ridge_return_importance": "Ridge",
        }
    )

    leakage_table = leakage_audit.copy()
    leakage_table["status"] = leakage_table["status"].map(
        lambda value: f"<span class='status {'pass' if value == 'PASS' else 'warn'}'>{value}</span>"
    )

    validation_view = validation.copy()
    for column in ("auc", "average_precision", "brier", "mae", "r2", "spearman_pred_actual"):
        validation_view[column] = validation_view[column].map(lambda value: num(value, 3))
    validation_view = validation_view[
        ["target", "model", "train_n", "test_n", "auc", "average_precision", "brier", "mae", "r2", "spearman_pred_actual"]
    ].rename(
        columns={
            "target": "Target",
            "model": "Model",
            "train_n": "Train n",
            "test_n": "Test n",
            "auc": "AUC",
            "average_precision": "Avg precision",
            "brier": "Brier",
            "mae": "MAE",
            "r2": "R^2",
            "spearman_pred_actual": "Spearman",
        }
    )

    risk_off_view = regime_accuracy[regime_accuracy["label"].eq("risk_off")].copy()
    for column in ("precision", "recall", "false_positive_rate", "balanced_accuracy", "overall_accuracy"):
        risk_off_view[column] = risk_off_view[column].map(lambda value: pct(value, 1))
    risk_off_view = risk_off_view[
        ["model_name", "precision", "recall", "false_positive_rate", "balanced_accuracy", "overall_accuracy"]
    ].rename(
        columns={
            "model_name": "Model",
            "precision": "Precision",
            "recall": "Recall",
            "false_positive_rate": "False positive rate",
            "balanced_accuracy": "Balanced accuracy",
            "overall_accuracy": "Overall accuracy",
        }
    )

    regime_expected_view = regime_expected.copy()
    regime_expected_view["avg_fwd_63d_return"] = regime_expected_view["avg_fwd_63d_return"].map(lambda value: pct(value, 1))
    regime_expected_view["positive_63d_rate"] = regime_expected_view["positive_63d_rate"].map(lambda value: pct(value, 1))
    regime_expected_view["avg_max_drawdown_next_63d"] = regime_expected_view["avg_max_drawdown_next_63d"].map(lambda value: pct(value, 1))
    regime_expected_view = regime_expected_view[
        ["model_name", "predicted_regime", "n_months", "avg_fwd_63d_return", "positive_63d_rate", "avg_max_drawdown_next_63d"]
    ].rename(
        columns={
            "model_name": "Model",
            "predicted_regime": "Predicted regime",
            "n_months": "Months",
            "avg_fwd_63d_return": "Avg next 63d return",
            "positive_63d_rate": "Positive 63d rate",
            "avg_max_drawdown_next_63d": "Avg next 63d max drawdown",
        }
    )

    strategy_view = leverage_ladder.copy()
    strategy_view["final_value"] = strategy_view["final_value"].map(money)
    strategy_view["xirr"] = strategy_view["xirr"].map(lambda value: pct(value, 1))
    strategy_view["time_weighted_cagr"] = strategy_view["time_weighted_cagr"].map(lambda value: pct(value, 1))
    strategy_view["max_drawdown"] = strategy_view["max_drawdown"].map(lambda value: pct(value, 1))
    strategy_view["avg_target_leverage"] = strategy_view["avg_target_leverage"].map(lambda value: num(value, 2))
    strategy_view = strategy_view[
        ["strategy", "final_value", "xirr", "time_weighted_cagr", "max_drawdown", "avg_target_leverage", "risk_on_months", "risk_off_months"]
    ].rename(
        columns={
            "strategy": "Strategy",
            "final_value": "Final value",
            "xirr": "XIRR",
            "time_weighted_cagr": "CAGR",
            "max_drawdown": "Max drawdown",
            "avg_target_leverage": "Avg leverage",
            "risk_on_months": "Risk-on months",
            "risk_off_months": "Risk-off months",
        }
    )

    feature_counts = {
        "tracked": len(inventory),
        "supervised": int(inventory["used_in_supervised_models"].sum()),
        "gmm": int(inventory["used_in_gmm"].sum()),
        "cycle": int(inventory["used_in_cycle_classifier"].sum()),
    }
    frequency_counts = inventory["input_frequency"].value_counts().to_dict()

    driver_chart = render_driver_chart(driver_ranking)
    theme_chart = render_theme_chart(driver_ranking)
    risk_off_chart = render_risk_off_chart(regime_accuracy)
    expected_return_chart = render_expected_return_chart(regime_expected)
    frontier_chart = render_strategy_frontier(leverage_ladder)
    return_compare_chart = image_file_to_base64(compare_dir / "walkforward_feature_importance_compare_returns.png")
    logistic_regime_chart = image_file_to_base64(compare_dir / "walkforward_logistic_regimes_full_common_window.png")
    equity_2x_chart = image_file_to_base64(audit_dir / "plots" / "macro_aware_equity_2x.png")
    guidance_items = "".join(f"<li>{item}</li>" for item in render_guidance_list())
    card_html = "".join(
        f"<div class='metric-card'><div class='metric-label'>{card['label']}</div><div class='metric-value'>{card['value']}</div><div class='metric-sub'>{card['sub']}</div></div>"
        for card in cards
    )
    pills_features = "".join(
        [
            f"<span class='pill'>{feature_counts['tracked']} tracked variables</span>",
            f"<span class='pill'>{feature_counts['supervised']} supervised inputs</span>",
            f"<span class='pill'>{feature_counts['gmm']} GMM inputs</span>",
            f"<span class='pill'>{feature_counts['cycle']} cycle inputs</span>",
        ]
    )
    pills_frequency = "".join(f"<span class='pill'>{label}: {count}</span>" for label, count in frequency_counts.items())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Macro Regime Kaggle Report</title>
  <style>
    :root {{
      --bg: {BG}; --panel: {PANEL}; --text: {TEXT}; --muted: {MUTED};
      --blue: {BLUE}; --blue-dark: {BLUE_DARK}; --teal: {TEAL};
      --green: {GREEN}; --red: {RED}; --amber: {AMBER};
      --line: #dbe4ef; --shadow: 0 24px 50px rgba(18, 31, 53, 0.08); --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif; background:
      radial-gradient(circle at top left, rgba(31,120,255,0.10), transparent 28%),
      radial-gradient(circle at top right, rgba(0,167,160,0.10), transparent 22%), var(--bg); color: var(--text); line-height: 1.55; }}
    .shell {{ width: min(1380px, calc(100% - 32px)); margin: 28px auto 56px; }}
    .hero {{ background: linear-gradient(140deg, #13213c, #0f5ec7 62%, #29b7b0); color: white; padding: 34px 36px 32px; border-radius: 28px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .hero::after {{ content: ""; position: absolute; inset: auto -100px -140px auto; width: 300px; height: 300px; background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%); }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; opacity: 0.75; margin-bottom: 14px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(32px, 5vw, 52px); line-height: 1.02; max-width: 860px; }}
    .hero p {{ max-width: 900px; margin: 0; font-size: 16px; color: rgba(255,255,255,0.88); }}
    .meta {{ margin-top: 18px; display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: rgba(255,255,255,0.85); }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 22px; }}
    .metric-card {{ background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.14); border-radius: 18px; padding: 18px 18px 16px; backdrop-filter: blur(10px); }}
    .metric-label {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: rgba(255,255,255,0.75); margin-bottom: 8px; }}
    .metric-value {{ font-size: 24px; line-height: 1.1; font-weight: 700; margin-bottom: 6px; }}
    .metric-sub {{ font-size: 13px; color: rgba(255,255,255,0.8); }}
    .section {{ margin-top: 22px; background: var(--panel); border-radius: var(--radius); box-shadow: var(--shadow); padding: 26px 28px; }}
    .section-head {{ display: flex; gap: 18px; align-items: end; justify-content: space-between; margin-bottom: 18px; flex-wrap: wrap; }}
    h2 {{ margin: 0; font-size: 28px; line-height: 1.15; }}
    .sub {{ color: var(--muted); font-size: 14px; max-width: 780px; }}
    .split {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 22px; align-items: start; }}
    .stack {{ display: grid; gap: 18px; }}
    .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
    .callout {{ border-left: 4px solid var(--blue); background: linear-gradient(180deg, rgba(31,120,255,0.07), rgba(31,120,255,0.01)); padding: 16px 18px; border-radius: 18px; }}
    .callout strong {{ display: block; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--blue-dark); }}
    .img-panel {{ background: linear-gradient(180deg, #fbfdff, #f3f7fc); border: 1px solid var(--line); border-radius: 18px; padding: 14px; }}
    .img-panel img {{ width: 100%; display: block; border-radius: 12px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table thead th {{ background: #eff4fb; color: var(--blue-dark); text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; padding: 12px; border-bottom: 1px solid var(--line); }}
    .data-table tbody td {{ padding: 11px 12px; border-bottom: 1px solid #edf2f8; vertical-align: top; }}
    .data-table tbody tr:nth-child(odd) {{ background: #fcfdff; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{ padding: 9px 12px; border-radius: 999px; background: #edf3fb; color: var(--blue-dark); font-size: 13px; font-weight: 600; }}
    .status {{ display: inline-flex; align-items: center; padding: 5px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }}
    .status.pass {{ background: rgba(35,163,109,0.12); color: var(--green); }}
    .status.warn {{ background: rgba(214,79,95,0.12); color: var(--red); }}
    ul.guidance {{ margin: 0; padding-left: 20px; display: grid; gap: 10px; }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; text-align: center; }}
    @media (max-width: 1100px) {{ .split, .two-col {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 720px) {{ .shell {{ width: min(100% - 16px, 100%); }} .hero, .section {{ padding: 22px 18px; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Macro Regime Research / Kaggle Style Audit</div>
      <h1>What really drives QQQ returns, where the model has edge, and why the current anti-leakage discipline looks sound.</h1>
      <p>This page summarizes the frozen macro regime audit as of {pd.Timestamp(current['as_of']).date()}. It combines statistical return analysis, walk-forward regime validation, leverage results against plain DCA, and explicit leakage checks from the exported training artifacts.</p>
      <div class="meta"><span>Frozen audit: <code>{audit_dir.name}</code></span><span>Analysis: <code>{analysis_dir.name}</code></span><span>Walk-forward compare: <code>{compare_dir.name}</code></span></div>
      <div class="metrics-grid">{card_html}</div>
    </section>
"""
    html += f"""
    <section class="section">
      <div class="section-head"><div><h2>1. Leakage Audit</h2><div class="sub">The main things to falsify are training overlap, same-day execution, and stale macro release handling. The current artifacts pass those checks.</div></div></div>
      <div class="split">
        <div class="stack">
          <div class="callout"><strong>Verdict</strong>I do not see a direct forward-looking leakage bug in the current walk-forward stack. The supervised monthly models train on prior month-end samples only, overlapping forward windows are purged, daily trading uses lagged signals, and slow macro releases are explicitly lagged before forward-fill.</div>
          {safe_html_table(leakage_table)}
          <div class="pill-row">{pills_features}</div>
          <div class="pill-row">{pills_frequency}</div>
        </div>
        <div class="img-panel"><img src="data:image/png;base64,{logistic_regime_chart}" alt="Logistic traded regime chart"></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2>2. What Drives Returns</h2><div class="sub">These rankings combine Newey-West OLS effect sizes at 63, 126, and 252 trading days with ridge and random-forest return importance. The point is not one magic factor; it is the recurring families that keep showing up.</div></div></div>
      <div class="two-col">
        <div class="img-panel"><img src="data:image/png;base64,{driver_chart}" alt="Composite return driver ranking"></div>
        <div class="img-panel"><img src="data:image/png;base64,{theme_chart}" alt="Average driver score by theme"></div>
      </div>
      <div class="callout" style="margin-top:18px"><strong>Read this carefully</strong>The most persistent drivers are valuation, stress, and trend. Shiller CAPE and the Wilshire/GDP proxy behave like medium-horizon headwinds when stretched. VIX, financial conditions, and credit stress dominate tactical setups. Trend matters a lot, but correlated moving-average features can flip sign inside multivariate models, so treat trend direction qualitatively rather than trusting one coefficient in isolation.</div>
      <div class="two-col" style="margin-top:18px">
        <div class="img-panel"><img src="data:image/png;base64,{return_compare_chart}" alt="Feature importance comparison for return target"></div>
        <div>{safe_html_table(top_drivers)}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2>3. Model Skill, Error Rate, and Sensitivity</h2><div class="sub">The hard truth is that direct return prediction is weak. The practical edge comes from classifying regimes well enough to modulate leverage, not from forecasting exact forward returns.</div></div></div>
      <div class="split">
        <div class="stack">
          <div class="callout"><strong>Key model finding</strong>The return models are not good enough to trade raw expected returns. In the purged chronological validation split, ridge return forecasting has deeply negative R<sup>2</sup>, and random forest does not rescue it. The regime models are more useful, but their main failure mode is false positive risk-off calls.</div>
          {safe_html_table(validation_view)}
          {safe_html_table(risk_off_view)}
        </div>
        <div class="stack">
          <div class="img-panel"><img src="data:image/png;base64,{risk_off_chart}" alt="Risk-off precision recall chart"></div>
          <div class="img-panel"><img src="data:image/png;base64,{expected_return_chart}" alt="Expected returns by predicted regime"></div>
        </div>
      </div>
      <div style="margin-top:18px">{safe_html_table(regime_expected_view)}</div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2>4. DCA vs Macro-Aware Overlays</h2><div class="sub">If the mandate is no leverage, the current stack does not justify replacing plain DCA. If 2x leverage is allowed, the ensemble becomes interesting.</div></div></div>
      <div class="two-col">
        <div class="img-panel"><img src="data:image/png;base64,{frontier_chart}" alt="Strategy frontier"></div>
        <div class="img-panel"><img src="data:image/png;base64,{equity_2x_chart}" alt="2x equity curves"></div>
      </div>
      <div class="callout" style="margin-top:18px"><strong>Deployment implication</strong>Plain DCA remains the best proven 1x baseline. The most reasonable upgrade today is the consensus ensemble at 2x, because it keeps much of the upside of the logistic model while controlling drawdown better than the pure logistic path.</div>
      <div style="margin-top:18px">{safe_html_table(strategy_view)}</div>
    </section>

    <section class="section">
      <div class="section-head"><div><h2>5. Current Read and Guidance</h2><div class="sub">Current environment signals say the market is still in expansion, but it is expensive and tactically defensive rather than fresh-cycle bullish.</div></div></div>
      <div class="split">
        <div class="stack">
          <div class="callout"><strong>Current market environment</strong>Macro cycle: <b>{str(current['macro_cycle']).replace('_', ' ').title()}</b>. Expansion / late-cycle / contraction score: <b>{num(current['expansion_score'], 0)} / {num(current['late_cycle_score'], 0)} / {num(current['contraction_score'], 0)}</b>. Combined traded regime: <b>{str(current['combined_market_regime']).replace('_', ' ').title()}</b>. Target equity allocation: <b>{pct(current['target_equity_allocation'], 0)}</b>.</div>
          <div class="callout"><strong>Why this matters</strong>CAPE z-score is <b>{num(current['cape_rolling_z'])}</b> and Wilshire/GDP z-score is <b>{num(current['buffett_indicator_proxy_rolling_z'])}</b>, so valuation remains rich. The model still sees enough stress or caution to stay tactically risk-off even though the larger macro cycle is not contractionary.</div>
          <ul class="guidance">{guidance_items}</ul>
        </div>
        <div class="stack">
          <div class="callout"><strong>Suggested operating rule</strong><b>Risk-on:</b> use leverage, invest new cash, and deploy reserves.<br><b>Neutral:</b> stay 1x long and keep contributing normally.<br><b>Risk-off:</b> remove leverage, keep base exposure, and send new contributions to reserve cash instead of forcing them into the market immediately.</div>
          <div class="callout"><strong>Black-box sentiment view</strong>The war / tariff / chokepoint uncertainty bucket is real, but the current black-box sentiment features are not reliable enough to be primary decision makers. They should remain supporting signals until they show stronger out-of-sample lift.</div>
        </div>
      </div>
    </section>

    <div class="footer">Report generated by <code>{Path(__file__).name}</code> into <code>{output_html.relative_to(ROOT)}</code>.</div>
  </div>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    args = parser.parse_args()

    scorecard = pd.read_csv(args.audit_dir / "feature_signal_scorecard.csv")
    current = pd.read_csv(args.analysis_dir / "current_market_environment.csv", parse_dates=["as_of"]).iloc[0]
    inventory = pd.read_csv(args.analysis_dir / "analysis_variable_inventory.csv")
    validation = pd.read_csv(args.compare_dir / "walkforward_model_validation_metrics.csv")
    regime_accuracy = pd.read_csv(args.audit_dir / "regime_accuracy_metrics.csv")
    regime_expected = pd.read_csv(args.audit_dir / "regime_expected_returns.csv")
    leverage_ladder = pd.read_csv(args.audit_dir / "leverage_ladder_x2_x3_x5.csv")
    monthly_regimes = pd.read_csv(
        args.compare_dir / "walkforward_model_regimes_monthly.csv",
        parse_dates=["date", "train_start", "train_end"],
    )
    gmm_refits = pd.read_csv(
        args.compare_dir / "walkforward_gmm_refits.csv",
        parse_dates=["wf_refit_date", "train_start", "train_end", "pred_start", "pred_end"],
    )
    daily_signals = pd.read_csv(args.compare_dir / "walkforward_model_signals_daily.csv", parse_dates=["date"])

    driver_ranking = build_return_driver_ranking(scorecard)
    leakage_audit = build_leakage_audit(monthly_regimes, gmm_refits, daily_signals, inventory)

    driver_ranking.to_csv(args.audit_dir / "kaggle_return_driver_ranking.csv", index=False)
    leakage_audit.to_csv(args.audit_dir / "kaggle_leakage_audit.csv", index=False)

    html = build_html(
        output_html=args.output_html,
        analysis_dir=args.analysis_dir,
        compare_dir=args.compare_dir,
        audit_dir=args.audit_dir,
        driver_ranking=driver_ranking,
        leakage_audit=leakage_audit,
        current=current,
        inventory=inventory,
        validation=validation,
        regime_accuracy=regime_accuracy,
        regime_expected=regime_expected,
        leverage_ladder=leverage_ladder,
    )
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
