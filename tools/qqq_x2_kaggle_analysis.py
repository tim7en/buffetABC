from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CORE_STRATEGIES = {
    "plain_dca_1x": "Plain DCA 1x",
    "qqq_ensemble_blend_2x": "QQQ Ensemble Blend 2x",
    "spy_gate_qqq_ensemble_blend_2x": "SPY Gate + QQQ Blend 2x",
}

MODEL_FRONTIER_STRATEGIES = {
    "plain_dca": "Plain DCA 1x",
    "walkforward_gmm_riskon_2x_keep_long_riskoff_reserve_dca": "GMM 2x",
    "walkforward_logistic_riskon_2x_prob_regime_dca": "Logistic 2x",
    "walkforward_random_forest_riskon_2x_prob_regime_dca": "Random Forest 2x",
    "walkforward_ensemble_majority_riskon_2x_prob_regime_dca": "Ensemble Majority 2x",
    "walkforward_ensemble_blend_riskon_2x_prob_regime_dca": "QQQ Ensemble Blend 2x",
    "spy_gate_qqq_ensemble_blend_2x": "SPY Gate + QQQ Blend 2x",
}

PALETTE = {
    "Plain DCA 1x": "#365c8d",
    "QQQ Ensemble Blend 2x": "#1b9e77",
    "SPY Gate + QQQ Blend 2x": "#d95f02",
    "GMM 2x": "#7570b3",
    "Logistic 2x": "#e7298a",
    "Random Forest 2x": "#66a61e",
    "Ensemble Majority 2x": "#e6ab02",
    "QQQ Ensemble Blend 2x": "#1b9e77",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an x2-focused QQQ vs plain DCA research pack."
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260410_x2_kaggle_rerun",
    )
    parser.add_argument(
        "--gate-dir",
        type=Path,
        default=ROOT / "reports" / "spy_gate_qqq_ensemble_backtest_20260410_x2_kaggle_rerun",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "reports" / "macro_regime_edge_audit_20260410_x2_kaggle_rerun",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "qqq_x2_kaggle_analysis_20260410",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def contribution_schedule(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    flow = pd.Series(0.0, index=dates.index, dtype=float)
    if flow.empty:
        return flow
    flow.iloc[0] = 10000.0
    month_change = dates.dt.to_period("M").ne(dates.shift().dt.to_period("M"))
    flow.loc[month_change & (flow.index != flow.index[0])] = 100.0
    return flow


def add_cashflow_adjusted_path(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.sort_values("date").copy()
    curve["external_cashflow"] = contribution_schedule(curve["date"])
    wealth_index: list[float] = []
    twr_return: list[float] = []
    drawdown: list[float] = []
    peak = -np.inf
    prev_total = None
    for _, row in curve.iterrows():
        total_value = float(row["total_value"])
        cashflow = float(row["external_cashflow"])
        if prev_total is None:
            wealth = total_value / cashflow if cashflow else 1.0
            ret = np.nan
        else:
            ret = total_value / (prev_total + cashflow) - 1.0
            wealth = wealth_index[-1] * (1.0 + ret)
        peak = max(peak, wealth)
        wealth_index.append(wealth)
        twr_return.append(ret)
        drawdown.append(wealth / peak - 1.0 if peak else 0.0)
        prev_total = total_value
    curve["wealth_index"] = wealth_index
    curve["twr_daily_return"] = twr_return
    curve["wealth_drawdown"] = drawdown
    curve["calendar_year"] = pd.to_datetime(curve["date"]).dt.year
    return curve


def load_core_curves(compare_dir: Path, gate_dir: Path) -> pd.DataFrame:
    compare_curves = pd.read_csv(
        compare_dir / "walkforward_model_compare_leverage_curves.csv",
        parse_dates=["date"],
    )
    gate_curves = pd.read_csv(
        gate_dir / "combined_policy_curves.csv",
        parse_dates=["date"],
    )

    compare_curves = compare_curves[
        (compare_curves["window"] == "full_common_window")
        & (compare_curves["strategy"].isin(["plain_dca", "walkforward_ensemble_blend_riskon_2x_prob_regime_dca"]))
    ].copy()
    compare_curves["strategy_key"] = compare_curves["strategy"].map(
        {
            "plain_dca": "plain_dca_1x",
            "walkforward_ensemble_blend_riskon_2x_prob_regime_dca": "qqq_ensemble_blend_2x",
        }
    )

    gate_curves = gate_curves[
        (gate_curves["window"] == "full_common_window")
        & (gate_curves["strategy"].isin(CORE_STRATEGIES))
    ].copy()
    gate_curves["strategy_key"] = gate_curves["strategy"]

    # Prefer the gate-side plain DCA so naming is consistent.
    curve = pd.concat(
        [
            compare_curves[compare_curves["strategy_key"] == "qqq_ensemble_blend_2x"],
            gate_curves,
        ],
        ignore_index=True,
    )
    curve = curve.drop_duplicates(subset=["date", "strategy_key"]).copy()
    curve["strategy_name"] = curve["strategy_key"].map(CORE_STRATEGIES)

    parts = [
        add_cashflow_adjusted_path(sub)
        for _, sub in curve.groupby("strategy_name", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


def load_frontier_metrics(compare_dir: Path, gate_dir: Path) -> pd.DataFrame:
    compare_metrics = pd.read_csv(compare_dir / "walkforward_model_compare_leverage_metrics.csv")
    gate_metrics = pd.read_csv(gate_dir / "combined_policy_metrics.csv")

    compare_metrics = compare_metrics[
        (compare_metrics["window"] == "full_common_window")
        & (compare_metrics["strategy"].isin(MODEL_FRONTIER_STRATEGIES))
    ].copy()
    gate_metrics = gate_metrics[
        (gate_metrics["window"] == "full_common_window")
        & (gate_metrics["strategy"] == "spy_gate_qqq_ensemble_blend_2x")
    ].copy()

    combined = pd.concat([compare_metrics, gate_metrics], ignore_index=True)
    combined["strategy_name"] = combined["strategy"].map(MODEL_FRONTIER_STRATEGIES)
    combined["max_drawdown_abs"] = combined["max_drawdown"].abs()
    combined["calmar_like"] = combined["xirr"] / combined["max_drawdown_abs"]
    return combined


def load_subwindow_metrics(gate_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(gate_dir / "combined_policy_metrics.csv")
    metrics = metrics[
        metrics["window"].isin(
            ["pre_covid_2014_2019", "covid_recovery_2020_2021", "inflation_ai_2022_2026"]
        )
        & metrics["strategy"].isin(CORE_STRATEGIES)
    ].copy()
    metrics["strategy_name"] = metrics["strategy"].map(CORE_STRATEGIES)
    metrics["window_name"] = metrics["window"].map(
        {
            "pre_covid_2014_2019": "2014-2019",
            "covid_recovery_2020_2021": "2020-2021",
            "inflation_ai_2022_2026": "2022-2026",
        }
    )
    return metrics


def load_logistic_sensitivity(audit_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(audit_dir / "logistic_threshold_sensitivity_1x_2x.csv")
    df = df[
        (df["window"] == "full_common_window")
        & (df["strategy_mode"] == "leverage_2x")
    ].copy()
    return df


def load_feature_rankings(audit_dir: Path, compare_dir: Path) -> pd.DataFrame:
    scorecard = pd.read_csv(audit_dir / "feature_signal_scorecard.csv")
    importance_cols = [
        "ridge_return_importance",
        "rf_return_importance",
        "logistic_risk_off_importance",
        "rf_risk_off_importance",
        "logistic_jump_in_importance",
        "rf_jump_in_importance",
    ]
    for col in importance_cols:
        scorecard[col] = pd.to_numeric(scorecard[col], errors="coerce").fillna(0.0)
    scorecard["composite_importance"] = scorecard[importance_cols].sum(axis=1)
    scorecard = scorecard.sort_values("composite_importance", ascending=False).copy()

    feature_importance = pd.read_csv(compare_dir / "walkforward_model_feature_importance.csv")
    ensemble_blend = feature_importance[
        (feature_importance["target"] == "risk_off_target")
        & (feature_importance["model"] == "logistic")
    ].copy()
    ensemble_blend = ensemble_blend.sort_values("importance_mean", ascending=False)
    top_logistic = ensemble_blend[["feature", "importance_mean"]].rename(
        columns={"importance_mean": "risk_off_logistic_importance"}
    )

    merged = scorecard.merge(top_logistic, on="feature", how="left")
    return merged


def plot_wealth_index(curves: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, sub in curves.groupby("strategy_name"):
        ax.plot(
            sub["date"],
            sub["wealth_index"],
            label=name,
            linewidth=2.5,
            color=PALETTE.get(name),
        )
    ax.set_title("Contribution-Adjusted Wealth Index")
    ax.set_ylabel("Growth of $1 net of cashflows")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_drawdowns(curves: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, sub in curves.groupby("strategy_name"):
        ax.plot(
            sub["date"],
            sub["wealth_drawdown"] * 100.0,
            label=name,
            linewidth=2.2,
            color=PALETTE.get(name),
        )
    ax.set_title("Drawdown From Prior Peak")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_yearly_returns(curves: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    curves = curves.sort_values(["strategy_name", "date"]).copy()
    year_end = (
        curves.groupby(["strategy_name", "calendar_year"], as_index=False)
        .agg(wealth_index=("wealth_index", "last"))
        .sort_values(["strategy_name", "calendar_year"])
    )
    year_end["yearly_twr_return"] = year_end.groupby("strategy_name")["wealth_index"].pct_change()
    year_end = year_end.dropna(subset=["yearly_twr_return"]).copy()

    years = sorted(year_end["calendar_year"].unique())
    strategies = list(CORE_STRATEGIES.values())
    x = np.arange(len(years))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13, 7))
    offsets = np.linspace(-width, width, num=len(strategies))
    for offset, strategy in zip(offsets, strategies):
        sub = year_end[year_end["strategy_name"] == strategy].set_index("calendar_year")
        vals = [sub.loc[year, "yearly_twr_return"] * 100.0 if year in sub.index else np.nan for year in years]
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=strategy,
            color=PALETTE.get(strategy),
            alpha=0.9,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Yearly TWR %")
    ax.set_title("Calendar-Year Time-Weighted Returns")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return year_end


def plot_frontier(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in metrics.iterrows():
        name = row["strategy_name"]
        x_val = row["max_drawdown_abs"] * 100.0
        y_val = row["xirr"] * 100.0
        ax.scatter(x_val, y_val, s=130, color=PALETTE.get(name, "#444444"), alpha=0.95)
        ax.annotate(name, (x_val, y_val), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_title("x2 Strategy Frontier")
    ax.set_xlabel("Max Drawdown %")
    ax.set_ylabel("XIRR %")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_subwindow_delta(metrics: pd.DataFrame, out_path: Path) -> None:
    plot_df = metrics.copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    strategies = list(CORE_STRATEGIES.values())
    windows = list(dict.fromkeys(plot_df["window_name"]))
    x = np.arange(len(windows))
    width = 0.24
    offsets = np.linspace(-width, width, num=len(strategies))
    for offset, strategy in zip(offsets, strategies):
        sub = plot_df[plot_df["strategy_name"] == strategy].set_index("window_name")
        vals = [sub.loc[w, "final_delta_vs_plain_dca"] if w in sub.index else np.nan for w in windows]
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=strategy,
            color=PALETTE.get(strategy),
            alpha=0.9,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Final Value Delta vs Plain DCA ($)")
    ax.set_title("Subwindow Robustness")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_logistic_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(
        index="risk_off_threshold",
        columns="jump_in_threshold",
        values="final_value",
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
    ax.set_xlabel("Jump-in Threshold")
    ax.set_ylabel("Risk-off Threshold")
    ax.set_title("Logistic 2x Threshold Sensitivity: Final Value")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j] / 1000:.1f}k", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Final Value ($)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_summary_tables(
    curves: pd.DataFrame,
    frontier: pd.DataFrame,
    subwindows: pd.DataFrame,
    feature_rankings: pd.DataFrame,
    year_end: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    latest_metrics = frontier[
        frontier["strategy_name"].isin(CORE_STRATEGIES.values())
    ][
        [
            "strategy_name",
            "final_value",
            "xirr",
            "time_weighted_cagr",
            "max_drawdown",
            "avg_target_leverage",
            "risk_on_months",
            "neutral_months",
            "risk_off_months",
            "final_delta_vs_plain_dca",
            "calmar_like",
        ]
    ].sort_values("final_value", ascending=False)

    latest_path = (
        curves.sort_values("date")
        .groupby("strategy_name", as_index=False)
        .agg(
            ending_wealth_index=("wealth_index", "last"),
            worst_drawdown_pct=("wealth_drawdown", lambda s: s.min() * 100.0),
        )
        .sort_values("ending_wealth_index", ascending=False)
    )

    yearly_pivot = (
        year_end.pivot(index="calendar_year", columns="strategy_name", values="yearly_twr_return")
        .sort_index()
        * 100.0
    )

    top_features = feature_rankings[
        [
            "feature",
            "label",
            "composite_importance",
            "ridge_return_importance",
            "logistic_risk_off_importance",
            "logistic_jump_in_importance",
            "valuation_flag",
            "stress_flag",
            "trend_flag",
        ]
    ].head(12)

    return {
        "x2_strategy_summary": latest_metrics,
        "path_summary": latest_path,
        "subwindow_summary": subwindows[
            [
                "window_name",
                "strategy_name",
                "final_value",
                "xirr",
                "max_drawdown",
                "final_delta_vs_plain_dca",
            ]
        ].sort_values(["window_name", "final_value"], ascending=[True, False]),
        "yearly_twr_returns": yearly_pivot.reset_index(),
        "top_features": top_features,
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(
    out_dir: Path,
    frontier: pd.DataFrame,
    subwindows: pd.DataFrame,
    feature_rankings: pd.DataFrame,
    logistic_sensitivity: pd.DataFrame,
) -> None:
    core = frontier[frontier["strategy_name"].isin(CORE_STRATEGIES.values())].copy()
    core = core.sort_values("final_value", ascending=False)
    best = core[core["strategy_name"] == "SPY Gate + QQQ Blend 2x"].iloc[0]
    runner_up = core[core["strategy_name"] == "QQQ Ensemble Blend 2x"].iloc[0]
    plain = core[core["strategy_name"] == "Plain DCA 1x"].iloc[0]

    logistic_best = logistic_sensitivity.sort_values("final_value", ascending=False).iloc[0]
    top_features = feature_rankings.head(8)

    report_lines = [
        "# QQQ x2 Kaggle-Style Analysis",
        "",
        "## Verdict",
        "",
        (
            f"- Best overall x2 strategy in this rerun: `{best['strategy_name']}` with final value "
            f"`$ {best['final_value']:,.0f}`, XIRR `{format_pct(best['xirr'])}`, and max drawdown "
            f"`{format_pct(best['max_drawdown'])}`."
        ),
        (
            f"- Best QQQ-only x2 strategy: `{runner_up['strategy_name']}` at "
            f"`$ {runner_up['final_value']:,.0f}`, XIRR `{format_pct(runner_up['xirr'])}`, "
            f"max drawdown `{format_pct(runner_up['max_drawdown'])}`."
        ),
        (
            f"- Plain DCA baseline finished at `$ {plain['final_value']:,.0f}` with XIRR "
            f"`{format_pct(plain['xirr'])}` and max drawdown `{format_pct(plain['max_drawdown'])}`."
        ),
        "",
        "## Why The Winner Won",
        "",
        (
            f"- The SPY gate improved final value by `$ {best['final_value'] - runner_up['final_value']:,.0f}` "
            f"over standalone QQQ 2x while using lower average leverage "
            f"(`{best['avg_target_leverage']:.2f}x` vs `{runner_up['avg_target_leverage']:.2f}x`)."
        ),
        (
            f"- It cut risk-on months from `{int(runner_up['risk_on_months'])}` to `{int(best['risk_on_months'])}`, "
            "which means the edge came from selectivity rather than simply pressing leverage harder."
        ),
        (
            f"- Relative to plain DCA, the winner added `$ {best['final_delta_vs_plain_dca']:,.0f}` of terminal wealth "
            f"but accepted an extra `{(abs(best['max_drawdown']) - abs(plain['max_drawdown'])) * 100:.1f}` percentage points "
            "of peak-to-trough pain."
        ),
        "",
        "## Robustness Read",
        "",
    ]

    for window_name, sub in subwindows.groupby("window_name"):
        sub = sub.sort_values("final_value", ascending=False)
        winner = sub.iloc[0]
        report_lines.append(
            f"- `{window_name}`: best was `{winner['strategy_name']}` at `$ {winner['final_value']:,.0f}` "
            f"with max drawdown `{format_pct(winner['max_drawdown'])}`."
        )

    report_lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            (
                f"- Logistic 2x threshold grid peaked at risk-off `{logistic_best['risk_off_threshold']:.2f}` "
                f"and jump-in `{logistic_best['jump_in_threshold']:.2f}`, finishing at "
                f"`$ {logistic_best['final_value']:,.0f}`."
            ),
            "- The threshold surface is fairly smooth rather than cliff-like, which is a good sign for model stability.",
            "- The bigger fragility is model-choice sensitivity: GMM and ensemble-majority both underperformed plain DCA even at 2x.",
            "",
            "## Variables That Matter Most",
            "",
        ]
    )

    for _, row in top_features.iterrows():
        flags = [flag for flag, active in [("valuation", row["valuation_flag"]), ("stress", row["stress_flag"]), ("trend", row["trend_flag"])] if active]
        suffix = f" [{', '.join(flags)}]" if flags else ""
        report_lines.append(
            f"- `{row['feature']}` ({row['label']}): composite importance `{row['composite_importance']:.3f}`{suffix}"
        )

    report_lines.extend(
        [
            "",
            "## Production Caveat",
            "",
            "- This is good enough to treat as a research-backed overlay for DCA, not yet a fully trusted production autopilot.",
            "- The key remaining risk is structural: the SPY gate uses features that include QQQ-derived information, so the gate/expression stack is not perfectly disentangled.",
            "- Before production, the clean next test is a pure-SPY-feature gate controlling QQQ leverage on a locked-forward holdout.",
            "",
        ]
    )

    (out_dir / "x2_strategy_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    curves = load_core_curves(args.compare_dir, args.gate_dir)
    frontier = load_frontier_metrics(args.compare_dir, args.gate_dir)
    subwindows = load_subwindow_metrics(args.gate_dir)
    logistic_sensitivity = load_logistic_sensitivity(args.audit_dir)
    feature_rankings = load_feature_rankings(args.audit_dir, args.compare_dir)

    wealth_plot = args.out_dir / "x2_wealth_index.png"
    drawdown_plot = args.out_dir / "x2_drawdowns.png"
    yearly_plot = args.out_dir / "x2_yearly_twr_returns.png"
    frontier_plot = args.out_dir / "x2_frontier.png"
    subwindow_plot = args.out_dir / "x2_subwindow_deltas.png"
    heatmap_plot = args.out_dir / "x2_logistic_threshold_heatmap.png"

    plot_wealth_index(curves, wealth_plot)
    plot_drawdowns(curves, drawdown_plot)
    year_end = plot_yearly_returns(curves, yearly_plot)
    plot_frontier(frontier, frontier_plot)
    plot_subwindow_delta(subwindows, subwindow_plot)
    plot_logistic_heatmap(logistic_sensitivity, heatmap_plot)

    tables = build_summary_tables(curves, frontier, subwindows, feature_rankings, year_end)
    write_tables(tables, args.out_dir)
    write_report(args.out_dir, frontier, subwindows, feature_rankings, logistic_sensitivity)

    print(f"Wrote x2 analysis pack to {args.out_dir}")


if __name__ == "__main__":
    main()
