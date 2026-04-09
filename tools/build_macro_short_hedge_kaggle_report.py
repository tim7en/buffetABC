"""Build a Kaggle-style HTML report for the short-horizon hedge audit."""

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
DEFAULT_AUDIT_DIR = ROOT / "reports" / "macro_short_hedge_audit_20260409"
DEFAULT_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_OUTPUT_HTML = DEFAULT_AUDIT_DIR / "macro_short_hedge_kaggle_report.html"

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


def pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value * 100:.{decimals}f}%"


def money(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"${value:,.0f}"


def num(value: float | int | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def safe_html_table(df: pd.DataFrame, classes: str = "data-table") -> str:
    return df.to_html(index=False, escape=False, border=0, classes=classes)


def build_risk_driver_ranking(scorecard: pd.DataFrame) -> pd.DataFrame:
    ranking = scorecard.copy()
    ranking["theme"] = np.where(
        ranking["valuation_flag"],
        "Valuation",
        np.where(ranking["stress_flag"], "Stress / liquidity", np.where(ranking["trend_flag"], "Trend", "Other")),
    )
    for outcome in ["qqq_fwd_42d_min_return", "qqq_fwd_42d_path_cvar20"]:
        coef_col = f"{outcome}_coef_pp_per_1sd"
        q_col = f"{outcome}_q_value"
        strength = ranking[coef_col].abs() * (1.0 - ranking[q_col].fillna(1.0).clip(0.0, 1.0))
        ranking[f"score_{outcome}"] = 0.0 if float(strength.max() or 0.0) == 0.0 else strength / strength.max()
    for column in ["logistic_light_importance", "logistic_strong_importance", "rf_light_importance", "rf_strong_importance"]:
        strength = ranking[column].abs()
        ranking[f"score_{column}"] = 0.0 if float(strength.max() or 0.0) == 0.0 else strength / strength.max()
    ranking["risk_driver_score"] = ranking[[col for col in ranking.columns if col.startswith("score_")]].mean(axis=1)
    adjusted = (
        -ranking["qqq_fwd_42d_min_return_coef_pp_per_1sd"].fillna(0.0)
        -ranking["qqq_fwd_42d_path_cvar20_coef_pp_per_1sd"].fillna(0.0)
        + ranking["logistic_light_importance"].fillna(0.0)
        + ranking["logistic_strong_importance"].fillna(0.0)
    )
    ranking["direction"] = np.where(adjusted > 0.0, "higher values raise hedge risk", "higher values reduce hedge risk")
    return ranking.sort_values(["risk_driver_score", "qqq_fwd_42d_path_cvar20_q_value"], ascending=[False, True]).reset_index(drop=True)


def build_leakage_audit(monthly_predictions: pd.DataFrame, daily_signals: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    audit_rows = []
    for model_name in ["logistic", "random_forest"]:
        subset = monthly_predictions[monthly_predictions["model_name"].eq(model_name) & monthly_predictions["train_end"].notna()].copy()
        subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
        subset["train_end"] = pd.to_datetime(subset["train_end"], errors="coerce")
        gap = (subset["date"] - subset["train_end"]).dt.days
        audit_rows.append(
            {
                "check": f"{model_name.replace('_', ' ').title()} hedge training ends before each prediction date",
                "status": "PASS" if bool((subset["train_end"] < subset["date"]).all()) else "FAIL",
                "evidence": f"{len(subset):,} predictions audited; min gap {int(gap.min())} days." if not subset.empty else "No rows.",
            }
        )
        first_pred = subset["date"].min()
        signal_col = f"{model_name}_hedge_state_lag1"
        first_sig = daily_signals.loc[daily_signals[signal_col].notna(), "date"].min()
        audit_rows.append(
            {
                "check": f"{model_name.replace('_', ' ').title()} hedge states are traded with lagged daily execution",
                "status": "PASS" if pd.notna(first_pred) and pd.notna(first_sig) and first_sig > first_pred else "FAIL",
                "evidence": f"First prediction {first_pred.date()} -> first traded daily state {first_sig.date()}." if pd.notna(first_pred) and pd.notna(first_sig) else "No rows.",
            }
        )
    lagged = inventory["availability_treatment"].str.contains("lagged", case=False, na=False)
    audit_rows.append(
        {
            "check": "Slow macro variables still use lagged release treatment",
            "status": "PASS" if int(lagged.sum()) >= 10 else "CHECK",
            "evidence": f"{int(lagged.sum())} of {len(inventory)} tracked variables use explicit lagged treatment.",
        }
    )
    return pd.DataFrame(audit_rows)


def render_driver_chart(ranking: pd.DataFrame) -> str:
    top = ranking.head(12).iloc[::-1]
    colors = [RED if "raise" in direction else GREEN for direction in top["direction"]]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["label"], top["risk_driver_score"], color=colors, edgecolor="none")
    ax.set_title("Composite hedge-risk driver ranking", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Driver score across 42d min-return, 42d path-CVaR, and hedge-model importance lenses")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_strategy_frontier(metrics: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    palette = {"baseline": SLATE, "random": TEAL, "consensus": BLUE, "logistic": RED}
    for _, row in metrics.iterrows():
        strategy = str(row["strategy"])
        family = "baseline" if strategy.startswith("baseline") else strategy.split("_")[0]
        ax.scatter(abs(row["max_drawdown"]), row["xirr"], s=max(90, row["avg_target_beta"] * 90), color=palette.get(family, AMBER), alpha=0.9, edgecolor=PANEL, linewidth=1.2)
        ax.text(abs(row["max_drawdown"]) + 0.01, row["xirr"], strategy, fontsize=9, va="center")
    ax.set_xlabel("Absolute max drawdown")
    ax.set_ylabel("XIRR")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.set_title("Hedge strategies: return vs drawdown", fontsize=18, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def render_validation_chart(validation: pd.DataFrame) -> str:
    subset = validation[validation["target"].isin(["hedge_light_target", "hedge_strong_target"])].copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(subset))
    width = 0.25
    ax.bar(x - width, subset["auc"], width=width, color=BLUE, label="AUC")
    ax.bar(x, subset["average_precision"], width=width, color=AMBER, label="Avg precision")
    ax.bar(x + width, subset["precision_at_50pct"], width=width, color=TEAL, label="Precision@50")
    labels = [f"{row['model']}\n{row['target'].replace('_target','')}" for _, row in subset.iterrows()]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("Hedge-model validation snapshot", fontsize=18, fontweight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig_to_base64(fig)


def build_html(
    output_html: Path,
    audit_dir: Path,
    analysis_dir: Path,
    ranking: pd.DataFrame,
    leakage: pd.DataFrame,
    current_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    validation: pd.DataFrame,
    expected_stats: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    current = current_summary.iloc[0]
    target_row = target_summary.iloc[0]
    card_html = "".join(
        [
            f"<div class='metric-card'><div class='metric-label'>Consensus hedge state</div><div class='metric-value'>{current['consensus_hedge_state']}</div><div class='metric-sub'>as of {pd.Timestamp(current['as_of']).date()}</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Light hedge event rate</div><div class='metric-value'>{pct(target_row['hedge_light_rate'],0)}</div><div class='metric-sub'>42d path pain events</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Strong hedge event rate</div><div class='metric-value'>{pct(target_row['hedge_strong_rate'],0)}</div><div class='metric-sub'>21d acute stress events</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Best 1x hedge</div><div class='metric-value'>RF hedge</div><div class='metric-sub'>{money(metrics.loc[metrics['strategy'].eq('random_forest_hedge_base_1x'), 'final_value'].iloc[0])} final value</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Best 2x hedge</div><div class='metric-value'>RF hedge</div><div class='metric-sub'>{money(metrics.loc[metrics['strategy'].eq('random_forest_hedge_base_2x'), 'final_value'].iloc[0])} final value</div></div>",
        ]
    )
    top_drivers = ranking.head(12).copy()
    top_drivers["risk_driver_score"] = top_drivers["risk_driver_score"].map(lambda value: f"{value:.2f}")
    top_drivers = top_drivers[["label", "theme", "risk_driver_score", "direction"]].rename(columns={"label": "Feature", "theme": "Theme", "risk_driver_score": "Driver score", "direction": "Interpretation"})
    leakage_view = leakage.copy()
    leakage_view["status"] = leakage_view["status"].map(lambda value: f"<span class='status {'pass' if value == 'PASS' else 'warn'}'>{value}</span>")
    validation_view = validation[validation["target"].isin(["hedge_light_target", "hedge_strong_target"])].copy()
    for col in ["auc", "average_precision", "brier", "precision_at_50pct", "recall_at_50pct"]:
        validation_view[col] = validation_view[col].map(lambda value: num(value, 3))
    validation_view = validation_view[["target", "model", "train_n", "test_n", "auc", "average_precision", "brier", "precision_at_50pct", "recall_at_50pct"]]
    stats_view = expected_stats.copy()
    for col in ["avg_fwd_21d_return", "avg_fwd_42d_return", "avg_fwd_21d_min_return", "avg_fwd_42d_min_return", "avg_fwd_42d_path_cvar20", "strong_event_rate"]:
        stats_view[col] = stats_view[col].map(lambda value: pct(value, 1))
    stats_view = stats_view[["model_name", "predicted_state", "n_months", "avg_fwd_42d_return", "avg_fwd_42d_min_return", "avg_fwd_42d_path_cvar20", "strong_event_rate"]]
    metrics_view = metrics.copy()
    for col in ["final_value"]:
        metrics_view[col] = metrics_view[col].map(money)
    for col in ["xirr", "time_weighted_cagr", "max_drawdown"]:
        metrics_view[col] = metrics_view[col].map(lambda value: pct(value, 1))
    metrics_view["avg_target_beta"] = metrics_view["avg_target_beta"].map(lambda value: num(value, 2))
    metrics_view["final_delta_vs_same_beta_baseline"] = metrics_view["final_delta_vs_same_beta_baseline"].map(money)
    metrics_view = metrics_view[["strategy", "base_beta", "final_value", "xirr", "max_drawdown", "avg_target_beta", "final_delta_vs_same_beta_baseline"]]

    driver_chart = render_driver_chart(ranking)
    frontier_chart = render_strategy_frontier(metrics)
    validation_chart = render_validation_chart(validation)
    equity_1x = image_file_to_base64(audit_dir / "plots" / "hedge_equity_base_1x.png")
    equity_2x = image_file_to_base64(audit_dir / "plots" / "hedge_equity_base_2x.png")
    state_chart = image_file_to_base64(audit_dir / "plots" / "consensus_hedge_states_full_common_window.png")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Short-Horizon Hedge Kaggle Report</title>
<style>
:root{{--bg:{BG};--panel:{PANEL};--text:{TEXT};--muted:{MUTED};--blue:{BLUE};--blue-dark:{BLUE_DARK};--teal:{TEAL};--green:{GREEN};--red:{RED};--amber:{AMBER};--line:#dbe4ef;--shadow:0 24px 50px rgba(18,31,53,.08);--radius:22px;}}
*{{box-sizing:border-box}} body{{margin:0;font-family:"IBM Plex Sans","Segoe UI",Arial,sans-serif;background:radial-gradient(circle at top left,rgba(31,120,255,.10),transparent 28%),radial-gradient(circle at top right,rgba(0,167,160,.10),transparent 22%),var(--bg);color:var(--text);line-height:1.55}}
.shell{{width:min(1380px,calc(100% - 32px));margin:28px auto 56px}} .hero{{background:linear-gradient(140deg,#13213c,#0f5ec7 62%,#29b7b0);color:#fff;padding:34px 36px 32px;border-radius:28px;box-shadow:var(--shadow)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:12px;opacity:.75;margin-bottom:14px}} h1{{margin:0 0 12px;font-size:clamp(32px,5vw,52px);line-height:1.02;max-width:860px}} .hero p{{max-width:900px;margin:0;font-size:16px;color:rgba(255,255,255,.88)}}
.metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:22px}} .metric-card{{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.14);border-radius:18px;padding:18px 18px 16px}}
.metric-label{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.75);margin-bottom:8px}} .metric-value{{font-size:24px;line-height:1.1;font-weight:700;margin-bottom:6px}} .metric-sub{{font-size:13px;color:rgba(255,255,255,.8)}}
.section{{margin-top:22px;background:var(--panel);border-radius:var(--radius);box-shadow:var(--shadow);padding:26px 28px}} .section-head{{display:flex;gap:18px;align-items:end;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap}} h2{{margin:0;font-size:28px;line-height:1.15}} .sub{{color:var(--muted);font-size:14px;max-width:780px}}
.split{{display:grid;grid-template-columns:1.25fr 1fr;gap:22px;align-items:start}} .two-col{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}} .stack{{display:grid;gap:18px}}
.callout{{border-left:4px solid var(--blue);background:linear-gradient(180deg,rgba(31,120,255,.07),rgba(31,120,255,.01));padding:16px 18px;border-radius:18px}} .callout strong{{display:block;margin-bottom:8px;font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--blue-dark)}}
.img-panel{{background:linear-gradient(180deg,#fbfdff,#f3f7fc);border:1px solid var(--line);border-radius:18px;padding:14px}} .img-panel img{{width:100%;display:block;border-radius:12px}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px}} .data-table thead th{{background:#eff4fb;color:var(--blue-dark);text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.05em;padding:12px;border-bottom:1px solid var(--line)}} .data-table tbody td{{padding:11px 12px;border-bottom:1px solid #edf2f8;vertical-align:top}} .data-table tbody tr:nth-child(odd){{background:#fcfdff}}
.status{{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;font-weight:700;font-size:12px;letter-spacing:.05em;text-transform:uppercase}} .status.pass{{background:rgba(35,163,109,.12);color:var(--green)}} .status.warn{{background:rgba(214,79,95,.12);color:var(--red)}}
ul.guidance{{margin:0;padding-left:20px;display:grid;gap:10px}} .footer{{margin-top:22px;color:var(--muted);font-size:13px;text-align:center}}
@media (max-width:1100px){{.split,.two-col{{grid-template-columns:1fr}}}} @media (max-width:720px){{.shell{{width:min(100% - 16px,100%)}} .hero,.section{{padding:22px 18px}}}}
</style></head><body><div class="shell">
<section class="hero"><div class="eyebrow">Short-Horizon Hedge / Kaggle Style Audit</div><h1>Can macro features predict path pain well enough to hedge, without flipping fully net short?</h1><p>This page summarizes the dedicated hedge overlay audit. The hedge model only cuts net beta to 0.6 or 0.3, uses purged month-end training, and trades lagged daily states. The honest benchmark is the same-beta unhedged baseline, not just plain DCA.</p><div class="metrics-grid">{card_html}</div></section>
<section class="section"><div class="section-head"><div><h2>1. Leakage Audit</h2><div class="sub">The hedge stack follows the same guardrails as the long-side work and keeps slow macro releases lagged.</div></div></div><div class="split"><div class="stack"><div class="callout"><strong>Verdict</strong>I do not see a direct leakage bug in the hedge stack. The monthly hedge models train strictly on earlier rows, overlap is purged using forward window end dates, and the daily hedge states start after the month-end prediction date.</div>{safe_html_table(leakage_view)}</div><div class="img-panel"><img src="data:image/png;base64,{state_chart}" alt="Consensus hedge states"></div></div></section>
<section class="section"><div class="section-head"><div><h2>2. What Drives Hedge Risk</h2><div class="sub">The ranking combines 42-day min-return and 42-day path-CVaR Newey-West effects with hedge-model importances.</div></div></div><div class="two-col"><div class="img-panel"><img src="data:image/png;base64,{driver_chart}" alt="Risk driver ranking"></div><div class="callout"><strong>Takeaway</strong>The hedge model leans hardest on valuation and stress, not the black-box sentiment inputs. CAPE, credit spreads, financial conditions, labor/inflation stress, and trend explain much more of future path pain than the sentiment composites.</div>{safe_html_table(top_drivers)}</div></div></section>
<section class="section"><div class="section-head"><div><h2>3. Model Skill</h2><div class="sub">A hedge model is only useful if it catches path-pain episodes early enough to justify the lost upside from hedging.</div></div></div><div class="split"><div class="stack"><div class="callout"><strong>Key finding</strong>Random forest is materially better than logistic for this hedge task. The strongest signal is on the acute 21-day severe-pain target, where random forest gets the best AUC, while logistic over-hedges and gives away too much return.</div>{safe_html_table(validation_view)}{safe_html_table(stats_view)}</div><div class="img-panel"><img src="data:image/png;base64,{validation_chart}" alt="Validation chart"></div></div></section>
<section class="section"><div class="section-head"><div><h2>4. Portfolio Impact</h2><div class="sub">This is the real decision layer: does the hedge improve the same-beta baseline enough to justify the drag?</div></div></div><div class="two-col"><div class="img-panel"><img src="data:image/png;base64,{frontier_chart}" alt="Strategy frontier"></div><div class="stack"><div class="img-panel"><img src="data:image/png;base64,{equity_1x}" alt="1x hedge equity curves"></div><div class="img-panel"><img src="data:image/png;base64,{equity_2x}" alt="2x hedge equity curves"></div></div></div><div class="callout" style="margin-top:18px"><strong>Read this honestly</strong>The first-pass hedge overlay does not beat the same-beta baseline on final wealth at 1x, 2x, or 3x. What it does buy is major drawdown relief. The only place it clearly improves both final value and drawdown is the very fragile 5x baseline, where the unhedged path is almost unusably dangerous.</div>{safe_html_table(metrics_view)}</section>
<section class="section"><div class="section-head"><div><h2>5. Guidance</h2><div class="sub">What I would actually do with this today.</div></div></div><div class="callout"><strong>Recommendation</strong>Use the hedge model as a risk governor, not as a return engine. The current evidence supports a random-forest hedge overlay more than a logistic one, especially if the base portfolio is already levered. At 1x, plain DCA remains stronger on terminal wealth; at 2x+, the hedge starts to look useful as a drawdown control sleeve.</div><ul class="guidance"><li>Default research candidate: <b>random_forest hedge on a 2x base beta</b>. It cuts drawdown sharply, even though it still gives up terminal value.</li><li>Do not deploy the logistic hedge as the default. It hedges too often and destroys too much upside.</li><li>Keep the hedge overlay beta-capped. Reducing to 0.6 or 0.3 is much more robust than trying to flip fully net short off noisy macro signals.</li><li>The next improvement should target calibration, not complexity: better probability calibration, class-weight tuning, and perhaps a meta-label that only turns on hedges when both path-CVaR and drawdown risk are elevated.</li><li>Quarter-end and turn-of-quarter effects should stay in the feature set, but they are supporting context, not primary hedge triggers.</li></ul></section>
<div class="footer">Report generated by <code>{Path(__file__).name}</code> into <code>{output_html.relative_to(ROOT)}</code>.</div></div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    args = parser.parse_args()

    scorecard = pd.read_csv(args.audit_dir / "hedge_feature_scorecard.csv")
    validation = pd.read_csv(args.audit_dir / "hedge_model_validation_metrics.csv")
    metrics = pd.read_csv(args.audit_dir / "hedge_strategy_metrics.csv")
    expected_stats = pd.read_csv(args.audit_dir / "hedge_expected_path_stats.csv")
    current_summary = pd.read_csv(args.audit_dir / "hedge_current_summary.csv", parse_dates=["as_of"])
    target_summary = pd.read_csv(args.audit_dir / "hedge_target_summary.csv")
    monthly_predictions = pd.concat(
        [
            pd.read_csv(args.audit_dir / "hedge_logistic_monthly_predictions.csv", parse_dates=["date"]),
            pd.read_csv(args.audit_dir / "hedge_random_forest_monthly_predictions.csv", parse_dates=["date"]),
        ],
        ignore_index=True,
    )
    daily_signals = pd.read_csv(args.audit_dir / "hedge_daily_signals.csv", parse_dates=["date"])
    inventory = pd.read_csv(args.analysis_dir / "analysis_variable_inventory.csv")

    ranking = build_risk_driver_ranking(scorecard)
    leakage = build_leakage_audit(monthly_predictions, daily_signals, inventory)
    ranking.to_csv(args.audit_dir / "hedge_kaggle_risk_driver_ranking.csv", index=False)
    leakage.to_csv(args.audit_dir / "hedge_kaggle_leakage_audit.csv", index=False)

    html = build_html(
        output_html=args.output_html,
        audit_dir=args.audit_dir,
        analysis_dir=args.analysis_dir,
        ranking=ranking,
        leakage=leakage,
        current_summary=current_summary,
        target_summary=target_summary,
        validation=validation,
        expected_stats=expected_stats,
        metrics=metrics,
    )
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
