"""Create a central log of macro-regime backtest runs and model candidates.

The goal is to keep model-selection evidence in one stable place as experiments
evolve, instead of relying on scattered report folders.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "reports" / "macro_backtest_model_log"

DEFAULT_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "qqq_20260410_ensemble_refresh": {
        "target_asset": "QQQ",
        "dependent_variable": "QQQ adjusted close",
        "independent_overlay": "Macro inputs plus QQQ target trend features",
        "compare_dir": ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260410_ensemble_refresh",
        "robustness_path": ROOT
        / "reports"
        / "macro_regime_edge_audit_20260410_ensemble_refresh"
        / "ensemble_robustness_subwindows.csv",
        "summary_path": ROOT
        / "reports"
        / "macro_regime_edge_audit_20260410_ensemble_refresh"
        / "macro_regime_edge_report.md",
        "notes": "QQQ is the higher-beta growth expression; useful for traded expression, not necessarily the cleanest macro target.",
    },
    "qqq_20260410_x2_kaggle_rerun": {
        "target_asset": "QQQ",
        "dependent_variable": "QQQ adjusted close",
        "independent_overlay": "Macro inputs plus QQQ target trend features; fresh x2 rerun used for the Kaggle-style analysis pack",
        "compare_dir": ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260410_x2_kaggle_rerun",
        "robustness_path": ROOT
        / "reports"
        / "spy_gate_qqq_ensemble_backtest_20260410_x2_kaggle_rerun"
        / "combined_policy_metrics.csv",
        "summary_path": ROOT
        / "reports"
        / "qqq_x2_kaggle_analysis_20260410"
        / "x2_strategy_report.md",
        "notes": "Fresh x2 rerun used for the decision pack; preferred standalone expression is QQQ ensemble_blend_2x.",
    },
    "spy_gate_qqq_20260410_x2": {
        "target_asset": "QQQ_with_SPY_gate",
        "dependent_variable": "QQQ adjusted close, conditioned on SPY ensemble_blend gate",
        "independent_overlay": "SPY ensemble_blend broad-market gate plus QQQ ensemble_blend higher-beta expression",
        "compare_dir": ROOT / "reports" / "spy_gate_qqq_ensemble_backtest_20260410_x2_kaggle_rerun",
        "metrics_filename": "combined_policy_metrics.csv",
        "report_filename": "spy_gate_qqq_backtest_summary.md",
        "validation_filename": "",
        "robustness_path": ROOT
        / "reports"
        / "spy_gate_qqq_ensemble_backtest_20260410_x2_kaggle_rerun"
        / "combined_policy_metrics.csv",
        "summary_path": ROOT
        / "reports"
        / "qqq_x2_kaggle_analysis_20260410"
        / "x2_strategy_report.md",
        "notes": "Research candidate overlay: use SPY ensemble_blend as the gate and QQQ ensemble_blend_2x as the high-beta expression when the gate allows it.",
    },
    "sp500_with_qqq_20260410": {
        "target_asset": "SP500_SPY",
        "dependent_variable": "SPY adjusted close as S&P 500 proxy",
        "independent_overlay": "Macro inputs plus separate growth_qqq_* variables",
        "compare_dir": ROOT / "reports" / "sp500_macro_regime_with_qqq_20260410" / "compare",
        "robustness_path": ROOT
        / "reports"
        / "sp500_macro_regime_with_qqq_20260410"
        / "sp500_ensemble_robustness_subwindows.csv",
        "summary_path": ROOT
        / "reports"
        / "sp500_macro_regime_with_qqq_20260410"
        / "sp500_qqq_backtest_summary.md",
        "notes": "SP500/SPY is the macro-regime target; QQQ is included as a higher-beta growth confirmation variable.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=["all", *DEFAULT_EXPERIMENTS.keys()],
        default=["all"],
        help="Experiment(s) to include. Defaults to all configured experiments.",
    )
    return parser.parse_args()


def simplify_strategy_name(strategy: str) -> str:
    if strategy == "plain_dca":
        return "plain_dca_1x"
    text = strategy.removeprefix("walkforward_")
    text = text.replace("_riskon_", "_")
    for suffix in ["_prob_regime_dca", "_keep_long_riskoff_reserve_dca"]:
        text = text.removesuffix(suffix)
    return text


def model_family(strategy_name: str) -> str:
    if strategy_name.startswith("plain"):
        return "plain"
    if strategy_name.startswith("spy_gate"):
        return "spy_gate"
    if strategy_name.startswith("qqq_ensemble_blend"):
        return "ensemble_blend"
    if strategy_name.startswith("random_forest"):
        return "random_forest"
    if strategy_name.startswith("ensemble_blend"):
        return "ensemble_blend"
    if strategy_name.startswith("ensemble_majority"):
        return "ensemble_majority"
    if strategy_name.startswith("logistic"):
        return "logistic"
    if strategy_name.startswith("gmm"):
        return "gmm"
    return strategy_name.split("_")[0]


def strategy_leverage(strategy_name: str) -> float:
    if strategy_name.startswith("plain"):
        return 1.0
    match = re.search(r"_(\d+)x$", strategy_name)
    return float(match.group(1)) if match else np.nan


def clean_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    return float(number)


def validation_map(validation: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if validation.empty:
        return out
    for _, row in validation.iterrows():
        target = str(row.get("target", ""))
        model = str(row.get("model", ""))
        if not model or model == "nan":
            continue
        out[(model, target)] = {
            "auc": clean_number(row.get("auc")),
            "average_precision": clean_number(row.get("average_precision")),
            "mae": clean_number(row.get("mae")),
            "r2": clean_number(row.get("r2")),
            "test_n": clean_number(row.get("test_n")),
        }
    return out


def robustness_win_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    strategy_col = "strategy" if "strategy" in data.columns else None
    if data.empty or "window" not in data or strategy_col is None:
        return {}
    counts: dict[str, int] = {}
    mask = ~data["window"].astype(str).str.contains("full_common", case=False, na=False)
    for window, group in data[mask].groupby("window"):
        _ = window
        best = group.sort_values("final_value", ascending=False).iloc[0]
        strategy = simplify_strategy_name(str(best[strategy_col]))
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


def caution_flags(row: pd.Series) -> str:
    flags: list[str] = []
    max_dd = abs(clean_number(row.get("max_drawdown")))
    risk_on_months = clean_number(row.get("risk_on_months"))
    xirr = clean_number(row.get("xirr"))
    if np.isfinite(max_dd) and max_dd >= 0.60:
        flags.append("severe_drawdown")
    if np.isfinite(risk_on_months) and risk_on_months < 12 and row.get("model_family") != "plain":
        flags.append("few_risk_on_months")
    if np.isfinite(xirr) and xirr <= 0.0:
        flags.append("non_positive_xirr")
    return ";".join(flags)


def build_experiment_rows(experiment_id: str, config: dict[str, Any], generated_at: str) -> list[dict[str, Any]]:
    compare_dir = Path(config["compare_dir"])
    metrics_path = compare_dir / config.get("metrics_filename", "walkforward_model_compare_leverage_metrics.csv")
    validation_filename = config.get("validation_filename", "walkforward_model_validation_metrics.csv")
    validation_path = compare_dir / validation_filename if validation_filename else None
    report_path = compare_dir / config.get("report_filename", "walkforward_model_compare_report.md")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing compare metrics for {experiment_id}: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    validation = pd.read_csv(validation_path) if validation_path is not None and validation_path.exists() else pd.DataFrame()
    validation_by_model = validation_map(validation)
    robustness_wins = robustness_win_counts(Path(config["robustness_path"]))

    full = metrics[metrics["window"].eq("full_common_window")].copy()
    if full.empty:
        raise ValueError(f"No full_common_window rows found in {metrics_path}")

    plain = full[full["strategy"].isin(["plain_dca", "plain_dca_1x"])]
    if plain.empty:
        plain = full[full["strategy"].astype(str).str.contains("plain_dca", na=False)]
    plain_xirr = clean_number(plain.iloc[0]["xirr"]) if not plain.empty else np.nan
    plain_final = clean_number(plain.iloc[0]["final_value"]) if not plain.empty else np.nan
    full["strategy_short"] = full["strategy"].map(lambda value: simplify_strategy_name(str(value)))
    full["model_family"] = full["strategy_short"].map(model_family)
    full["strategy_leverage"] = full["strategy_short"].map(strategy_leverage)
    full["xirr_to_abs_dd"] = full.apply(
        lambda row: clean_number(row["xirr"]) / abs(clean_number(row["max_drawdown"]))
        if abs(clean_number(row["max_drawdown"])) > 0.0
        else np.nan,
        axis=1,
    )
    full["rank_final_value"] = full["final_value"].rank(method="min", ascending=False).astype(int)
    full["rank_xirr_to_abs_dd"] = full["xirr_to_abs_dd"].rank(method="min", ascending=False).astype(int)

    rows: list[dict[str, Any]] = []
    for _, row in full.sort_values("rank_final_value").iterrows():
        family = str(row["model_family"])
        risk_off = validation_by_model.get((family, "risk_off_target"), {})
        jump_in = validation_by_model.get((family, "jump_in_target"), {})
        forward = validation_by_model.get((family, "qqq_fwd_63d_return"), {})
        strategy_short = str(row["strategy_short"])
        log_row = {
            "generated_at_utc": generated_at,
            "experiment_id": experiment_id,
            "target_asset": config["target_asset"],
            "dependent_variable": config["dependent_variable"],
            "independent_overlay": config["independent_overlay"],
            "window": "full_common_window",
            "strategy": strategy_short,
            "source_strategy": row["strategy"],
            "model_family": family,
            "strategy_leverage": clean_number(row["strategy_leverage"]),
            "start_date": row.get("start_date"),
            "end_date": row.get("end_date"),
            "final_value": clean_number(row.get("final_value")),
            "xirr": clean_number(row.get("xirr")),
            "xirr_excess_vs_plain": clean_number(row.get("xirr")) - plain_xirr,
            "time_weighted_cagr": clean_number(row.get("time_weighted_cagr")),
            "max_drawdown": clean_number(row.get("max_drawdown")),
            "avg_target_leverage": clean_number(row.get("avg_target_leverage")),
            "risk_on_months": clean_number(row.get("risk_on_months")),
            "neutral_months": clean_number(row.get("neutral_months")),
            "risk_off_months": clean_number(row.get("risk_off_months")),
            "final_delta_vs_plain": clean_number(row.get("final_value")) - plain_final,
            "rank_final_value": int(row["rank_final_value"]),
            "rank_xirr_to_abs_dd": int(row["rank_xirr_to_abs_dd"]),
            "xirr_to_abs_dd": clean_number(row.get("xirr_to_abs_dd")),
            "subwindow_win_count_2x": robustness_wins.get(strategy_short, 0),
            "risk_off_auc": risk_off.get("auc", np.nan),
            "risk_off_average_precision": risk_off.get("average_precision", np.nan),
            "jump_in_auc": jump_in.get("auc", np.nan),
            "jump_in_average_precision": jump_in.get("average_precision", np.nan),
            "forward_63d_mae": forward.get("mae", np.nan),
            "forward_63d_r2": forward.get("r2", np.nan),
            "validation_test_n": max(
                [value for value in [risk_off.get("test_n"), jump_in.get("test_n"), forward.get("test_n")] if value is not None],
                default=np.nan,
            ),
            "caution_flags": "",
            "compare_dir": str(compare_dir.relative_to(ROOT)),
            "compare_report": str(report_path.relative_to(ROOT)) if report_path.exists() else "",
            "summary_path": str(Path(config["summary_path"]).relative_to(ROOT)) if Path(config["summary_path"]).exists() else "",
            "notes": config["notes"],
        }
        log_row["caution_flags"] = caution_flags(pd.Series(log_row))
        rows.append(log_row)
    return rows


def format_pct(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value * 100:.{decimals}f}%"


def format_money(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"${value:,.0f}"


def write_summary(log_dir: Path, registry: pd.DataFrame, generated_at: str) -> None:
    lines = [
        "# Macro Backtest Model Log",
        "",
        f"- Generated at UTC: `{generated_at}`",
        "- Source of truth CSV: `backtest_registry.csv`",
        "- Validation CSV: `validation_registry.csv`",
        "- This log is intended to be rerun after each macro-regime backtest so model selection is auditable.",
        "",
        "## Current Leaders",
        "",
    ]
    for experiment_id, group in registry.groupby("experiment_id", sort=False):
        sorted_final = group.sort_values("final_value", ascending=False)
        non_plain = group[group["model_family"].ne("plain")].copy()
        two_x = non_plain[non_plain["strategy_leverage"].eq(2.0)].sort_values("final_value", ascending=False)
        risk_adjusted = non_plain.sort_values("xirr_to_abs_dd", ascending=False)
        best_final = sorted_final.iloc[0]
        best_2x = two_x.iloc[0] if not two_x.empty else None
        best_risk = risk_adjusted.iloc[0] if not risk_adjusted.empty else None
        lines.extend(
            [
                f"### {experiment_id}",
                "",
                f"- Target: `{best_final['target_asset']}`",
                f"- Best final value: `{best_final['strategy']}` at {format_money(best_final['final_value'])}, XIRR {format_pct(best_final['xirr'])}, max DD {format_pct(best_final['max_drawdown'])}.",
            ]
        )
        if best_2x is not None:
            lines.append(
                f"- Best 2x candidate: `{best_2x['strategy']}` at {format_money(best_2x['final_value'])}, "
                f"XIRR {format_pct(best_2x['xirr'])}, max DD {format_pct(best_2x['max_drawdown'])}."
            )
        if best_risk is not None:
            lines.append(
                f"- Best XIRR/drawdown candidate: `{best_risk['strategy']}` with score {best_risk['xirr_to_abs_dd']:.3f}."
            )
        cautions = sorted({flag for text in group["caution_flags"].dropna() for flag in str(text).split(";") if flag})
        lines.append(f"- Caution flags present: `{', '.join(cautions) if cautions else 'none'}`")
        lines.append("")

    lines.extend(
        [
            "## Selection Rule Of Thumb",
            "",
            "- Prefer candidates that beat plain DCA on XIRR and final value without severe drawdown flags.",
            "- Prefer 2x over 3x when the XIRR improvement is small relative to drawdown increase.",
            "- Treat `few_risk_on_months` as a fragility warning, not a rejection by itself.",
            "- Do not promote a model solely because it wins one full-window run; check `subwindow_win_count_2x` and validation metrics.",
            "",
            "## Refresh Command",
            "",
            "```powershell",
            "python tools/update_macro_backtest_log.py",
            "```",
            "",
        ]
    )
    (log_dir / "model_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_leaders_csv(log_dir: Path, registry: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for experiment_id, group in registry.groupby("experiment_id", sort=False):
        non_plain = group[group["model_family"].ne("plain")].copy()
        best_final = group.sort_values("final_value", ascending=False).iloc[0]
        two_x = non_plain[non_plain["strategy_leverage"].eq(2.0)].sort_values("final_value", ascending=False)
        risk_adjusted = non_plain.sort_values("xirr_to_abs_dd", ascending=False)
        best_2x = two_x.iloc[0] if not two_x.empty else None
        best_risk = risk_adjusted.iloc[0] if not risk_adjusted.empty else None
        practical = best_risk if best_risk is not None else best_2x
        if practical is None:
            practical = best_final
        rows.append(
            {
                "experiment_id": experiment_id,
                "target_asset": best_final["target_asset"],
                "best_final_value_strategy": best_final["strategy"],
                "best_final_value": best_final["final_value"],
                "best_final_xirr": best_final["xirr"],
                "best_final_max_drawdown": best_final["max_drawdown"],
                "best_2x_strategy": None if best_2x is None else best_2x["strategy"],
                "best_2x_xirr": np.nan if best_2x is None else best_2x["xirr"],
                "best_2x_max_drawdown": np.nan if best_2x is None else best_2x["max_drawdown"],
                "best_xirr_to_abs_dd_strategy": None if best_risk is None else best_risk["strategy"],
                "best_xirr_to_abs_dd": np.nan if best_risk is None else best_risk["xirr_to_abs_dd"],
                "practical_candidate_strategy": practical["strategy"],
                "practical_candidate_xirr": practical["xirr"],
                "practical_candidate_max_drawdown": practical["max_drawdown"],
                "practical_candidate_caution_flags": practical["caution_flags"],
                "practical_candidate_note": "research_candidate_not_production_ready",
            }
        )
    pd.DataFrame(rows).to_csv(log_dir / "model_selection_leaders.csv", index=False)


def write_readme(log_dir: Path) -> None:
    lines = [
        "# Macro Backtest Logs",
        "",
        "This folder records model-selection evidence for macro-regime experiments.",
        "",
        "Files:",
        "",
        "- `backtest_registry.csv`: full-window strategy metrics across configured experiments.",
        "- `validation_registry.csv`: model validation metrics copied from each compare run.",
        "- `model_selection_leaders.csv`: compact best-final, best-2x, and practical-candidate rows.",
        "- `model_selection_summary.md`: compact leader/caution summary.",
        "- `latest_model_selection.json`: best final-value, best 2x, and best XIRR/drawdown choices by experiment.",
        "",
        "Refresh after rerunning backtests:",
        "",
        "```powershell",
        "python tools/update_macro_backtest_log.py",
        "```",
    ]
    (log_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_latest_json(log_dir: Path, registry: pd.DataFrame, generated_at: str) -> None:
    output: dict[str, Any] = {"generated_at_utc": generated_at, "experiments": {}}
    for experiment_id, group in registry.groupby("experiment_id", sort=False):
        non_plain = group[group["model_family"].ne("plain")]
        best_final = group.sort_values("final_value", ascending=False).iloc[0]
        two_x = non_plain[non_plain["strategy_leverage"].eq(2.0)].sort_values("final_value", ascending=False)
        risk_adjusted = non_plain.sort_values("xirr_to_abs_dd", ascending=False)
        output["experiments"][experiment_id] = {
            "target_asset": best_final["target_asset"],
            "best_final_value_strategy": best_final["strategy"],
            "best_2x_strategy": None if two_x.empty else two_x.iloc[0]["strategy"],
            "best_xirr_to_drawdown_strategy": None if risk_adjusted.empty else risk_adjusted.iloc[0]["strategy"],
            "best_final_value": None if pd.isna(best_final["final_value"]) else float(best_final["final_value"]),
            "best_final_xirr": None if pd.isna(best_final["xirr"]) else float(best_final["xirr"]),
            "best_final_max_drawdown": None if pd.isna(best_final["max_drawdown"]) else float(best_final["max_drawdown"]),
        }
    (log_dir / "latest_model_selection.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def selected_experiments(values: list[str]) -> list[str]:
    if "all" in values:
        return list(DEFAULT_EXPERIMENTS)
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []
    for experiment_id in selected_experiments(args.experiment):
        config = DEFAULT_EXPERIMENTS[experiment_id]
        rows.extend(build_experiment_rows(experiment_id, config, generated_at))
        validation_filename = config.get("validation_filename", "walkforward_model_validation_metrics.csv")
        validation_path = Path(config["compare_dir"]) / validation_filename if validation_filename else None
        if validation_path is not None and validation_path.exists():
            validation = pd.read_csv(validation_path)
            validation.insert(0, "experiment_id", experiment_id)
            validation.insert(1, "target_asset", config["target_asset"])
            validation_rows.append(validation)

    registry = pd.DataFrame(rows).sort_values(["experiment_id", "rank_final_value"])
    registry.to_csv(args.log_dir / "backtest_registry.csv", index=False)
    if validation_rows:
        pd.concat(validation_rows, ignore_index=True).to_csv(args.log_dir / "validation_registry.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.log_dir / "validation_registry.csv", index=False)
    write_leaders_csv(args.log_dir, registry)
    write_summary(args.log_dir, registry, generated_at)
    write_latest_json(args.log_dir, registry, generated_at)
    write_readme(args.log_dir)
    print(f"Wrote macro backtest model log to {args.log_dir}")


if __name__ == "__main__":
    main()
