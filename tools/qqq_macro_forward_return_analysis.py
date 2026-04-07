"""Simple macro feature audit for future QQQ returns.

The goal is descriptive, not predictive precision:
- use only macro values available on or before each QQQ trading day
- sample month-end observations to reduce daily overlap noise
- test simple univariate relationships and fixed logical rules
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QQQ_PATH = ROOT / "cache" / "cache" / "cache" / "QQQ_daily.parquet"
DEFAULT_MACRO_PATH = ROOT / "cache" / "cache" / "macro_daily_1999.parquet"
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_macro_forward_return_analysis"
TRADING_DAYS_PER_YEAR = 252


FEATURE_DEFINITIONS: dict[str, str] = {
    "dxy_level": "DXY level",
    "dxy_3m_return": "DXY 3-month percent change",
    "dxy_12m_return": "DXY 12-month percent change",
    "us2y_level": "2Y Treasury yield level",
    "us10y_level": "10Y Treasury yield level",
    "us30y_level": "30Y Treasury yield level",
    "us2y_3m_change_pp": "2Y Treasury yield 3-month change in percentage points",
    "us2y_12m_change_pp": "2Y Treasury yield 12-month change in percentage points",
    "us10y_3m_change_pp": "10Y Treasury yield 3-month change in percentage points",
    "us10y_12m_change_pp": "10Y Treasury yield 12-month change in percentage points",
    "us30y_12m_change_pp": "30Y Treasury yield 12-month change in percentage points",
    "curve_10y2y_level": "10Y minus 2Y yield curve level",
    "curve_30y2y_level": "30Y minus 2Y yield curve level",
    "curve_10y2y_12m_change_pp": "10Y-2Y curve 12-month change in percentage points",
    "curve_30y2y_12m_change_pp": "30Y-2Y curve 12-month change in percentage points",
    "wti_level": "WTI spot price level",
    "wti_3m_return": "WTI 3-month percent change",
    "wti_12m_return": "WTI 12-month percent change",
}


def _load_qqq(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    date = pd.to_datetime(df["date"]).dt.tz_localize(None)
    close_col = "adj_c" if "adj_c" in df.columns else "c"
    close = pd.Series(pd.to_numeric(df[close_col], errors="coerce").to_numpy(), index=date, name="qqq_adj_close")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.dropna()


def _load_macro(path: Path, qqq_index: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.read_parquet(path)
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro.set_index("date").sort_index()
    macro = macro.rename(
        columns={
            "dxy_close": "dxy",
            "us_2y_yield": "us2y",
            "us_10y_yield": "us10y",
            "us_30y_yield": "us30y",
            "wti_usd_per_bbl": "wti",
        }
    )
    cols = ["dxy", "us2y", "us10y", "us30y", "wti"]
    macro = macro[cols].apply(pd.to_numeric, errors="coerce")
    # Forward-fill onto QQQ trading days so holidays use the most recent known macro print.
    aligned = macro.reindex(macro.index.union(qqq_index)).sort_index().ffill().reindex(qqq_index)
    return aligned


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1.0


def build_dataset(qqq_close: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=qqq_close.index)
    df["qqq_adj_close"] = qqq_close
    df["dxy_level"] = macro["dxy"]
    df["dxy_3m_return"] = _pct_change(macro["dxy"], 63)
    df["dxy_12m_return"] = _pct_change(macro["dxy"], 252)
    df["us2y_level"] = macro["us2y"]
    df["us10y_level"] = macro["us10y"]
    df["us30y_level"] = macro["us30y"]
    df["us2y_3m_change_pp"] = macro["us2y"].diff(63)
    df["us2y_12m_change_pp"] = macro["us2y"].diff(252)
    df["us10y_3m_change_pp"] = macro["us10y"].diff(63)
    df["us10y_12m_change_pp"] = macro["us10y"].diff(252)
    df["us30y_12m_change_pp"] = macro["us30y"].diff(252)
    df["curve_10y2y_level"] = macro["us10y"] - macro["us2y"]
    df["curve_30y2y_level"] = macro["us30y"] - macro["us2y"]
    df["curve_10y2y_12m_change_pp"] = df["curve_10y2y_level"].diff(252)
    df["curve_30y2y_12m_change_pp"] = df["curve_30y2y_level"].diff(252)
    df["wti_level"] = macro["wti"]
    df["wti_3m_return"] = _pct_change(macro["wti"], 63)
    df["wti_12m_return"] = _pct_change(macro["wti"], 252)

    for horizon in [252, 504]:
        forward_return = qqq_close.shift(-horizon) / qqq_close - 1.0
        df[f"qqq_forward_{horizon}d_return"] = forward_return
        df[f"qqq_forward_{horizon}d_cagr"] = (1.0 + forward_return).pow(TRADING_DAYS_PER_YEAR / horizon) - 1.0
    return df


def month_end_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.copy()
    sample["month"] = sample.index.to_period("M")
    sample = sample.groupby("month", sort=True).tail(1).drop(columns=["month"])
    return sample


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 24:
        return np.nan
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def _bucket_stats(df: pd.DataFrame, feature: str, outcome: str) -> dict[str, Any]:
    valid = df[[feature, outcome]].dropna()
    if len(valid) < 36:
        return {}
    valid = valid.copy()
    valid["bucket"] = pd.qcut(valid[feature], q=3, labels=["low", "mid", "high"], duplicates="drop")
    grouped = valid.groupby("bucket", observed=True)[outcome]
    stats = grouped.agg(["count", "mean", "median"]).reset_index()
    out: dict[str, Any] = {
        "low_count": np.nan,
        "mid_count": np.nan,
        "high_count": np.nan,
        "low_mean": np.nan,
        "mid_mean": np.nan,
        "high_mean": np.nan,
        "low_median": np.nan,
        "mid_median": np.nan,
        "high_median": np.nan,
    }
    for _, row in stats.iterrows():
        bucket = str(row["bucket"])
        out[f"{bucket}_count"] = int(row["count"])
        out[f"{bucket}_mean"] = float(row["mean"])
        out[f"{bucket}_median"] = float(row["median"])
    out["high_minus_low_mean"] = out["high_mean"] - out["low_mean"]
    out["best_bucket_mean"] = max(out["low_mean"], out["mid_mean"], out["high_mean"])
    out["worst_bucket_mean"] = min(out["low_mean"], out["mid_mean"], out["high_mean"])
    out["best_minus_worst_mean"] = out["best_bucket_mean"] - out["worst_bucket_mean"]
    return out


def univariate_feature_audit(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        for horizon in [252, 504]:
            outcome = f"qqq_forward_{horizon}d_cagr"
            bucket = _bucket_stats(sample, feature, outcome)
            if not bucket:
                continue
            row = {
                "feature": feature,
                "definition": FEATURE_DEFINITIONS.get(feature, feature),
                "horizon_days": horizon,
                "spearman_corr": _rank_corr(sample[feature], sample[outcome]),
            }
            row.update(bucket)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman_corr"] = out["spearman_corr"].abs()
        out["abs_high_minus_low_mean"] = out["high_minus_low_mean"].abs()
        out = out.sort_values(
            ["horizon_days", "abs_high_minus_low_mean", "abs_spearman_corr"],
            ascending=[True, False, False],
        )
    return out


def _rule_rows(sample: pd.DataFrame, rule_name: str, condition: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in [252, 504]:
        outcome = f"qqq_forward_{horizon}d_cagr"
        valid = sample[[outcome]].copy()
        valid["condition"] = condition.reindex(sample.index)
        valid = valid.dropna()
        valid["condition"] = valid["condition"].astype(bool)
        if valid["condition"].sum() < 12 or (~valid["condition"]).sum() < 12:
            continue
        on = valid.loc[valid["condition"], outcome]
        off = valid.loc[~valid["condition"], outcome]
        rows.append(
            {
                "rule": rule_name,
                "horizon_days": horizon,
                "on_count": int(len(on)),
                "off_count": int(len(off)),
                "on_mean_cagr": float(on.mean()),
                "off_mean_cagr": float(off.mean()),
                "on_minus_off_mean_cagr": float(on.mean() - off.mean()),
                "on_median_cagr": float(on.median()),
                "off_median_cagr": float(off.median()),
                "on_minus_off_median_cagr": float(on.median() - off.median()),
                "on_positive_rate": float((on > 0.0).mean()),
                "off_positive_rate": float((off > 0.0).mean()),
            }
        )
    return rows


def logical_rule_audit(sample: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "DXY falling YoY": sample["dxy_12m_return"] < 0.0,
        "DXY rising YoY": sample["dxy_12m_return"] > 0.0,
        "2Y yield falling YoY": sample["us2y_12m_change_pp"] < 0.0,
        "2Y yield rising YoY": sample["us2y_12m_change_pp"] > 0.0,
        "10Y yield falling YoY": sample["us10y_12m_change_pp"] < 0.0,
        "10Y yield rising YoY": sample["us10y_12m_change_pp"] > 0.0,
        "10Y-2Y curve steepening YoY": sample["curve_10y2y_12m_change_pp"] > 0.0,
        "10Y-2Y curve flattening YoY": sample["curve_10y2y_12m_change_pp"] < 0.0,
        "WTI falling YoY": sample["wti_12m_return"] < 0.0,
        "WTI rising YoY": sample["wti_12m_return"] > 0.0,
        "Rates easing and dollar falling": (sample["us10y_12m_change_pp"] < 0.0) & (sample["dxy_12m_return"] < 0.0),
        "Rates rising and dollar rising": (sample["us10y_12m_change_pp"] > 0.0) & (sample["dxy_12m_return"] > 0.0),
        "Rates easing, curve steepening": (sample["us2y_12m_change_pp"] < 0.0)
        & (sample["curve_10y2y_12m_change_pp"] > 0.0),
        "Rates rising, curve flattening": (sample["us2y_12m_change_pp"] > 0.0)
        & (sample["curve_10y2y_12m_change_pp"] < 0.0),
    }
    rows: list[dict[str, Any]] = []
    for name, condition in rules.items():
        rows.extend(_rule_rows(sample, name, condition))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["horizon_days", "on_minus_off_mean_cagr", "on_minus_off_median_cagr"],
            ascending=[True, False, False],
        )
    return out


def robustness_audit(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    windows = {
        "full": sample,
        "1999_2011": sample[sample.index < "2012-01-01"],
        "2012_plus": sample[sample.index >= "2012-01-01"],
        "december_only": sample[sample.index.month == 12],
    }
    rows: list[dict[str, Any]] = []
    for window_name, window_df in windows.items():
        for feature in features:
            for horizon in [252, 504]:
                outcome = f"qqq_forward_{horizon}d_cagr"
                valid = window_df[[feature, outcome]].dropna()
                if len(valid) < 20:
                    continue
                bucket = _bucket_stats(window_df, feature, outcome)
                rows.append(
                    {
                        "window": window_name,
                        "feature": feature,
                        "horizon_days": horizon,
                        "n": int(len(valid)),
                        "spearman_corr": _rank_corr(valid[feature], valid[outcome]),
                        "high_minus_low_mean": bucket.get("high_minus_low_mean", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def write_report(
    out_dir: Path,
    dataset: pd.DataFrame,
    sample: pd.DataFrame,
    feature_audit: pd.DataFrame,
    rule_audit: pd.DataFrame,
    robustness: pd.DataFrame,
    qqq_path: Path,
    macro_path: Path,
) -> None:
    lines = [
        "# QQQ Macro Forward Return Analysis",
        "",
        "This is a descriptive research audit, not a forecast or investment recommendation.",
        "",
        "## Data",
        "",
        f"- QQQ source: `{qqq_path}`",
        f"- Macro source: `{macro_path}`",
        f"- Daily aligned range: `{dataset.index[0].date()}` to `{dataset.index[-1].date()}`",
        f"- Month-end observations: `{len(sample)}`",
        "- Macro values are forward-filled to QQQ trading days, so signals use the latest value known on or before the observation date.",
        "- Outcomes are QQQ adjusted-close forward CAGRs over 252 and 504 trading days.",
        "- Features use simple levels, 3-month changes, 12-month changes, and yield-curve spreads.",
        "",
        "## Strongest Univariate Relationships",
        "",
    ]

    for horizon in [252, 504]:
        top = feature_audit[feature_audit["horizon_days"] == horizon].head(8)
        lines.extend(
            [
                f"### {horizon} Trading Days",
                "",
                "| Feature | Spearman | Low Third Avg | Mid Third Avg | High Third Avg | High-Low |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in top.iterrows():
            lines.append(
                "| {feature} | {corr:.2f} | {low} | {mid} | {high} | {spread} |".format(
                    feature=row["feature"],
                    corr=row["spearman_corr"],
                    low=_fmt_pct(row["low_mean"]),
                    mid=_fmt_pct(row["mid_mean"]),
                    high=_fmt_pct(row["high_mean"]),
                    spread=_fmt_pct(row["high_minus_low_mean"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Best Logical Rules",
            "",
            "| Rule | Horizon | On Avg CAGR | Off Avg CAGR | Difference | On Count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in rule_audit.groupby("horizon_days", group_keys=False).head(8).iterrows():
        lines.append(
            "| {rule} | {horizon} | {on} | {off} | {diff} | {count} |".format(
                rule=row["rule"],
                horizon=int(row["horizon_days"]),
                on=_fmt_pct(row["on_mean_cagr"]),
                off=_fmt_pct(row["off_mean_cagr"]),
                diff=_fmt_pct(row["on_minus_off_mean_cagr"]),
                count=int(row["on_count"]),
            )
        )

    lines.extend(
        [
            "",
            "## Robustness Notes",
            "",
            "| Feature | Horizon | Full Spearman | 1999-2011 Spearman | 2012+ Spearman | December-Only Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    focus_features = ["us10y_level", "us2y_level", "wti_12m_return", "dxy_12m_return", "curve_10y2y_level"]
    for feature in focus_features:
        for horizon in [252, 504]:
            rows = robustness[(robustness["feature"] == feature) & (robustness["horizon_days"] == horizon)]
            by_window = rows.set_index("window")["spearman_corr"].to_dict()
            lines.append(
                "| {feature} | {horizon} | {full} | {early} | {late} | {december} |".format(
                    feature=feature,
                    horizon=horizon,
                    full="" if "full" not in by_window else f"{by_window['full']:.2f}",
                    early="" if "1999_2011" not in by_window else f"{by_window['1999_2011']:.2f}",
                    late="" if "2012_plus" not in by_window else f"{by_window['2012_plus']:.2f}",
                    december="" if "december_only" not in by_window else f"{by_window['december_only']:.2f}",
                )
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `aligned_daily_dataset.csv`: daily aligned QQQ and macro features",
            "- `month_end_sample.csv`: month-end sample used for the audit",
            "- `feature_audit.csv`: univariate feature correlations and tercile buckets",
            "- `logical_rule_audit.csv`: fixed simple rule outcomes",
            "- `robustness_audit.csv`: era split and December-only robustness checks",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit macro relationships with 1Y/2Y forward QQQ returns.")
    parser.add_argument("--qqq-path", type=Path, default=DEFAULT_QQQ_PATH)
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    qqq_close = _load_qqq(args.qqq_path)
    macro = _load_macro(args.macro_path, qqq_close.index)
    dataset = build_dataset(qqq_close, macro)
    sample = month_end_sample(dataset).dropna(subset=["qqq_forward_252d_cagr", "qqq_forward_504d_cagr"])
    features = list(FEATURE_DEFINITIONS)
    feature_audit = univariate_feature_audit(sample, features)
    rule_audit = logical_rule_audit(sample)
    robustness = robustness_audit(sample, features)

    dataset.to_csv(out_dir / "aligned_daily_dataset.csv", index_label="date")
    sample.to_csv(out_dir / "month_end_sample.csv", index_label="date")
    feature_audit.to_csv(out_dir / "feature_audit.csv", index=False)
    rule_audit.to_csv(out_dir / "logical_rule_audit.csv", index=False)
    robustness.to_csv(out_dir / "robustness_audit.csv", index=False)
    write_report(out_dir, dataset, sample, feature_audit, rule_audit, robustness, args.qqq_path, args.macro_path)

    print(f"Saved macro forward-return report under: {out_dir}")
    print()
    print("Top 1Y relationships:")
    print(
        feature_audit[feature_audit["horizon_days"] == 252][
            ["feature", "spearman_corr", "low_mean", "mid_mean", "high_mean", "high_minus_low_mean"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    print("Top 2Y relationships:")
    print(
        feature_audit[feature_audit["horizon_days"] == 504][
            ["feature", "spearman_corr", "low_mean", "mid_mean", "high_mean", "high_minus_low_mean"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
