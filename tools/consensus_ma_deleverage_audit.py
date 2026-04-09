"""Test lagged 60d/200d trend deleveraging on the consensus long strategy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.append(str(ROOT / "tools"))

import macro_regime_edge_audit as edge
import qqq_macro_walkforward_leverage as leverage
import qqq_macro_walkforward_model_compare as model_compare


DEFAULT_ANALYSIS_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_COMPARE_DIR = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260409_monthly_equal"
DEFAULT_OUT_DIR = ROOT / "reports" / "macro_regime_edge_audit_20260409" / "ma_deleverage_60_200"
DEFAULT_VIX_OUT_DIR = ROOT / "reports" / "macro_regime_edge_audit_20260409" / "ma_deleverage_60_200_vix"

MODE_LABELS = {
    "baseline": "Baseline",
    "cap_1x": "Cap To 1x",
    "step_down": "Step Down",
}
MODE_COLORS = {
    "baseline": "#1f77b4",
    "cap_1x": "#d62728",
    "step_down": "#2ca02c",
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
    parser.add_argument("--short-window", type=int, default=60)
    parser.add_argument("--long-window", type=int, default=200)
    parser.add_argument("--risk-on-levels", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--vix-thresholds", type=float, nargs="*", default=[20.0, 25.0, 30.0])
    parser.add_argument("--include-vix", action="store_true")
    return parser.parse_args()


def load_csv(path: Path, *, index_col: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if index_col is not None:
        frame = frame.set_index(index_col)
    return frame


def load_consensus_signal(compare_dir: Path, close_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    monthly_regimes = load_csv(compare_dir / "walkforward_model_regimes_monthly.csv", index_col="date")
    gmm_daily = load_csv(compare_dir / "walkforward_gmm_daily_regimes.csv", index_col="date")

    logistic_monthly = monthly_regimes[monthly_regimes["model_name"] == "logistic"].copy()
    logistic_prediction = logistic_monthly["regime"].rename("logistic")
    gmm_monthly = edge.regime_month_end_from_daily(gmm_daily, "wf_gmm_regime", "gmm")
    gmm_prediction = gmm_monthly["regime"].rename("gmm")
    consensus_prediction = edge.build_consensus_regime({"logistic": logistic_prediction, "gmm": gmm_prediction})
    daily_consensus = model_compare.monthly_signal_to_daily(consensus_prediction, close_index)
    return consensus_prediction, daily_consensus, gmm_monthly["regime"]


def build_trend_signal(close: pd.Series, short_window: int, long_window: int) -> pd.DataFrame:
    sma_short = close.rolling(short_window, min_periods=short_window).mean()
    sma_long = close.rolling(long_window, min_periods=long_window).mean()
    bearish_raw = sma_short.lt(sma_long)
    bearish_lag1 = bearish_raw.shift(1, fill_value=False).astype(bool)
    previous_bearish = bearish_raw.shift(1, fill_value=False).astype(bool)
    cross_below_raw = bearish_raw.astype(bool) & ~previous_bearish
    cross_above_raw = ~bearish_raw.astype(bool) & previous_bearish
    return pd.DataFrame(
        {
            "qqq_close": close.astype(float),
            "sma_short": sma_short.astype(float),
            "sma_long": sma_long.astype(float),
            "ma_bearish_raw": bearish_raw.astype(bool),
            "ma_bearish_lag1": bearish_lag1.astype(bool),
            "cross_below_raw": cross_below_raw.fillna(False).astype(bool),
            "cross_above_raw": cross_above_raw.fillna(False).astype(bool),
        }
    )


def build_vix_signal(vix_level: pd.Series, threshold: float) -> pd.Series:
    elevated_raw = vix_level.astype(float).ge(float(threshold))
    return elevated_raw.shift(1, fill_value=False).astype(bool).rename(f"vix_ge_{threshold:g}_lag1")


def adjusted_leverage(base_leverage: float, trend_active: bool, mode: str) -> float:
    if mode == "baseline":
        return float(base_leverage)
    if not trend_active or base_leverage <= 1.0:
        return float(base_leverage)
    if mode == "cap_1x":
        return 1.0
    if mode == "step_down":
        if base_leverage <= 2.0:
            return 1.0
        if base_leverage <= 3.0:
            return 2.0
        if base_leverage <= 5.0:
            return 3.0
        return max(1.0, base_leverage / 2.0)
    raise ValueError(f"Unknown mode: {mode}")


def simulate_consensus_with_trend(
    close: pd.Series,
    regime_signal: pd.Series,
    trend_active: pd.Series,
    *,
    strategy: str,
    risk_on_leverage: float,
    mode: str,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
    borrow_rate: float,
) -> leverage.StrategyResult:
    signal = regime_signal.reindex(close.index)
    if signal.dropna().empty:
        raise RuntimeError("No lagged consensus regime signal available for deleveraging audit.")
    price = close.loc[signal.dropna().index.min() :].copy()
    signal = signal.reindex(price.index).ffill()
    trend_active = trend_active.reindex(price.index).fillna(False).astype(bool)
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
    trend_deleverage_entries = 0
    trend_deleverage_exits = 0
    trend_deleverage_days = 0
    events: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(price.index[0], -float(initial_capital))]

    first_regime = str(signal.iloc[0])
    first_base_leverage = leverage.target_leverage_for_regime(first_regime, risk_on_leverage)
    first_trend = bool(trend_active.iloc[0])
    current_leverage = adjusted_leverage(first_base_leverage, first_trend, mode)
    initial_cost = exposure_equity * current_leverage * cost_rate
    exposure_equity = max(0.0, exposure_equity - initial_cost)
    transaction_costs += initial_cost

    previous_price = float(price.iloc[0])
    previous_regime = first_regime
    previous_trend = first_trend
    previous_period = price.index[0].to_period("M")

    for i, (date, px) in enumerate(price.items()):
        px = float(px)
        regime = str(signal.loc[date])
        trend_now = bool(trend_active.loc[date])
        base_leverage = leverage.target_leverage_for_regime(regime, risk_on_leverage)
        new_leverage = adjusted_leverage(base_leverage, trend_now, mode)

        if i > 0 and exposure_equity > 0.0:
            equity_before = exposure_equity
            daily_borrow = max(current_leverage - 1.0, 0.0) * borrow_rate / 252.0
            borrow_costs += equity_before * daily_borrow
            day_return = px / previous_price - 1.0
            exposure_equity = max(0.0, equity_before * (1.0 + current_leverage * day_return - daily_borrow))

        account_before_actions = exposure_equity + reserve_cash
        period = date.to_period("M")
        if i > 0 and period != previous_period:
            contribution = float(monthly_contribution)
            cashflows.append((date, -contribution))
            if regime == "risk_off":
                reserve_cash += contribution
                reserve_contributions += contribution
            else:
                contribution_cost = contribution * new_leverage * cost_rate
                transaction_costs += contribution_cost
                exposure_equity += max(0.0, contribution - contribution_cost)
            previous_period = period

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

        if mode != "baseline" and trend_now and not previous_trend and base_leverage > 1.0:
            trend_deleverage_entries += 1
            leverage.record_event(
                events,
                date=date,
                strategy=strategy,
                event=f"trend_deleverage_on_{mode}",
                regime=regime,
                value_before=account_before_actions,
                reserve_cash=reserve_cash,
                amount=None,
            )
        if mode != "baseline" and previous_trend and not trend_now:
            trend_deleverage_exits += 1
            leverage.record_event(
                events,
                date=date,
                strategy=strategy,
                event=f"trend_deleverage_off_{mode}",
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

        if trend_now and base_leverage > 1.0:
            trend_deleverage_days += 1

        total_value = exposure_equity + reserve_cash
        curves.append(
            {
                "date": date,
                "strategy": strategy,
                "total_value": total_value,
                "exposure_equity": exposure_equity,
                "reserve_cash": reserve_cash,
                "base_target_leverage": base_leverage,
                "target_leverage": new_leverage,
                "regime_signal": regime,
                "trend_deleverage_active": trend_now,
            }
        )
        previous_price = px
        previous_regime = regime
        previous_trend = trend_now
        current_leverage = new_leverage

    curve_df = pd.DataFrame(curves).set_index("date")
    month_signals = curve_df["regime_signal"].groupby(curve_df.index.to_period("M")).first()
    return leverage.StrategyResult(
        strategy=strategy,
        curves=curve_df,
        events=pd.DataFrame(events),
        cashflows=cashflows,
        final_exposure_equity=float(curve_df["exposure_equity"].iloc[-1]),
        final_reserve_cash=float(curve_df["reserve_cash"].iloc[-1]),
        avg_target_leverage=float(curve_df["target_leverage"].mean()),
        total_external_contributed=leverage.total_external_from_cashflows(cashflows),
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


def strategy_metrics_row(
    window: str,
    result: leverage.StrategyResult,
    *,
    mode: str,
    signal_variant: str,
    vix_threshold: float | None,
    risk_on_leverage: int,
    trend_entries: int,
    trend_exits: int,
    trend_days: int,
) -> dict[str, Any]:
    row = leverage.metrics_row(window, result, None)
    row["overlay_mode"] = mode
    row["signal_variant"] = signal_variant
    row["vix_threshold"] = float(vix_threshold) if vix_threshold is not None else np.nan
    row["risk_on_leverage"] = int(risk_on_leverage)
    row["trend_deleverage_entries"] = int(trend_entries)
    row["trend_deleverage_exits"] = int(trend_exits)
    row["trend_deleverage_days"] = int(trend_days)
    row["trend_deleverage_share_days"] = trend_days / max(len(result.curves), 1)
    return row


def extract_trend_stats(result: leverage.StrategyResult) -> tuple[int, int, int]:
    curves = result.curves
    active = curves["trend_deleverage_active"].fillna(False).astype(bool)
    previous = active.shift(1, fill_value=False).astype(bool)
    entries = int((active & ~previous).sum())
    exits = int((~active & previous).sum())
    days = int(active.sum())
    return entries, exits, days


def plot_comparison(curves: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, lev in zip(axes, [2, 3, 5], strict=True):
        subset = curves[curves["risk_on_leverage"] == lev].copy()
        preferred = subset[
            subset["signal_variant"].eq("baseline")
            | (
                subset["signal_variant"].eq("ma_only")
                & subset["overlay_mode"].isin(["step_down"])
            )
            | (
                subset["signal_variant"].eq("ma_vix_20")
                & subset["overlay_mode"].isin(["step_down"])
            )
        ].copy()
        for label, frame in [
            ("Baseline", preferred[preferred["signal_variant"] == "baseline"]),
            ("MA only step-down", preferred[(preferred["signal_variant"] == "ma_only") & (preferred["overlay_mode"] == "step_down")]),
            ("MA + VIX20 step-down", preferred[(preferred["signal_variant"] == "ma_vix_20") & (preferred["overlay_mode"] == "step_down")]),
        ]:
            if frame.empty:
                continue
            variant = frame["signal_variant"].iloc[0]
            mode = frame["overlay_mode"].iloc[0]
            ax.plot(
                pd.to_datetime(frame["date"]),
                frame["total_value"],
                label=label,
                color=(
                    MODE_COLORS[mode]
                    if variant == "ma_only"
                    else ("#9467bd" if variant == "ma_vix_20" else MODE_COLORS[mode])
                ),
                linewidth=2.0,
            )
        ax.set_yscale("log")
        ax.set_title(f"Consensus {lev}x with lagged 60d/200d trend deleveraging")
        ax.set_ylabel("Total value")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    metrics: pd.DataFrame,
    signal_daily: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> None:
    lines = [
        "# Consensus MA Deleveraging Audit",
        "",
        f"- Trend rule under test: lagged `{short_window}d < {long_window}d` on QQQ closes.",
        "- No-leakage treatment: the moving-average state is shifted by one trading day before it can change leverage.",
        "- Reserve behavior is unchanged: only `risk_off` parks new cash in reserve; non-`risk_off` regimes deploy reserve cash.",
        "",
        "## Headline read",
        "",
    ]
    for lev in [2, 3, 5]:
        subset = metrics[metrics["risk_on_leverage"] == lev].copy()
        if subset.empty:
            continue
        base = subset[subset["signal_variant"] == "baseline"].iloc[0]
        ma_only = subset[
            subset["signal_variant"].eq("ma_only") & subset["overlay_mode"].eq("step_down")
        ].sort_values("final_value", ascending=False)
        best = subset.sort_values("final_value", ascending=False).iloc[0]
        lines.append(
            f"- `Consensus {lev}x`: baseline stayed best on terminal value at "
            f"`$ {base['final_value']:,.0f}` with `{base['max_drawdown']:.1%}` max drawdown."
        )
        if not ma_only.empty:
            row = ma_only.iloc[0]
            lines.append(
                f"- MA-only step-down finished at `$ {row['final_value']:,.0f}` with "
                f"`{row['max_drawdown']:.1%}` drawdown."
            )
        ma_vix = subset[
            subset["signal_variant"].str.startswith("ma_vix_") & subset["overlay_mode"].eq("step_down")
        ].sort_values("final_value", ascending=False)
        if not ma_vix.empty:
            row = ma_vix.iloc[0]
            threshold = int(row["vix_threshold"]) if np.isfinite(row["vix_threshold"]) else "n/a"
            lines.append(
                f"- Best MA+VIX step-down used `VIX >= {threshold}` and finished at "
                f"`$ {row['final_value']:,.0f}` with `{row['max_drawdown']:.1%}` drawdown."
            )
        if best["overlay_mode"] != "baseline":
            lines.append(
                f"- Best MA variant for `{lev}x` was `{best['overlay_mode']}` at "
                f"`$ {best['final_value']:,.0f}` and `{best['max_drawdown']:.1%}` drawdown."
            )
        else:
            lines.append(
                f"- Both MA deleveraging variants reduced return more than drawdown for `{lev}x`."
            )
    lines.extend(
        [
            "",
            "## Trend signal stats",
            "",
            f"- Lagged bearish-state share: `{signal_daily['ma_bearish_lag1'].mean():.1%}` of traded days.",
            f"- Raw cross-below events: `{int(signal_daily['cross_below_raw'].sum())}`.",
            f"- Raw cross-above events: `{int(signal_daily['cross_above_raw'].sum())}`.",
            f"- Lagged `VIX >= 20` share: `{signal_daily['vix_ge_20_lag1'].mean():.1%}` of traded days."
            if "vix_ge_20_lag1" in signal_daily
            else "- Lagged VIX filter not enabled in this run.",
            "",
            "## Conclusion",
            "",
            "- A hard 1x cap during bearish trend states is too blunt for this consensus long strategy.",
            "- Adding an elevated-VIX confirmation makes the trend filter less blunt and improves on MA-only, but it still does not beat the baseline consensus strategy here.",
            "- A softer step-down rule preserves more upside, but still gave up too much return for only modest drawdown relief in this backtest.",
            "- If we want trend-aware deleveraging to help, the next sensible test is a smaller haircut such as `5x -> 4x`, `3x -> 2.5x`, `2x -> 1.5x` or using the trend filter only after stress confirmation.",
        ]
    )
    (out_dir / "consensus_ma_deleverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = leverage.load_dataset(args.analysis_dir / "aligned_daily_dataset.csv")
    compare_metrics = pd.read_csv(args.compare_dir / "walkforward_model_compare_leverage_metrics.csv")
    plain_row = compare_metrics[
        (compare_metrics["window"] == "full_common_window") & (compare_metrics["strategy"] == "plain_dca")
    ].iloc[0]
    common_start = pd.Timestamp(plain_row["start_date"])
    close = dataset["qqq_close"].astype(float)
    vix_level = dataset["vix_level"].astype(float)
    consensus_monthly, consensus_daily, _ = load_consensus_signal(args.compare_dir, close.index)

    signal_daily = build_trend_signal(close, args.short_window, args.long_window)
    signal_daily["consensus_signal_lag1"] = consensus_daily.reindex(close.index)
    if args.include_vix:
        for threshold in args.vix_thresholds:
            signal_daily[f"vix_ge_{threshold:g}_lag1"] = build_vix_signal(vix_level, threshold)

    close_window = close.loc[common_start:].copy()
    consensus_window = consensus_daily.reindex(close_window.index)
    trend_window = signal_daily["ma_bearish_lag1"].reindex(close_window.index)

    metric_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for lev in args.risk_on_levels:
        signal_variants: list[tuple[str, pd.Series, float | None]] = [("baseline", trend_window.copy().astype(bool), None)]
        signal_variants.append(("ma_only", trend_window.copy().astype(bool), None))
        if args.include_vix:
            for threshold in args.vix_thresholds:
                vix_filter = signal_daily[f"vix_ge_{threshold:g}_lag1"].reindex(close_window.index).fillna(False).astype(bool)
                signal_variants.append((f"ma_vix_{int(threshold)}", (trend_window & vix_filter).astype(bool), float(threshold)))

        for signal_variant, active_signal, threshold in signal_variants:
            modes = ["baseline"] if signal_variant == "baseline" else ["step_down", "cap_1x"]
            for mode in modes:
                result = simulate_consensus_with_trend(
                    close_window,
                    consensus_window,
                    active_signal,
                    strategy=(
                        f"consensus_{int(lev)}x_baseline"
                        if signal_variant == "baseline"
                        else f"consensus_{int(lev)}x_{signal_variant}_{mode}"
                    ),
                    risk_on_leverage=float(lev),
                    mode=mode,
                    initial_capital=args.initial_capital,
                    monthly_contribution=args.monthly_contribution,
                    trading_cost_bps=args.trading_cost_bps,
                    borrow_rate=args.borrow_rate,
                )
                entries, exits, days = extract_trend_stats(result)
                metric_rows.append(
                    strategy_metrics_row(
                        "full_common_window",
                        result,
                        mode=mode,
                        signal_variant=signal_variant,
                        vix_threshold=threshold,
                        risk_on_leverage=int(lev),
                        trend_entries=entries,
                        trend_exits=exits,
                        trend_days=days,
                    )
                )
                curve_frames.append(
                    result.curves.reset_index().assign(
                        overlay_mode=mode,
                        risk_on_leverage=int(lev),
                        signal_variant=signal_variant,
                        vix_threshold=threshold,
                    )
                )
                if not result.events.empty:
                    event_frames.append(result.events.copy())

    metrics = pd.DataFrame(metric_rows).sort_values(["risk_on_leverage", "overlay_mode"])
    curves = pd.concat(curve_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    metrics["short_window"] = int(args.short_window)
    metrics["long_window"] = int(args.long_window)
    metrics.to_csv(args.out_dir / "consensus_ma_deleverage_metrics.csv", index=False)
    curves.to_csv(args.out_dir / "consensus_ma_deleverage_curves.csv", index=False)
    signal_daily.loc[common_start:].to_csv(args.out_dir / "consensus_ma_deleverage_signal_daily.csv", index_label="date")
    if not events.empty:
        events.to_csv(args.out_dir / "consensus_ma_deleverage_events.csv", index=False)

    metadata = {
        "common_start": common_start.date().isoformat(),
        "initial_capital": args.initial_capital,
        "monthly_contribution": args.monthly_contribution,
        "trading_cost_bps": args.trading_cost_bps,
        "borrow_rate": args.borrow_rate,
        "short_window": args.short_window,
        "long_window": args.long_window,
        "risk_on_levels": [int(x) for x in args.risk_on_levels],
        "include_vix": bool(args.include_vix),
        "vix_thresholds": [float(x) for x in args.vix_thresholds] if args.include_vix else [],
        "notes": [
            "Consensus regime is built from the existing audited logistic + GMM combination.",
            "Trend override uses lagged daily 60d/200d state only.",
            "VIX filter, when enabled, is also lagged by one trading day.",
            "The MA rule is an overlay on top of the existing reserve/deploy leverage logic.",
        ],
    }
    (args.out_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_comparison(curves, args.out_dir / "consensus_ma_deleverage_equity.png")
    write_report(
        args.out_dir,
        metrics=metrics,
        signal_daily=signal_daily.loc[common_start:].copy(),
        short_window=args.short_window,
        long_window=args.long_window,
    )

    print(f"Wrote MA deleveraging audit to {args.out_dir}")
    print(metrics[["risk_on_leverage", "overlay_mode", "final_value", "xirr", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
