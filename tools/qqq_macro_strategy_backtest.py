"""Backtest QQQ DCA and moving-average strategies with macro regime overlays.

This is research tooling, not investment advice. Signals are evaluated with
information known at the prior close and applied to the next close-to-close
return. The script compares buy-and-hold, monthly DCA, moving-average filters,
and regime-aware long/flat/short/leverage variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QQQ_PATH = ROOT / "cache" / "cache" / "cache" / "QQQ_daily.parquet"
DEFAULT_REGIME_PATH = (
    ROOT / "reports" / "qqq_macro_regime_shock_audit_3m_ma" / "derived_daily_regime_dataset.csv"
)
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_macro_strategy_backtest_3m_ma"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_qqq(path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing QQQ daily parquet: {path}")

    raw = pd.read_parquet(path).copy()
    if "date" in raw:
        index = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    elif "time" in raw:
        index = pd.to_datetime(raw["time"]).dt.tz_localize(None)
    else:
        raise ValueError("QQQ daily parquet must contain either date or time")

    required = {"o", "h", "l", "c", "adj_c", "v"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"QQQ daily parquet missing columns: {sorted(missing)}")

    factor = (raw["adj_c"].astype(float) / raw["c"].replace(0.0, np.nan).astype(float)).replace(
        [np.inf, -np.inf], np.nan
    )
    factor = factor.fillna(1.0)
    out = pd.DataFrame(
        {
            "open": raw["o"].astype(float).to_numpy() * factor.to_numpy(),
            "high": raw["h"].astype(float).to_numpy() * factor.to_numpy(),
            "low": raw["l"].astype(float).to_numpy() * factor.to_numpy(),
            "close": raw["adj_c"].astype(float).to_numpy(),
            "volume": raw["v"].astype(float).to_numpy(),
        },
        index=index,
    ).sort_index()
    out.index.name = "date"

    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    if len(out) < 260:
        raise ValueError(f"Only {len(out)} QQQ rows after filtering; need at least about 1 year.")
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"}).fillna(False)


def load_regime(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing macro regime dataset: {path}. Run tools/qqq_macro_regime_shock_audit.py first."
        )

    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    regime = raw.reindex(raw.index.union(index)).sort_index().ffill().reindex(index)

    required = [
        "risk_regime",
        "risk_score_0_100",
        "major_shock",
        "oil_up_shock",
        "shock_count",
    ]
    missing = [column for column in required if column not in regime.columns]
    if missing:
        raise ValueError(f"Regime dataset missing columns: {missing}")

    for column in [column for column in regime.columns if column.endswith("_shock") or column == "major_shock"]:
        regime[column] = _as_bool(regime[column])
    regime["risk_regime"] = regime["risk_regime"].fillna("unknown").astype(str)
    regime["shock_count"] = pd.to_numeric(regime["shock_count"], errors="coerce").fillna(0.0)
    regime["risk_score_0_100"] = pd.to_numeric(regime["risk_score_0_100"], errors="coerce")
    return regime


def build_targets(price: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    close = price["close"]
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    above_sma200 = close > sma200
    bull_ma = (sma50 > sma200) & above_sma200
    bear_ma = close < sma200
    risk_on = regime["risk_regime"] == "risk_on"
    neutral = regime["risk_regime"] == "neutral"
    risk_off = regime["risk_regime"] == "risk_off"
    major_shock = regime["major_shock"]
    oil_up = regime["oil_up_shock"]

    targets = pd.DataFrame(index=close.index)
    targets["Buy & Hold QQQ 1x"] = 1.0
    targets["Buy & Hold QQQ 2x"] = 2.0
    targets["SMA200 Long/Cash"] = np.where(above_sma200, 1.0, 0.0)
    targets["SMA200 Long/Short 50%"] = np.where(above_sma200, 1.0, -0.5)
    targets["SMA50/200 Trend 2x"] = np.where(bull_ma, 2.0, 0.0)

    regime_scaled = pd.Series(1.0, index=close.index)
    regime_scaled.loc[risk_on] = 1.5
    regime_scaled.loc[neutral] = 1.0
    regime_scaled.loc[risk_off] = 0.25
    regime_scaled.loc[major_shock] = np.minimum(regime_scaled.loc[major_shock], 0.25)
    regime_scaled.loc[oil_up] = np.minimum(regime_scaled.loc[oil_up], 1.0)
    targets["Regime Scaled Long Only"] = regime_scaled

    regime_ma_flat = pd.Series(0.0, index=close.index)
    regime_ma_flat.loc[above_sma200 & risk_on] = 1.5
    regime_ma_flat.loc[above_sma200 & neutral] = 1.0
    regime_ma_flat.loc[above_sma200 & risk_off] = 0.5
    regime_ma_flat.loc[~above_sma200 & risk_on] = 0.75
    regime_ma_flat.loc[major_shock] = np.minimum(regime_ma_flat.loc[major_shock], 0.25)
    regime_ma_flat.loc[oil_up] = np.minimum(regime_ma_flat.loc[oil_up], 1.0)
    targets["Regime + SMA Long/Flat"] = regime_ma_flat

    regime_ma_ls = pd.Series(0.0, index=close.index)
    regime_ma_ls.loc[above_sma200 & risk_on] = 2.0
    regime_ma_ls.loc[above_sma200 & neutral] = 1.0
    regime_ma_ls.loc[above_sma200 & risk_off] = 0.25
    regime_ma_ls.loc[bear_ma & risk_on] = 0.5
    regime_ma_ls.loc[bear_ma & risk_off] = -0.75
    regime_ma_ls.loc[bear_ma & major_shock] = -0.5
    regime_ma_ls.loc[oil_up & (regime_ma_ls > 1.0)] = 1.0
    targets["Regime + SMA Long/Short"] = regime_ma_ls

    regime_only_ls = pd.Series(1.0, index=close.index)
    regime_only_ls.loc[risk_on] = 2.0
    regime_only_ls.loc[neutral] = 1.0
    regime_only_ls.loc[risk_off] = -0.5
    regime_only_ls.loc[major_shock] = np.minimum(regime_only_ls.loc[major_shock], -0.25)
    regime_only_ls.loc[oil_up & (regime_only_ls > 0.75)] = 0.75
    targets["Regime Only Long/Short"] = regime_only_ls

    targets = targets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return targets


def financing_cost(exposure: float, borrow_rate: float, short_borrow_rate: float) -> float:
    gross = abs(float(exposure))
    leverage_cost = max(gross - 1.0, 0.0) * borrow_rate / 252.0
    short_cost = max(-float(exposure), 0.0) * short_borrow_rate / 252.0
    return leverage_cost + short_cost


def simulate_lump_sum(
    returns: pd.Series,
    target: pd.Series,
    *,
    capital: float,
    borrow_rate: float,
    short_borrow_rate: float,
    trading_cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    cost_rate = trading_cost_bps / 10_000.0
    signal = target.shift(1).fillna(0.0).astype(float)
    equity = np.empty(len(returns), dtype=float)
    realized_exposure = np.empty(len(returns), dtype=float)
    account_equity = float(capital)
    active_exposure = 0.0
    equity[0] = account_equity
    realized_exposure[0] = active_exposure

    for i in range(1, len(returns)):
        next_exposure = float(signal.iloc[i])
        if account_equity > 0.0 and next_exposure != active_exposure:
            account_equity -= account_equity * abs(next_exposure - active_exposure) * cost_rate
            active_exposure = next_exposure
        if account_equity > 0.0:
            cost = financing_cost(active_exposure, borrow_rate, short_borrow_rate)
            account_equity *= 1.0 + active_exposure * float(returns.iloc[i]) - cost
            if account_equity <= 0.0:
                account_equity = 0.0
                active_exposure = 0.0
        equity[i] = account_equity
        realized_exposure[i] = active_exposure

    return (
        pd.Series(equity, index=returns.index, name=target.name),
        pd.Series(realized_exposure, index=returns.index, name=target.name),
    )


def simulate_dca(
    returns: pd.Series,
    target: pd.Series,
    *,
    capital: float,
    borrow_rate: float,
    short_borrow_rate: float,
    trading_cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    cost_rate = trading_cost_bps / 10_000.0
    signal_for_return = target.shift(1).fillna(0.0).astype(float)
    signal_after_close = target.fillna(0.0).astype(float)
    months = pd.PeriodIndex(returns.index, freq="M").unique()
    installment = capital / len(months)

    outside_cash = float(capital)
    account_equity = 0.0
    active_exposure = 0.0
    previous_month: pd.Period | None = None
    equity: list[float] = []
    realized_exposure: list[float] = []

    for i, date in enumerate(returns.index):
        if i > 0 and account_equity > 0.0:
            next_exposure = float(signal_for_return.iloc[i])
            if next_exposure != active_exposure:
                account_equity -= account_equity * abs(next_exposure - active_exposure) * cost_rate
                active_exposure = next_exposure
            cost = financing_cost(active_exposure, borrow_rate, short_borrow_rate)
            account_equity *= 1.0 + active_exposure * float(returns.iloc[i]) - cost
            if account_equity <= 0.0:
                account_equity = 0.0
                active_exposure = 0.0

        month = pd.Period(date, freq="M")
        if month != previous_month:
            contribution = min(installment, outside_cash)
            if contribution > 0.0:
                target_after_close = float(signal_after_close.iloc[i])
                trade_cost = contribution * abs(target_after_close) * cost_rate
                account_equity += max(0.0, contribution - trade_cost)
                outside_cash -= contribution
            previous_month = month

        total_equity = outside_cash + account_equity
        equity.append(total_equity)
        realized_exposure.append((account_equity * active_exposure / total_equity) if total_equity > 0.0 else 0.0)

    return (
        pd.Series(equity, index=returns.index, name=f"DCA {target.name}"),
        pd.Series(realized_exposure, index=returns.index, name=f"DCA {target.name}"),
    )


def max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return np.nan
    return float((clean / clean.cummax() - 1.0).min())


def strategy_metrics(equity: pd.Series, exposure: pd.Series, label: str) -> dict[str, Any]:
    eq = equity.dropna()
    daily = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0 if eq.iloc[0] > 0.0 else np.nan
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0.0 and total_return > -1.0 else -1.0
    volatility = float(daily.std() * np.sqrt(252.0)) if len(daily) > 1 else np.nan
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252.0)) if len(daily) > 1 and daily.std() > 0.0 else np.nan
    downside = daily[daily < 0.0]
    sortino = (
        float(daily.mean() / downside.std() * np.sqrt(252.0))
        if len(downside) > 1 and downside.std() > 0.0
        else np.nan
    )
    dd = max_drawdown(eq)
    exp = exposure.reindex(eq.index).fillna(0.0)
    return {
        "strategy": label,
        "start_date": eq.index[0].date().isoformat(),
        "end_date": eq.index[-1].date().isoformat(),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": dd,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(cagr / abs(dd)) if dd < 0.0 else np.nan,
        "avg_signed_exposure": float(exp.mean()),
        "avg_gross_exposure": float(exp.abs().mean()),
        "max_gross_exposure": float(exp.abs().max()),
        "pct_days_long": float((exp > 0.05).mean()),
        "pct_days_short": float((exp < -0.05).mean()),
        "pct_days_flat": float((exp.abs() <= 0.05).mean()),
        "turnover": float(exp.diff().abs().sum()),
        "blown_up": bool(eq.iloc[-1] <= 0.0),
    }


def yearly_returns(equity_curves: pd.DataFrame) -> pd.DataFrame:
    annual = equity_curves.resample("YE").last().pct_change()
    annual.index = annual.index.year
    annual.index.name = "year"
    return annual


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.1f}%"


def _fmt_currency(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${float(value):,.0f}"


def write_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    current_targets: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    ranked = metrics.sort_values(["sharpe", "cagr"], ascending=False)
    top = ranked.head(14)
    lines = [
        "# QQQ Macro-Regime DCA and Moving-Average Strategy Backtest",
        "",
        "This is a research backtest, not investment advice or a live trading recommendation.",
        "",
        "## Method",
        "",
        "- Uses the 3-month-smoothed macro risk-regime dataset from the prior audit.",
        "- Signals are shifted by one trading day before applying returns to reduce look-ahead bias.",
        "- Monthly DCA splits initial capital into equal monthly installments; undeployed cash earns 0%.",
        "- Leverage is modeled as daily rebalanced exposure with borrow cost on gross exposure above 1x.",
        "- Short exposure is charged a simple annualized short-borrow cost and can lose money quickly in rallies.",
        "",
        "## Configuration",
        "",
        f"- Date range: `{summary['start_date']}` to `{summary['end_date']}`",
        f"- Initial capital: `{_fmt_currency(summary['initial_capital'])}`",
        f"- Borrow rate: `{_fmt_pct(summary['borrow_rate'])}` annual",
        f"- Short borrow rate: `{_fmt_pct(summary['short_borrow_rate'])}` annual",
        f"- Trading cost: `{summary['trading_cost_bps']:.1f}` bps per notional exposure change",
        f"- Latest QQQ adjusted close: `{summary['latest_close']:.2f}`",
        f"- Latest macro regime: `{summary['latest_regime']}`",
        f"- Latest risk score 0-100: `{summary['latest_risk_score_0_100']:.1f}`",
        f"- Latest major shock: `{summary['latest_major_shock']}`",
        f"- Latest active oil-up shock: `{summary['latest_oil_up_shock']}`",
        "",
        "## Ranked Results",
        "",
        "| Strategy | Final | CAGR | Max DD | Sharpe | Avg Gross Exp | Short Days |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {strategy} | {final} | {cagr} | {dd} | {sharpe:.2f} | {gross} | {short} |".format(
                strategy=row["strategy"],
                final=_fmt_currency(row["final_equity"]),
                cagr=_fmt_pct(row["cagr"]),
                dd=_fmt_pct(row["max_drawdown"]),
                sharpe=float(row["sharpe"]) if pd.notna(row["sharpe"]) else float("nan"),
                gross=_fmt_pct(row["avg_gross_exposure"]),
                short=_fmt_pct(row["pct_days_short"]),
            )
        )

    lines.extend(
        [
            "",
            "## Current Targets",
            "",
            "| Strategy | Current Target Exposure | Prior-Day Applied Exposure |",
            "|---|---:|---:|",
        ]
    )
    for _, row in current_targets.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['current_target_exposure']:.2f}x | {row['prior_day_applied_exposure']:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the table as a hypothesis generator, not as a final allocation model.",
            "- Compare Sharpe, max drawdown, and short-day exposure before comparing final equity.",
            "- Leveraged strategies need stress testing beyond this simple daily-rebalanced model.",
            "- Short variants are included because the account can short; they should be treated as risk overlays, not default DCA behavior.",
            "",
            "## Files",
            "",
            "- `metrics.csv`: numeric performance metrics for each strategy",
            "- `equity_curves.csv`: daily equity curves",
            "- `target_exposure.csv`: daily signed target exposure before one-day signal shift",
            "- `realized_exposure.csv`: daily signed exposure actually applied in the simulation",
            "- `current_targets.csv`: latest target exposures",
            "- `yearly_returns.csv`: calendar-year return table",
            "- `summary.json`: run configuration and latest regime snapshot",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest QQQ DCA/MA strategies with macro regime overlays.")
    parser.add_argument("--qqq-path", type=Path, default=DEFAULT_QQQ_PATH)
    parser.add_argument("--regime-path", type=Path, default=DEFAULT_REGIME_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument("--short-borrow-rate", type=float, default=0.02)
    parser.add_argument("--trading-cost-bps", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qqq = load_qqq(args.qqq_path, args.start, args.end)
    regime = load_regime(args.regime_path, qqq.index)
    returns = qqq["close"].pct_change().fillna(0.0)
    targets = build_targets(qqq, regime)

    strategies: dict[str, tuple[pd.Series, pd.Series]] = {}
    for name in targets.columns:
        equity, exposure = simulate_lump_sum(
            returns,
            targets[name].rename(name),
            capital=args.initial_capital,
            borrow_rate=args.borrow_rate,
            short_borrow_rate=args.short_borrow_rate,
            trading_cost_bps=args.trading_cost_bps,
        )
        strategies[name] = (equity.rename(name), exposure.rename(name))

    dca_names = [
        "Buy & Hold QQQ 1x",
        "Buy & Hold QQQ 2x",
        "Regime Scaled Long Only",
        "Regime + SMA Long/Flat",
        "Regime + SMA Long/Short",
    ]
    for name in dca_names:
        label = f"DCA {name}"
        equity, exposure = simulate_dca(
            returns,
            targets[name].rename(name),
            capital=args.initial_capital,
            borrow_rate=args.borrow_rate,
            short_borrow_rate=args.short_borrow_rate,
            trading_cost_bps=args.trading_cost_bps,
        )
        strategies[label] = (equity.rename(label), exposure.rename(label))

    equity_curves = pd.DataFrame({name: result[0] for name, result in strategies.items()})
    realized_exposure = pd.DataFrame({name: result[1] for name, result in strategies.items()})
    metrics = pd.DataFrame(
        [strategy_metrics(equity, realized_exposure[name], name) for name, (equity, _) in strategies.items()]
    ).sort_values(["sharpe", "cagr"], ascending=False)

    current_targets = pd.DataFrame(
        {
            "strategy": targets.columns,
            "current_target_exposure": targets.iloc[-1].to_numpy(dtype=float),
            "prior_day_applied_exposure": targets.shift(1).fillna(0.0).iloc[-1].to_numpy(dtype=float),
        }
    )

    summary = {
        "qqq_path": str(args.qqq_path),
        "regime_path": str(args.regime_path),
        "start_date": qqq.index[0].date().isoformat(),
        "end_date": qqq.index[-1].date().isoformat(),
        "rows": int(len(qqq)),
        "initial_capital": float(args.initial_capital),
        "borrow_rate": float(args.borrow_rate),
        "short_borrow_rate": float(args.short_borrow_rate),
        "trading_cost_bps": float(args.trading_cost_bps),
        "latest_close": float(qqq["close"].iloc[-1]),
        "latest_regime": str(regime["risk_regime"].iloc[-1]),
        "latest_risk_score_0_100": float(regime["risk_score_0_100"].iloc[-1]),
        "latest_major_shock": bool(regime["major_shock"].iloc[-1]),
        "latest_oil_up_shock": bool(regime["oil_up_shock"].iloc[-1]),
    }

    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    equity_curves.to_csv(args.out_dir / "equity_curves.csv", index_label="date")
    targets.to_csv(args.out_dir / "target_exposure.csv", index_label="date")
    realized_exposure.to_csv(args.out_dir / "realized_exposure.csv", index_label="date")
    current_targets.to_csv(args.out_dir / "current_targets.csv", index=False)
    yearly_returns(equity_curves).to_csv(args.out_dir / "yearly_returns.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    write_report(args.out_dir, metrics, current_targets, summary)

    print(f"Saved QQQ macro strategy backtest under: {args.out_dir}")
    print(metrics[["strategy", "final_equity", "cagr", "max_drawdown", "sharpe"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
