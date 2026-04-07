"""Backtest QQQ turtle-trading variants against QQQ DCA.

The script uses the local Tiingo 5-minute parquet cache, aggregates it to
regular-session daily OHLCV, and compares buy-and-hold, monthly DCA, and
turtle/Donchian variants with a hard leverage cap.

This is research tooling, not investment advice. Signals are evaluated on the
daily close and exposure changes take effect on the next close-to-close return.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "cache" / "cache" / "tiingo"
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_turtle_dca_backtest"


@dataclass(frozen=True)
class TurtleConfig:
    name: str
    initial_capital: float
    entry_period: int
    exit_period: int
    atr_period: int
    sma_period: int
    volume_period: int
    max_leverage: float
    borrow_rate: float
    slippage_bps: float
    commission_bps: float
    sizing: str
    risk_per_unit: float
    boosted_risk_per_unit: float
    use_volume_filter: bool
    use_volume_risk_boost: bool
    min_rel_volume: float
    close_location_threshold: float
    use_sma_filter: bool
    allow_longs: bool
    allow_shorts: bool
    enable_pyramiding: bool
    max_units: int
    add_atr_interval: float
    atr_stop_mult: float


@dataclass
class Position:
    direction: int
    leverage: float
    units: int
    entry_i: int
    entry_date: pd.Timestamp
    entry_price: float
    entry_equity: float
    avg_entry: float
    last_add_price: float
    stop: float
    entry_rel_volume: float
    entry_volume_confirmed: bool
    max_leverage_seen: float


@dataclass
class BacktestResult:
    equity: pd.Series
    leverage: pd.Series
    trades: list[dict[str, Any]]


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


def load_daily_ohlcv(
    symbol: str,
    data_dir: Path,
    min_bars_per_day: int,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, Path]:
    path = data_dir / f"{symbol.upper()}_5m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing local data file: {path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the local Tiingo parquet cache") from exc

    table = pq.read_table(path, columns=["time", "o", "h", "l", "c", "v"])
    raw = table.to_pandas()
    raw.columns = ["time", "open", "high", "low", "close", "volume"]
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw["ny_date"] = raw["time"].dt.tz_convert("America/New_York").dt.date

    daily = (
        raw.groupby("ny_date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bars=("close", "size"),
            first_bar_utc=("time", "first"),
            last_bar_utc=("time", "last"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["ny_date"])
    daily = daily.drop(columns=["ny_date"]).set_index("date").sort_index()

    # The local cache can end mid-session. Dropping thin days avoids treating a
    # partial trading day as a real daily close.
    daily = daily[daily["bars"] >= int(min_bars_per_day)].copy()

    if start:
        daily = daily[daily.index >= pd.Timestamp(start)]
    if end:
        daily = daily[daily.index <= pd.Timestamp(end)]

    if len(daily) < 260:
        raise ValueError(
            f"Only {len(daily)} usable daily bars after filtering; need at least about 1 year."
        )

    return daily, path


def add_indicators(
    daily: pd.DataFrame,
    entry_period: int,
    exit_period: int,
    atr_period: int,
    sma_period: int,
    volume_period: int,
) -> pd.DataFrame:
    df = daily.copy()
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["return"] = df["close"].pct_change().fillna(0.0)
    df["atr"] = true_range.rolling(atr_period).mean()
    df["sma"] = df["close"].rolling(sma_period).mean()
    df["entry_high"] = df["high"].rolling(entry_period).max().shift(1)
    df["entry_low"] = df["low"].rolling(entry_period).min().shift(1)
    df["exit_high"] = df["high"].rolling(exit_period).max().shift(1)
    df["exit_low"] = df["low"].rolling(exit_period).min().shift(1)
    df["volume_sma"] = df["volume"].rolling(volume_period).mean().shift(1)
    df["rel_volume"] = df["volume"] / df["volume_sma"]

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["close_location"] = ((df["close"] - df["low"]) / bar_range).clip(0.0, 1.0)
    df["close_location"] = df["close_location"].fillna(0.5)
    return df


def buy_and_hold(close: pd.Series, capital: float, label: str) -> pd.Series:
    shares = capital / float(close.iloc[0])
    return (shares * close).rename(label)


def dca_monthly(close: pd.Series, capital: float, label: str) -> pd.Series:
    months = pd.PeriodIndex(close.index, freq="M").unique()
    installment = capital / len(months)
    cash = capital
    shares = 0.0
    equity: list[float] = []
    previous_month: pd.Period | None = None

    for date, price in close.items():
        month = pd.Period(date, freq="M")
        if month != previous_month:
            contribution = min(installment, cash)
            shares += contribution / float(price)
            cash -= contribution
            previous_month = month
        equity.append(shares * float(price) + cash)

    return pd.Series(equity, index=close.index, name=label)


def leveraged_dca_monthly(
    close: pd.Series,
    capital: float,
    leverage: float,
    borrow_rate: float,
    slippage_bps: float,
    commission_bps: float,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    """Monthly DCA into a daily-rebalanced leveraged QQQ sleeve.

    Undeployed capital remains in cash at 0%. Deployed capital earns
    ``leverage * QQQ daily return`` less borrow cost on leverage above 1x.
    """
    months = pd.PeriodIndex(close.index, freq="M").unique()
    installment = capital / len(months)
    trading_cost_rate = (slippage_bps + commission_bps) / 10_000.0
    cash = capital
    sleeve_equity = 0.0
    equity: list[float] = []
    portfolio_leverage: list[float] = []
    previous_month: pd.Period | None = None
    previous_price: float | None = None

    for date, price in close.items():
        current_price = float(price)
        if previous_price is not None and sleeve_equity > 0.0:
            day_return = current_price / previous_price - 1.0
            borrow_cost = max(leverage - 1.0, 0.0) * borrow_rate / 252.0
            sleeve_equity *= max(0.0, 1.0 + leverage * day_return - borrow_cost)

        month = pd.Period(date, freq="M")
        if month != previous_month:
            contribution = min(installment, cash)
            if contribution > 0.0:
                trade_cost = contribution * leverage * trading_cost_rate
                sleeve_equity += max(0.0, contribution - trade_cost)
                cash -= contribution
            previous_month = month

        total_equity = cash + sleeve_equity
        equity.append(total_equity)
        gross_exposure = sleeve_equity * leverage
        portfolio_leverage.append(gross_exposure / total_equity if total_equity > 0.0 else 0.0)
        previous_price = current_price

    equity_series = pd.Series(equity, index=close.index, name=label)
    leverage_series = pd.Series(portfolio_leverage, index=close.index, name=f"{label} leverage")
    return equity_series, leverage_series


def ma_tactical_dca_sleeve(
    close: pd.Series,
    capital: float,
    leverage: float,
    borrow_rate: float,
    slippage_bps: float,
    commission_bps: float,
    fast_ma: int,
    slow_ma: int,
    threshold: float,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    """DCA normally, but route cheap-regime contributions into a 3x sleeve."""
    months = pd.PeriodIndex(close.index, freq="M").unique()
    installment = capital / len(months)
    cost_rate = (slippage_bps + commission_bps) / 10_000.0
    ma_ratio = (close.rolling(fast_ma).mean() / close.rolling(slow_ma).mean()).shift(1)
    cash = capital
    normal_shares = 0.0
    leveraged_equity = 0.0
    equity: list[float] = []
    portfolio_leverage: list[float] = []
    previous_month: pd.Period | None = None
    previous_price: float | None = None

    for date, price in close.items():
        current_price = float(price)
        ratio = float(ma_ratio.loc[date]) if pd.notna(ma_ratio.loc[date]) else np.nan
        cheap_regime = bool(np.isfinite(ratio) and ratio < threshold)

        if previous_price is not None:
            if not cheap_regime and leveraged_equity > 0.0:
                post_sell = max(0.0, leveraged_equity - leveraged_equity * leverage * cost_rate)
                post_buy = max(0.0, post_sell - post_sell * cost_rate)
                normal_shares += post_buy / previous_price
                leveraged_equity = 0.0
            if leveraged_equity > 0.0:
                day_return = current_price / previous_price - 1.0
                borrow_cost = max(leverage - 1.0, 0.0) * borrow_rate / 252.0
                leveraged_equity *= max(0.0, 1.0 + leverage * day_return - borrow_cost)

        month = pd.Period(date, freq="M")
        if month != previous_month:
            contribution = min(installment, cash)
            if contribution > 0.0:
                if cheap_regime:
                    leveraged_equity += max(0.0, contribution - contribution * leverage * cost_rate)
                else:
                    normal_shares += max(0.0, contribution - contribution * cost_rate) / current_price
                cash -= contribution
            previous_month = month

        normal_equity = normal_shares * current_price
        total_equity = cash + normal_equity + leveraged_equity
        gross_exposure = normal_equity + leveraged_equity * leverage
        equity.append(total_equity)
        portfolio_leverage.append(gross_exposure / total_equity if total_equity > 0.0 else 0.0)
        previous_price = current_price

    equity_series = pd.Series(equity, index=close.index, name=label)
    leverage_series = pd.Series(portfolio_leverage, index=close.index, name=f"{label} leverage")
    return equity_series, leverage_series


def ma_tactical_dca_portfolio(
    close: pd.Series,
    capital: float,
    leverage: float,
    borrow_rate: float,
    slippage_bps: float,
    commission_bps: float,
    fast_ma: int,
    slow_ma: int,
    threshold: float,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    """DCA while switching the whole invested balance between 1x and 3x."""
    months = pd.PeriodIndex(close.index, freq="M").unique()
    installment = capital / len(months)
    cost_rate = (slippage_bps + commission_bps) / 10_000.0
    ma_ratio = (close.rolling(fast_ma).mean() / close.rolling(slow_ma).mean()).shift(1)
    cash = capital
    invested_equity = 0.0
    active_leverage = 1.0
    equity: list[float] = []
    portfolio_leverage: list[float] = []
    previous_month: pd.Period | None = None
    previous_price: float | None = None

    for date, price in close.items():
        current_price = float(price)
        ratio = float(ma_ratio.loc[date]) if pd.notna(ma_ratio.loc[date]) else np.nan
        cheap_regime = bool(np.isfinite(ratio) and ratio < threshold)
        target_leverage = leverage if cheap_regime else 1.0

        if previous_price is not None:
            if invested_equity > 0.0 and target_leverage != active_leverage:
                leverage_delta = abs(target_leverage - active_leverage)
                invested_equity = max(0.0, invested_equity - invested_equity * leverage_delta * cost_rate)
            active_leverage = target_leverage
            if invested_equity > 0.0:
                day_return = current_price / previous_price - 1.0
                borrow_cost = max(active_leverage - 1.0, 0.0) * borrow_rate / 252.0
                invested_equity *= max(0.0, 1.0 + active_leverage * day_return - borrow_cost)
        else:
            active_leverage = target_leverage

        month = pd.Period(date, freq="M")
        if month != previous_month:
            contribution = min(installment, cash)
            if contribution > 0.0:
                invested_equity += max(0.0, contribution - contribution * active_leverage * cost_rate)
                cash -= contribution
            previous_month = month

        total_equity = cash + invested_equity
        gross_exposure = invested_equity * active_leverage
        equity.append(total_equity)
        portfolio_leverage.append(gross_exposure / total_equity if total_equity > 0.0 else 0.0)
        previous_price = current_price

    equity_series = pd.Series(equity, index=close.index, name=label)
    leverage_series = pd.Series(portfolio_leverage, index=close.index, name=f"{label} leverage")
    return equity_series, leverage_series


def _volume_confirmed(row: pd.Series, direction: int, cfg: TurtleConfig) -> bool:
    rel_volume = float(row.get("rel_volume", np.nan))
    if not np.isfinite(rel_volume):
        return False

    close_location = float(row.get("close_location", 0.5))
    if direction == 1:
        location_ok = close_location >= cfg.close_location_threshold
    else:
        location_ok = close_location <= 1.0 - cfg.close_location_threshold
    return rel_volume >= cfg.min_rel_volume and location_ok


def _target_leverage(row: pd.Series, cfg: TurtleConfig, volume_confirmed: bool) -> float:
    if cfg.sizing == "fixed":
        return cfg.max_leverage
    if cfg.sizing != "atr":
        raise ValueError(f"Unsupported sizing mode: {cfg.sizing}")

    atr = float(row["atr"])
    close = float(row["close"])
    if atr <= 0 or close <= 0:
        return 0.0

    risk = cfg.risk_per_unit
    if cfg.use_volume_risk_boost and volume_confirmed:
        risk = cfg.boosted_risk_per_unit
    return min(max(risk / (atr / close), 0.0), cfg.max_leverage)


def _apply_trade_cost(equity: float, leverage_delta: float, cfg: TurtleConfig) -> float:
    cost_rate = (cfg.slippage_bps + cfg.commission_bps) / 10_000.0
    return max(0.0, equity - equity * abs(leverage_delta) * cost_rate)


def _new_stop(direction: int, avg_entry: float, atr: float, cfg: TurtleConfig) -> float:
    if direction == 1:
        return avg_entry - cfg.atr_stop_mult * atr
    return avg_entry + cfg.atr_stop_mult * atr


def _trade_row(
    cfg: TurtleConfig,
    position: Position,
    exit_i: int,
    exit_date: pd.Timestamp,
    exit_price: float,
    current_equity: float,
    exit_reason: str,
) -> dict[str, Any]:
    return {
        "strategy": cfg.name,
        "direction": "long" if position.direction == 1 else "short",
        "entry_date": position.entry_date,
        "exit_date": exit_date,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "entry_equity": position.entry_equity,
        "exit_equity": current_equity,
        "pnl_dollars": current_equity - position.entry_equity,
        "pnl_pct": current_equity / position.entry_equity - 1.0,
        "entry_leverage": position.leverage,
        "max_leverage_seen": position.max_leverage_seen,
        "units": position.units,
        "bars_held": exit_i - position.entry_i,
        "entry_rel_volume": position.entry_rel_volume,
        "entry_volume_confirmed": position.entry_volume_confirmed,
        "exit_reason": exit_reason,
    }


def run_turtle_strategy(daily: pd.DataFrame, cfg: TurtleConfig) -> BacktestResult:
    df = add_indicators(
        daily,
        entry_period=cfg.entry_period,
        exit_period=cfg.exit_period,
        atr_period=cfg.atr_period,
        sma_period=cfg.sma_period,
        volume_period=cfg.volume_period,
    )
    n = len(df)
    equity = np.full(n, np.nan)
    leverage = np.zeros(n)
    equity[0] = cfg.initial_capital
    current_equity = cfg.initial_capital
    position: Position | None = None
    trades: list[dict[str, Any]] = []

    warmup = max(
        cfg.entry_period,
        cfg.exit_period,
        cfg.atr_period,
        cfg.sma_period if cfg.use_sma_filter else 0,
        cfg.volume_period if (cfg.use_volume_filter or cfg.use_volume_risk_boost) else 0,
    )

    for i in range(1, n):
        row = df.iloc[i]
        prev_close = float(df["close"].iloc[i - 1])
        close = float(row["close"])
        day_return = 0.0 if prev_close <= 0 else close / prev_close - 1.0

        if position is not None:
            gross_leverage = min(abs(position.leverage), cfg.max_leverage)
            signed_leverage = position.direction * gross_leverage
            daily_borrow_cost = max(gross_leverage - 1.0, 0.0) * cfg.borrow_rate / 252.0
            current_equity *= max(0.0, 1.0 + signed_leverage * day_return - daily_borrow_cost)
            leverage[i] = signed_leverage
            position.max_leverage_seen = max(position.max_leverage_seen, gross_leverage)

        equity[i] = current_equity
        if current_equity <= 0.0:
            equity[i:] = 0.0
            break

        if i < warmup:
            continue

        required_cols = ["atr", "entry_high", "entry_low", "exit_high", "exit_low"]
        if cfg.use_sma_filter:
            required_cols.append("sma")
        if any(not np.isfinite(float(row[col])) for col in required_cols):
            continue

        if position is not None:
            direction = position.direction
            exit_reason = ""
            if direction == 1:
                if close < float(row["exit_low"]):
                    exit_reason = "donchian_exit"
                elif close < position.stop:
                    exit_reason = "atr_stop"
                elif cfg.use_sma_filter and close < float(row["sma"]):
                    exit_reason = "sma_break"
            else:
                if close > float(row["exit_high"]):
                    exit_reason = "donchian_exit"
                elif close > position.stop:
                    exit_reason = "atr_stop"
                elif cfg.use_sma_filter and close > float(row["sma"]):
                    exit_reason = "sma_break"

            if exit_reason:
                current_equity = _apply_trade_cost(current_equity, position.leverage, cfg)
                equity[i] = current_equity
                trades.append(
                    _trade_row(
                        cfg,
                        position,
                        exit_i=i,
                        exit_date=df.index[i],
                        exit_price=close,
                        current_equity=current_equity,
                        exit_reason=exit_reason,
                    )
                )
                position = None
                continue

            if cfg.enable_pyramiding and position.units < cfg.max_units:
                if direction == 1:
                    add_signal = close >= position.last_add_price + cfg.add_atr_interval * float(row["atr"])
                else:
                    add_signal = close <= position.last_add_price - cfg.add_atr_interval * float(row["atr"])

                if add_signal and position.leverage < cfg.max_leverage:
                    add_leverage = _target_leverage(row, cfg, volume_confirmed=False)
                    add_leverage = min(add_leverage, cfg.max_leverage - position.leverage)
                    if add_leverage > 1e-9:
                        next_leverage = position.leverage + add_leverage
                        current_equity = _apply_trade_cost(current_equity, add_leverage, cfg)
                        equity[i] = current_equity
                        position.avg_entry = (
                            position.avg_entry * position.leverage + close * add_leverage
                        ) / next_leverage
                        position.leverage = next_leverage
                        position.units += 1
                        position.last_add_price = close
                        position.stop = _new_stop(direction, position.avg_entry, float(row["atr"]), cfg)
                        position.max_leverage_seen = max(position.max_leverage_seen, next_leverage)
            continue

        above_sma = (not cfg.use_sma_filter) or close > float(row["sma"])
        below_sma = (not cfg.use_sma_filter) or close < float(row["sma"])
        long_breakout = cfg.allow_longs and close > float(row["entry_high"]) and above_sma
        short_breakout = cfg.allow_shorts and close < float(row["entry_low"]) and below_sma

        direction = 0
        if long_breakout:
            direction = 1
        elif short_breakout:
            direction = -1
        if direction == 0:
            continue

        volume_confirmed = _volume_confirmed(row, direction, cfg)
        if cfg.use_volume_filter and not volume_confirmed:
            continue

        target_leverage = _target_leverage(row, cfg, volume_confirmed=volume_confirmed)
        if target_leverage <= 1e-9:
            continue

        current_equity = _apply_trade_cost(current_equity, target_leverage, cfg)
        equity[i] = current_equity
        position = Position(
            direction=direction,
            leverage=target_leverage,
            units=1,
            entry_i=i,
            entry_date=df.index[i],
            entry_price=close,
            entry_equity=current_equity,
            avg_entry=close,
            last_add_price=close,
            stop=_new_stop(direction, close, float(row["atr"]), cfg),
            entry_rel_volume=float(row.get("rel_volume", np.nan)),
            entry_volume_confirmed=volume_confirmed,
            max_leverage_seen=target_leverage,
        )

    if position is not None and current_equity > 0.0:
        current_equity = _apply_trade_cost(current_equity, position.leverage, cfg)
        equity[-1] = current_equity
        trades.append(
            _trade_row(
                cfg,
                position,
                exit_i=n - 1,
                exit_date=df.index[-1],
                exit_price=float(df["close"].iloc[-1]),
                current_equity=current_equity,
                exit_reason="end_of_data",
            )
        )

    equity_series = pd.Series(equity, index=df.index, name=cfg.name).ffill()
    leverage_series = pd.Series(leverage, index=df.index, name=f"{cfg.name} leverage")
    return BacktestResult(equity=equity_series, leverage=leverage_series, trades=trades)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.ffill().dropna()
    drawdown = eq / eq.cummax() - 1.0
    return float(drawdown.min())


def compute_metrics(
    equity: pd.Series,
    label: str,
    leverage: pd.Series | None = None,
    trades: list[dict[str, Any]] | None = None,
    dca_equity: pd.Series | None = None,
) -> dict[str, Any]:
    eq = equity.ffill().dropna()
    daily_returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    dd = max_drawdown(eq)
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252.0)
        if len(daily_returns) > 1 and daily_returns.std() > 0
        else 0.0
    )
    downside = daily_returns[daily_returns < 0.0]
    sortino = (
        daily_returns.mean() / downside.std() * np.sqrt(252.0)
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )
    calmar = cagr / abs(dd) if dd < 0 else 0.0

    active = pd.Series(dtype=float)
    if leverage is not None:
        active = leverage.abs()[leverage.abs() > 1e-9]

    trades = trades or []
    wins = [t for t in trades if float(t.get("pnl_dollars", 0.0)) > 0.0]
    losses = [t for t in trades if float(t.get("pnl_dollars", 0.0)) < 0.0]
    gross_profit = sum(float(t.get("pnl_dollars", 0.0)) for t in wins)
    gross_loss = abs(sum(float(t.get("pnl_dollars", 0.0)) for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    row: dict[str, Any] = {
        "strategy": label,
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(dd * 100.0),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "time_in_market_pct": (
            float((leverage.abs() > 1e-9).mean() * 100.0) if leverage is not None else np.nan
        ),
        "avg_abs_leverage_active": float(active.mean()) if len(active) else np.nan,
        "max_abs_leverage": float(leverage.abs().max()) if leverage is not None else np.nan,
        "trades": len(trades),
        "win_rate_pct": float(len(wins) / len(trades) * 100.0) if trades else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.nan,
    }
    if dca_equity is not None:
        dca_final = float(dca_equity.ffill().iloc[-1])
        row["final_vs_dca_pct"] = (row["final_equity"] / dca_final - 1.0) * 100.0
        row["beats_dca_final"] = row["final_equity"] > dca_final
    return row


def build_default_configs(args: argparse.Namespace) -> list[TurtleConfig]:
    base = dict(
        initial_capital=args.initial_capital,
        entry_period=args.entry_period,
        exit_period=args.exit_period,
        atr_period=args.atr_period,
        sma_period=args.sma_period,
        volume_period=args.volume_period,
        max_leverage=args.max_leverage,
        borrow_rate=args.borrow_rate,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        risk_per_unit=args.risk_per_unit,
        boosted_risk_per_unit=args.boosted_risk_per_unit,
        min_rel_volume=args.min_rel_volume,
        close_location_threshold=args.close_location_threshold,
        use_sma_filter=True,
        allow_longs=True,
        allow_shorts=args.allow_shorts,
        max_units=args.max_units,
        add_atr_interval=args.add_atr_interval,
        atr_stop_mult=args.atr_stop_mult,
    )
    lev_label = f"{args.max_leverage:g}x"
    suffix = " L/S" if args.allow_shorts else ""
    return [
        TurtleConfig(
            name=f"Turtle Fixed {lev_label}{suffix}",
            sizing="fixed",
            use_volume_filter=False,
            use_volume_risk_boost=False,
            enable_pyramiding=False,
            **base,
        ),
        TurtleConfig(
            name=f"Turtle Fixed {lev_label} RVOL Filter{suffix}",
            sizing="fixed",
            use_volume_filter=True,
            use_volume_risk_boost=False,
            enable_pyramiding=False,
            **base,
        ),
        TurtleConfig(
            name=f"Turtle ATR {lev_label} Cap{suffix}",
            sizing="atr",
            use_volume_filter=False,
            use_volume_risk_boost=False,
            enable_pyramiding=False,
            **base,
        ),
        TurtleConfig(
            name=f"Turtle ATR {lev_label} Cap RVOL Boost{suffix}",
            sizing="atr",
            use_volume_filter=False,
            use_volume_risk_boost=True,
            enable_pyramiding=False,
            **base,
        ),
        TurtleConfig(
            name=f"Turtle ATR Pyramid {lev_label} Cap RVOL Boost{suffix}",
            sizing="atr",
            use_volume_filter=False,
            use_volume_risk_boost=True,
            enable_pyramiding=True,
            **base,
        ),
    ]


def run_grid(
    daily: pd.DataFrame,
    args: argparse.Namespace,
    dca: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry_period, exit_period, max_leverage, require_volume in product(
        args.grid_entry_periods,
        args.grid_exit_periods,
        args.grid_leverages,
        [False, True],
    ):
        min_rel_volume_options = args.grid_min_rel_volumes if require_volume else [args.min_rel_volume]
        for min_rel_volume in min_rel_volume_options:
            if exit_period >= entry_period:
                continue
            cfg = TurtleConfig(
                name=(
                    f"grid_fixed_e{entry_period}_x{exit_period}_"
                    f"lev{max_leverage:g}_rvol{min_rel_volume:g}_req{int(require_volume)}"
                ),
                initial_capital=args.initial_capital,
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=args.atr_period,
                sma_period=args.sma_period,
                volume_period=args.volume_period,
                max_leverage=max_leverage,
                borrow_rate=args.borrow_rate,
                slippage_bps=args.slippage_bps,
                commission_bps=args.commission_bps,
                sizing="fixed",
                risk_per_unit=args.risk_per_unit,
                boosted_risk_per_unit=args.boosted_risk_per_unit,
                use_volume_filter=require_volume,
                use_volume_risk_boost=False,
                min_rel_volume=min_rel_volume,
                close_location_threshold=args.close_location_threshold,
                use_sma_filter=True,
                allow_longs=True,
                allow_shorts=args.allow_shorts,
                enable_pyramiding=False,
                max_units=args.max_units,
                add_atr_interval=args.add_atr_interval,
                atr_stop_mult=args.atr_stop_mult,
            )
            result = run_turtle_strategy(daily, cfg)
            row = compute_metrics(
                result.equity,
                cfg.name,
                leverage=result.leverage,
                trades=result.trades,
                dca_equity=dca,
            )
            row.update(
                {
                    "entry_period": entry_period,
                    "exit_period": exit_period,
                    "max_leverage_setting": max_leverage,
                    "require_volume": require_volume,
                    "min_rel_volume": min_rel_volume,
                }
            )
            rows.append(row)

    grid = pd.DataFrame(rows)
    if not grid.empty:
        grid = grid.sort_values(
            ["beats_dca_final", "cagr_pct", "max_drawdown_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return grid


def run_tactical_ma_grid(
    close: pd.Series,
    args: argparse.Namespace,
    dca: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fast_ma, slow_ma in product(args.tactical_grid_fast_mas, args.tactical_grid_slow_mas):
        if fast_ma >= slow_ma:
            continue
        configs = [
            (
                "sleeve",
                ma_tactical_dca_sleeve,
                f"MA{fast_ma}/{slow_ma} Tactical Sleeve DCA {args.max_leverage:g}x",
            ),
            (
                "portfolio",
                ma_tactical_dca_portfolio,
                f"MA{fast_ma}/{slow_ma} Tactical Portfolio DCA {args.max_leverage:g}x",
            ),
        ]
        for variant, runner, label in configs:
            equity, leverage = runner(
                close,
                args.initial_capital,
                leverage=args.max_leverage,
                borrow_rate=args.borrow_rate,
                slippage_bps=args.slippage_bps,
                commission_bps=args.commission_bps,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                threshold=args.tactical_ma_threshold,
                label=label,
            )
            row = compute_metrics(equity, label, leverage=leverage, dca_equity=dca)
            row.update(
                {
                    "variant": variant,
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "threshold": args.tactical_ma_threshold,
                    "days_over_1_5x": int((leverage > 1.5).sum()),
                    "avg_portfolio_leverage": float(leverage.mean()),
                }
            )
            rows.append(row)

    grid = pd.DataFrame(rows)
    if not grid.empty:
        grid = grid.sort_values(
            ["beats_dca_final", "cagr_pct", "max_drawdown_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return grid


def plot_results(
    equity_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, height_ratios=[2.4, 1.3, 1.2, 1.8])
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1])
    ax_lv = fig.add_subplot(gs[2])
    ax_tb = fig.add_subplot(gs[3])
    ax_tb.axis("off")

    for column in equity_df.columns:
        series = equity_df[column].ffill()
        ax_eq.plot(series.index, series / series.iloc[0], linewidth=1.5, label=column)
        drawdown = series / series.cummax() - 1.0
        ax_dd.plot(series.index, drawdown * 100.0, linewidth=1.2, label=column)

    ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax_eq.set_title("Equity Curves")
    ax_eq.set_ylabel("Normalized value")
    ax_eq.legend(fontsize=8, ncol=2)
    ax_eq.grid(alpha=0.25)

    ax_dd.axhline(0.0, color="black", linewidth=0.6)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.legend(fontsize=8, ncol=2)
    ax_dd.grid(alpha=0.25)

    for column in leverage_df.columns:
        ax_lv.plot(leverage_df.index, leverage_df[column], linewidth=1.1, label=column)
    if not leverage_df.empty:
        max_lev = float(leverage_df.abs().max().max())
        ax_lv.axhline(max_lev, color="red", linewidth=0.6, linestyle=":")
        ax_lv.axhline(-max_lev, color="red", linewidth=0.6, linestyle=":")
    ax_lv.axhline(0.0, color="black", linewidth=0.6)
    ax_lv.set_title("Signed Leverage")
    ax_lv.set_ylabel("Leverage")
    ax_lv.legend(fontsize=8, ncol=2)
    ax_lv.grid(alpha=0.25)

    table_cols = [
        "strategy",
        "final_equity",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe",
        "final_vs_dca_pct",
        "trades",
    ]
    table_df = metrics_df[table_cols].copy()
    table_df["final_equity"] = table_df["final_equity"].map(lambda value: f"${value:,.0f}")
    for col in ["cagr_pct", "max_drawdown_pct", "final_vs_dca_pct"]:
        table_df[col] = table_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.1f}%")
    table_df["sharpe"] = table_df["sharpe"].map(lambda value: f"{value:.2f}")

    table = ax_tb.table(
        cellText=table_df.values.tolist(),
        colLabels=table_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for col_idx in range(len(table_df.columns)):
        table[0, col_idx].set_facecolor("#263238")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(table_df) + 1):
        for col_idx in range(len(table_df.columns)):
            table[row_idx, col_idx].set_facecolor("#ECEFF1" if row_idx % 2 == 0 else "white")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    symbol: str,
    data_path: Path,
    daily: pd.DataFrame,
    metrics_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    sorted_metrics = metrics_df.sort_values("final_equity", ascending=False).copy()
    lines = [
        f"# {symbol} Turtle vs DCA Backtest",
        "",
        "This is a research backtest, not a promise of future performance.",
        "",
        "## Data",
        "",
        f"- Source: `{data_path}`",
        f"- Daily bars used: `{len(daily)}`",
        f"- Date range: `{daily.index[0].date()}` to `{daily.index[-1].date()}`",
        f"- Partial days dropped when fewer than `{args.min_bars_per_day}` 5-minute bars were present",
        "",
        "## Assumptions",
        "",
        f"- Initial capital: `${args.initial_capital:,.2f}`",
        f"- Max leverage: `{args.max_leverage:g}x`",
        f"- Borrow rate on leverage above 1x: `{args.borrow_rate * 100:.2f}%` annual",
        f"- Trading cost: `{args.slippage_bps:g}` bps slippage + `{args.commission_bps:g}` bps commission per exposure change",
        f"- Turtle entry/exit: `{args.entry_period}`-day breakout / `{args.exit_period}`-day exit channel",
        f"- Trend filter: close above/below `{args.sma_period}`-day SMA",
        f"- RVOL confirmation: volume >= `{args.min_rel_volume:g}x` prior `{args.volume_period}`-day average and close location threshold `{args.close_location_threshold:g}`",
        f"- Tactical MA DCA: cheap when `{args.tactical_fast_ma}`-day SMA / `{args.tactical_slow_ma}`-day SMA < `{args.tactical_ma_threshold:g}`",
        f"- Shorting enabled: `{bool(args.allow_shorts)}`",
        "",
        "## Results",
        "",
        "| Strategy | Final | CAGR | Max DD | Sharpe | vs DCA | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in sorted_metrics.iterrows():
        vs_dca = row.get("final_vs_dca_pct", np.nan)
        lines.append(
            "| {strategy} | ${final:,.0f} | {cagr:.1f}% | {dd:.1f}% | {sharpe:.2f} | {vs_dca} | {trades} |".format(
                strategy=row["strategy"],
                final=row["final_equity"],
                cagr=row["cagr_pct"],
                dd=row["max_drawdown_pct"],
                sharpe=row["sharpe"],
                vs_dca="" if pd.isna(vs_dca) else f"{vs_dca:.1f}%",
                trades=int(row["trades"]) if not pd.isna(row["trades"]) else "",
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: numeric strategy metrics",
            "- `equity_curves.csv`: daily equity for each benchmark and strategy",
            "- `leverage.csv`: daily signed leverage for turtle strategies",
            "- `trades.csv`: entry/exit log for turtle strategies",
            "- `equity_curves.png`: visual summary",
            "- `summary.json`: config and data metadata",
        ]
    )
    if args.run_grid:
        lines.append("- `grid_results.csv`: optional fixed-leverage parameter scan")
        lines.append("- `tactical_ma_grid_results.csv`: optional tactical MA DCA parameter scan")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest QQQ turtle trading variants against QQQ monthly DCA."
    )
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default=None, help="Optional YYYY-MM-DD start date")
    parser.add_argument("--end", default=None, help="Optional YYYY-MM-DD end date")
    parser.add_argument("--min-bars-per-day", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--borrow-rate", type=float, default=0.055)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--entry-period", type=int, default=55)
    parser.add_argument("--exit-period", type=int, default=20)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--sma-period", type=int, default=200)
    parser.add_argument("--volume-period", type=int, default=40)
    parser.add_argument("--tactical-fast-ma", type=int, default=50)
    parser.add_argument("--tactical-slow-ma", type=int, default=200)
    parser.add_argument("--tactical-ma-threshold", type=float, default=1.0)
    parser.add_argument("--tactical-grid-fast-mas", type=int, nargs="+", default=[50, 60, 70])
    parser.add_argument("--tactical-grid-slow-mas", type=int, nargs="+", default=[200, 210, 220])
    parser.add_argument("--risk-per-unit", type=float, default=0.02)
    parser.add_argument("--boosted-risk-per-unit", type=float, default=0.03)
    parser.add_argument("--min-rel-volume", type=float, default=1.25)
    parser.add_argument("--close-location-threshold", type=float, default=0.65)
    parser.add_argument("--atr-stop-mult", type=float, default=2.0)
    parser.add_argument("--max-units", type=int, default=3)
    parser.add_argument("--add-atr-interval", type=float, default=0.5)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--run-grid", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--grid-entry-periods",
        type=int,
        nargs="+",
        default=[20, 40, 55, 65],
    )
    parser.add_argument(
        "--grid-exit-periods",
        type=int,
        nargs="+",
        default=[10, 20, 25],
    )
    parser.add_argument(
        "--grid-leverages",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
    )
    parser.add_argument(
        "--grid-min-rel-volumes",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 1.5],
    )
    args = parser.parse_args()

    if args.max_leverage <= 0 or args.max_leverage > 3.0:
        raise ValueError("--max-leverage must be > 0 and <= 3.0 for this research setup")
    if args.exit_period >= args.entry_period:
        raise ValueError("--exit-period should be smaller than --entry-period")
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    daily, data_path = load_daily_ohlcv(
        symbol=args.symbol,
        data_dir=args.data_dir,
        min_bars_per_day=args.min_bars_per_day,
        start=args.start,
        end=args.end,
    )

    symbol = args.symbol.upper()
    close = daily["close"]
    dca = dca_monthly(close, args.initial_capital, f"DCA Monthly {symbol} 1x")
    dca_lev, dca_lev_leverage = leveraged_dca_monthly(
        close,
        args.initial_capital,
        leverage=args.max_leverage,
        borrow_rate=args.borrow_rate,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        label=f"DCA Monthly {symbol} {args.max_leverage:g}x",
    )
    tactical_sleeve, tactical_sleeve_leverage = ma_tactical_dca_sleeve(
        close,
        args.initial_capital,
        leverage=args.max_leverage,
        borrow_rate=args.borrow_rate,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        fast_ma=args.tactical_fast_ma,
        slow_ma=args.tactical_slow_ma,
        threshold=args.tactical_ma_threshold,
        label=f"MA{args.tactical_fast_ma}/{args.tactical_slow_ma} Tactical Sleeve DCA {args.max_leverage:g}x",
    )
    tactical_portfolio, tactical_portfolio_leverage = ma_tactical_dca_portfolio(
        close,
        args.initial_capital,
        leverage=args.max_leverage,
        borrow_rate=args.borrow_rate,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        fast_ma=args.tactical_fast_ma,
        slow_ma=args.tactical_slow_ma,
        threshold=args.tactical_ma_threshold,
        label=f"MA{args.tactical_fast_ma}/{args.tactical_slow_ma} Tactical Portfolio DCA {args.max_leverage:g}x",
    )
    buy_hold = buy_and_hold(close, args.initial_capital, f"Buy & Hold {symbol} 1x")

    equity_curves = [buy_hold, dca, dca_lev, tactical_sleeve, tactical_portfolio]
    leverage_curves: list[pd.Series] = [
        dca_lev_leverage,
        tactical_sleeve_leverage,
        tactical_portfolio_leverage,
    ]
    all_trades: list[dict[str, Any]] = []
    metrics_rows = [
        compute_metrics(buy_hold, buy_hold.name, dca_equity=dca),
        compute_metrics(dca, dca.name, dca_equity=dca),
        compute_metrics(dca_lev, dca_lev.name, leverage=dca_lev_leverage, dca_equity=dca),
        compute_metrics(
            tactical_sleeve,
            tactical_sleeve.name,
            leverage=tactical_sleeve_leverage,
            dca_equity=dca,
        ),
        compute_metrics(
            tactical_portfolio,
            tactical_portfolio.name,
            leverage=tactical_portfolio_leverage,
            dca_equity=dca,
        ),
    ]

    configs = build_default_configs(args)
    for cfg in configs:
        result = run_turtle_strategy(daily, cfg)
        equity_curves.append(result.equity)
        leverage_curves.append(result.leverage.rename(cfg.name))
        all_trades.extend(result.trades)
        metrics_rows.append(
            compute_metrics(
                result.equity,
                cfg.name,
                leverage=result.leverage,
                trades=result.trades,
                dca_equity=dca,
            )
        )

    equity_df = pd.concat(equity_curves, axis=1)
    leverage_df = pd.concat(leverage_curves, axis=1) if leverage_curves else pd.DataFrame(index=daily.index)
    metrics_df = pd.DataFrame(metrics_rows)
    trades_df = pd.DataFrame(all_trades)

    equity_df.to_csv(out_dir / "equity_curves.csv", index_label="date")
    leverage_df.to_csv(out_dir / "leverage.csv", index_label="date")
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    if args.run_grid:
        grid_df = run_grid(daily, args, dca=dca)
        grid_df.to_csv(out_dir / "grid_results.csv", index=False)
        tactical_grid_df = run_tactical_ma_grid(close, args, dca=dca)
        tactical_grid_df.to_csv(out_dir / "tactical_ma_grid_results.csv", index=False)

    if not args.no_plot:
        plot_results(
            equity_df,
            leverage_df,
            metrics_df,
            out_dir / "equity_curves.png",
            title=(
                f"{symbol} Turtle vs DCA "
                f"({daily.index[0].date()} to {daily.index[-1].date()}, max {args.max_leverage:g}x)"
            ),
        )

    summary = {
        "symbol": symbol,
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "date_start": daily.index[0].date().isoformat(),
        "date_end": daily.index[-1].date().isoformat(),
        "daily_bars": len(daily),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "turtle_configs": [asdict(cfg) for cfg in configs],
        "best_by_final_equity": metrics_df.sort_values("final_equity", ascending=False)
        .iloc[0]
        .to_dict(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_report(out_dir, symbol, data_path, daily, metrics_df, args)

    display_cols = [
        "strategy",
        "final_equity",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe",
        "final_vs_dca_pct",
        "trades",
    ]
    print(metrics_df[display_cols].sort_values("final_equity", ascending=False).to_string(index=False))
    print(f"\nSaved report files under: {out_dir}")


if __name__ == "__main__":
    main()
