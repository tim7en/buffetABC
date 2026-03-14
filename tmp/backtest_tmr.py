#!/usr/bin/env python3
"""
BTC/ETH Spread Mean Reversion — Python Backtest
===============================================

This version replaces the single-asset price fade with a two-leg spread trade:
  BTC price = alpha + beta * ETH price + residual

The residual is treated as the tradable mean-reverting spread.
Entries are taken in both directions:
  - Long spread : long BTC, short beta * ETH
  - Short spread: short BTC, long beta * ETH

The strategy uses:
  - rolling OLS hedge ratio
  - rolling residual Z-score
  - R^2 and OU half-life regime filters
  - reversal confirmation at the extreme
  - spread-based stop/target and pair-aware sizing
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Data loader (use production standalone to avoid Django deps) ───────────────
_ROOT     = Path(__file__).resolve().parent.parent
_PROD_DIR = _ROOT / "production" / "session_turtle_core_x2"
sys.path.insert(0, str(_PROD_DIR))
from binance_data import load_local_binance_klines  # noqa: E402

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

PARAMS: dict[str, Any] = {
    "initial_capital":  10_000.0,
    "commission_pct":   0.05 / 100,
    "interval":         "5m",
    "lookback_years":   3.0,
    "warmup_days":      60,

    "pair_y": "BTC-USD",   # dependent leg in hedge regression
    "pair_x": "ETH-USD",   # hedge leg in regression

    # Rolling hedge ratio / residual model
    "hedge_len":   2016,   # 7 days of 5m bars
    "spread_len":  288,    # 1 day residual Z-score window

    # Entry / exit on residual Z-score
    "entry_z":  2.25,
    "exit_z":   0.50,
    "stop_z":   3.75,
    "max_bars": 96,

    # Sizing
    "risk_pct": 1.00,
    "leverage": 4.0,

    # Regime filters
    "use_r2":  True,
    "min_r2":  0.85,
    "min_beta": 5.0,
    "max_beta": 40.0,

    "use_ou":            True,
    "ou_len":            288,
    "ou_half_life_min":  3.0,
    "ou_half_life_max":  72.0,
    "hl_exit_mult":      3.0,
    "hl_min_bars":       12,

    # Reversal confirmation
    "use_reversal":    True,
    "min_reversal_z":  0.05,

    # Directions
    "allow_long_spread":  True,
    "allow_short_spread": True,
}


# =============================================================================
# HELPERS
# =============================================================================

class RollingWindow:
    """Fixed-size rolling buffer with O(1) SMA and population STDEV."""
    def __init__(self, size: int) -> None:
        self.buf: deque[float] = deque(maxlen=size)
        self.size = size
        self._sum = 0.0
        self._sum_sq = 0.0

    def push(self, val: float) -> None:
        if len(self.buf) == self.size:
            old = self.buf.popleft()
            self._sum -= old
            self._sum_sq -= old * old
        self.buf.append(val)
        self._sum += val
        self._sum_sq += val * val

    def full(self) -> bool:
        return len(self.buf) == self.size

    def sma(self) -> float | None:
        if not self.full():
            return None
        return self._sum / len(self.buf)

    def stdev(self) -> float | None:
        if not self.full():
            return None
        n = len(self.buf)
        mean = self._sum / n
        var = max(self._sum_sq / n - mean * mean, 0.0)
        return math.sqrt(var)

    def values(self) -> list[float]:
        return list(self.buf)


class HalfLifeEstimator:
    """Rolling OU-style half-life estimate on a spread series."""
    def __init__(self, window: int) -> None:
        self.window = RollingWindow(window)
        self.lambda_: float | None = None
        self.half_life: float | None = None

    def update(self, value: float) -> float | None:
        self.window.push(value)
        if not self.window.full():
            return self.half_life

        vals = self.window.values()
        x = vals[:-1]
        y = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        if not x:
            return self.half_life

        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        var_x = sum((xi - mx) ** 2 for xi in x)

        if var_x <= 1e-12:
            self.lambda_ = 0.0
            self.half_life = math.inf
            return self.half_life

        cov_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        self.lambda_ = cov_xy / var_x
        if self.lambda_ >= 0:
            self.half_life = math.inf
        else:
            self.half_life = -math.log(2.0) / self.lambda_
        return self.half_life


class RollingOLS:
    """Rolling OLS with intercept using O(1) sufficient statistics."""
    def __init__(self, size: int) -> None:
        self.size = size
        self.x: deque[float] = deque(maxlen=size)
        self.y: deque[float] = deque(maxlen=size)
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0

    def push(self, x: float, y: float) -> None:
        if len(self.x) == self.size:
            old_x = self.x.popleft()
            old_y = self.y.popleft()
            self.sx -= old_x
            self.sy -= old_y
            self.sxx -= old_x * old_x
            self.syy -= old_y * old_y
            self.sxy -= old_x * old_y

        self.x.append(x)
        self.y.append(y)
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def full(self) -> bool:
        return len(self.x) == self.size

    def coeffs(self) -> tuple[float | None, float | None, float | None]:
        if not self.full():
            return None, None, None

        n = len(self.x)
        var_x = self.sxx - (self.sx * self.sx) / n
        var_y = self.syy - (self.sy * self.sy) / n
        if var_x <= 1e-12 or var_y <= 1e-12:
            return None, None, None

        cov_xy = self.sxy - (self.sx * self.sy) / n
        beta = cov_xy / var_x
        mean_x = self.sx / n
        mean_y = self.sy / n
        alpha = mean_y - beta * mean_x
        r2 = max(min((cov_xy * cov_xy) / (var_x * var_y), 1.0), 0.0)
        return alpha, beta, r2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    direction: str
    pair_qty: float
    btc_qty: float
    eth_qty: float
    beta: float
    alpha: float
    btc_entry: float
    eth_entry: float
    entry_spread: float
    stop_spread: float
    target_spread: float
    entry_fee: float
    r2: float
    half_life: float | None
    max_hold_bars: int
    bars_held: int = 0
    entry_ts: datetime | None = None


@dataclass
class Trade:
    direction: str
    entry_ts: datetime
    exit_ts: datetime
    pair_qty: float
    beta: float
    entry_spread: float
    exit_spread: float
    entry_btc: float
    exit_btc: float
    entry_eth: float
    exit_eth: float
    pnl: float
    exit_reason: str
    r2: float
    half_life: float | None


# =============================================================================
# CORE LOGIC
# =============================================================================

def load_pair_bars(y_ticker: str, x_ticker: str, p: dict) -> tuple[list[dict[str, Any]], str, str]:
    y_bars, y_symbol = load_local_binance_klines(
        ticker=y_ticker,
        interval=p["interval"],
        lookback_years=p["lookback_years"],
        warmup_days=p["warmup_days"],
    )
    x_bars, x_symbol = load_local_binance_klines(
        ticker=x_ticker,
        interval=p["interval"],
        lookback_years=p["lookback_years"],
        warmup_days=p["warmup_days"],
    )

    x_by_ts = {bar["timestamp"]: bar for bar in x_bars}
    aligned: list[dict[str, Any]] = []
    for y_bar in y_bars:
        x_bar = x_by_ts.get(y_bar["timestamp"])
        if x_bar is None:
            continue
        aligned.append(
            {
                "timestamp": y_bar["timestamp"],
                "btc_close": float(y_bar["close"]),
                "eth_close": float(x_bar["close"]),
            }
        )

    if not aligned:
        raise ValueError(f"No overlapping bars for {y_symbol}/{x_symbol}")

    return aligned, y_symbol, x_symbol


def derive_hold_limit(entry_half_life: float | None, p: dict) -> int:
    base_limit = p["max_bars"]
    if not p["use_ou"] or entry_half_life is None or not math.isfinite(entry_half_life):
        return base_limit

    hl_limit = max(int(round(entry_half_life * p["hl_exit_mult"])), p["hl_min_bars"])
    if base_limit > 0:
        return min(base_limit, hl_limit)
    return hl_limit


def calc_pair_qty(
    equity: float,
    risk_pct: float,
    btc_price: float,
    eth_price: float,
    beta: float,
    entry_spread: float,
    stop_spread: float,
    leverage: float,
) -> float:
    stop_d = abs(entry_spread - stop_spread)
    gross_unit = abs(btc_price) + abs(beta) * abs(eth_price)
    if stop_d <= 0 or gross_unit <= 0 or equity <= 0:
        return 0.0

    risk_amt = equity * risk_pct / 100.0
    qty_risk = risk_amt / stop_d
    qty_lev = (equity * leverage) / gross_unit
    return max(min(qty_risk, qty_lev), 0.0)


def unrealized_pnl(pos: Position, btc_price: float, eth_price: float) -> float:
    return pos.btc_qty * (btc_price - pos.btc_entry) + pos.eth_qty * (eth_price - pos.eth_entry)


def run_backtest(p: dict) -> dict[str, Any]:
    bars, btc_symbol, eth_symbol = load_pair_bars(p["pair_y"], p["pair_x"], p)

    equity = p["initial_capital"]
    equity_peak = equity
    pos: Position | None = None
    trades: list[Trade] = []
    equity_curve: list[tuple[datetime, float]] = []

    ols = RollingOLS(p["hedge_len"])
    spread_win = RollingWindow(p["spread_len"])
    hl_est = HalfLifeEstimator(p["ou_len"])
    prev_z: float | None = None

    for bar in bars:
        ts = bar["timestamp"]
        btc = bar["btc_close"]
        eth = bar["eth_close"]

        ols.push(eth, btc)
        alpha, beta, r2 = ols.coeffs()
        if alpha is None or beta is None or r2 is None:
            continue

        if not math.isfinite(alpha) or not math.isfinite(beta) or not math.isfinite(r2):
            continue

        spread = btc - (alpha + beta * eth)
        half_life = hl_est.update(spread)

        z: float | None = None
        spread_mean = spread_win.sma()
        spread_std = spread_win.stdev()
        if spread_mean is not None and spread_std is not None and spread_std > 1e-12:
            z = (spread - spread_mean) / spread_std

        # Exit on the current spread first.
        exit_reason: str | None = None
        if pos is not None:
            pos.bars_held += 1
            hold_limit = pos.max_hold_bars if pos.max_hold_bars > 0 else p["max_bars"]
            if pos.direction == "long_spread":
                if spread <= pos.stop_spread:
                    exit_reason = "Z-Stop"
                elif spread >= pos.target_spread:
                    exit_reason = "Target"
                elif z is not None and z >= -p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif hold_limit > 0 and pos.bars_held >= hold_limit:
                    exit_reason = "Time-Stop"
            else:
                if spread >= pos.stop_spread:
                    exit_reason = "Z-Stop"
                elif spread <= pos.target_spread:
                    exit_reason = "Target"
                elif z is not None and z <= p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif hold_limit > 0 and pos.bars_held >= hold_limit:
                    exit_reason = "Time-Stop"

        if exit_reason is not None and pos is not None:
            gross_pnl = unrealized_pnl(pos, btc, eth)
            exit_notional = abs(pos.btc_qty * btc) + abs(pos.eth_qty * eth)
            exit_fee = exit_notional * p["commission_pct"]
            total_pnl = gross_pnl - pos.entry_fee - exit_fee
            equity += gross_pnl - exit_fee

            trades.append(
                Trade(
                    direction=pos.direction,
                    entry_ts=pos.entry_ts,
                    exit_ts=ts,
                    pair_qty=pos.pair_qty,
                    beta=pos.beta,
                    entry_spread=pos.entry_spread,
                    exit_spread=spread,
                    entry_btc=pos.btc_entry,
                    exit_btc=btc,
                    entry_eth=pos.eth_entry,
                    exit_eth=eth,
                    pnl=total_pnl,
                    exit_reason=exit_reason,
                    r2=pos.r2,
                    half_life=pos.half_life,
                )
            )
            pos = None

        # Entry after exits.
        beta_ok = p["min_beta"] <= beta <= p["max_beta"]
        r2_ok = (not p["use_r2"]) or r2 >= p["min_r2"]
        hl_ok = (
            not p["use_ou"] or
            half_life is None or
            (
                math.isfinite(half_life) and
                p["ou_half_life_min"] <= half_life <= p["ou_half_life_max"]
            )
        )

        if pos is None and z is not None and prev_z is not None and beta_ok and r2_ok and hl_ok:
            long_reversal = (
                p["allow_long_spread"] and
                z <= -p["entry_z"] and
                (
                    not p["use_reversal"] or
                    (z > prev_z and (z - prev_z) >= p["min_reversal_z"])
                )
            )
            short_reversal = (
                p["allow_short_spread"] and
                z >= p["entry_z"] and
                (
                    not p["use_reversal"] or
                    (z < prev_z and (prev_z - z) >= p["min_reversal_z"])
                )
            )

            if long_reversal or short_reversal:
                direction = "long_spread" if long_reversal else "short_spread"
                if direction == "long_spread":
                    stop_spread = spread_mean - p["stop_z"] * spread_std
                    target_spread = spread_mean - p["exit_z"] * spread_std
                    pair_qty = calc_pair_qty(
                        equity, p["risk_pct"], btc, eth, beta, spread, stop_spread, p["leverage"]
                    )
                    btc_qty = pair_qty
                    eth_qty = -pair_qty * beta
                else:
                    stop_spread = spread_mean + p["stop_z"] * spread_std
                    target_spread = spread_mean + p["exit_z"] * spread_std
                    pair_qty = calc_pair_qty(
                        equity, p["risk_pct"], btc, eth, beta, spread, stop_spread, p["leverage"]
                    )
                    btc_qty = -pair_qty
                    eth_qty = pair_qty * beta

                if pair_qty > 0:
                    entry_notional = abs(btc_qty * btc) + abs(eth_qty * eth)
                    entry_fee = entry_notional * p["commission_pct"]
                    equity -= entry_fee
                    hold_limit = derive_hold_limit(half_life, p)
                    pos = Position(
                        direction=direction,
                        pair_qty=pair_qty,
                        btc_qty=btc_qty,
                        eth_qty=eth_qty,
                        beta=beta,
                        alpha=alpha,
                        btc_entry=btc,
                        eth_entry=eth,
                        entry_spread=spread,
                        stop_spread=stop_spread,
                        target_spread=target_spread,
                        entry_fee=entry_fee,
                        r2=r2,
                        half_life=half_life,
                        max_hold_bars=hold_limit,
                        bars_held=0,
                        entry_ts=ts,
                    )

        mtm = equity
        if pos is not None:
            mtm += unrealized_pnl(pos, btc, eth)
        equity_peak = max(equity_peak, mtm)
        equity_curve.append((ts, round(mtm, 4)))

        spread_win.push(spread)
        if z is not None:
            prev_z = z

    if pos is not None and equity_curve:
        last_ts = bars[-1]["timestamp"]
        last_btc = bars[-1]["btc_close"]
        last_eth = bars[-1]["eth_close"]
        spread = last_btc - (pos.alpha + pos.beta * last_eth)
        gross_pnl = unrealized_pnl(pos, last_btc, last_eth)
        exit_notional = abs(pos.btc_qty * last_btc) + abs(pos.eth_qty * last_eth)
        exit_fee = exit_notional * p["commission_pct"]
        total_pnl = gross_pnl - pos.entry_fee - exit_fee
        equity += gross_pnl - exit_fee

        trades.append(
            Trade(
                direction=pos.direction,
                entry_ts=pos.entry_ts,
                exit_ts=last_ts,
                pair_qty=pos.pair_qty,
                beta=pos.beta,
                entry_spread=pos.entry_spread,
                exit_spread=spread,
                entry_btc=pos.btc_entry,
                exit_btc=last_btc,
                entry_eth=pos.eth_entry,
                exit_eth=last_eth,
                pnl=total_pnl,
                exit_reason="End-Of-Data",
                r2=pos.r2,
                half_life=pos.half_life,
            )
        )

    return _compute_metrics(btc_symbol, eth_symbol, equity, equity_curve, trades, p)


# =============================================================================
# METRICS
# =============================================================================

def _compute_metrics(
    btc_symbol: str,
    eth_symbol: str,
    final_equity: float,
    equity_curve: list[tuple[datetime, float]],
    trades: list[Trade],
    p: dict,
) -> dict[str, Any]:
    n = len(trades)
    init = p["initial_capital"]
    total_ret = (final_equity - init) / init * 100.0

    if len(equity_curve) >= 2:
        yrs = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / (365.25 * 86400)
        cagr = ((final_equity / init) ** (1 / max(yrs, 0.01)) - 1) * 100.0
    else:
        cagr = 0.0

    peak = init
    mdd = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak * 100.0)

    rets = []
    for i in range(1, len(equity_curve)):
        prev_eq = equity_curve[i - 1][1]
        cur_eq = equity_curve[i][1]
        if prev_eq > 0:
            rets.append((cur_eq - prev_eq) / prev_eq)
    sharpe = 0.0
    if len(rets) > 20:
        mean_ret = sum(rets) / len(rets)
        stdev_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in rets) / len(rets))
        bars_per_year = (24 * 60) // {"5m": 5, "15m": 15, "30m": 30, "60m": 60}.get(p["interval"], 60) * 365
        if stdev_ret > 0:
            sharpe = mean_ret / stdev_ret * math.sqrt(bars_per_year)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins) / n * 100.0 if n else 0.0
    pf_num = sum(t.pnl for t in wins)
    pf_den = abs(sum(t.pnl for t in losses))
    pf = pf_num / pf_den if pf_den > 0 else (999.0 if pf_num > 0 else 0.0)
    avg_w = pf_num / len(wins) if wins else 0.0
    avg_l = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    exits: dict[str, int] = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    long_trades = [t for t in trades if t.direction == "long_spread"]
    short_trades = [t for t in trades if t.direction == "short_spread"]

    def dir_wr(items: list[Trade]) -> float:
        return sum(1 for t in items if t.pnl > 0) / len(items) * 100.0 if items else 0.0

    avg_r2 = sum(t.r2 for t in trades) / n if n else 0.0
    finite_hl = [t.half_life for t in trades if t.half_life is not None and math.isfinite(t.half_life)]
    avg_hl = sum(finite_hl) / len(finite_hl) if finite_hl else 0.0
    avg_beta = sum(t.beta for t in trades) / n if n else 0.0

    return {
        "pair": f"{btc_symbol}/{eth_symbol}",
        "bars_total": len(equity_curve),
        "initial": init,
        "final_equity": round(final_equity, 2),
        "total_ret_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(mdd, 2),
        "sharpe": round(sharpe, 3),
        "n_trades": n,
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "long_spread_n": len(long_trades),
        "short_spread_n": len(short_trades),
        "long_spread_wr_pct": round(dir_wr(long_trades), 1),
        "short_spread_wr_pct": round(dir_wr(short_trades), 1),
        "long_spread_pnl": round(sum(t.pnl for t in long_trades), 2),
        "short_spread_pnl": round(sum(t.pnl for t in short_trades), 2),
        "avg_beta": round(avg_beta, 3),
        "avg_r2": round(avg_r2, 3),
        "avg_half_life": round(avg_hl, 2),
        "exit_reasons": exits,
        "equity_curve": [(str(ts), eq) for ts, eq in equity_curve[::288]],
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_summary(result: dict[str, Any], p: dict) -> None:
    print()
    print("═" * 72)
    print("  BTC/ETH SPREAD MEAN REVERSION — BACKTEST RESULTS")
    print("═" * 72)
    print(f"  Pair                {result['pair']}")
    print(f"  Final equity        ${result['final_equity']:,.2f}")
    print(f"  Total return        {result['total_ret_pct']:+.2f}%")
    print(f"  CAGR                {result['cagr_pct']:+.2f}%")
    print(f"  Max drawdown        {result['max_dd_pct']:.2f}%")
    print(f"  Sharpe              {result['sharpe']:.3f}")
    print("  " + "─" * 66)
    print(f"  Trades              {result['n_trades']}")
    print(f"  Win rate            {result['win_rate_pct']:.1f}%")
    print(f"  Profit factor       {result['profit_factor']:.2f}x")
    print(f"  Avg win / loss      ${result['avg_win']:+.2f} / ${result['avg_loss']:+.2f}")
    print("  " + "─" * 66)
    print(f"  Long spread         {result['long_spread_n']} trades, WR {result['long_spread_wr_pct']:.1f}%, PnL ${result['long_spread_pnl']:+.2f}")
    print(f"  Short spread        {result['short_spread_n']} trades, WR {result['short_spread_wr_pct']:.1f}%, PnL ${result['short_spread_pnl']:+.2f}")
    print(f"  Avg beta / R²       {result['avg_beta']:.3f} / {result['avg_r2']:.3f}")
    print(f"  Avg half-life       {result['avg_half_life']:.2f} bars")
    print("  " + "─" * 66)
    print("  Exit breakdown:")
    for reason, count in sorted(result["exit_reasons"].items(), key=lambda item: (-item[1], item[0])):
        pct = count / result["n_trades"] * 100.0 if result["n_trades"] else 0.0
        print(f"    {reason:<16} {count:4d} ({pct:.1f}%)")
    print("  " + "─" * 66)
    print(
        f"  Params: hedge_len={p['hedge_len']} spread_len={p['spread_len']}"
        f"  entry={p['entry_z']} exit={p['exit_z']} stop={p['stop_z']}"
    )
    print(
        f"  Risk: risk_pct={p['risk_pct']} leverage={p['leverage']} max_bars={p['max_bars']}"
        f"  both_dirs={p['allow_long_spread'] and p['allow_short_spread']}"
    )
    print(
        f"  Filters: R²>={p['min_r2']:.2f} beta={p['min_beta']}..{p['max_beta']}"
        f"  half_life={p['ou_half_life_min']:.0f}..{p['ou_half_life_max']:.0f}"
    )
    print("═" * 72)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_backtest(PARAMS)
    print_summary(result, PARAMS)

    out_path = Path(__file__).parent / "backtest_tmr_results.json"
    payload = {
        "params": {k: v for k, v in PARAMS.items()},
        "results": [{k: v for k, v in result.items() if k != "equity_curve"}],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Results saved → {out_path.name}\n")
