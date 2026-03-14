#!/usr/bin/env python3
"""
BTC/ETH Ratio Mean Reversion — Python Backtest
==============================================

This version is a market-neutral BTC/ETH ratio strategy for 5m bars.
The traded signal is the log-ratio:

    ratio = log(BTC / ETH)

The ratio is faded in both directions:
  - Long ratio : long BTC, short ETH
  - Short ratio: short BTC, long ETH

Compared with the rolling-regression spread version, this simpler ratio model
behaved materially better on the local Binance sample. It is still strict:
  - very deep Z-score entry
  - reversal confirmation
  - OU half-life regime filter
  - short holding period
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
    "initial_capital": 10_000.0,
    "commission_pct":  0.05 / 100,
    "interval":        "5m",
    "lookback_years":  3.0,
    "warmup_days":     60,

    "btc_ticker": "BTC-USD",
    "eth_ticker": "ETH-USD",

    # Ratio window
    "ratio_len": 4032,   # 14 days of 5m bars

    # Entry / exit
    "entry_z":  3.5,
    "exit_z":   1.0,
    "stop_z":   5.5,
    "max_bars": 24,

    # Sizing
    "risk_pct": 0.75,
    "leverage": 4.0,

    # OU-style regime filter on the ratio itself
    "use_ou":            True,
    "ou_len":            288,
    "ou_half_life_min":  3.0,
    "ou_half_life_max": 72.0,
    "hl_exit_mult":     3.0,
    "hl_min_bars":      12,

    # Entry confirmation
    "use_reversal":   True,
    "min_reversal_z": 0.05,

    # Directions
    "allow_long_ratio":  True,
    "allow_short_ratio": True,
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
    """Rolling OU-style half-life estimate on the ratio series."""
    def __init__(self, window: int) -> None:
        self.window = RollingWindow(window)
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
            self.half_life = math.inf
            return self.half_life

        cov_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        lam = cov_xy / var_x
        self.half_life = math.inf if lam >= 0 else -math.log(2.0) / lam
        return self.half_life


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    direction: str
    gross_notional: float
    btc_qty: float
    eth_qty: float
    btc_entry: float
    eth_entry: float
    entry_ratio: float
    stop_ratio: float
    target_ratio: float
    entry_fee: float
    half_life: float | None
    max_hold_bars: int
    bars_held: int = 0
    entry_ts: datetime | None = None


@dataclass
class Trade:
    direction: str
    entry_ts: datetime
    exit_ts: datetime
    gross_notional: float
    entry_ratio: float
    exit_ratio: float
    entry_btc: float
    exit_btc: float
    entry_eth: float
    exit_eth: float
    pnl: float
    exit_reason: str
    half_life: float | None


# =============================================================================
# CORE LOGIC
# =============================================================================

def load_pair_bars(btc_ticker: str, eth_ticker: str, p: dict) -> tuple[list[dict[str, Any]], str, str]:
    btc_bars, btc_symbol = load_local_binance_klines(
        ticker=btc_ticker,
        interval=p["interval"],
        lookback_years=p["lookback_years"],
        warmup_days=p["warmup_days"],
    )
    eth_bars, eth_symbol = load_local_binance_klines(
        ticker=eth_ticker,
        interval=p["interval"],
        lookback_years=p["lookback_years"],
        warmup_days=p["warmup_days"],
    )

    eth_by_ts = {bar["timestamp"]: bar for bar in eth_bars}
    aligned: list[dict[str, Any]] = []
    for btc_bar in btc_bars:
        eth_bar = eth_by_ts.get(btc_bar["timestamp"])
        if eth_bar is None:
            continue
        aligned.append(
            {
                "timestamp": btc_bar["timestamp"],
                "btc_close": float(btc_bar["close"]),
                "eth_close": float(eth_bar["close"]),
            }
        )

    if not aligned:
        raise ValueError(f"No overlapping bars for {btc_symbol}/{eth_symbol}")

    return aligned, btc_symbol, eth_symbol


def derive_hold_limit(entry_half_life: float | None, p: dict) -> int:
    base_limit = p["max_bars"]
    if not p["use_ou"] or entry_half_life is None or not math.isfinite(entry_half_life):
        return base_limit

    hl_limit = max(int(round(entry_half_life * p["hl_exit_mult"])), p["hl_min_bars"])
    if base_limit > 0:
        return min(base_limit, hl_limit)
    return hl_limit


def calc_gross_notional(
    equity: float,
    risk_pct: float,
    entry_ratio: float,
    stop_ratio: float,
    leverage: float,
) -> float:
    """
    Equal-dollar BTC/ETH pair.
    For a small log-ratio move dR, portfolio PnL is approximately:
      gross_notional / 2 * dR
    """
    stop_d = abs(entry_ratio - stop_ratio)
    if stop_d <= 0 or equity <= 0:
        return 0.0

    risk_amt = equity * risk_pct / 100.0
    gross_risk = (2.0 * risk_amt) / stop_d
    gross_lev = equity * leverage
    return max(min(gross_risk, gross_lev), 0.0)


def unrealized_pnl(pos: Position, btc_price: float, eth_price: float) -> float:
    return pos.btc_qty * (btc_price - pos.btc_entry) + pos.eth_qty * (eth_price - pos.eth_entry)


def run_backtest(p: dict) -> dict[str, Any]:
    bars, btc_symbol, eth_symbol = load_pair_bars(p["btc_ticker"], p["eth_ticker"], p)

    equity = p["initial_capital"]
    pos: Position | None = None
    trades: list[Trade] = []
    equity_curve: list[tuple[datetime, float]] = []

    ratio_win = RollingWindow(p["ratio_len"])
    hl_est = HalfLifeEstimator(p["ou_len"])
    prev_z: float | None = None

    for bar in bars:
        ts = bar["timestamp"]
        btc = bar["btc_close"]
        eth = bar["eth_close"]
        ratio = math.log(btc / eth)
        half_life = hl_est.update(ratio)

        ratio_mean = ratio_win.sma()
        ratio_std = ratio_win.stdev()
        z: float | None = None
        if ratio_mean is not None and ratio_std is not None and ratio_std > 1e-12:
            z = (ratio - ratio_mean) / ratio_std

        exit_reason: str | None = None
        if pos is not None:
            pos.bars_held += 1
            hold_limit = pos.max_hold_bars if pos.max_hold_bars > 0 else p["max_bars"]
            if pos.direction == "long_ratio":
                if ratio <= pos.stop_ratio:
                    exit_reason = "Z-Stop"
                elif ratio >= pos.target_ratio:
                    exit_reason = "Target"
                elif z is not None and z >= -p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif hold_limit > 0 and pos.bars_held >= hold_limit:
                    exit_reason = "Time-Stop"
            else:
                if ratio >= pos.stop_ratio:
                    exit_reason = "Z-Stop"
                elif ratio <= pos.target_ratio:
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
                    gross_notional=pos.gross_notional,
                    entry_ratio=pos.entry_ratio,
                    exit_ratio=ratio,
                    entry_btc=pos.btc_entry,
                    exit_btc=btc,
                    entry_eth=pos.eth_entry,
                    exit_eth=eth,
                    pnl=total_pnl,
                    exit_reason=exit_reason,
                    half_life=pos.half_life,
                )
            )
            pos = None

        hl_ok = (
            not p["use_ou"] or
            half_life is None or
            (
                math.isfinite(half_life) and
                p["ou_half_life_min"] <= half_life <= p["ou_half_life_max"]
            )
        )

        if pos is None and z is not None and prev_z is not None and hl_ok:
            long_reversal = (
                p["allow_long_ratio"] and
                z <= -p["entry_z"] and
                (
                    not p["use_reversal"] or
                    (z > prev_z and (z - prev_z) >= p["min_reversal_z"])
                )
            )
            short_reversal = (
                p["allow_short_ratio"] and
                z >= p["entry_z"] and
                (
                    not p["use_reversal"] or
                    (z < prev_z and (prev_z - z) >= p["min_reversal_z"])
                )
            )

            if long_reversal or short_reversal:
                direction = "long_ratio" if long_reversal else "short_ratio"
                stop_ratio = (
                    ratio_mean - p["stop_z"] * ratio_std
                    if direction == "long_ratio"
                    else ratio_mean + p["stop_z"] * ratio_std
                )
                target_ratio = (
                    ratio_mean - p["exit_z"] * ratio_std
                    if direction == "long_ratio"
                    else ratio_mean + p["exit_z"] * ratio_std
                )

                gross_notional = calc_gross_notional(
                    equity, p["risk_pct"], ratio, stop_ratio, p["leverage"]
                )
                if gross_notional > 0:
                    side = 1.0 if direction == "long_ratio" else -1.0
                    btc_qty = side * (gross_notional / 2.0) / btc
                    eth_qty = -side * (gross_notional / 2.0) / eth
                    entry_fee = gross_notional * p["commission_pct"]
                    equity -= entry_fee
                    hold_limit = derive_hold_limit(half_life, p)
                    pos = Position(
                        direction=direction,
                        gross_notional=gross_notional,
                        btc_qty=btc_qty,
                        eth_qty=eth_qty,
                        btc_entry=btc,
                        eth_entry=eth,
                        entry_ratio=ratio,
                        stop_ratio=stop_ratio,
                        target_ratio=target_ratio,
                        entry_fee=entry_fee,
                        half_life=half_life,
                        max_hold_bars=hold_limit,
                        bars_held=0,
                        entry_ts=ts,
                    )

        mtm = equity
        if pos is not None:
            mtm += unrealized_pnl(pos, btc, eth)
        equity_curve.append((ts, round(mtm, 4)))

        ratio_win.push(ratio)
        if z is not None:
            prev_z = z

    if pos is not None and equity_curve:
        last_ts = bars[-1]["timestamp"]
        last_btc = bars[-1]["btc_close"]
        last_eth = bars[-1]["eth_close"]
        ratio = math.log(last_btc / last_eth)
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
                gross_notional=pos.gross_notional,
                entry_ratio=pos.entry_ratio,
                exit_ratio=ratio,
                entry_btc=pos.btc_entry,
                exit_btc=last_btc,
                entry_eth=pos.eth_entry,
                exit_eth=last_eth,
                pnl=total_pnl,
                exit_reason="End-Of-Data",
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

    long_trades = [t for t in trades if t.direction == "long_ratio"]
    short_trades = [t for t in trades if t.direction == "short_ratio"]

    def dir_wr(items: list[Trade]) -> float:
        return sum(1 for t in items if t.pnl > 0) / len(items) * 100.0 if items else 0.0

    finite_hl = [t.half_life for t in trades if t.half_life is not None and math.isfinite(t.half_life)]
    avg_hl = sum(finite_hl) / len(finite_hl) if finite_hl else 0.0
    avg_notional = sum(t.gross_notional for t in trades) / n if n else 0.0

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
        "long_ratio_n": len(long_trades),
        "short_ratio_n": len(short_trades),
        "long_ratio_wr_pct": round(dir_wr(long_trades), 1),
        "short_ratio_wr_pct": round(dir_wr(short_trades), 1),
        "long_ratio_pnl": round(sum(t.pnl for t in long_trades), 2),
        "short_ratio_pnl": round(sum(t.pnl for t in short_trades), 2),
        "avg_half_life": round(avg_hl, 2),
        "avg_gross_notional": round(avg_notional, 2),
        "exit_reasons": exits,
        "equity_curve": [(str(ts), eq) for ts, eq in equity_curve[::288]],
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_summary(result: dict[str, Any], p: dict) -> None:
    print()
    print("═" * 72)
    print("  BTC/ETH RATIO MEAN REVERSION — BACKTEST RESULTS")
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
    print(f"  Long ratio          {result['long_ratio_n']} trades, WR {result['long_ratio_wr_pct']:.1f}%, PnL ${result['long_ratio_pnl']:+.2f}")
    print(f"  Short ratio         {result['short_ratio_n']} trades, WR {result['short_ratio_wr_pct']:.1f}%, PnL ${result['short_ratio_pnl']:+.2f}")
    print(f"  Avg half-life       {result['avg_half_life']:.2f} bars")
    print(f"  Avg gross notional  ${result['avg_gross_notional']:,.2f}")
    print("  " + "─" * 66)
    print("  Exit breakdown:")
    for reason, count in sorted(result["exit_reasons"].items(), key=lambda item: (-item[1], item[0])):
        pct = count / result["n_trades"] * 100.0 if result["n_trades"] else 0.0
        print(f"    {reason:<16} {count:4d} ({pct:.1f}%)")
    print("  " + "─" * 66)
    print(
        f"  Params: ratio_len={p['ratio_len']} entry={p['entry_z']}"
        f" exit={p['exit_z']} stop={p['stop_z']} max_bars={p['max_bars']}"
    )
    print(
        f"  Risk: risk_pct={p['risk_pct']} leverage={p['leverage']}"
        f" both_dirs={p['allow_long_ratio'] and p['allow_short_ratio']}"
    )
    print(
        f"  Half-life filter: {p['ou_half_life_min']:.0f}..{p['ou_half_life_max']:.0f}"
        f"  reversal={p['use_reversal']}"
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
