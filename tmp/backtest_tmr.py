#!/usr/bin/env python3
"""
Tiered Mean Reversion — Python Backtest
========================================
Strategy port of tiered_mean_reversion_v2.pine with additional features:
  - Previous session H/L levels as off-hours anomaly boost
  - Weekend overextension detection
  - Leverage-capped position sizing (up to x20)

Risk formula:
  final_risk = base_tier_risk × session_mult × macro_mult × vol_mult
              × pdhl_boost × weekend_boost

Assets tested: BTCUSDT, ETHUSDT, SOLUSDT  (1h bars, local Binance cache)
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
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
    # Capital & costs
    "initial_capital":  10_000.0,
    "commission_pct":   0.05 / 100,   # 0.05 % taker fee per side
    "interval":         "5m",
    "lookback_years":   3.0,
    "warmup_days":      60,

    # Z-score / Bollinger  (scaled to 5m resolution)
    # 144 bars × 5m = 12 h lookback — stable intraday mean.
    # Z-score / Bollinger  (scaled to 5m resolution)
    # 144 bars × 5m = 12 h lookback — stable intraday mean.
    "bb_len":   144,
    "t1_z":     2.0,   # T1 only — t2/t3 set high to disable pyramiding
    "t2_z":     99.0,  # effectively disabled
    "t3_z":     99.0,  # effectively disabled
    "stop_z":   3.5,   # R:R = (2.0-0.3)/(3.5-2.0) = 1.7/1.5 ≈ 1.13; breakeven WR ~47%
    "exit_z":   0.3,   # exit at 0.3σ from mean

    # Base risk per tier (T2/T3 disabled, only t1_risk matters)
    "t1_risk":  2.0,
    "t2_risk":  1.0,
    "t3_risk":  0.5,

    # Leverage: max position value = equity × leverage
    "leverage": 5.0,

    # Session risk multipliers (UTC clock-time sessions)
    "use_sessions":  True,
    "skip_offhours": True,
    "sess_ovl_ln":   1.25,   # London / NY overlap  (~13:30–16:00 UTC)
    "sess_al":       0.90,   # Asia / London overlap (~07:00–08:30 UTC)
    "sess_london":   1.00,
    "sess_ny":       1.00,
    "sess_tokyo":    0.75,
    "sess_offhours": 0.50,

    # US macro calendar guard (approximate release schedule)
    "use_macro":       True,
    "macro_post_mult": 1.25,   # post-release reversion boost (2 h window)

    # Volume risk adjustment  (vol_len scaled to 5m: 144 = 12 h average)
    "use_vol":       True,
    "vol_len":       144,
    "vol_th":        1.20,
    "close_loc_th":  0.65,
    "vol_boost":     1.20,
    "vol_penalty":   0.70,

    # HTF trend bias — EMA length in 5m bars
    # 600 bars × 5m = 50 h ≈ 2-day trend proxy
    "use_htf":     True,
    "htf_ema_len": 600,

    # Max bars in trade  (288 × 5m = 24 h)
    "max_bars": 288,

    # Drawdown governor
    "use_dd":       False,
    "dd_halt_pct":  15.0,

    # Previous Day High/Low boost (off-hours anomaly)
    # If price is beyond PDL (for longs) or PDH (for shorts) during off-hours,
    # this boosts risk — the market is more likely to snap back to those levels.
    "use_pdhl":    True,
    "pdhl_boost":  1.15,

    # Weekend overextension boost
    # Crypto weekend moves in thin liquidity often revert on Monday open.
    "use_weekend": True,
    "weekend_boost": 1.12,

    # Directions
    "allow_long":  True,
    "allow_short": True,
}

# =============================================================================
# INDICATOR HELPERS
# =============================================================================

class RollingWindow:
    """Fixed-size rolling buffer with SMA and population-STDEV."""
    def __init__(self, size: int) -> None:
        self.buf: deque[float] = deque(maxlen=size)
        self.size = size

    def push(self, val: float) -> None:
        self.buf.append(val)

    def full(self) -> bool:
        return len(self.buf) >= self.size

    def sma(self) -> float | None:
        if not self.full():
            return None
        return sum(self.buf) / len(self.buf)

    def stdev(self) -> float | None:
        if not self.full():
            return None
        n    = len(self.buf)
        mean = sum(self.buf) / n
        var  = sum((x - mean) ** 2 for x in self.buf) / n
        return math.sqrt(var)


class EMA:
    """Exponential moving average (Wilder-style initialised with first value)."""
    def __init__(self, period: int) -> None:
        self.k     = 2.0 / (period + 1)
        self.value: float | None = None

    def update(self, price: float) -> float | None:
        if self.value is None:
            self.value = price
        else:
            self.value = self.value * (1 - self.k) + price * self.k
        return self.value


# =============================================================================
# SESSION DETECTION  (UTC naive datetimes)
# =============================================================================

def session_info(ts: datetime) -> dict[str, Any]:
    """Classify a UTC bar timestamp into trading sessions."""
    t    = ts.time()
    wday = ts.weekday()          # 0 = Mon … 6 = Sun

    # Approximate UTC session windows (simplified, no DST)
    in_tokyo  = dtime(0,  0) <= t < dtime(8, 30)   # 09:00–15:30 JST + overlap buffer
    in_london = dtime(7,  0) <= t < dtime(16, 0)   # 08:00–17:00 BST approx
    in_ny     = dtime(13, 30) <= t < dtime(21, 0)  # 09:30–17:00 ET  approx

    overlap_ln = in_london and in_ny    # ~13:30–16:00 UTC — most liquid
    overlap_al = in_tokyo  and in_london # ~07:00–08:30 UTC — Asia handoff

    in_any   = in_tokyo or in_london or in_ny
    is_wknd  = wday >= 5

    if   overlap_ln:   label = "OVL_LN"
    elif overlap_al:   label = "OVL_AL"
    elif in_london:    label = "LONDON"
    elif in_ny:        label = "NY"
    elif in_tokyo:     label = "TOKYO"
    else:              label = "OFF"

    return {
        "label":      label,
        "overlap_ln": overlap_ln,
        "overlap_al": overlap_al,
        "in_london":  in_london,
        "in_ny":      in_ny,
        "in_tokyo":   in_tokyo,
        "in_any":     in_any,
        "is_weekend": is_wknd,
    }


# =============================================================================
# US MACRO CALENDAR APPROXIMATION
# =============================================================================

def macro_status(ts: datetime, p: dict) -> dict[str, Any]:
    """
    Approximate US macro release schedule.
      NFP    : 1st Friday  of month      (typically 08:30 ET = 13:30 UTC)
      CPI/PPI: 2nd–3rd Tue/Wed           (same time)
      GDP/PCE: last Thursday             (08:30 ET)
    """
    if not p["use_macro"]:
        return {"is_macro": False, "label": "NORMAL", "mult": 1.0, "halted": False}

    wday = ts.weekday()   # Python: Mon=0 … Fri=4
    dom  = ts.day
    t    = ts.time()

    raw_nfp = (wday == 4) and (dom <= 7)
    raw_cpi = (wday in (1, 2)) and (8 <= dom <= 21)
    raw_gdp = (wday == 3) and (dom >= 22)

    if not (raw_nfp or raw_cpi or raw_gdp):
        return {"is_macro": False, "label": "NORMAL", "mult": 1.0, "halted": False}

    event   = "NFP" if raw_nfp else ("CPI" if raw_cpi else "GDP")
    pre_win = dtime(13,  0) <= t < dtime(13, 30)   # 30-min pre-release
    rel_win = dtime(13, 30) <= t < dtime(14, 30)   # 60-min release window
    pst_win = dtime(14, 30) <= t < dtime(16, 30)   # 2-h post-release boost

    if pre_win or rel_win:
        return {"is_macro": True, "label": f"{event}-HALT", "mult": 0.0, "halted": True}
    if pst_win:
        return {"is_macro": True, "label": f"{event}-POST", "mult": p["macro_post_mult"], "halted": False}
    return {"is_macro": True, "label": event, "mult": 1.0, "halted": False}


# =============================================================================
# POSITION & TRADE DATA CLASSES
# =============================================================================

@dataclass
class Position:
    direction:  str                # 'long' or 'short'
    tiers:      int                # 1, 2, or 3
    total_qty:  float
    avg_entry:  float
    stop_price: float
    bars_held:  int = 0
    entry_ts:   datetime | None = None

    def add_tier(self, price: float, qty: float) -> None:
        old_cost      = self.avg_entry * self.total_qty
        self.total_qty += qty
        self.avg_entry = (old_cost + price * qty) / self.total_qty
        self.tiers    += 1


@dataclass
class Trade:
    direction:    str
    entry_ts:     datetime
    exit_ts:      datetime
    entry_price:  float
    exit_price:   float
    qty:          float
    tiers:        int
    pnl:          float
    exit_reason:  str
    session:      str


# =============================================================================
# RISK SIZING
# =============================================================================

def calc_qty(equity: float, base_risk: float, entry: float,
             stop: float, mult: float, leverage: float) -> float:
    """
    qty = min(risk-based size, leverage cap)
    risk-based: equity × adj_risk% / stop_distance
    leverage cap: equity × leverage / entry_price
    """
    adj_risk = base_risk * mult / 100.0
    stop_d   = abs(entry - stop)
    if stop_d <= 0 or entry <= 0 or equity <= 0:
        return 0.0
    qty_risk = (equity * adj_risk) / stop_d
    qty_lev  = (equity * leverage) / entry
    return max(min(qty_risk, qty_lev), 0.0)


# =============================================================================
# CORE BACKTEST
# =============================================================================

def run_backtest(ticker: str, p: dict) -> dict:
    bars, symbol = load_local_binance_klines(
        ticker=ticker,
        interval=p["interval"],
        lookback_years=p["lookback_years"],
        warmup_days=p["warmup_days"],
    )
    if not bars:
        return {"error": f"No data for {ticker}"}

    equity      = p["initial_capital"]
    equity_peak = equity
    max_dd      = 0.0

    pos: Position | None = None
    trades: list[Trade]  = []
    equity_curve: list[tuple[datetime, float]] = []

    bb_win  = RollingWindow(p["bb_len"])
    vol_win = RollingWindow(p["vol_len"])
    htf_ema = EMA(p["htf_ema_len"])

    prev_z: float = 0.0

    # Previous day H/L tracking
    pdh: float = float("nan")
    pdl: float = float("nan")
    day_h: float = float("-inf")
    day_l: float = float("inf")
    cur_date = None

    # 5m bars: 24 h × 12 bars/h = 288 per day  (gracefully handles other intervals too)
    _mins_per_bar = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}.get(p["interval"], 60)
    _bars_per_day = (24 * 60) // _mins_per_bar
    warmup_bars   = p["warmup_days"] * _bars_per_day

    for idx, bar in enumerate(bars):
        ts = bar["timestamp"]
        o, h, l, c, v = (
            bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]
        )

        # ── Previous day H/L ─────────────────────────────────────────────────
        bar_date = ts.date()
        if cur_date is None:
            cur_date = bar_date
        if bar_date != cur_date:
            pdh, pdl  = day_h, day_l
            day_h, day_l = h, l
            cur_date  = bar_date
        else:
            day_h = max(day_h, h)
            day_l = min(day_l, l)

        # ── Update indicators ─────────────────────────────────────────────────
        bb_win.push(c)
        vol_win.push(v)
        htf_ema.update(c)

        # Update prev_z even during warmup so it's valid at warmup boundary
        if bb_win.full():
            bm = bb_win.sma()
            bs = bb_win.stdev()
            _z = (c - bm) / bs if (bs and bs > 0) else 0.0
        else:
            _z = 0.0

        if idx < warmup_bars:
            prev_z = _z
            continue

        if not bb_win.full():
            prev_z = _z
            continue

        bb_mean = bb_win.sma()
        bb_std  = bb_win.stdev()
        if bb_std is None or bb_std <= 0:
            prev_z = _z
            continue

        z       = _z
        lvl_stl = bb_mean - p["stop_z"] * bb_std   # long stop
        lvl_stu = bb_mean + p["stop_z"] * bb_std   # short stop

        # ── Session ───────────────────────────────────────────────────────────
        sess = session_info(ts)
        if p["use_sessions"]:
            if   sess["overlap_ln"]: sm = p["sess_ovl_ln"]
            elif sess["overlap_al"]: sm = p["sess_al"]
            elif sess["in_london"]:  sm = p["sess_london"]
            elif sess["in_ny"]:      sm = p["sess_ny"]
            elif sess["in_tokyo"]:   sm = p["sess_tokyo"]
            else:                    sm = p["sess_offhours"]
        else:
            sm = 1.0
        ok_sess = not p["skip_offhours"] or sess["in_any"]

        # ── Macro ─────────────────────────────────────────────────────────────
        macro  = macro_status(ts, p)
        mm     = macro["mult"]
        halted = macro["halted"]

        # ── Volume ────────────────────────────────────────────────────────────
        vsma      = vol_win.sma() or 1.0
        rel_vol   = v / vsma if vsma > 0 else 1.0
        bar_rng   = max(h - l, 1e-12)
        close_loc = (c - l) / bar_rng

        vol_high = rel_vol >= p["vol_th"]
        vcl = vol_high and close_loc >= p["close_loc_th"]            # long confirm
        vcs = vol_high and close_loc <= (1.0 - p["close_loc_th"])    # short confirm

        if p["use_vol"]:
            vml = p["vol_boost"] if vcl else (1.0 if vol_high else p["vol_penalty"])
            vms = p["vol_boost"] if vcs else (1.0 if vol_high else p["vol_penalty"])
        else:
            vml = vms = 1.0

        # ── HTF trend bias ────────────────────────────────────────────────────
        htf_val  = htf_ema.value
        ok_long  = p["allow_long"]  and (not p["use_htf"] or htf_val is None or c > htf_val)
        ok_short = p["allow_short"] and (not p["use_htf"] or htf_val is None or c < htf_val)

        # ── Previous day H/L boost (off-hours anomaly) ────────────────────────
        # Price trading below PDL or above PDH in thin off-hours → strong snap-back signal.
        pdh_ok = not math.isnan(pdh)
        pdl_ok = not math.isnan(pdl)
        if p["use_pdhl"] and not sess["in_any"]:
            pb_l = p["pdhl_boost"] if (pdl_ok and c < pdl) else 1.0
            pb_s = p["pdhl_boost"] if (pdh_ok and c > pdh) else 1.0
        else:
            pb_l = pb_s = 1.0

        # ── Weekend boost ─────────────────────────────────────────────────────
        # Weekend crypto moves in thin liquidity often fully retrace by Monday.
        wb = p["weekend_boost"] if (p["use_weekend"] and sess["is_weekend"] and abs(z) >= p["t2_z"]) else 1.0

        # ── Drawdown governor ─────────────────────────────────────────────────
        dd_now  = (equity_peak - equity) / equity_peak * 100.0 if equity_peak > 0 else 0.0
        dd_halt = p["use_dd"] and dd_now >= p["dd_halt_pct"]

        ok_entry = ok_sess and not halted and not dd_halt

        # ── Composite final risk multipliers ──────────────────────────────────
        fm_l = sm * mm * vml * pb_l * wb
        fm_s = sm * mm * vms * pb_s * wb

        # ── Update bars-held & dynamic stop ──────────────────────────────────
        if pos is not None:
            pos.bars_held += 1
            pos.stop_price = lvl_stl if pos.direction == "long" else lvl_stu

        # ── EXIT LOGIC ────────────────────────────────────────────────────────
        exit_reason: str | None = None
        if pos is not None:
            if pos.direction == "long":
                if z <= -p["stop_z"]:
                    exit_reason = "Z-Stop"
                elif prev_z < -p["exit_z"] and z >= -p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif p["max_bars"] > 0 and pos.bars_held >= p["max_bars"]:
                    exit_reason = "Time-Stop"
            else:  # short
                if z >= p["stop_z"]:
                    exit_reason = "Z-Stop"
                elif prev_z > p["exit_z"] and z <= p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif p["max_bars"] > 0 and pos.bars_held >= p["max_bars"]:
                    exit_reason = "Time-Stop"

        if exit_reason and pos is not None:
            sign      = 1 if pos.direction == "long" else -1
            gross_pnl = pos.total_qty * (c - pos.avg_entry) * sign
            fees      = (pos.avg_entry + c) * pos.total_qty * p["commission_pct"]
            net_pnl   = gross_pnl - fees
            equity   += net_pnl

            trades.append(Trade(
                direction=pos.direction,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                entry_price=pos.avg_entry,
                exit_price=c,
                qty=pos.total_qty,
                tiers=pos.tiers,
                pnl=net_pnl,
                exit_reason=exit_reason,
                session=sess["label"],
            ))
            pos = None

        # ── ENTRY LOGIC ───────────────────────────────────────────────────────
        if ok_entry:
            if pos is None:
                # T1 Long: Z crosses below -t1_z
                if ok_long and prev_z >= -p["t1_z"] and z < -p["t1_z"]:
                    qty = calc_qty(equity, p["t1_risk"], c, lvl_stl, fm_l, p["leverage"])
                    if qty > 0:
                        pos    = Position("long", 1, qty, c, lvl_stl, 0, ts)
                # T1 Short: Z crosses above +t1_z
                elif ok_short and prev_z <= p["t1_z"] and z > p["t1_z"]:
                    qty = calc_qty(equity, p["t1_risk"], c, lvl_stu, fm_s, p["leverage"])
                    if qty > 0:
                        pos    = Position("short", 1, qty, c, lvl_stu, 0, ts)

            elif pos.direction == "long" and pos.tiers == 1:
                if ok_long and prev_z >= -p["t2_z"] and z < -p["t2_z"]:
                    qty = calc_qty(equity, p["t2_risk"], c, lvl_stl, fm_l, p["leverage"])
                    if qty > 0:
                        pos.add_tier(c, qty)

            elif pos.direction == "long" and pos.tiers == 2:
                if ok_long and prev_z >= -p["t3_z"] and z < -p["t3_z"]:
                    qty = calc_qty(equity, p["t3_risk"], c, lvl_stl, fm_l, p["leverage"])
                    if qty > 0:
                        pos.add_tier(c, qty)

            elif pos.direction == "short" and pos.tiers == 1:
                if ok_short and prev_z <= p["t2_z"] and z > p["t2_z"]:
                    qty = calc_qty(equity, p["t2_risk"], c, lvl_stu, fm_s, p["leverage"])
                    if qty > 0:
                        pos.add_tier(c, qty)

            elif pos.direction == "short" and pos.tiers == 2:
                if ok_short and prev_z <= p["t3_z"] and z > p["t3_z"]:
                    qty = calc_qty(equity, p["t3_risk"], c, lvl_stu, fm_s, p["leverage"])
                    if qty > 0:
                        pos.add_tier(c, qty)

        # ── Mark-to-market equity curve ───────────────────────────────────────
        mtm = equity
        if pos is not None:
            sign = 1 if pos.direction == "long" else -1
            mtm  = equity + pos.total_qty * (c - pos.avg_entry) * sign

        equity_peak = max(equity_peak, mtm)
        dd_bar      = (equity_peak - mtm) / equity_peak * 100.0 if equity_peak > 0 else 0.0
        max_dd      = max(max_dd, dd_bar)
        equity_curve.append((ts, round(mtm, 4)))

        prev_z = z

    # Close any open position at last bar close
    if pos is not None and equity_curve:
        last_ts, last_c = equity_curve[-1][0], bars[-1]["close"]
        sign      = 1 if pos.direction == "long" else -1
        gross_pnl = pos.total_qty * (last_c - pos.avg_entry) * sign
        fees      = (pos.avg_entry + last_c) * pos.total_qty * p["commission_pct"]
        equity   += gross_pnl - fees
        trades.append(Trade(
            direction=pos.direction, entry_ts=pos.entry_ts, exit_ts=last_ts,
            entry_price=pos.avg_entry, exit_price=last_c, qty=pos.total_qty,
            tiers=pos.tiers, pnl=gross_pnl - fees, exit_reason="End-Of-Data",
            session="END",
        ))

    return _compute_metrics(ticker, symbol, equity, equity_curve, trades, p)


# =============================================================================
# METRICS
# =============================================================================

def _compute_metrics(ticker: str, symbol: str, final_equity: float,
                     equity_curve: list, trades: list[Trade], p: dict) -> dict:
    n = len(trades)
    init = p["initial_capital"]

    total_ret = (final_equity - init) / init * 100.0

    # CAGR
    if len(equity_curve) >= 2:
        yrs  = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / (365.25 * 86400)
        cagr = ((final_equity / init) ** (1 / max(yrs, 0.01)) - 1) * 100.0
    else:
        cagr = 0.0

    # Max drawdown (already tracked per-bar, recompute clean here)
    peak = init
    mdd  = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        mdd  = max(mdd, (peak - eq) / peak * 100.0 if peak > 0 else 0.0)

    # Sharpe (hourly returns, annualised)
    rets = []
    for i in range(1, len(equity_curve)):
        pe = equity_curve[i - 1][1]
        ce = equity_curve[i][1]
        if pe > 0:
            rets.append((ce - pe) / pe)
    sharpe = 0.0
    if len(rets) > 20:
        mr = sum(rets) / len(rets)
        sr = math.sqrt(sum((r - mr) ** 2 for r in rets) / len(rets))
        sharpe = (mr / sr * math.sqrt(24 * 365)) if sr > 0 else 0.0

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr     = len(wins) / n * 100.0 if n > 0 else 0.0
    pf_num = sum(t.pnl for t in wins)
    pf_den = abs(sum(t.pnl for t in losses))
    pf     = pf_num / pf_den if pf_den > 0 else 999.0
    avg_w  = pf_num / len(wins)   if wins   else 0.0
    avg_l  = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    # Exit reason breakdown
    exits: dict[str, int] = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    # Tier breakdown
    tiers: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for t in trades:
        tiers[t.tiers] = tiers.get(t.tiers, 0) + 1

    # Session win rates
    sess_tot: dict[str, int] = {}
    sess_win: dict[str, int] = {}
    for t in trades:
        sess_tot[t.session] = sess_tot.get(t.session, 0) + 1
        if t.pnl > 0:
            sess_win[t.session] = sess_win.get(t.session, 0) + 1
    sess_wr = {k: round(sess_win.get(k, 0) / v * 100, 1) for k, v in sess_tot.items()}

    # Direction split
    long_pnl  = sum(t.pnl for t in trades if t.direction == "long")
    short_pnl = sum(t.pnl for t in trades if t.direction == "short")
    long_n    = sum(1 for t in trades if t.direction == "long")
    short_n   = sum(1 for t in trades if t.direction == "short")
    long_wr   = sum(1 for t in trades if t.direction == "long"  and t.pnl > 0) / long_n  * 100 if long_n  else 0.0
    short_wr  = sum(1 for t in trades if t.direction == "short" and t.pnl > 0) / short_n * 100 if short_n else 0.0

    return {
        "ticker":         ticker,
        "symbol":         symbol,
        "bars_total":     len(equity_curve),
        "initial":        init,
        "final_equity":   round(final_equity, 2),
        "total_ret_pct":  round(total_ret, 2),
        "cagr_pct":       round(cagr, 2),
        "max_dd_pct":     round(mdd, 2),
        "sharpe":         round(sharpe, 3),
        "n_trades":       n,
        "win_rate_pct":   round(wr, 1),
        "profit_factor":  round(pf, 3),
        "avg_win":        round(avg_w, 2),
        "avg_loss":       round(avg_l, 2),
        "long_n":         long_n,
        "short_n":        short_n,
        "long_pnl":       round(long_pnl, 2),
        "short_pnl":      round(short_pnl, 2),
        "long_wr_pct":    round(long_wr, 1),
        "short_wr_pct":   round(short_wr, 1),
        "exit_reasons":   exits,
        "tier_breakdown": tiers,
        "session_wr":     sess_wr,
        "equity_curve":   [(str(ts), eq) for ts, eq in equity_curve[::288]],  # ~daily sample
    }


# =============================================================================
# DISPLAY
# =============================================================================

def _bar(label: str, width: int = 28) -> str:
    return f"{label:<{width}}"

def print_summary(results: list[dict]) -> None:
    W = 14
    tickers = [r["ticker"] for r in results]
    line    = "─" * (30 + W * len(tickers))

    print()
    print("═" * (30 + W * len(tickers)))
    print(f"  TIERED MEAN REVERSION — BACKTEST RESULTS  (x{PARAMS['leverage']:.0f} leverage)")
    print("═" * (30 + W * len(tickers)))

    def row(label: str, vals: list[str]) -> None:
        cells = "".join(f"{v:>{W}}" for v in vals)
        print(f"  {label:<28}{cells}")

    row("", tickers)
    print(line)
    row("Final equity ($)",    [f"{r['final_equity']:,.0f}"      for r in results])
    row("Total return (%)",    [f"{r['total_ret_pct']:+.1f}%"    for r in results])
    row("CAGR (%)",            [f"{r['cagr_pct']:+.1f}%"         for r in results])
    row("Max drawdown (%)",    [f"{r['max_dd_pct']:.1f}%"        for r in results])
    row("Sharpe ratio",        [f"{r['sharpe']:.3f}"             for r in results])
    print(line)
    row("Total trades",        [str(r["n_trades"])                for r in results])
    row("Win rate (%)",        [f"{r['win_rate_pct']:.1f}%"      for r in results])
    row("Profit factor",       [f"{r['profit_factor']:.2f}x"     for r in results])
    row("Avg win ($)",         [f"{r['avg_win']:+.2f}"           for r in results])
    row("Avg loss ($)",        [f"{r['avg_loss']:+.2f}"          for r in results])
    print(line)
    row("Long trades / WR",   [f"{r['long_n']} / {r['long_wr_pct']:.0f}%"  for r in results])
    row("Short trades / WR",  [f"{r['short_n']} / {r['short_wr_pct']:.0f}%" for r in results])
    row("Long PnL ($)",       [f"{r['long_pnl']:+,.0f}"          for r in results])
    row("Short PnL ($)",      [f"{r['short_pnl']:+,.0f}"         for r in results])
    print(line)
    row("Tier-1 only exits",  [str(r["tier_breakdown"].get(1, 0)) for r in results])
    row("Tier-2 exits",       [str(r["tier_breakdown"].get(2, 0)) for r in results])
    row("Tier-3 exits",       [str(r["tier_breakdown"].get(3, 0)) for r in results])
    print(line)

    for r in results:
        er = r["exit_reasons"]
        print(f"\n  {r['ticker']} exit breakdown:")
        for reason, cnt in sorted(er.items(), key=lambda x: -x[1]):
            pct = cnt / r["n_trades"] * 100 if r["n_trades"] else 0
            print(f"    {reason:<20} {cnt:4d}  ({pct:.1f}%)")
        print(f"  {r['ticker']} session win-rates:")
        for sess, wr in sorted(r["session_wr"].items()):
            print(f"    {sess:<12} {wr:.1f}%")

    print()
    print("═" * (30 + W * len(tickers)))
    print(f"  Params: bb_len={PARAMS['bb_len']}, Z=[{PARAMS['t1_z']},{PARAMS['t2_z']},{PARAMS['t3_z']}],"
          f" stop_z={PARAMS['stop_z']}, leverage={PARAMS['leverage']}x,"
          f" risk=[{PARAMS['t1_risk']},{PARAMS['t2_risk']},{PARAMS['t3_risk']}]%")
    print(f"  Filters: sessions={PARAMS['use_sessions']}, macro={PARAMS['use_macro']},"
          f" vol={PARAMS['use_vol']}, htf_ema={PARAMS['htf_ema_len']}, pdhl={PARAMS['use_pdhl']},"
          f" weekend={PARAMS['use_weekend']}")
    print("═" * (30 + W * len(tickers)))
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

    results = []
    for asset in ASSETS:
        print(f"  Running {asset}...", end=" ", flush=True)
        try:
            r = run_backtest(asset, PARAMS)
            results.append(r)
            print(f"done  ({r['n_trades']} trades, {r['total_ret_pct']:+.1f}%)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"ticker": asset, "error": str(exc)})

    valid = [r for r in results if "error" not in r]
    if valid:
        print_summary(valid)

    # Save JSON results
    out_path = Path(__file__).parent / "backtest_tmr_results.json"
    payload  = {
        "params":  {k: v for k, v in PARAMS.items() if k != "equity_curve"},
        "results": [{k: v for k, v in r.items() if k != "equity_curve"} for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Results saved → {out_path.name}\n")
