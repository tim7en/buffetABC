#!/usr/bin/env python3
"""
Tiered Mean Reversion — Python Backtest
========================================
Tier classification model:
  Each potential entry is scored by counting aligned confluences.
  The score drives tier assignment which sets position size.

  Confluences scored (0–8 points):
    1. |Z| ≥ t2_z  (deeper deviation)
    2. |Z| ≥ t1_z  (very deep deviation)
    3. Liquid session (London, NY, or their overlaps)
    4. Directional volume confirmation (selling/buying climax)
    5. HTF trend aligned (price side of long-term EMA)
    6. Previous-day H/L anomaly (price beyond PDH or PDL)
    7. Post macro-release reversion window
    8. Weekend overextension (thin liquidity)

  Score ≥ tier1_score  →  Tier 1 (highest conviction, biggest size)
  Score ≥ tier2_score  →  Tier 2 (medium conviction)
  Score <  tier2_score →  Tier 3 (lowest conviction, smallest size)

Assets tested: BTCUSDT, ETHUSDT, SOLUSDT  (5m bars, local Binance cache)
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
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

    # Bollinger / Z-score window
    # 1440 bars × 5m = 5-day lookback.
    # A 5-day mean is slow enough that Z reversion reflects genuine price
    # movement back toward the mean, not just the mean chasing price.
    "bb_len":  1440,

    # Z thresholds used in SCORING (not sequential tiers)
    #   t3_z = minimum Z to trigger any entry
    #   t2_z = deeper deviation (+1 point)
    #   t1_z = very deep deviation (+1 point, stacks with t2)
    "t3_z":   2.0,   # entry gate
    "t2_z":   2.5,   # score bonus
    "t1_z":   3.0,   # score bonus

    # Exit: rolling Z threshold (Z returns within exit_z of mean → close)
    # Stop: fixed price level set at entry time (entry_mean ± stop_z × entry_std)
    "stop_z": 4.0,   # hard stop
    "exit_z": 0.5,   # exit when rolling Z returns within 0.5σ (wider to avoid noise)

    # Confluence score boundaries
    "tier1_score": 4,   # score ≥ 4 → Tier 1
    "tier2_score": 2,   # score 2–3 → Tier 2; score < 2 → Tier 3
    "min_score":   2,   # skip weak single-confluence fades

    # Risk per tier (% of equity per trade)
    "t1_risk": 3.0,
    "t2_risk": 1.5,
    "t3_risk": 0.75,

    # Leverage: max position notional = equity × leverage
    "leverage": 5.0,

    # Sessions (UTC naive)
    "skip_offhours": True,   # block new entries during off-hours

    # US macro calendar guard
    "use_macro": True,       # halt entries during pre/release windows

    # Volume confirmation
    "use_vol":      True,
    "vol_len":      288,     # 1-day volume average (independent of bb_len)
    "vol_th":       1.20,    # relative-volume threshold for "high volume"
    "close_loc_th": 0.65,    # close location threshold for directional confirm

    # Regime filter: mean reversion performs poorly in strong directional trends
    "use_adx": True,
    "adx_len": 14,
    "adx_max": 25.0,

    # HTF trend bias (EMA in 5m bars; 600 × 5m ≈ 50 h)
    "use_htf":     True,
    "htf_ema_len": 600,

    # OU-style half-life gate on the Z-score itself.
    # Only fade deviations whose local reversion speed is still finite/usable.
    "use_ou":            True,
    "ou_len":            288,   # 1-day regime window
    "ou_recalc":         12,    # recompute roughly once per hour on 5m data
    "ou_half_life_min":  3.0,
    "ou_half_life_max": 72.0,
    "hl_exit_mult":      3.0,
    "hl_min_bars":       12,

    # Entry confirmation: wait for the extreme to start reverting first
    "use_reversal":            True,
    "min_reversal_z":          0.10,
    "require_close_reversal":  True,

    # Exits
    "use_fixed_target": True,

    # Max bars in trade (288 × 5m = 24 h time stop)
    "max_bars": 288,

    # Drawdown governor
    "use_dd":      False,
    "dd_halt_pct": 15.0,

    # Previous Day High/Low scoring
    "use_pdhl": True,   # add score point when price beyond PDH/PDL off-hours

    # Weekend overextension scoring
    "use_weekend": True,

    # Directions
    "allow_long":  True,
    "allow_short": True,
}

# =============================================================================
# INDICATOR HELPERS
# =============================================================================

class RollingWindow:
    """Fixed-size rolling buffer with O(1) SMA and population-STDEV."""
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
        n    = len(self.buf)
        mean = self._sum / n
        var  = max(self._sum_sq / n - mean * mean, 0.0)
        return math.sqrt(var)

    def values(self) -> list[float]:
        return list(self.buf)


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


class ADX:
    """Wilder ADX implementation for regime filtering."""
    def __init__(self, period: int) -> None:
        self.period = period
        self.prev_h: float | None = None
        self.prev_l: float | None = None
        self.prev_c: float | None = None

        self._seed_n = 0
        self._tr_sum = 0.0
        self._plus_dm_sum = 0.0
        self._minus_dm_sum = 0.0

        self._sm_tr: float | None = None
        self._sm_plus_dm: float | None = None
        self._sm_minus_dm: float | None = None
        self._dx_seed: deque[float] = deque(maxlen=period)
        self.value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self.prev_h is None or self.prev_l is None or self.prev_c is None:
            self.prev_h, self.prev_l, self.prev_c = high, low, close
            return self.value

        up_move = high - self.prev_h
        down_move = self.prev_l - low
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
        tr = max(high - low, abs(high - self.prev_c), abs(low - self.prev_c))

        if self._sm_tr is None:
            self._seed_n += 1
            self._tr_sum += tr
            self._plus_dm_sum += plus_dm
            self._minus_dm_sum += minus_dm

            if self._seed_n == self.period:
                self._sm_tr = self._tr_sum
                self._sm_plus_dm = self._plus_dm_sum
                self._sm_minus_dm = self._minus_dm_sum
                dx = self._dx()
                self._dx_seed.append(dx)
        else:
            self._sm_tr = self._sm_tr - (self._sm_tr / self.period) + tr
            self._sm_plus_dm = self._sm_plus_dm - (self._sm_plus_dm / self.period) + plus_dm
            self._sm_minus_dm = self._sm_minus_dm - (self._sm_minus_dm / self.period) + minus_dm

            dx = self._dx()
            if self.value is None:
                self._dx_seed.append(dx)
                if len(self._dx_seed) == self.period:
                    self.value = sum(self._dx_seed) / self.period
            else:
                self.value = ((self.value * (self.period - 1)) + dx) / self.period

        self.prev_h, self.prev_l, self.prev_c = high, low, close
        return self.value

    def _dx(self) -> float:
        if not self._sm_tr or self._sm_tr <= 0:
            return 0.0
        plus_di = 100.0 * (self._sm_plus_dm or 0.0) / self._sm_tr
        minus_di = 100.0 * (self._sm_minus_dm or 0.0) / self._sm_tr
        di_sum = plus_di + minus_di
        if di_sum <= 0:
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / di_sum


class HalfLifeEstimator:
    """Rolling OU-style half-life estimate on a stationary series."""
    def __init__(self, window: int, recalc_every: int = 1) -> None:
        self.window = RollingWindow(window)
        self.recalc_every = max(recalc_every, 1)
        self._since_recalc = 0
        self.lambda_: float | None = None
        self.half_life: float | None = None

    def update(self, value: float) -> float | None:
        self.window.push(value)
        if not self.window.full():
            return self.half_life

        self._since_recalc += 1
        if self.half_life is not None and self._since_recalc < self.recalc_every:
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
        else:
            cov_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            self.lambda_ = cov_xy / var_x
            if self.lambda_ >= 0:
                self.half_life = math.inf
            else:
                self.half_life = -math.log(2.0) / self.lambda_

        self._since_recalc = 0
        return self.half_life


# =============================================================================
# SESSION DETECTION  (UTC naive datetimes)
# =============================================================================

def session_info(ts: datetime) -> dict[str, Any]:
    """Classify a UTC bar timestamp into trading sessions."""
    t    = ts.time()
    wday = ts.weekday()          # 0 = Mon … 6 = Sun

    in_tokyo  = dtime(0,  0) <= t < dtime(8, 30)
    in_london = dtime(7,  0) <= t < dtime(16, 0)
    in_ny     = dtime(13, 30) <= t < dtime(21, 0)

    overlap_ln = in_london and in_ny
    overlap_al = in_tokyo  and in_london

    in_any  = in_tokyo or in_london or in_ny
    is_wknd = wday >= 5

    if   overlap_ln: label = "OVL_LN"
    elif overlap_al: label = "OVL_AL"
    elif in_london:  label = "LONDON"
    elif in_ny:      label = "NY"
    elif in_tokyo:   label = "TOKYO"
    else:            label = "OFF"

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
      NFP    : 1st Friday  of month      (08:30 ET = 13:30 UTC)
      CPI/PPI: 2nd–3rd Tue/Wed           (same time)
      GDP/PCE: last Thursday             (08:30 ET)
    """
    if not p["use_macro"]:
        return {"is_macro": False, "label": "NORMAL", "is_post": False, "halted": False}

    wday = ts.weekday()
    dom  = ts.day
    t    = ts.time()

    raw_nfp = (wday == 4) and (dom <= 7)
    raw_cpi = (wday in (1, 2)) and (8 <= dom <= 21)
    raw_gdp = (wday == 3) and (dom >= 22)

    if not (raw_nfp or raw_cpi or raw_gdp):
        return {"is_macro": False, "label": "NORMAL", "is_post": False, "halted": False}

    event   = "NFP" if raw_nfp else ("CPI" if raw_cpi else "GDP")
    pre_win = dtime(13,  0) <= t < dtime(13, 30)
    rel_win = dtime(13, 30) <= t < dtime(14, 30)
    pst_win = dtime(14, 30) <= t < dtime(16, 30)

    if pre_win or rel_win:
        return {"is_macro": True, "label": f"{event}-HALT", "is_post": False, "halted": True}
    if pst_win:
        return {"is_macro": True, "label": f"{event}-POST", "is_post": True,  "halted": False}
    return {"is_macro": True, "label": event, "is_post": False, "halted": False}


# =============================================================================
# CONFLUENCE SCORING
# =============================================================================

def score_entry(
    direction: str,
    z: float,
    sess: dict,
    vcl: bool,
    vcs: bool,
    htf_aligned: bool,
    pdhl_triggered: bool,
    is_macro_post: bool,
    is_weekend_ext: bool,
    p: dict,
) -> int:
    """
    Count aligned confluences and return a score 0–8.

    Points:
      1  |Z| ≥ t2_z   (deeper deviation)
      2  |Z| ≥ t1_z   (very deep, stacks with #1)
      3  Liquid session (London, NY, or overlaps)
      4  Directional volume climax (confirms the reversion direction)
      5  HTF trend aligned (temporary deviation within larger trend)
      6  PDH/PDL anomaly (price beyond previous day's range off-hours)
      7  Post macro-release reversion window
      8  Weekend overextension in thin liquidity
    """
    score  = 0
    abs_z  = abs(z)

    # Z depth
    if abs_z >= p["t2_z"]:  score += 1
    if abs_z >= p["t1_z"]:  score += 1   # stacks: 3σ+ gets 2 points total

    # Liquid session
    if sess["in_london"] or sess["in_ny"] or sess["overlap_ln"] or sess["overlap_al"]:
        score += 1

    # Directional volume climax
    if direction == "long"  and vcl:  score += 1
    if direction == "short" and vcs:  score += 1

    # HTF alignment
    if htf_aligned:  score += 1

    # Off-hours PDH/PDL anomaly
    if p["use_pdhl"] and pdhl_triggered:  score += 1

    # Post macro-release
    if p["use_macro"] and is_macro_post:  score += 1

    # Weekend overextension
    if p["use_weekend"] and is_weekend_ext:  score += 1

    return score


def tier_from_score(score: int, p: dict) -> int:
    """Map confluence score → tier (1 = best, 3 = weakest)."""
    if score >= p["tier1_score"]:  return 1
    if score >= p["tier2_score"]:  return 2
    return 3


# =============================================================================
# POSITION & TRADE DATA CLASSES
# =============================================================================

@dataclass
class Position:
    direction:   str
    tier:        int
    score:       int
    total_qty:   float
    avg_entry:   float
    stop_price:  float   # fixed at entry time (entry_mean ± stop_z × entry_std)
    exit_target: float   # fixed at entry time (entry_mean ± exit_z × entry_std)
    half_life:   float | None
    max_hold_bars: int
    bars_held:   int = 0
    entry_ts:    datetime | None = None


@dataclass
class Trade:
    direction:   str
    tier:        int
    score:       int
    entry_ts:    datetime
    exit_ts:     datetime
    entry_price: float
    exit_price:  float
    qty:         float
    pnl:         float
    exit_reason: str
    session:     str


# =============================================================================
# RISK SIZING
# =============================================================================

def calc_qty(equity: float, base_risk: float, entry: float,
             stop: float, leverage: float) -> float:
    """
    qty = min(risk-based size, leverage cap)
    risk-based : equity × risk% / stop_distance
    leverage cap: equity × leverage / entry_price
    """
    adj_risk = base_risk / 100.0
    stop_d   = abs(entry - stop)
    if stop_d <= 0 or entry <= 0 or equity <= 0:
        return 0.0
    qty_risk = (equity * adj_risk) / stop_d
    qty_lev  = (equity * leverage) / entry
    return max(min(qty_risk, qty_lev), 0.0)


def derive_hold_limit(entry_half_life: float | None, p: dict) -> int:
    """Convert the estimated half-life into a max holding window."""
    base_limit = p["max_bars"]
    if not p["use_ou"] or entry_half_life is None or not math.isfinite(entry_half_life):
        return base_limit

    hl_limit = max(int(round(entry_half_life * p["hl_exit_mult"])), p["hl_min_bars"])
    if base_limit > 0:
        return min(base_limit, hl_limit)
    return hl_limit


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
    adx_calc = ADX(p["adx_len"])
    hl_est = HalfLifeEstimator(p["ou_len"], p["ou_recalc"])

    prev_z: float = 0.0
    prev_c: float | None = None

    # Previous day H/L tracking
    pdh: float = float("nan")
    pdl: float = float("nan")
    day_h: float = float("-inf")
    day_l: float = float("inf")
    cur_date = None

    _mins_per_bar = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}.get(p["interval"], 60)
    _bars_per_day = (24 * 60) // _mins_per_bar
    warmup_bars   = p["warmup_days"] * _bars_per_day

    for idx, bar in enumerate(bars):
        ts = bar["timestamp"]
        h, l, c, v = bar["high"], bar["low"], bar["close"], bar["volume"]

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
        adx_val = adx_calc.update(h, l, c)

        if bb_win.full():
            bm = bb_win.sma()
            bs = bb_win.stdev()
            _z = (c - bm) / bs if (bs and bs > 0) else 0.0
            half_life = hl_est.update(_z)
        else:
            _z = 0.0
            half_life = None

        if idx < warmup_bars:
            prev_z = _z
            prev_c = c
            continue

        if not bb_win.full():
            prev_z = _z
            prev_c = c
            continue

        bb_mean = bb_win.sma()
        bb_std  = bb_win.stdev()
        if bb_std is None or bb_std <= 0:
            prev_z = _z
            prev_c = c
            continue

        z = _z

        # ── Session ───────────────────────────────────────────────────────────
        sess    = session_info(ts)
        ok_sess = not p["skip_offhours"] or sess["in_any"]

        # ── Macro ─────────────────────────────────────────────────────────────
        macro   = macro_status(ts, p)
        halted  = macro["halted"]

        # ── Volume ────────────────────────────────────────────────────────────
        vsma      = vol_win.sma() or 1.0
        rel_vol   = v / vsma if vsma > 0 else 1.0
        bar_rng   = max(h - l, 1e-12)
        close_loc = (c - l) / bar_rng

        vol_high = rel_vol >= p["vol_th"]
        vcl = p["use_vol"] and vol_high and close_loc >= p["close_loc_th"]           # long vol confirm
        vcs = p["use_vol"] and vol_high and close_loc <= (1.0 - p["close_loc_th"])   # short vol confirm

        # ── HTF trend bias ────────────────────────────────────────────────────
        htf_val   = htf_ema.value
        htf_long  = not p["use_htf"] or htf_val is None or c > htf_val
        htf_short = not p["use_htf"] or htf_val is None or c < htf_val

        # ── Regime filters ────────────────────────────────────────────────────
        is_ranging = not p["use_adx"] or adx_val is None or adx_val <= p["adx_max"]
        hl_ok = (
            not p["use_ou"] or
            half_life is None or
            (
                math.isfinite(half_life) and
                p["ou_half_life_min"] <= half_life <= p["ou_half_life_max"]
            )
        )

        # ── Drawdown governor ─────────────────────────────────────────────────
        dd_now  = (equity_peak - equity) / equity_peak * 100.0 if equity_peak > 0 else 0.0
        dd_halt = p["use_dd"] and dd_now >= p["dd_halt_pct"]

        ok_entry = ok_sess and not halted and not dd_halt and is_ranging and hl_ok

        # ── Update bars-held (stop/target are FIXED at entry time) ───────────
        if pos is not None:
            pos.bars_held += 1

        # ── EXIT LOGIC ────────────────────────────────────────────────────────
        # Stop   : fixed price level captured at entry (can't be mean-chased away).
        # Target : rolling Z threshold — works correctly with a slow 5-day mean
        #          because Z reverts mainly via price movement, not mean drift.
        exit_reason: str | None = None
        if pos is not None:
            hold_limit = pos.max_hold_bars if pos.max_hold_bars > 0 else p["max_bars"]
            if pos.direction == "long":
                if c <= pos.stop_price:
                    exit_reason = "Z-Stop"
                elif p["use_fixed_target"] and c >= pos.exit_target:
                    exit_reason = "Fixed-Target"
                elif prev_z < -p["exit_z"] and z >= -p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif hold_limit > 0 and pos.bars_held >= hold_limit:
                    exit_reason = "Time-Stop"
            else:  # short
                if c >= pos.stop_price:
                    exit_reason = "Z-Stop"
                elif p["use_fixed_target"] and c <= pos.exit_target:
                    exit_reason = "Fixed-Target"
                elif prev_z > p["exit_z"] and z <= p["exit_z"]:
                    exit_reason = "Mean-Revert"
                elif hold_limit > 0 and pos.bars_held >= hold_limit:
                    exit_reason = "Time-Stop"

        if exit_reason and pos is not None:
            sign      = 1 if pos.direction == "long" else -1
            gross_pnl = pos.total_qty * (c - pos.avg_entry) * sign
            fees      = (pos.avg_entry + c) * pos.total_qty * p["commission_pct"]
            net_pnl   = gross_pnl - fees
            equity   += net_pnl

            trades.append(Trade(
                direction=pos.direction,
                tier=pos.tier,
                score=pos.score,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                entry_price=pos.avg_entry,
                exit_price=c,
                qty=pos.total_qty,
                pnl=net_pnl,
                exit_reason=exit_reason,
                session=sess["label"],
            ))
            pos = None

        # ── ENTRY LOGIC (confluence scoring) ──────────────────────────────────
        if ok_entry and pos is None:
            direction: str | None = None
            long_extreme = min(prev_z, z) <= -p["t3_z"]
            short_extreme = max(prev_z, z) >= p["t3_z"]
            close_up = prev_c is not None and c > prev_c
            close_down = prev_c is not None and c < prev_c
            long_reversal = (
                not p["use_reversal"] or
                (
                    long_extreme and
                    z < -p["exit_z"] and
                    (z - prev_z) >= p["min_reversal_z"] and
                    (not p["require_close_reversal"] or close_up)
                )
            )
            short_reversal = (
                not p["use_reversal"] or
                (
                    short_extreme and
                    z > p["exit_z"] and
                    (prev_z - z) >= p["min_reversal_z"] and
                    (not p["require_close_reversal"] or close_down)
                )
            )

            if p["allow_long"] and long_reversal:
                direction = "long"
            elif p["allow_short"] and short_reversal:
                direction = "short"

            if direction is not None:
                pdh_ok = not math.isnan(pdh)
                pdl_ok = not math.isnan(pdl)

                # PDH/PDL anomaly: price beyond yesterday's range in off-hours
                pdhl_trig = (
                    (direction == "long"  and pdl_ok and c < pdl and not sess["in_any"]) or
                    (direction == "short" and pdh_ok and c > pdh and not sess["in_any"])
                )

                # Weekend overextension: big move on thin-liquidity weekend
                wknd_ext = sess["is_weekend"] and abs(z) >= p["t2_z"]

                htf_aln = htf_long if direction == "long" else htf_short

                sc   = score_entry(
                    direction, z, sess, vcl, vcs,
                    htf_aln, pdhl_trig, macro["is_post"], wknd_ext, p
                )

                if sc >= p["min_score"]:
                    tier      = tier_from_score(sc, p)
                    base_risk = {1: p["t1_risk"], 2: p["t2_risk"], 3: p["t3_risk"]}[tier]

                    # Fixed price levels from entry-bar bb_mean/bb_std
                    if direction == "long":
                        stop   = bb_mean - p["stop_z"] * bb_std
                        target = bb_mean - p["exit_z"] * bb_std
                    else:
                        stop   = bb_mean + p["stop_z"] * bb_std
                        target = bb_mean + p["exit_z"] * bb_std

                    qty = calc_qty(equity, base_risk, c, stop, p["leverage"])
                    if qty > 0:
                        hold_limit = derive_hold_limit(half_life, p)
                        pos = Position(direction, tier, sc, qty, c, stop, target, half_life, hold_limit, 0, ts)

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
        prev_c = c

    # Close any open position at last bar close
    if pos is not None and equity_curve:
        last_ts = equity_curve[-1][0]
        last_c  = bars[-1]["close"]
        sign      = 1 if pos.direction == "long" else -1
        gross_pnl = pos.total_qty * (last_c - pos.avg_entry) * sign
        fees      = (pos.avg_entry + last_c) * pos.total_qty * p["commission_pct"]
        equity   += gross_pnl - fees
        trades.append(Trade(
            direction=pos.direction, tier=pos.tier, score=pos.score,
            entry_ts=pos.entry_ts, exit_ts=last_ts,
            entry_price=pos.avg_entry, exit_price=last_c, qty=pos.total_qty,
            pnl=gross_pnl - fees, exit_reason="End-Of-Data", session="END",
        ))

    return _compute_metrics(ticker, symbol, equity, equity_curve, trades, p)


# =============================================================================
# METRICS
# =============================================================================

def _compute_metrics(ticker: str, symbol: str, final_equity: float,
                     equity_curve: list, trades: list[Trade], p: dict) -> dict:
    n    = len(trades)
    init = p["initial_capital"]

    total_ret = (final_equity - init) / init * 100.0

    # CAGR
    if len(equity_curve) >= 2:
        yrs  = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / (365.25 * 86400)
        cagr = ((final_equity / init) ** (1 / max(yrs, 0.01)) - 1) * 100.0
    else:
        cagr = 0.0

    # Max drawdown
    peak = init
    mdd  = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        mdd  = max(mdd, (peak - eq) / peak * 100.0 if peak > 0 else 0.0)

    # Sharpe (bar-level returns, annualised)
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
        bars_per_year = (24 * 60) // {"5m": 5, "15m": 15, "30m": 30, "60m": 60}.get(p["interval"], 60) * 365
        sharpe = (mr / sr * math.sqrt(bars_per_year)) if sr > 0 else 0.0

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr     = len(wins) / n * 100.0 if n > 0 else 0.0
    pf_num = sum(t.pnl for t in wins)
    pf_den = abs(sum(t.pnl for t in losses))
    pf     = pf_num / pf_den if pf_den > 0 else (999.0 if pf_num > 0 else 0.0)
    avg_w  = pf_num / len(wins)   if wins   else 0.0
    avg_l  = sum(t.pnl for t in losses) / len(losses) if losses else 0.0

    # Exit reason breakdown
    exits: dict[str, int] = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    # Tier breakdown
    tier_n:   dict[int, int]   = {1: 0, 2: 0, 3: 0}
    tier_pnl: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
    tier_win: dict[int, int]   = {1: 0, 2: 0, 3: 0}
    for t in trades:
        tier_n[t.tier]   = tier_n.get(t.tier, 0)   + 1
        tier_pnl[t.tier] = tier_pnl.get(t.tier, 0.0) + t.pnl
        if t.pnl > 0:
            tier_win[t.tier] = tier_win.get(t.tier, 0) + 1
    tier_wr = {k: round(tier_win.get(k, 0) / tier_n[k] * 100, 1) if tier_n[k] > 0 else 0.0
               for k in (1, 2, 3)}

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

    # Score distribution
    score_dist: dict[int, int] = {}
    for t in trades:
        score_dist[t.score] = score_dist.get(t.score, 0) + 1

    return {
        "ticker":          ticker,
        "symbol":          symbol,
        "bars_total":      len(equity_curve),
        "initial":         init,
        "final_equity":    round(final_equity, 2),
        "total_ret_pct":   round(total_ret, 2),
        "cagr_pct":        round(cagr, 2),
        "max_dd_pct":      round(mdd, 2),
        "sharpe":          round(sharpe, 3),
        "n_trades":        n,
        "win_rate_pct":    round(wr, 1),
        "profit_factor":   round(pf, 3),
        "avg_win":         round(avg_w, 2),
        "avg_loss":        round(avg_l, 2),
        "long_n":          long_n,
        "short_n":         short_n,
        "long_pnl":        round(long_pnl, 2),
        "short_pnl":       round(short_pnl, 2),
        "long_wr_pct":     round(long_wr, 1),
        "short_wr_pct":    round(short_wr, 1),
        "tier_n":          tier_n,
        "tier_pnl":        {k: round(v, 2) for k, v in tier_pnl.items()},
        "tier_wr_pct":     tier_wr,
        "exit_reasons":    exits,
        "session_wr":      sess_wr,
        "score_dist":      dict(sorted(score_dist.items())),
        "equity_curve":    [(str(ts), eq) for ts, eq in equity_curve[::288]],
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_summary(results: list[dict]) -> None:
    W       = 14
    tickers = [r["ticker"] for r in results]
    line    = "─" * (32 + W * len(tickers))

    print()
    print("═" * (32 + W * len(tickers)))
    print(f"  TIERED MEAN REVERSION — BACKTEST RESULTS  (x{PARAMS['leverage']:.0f} leverage)")
    print("═" * (32 + W * len(tickers)))

    def row(label: str, vals: list[str]) -> None:
        cells = "".join(f"{v:>{W}}" for v in vals)
        print(f"  {label:<30}{cells}")

    row("", tickers)
    print(line)
    row("Final equity ($)",    [f"{r['final_equity']:,.0f}"      for r in results])
    row("Total return (%)",    [f"{r['total_ret_pct']:+.1f}%"    for r in results])
    row("CAGR (%)",            [f"{r['cagr_pct']:+.1f}%"         for r in results])
    row("Max drawdown (%)",    [f"{r['max_dd_pct']:.1f}%"        for r in results])
    row("Sharpe ratio",        [f"{r['sharpe']:.3f}"             for r in results])
    print(line)
    row("Total trades",        [str(r["n_trades"])               for r in results])
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

    for r in results:
        tn = r["tier_n"]; tp = r["tier_pnl"]; tw = r["tier_wr_pct"]
        print(f"\n  {r['ticker']} tier breakdown:")
        for tier in (1, 2, 3):
            n = tn.get(tier, 0)
            if n:
                print(f"    T{tier}  {n:4d} trades  WR {tw.get(tier,0):.1f}%  PnL ${tp.get(tier,0):+,.0f}")
        print(f"  {r['ticker']} score distribution:")
        for sc, cnt in sorted(r["score_dist"].items()):
            print(f"    score={sc}  {cnt:4d} trades")
        print(f"  {r['ticker']} exit breakdown:")
        for reason, cnt in sorted(r["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = cnt / r["n_trades"] * 100 if r["n_trades"] else 0
            print(f"    {reason:<20} {cnt:4d}  ({pct:.1f}%)")
        print(f"  {r['ticker']} session win-rates:")
        for s, wr in sorted(r["session_wr"].items()):
            print(f"    {s:<12} {wr:.1f}%")

    print()
    print("═" * (32 + W * len(tickers)))
    print(f"  Params: bb_len={PARAMS['bb_len']}, entry_z≥{PARAMS['t3_z']}"
          f"  score→tier: T1≥{PARAMS['tier1_score']} T2≥{PARAMS['tier2_score']}")
    print(f"  Z thresholds: t2={PARAMS['t2_z']} t1={PARAMS['t1_z']}"
          f"  stop={PARAMS['stop_z']}  exit={PARAMS['exit_z']}")
    print(f"  Risk: T1={PARAMS['t1_risk']}%  T2={PARAMS['t2_risk']}%  T3={PARAMS['t3_risk']}%"
          f"  leverage={PARAMS['leverage']}x  min_score={PARAMS['min_score']}")
    print(f"  Regime: ADX≤{PARAMS['adx_max']} ({'on' if PARAMS['use_adx'] else 'off'})"
          f"  OU half-life={PARAMS['ou_half_life_min']:.0f}..{PARAMS['ou_half_life_max']:.0f}"
          f"  reversal={'on' if PARAMS['use_reversal'] else 'off'}")
    print("═" * (32 + W * len(tickers)))
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
            import traceback
            traceback.print_exc()
            print(f"ERROR: {exc}")
            results.append({"ticker": asset, "error": str(exc)})

    valid = [r for r in results if "error" not in r]
    if valid:
        print_summary(valid)

    out_path = Path(__file__).parent / "backtest_tmr_results.json"
    payload  = {
        "params":  {k: v for k, v in PARAMS.items()},
        "results": [{k: v for k, v in r.items() if k != "equity_curve"} for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Results saved → {out_path.name}\n")
