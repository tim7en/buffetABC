"""
tools/turtle_atr_leverage.py

ATR-sized Turtle system with SMA-200 regime filters and configurable leverage.

Mechanics
---------
Long regime    : close > SMA(200)
  Entry        : close breaks above Donchian(55) high
  Exit         : close breaks below Donchian(20) low, falls below SMA(200),
                 or hits the ATR stop

Short regime   : close < SMA(200)
  Entry        : close breaks below Donchian(55) low
  Exit         : close breaks above Donchian(20) high, rises above SMA(200),
                 or hits the ATR stop

Position size (turtle N-unit rule)
  N        = 20-day ATR (in dollars per share)
  1 unit   = equity * risk_per_unit / N   (shares)
  Leverage = unit_value / equity = risk_per_unit * price / N = risk_per_unit / atr_pct
  Cap      : total leverage <= max_lev (3x here)

Pyramiding (optional)
  Longs  : add 1 unit each time price advances 0.5*N above the last add price
  Shorts : add 1 unit each time price falls 0.5*N below the last add price
  Maximum 3 units total. Aggregate stop stays 2*N away from average entry.

Borrow cost  : 5.5% annual on leveraged capital above 1x while in a trade.

Strategies compared
-------------------
1. Long-only Donchian+SMA200 reference
2. Long-only ATR-sized variants
3. Long/short regime variants that short below the SMA-200

Applied to both QQQ and SPY.

Output
------
  reports/turtle_atr_leverage/turtle_atr_leverage.png
  reports/turtle_atr_leverage/metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq

# ── config ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]
TIINGO = ROOT / "cache" / "cache" / "tiingo"
OUT    = ROOT / "reports" / "turtle_atr_leverage"
OUT.mkdir(parents=True, exist_ok=True)

CAPITAL       = 10_000.0
BORROW_RATE   = 0.055      # annual, on borrowed portion
RISK_PER_UNIT = 0.01       # 1 % of equity risked per ATR unit
ATR_PERIOD    = 20
SMA_PERIOD    = 200
ENTRY_PERIOD  = 55
EXIT_PERIOD   = 20
ATR_STOP_MULT = 2.0        # stop = entry - 2 * ATR
ADD_INTERVAL  = 0.5        # add next unit every +0.5 ATR price advance
MAX_UNITS     = 3          # max pyramid units
MAX_LEV       = 3.0        # hard cap on total leverage
VOL_GATE_ANN  = 0.25       # vol gate threshold (25% ann) for strategy 4
RATCHET_MIN_EXIT = 5       # floor for dynamic Donchian exit period
RATCHET_TIGHTEN_DAYS = 5   # shorten exit by 5 days at each profit milestone

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_daily(symbol: str) -> pd.DataFrame:
    path = TIINGO / f"{symbol}_5m.parquet"
    tbl  = pq.read_table(path, columns=["time", "o", "h", "l", "c", "v"])
    df   = tbl.to_pandas()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_convert("America/New_York").dt.date
    daily = (
        df.groupby("date")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"),  close=("close","last"), volume=("volume","sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder ATR (true range smoothed by simple average for stability)."""
    n   = len(close)
    tr  = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    atr = np.zeros(n)
    for i in range(period, n):
        atr[i] = tr[i-period+1:i+1].mean()
    # seed initial values
    atr[:period] = tr[1:period+1].mean() if period > 1 else tr[1]
    return atr


def _rolling_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        out[i] = arr[max(0, i-period+1):i+1].mean()
    return out


def _rolling_max(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), arr[0])
    for i in range(1, len(arr)):
        out[i] = arr[max(0, i-period):i].max()
    return out


def _rolling_min(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), arr[0])
    for i in range(1, len(arr)):
        out[i] = arr[max(0, i-period):i].min()
    return out


def _rolling_vol_ann(returns: np.ndarray, period: int = 20) -> np.ndarray:
    out = np.zeros(len(returns))
    for i in range(1, len(returns)):
        w = returns[max(0, i-period):i]
        out[i] = w.std() * np.sqrt(252) if len(w) > 1 else 0.0
    return out


def _metrics(eq: pd.Series, label: str) -> dict:
    eq = eq.ffill().dropna()
    if len(eq) < 2 or eq.iloc[-1] <= 1e-3:
        return {"Strategy": label, "Total Return": "BLOWN UP", "CAGR": "-",
                "Max DD": "-", "Sharpe": "-", "Final $": "$0",
                "Avg Lev": "-", "Time in Mkt": "-"}
    years    = (eq.index[-1] - eq.index[0]).days / 365.25
    tot_r    = eq.iloc[-1] / eq.iloc[0] - 1
    cagr     = (1 + tot_r) ** (1 / years) - 1 if years > 0 else 0.0
    roll_max = eq.cummax()
    max_dd   = ((eq - roll_max) / roll_max).min()
    dr       = eq.pct_change().dropna()
    sharpe   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    return {
        "Strategy":    label,
        "Total Return": f"{tot_r*100:.1f}%",
        "CAGR":        f"{cagr*100:.1f}%",
        "Max DD":      f"{max_dd*100:.1f}%",
        "Sharpe":      f"{sharpe:.2f}",
        "Final $":     f"${eq.iloc[-1]:,.0f}",
    }

# ── load & pre-compute indicators ─────────────────────────────────────────────

print("Loading data …")
qqq = _load_daily("QQQ")
spy = _load_daily("SPY")
common = sorted(set(qqq["date"]) & set(spy["date"]))
qqq = qqq[qqq["date"].isin(common)].set_index("date").sort_index()
spy = spy[spy["date"].isin(common)].set_index("date").sort_index()
n   = len(qqq)
print(f"  {n} daily bars  {qqq.index[0].date()} to {qqq.index[-1].date()}")


def _precompute(df: pd.DataFrame) -> dict:
    c  = df["close"].values
    h  = df["high"].values
    lo = df["low"].values
    r  = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    entry_high = _rolling_max(c, ENTRY_PERIOD)
    entry_low  = _rolling_min(c, ENTRY_PERIOD)
    exit_high  = _rolling_max(c, EXIT_PERIOD)
    exit_low   = _rolling_min(c, EXIT_PERIOD)
    return {
        "close":    c,
        "high":     h,
        "low":      lo,
        "returns":  r,
        "atr":      _atr(h, lo, c),
        "sma200":   _rolling_sma(c, SMA_PERIOD),
        "entry_high": entry_high,
        "entry_low":  entry_low,
        "exit_high":  exit_high,
        "exit_low":   exit_low,
        "don_high": entry_high,   # compatibility alias for long-only entry
        "don_low":  exit_low,     # compatibility alias for long-only exit
        "vol_ann":  _rolling_vol_ann(r),
    }

print("Pre-computing indicators …")
Q = _precompute(qqq)
S = _precompute(spy)

# ── Strategy 1: Fixed 3x Donchian+SMA200 (reference) ─────────────────────────

def fixed_lev_don_sma(
    ind: dict,
    lev: float = 3.0,
    entry_period: int = ENTRY_PERIOD,
    exit_period: int = EXIT_PERIOD,
    sma_period: int = SMA_PERIOD,
) -> pd.Series:
    """Flat leverage when Donchian+SMA is active, applied from the next bar."""
    c, r = ind["close"], ind["returns"]
    dh = ind["entry_high"] if entry_period == ENTRY_PERIOD else _rolling_max(c, entry_period)
    dl = ind["exit_low"] if exit_period == EXIT_PERIOD else _rolling_min(c, exit_period)
    sma = ind["sma200"] if sma_period == SMA_PERIOD else _rolling_sma(c, sma_period)
    warmup = max(entry_period, sma_period)

    eq       = np.empty(n)
    eq[0]    = CAPITAL
    in_trade = False

    for i in range(1, n):
        if in_trade:
            cost   = (lev - 1) * BORROW_RATE / 252
            eq[i]  = eq[i-1] * max(0.0, 1 + lev * r[i] - cost)
        else:
            eq[i]  = eq[i-1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        if in_trade:
            don_exit  = c[i] < dl[i]
            if don_exit or not above_sma:
                in_trade = False
        else:
            if c[i] > dh[i] and above_sma:
                in_trade = True

    return pd.Series(eq, index=qqq.index)


def fixed_lev_don_sma_profit_ratchet(
    ind: dict,
    lev: float = 3.0,
    profit_step: float = 0.03,
    tighten_days: int = RATCHET_TIGHTEN_DAYS,
    min_exit_period: int = RATCHET_MIN_EXIT,
) -> pd.Series:
    """
    Long-only 3x Donchian+SMA strategy with a profit ratchet on the exit channel.

    Once a trade is open, every `profit_step` of open profit shortens the exit
    Donchian by `tighten_days`, down to `min_exit_period`. The base exit resets
    back to EXIT_PERIOD after the trade closes.
    """
    c, r = ind["close"], ind["returns"]
    dh = ind["don_high"]
    sma = ind["sma200"]
    warmup = max(ENTRY_PERIOD, SMA_PERIOD)

    eq = np.empty(n)
    eq[0] = CAPITAL
    in_trade = False
    entry_price = 0.0

    for i in range(1, n):
        if in_trade:
            cost = (lev - 1) * BORROW_RATE / 252
            eq[i] = eq[i - 1] * max(0.0, 1 + lev * r[i] - cost)
        else:
            eq[i] = eq[i - 1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        if in_trade:
            open_profit = c[i] / entry_price - 1 if entry_price > 0 else 0.0
            tighten_steps = max(0, int(np.floor(open_profit / profit_step)))
            exit_period = max(min_exit_period, EXIT_PERIOD - tighten_days * tighten_steps)
            exit_low = c[max(0, i - exit_period):i].min()
            if c[i] < exit_low or not above_sma:
                in_trade = False
                entry_price = 0.0
        else:
            if c[i] > dh[i] and above_sma:
                in_trade = True
                entry_price = c[i]

    return pd.Series(eq, index=qqq.index)

# ── Strategy 2: ATR-sized, no pyramid, 3x cap ────────────────────────────────

def atr_sized_no_pyramid(ind: dict, max_lev: float = MAX_LEV,
                          risk: float = RISK_PER_UNIT) -> tuple[pd.Series, pd.Series]:
    """
    Size position via ATR:
      leverage = risk / atr_pct  (capped at max_lev)
    Stop: 2*ATR below entry.
    Exit: Donchian exit OR stop OR below SMA200.
    Returns (equity_series, leverage_series).
    """
    c, h, lo = ind["close"], ind["high"], ind["low"]
    r        = ind["returns"]
    atr      = ind["atr"]
    sma      = ind["sma200"]
    dh       = ind["don_high"]
    dl       = ind["don_low"]
    warmup   = max(ENTRY_PERIOD, SMA_PERIOD, ATR_PERIOD)

    eq       = np.empty(n)
    eq[0]    = CAPITAL
    lev_log  = np.zeros(n)
    in_trade = False
    lev      = 0.0
    stop     = 0.0

    for i in range(1, n):
        active_lev = lev if in_trade else 0.0
        if active_lev > 0:
            cost   = (active_lev - 1) * BORROW_RATE / 252 if active_lev > 1 else 0.0
            eq[i]  = eq[i-1] * max(0.0, 1 + active_lev * r[i] - cost)
            lev_log[i] = active_lev
        else:
            eq[i] = eq[i-1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        if in_trade:
            don_exit  = c[i] < dl[i]
            stop_hit  = c[i] < stop
            if don_exit or stop_hit or not above_sma:
                in_trade = False
                lev      = 0.0
        elif atr[i] > 0 and c[i] > dh[i] and above_sma:
            atr_pct  = atr[i] / c[i]
            lev      = min(risk / atr_pct, max_lev)
            stop     = c[i] - ATR_STOP_MULT * atr[i]
            in_trade = True

    return pd.Series(eq, index=qqq.index), pd.Series(lev_log, index=qqq.index)


# ── Strategy 3: ATR-sized + pyramid (up to MAX_UNITS) ────────────────────────

def atr_sized_pyramid(ind: dict, max_lev: float = MAX_LEV,
                       risk: float = RISK_PER_UNIT) -> tuple[pd.Series, pd.Series]:
    """
    Same ATR sizing but add 1 unit each time price advances 0.5*N from last add.
    Max MAX_UNITS units.  Aggregate stop = 2*ATR below avg entry price.
    """
    c        = ind["close"]
    r        = ind["returns"]
    atr      = ind["atr"]
    sma      = ind["sma200"]
    dh       = ind["don_high"]
    dl       = ind["don_low"]
    warmup   = max(ENTRY_PERIOD, SMA_PERIOD, ATR_PERIOD)

    eq       = np.empty(n)
    eq[0]    = CAPITAL
    lev_log  = np.zeros(n)
    in_trade = False
    units    = 0
    total_lev = 0.0
    last_add  = 0.0
    avg_entry = 0.0
    stop      = 0.0

    for i in range(1, n):
        active_lev = total_lev if in_trade else 0.0
        if active_lev > 0:
            cost  = (active_lev - 1) * BORROW_RATE / 252 if active_lev > 1 else 0.0
            eq[i] = eq[i-1] * max(0.0, 1 + active_lev * r[i] - cost)
            lev_log[i] = active_lev
        else:
            eq[i] = eq[i-1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        if in_trade:
            don_exit  = c[i] < dl[i]
            stop_hit  = c[i] < stop
            if don_exit or stop_hit or not above_sma:
                in_trade  = False
                units     = 0
                total_lev = 0.0
            elif units < MAX_UNITS and atr[i] > 0 and c[i] >= last_add + ADD_INTERVAL * atr[i]:
                atr_pct   = atr[i] / c[i]
                unit_lev  = min(risk / atr_pct, max_lev - total_lev)
                unit_lev  = max(unit_lev, 0.0)
                total_lev = min(total_lev + unit_lev, max_lev)
                avg_entry = (avg_entry * (units / (units + 1)) +
                             c[i] * (1.0 / (units + 1)))
                stop      = avg_entry - ATR_STOP_MULT * atr[i]
                last_add  = c[i]
                units    += 1
        elif atr[i] > 0 and c[i] > dh[i] and above_sma:
            atr_pct   = atr[i] / c[i]
            unit_lev  = min(risk / atr_pct, max_lev)
            total_lev = unit_lev
            avg_entry = c[i]
            stop      = c[i] - ATR_STOP_MULT * atr[i]
            last_add  = c[i]
            units     = 1
            in_trade  = True

    return pd.Series(eq, index=qqq.index), pd.Series(lev_log, index=qqq.index)


# ── Strategy 4: ATR-sized pyramid + vol gate ─────────────────────────────────

def atr_pyramid_vol_gate(ind: dict, max_lev_normal: float = MAX_LEV,
                          max_lev_highvol: float = 1.0,
                          risk: float = RISK_PER_UNIT) -> tuple[pd.Series, pd.Series]:
    """
    Same as pyramid but caps leverage at 1x when realised 20-day vol > VOL_GATE_ANN.
    This keeps position size conservative during volatile regimes.
    """
    c        = ind["close"]
    r        = ind["returns"]
    atr      = ind["atr"]
    sma      = ind["sma200"]
    dh       = ind["don_high"]
    dl       = ind["don_low"]
    vol_ann  = ind["vol_ann"]
    warmup   = max(ENTRY_PERIOD, SMA_PERIOD, ATR_PERIOD)

    eq        = np.empty(n)
    eq[0]     = CAPITAL
    lev_log   = np.zeros(n)
    in_trade  = False
    units     = 0
    total_lev = 0.0
    last_add  = 0.0
    avg_entry = 0.0
    stop      = 0.0

    for i in range(1, n):
        active_lev = total_lev if in_trade else 0.0
        if active_lev > 0:
            cost  = (active_lev - 1) * BORROW_RATE / 252 if active_lev > 1 else 0.0
            eq[i] = eq[i-1] * max(0.0, 1 + active_lev * r[i] - cost)
            lev_log[i] = active_lev
        else:
            eq[i] = eq[i-1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        hi_vol    = vol_ann[i] > VOL_GATE_ANN
        cur_max   = max_lev_highvol if hi_vol else max_lev_normal

        if in_trade:
            don_exit = c[i] < dl[i]
            stop_hit = c[i] < stop
            if don_exit or stop_hit or not above_sma:
                in_trade  = False
                units     = 0
                total_lev = 0.0
            else:
                if hi_vol and total_lev > max_lev_highvol:
                    total_lev = max_lev_highvol

                if (units < MAX_UNITS and atr[i] > 0 and
                        c[i] >= last_add + ADD_INTERVAL * atr[i] and not hi_vol):
                    atr_pct   = atr[i] / c[i]
                    unit_lev  = min(risk / atr_pct, cur_max - total_lev)
                    unit_lev  = max(unit_lev, 0.0)
                    total_lev = min(total_lev + unit_lev, cur_max)
                    avg_entry = (avg_entry * (units / (units + 1)) +
                                 c[i] * (1.0 / (units + 1)))
                    stop      = avg_entry - ATR_STOP_MULT * atr[i]
                    last_add  = c[i]
                    units    += 1
        elif atr[i] > 0 and c[i] > dh[i] and above_sma:
            atr_pct   = atr[i] / c[i]
            unit_lev  = min(risk / atr_pct, cur_max)
            total_lev = unit_lev
            avg_entry = c[i]
            stop      = c[i] - ATR_STOP_MULT * atr[i]
            last_add  = c[i]
            units     = 1
            in_trade  = True

    return pd.Series(eq, index=qqq.index), pd.Series(lev_log, index=qqq.index)


def fixed_lev_don_sma_regime(ind: dict, lev: float = 3.0) -> pd.Series:
    """Flat leverage with mirrored long/short Donchian regimes, applied next bar."""
    c = ind["close"]
    r = ind["returns"]
    sma = ind["sma200"]
    entry_high = ind["entry_high"]
    entry_low = ind["entry_low"]
    exit_high = ind["exit_high"]
    exit_low = ind["exit_low"]
    warmup = max(ENTRY_PERIOD, SMA_PERIOD)

    eq = np.empty(n)
    eq[0] = CAPITAL
    direction = 0

    for i in range(1, n):
        signed_lev = direction * lev
        if signed_lev:
            cost = (lev - 1) * BORROW_RATE / 252 if lev > 1 else 0.0
            eq[i] = eq[i - 1] * max(0.0, 1 + signed_lev * r[i] - cost)
        else:
            eq[i] = eq[i - 1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        below_sma = c[i] < sma[i]
        if direction == 1:
            if c[i] < exit_low[i] or not above_sma:
                direction = 0
        elif direction == -1:
            if c[i] > exit_high[i] or not below_sma:
                direction = 0
        else:
            if c[i] > entry_high[i] and above_sma:
                direction = 1
            elif c[i] < entry_low[i] and below_sma:
                direction = -1

    return pd.Series(eq, index=qqq.index)


def atr_regime_pyramid(ind: dict, max_lev: float = MAX_LEV,
                       risk: float = RISK_PER_UNIT,
                       use_vol_gate: bool = False,
                       max_lev_highvol: float = 1.0,
                       short_slope_window: int = 0) -> tuple[pd.Series, pd.Series]:
    """ATR-sized long/short Turtle with pyramiding, vol gating, and optional SMA slope filter."""
    c = ind["close"]
    r = ind["returns"]
    atr = ind["atr"]
    sma = ind["sma200"]
    entry_high = ind["entry_high"]
    entry_low = ind["entry_low"]
    exit_high = ind["exit_high"]
    exit_low = ind["exit_low"]
    vol_ann = ind["vol_ann"]
    warmup = max(ENTRY_PERIOD, SMA_PERIOD, ATR_PERIOD)

    eq = np.empty(n)
    eq[0] = CAPITAL
    lev_log = np.zeros(n)
    direction = 0
    units = 0
    gross_lev = 0.0
    last_add = 0.0
    avg_entry = 0.0
    stop = 0.0

    for i in range(1, n):
        signed_lev = direction * gross_lev
        if signed_lev:
            cost = (gross_lev - 1) * BORROW_RATE / 252 if gross_lev > 1 else 0.0
            eq[i] = eq[i - 1] * max(0.0, 1 + signed_lev * r[i] - cost)
            lev_log[i] = signed_lev
        else:
            eq[i] = eq[i - 1]

        if eq[i] <= 0:
            eq[i:] = 0.0
            break

        if i < warmup:
            continue

        above_sma = c[i] > sma[i]
        below_sma = c[i] < sma[i]
        short_trend_ok = (
            short_slope_window <= 0 or
            (i >= short_slope_window and sma[i] < sma[i - short_slope_window])
        )
        hi_vol = use_vol_gate and vol_ann[i] > VOL_GATE_ANN
        cur_max = max_lev_highvol if hi_vol else max_lev

        if direction:
            if hi_vol and gross_lev > cur_max:
                gross_lev = cur_max

            if direction == 1:
                exit_signal = c[i] < exit_low[i] or c[i] < stop or not above_sma
                add_signal = (
                    units < MAX_UNITS and atr[i] > 0 and not hi_vol and
                    c[i] >= last_add + ADD_INTERVAL * atr[i]
                )
            else:
                exit_signal = (
                    c[i] > exit_high[i] or c[i] > stop or
                    not below_sma or not short_trend_ok
                )
                add_signal = (
                    units < MAX_UNITS and atr[i] > 0 and not hi_vol and
                    c[i] <= last_add - ADD_INTERVAL * atr[i]
                )

            if exit_signal:
                direction = 0
                units = 0
                gross_lev = 0.0
                last_add = 0.0
                avg_entry = 0.0
                stop = 0.0
            elif add_signal:
                atr_pct = atr[i] / c[i]
                unit_lev = min(risk / atr_pct, cur_max - gross_lev)
                unit_lev = max(unit_lev, 0.0)
                if unit_lev > 0:
                    gross_lev = min(gross_lev + unit_lev, cur_max)
                    avg_entry = (avg_entry * units + c[i]) / (units + 1)
                    stop = (
                        avg_entry - ATR_STOP_MULT * atr[i]
                        if direction == 1 else
                        avg_entry + ATR_STOP_MULT * atr[i]
                    )
                    last_add = c[i]
                    units += 1
        elif atr[i] > 0:
            atr_pct = atr[i] / c[i]
            unit_lev = min(risk / atr_pct, cur_max)
            if c[i] > entry_high[i] and above_sma:
                direction = 1
                units = 1
                gross_lev = unit_lev
                last_add = c[i]
                avg_entry = c[i]
                stop = c[i] - ATR_STOP_MULT * atr[i]
            elif c[i] < entry_low[i] and below_sma and short_trend_ok:
                direction = -1
                units = 1
                gross_lev = unit_lev
                last_add = c[i]
                avg_entry = c[i]
                stop = c[i] + ATR_STOP_MULT * atr[i]

    return pd.Series(eq, index=qqq.index), pd.Series(lev_log, index=qqq.index)


# ── run all strategies ────────────────────────────────────────────────────────

print("Running strategies …")

results: dict[str, pd.Series] = {}
lev_series: dict[str, pd.Series] = {}

# reference baselines
results["B&H QQQ 1x"]            = (qqq["close"] / qqq["close"].iloc[0] * CAPITAL)
results["Fixed 3x Don+SMA QQQ"]  = fixed_lev_don_sma(Q, lev=3.0)
results["Fixed 3x Don65/25+SMA225 QQQ"] = fixed_lev_don_sma(
    Q, lev=3.0, entry_period=65, exit_period=25, sma_period=225
)
results["Fixed 3x Don+SMA QQQ Ratchet 3%"] = fixed_lev_don_sma_profit_ratchet(Q, lev=3.0, profit_step=0.03)
results["Fixed 3x Don+SMA QQQ Ratchet 2%"] = fixed_lev_don_sma_profit_ratchet(Q, lev=3.0, profit_step=0.02)
results["Fixed 3x Don+SMA SPY"]  = fixed_lev_don_sma(S, lev=3.0)

# ATR-sized strategies on QQQ
eq2q, lv2q = atr_sized_no_pyramid(Q)
results["ATR-sized 3x QQQ"]         = eq2q
lev_series["ATR-sized 3x QQQ"]      = lv2q

eq3q, lv3q = atr_sized_pyramid(Q)
results["ATR-pyramid 3x QQQ"]        = eq3q
lev_series["ATR-pyramid 3x QQQ"]     = lv3q

eq4q, lv4q = atr_pyramid_vol_gate(Q)
results["ATR-pyramid+VolGate QQQ"]   = eq4q
lev_series["ATR-pyramid+VolGate QQQ"]= lv4q

# ATR-sized strategies on SPY
eq2s, lv2s = atr_sized_no_pyramid(S)
results["ATR-sized 3x SPY"]          = eq2s
lev_series["ATR-sized 3x SPY"]       = lv2s

eq3s, lv3s = atr_sized_pyramid(S)
results["ATR-pyramid 3x SPY"]        = eq3s
lev_series["ATR-pyramid 3x SPY"]     = lv3s

eq4s, lv4s = atr_pyramid_vol_gate(S)
results["ATR-pyramid+VolGate SPY"]   = eq4s
lev_series["ATR-pyramid+VolGate SPY"]= lv4s

# Long/short regime strategies on QQQ
results["Fixed 3x Regime L/S QQQ"] = fixed_lev_don_sma_regime(Q, lev=3.0)

eq5q, lv5q = atr_regime_pyramid(Q)
results["ATR-pyramid 3x Regime L/S QQQ"] = eq5q
lev_series["ATR-pyramid 3x Regime L/S QQQ"] = lv5q

eq6q, lv6q = atr_regime_pyramid(Q, use_vol_gate=True)
results["ATR-pyramid+VolGate Regime L/S QQQ"] = eq6q
lev_series["ATR-pyramid+VolGate Regime L/S QQQ"] = lv6q

eq7q, lv7q = atr_regime_pyramid(Q, use_vol_gate=True, short_slope_window=40)
results["ATR-pyramid+VolGate+Slope40 L/S QQQ"] = eq7q
lev_series["ATR-pyramid+VolGate+Slope40 L/S QQQ"] = lv7q

# ── metrics ───────────────────────────────────────────────────────────────────

rows = [_metrics(v.rename(k), k) for k, v in results.items()]

# append avg leverage where available
for row in rows:
    name = row["Strategy"]
    if name in lev_series:
        lv = lev_series[name]
        active = lv.abs() > 0
        in_mkt = active.mean() * 100
        avg_lv = lv[active].abs().mean() if active.any() else 0.0
        row["Avg Lev"]    = f"{avg_lv:.2f}x"
        row["% in Mkt"]   = f"{in_mkt:.0f}%"
    else:
        row["Avg Lev"]  = "-"
        row["% in Mkt"] = "-"

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT / "metrics.csv", index=False)
print("\n" + metrics_df.to_string(index=False))

# ── Figure: equity curves + drawdown + leverage + table ──────────────────────

COLORS = {
    "B&H QQQ 1x":                "#90A4AE",
    "Fixed 3x Don+SMA QQQ":      "#1565C0",
    "Fixed 3x Don65/25+SMA225 QQQ": "#1E88E5",
    "Fixed 3x Don+SMA QQQ Ratchet 3%": "#3949AB",
    "Fixed 3x Don+SMA QQQ Ratchet 2%": "#5C6BC0",
    "Fixed 3x Don+SMA SPY":      "#42A5F5",
    "ATR-sized 3x QQQ":          "#2E7D32",
    "ATR-pyramid 3x QQQ":        "#F57F17",
    "ATR-pyramid+VolGate QQQ":   "#AD1457",
    "ATR-sized 3x SPY":          "#558B2F",
    "ATR-pyramid 3x SPY":        "#FF8F00",
    "ATR-pyramid+VolGate SPY":   "#880E4F",
    "Fixed 3x Regime L/S QQQ":   "#6D4C41",
    "ATR-pyramid 3x Regime L/S QQQ": "#00897B",
    "ATR-pyramid+VolGate Regime L/S QQQ": "#00695C",
    "ATR-pyramid+VolGate+Slope40 L/S QQQ": "#004D40",
}
DASH = {k: "--" if "SPY" in k else "-" for k in COLORS}

fig = plt.figure(figsize=(20, 20))
fig.suptitle(
    "Turtle ATR Leverage With Regime Shorts  (SMA-200 gate, Donchian 55/20, max 3x, 5.5% borrow)",
    fontsize=13, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.42,
                       height_ratios=[2.5, 1.5, 1.2, 1.8])
ax_eq  = fig.add_subplot(gs[0])
ax_dd  = fig.add_subplot(gs[1])
ax_lv  = fig.add_subplot(gs[2])
ax_tb  = fig.add_subplot(gs[3])
ax_tb.axis("off")

for label, eq in results.items():
    eq = eq.ffill()
    if eq.iloc[-1] <= 0:
        continue
    norm = eq / eq.iloc[0]
    ax_eq.plot(eq.index, norm,
               color=COLORS.get(label, "grey"),
               linestyle=DASH.get(label, "-"),
               linewidth=1.6, label=label)

ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
ax_eq.set_ylabel("Normalised Value")
ax_eq.set_title("Equity Curves")
ax_eq.legend(fontsize=8, ncol=2, loc="upper left")
ax_eq.grid(alpha=0.25)

for label, eq in results.items():
    eq = eq.ffill()
    if eq.iloc[-1] <= 0:
        continue
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax_dd.plot(eq.index, dd,
               color=COLORS.get(label, "grey"),
               linestyle=DASH.get(label, "-"),
               linewidth=1.2, label=label)

ax_dd.axhline(0, color="black", linewidth=0.6)
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=8, ncol=2, loc="lower left")
ax_dd.grid(alpha=0.25)

# leverage over time for QQQ variants
for label, lv in lev_series.items():
    if "QQQ" not in label:
        continue
    ax_lv.fill_between(lv.index, 0, lv.values,
                        color=COLORS.get(label, "grey"), alpha=0.35, label=label)
    ax_lv.plot(lv.index, lv.values,
               color=COLORS.get(label, "grey"), linewidth=0.8)

ax_lv.axhline(MAX_LEV, color="red", linewidth=0.8, linestyle=":", label=f"+{MAX_LEV}x cap")
ax_lv.axhline(-MAX_LEV, color="red", linewidth=0.8, linestyle=":")
ax_lv.axhline(0, color="black", linewidth=0.6)
ax_lv.set_ylabel("Signed Leverage")
ax_lv.set_title("Leverage Over Time  (QQQ strategies, shorts plot below zero)")
ax_lv.legend(fontsize=8, ncol=2)
ax_lv.grid(alpha=0.25)

# table
col_labels = list(metrics_df.columns)
cell_text  = metrics_df.values.tolist()
tbl = ax_tb.table(cellText=cell_text, colLabels=col_labels,
                  cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.35)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#263238")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(cell_text) + 1):
    for j in range(len(col_labels)):
        clr = "#FFEBEE" if "BLOWN" in str(cell_text[i-1][j]) else \
              ("#ECEFF1" if i % 2 == 0 else "white")
        tbl[i, j].set_facecolor(clr)

fig.savefig(OUT / "turtle_atr_leverage.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT / 'turtle_atr_leverage.png'}")
plt.show()
