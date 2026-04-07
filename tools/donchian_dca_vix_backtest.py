"""
tools/donchian_dca_vix_backtest.py

Pure MA + volume strategy. No Donchians.

Entry  : price crosses above SMA-65
Exit   : price crosses below SMA-65
         OR price crosses below SMA-215 on HIGH volume (>1.5× 20d avg vol)

Volume-gated DCA (monthly, first trading day)
  price > SMA-65                        →  $500   (normal, in bull regime)
  SMA-215 < price ≤ SMA-65             →  $250   (cautious, watching)
  price < SMA-215, vol ≤ vol MA        →  $1000  (capitulation dip, buy more)
  price < SMA-215, vol > 1.5× vol MA  →  $0     (high-vol breakdown, wait)

Position : 80% of NAV deployed, 20% cash at risk-free rate
Leverage : x3, borrow cost 5.5% annual

Benchmarks
  1. Strategy
  2. Pure DCA QQQ x1 (flat $500/month)

Output
------
  reports/donchian_dca_vix/ma_volume_backtest.png
  reports/donchian_dca_vix/metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pyarrow.parquet as pq

# ── config ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]
TIINGO = ROOT / "cache" / "cache" / "tiingo"
OUT    = ROOT / "reports" / "donchian_dca_vix"
OUT.mkdir(parents=True, exist_ok=True)

SMA_REGIME     = 65     # above = long, cross above = entry, cross below = exit
SMA_TREND      = 215    # vol-gated hard exit + DCA zone boundary
VOL_MA_PERIOD  = 20
VOL_HIGH_MULT  = 1.5
LEVERAGE       = 3.0
DEPLOY         = 0.80
RISK_FREE_RATE = 0.04
BORROW_RATE    = 0.055

DCA_NORMAL   = 500.0
DCA_CAUTIOUS = 250.0
DCA_DIP      = 1000.0
DCA_NONE     = 0.0

WARMUP = SMA_TREND   # need SMA-215 to be valid before we start

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_daily(symbol: str) -> pd.DataFrame:
    tbl = pq.read_table(TIINGO / f"{symbol}_5m.parquet",
                        columns=["time", "o", "h", "l", "c", "v"])
    df = tbl.to_pandas()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.tz_convert("America/New_York").dt.date
    daily = (
        df.groupby("date")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"),    close=("close", "last"),
             volume=("volume", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1 : i + 1].mean()
    return out


def _cagr(final: float, invested: float, years: float) -> float:
    if years <= 0 or invested <= 0:
        return 0.0
    return (final / invested) ** (1.0 / years) - 1.0


def _max_drawdown(eq: np.ndarray) -> float:
    pos = eq[eq > 0]
    if len(pos) == 0:
        return 0.0
    peak = np.maximum.accumulate(pos)
    return float(((pos - peak) / peak).min()) * 100


# ── load & indicators ─────────────────────────────────────────────────────────

print("Loading QQQ …")
df = _load_daily("QQQ")
print(f"Data: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}  ({len(df)} days)")

c   = df["close"].to_numpy(float)
vol = df["volume"].to_numpy(float)
n   = len(df)

sma_reg   = _sma(c,   SMA_REGIME)
sma_trend = _sma(c,   SMA_TREND)
vol_ma    = _sma(vol, VOL_MA_PERIOD)

daily_ret    = np.zeros(n)
daily_ret[1:] = (c[1:] - c[:-1]) / c[:-1]
daily_rf     = RISK_FREE_RATE / 252.0
daily_borrow = BORROW_RATE    / 252.0

# ── simulation ────────────────────────────────────────────────────────────────

pos    = 0
lev_eq = 0.0
cash   = 0.0
bh_shares = 0.0

total_contributions = 0.0
last_dca_month      = -1
long_trades         = 0
vol_exits           = 0
total_bars = long_bars = 0
dca_normal_total = dca_cautious_total = dca_dip_total = dca_skipped_total = 0.0

strat_val  = np.zeros(n)
bh_val     = np.zeros(n)
contrib_arr= np.zeros(n)
pos_arr    = np.zeros(n, dtype=int)
vol_zone   = np.zeros(n, dtype=int)   # 0=normal 1=cautious 2=dip 3=knife

for i in range(n):
    date = df["date"].iloc[i]

    # ── classify volume/price zone ────────────────────────────────────────────
    if i >= WARMUP and not np.isnan(sma_reg[i]) and not np.isnan(sma_trend[i]):
        high_vol     = vol[i] > VOL_HIGH_MULT * vol_ma[i]
        above_regime = c[i] > sma_reg[i]
        above_trend  = c[i] > sma_trend[i]
        if above_regime:
            vol_zone[i] = 0
        elif above_trend:
            vol_zone[i] = 1
        elif not high_vol:
            vol_zone[i] = 2
        else:
            vol_zone[i] = 3

    # ── DCA ───────────────────────────────────────────────────────────────────
    if date.month != last_dca_month and i >= WARMUP:
        last_dca_month = date.month
        zone = vol_zone[i]
        dca  = [DCA_NORMAL, DCA_CAUTIOUS, DCA_DIP, DCA_NONE][zone]

        total_contributions += dca
        if dca > 0:
            lev_eq += dca if pos == 1 else 0
            cash   += dca if pos != 1 else 0

        [dca_normal_total, dca_cautious_total,
         dca_dip_total,    dca_skipped_total][zone]  # just for clarity
        if zone == 0: dca_normal_total   += dca
        elif zone==1: dca_cautious_total += dca
        elif zone==2: dca_dip_total      += dca
        else:         dca_skipped_total  += DCA_NORMAL

        bh_shares += DCA_NORMAL / c[i]

    # ── apply returns ─────────────────────────────────────────────────────────
    if i > WARMUP:
        if pos == 1 and lev_eq > 0:
            lev_eq *= (1.0 + daily_ret[i] * LEVERAGE)
            lev_eq -= daily_borrow * (LEVERAGE - 1.0) * lev_eq
        if cash > 0:
            cash *= (1.0 + daily_rf)

    # ── exits ─────────────────────────────────────────────────────────────────
    if i >= WARMUP and pos == 1:
        # 1. price crosses below SMA-65
        sma_exit = c[i] < sma_reg[i] and c[i-1] >= sma_reg[i-1]
        # 2. high-conviction breakdown below SMA-215
        vol_exit = (
            not np.isnan(sma_trend[i]) and not np.isnan(sma_trend[i-1])
            and c[i] < sma_trend[i] and c[i-1] >= sma_trend[i-1]
            and vol[i] > VOL_HIGH_MULT * vol_ma[i]
        )
        if vol_exit:
            vol_exits += 1
        if sma_exit or vol_exit:
            cash  += lev_eq
            lev_eq = 0.0
            pos    = 0

    # ── entry ─────────────────────────────────────────────────────────────────
    if i >= WARMUP and pos == 0 and not np.isnan(sma_reg[i]):
        # price crosses above SMA-65
        if c[i] > sma_reg[i] and c[i-1] <= sma_reg[i-1]:
            nav    = lev_eq + cash
            lev_eq = nav * DEPLOY
            cash   = nav * (1.0 - DEPLOY)
            pos    = 1
            long_trades += 1

    # ── record ────────────────────────────────────────────────────────────────
    strat_val[i]   = lev_eq + cash if i >= WARMUP else 0.0
    bh_val[i]      = bh_shares * c[i]
    contrib_arr[i] = total_contributions
    pos_arr[i]     = pos
    if i >= WARMUP:
        total_bars += 1
        if pos == 1:
            long_bars += 1

# ── trim ──────────────────────────────────────────────────────────────────────

sl         = slice(WARMUP, None)
dates      = df["date"].iloc[WARMUP:].to_numpy()
strat_v    = strat_val[sl]
bh_v       = bh_val[sl]
cont_v     = contrib_arr[sl]
pos_v      = pos_arr[sl]
vol_zone_v = vol_zone[sl]
c_v        = c[sl]
sma_reg_v  = sma_reg[sl]
sma_trend_v= sma_trend[sl]
vol_v      = vol[sl]
vol_ma_v   = vol_ma[sl]

years     = total_bars / 252.0
time_long = long_bars / total_bars * 100 if total_bars else 0
time_flat = 100 - time_long

bh_inv = sum(DCA_NORMAL for i in range(WARMUP, n)
             if df["date"].iloc[i].month != df["date"].iloc[i-1].month)

# ── metrics ───────────────────────────────────────────────────────────────────

def _m(vals, inv, label):
    final  = float(vals[-1])
    profit = final - inv
    return dict(label=label, final=final, invested=inv, profit=profit,
                ret=profit / inv * 100 if inv else 0,
                cagr=_cagr(final, inv, years) * 100,
                mdd=_max_drawdown(vals))

ms = _m(strat_v, total_contributions, "Strategy")
mb = _m(bh_v,    bh_inv,              "DCA QQQ x1")

for m in [ms, mb]:
    print(f"\n{m['label']}")
    print(f"  Final     : ${m['final']:>10,.0f}")
    print(f"  Invested  : ${m['invested']:>10,.0f}")
    print(f"  Profit    : ${m['profit']:>10,.0f}")
    print(f"  Return    : {m['ret']:>7.1f}%")
    print(f"  CAGR      : {m['cagr']:>7.1f}%")
    print(f"  Max DD    : {m['mdd']:>7.1f}%")

print(f"\nLong trades   : {long_trades}  ({time_long:.1f}% of time)")
print(f"Vol exits     : {vol_exits}")
print(f"Cash (flat)   : {time_flat:.1f}% of time")
print(f"Years active  : {years:.1f}")
print(f"\nDCA breakdown:")
print(f"  Normal   $500  : ${dca_normal_total:,.0f}")
print(f"  Cautious $250  : ${dca_cautious_total:,.0f}")
print(f"  Dip     $1000  : ${dca_dip_total:,.0f}")
print(f"  Skipped        : ${dca_skipped_total:,.0f} withheld")

pd.DataFrame([ms, mb]).to_csv(OUT / "metrics.csv", index=False)

# ── plot ──────────────────────────────────────────────────────────────────────

BG    = "#0d1117"
PANEL = "#111827"
GREEN = "#00e676"
RED   = "#ef5350"
GRAY  = "#90a4ae"
AMBER = "#ffb300"
BLUE  = "#42a5f5"
PINK  = "#f48fb1"
DIM   = "#546e7a"

fig = plt.figure(figsize=(16, 14), facecolor=BG)
gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0.8],
                        hspace=0.06, top=0.93, bottom=0.06,
                        left=0.07, right=0.95)

date_nums = mdates.date2num(pd.to_datetime(dates))

def _style(ax, ylabel, fontsize=9):
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, color="white", fontsize=fontsize)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(alpha=0.07, color="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a3a")
    ax.xaxis_date()

def _shade_long(ax):
    mask  = pos_v == 1
    trans = np.diff(mask.astype(int), prepend=0)
    s = date_nums[trans == 1]
    e = date_nums[trans == -1]
    if mask[-1]: e = np.append(e, date_nums[-1])
    for a, b in zip(s, e):
        ax.axvspan(a, b, alpha=0.08, color=GREEN, zorder=0)

def _shade_zones(ax):
    for zone, col in [(1, AMBER), (2, GREEN), (3, RED)]:
        mask  = vol_zone_v == zone
        trans = np.diff(mask.astype(int), prepend=0)
        s = date_nums[trans == 1]
        e = date_nums[trans == -1]
        if mask[-1]: e = np.append(e, date_nums[-1])
        for a, b in zip(s, e):
            ax.axvspan(a, b, alpha=0.07, color=col, zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 – Portfolio value
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
_style(ax1, "Portfolio Value ($)", fontsize=10)
_shade_long(ax1)

ax1.plot(date_nums, strat_v, color=GREEN, lw=2.2,
         label=f"Strategy  →  ${strat_v[-1]:,.0f}")
ax1.plot(date_nums, bh_v,   color=GRAY,  lw=1.5,
         label=f"DCA QQQ   →  ${bh_v[-1]:,.0f}")
ax1.plot(date_nums, cont_v, color=DIM,   lw=0.9, ls="--",
         label=f"Contributed  →  ${cont_v[-1]:,.0f}")

ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.legend(loc="upper left", fontsize=9, framealpha=0.3,
           labelcolor="white", facecolor="#1a1a2e", edgecolor="#333")

stats = (
    f"{'':18} {'Strategy':>12}  {'DCA QQQ x1':>11}\n"
    f"{'─'*45}\n"
    f"{'Final value':18} ${ms['final']:>11,.0f}  ${mb['final']:>10,.0f}\n"
    f"{'Invested':18} ${ms['invested']:>11,.0f}  ${mb['invested']:>10,.0f}\n"
    f"{'Profit':18} ${ms['profit']:>11,.0f}  ${mb['profit']:>10,.0f}\n"
    f"{'Total return':18} {ms['ret']:>10.1f}%  {mb['ret']:>10.1f}%\n"
    f"{'CAGR':18} {ms['cagr']:>10.1f}%  {mb['cagr']:>10.1f}%\n"
    f"{'Max drawdown':18} {ms['mdd']:>10.1f}%  {mb['mdd']:>10.1f}%\n"
    f"{'Long trades':18} {long_trades:>12}  {'∞ (DCA)':>11}\n"
    f"{'Vol exits':18} {vol_exits:>12}\n"
    f"{'Time long':18} {time_long:>9.1f}%\n"
    f"{'DCA $500 normal':18} ${dca_normal_total:>11,.0f}\n"
    f"{'DCA $250 cautious':18} ${dca_cautious_total:>11,.0f}\n"
    f"{'DCA $1000 dip':18} ${dca_dip_total:>11,.0f}\n"
    f"{'DCA skipped':18} ${dca_skipped_total:>11,.0f}\n"
    f"{'Years':18} {years:>10.1f}"
)
ax1.text(0.99, 0.97, stats, transform=ax1.transAxes,
         family="monospace", fontsize=7.5, color="white", va="top", ha="right",
         bbox=dict(facecolor="#0d1117", alpha=0.78, edgecolor="#2a2a3a",
                   boxstyle="round,pad=0.45"))

ax1.set_title(
    f"SMA-{SMA_REGIME} entry/exit  ·  SMA-{SMA_TREND} vol-exit  ·  "
    f"volume-gated DCA  ·  x{LEVERAGE:.0f} lev  ·  {int(DEPLOY*100)}% deploy",
    color="white", fontsize=10, pad=8, loc="left")
plt.setp(ax1.get_xticklabels(), visible=False)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 – QQQ price + SMAs + zone backgrounds + signals
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
_style(ax2, "QQQ Price ($)", fontsize=10)
_shade_zones(ax2)
_shade_long(ax2)

# SMA fills
ax2.fill_between(date_nums, sma_reg_v, sma_trend_v,
                 where=~(np.isnan(sma_reg_v) | np.isnan(sma_trend_v)),
                 color=BLUE, alpha=0.05, zorder=1)

# SMA lines
ax2.plot(date_nums, sma_reg_v,   color=BLUE, lw=2.2, zorder=3,
         label=f"SMA {SMA_REGIME}  — entry/exit signal")
ax2.plot(date_nums, sma_trend_v, color=PINK, lw=1.8, zorder=3,
         label=f"SMA {SMA_TREND} — vol-exit + DCA zones")

# QQQ price
ax2.plot(date_nums, c_v, color="#cfd8dc", lw=1.0, alpha=0.9, zorder=4,
         label="QQQ close")

# Entry / exit markers
entry_m = np.diff((pos_v == 1).astype(int), prepend=0) == 1
exit_m  = np.diff((pos_v == 1).astype(int), prepend=0) == -1
ax2.scatter(date_nums[entry_m], c_v[entry_m] * 0.982, marker="^",
            color=GREEN, s=90, zorder=6, label=f"Entry  ({entry_m.sum()})")
ax2.scatter(date_nums[exit_m],  c_v[exit_m]  * 1.018, marker="v",
            color=RED,   s=90, zorder=6, label=f"Exit   ({exit_m.sum()})")

# zone patches for legend
zone_patches = [
    mpatches.Patch(color=GREEN,  alpha=0.4, label="Long (above SMA-65)"),
    mpatches.Patch(color=AMBER,  alpha=0.4, label="Cautious: SMA-65→215  ($250 DCA)"),
    mpatches.Patch(color=GREEN,  alpha=0.6, label="Dip: below SMA-215, low vol  ($1000 DCA)"),
    mpatches.Patch(color=RED,    alpha=0.4, label="Knife: below SMA-215, high vol  ($0 DCA)"),
]
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles + zone_patches, labels + [p.get_label() for p in zone_patches],
           loc="upper left", fontsize=7.5, framealpha=0.35,
           labelcolor="white", facecolor="#1a1a2e", edgecolor="#333", ncol=2)

ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

for val, lbl, col in [
    (sma_reg_v[-1],   f"SMA{SMA_REGIME}  ${sma_reg_v[-1]:.0f}",   BLUE),
    (sma_trend_v[-1], f"SMA{SMA_TREND} ${sma_trend_v[-1]:.0f}",   PINK),
]:
    if not np.isnan(val):
        ax2.annotate(lbl, xy=(date_nums[-1], val),
                     xytext=(5, 0), textcoords="offset points",
                     color=col, fontsize=8, va="center",
                     annotation_clip=False)

plt.setp(ax2.get_xticklabels(), visible=False)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 – Volume
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
_style(ax3, "Volume", fontsize=9)

bar_colors = np.where(
    vol_zone_v == 3, RED,
    np.where(vol_zone_v == 2, GREEN,
    np.where(vol_zone_v == 1, AMBER, GRAY))
)
ax3.bar(date_nums, vol_v / 1e6, width=0.6, color=bar_colors, alpha=0.7, zorder=2)
ax3.plot(date_nums, vol_ma_v / 1e6,                   color="white", lw=1.2,
         label=f"Vol MA {VOL_MA_PERIOD}d")
ax3.plot(date_nums, vol_ma_v * VOL_HIGH_MULT / 1e6,   color=RED, lw=1.0, ls="--",
         label=f"{VOL_HIGH_MULT}× vol MA  (high-vol threshold)")

ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.tick_params(axis="x", rotation=25, labelsize=7.5)
ax3.legend(loc="upper right", fontsize=7.5, framealpha=0.3,
           labelcolor="white", facecolor="#1a1a2e", edgecolor="#333")
ax3.text(0.01, 0.93,
         "Gray = normal  ·  Amber = cautious  ·  Green = dip buy  ·  Red = high-vol knife",
         transform=ax3.transAxes, fontsize=7, color="white", va="top", alpha=0.8)

out_path = OUT / "ma_volume_backtest.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nChart → {out_path}")
