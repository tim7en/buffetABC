"""
tools/leverage_comparison.py

Explores leverage (1x–10x) applied to the best QQQ/SPY strategies.
Models daily-rebalanced leverage with realistic margin borrowing costs.

Key mechanics modelled
----------------------
* Daily P&L:  equity *= (1 + L*r - (L-1)*borrow/252)
* Margin call: equity <= 0  →  strategy blows up, final value = $0
* Volatility drag:  CAGR peaks at Kelly leverage  L* = (mu - borrow/252) / sigma²
  Beyond L*, higher leverage destroys more via drag than it adds via return.

Strategies compared
-------------------
A. QQQ Buy & Hold at fixed leverage 1x … 10x  (leverage scan)
B. SPY Buy & Hold at fixed leverage 1x … 10x
C. Donchian 55/20 + SMA-200 QQQ at fixed leverage 1x … 5x
D. Volatility-targeted (target 15 % / 20 % / 25 % ann. vol) on QQQ
E. Half-Kelly dynamic leverage on QQQ

Outputs
-------
  reports/leverage_comparison/leverage_scan.png
  reports/leverage_comparison/equity_curves.png
  reports/leverage_comparison/leverage_metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq

# ── config ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
TIINGO  = ROOT / "cache" / "cache" / "tiingo"
OUT     = ROOT / "reports" / "leverage_comparison"
OUT.mkdir(parents=True, exist_ok=True)

CAPITAL      = 10_000.0
BORROW_RATE  = 0.055          # 5.5 % annual margin cost on borrowed capital
VOL_WINDOW   = 20             # days for realized-vol estimate
KELLY_WINDOW = 60             # days for rolling Kelly estimate
MAX_LEV      = 10.0

# ── data loading ──────────────────────────────────────────────────────────────

def _load_daily(symbol: str) -> pd.DataFrame:
    path = TIINGO / f"{symbol}_5m.parquet"
    tbl  = pq.read_table(path, columns=["time", "o", "h", "l", "c", "v"])
    df   = tbl.to_pandas()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"]  = pd.to_datetime(df["time"], utc=True)
    df["date"]  = df["time"].dt.tz_convert("America/New_York").dt.date
    daily = (
        df.groupby("date")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


print("Loading data …")
qqq = _load_daily("QQQ")
spy = _load_daily("SPY")
common = sorted(set(qqq["date"]) & set(spy["date"]))
qqq = qqq[qqq["date"].isin(common)].set_index("date").sort_index()
spy = spy[spy["date"].isin(common)].set_index("date").sort_index()
n   = len(qqq)
print(f"  {n} daily bars  {qqq.index[0].date()} to {qqq.index[-1].date()}")

qqq_r = qqq["close"].pct_change().fillna(0).values
spy_r = spy["close"].pct_change().fillna(0).values
qqq_c = qqq["close"].values
spy_c = spy["close"].values

# ── core helpers ──────────────────────────────────────────────────────────────

def _metrics(eq: pd.Series, label: str) -> dict:
    eq = eq.ffill().dropna()
    if len(eq) < 2 or eq.iloc[-1] <= 0:
        blown = eq.iloc[-1] <= 0
        return {"Strategy": label, "Total Return": "BLOWN UP" if blown else "n/a",
                "CAGR": "-", "Max DD": "-", "Sharpe": "-",
                "Final $": "$0" if blown else "-", "Blown Up": blown}
    years     = (eq.index[-1] - eq.index[0]).days / 365.25
    tot_r     = eq.iloc[-1] / eq.iloc[0] - 1
    cagr      = (1 + tot_r) ** (1 / years) - 1 if years > 0 else 0.0
    roll_max  = eq.cummax()
    max_dd    = ((eq - roll_max) / roll_max).min()
    dr        = eq.pct_change().dropna()
    sharpe    = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    return {
        "Strategy":     label,
        "Total Return": f"{tot_r*100:.1f}%",
        "CAGR":         f"{cagr*100:.1f}%",
        "Max DD":       f"{max_dd*100:.1f}%",
        "Sharpe":       f"{sharpe:.2f}",
        "Final $":      f"${eq.iloc[-1]:,.0f}",
        "Blown Up":     False,
    }


# ── A. Fixed-leverage buy-and-hold (leverage scan) ────────────────────────────

def fixed_lev_bh(returns: np.ndarray, lev: float, borrow: float = BORROW_RATE) -> pd.Series:
    """Daily-rebalanced leveraged equity curve."""
    eq = np.empty(n)
    eq[0] = CAPITAL
    for i in range(1, n):
        r    = returns[i]
        cost = (lev - 1) * borrow / 252 if lev > 1 else 0.0
        eq[i] = eq[i - 1] * max(0.0, 1 + lev * r - cost)
        if eq[i] <= 0:
            eq[i:] = 0.0
            break
    return pd.Series(eq, index=qqq.index)


levers = np.arange(1.0, 10.5, 0.5)

scan_qqq: list[dict] = []
scan_spy: list[dict] = []

for lv in levers:
    eq = fixed_lev_bh(qqq_r, lv)
    m  = _metrics(eq, f"QQQ {lv:.1f}x")
    m["lev"] = lv
    scan_qqq.append(m)

    eq = fixed_lev_bh(spy_r, lv)
    m  = _metrics(eq, f"SPY {lv:.1f}x")
    m["lev"] = lv
    scan_spy.append(m)


def _scan_series(scan: list[dict], col: str) -> np.ndarray:
    out = []
    for m in scan:
        v = m[col]
        if v in ("-", "BLOWN UP"):
            out.append(np.nan)
        else:
            out.append(float(v.replace("%","").replace("$","").replace(",","")))
    return np.array(out)


# ── B. Donchian 55/20 + SMA-200 QQQ at leverage 1x … 5x ──────────────────────

def donchian_sma_signals(closes: np.ndarray, entry=55, exit_p=20, sma=200) -> np.ndarray:
    """Return daily position size (0 = cash, 1 = fully invested)."""
    sma_val = np.array([closes[max(0,i-sma+1):i+1].mean() for i in range(n)])
    warmup  = max(entry, sma)
    pos     = np.zeros(n)
    in_trade = False

    for i in range(warmup, n):
        c    = closes[i]
        high = closes[max(0, i-entry):i].max()
        low  = closes[max(0, i-exit_p):i].min()
        above = c > sma_val[i]

        if in_trade:
            if c < low or not above:
                in_trade = False
        else:
            if c > high and above:
                in_trade = True
        pos[i] = 1.0 if in_trade else 0.0
    return pos


print("Computing Donchian signals …")
don_pos = donchian_sma_signals(qqq_c)

def donchian_lev(lev: float, borrow: float = BORROW_RATE) -> pd.Series:
    """Apply leverage only when the Donchian strategy is in a trade."""
    eq      = np.empty(n)
    eq[0]   = CAPITAL
    for i in range(1, n):
        in_t = don_pos[i - 1]      # use prior-day signal (no look-ahead)
        if in_t:
            r    = qqq_r[i]
            cost = (lev - 1) * borrow / 252 if lev > 1 else 0.0
            eq[i] = eq[i-1] * max(0.0, 1 + lev * r - cost)
        else:
            eq[i] = eq[i-1]        # in cash — no leverage cost
        if eq[i] <= 0:
            eq[i:] = 0.0
            break
    return pd.Series(eq, index=qqq.index)

don_levers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
scan_don: list[dict] = []
for lv in don_levers:
    eq = donchian_lev(lv)
    m  = _metrics(eq, f"Don55+SMA QQQ {lv:.1f}x")
    m["lev"] = lv
    scan_don.append(m)

# ── C. Volatility-targeted leverage on QQQ ────────────────────────────────────

def vol_targeted(
    returns: np.ndarray,
    target_ann_vol: float,
    max_lev: float = MAX_LEV,
    borrow: float  = BORROW_RATE,
) -> pd.Series:
    """
    Each day compute trailing 20-day realized vol, set:
        leverage = min(target_vol / realized_vol, max_lev)
    Rebalance at open (use prior-day leverage).
    """
    target_daily = target_ann_vol / np.sqrt(252)
    eq = np.empty(n)
    eq[0] = CAPITAL

    for i in range(1, n):
        window = returns[max(0, i - VOL_WINDOW): i]
        rv     = window.std() if len(window) > 1 else target_daily
        lev    = min(target_daily / rv, max_lev) if rv > 0 else 1.0
        cost   = (lev - 1) * borrow / 252 if lev > 1 else 0.0
        eq[i]  = eq[i-1] * max(0.0, 1 + lev * returns[i] - cost)
        if eq[i] <= 0:
            eq[i:] = 0.0
            break

    return pd.Series(eq, index=qqq.index)


print("Computing vol-targeted curves …")
eq_vt15  = vol_targeted(qqq_r, 0.15)
eq_vt20  = vol_targeted(qqq_r, 0.20)
eq_vt30  = vol_targeted(qqq_r, 0.30)
eq_vt40  = vol_targeted(qqq_r, 0.40)

# ── D. Half-Kelly dynamic leverage on QQQ ────────────────────────────────────

def half_kelly(
    returns: np.ndarray,
    kelly_frac: float = 0.5,
    max_lev: float    = MAX_LEV,
    borrow: float     = BORROW_RATE,
) -> pd.Series:
    """
    Rolling Kelly:  L* = (mu_rolling - borrow/252) / var_rolling
    Apply kelly_frac * L*, capped at max_lev.
    """
    eq = np.empty(n)
    eq[0] = CAPITAL

    for i in range(1, n):
        window = returns[max(0, i - KELLY_WINDOW): i]
        if len(window) < 10:
            lev = 1.0
        else:
            mu  = window.mean()
            var = window.var()
            if var <= 0:
                lev = 1.0
            else:
                raw_kelly = (mu - borrow / 252) / var
                lev = float(np.clip(kelly_frac * raw_kelly, 0.0, max_lev))

        cost  = (lev - 1) * borrow / 252 if lev > 1 else 0.0
        eq[i] = eq[i-1] * max(0.0, 1 + lev * returns[i] - cost)
        if eq[i] <= 0:
            eq[i:] = 0.0
            break

    return pd.Series(eq, index=qqq.index)


print("Computing Kelly curves …")
eq_hk    = half_kelly(qqq_r, kelly_frac=0.5)
eq_fk    = half_kelly(qqq_r, kelly_frac=1.0)

# ── build comparison table ────────────────────────────────────────────────────

key_curves = {
    "B&H QQQ 1x (baseline)":     fixed_lev_bh(qqq_r, 1.0),
    "B&H QQQ 2x":                 fixed_lev_bh(qqq_r, 2.0),
    "B&H QQQ 3x (TQQQ-like)":    fixed_lev_bh(qqq_r, 3.0),
    "B&H QQQ 5x":                 fixed_lev_bh(qqq_r, 5.0),
    "Don55+SMA QQQ 1x":           donchian_lev(1.0),
    "Don55+SMA QQQ 2x":           donchian_lev(2.0),
    "Don55+SMA QQQ 3x":           donchian_lev(3.0),
    "Vol-Target 15% ann":         eq_vt15,
    "Vol-Target 20% ann":         eq_vt20,
    "Vol-Target 30% ann":         eq_vt30,
    "Vol-Target 40% ann":         eq_vt40,
    "Half-Kelly QQQ":             eq_hk,
    "Full-Kelly QQQ":             eq_fk,
}

rows = [_metrics(v.rename(k), k) for k, v in key_curves.items()]
metrics_df = pd.DataFrame(rows).drop(columns=["Blown Up"])
metrics_df.to_csv(OUT / "leverage_metrics.csv", index=False)
print("\n" + metrics_df.to_string(index=False))

# ── Figure 1: Leverage scan ────────────────────────────────────────────────────

def _num(scan: list[dict], col: str) -> np.ndarray:
    out = []
    for m in scan:
        v = m.get(col, "-")
        if v in ("-", "BLOWN UP", None):
            out.append(np.nan)
        else:
            out.append(float(str(v).replace("%","").replace("$","").replace(",","")))
    return np.array(out)

fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle(
    "Leverage Scan: Buy & Hold QQQ vs SPY  (daily-rebalanced, 5.5% borrow cost)",
    fontsize=13, fontweight="bold",
)

lv_arr = np.array([m["lev"] for m in scan_qqq])
cagr_q = _num(scan_qqq, "CAGR")
cagr_s = _num(scan_spy,  "CAGR")
dd_q   = _num(scan_qqq, "Max DD")
dd_s   = _num(scan_spy,  "Max DD")
sr_q   = _num(scan_qqq, "Sharpe")
sr_s   = _num(scan_spy,  "Sharpe")

ax = axes[0]
ax.plot(lv_arr, cagr_q, "o-", color="#1565C0", label="QQQ")
ax.plot(lv_arr, cagr_s, "s--", color="#42A5F5", label="SPY")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel("Leverage"); ax.set_ylabel("CAGR (%)")
ax.set_title("CAGR vs Leverage"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(lv_arr, dd_q, "o-", color="#1565C0", label="QQQ")
ax.plot(lv_arr, dd_s, "s--", color="#42A5F5", label="SPY")
ax.axhline(-100, color="red", linewidth=0.8, linestyle=":")
ax.set_xlabel("Leverage"); ax.set_ylabel("Max Drawdown (%)")
ax.set_title("Max Drawdown vs Leverage"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(lv_arr, sr_q, "o-", color="#1565C0", label="QQQ")
ax.plot(lv_arr, sr_s, "s--", color="#42A5F5", label="SPY")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel("Leverage"); ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe vs Leverage"); ax.legend(); ax.grid(alpha=0.3)

# mark Kelly-optimal on CAGR chart (rough theoretical: peak of the curve)
valid_q = ~np.isnan(cagr_q)
if valid_q.any():
    peak_idx = np.nanargmax(cagr_q)
    axes[0].axvline(lv_arr[peak_idx], color="#1565C0", linestyle=":", alpha=0.7,
                    label=f"QQQ peak ~{lv_arr[peak_idx]:.1f}x")
    axes[0].legend(fontsize=8)

fig1.tight_layout()
fig1.savefig(OUT / "leverage_scan.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'leverage_scan.png'}")

# ── Figure 2: Equity curves of key strategies ─────────────────────────────────

COLORS2 = {
    "B&H QQQ 1x (baseline)":    "#90A4AE",
    "B&H QQQ 2x":               "#1565C0",
    "B&H QQQ 3x (TQQQ-like)":  "#0D47A1",
    "B&H QQQ 5x":               "#01579B",
    "Don55+SMA QQQ 1x":         "#2E7D32",
    "Don55+SMA QQQ 2x":         "#388E3C",
    "Don55+SMA QQQ 3x":         "#1B5E20",
    "Vol-Target 15% ann":       "#F9A825",
    "Vol-Target 20% ann":       "#F57F17",
    "Vol-Target 30% ann":       "#E65100",
    "Vol-Target 40% ann":       "#BF360C",
    "Half-Kelly QQQ":           "#6A1B9A",
    "Full-Kelly QQQ":           "#4A148C",
}
STYLES2 = {k: "--" if "SPY" in k or "1x" in k else "-" for k in COLORS2}

fig2 = plt.figure(figsize=(20, 16))
gs2 = gridspec.GridSpec(3, 1, figure=fig2, hspace=0.40, height_ratios=[3, 2, 1.6])
ax_eq = fig2.add_subplot(gs2[0])
ax_dd = fig2.add_subplot(gs2[1])
ax_tb = fig2.add_subplot(gs2[2])
ax_tb.axis("off")

fig2.suptitle(
    "Leveraged Strategies on QQQ  (Mar 2021 to Mar 2026, $10 000 start, 5.5% borrow)",
    fontsize=13, fontweight="bold", y=0.99,
)

for label, eq in key_curves.items():
    eq = eq.ffill()
    if eq.iloc[-1] <= 0:
        continue   # skip blown-up curves from equity chart
    norm = eq / eq.iloc[0]
    ax_eq.plot(eq.index, norm,
               color=COLORS2.get(label, "grey"),
               linestyle=STYLES2.get(label, "-"),
               linewidth=1.5, label=label)

ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
ax_eq.set_ylabel("Normalised Value")
ax_eq.set_title("Equity Curves  (normalised to 1.0 at start)")
ax_eq.legend(fontsize=8, ncol=2, loc="upper left")
ax_eq.grid(alpha=0.25)

for label, eq in key_curves.items():
    eq = eq.ffill()
    if eq.iloc[-1] <= 0:
        continue
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    ax_dd.plot(eq.index, dd,
               color=COLORS2.get(label, "grey"),
               linestyle=STYLES2.get(label, "-"),
               linewidth=1.2, label=label)

ax_dd.axhline(0, color="black", linewidth=0.6)
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown")
ax_dd.legend(fontsize=8, ncol=2, loc="lower left")
ax_dd.grid(alpha=0.25)

# table
col_labels = [c for c in metrics_df.columns]
cell_text  = metrics_df.values.tolist()
tbl = ax_tb.table(cellText=cell_text, colLabels=col_labels,
                  cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.35)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#263238")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(cell_text) + 1):
    for j in range(len(col_labels)):
        clr = "#FFEBEE" if "BLOWN" in str(cell_text[i-1][j]) else \
              ("#ECEFF1" if i % 2 == 0 else "white")
        tbl[i, j].set_facecolor(clr)

fig2.savefig(OUT / "equity_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'equity_curves.png'}")
plt.show()
