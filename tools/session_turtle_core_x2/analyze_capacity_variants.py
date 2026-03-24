"""
Deep analysis of capacity variants: Sharpe, Calmar, drawdown duration,
time-underwater, and professional recommendation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

REPORT_DIR = Path("reports/capacity_test_core_x2")
OUT_DIR    = Path("reports/capacity_test_core_x2/analysis")

VARIANTS = [
    ("Baseline  (exp=2x)", "trades_baseline_risk_5_exp_2x.csv"),
    ("High exp  (exp=3x)", "trades_high_exp_risk_5_exp_3x.csv"),
    ("High exp  (exp=4x)", "trades_high_exp_risk_5_exp_4x.csv"),
]

COLORS = ["#2196F3", "#FF9800", "#E53935"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trades(fname: str) -> pd.DataFrame:
    path = REPORT_DIR / fname
    df = pd.read_csv(path, parse_dates=["entry_ts", "exit_ts"])
    df["net_pnl"]          = pd.to_numeric(df["net_pnl"], errors="coerce")
    df["equity_after_exit"] = pd.to_numeric(df["equity_after_exit"], errors="coerce")
    df = df.sort_values("exit_ts").reset_index(drop=True)
    return df


def build_daily_equity(df: pd.DataFrame, start_date: pd.Timestamp,
                       end_date: pd.Timestamp, initial: float = 1000.0) -> pd.Series:
    """Forward-fill equity on a daily index from exit events."""
    idx = pd.date_range(start_date, end_date, freq="D")
    eq  = pd.Series(np.nan, index=idx, dtype=float)
    eq.iloc[0] = initial
    # take the last (highest) equity value recorded on each exit day
    daily = (
        df.set_index(df["exit_ts"].dt.normalize())["equity_after_exit"]
        .groupby(level=0)
        .last()
    )
    eq.update(daily)
    eq = eq.ffill()
    return eq


def drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max * 100   # negative %


def drawdown_stats(dd: pd.Series) -> dict:
    """
    Returns:
      max_dd_pct        — worst peak-to-trough %
      calmar            — CAGR / abs(max_dd)
      avg_dd_duration   — average days per drawdown episode
      max_dd_duration   — longest drawdown in days
      time_underwater_pct — % of days below prior equity peak
    """
    underwater = dd < -0.001   # small tolerance
    time_underwater_pct = underwater.mean() * 100

    # episode lengths
    durations = []
    in_dd = False
    length = 0
    for uw in underwater:
        if uw:
            in_dd = True
            length += 1
        else:
            if in_dd:
                durations.append(length)
            in_dd = False
            length = 0
    if in_dd:
        durations.append(length)

    avg_dd_dur = float(np.mean(durations)) if durations else 0.0
    max_dd_dur = int(max(durations)) if durations else 0

    return dict(
        max_dd_pct=float(dd.min()),
        time_underwater_pct=float(time_underwater_pct),
        avg_dd_duration=avg_dd_dur,
        max_dd_duration=max_dd_dur,
        n_episodes=len(durations),
    )


def sharpe(equity: pd.Series, rfr_annual: float = 0.05) -> float:
    """Annualised Sharpe on daily returns, rfr = 5% annual."""
    daily_ret = equity.pct_change().dropna()
    rfr_daily = (1 + rfr_annual) ** (1 / 252) - 1
    excess = daily_ret - rfr_daily
    if excess.std() < 1e-12:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def sortino(equity: pd.Series, rfr_annual: float = 0.05) -> float:
    """Annualised Sortino on daily returns."""
    daily_ret = equity.pct_change().dropna()
    rfr_daily = (1 + rfr_annual) ** (1 / 252) - 1
    excess = daily_ret - rfr_daily
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() < 1e-12:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years < 0.01:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_df    = {}
    all_eq    = {}
    all_dd    = {}
    all_stats = {}

    initial = 1_000.0
    global_start = pd.Timestamp("2022-02-01")
    global_end   = pd.Timestamp("2026-03-21")

    for label, fname in VARIANTS:
        df  = load_trades(fname)
        eq  = build_daily_equity(df, global_start, global_end, initial)
        dd  = drawdown_series(eq)
        dds = drawdown_stats(dd)
        c   = cagr(eq)
        s   = sharpe(eq)
        so  = sortino(eq)
        calmar_r = c / abs(dds["max_dd_pct"]) if dds["max_dd_pct"] < -1e-6 else 0.0

        all_df[label]    = df
        all_eq[label]    = eq
        all_dd[label]    = dd
        all_stats[label] = dict(
            label=label,
            trades=len(df),
            cagr_pct=round(c, 2),
            max_dd_pct=round(dds["max_dd_pct"], 2),
            calmar=round(calmar_r, 3),
            sharpe=round(s, 3),
            sortino=round(so, 3),
            time_underwater_pct=round(dds["time_underwater_pct"], 1),
            avg_dd_duration=round(dds["avg_dd_duration"], 1),
            max_dd_duration=dds["max_dd_duration"],
            n_dd_episodes=dds["n_episodes"],
        )

    # ── print table ───────────────────────────────────────────────────────
    cols = [
        ("Variant",        "label",                 "<26"),
        ("Trades",         "trades",                ">7"),
        ("CAGR %",         "cagr_pct",              ">8"),
        ("Max DD %",       "max_dd_pct",            ">9"),
        ("Calmar",         "calmar",                ">7"),
        ("Sharpe",         "sharpe",                ">7"),
        ("Sortino",        "sortino",               ">8"),
        ("UW %",           "time_underwater_pct",   ">7"),
        ("Avg DD d",       "avg_dd_duration",       ">9"),
        ("Max DD d",       "max_dd_duration",       ">9"),
        ("# DD eps",       "n_dd_episodes",         ">9"),
    ]
    header = "  ".join(f"{t:{s}}" for t, _, s in cols)
    sep    = "  ".join("-" * int(s.strip("<>")) for _, _, s in cols)
    print("\n" + header); print(sep)
    for lbl, _ in VARIANTS:
        row = all_stats[lbl]
        parts = []
        for _, key, spec in cols:
            raw  = row.get(key, "-")
            text = f"{raw:.3f}" if isinstance(raw, float) and key in ("calmar","sharpe","sortino") \
                   else (f"{raw:.2f}" if isinstance(raw, float) else str(raw))
            parts.append(f"{text:{spec}}")
        print("  ".join(parts))
    print()

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("Capacity Variant Analysis: Risk-Adjusted Performance",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_eq  = fig.add_subplot(gs[0, :])   # full width — equity curves
    ax_dd  = fig.add_subplot(gs[1, :])   # full width — drawdown
    ax_sh  = fig.add_subplot(gs[2, 0])   # Sharpe / Sortino / Calmar
    ax_uw  = fig.add_subplot(gs[2, 1])   # Time underwater
    ax_dur = fig.add_subplot(gs[3, 0])   # DD duration
    ax_mo  = fig.add_subplot(gs[3, 1])   # Monthly return distribution

    # 1. Equity curves (log scale)
    for (lbl, _), color in zip(VARIANTS, COLORS):
        eq = all_eq[lbl]
        ax_eq.plot(eq.index, eq.values, label=lbl, color=color, linewidth=1.8)
    ax_eq.set_yscale("log")
    ax_eq.set_title("Equity Curve (log scale, $1,000 start)", fontsize=11)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.legend(fontsize=9)
    ax_eq.grid(alpha=0.3)

    # 2. Drawdown
    for (lbl, _), color in zip(VARIANTS, COLORS):
        dd = all_dd[lbl]
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color)
        ax_dd.plot(dd.index, dd.values, label=lbl, color=color, linewidth=1.0)
    ax_dd.set_title("Drawdown (%)", fontsize=11)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.yaxis.set_major_formatter(PercentFormatter())
    ax_dd.legend(fontsize=9)
    ax_dd.grid(alpha=0.3)

    # 3. Risk ratios bar chart
    labels_short = ["2x", "3x", "4x"]
    x = np.arange(3)
    w = 0.25
    sharpes  = [all_stats[l]["sharpe"]  for l, _ in VARIANTS]
    sortinos = [all_stats[l]["sortino"] for l, _ in VARIANTS]
    calmars  = [all_stats[l]["calmar"]  for l, _ in VARIANTS]
    ax_sh.bar(x - w, sharpes,  w, label="Sharpe",  color="#42A5F5")
    ax_sh.bar(x,     sortinos, w, label="Sortino", color="#66BB6A")
    ax_sh.bar(x + w, calmars,  w, label="Calmar",  color="#FFA726")
    ax_sh.set_xticks(x); ax_sh.set_xticklabels(labels_short)
    ax_sh.set_title("Risk-Adjusted Ratios", fontsize=11)
    ax_sh.legend(fontsize=9); ax_sh.grid(alpha=0.3, axis="y")
    for bar in ax_sh.patches:
        h = bar.get_height()
        ax_sh.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                   ha="center", va="bottom", fontsize=7)

    # 4. Time underwater
    uw_vals = [all_stats[l]["time_underwater_pct"] for l, _ in VARIANTS]
    bars = ax_uw.bar(labels_short, uw_vals, color=COLORS, alpha=0.8)
    ax_uw.set_title("Time Underwater (% of days)", fontsize=11)
    ax_uw.set_ylabel("% days below peak")
    ax_uw.yaxis.set_major_formatter(PercentFormatter())
    for bar, v in zip(bars, uw_vals):
        ax_uw.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                   ha="center", va="bottom", fontsize=9)
    ax_uw.grid(alpha=0.3, axis="y")

    # 5. Drawdown duration
    avg_durs = [all_stats[l]["avg_dd_duration"] for l, _ in VARIANTS]
    max_durs = [all_stats[l]["max_dd_duration"] for l, _ in VARIANTS]
    x = np.arange(3); w = 0.35
    ax_dur.bar(x - w/2, avg_durs, w, label="Avg DD (days)", color="#78909C")
    ax_dur.bar(x + w/2, max_durs, w, label="Max DD (days)", color="#EF5350")
    ax_dur.set_xticks(x); ax_dur.set_xticklabels(labels_short)
    ax_dur.set_title("Drawdown Duration (days)", fontsize=11)
    ax_dur.legend(fontsize=9); ax_dur.grid(alpha=0.3, axis="y")
    for bar in ax_dur.patches:
        h = bar.get_height()
        ax_dur.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8)

    # 6. Monthly return distribution (violin/box)
    monthly_rets = []
    for lbl, _ in VARIANTS:
        eq = all_eq[lbl]
        m  = eq.resample("ME").last().pct_change().dropna() * 100
        monthly_rets.append(m.values)
    vp = ax_mo.violinplot(monthly_rets, positions=[1, 2, 3],
                          showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], COLORS):
        body.set_facecolor(color); body.set_alpha(0.6)
    ax_mo.set_xticks([1, 2, 3]); ax_mo.set_xticklabels(labels_short)
    ax_mo.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_mo.set_title("Monthly Return Distribution (%)", fontsize=11)
    ax_mo.set_ylabel("Monthly return %")
    ax_mo.grid(alpha=0.3, axis="y")

    fig.savefig(OUT_DIR / "capacity_variant_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {OUT_DIR / 'capacity_variant_analysis.png'}")
    plt.close(fig)

    # ── professional recommendation ───────────────────────────────────────
    print("=" * 72)
    print("PROFESSIONAL RECOMMENDATION")
    print("=" * 72)
    for lbl, _ in VARIANTS:
        s = all_stats[lbl]
        print(
            f"\n{s['label']}"
            f"\n  CAGR {s['cagr_pct']:.1f}%  |  Max DD {s['max_dd_pct']:.1f}%  |  "
            f"Calmar {s['calmar']:.2f}  |  Sharpe {s['sharpe']:.2f}  |  Sortino {s['sortino']:.2f}"
            f"\n  Time underwater {s['time_underwater_pct']:.1f}%  |  "
            f"Avg DD episode {s['avg_dd_duration']:.0f}d  |  Max DD episode {s['max_dd_duration']}d"
        )
    print()


if __name__ == "__main__":
    main()
