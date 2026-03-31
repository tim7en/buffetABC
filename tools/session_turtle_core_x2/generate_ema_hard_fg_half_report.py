"""
Report generator — Session Turtle Core x3, ema_hard_fg_half configuration.

Produces:
  1. Text summary printed to stdout
  2. charts/01_equity_drawdown.png    — equity curve + drawdown
  3. charts/02_candidate_funnel.png   — how many candidates the overlay filters
  4. charts/03_overlay_regime_impact.png — EMA / F&G / VIX regime breakdown
  5. charts/04_winner_vs_loser.png    — overlay state for winners vs losers
  6. charts/05_pnl_by_bucket.png      — P&L and trade count by asset bucket
  7. charts/06_trade_dist.png         — R-multiple and duration distributions

Run from repo root:
  python tools/session_turtle_core_x2/generate_ema_hard_fg_half_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPORT_DIR = Path("reports/session_turtle_x3_ema_hard_fg_half_production")
OUT_DIR    = REPORT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = REPORT_DIR / "summary.json"
TRADES_FILE  = REPORT_DIR / "trades.csv"

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
BORDER    = "#2d3048"
TEXT      = "#e2e8f0"
MUTED     = "#64748b"
GREEN     = "#10b981"
RED       = "#ef4444"
AMBER     = "#f59e0b"
BLUE      = "#3b82f6"
PURPLE    = "#a855f7"
CYAN      = "#06b6d4"
BUCKET_COLORS = {"equity": BLUE, "crypto": PURPLE, "gold": AMBER, "metals": CYAN}

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.6,
    "legend.facecolor": PANEL_BG, "legend.edgecolor": BORDER,
    "font.size": 9,
})


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def _load() -> tuple[dict, list[dict]]:
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    with TRADES_FILE.open(newline="", encoding="utf-8") as fh:
        trades = list(csv.DictReader(fh))
    # coerce numeric fields
    for t in trades:
        for k in ("net_pnl", "equity_after_exit", "notional", "shares",
                  "ext_hours_proxy_mult", "ext_hours_proxy_score",
                  "technical_ema_mult", "entry_exposure_mult",
                  "entry_rel_volume", "conviction_mult"):
            try:
                t[k] = float(t[k]) if t[k] not in ("", None) else None
            except (ValueError, TypeError):
                t[k] = None
    return summary, trades


# ── Chart 1: Equity + Drawdown ─────────────────────────────────────────────────
def chart_equity(trades: list[dict], summary: dict) -> None:
    # Build equity curve by sorting on exit_ts — equity_after_exit is stamped
    # at the moment of exit, so chronological exit order is the correct sequence.
    # Sorting by entry_ts causes zigzags when concurrent positions exit out of order.
    initial = float(summary["initial_capital"])
    sorted_trades = sorted(
        [t for t in trades if t["equity_after_exit"] is not None],
        key=lambda t: t["exit_ts"],
    )

    dates, equities = [datetime.fromisoformat(summary["start_date"])], [initial]
    for t in sorted_trades:
        dates.append(datetime.fromisoformat(t["exit_ts"]))
        equities.append(t["equity_after_exit"])

    peak_eq, drawdowns = initial, []
    for eq in equities:
        peak_eq = max(peak_eq, eq)
        drawdowns.append((peak_eq - eq) / peak_eq * 100.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    fig.suptitle(
        f"Session Turtle Core x3  |  ema_hard_fg_half  |  "
        f"{summary['start_date'][:10]} to {summary['end_date'][:10]}",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    # Equity
    ax1.plot(dates, equities, color=GREEN, linewidth=1.8, label="Equity")
    ax1.fill_between(dates, initial, equities, color=GREEN, alpha=0.12)
    peak_vals = [max(equities[:i+1]) for i in range(len(equities))]
    ax1.plot(dates, peak_vals, color=MUTED, linewidth=0.9, linestyle="--", label="Peak", alpha=0.7)
    ax1.set_ylabel("Portfolio Value ($)", color=TEXT)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Key stats annotation
    ret_pct   = summary["total_return_pct"]
    cagr_pct  = summary["cagr_pct"]
    maxdd_pct = summary["max_realized_drawdown_pct"]
    pf        = summary["profit_factor"]
    wr        = summary["win_rate_pct"]
    ax1.text(
        0.99, 0.06,
        f"Return {ret_pct:.0f}%   CAGR {cagr_pct:.1f}%   MaxDD {maxdd_pct:.1f}%   PF {pf:.2f}   WR {wr:.1f}%",
        transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG, edgecolor=BORDER),
    )

    # Drawdown
    ax2.fill_between(dates, drawdowns, 0, color=RED, alpha=0.8)
    ax2.set_ylabel("Drawdown %", color=TEXT)
    ax2.set_xlabel("Date", color=TEXT)
    ax2.set_ylim(top=0)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(-maxdd_pct, color=RED, linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.text(dates[-1], -maxdd_pct - 0.5, f"Max DD {maxdd_pct:.1f}%", color=RED, fontsize=7, ha="right")

    fig.autofmt_xdate()
    _savefig(fig, OUT_DIR / "01_equity_drawdown.png")


# ── Chart 2: Candidate Funnel ──────────────────────────────────────────────────
def chart_candidate_funnel(summary: dict) -> None:
    s = summary
    total_candidates    = int(s["candidate_trades"])
    skip_same_ticker    = int(s["skipped_same_ticker"])
    skip_no_capacity    = int(s["skipped_no_capacity"])
    executed            = int(s["executed_trades"])

    # Overlay-level breakdown of executed trades
    ext_risk_on   = int(s["entries_ext_hours_risk_on_micro"])   # shorts suppressed in risk-on
    ext_risk_off  = int(s["entries_ext_hours_risk_off_micro"])  # longs halved in risk-off
    ext_neutral   = int(s["entries_ext_hours_neutral_micro"])
    ext_scaled    = int(s["entries_ext_hours_proxy_scaled"])    # actually had mult < 1

    above_ema = int(s["entries_above_ema"])
    below_ema = int(s["entries_below_ema"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Candidate Funnel  |  How Overlays Filter the Signal Pool", color=TEXT, fontsize=11, fontweight="bold")

    # ── Left: Waterfall funnel ──
    ax = axes[0]
    stages = [
        ("Raw candidates", total_candidates, BLUE),
        ("Skip: same ticker", skip_same_ticker, AMBER),
        ("Skip: no capacity", skip_no_capacity, AMBER),
        ("Executed trades", executed, GREEN),
    ]
    ys     = range(len(stages))
    labels = [s[0] for s in stages]
    vals   = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    bars   = ax.barh(ys, vals, color=colors, alpha=0.85, edgecolor=BORDER, height=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                f"{val}", va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Signal Pool Waterfall", color=TEXT)
    ax.grid(True, axis="x", alpha=0.3)
    # execution rate annotation
    exec_rate = executed / total_candidates * 100
    ax.text(0.98, 0.02, f"Execution rate: {exec_rate:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)

    # ── Right: Overlay regime at time of executed entries ──
    ax2 = axes[1]
    cats = [
        ("EMA: above (with-trend long / short off)", above_ema, GREEN),
        ("EMA: below (with-trend short / long off)", below_ema, AMBER),
        ("Overlay: risk-on micro (shorts suppressed)", ext_risk_on, PURPLE),
        ("Overlay: neutral micro (both full)", ext_neutral, BLUE),
        ("Overlay: risk-off micro (longs halved)", ext_risk_off, RED),
        ("Overlay: actually size-reduced (EH mult<1)", ext_scaled, RED),
    ]
    cat_labels = [c[0] for c in cats]
    cat_vals   = [c[1] for c in cats]
    cat_colors = [c[2] for c in cats]
    ys2 = range(len(cats))
    bars2 = ax2.barh(ys2, cat_vals, color=cat_colors, alpha=0.85, edgecolor=BORDER, height=0.55)
    for bar, val in zip(bars2, cat_vals):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val}", va="center", ha="left", fontsize=9, color=TEXT)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels(cat_labels, fontsize=8)
    ax2.set_xlabel("Count (of executed trades)")
    ax2.set_title("Overlay State at Entry (Executed Trades)", color=TEXT)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, OUT_DIR / "02_candidate_funnel.png")


# ── Chart 3: Overlay regime breakdown ─────────────────────────────────────────
def chart_overlay_regimes(trades: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Overlay Regime Breakdown  |  Win Rate & P&L by Regime State at Entry", color=TEXT, fontsize=11, fontweight="bold")

    def regime_stats(key: str, valid_vals: list[str]) -> dict:
        groups: dict[str, list] = defaultdict(list)
        for t in trades:
            val = t.get(key) or "unknown"
            if not valid_vals or val in valid_vals:
                groups[val].append(float(t["net_pnl"]) if t["net_pnl"] is not None else 0.0)
        return groups

    # ── EMA regime ──
    ax = axes[0]
    ema_groups = regime_stats("technical_ema_regime", ["above_ema", "below_ema", ""])
    ema_order  = ["above_ema", "below_ema", "unknown"]
    ema_labels = {"above_ema": "Above EMA\n(with-trend long)", "below_ema": "Below EMA\n(with-trend short)", "unknown": "No EMA\n(non-crypto)"}
    ema_colors = {"above_ema": GREEN, "below_ema": AMBER, "unknown": MUTED}
    for i, key in enumerate(ema_order):
        pnls = ema_groups.get(key, [])
        if not pnls:
            continue
        wins    = sum(1 for p in pnls if p > 0)
        wr      = wins / len(pnls) * 100
        total   = sum(pnls)
        color   = ema_colors[key]
        label   = ema_labels[key]
        ax.bar(i, wr, color=color, alpha=0.85, edgecolor=BORDER)
        ax.text(i, wr + 1.5, f"WR {wr:.0f}%\n{len(pnls)} trades\nP&L ${total:+,.0f}", ha="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(range(len(ema_order)))
    ax.set_xticklabels([ema_labels[k] for k in ema_order], fontsize=7.5)
    ax.set_ylabel("Win Rate %")
    ax.set_ylim(0, 100)
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("EMA Hard Stop Regime", color=TEXT)
    ax.grid(True, axis="y", alpha=0.3)

    # ── F&G / VIX macro overlay regime ──
    ax2 = axes[1]
    eh_groups = regime_stats("ext_hours_proxy_regime", [])
    regime_order  = ["risk_on_micro", "neutral_micro", "risk_off_micro", "unknown"]
    regime_labels = {
        "risk_on_micro":  "Risk-On\n(shorts ~0×)",
        "neutral_micro":  "Neutral\n(both 1×)",
        "risk_off_micro": "Risk-Off\n(longs 0.5×)",
        "unknown":        "No data",
    }
    regime_colors = {"risk_on_micro": GREEN, "neutral_micro": BLUE, "risk_off_micro": RED, "unknown": MUTED}
    for i, key in enumerate(regime_order):
        pnls = eh_groups.get(key, [])
        if not pnls:
            continue
        wins  = sum(1 for p in pnls if p > 0)
        wr    = wins / len(pnls) * 100
        total = sum(pnls)
        ax2.bar(i, wr, color=regime_colors[key], alpha=0.85, edgecolor=BORDER)
        ax2.text(i, wr + 1.5, f"WR {wr:.0f}%\n{len(pnls)} trades\nP&L ${total:+,.0f}", ha="center", fontsize=7.5, color=TEXT)
    ax2.set_xticks(range(len(regime_order)))
    ax2.set_xticklabels([regime_labels[k] for k in regime_order], fontsize=7.5)
    ax2.set_ylabel("Win Rate %")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_title("VIX / F&G Macro Regime", color=TEXT)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Overlay mult distribution (size impact) ──
    ax3 = axes[2]
    mults = [t["ext_hours_proxy_mult"] for t in trades if t["ext_hours_proxy_mult"] is not None]
    ax3.hist(mults, bins=20, color=PURPLE, alpha=0.8, edgecolor=BORDER)
    scaled_count = sum(1 for m in mults if m < 0.999)
    ax3.set_xlabel("Ext-Hours Overlay Multiplier")
    ax3.set_ylabel("Trade Count")
    ax3.set_title(f"Overlay Size Multiplier Distribution\n({scaled_count} trades reduced below 1.0×)", color=TEXT)
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.axvline(1.0, color=MUTED, linewidth=1.2, linestyle="--", alpha=0.7, label="Full size (1.0×)")
    ax3.axvline(0.5, color=RED, linewidth=1.2, linestyle="--", alpha=0.7, label="Half size (0.5×)")
    ax3.legend(fontsize=7.5)

    fig.tight_layout()
    _savefig(fig, OUT_DIR / "03_overlay_regime_impact.png")


# ── Chart 4: Winners vs Losers profile ────────────────────────────────────────
def chart_winner_vs_loser(trades: list[dict]) -> None:
    winners = [t for t in trades if (t["net_pnl"] or 0) > 0]
    losers  = [t for t in trades if (t["net_pnl"] or 0) <= 0]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        f"Good vs Bad Candidates  |  {len(winners)} Winners vs {len(losers)} Losers",
        color=TEXT, fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── P&L distribution ──
    ax_pnl = fig.add_subplot(gs[0, :])
    winner_pnls = [float(t["net_pnl"]) for t in winners]
    loser_pnls  = [float(t["net_pnl"]) for t in losers]
    all_pnl     = winner_pnls + loser_pnls
    bins = np.linspace(min(all_pnl) * 1.05, max(all_pnl) * 1.05, 40)
    ax_pnl.hist(winner_pnls, bins=bins, color=GREEN, alpha=0.75, label=f"Winners (n={len(winners)})", edgecolor=BORDER)
    ax_pnl.hist(loser_pnls,  bins=bins, color=RED,   alpha=0.75, label=f"Losers (n={len(losers)})",  edgecolor=BORDER)
    avg_win  = np.mean(winner_pnls) if winner_pnls else 0
    avg_loss = np.mean(loser_pnls)  if loser_pnls  else 0
    ax_pnl.axvline(avg_win,  color=GREEN, linewidth=1.5, linestyle="--", alpha=0.9, label=f"Avg win  ${avg_win:+,.0f}")
    ax_pnl.axvline(avg_loss, color=RED,   linewidth=1.5, linestyle="--", alpha=0.9, label=f"Avg loss ${avg_loss:+,.0f}")
    ax_pnl.axvline(0, color=MUTED, linewidth=1.0, linestyle="-", alpha=0.4)
    ax_pnl.set_xlabel("Net P&L ($)")
    ax_pnl.set_ylabel("Count")
    ax_pnl.set_title("P&L Distribution: Winners vs Losers", color=TEXT)
    ax_pnl.legend(fontsize=7.5)
    ax_pnl.grid(True, axis="y", alpha=0.3)

    # ── EMA regime breakdown for winners vs losers ──
    ax_ema = fig.add_subplot(gs[1, 0])
    ema_vals = ["above_ema", "below_ema", ""]
    ema_labels_short = {"above_ema": "Above EMA", "below_ema": "Below EMA", "": "No EMA"}
    x = np.arange(len(ema_vals))
    w_counts = [sum(1 for t in winners if (t["technical_ema_regime"] or "") == v) for v in ema_vals]
    l_counts = [sum(1 for t in losers  if (t["technical_ema_regime"] or "") == v) for v in ema_vals]
    ax_ema.bar(x - 0.2, w_counts, 0.35, label="Winners", color=GREEN, alpha=0.85, edgecolor=BORDER)
    ax_ema.bar(x + 0.2, l_counts, 0.35, label="Losers",  color=RED,   alpha=0.85, edgecolor=BORDER)
    ax_ema.set_xticks(x)
    ax_ema.set_xticklabels([ema_labels_short[v] for v in ema_vals], fontsize=7.5)
    ax_ema.set_title("EMA Regime at Entry", color=TEXT)
    ax_ema.set_ylabel("Count")
    ax_ema.legend(fontsize=7)
    ax_ema.grid(True, axis="y", alpha=0.3)

    # ── Macro overlay regime ──
    ax_mac = fig.add_subplot(gs[1, 1])
    mac_vals = ["risk_on_micro", "neutral_micro", "risk_off_micro"]
    mac_labels_short = {"risk_on_micro": "Risk-On", "neutral_micro": "Neutral", "risk_off_micro": "Risk-Off"}
    x2 = np.arange(len(mac_vals))
    w2 = [sum(1 for t in winners if t.get("ext_hours_proxy_regime") == v) for v in mac_vals]
    l2 = [sum(1 for t in losers  if t.get("ext_hours_proxy_regime") == v) for v in mac_vals]
    ax_mac.bar(x2 - 0.2, w2, 0.35, label="Winners", color=GREEN, alpha=0.85, edgecolor=BORDER)
    ax_mac.bar(x2 + 0.2, l2, 0.35, label="Losers",  color=RED,   alpha=0.85, edgecolor=BORDER)
    ax_mac.set_xticks(x2)
    ax_mac.set_xticklabels([mac_labels_short[v] for v in mac_vals], fontsize=7.5)
    ax_mac.set_title("Macro Overlay Regime at Entry", color=TEXT)
    ax_mac.set_ylabel("Count")
    ax_mac.legend(fontsize=7)
    ax_mac.grid(True, axis="y", alpha=0.3)

    # ── Asset bucket breakdown ──
    ax_bkt = fig.add_subplot(gs[1, 2])
    buckets = ["equity", "crypto", "gold", "metals"]
    x3 = np.arange(len(buckets))
    w3 = [sum(1 for t in winners if t.get("asset_bucket") == b) for b in buckets]
    l3 = [sum(1 for t in losers  if t.get("asset_bucket") == b) for b in buckets]
    ax_bkt.bar(x3 - 0.2, w3, 0.35, label="Winners", color=GREEN, alpha=0.85, edgecolor=BORDER)
    ax_bkt.bar(x3 + 0.2, l3, 0.35, label="Losers",  color=RED,   alpha=0.85, edgecolor=BORDER)
    ax_bkt.set_xticks(x3)
    ax_bkt.set_xticklabels([b.capitalize() for b in buckets], fontsize=7.5)
    ax_bkt.set_title("Asset Bucket Breakdown", color=TEXT)
    ax_bkt.set_ylabel("Count")
    ax_bkt.legend(fontsize=7)
    ax_bkt.grid(True, axis="y", alpha=0.3)

    _savefig(fig, OUT_DIR / "04_winner_vs_loser.png")


# ── Chart 5: P&L by bucket ────────────────────────────────────────────────────
def chart_pnl_by_bucket(trades: list[dict], summary: dict) -> None:
    buckets = ["equity", "crypto", "gold", "metals"]
    bucket_pnl    = {b: float(summary.get(f"{b}_pnl",   0)) for b in buckets}
    bucket_trades = {b: int(summary.get(f"{b}_trades", 0)) for b in buckets}

    # Win rate per bucket
    bucket_wr = {}
    for b in buckets:
        bt = [t for t in trades if t.get("asset_bucket") == b]
        if bt:
            bucket_wr[b] = sum(1 for t in bt if (t["net_pnl"] or 0) > 0) / len(bt) * 100
        else:
            bucket_wr[b] = 0.0

    # Trade direction split
    bucket_long  = {b: sum(1 for t in trades if t.get("asset_bucket") == b and t.get("direction") == "long")  for b in buckets}
    bucket_short = {b: sum(1 for t in trades if t.get("asset_bucket") == b and t.get("direction") == "short") for b in buckets}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("P&L & Activity by Asset Bucket", color=TEXT, fontsize=11, fontweight="bold")

    # P&L bar
    ax1 = axes[0]
    colors = [GREEN if bucket_pnl[b] >= 0 else RED for b in buckets]
    ax1.bar(buckets, [bucket_pnl[b] for b in buckets], color=colors, alpha=0.85, edgecolor=BORDER)
    for i, b in enumerate(buckets):
        ax1.text(i, bucket_pnl[b] + (300 if bucket_pnl[b] >= 0 else -800),
                 f"${bucket_pnl[b]:+,.0f}", ha="center", fontsize=8, color=TEXT)
    ax1.set_ylabel("Net P&L ($)")
    ax1.set_title("Net P&L by Bucket", color=TEXT)
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Trade count with long/short split
    ax2 = axes[1]
    x = np.arange(len(buckets))
    ax2.bar(x, [bucket_long[b]  for b in buckets], label="Long",  color=GREEN, alpha=0.85, edgecolor=BORDER)
    ax2.bar(x, [bucket_short[b] for b in buckets], label="Short", color=RED,   alpha=0.85, edgecolor=BORDER,
            bottom=[bucket_long[b] for b in buckets])
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.capitalize() for b in buckets])
    ax2.set_ylabel("Trade Count")
    ax2.set_title("Trade Count (Long / Short Split)", color=TEXT)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, axis="y", alpha=0.3)

    # Win rate
    ax3 = axes[2]
    wr_vals = [bucket_wr[b] for b in buckets]
    bar_colors = [GREEN if w >= 50 else AMBER if w >= 35 else RED for w in wr_vals]
    ax3.bar(buckets, wr_vals, color=bar_colors, alpha=0.85, edgecolor=BORDER)
    ax3.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6, label="50% line")
    for i, (b, w) in enumerate(zip(buckets, wr_vals)):
        ax3.text(i, w + 1.5, f"{w:.0f}%", ha="center", fontsize=8.5, color=TEXT)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Win Rate %")
    ax3.set_title("Win Rate by Bucket", color=TEXT)
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, OUT_DIR / "05_pnl_by_bucket.png")


# ── Chart 6: Trade duration & R-multiple ──────────────────────────────────────
def chart_trade_distributions(trades: list[dict]) -> None:
    # Duration in hours
    durations = []
    for t in trades:
        try:
            entry = datetime.fromisoformat(t["entry_ts"])
            exit_ = datetime.fromisoformat(t["exit_ts"])
            durations.append((exit_ - entry).total_seconds() / 3600)
        except Exception:
            pass

    # Approximate R-multiple: P&L / (notional * fixed_stop_pct)
    r_mults = []
    for t in trades:
        pnl = t["net_pnl"]
        notional = t["notional"]
        if pnl is not None and notional is not None and notional > 0:
            r = pnl / (notional * 0.10)   # fixed_stop_pct = 0.10
            r_mults.append(r)

    winner_r = [r for r, t in zip(r_mults, trades) if (t["net_pnl"] or 0) > 0]
    loser_r  = [r for r, t in zip(r_mults, trades) if (t["net_pnl"] or 0) <= 0]

    # Direction breakdown by month
    sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
    monthly: dict[str, dict] = defaultdict(lambda: {"long": 0, "short": 0, "wins": 0, "total": 0})
    for t in sorted_trades:
        mo = t["entry_ts"][:7]  # YYYY-MM
        d  = t.get("direction", "long")
        monthly[mo][d] += 1
        monthly[mo]["total"] += 1
        if (t["net_pnl"] or 0) > 0:
            monthly[mo]["wins"] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Trade Characteristics — Duration, R-Multiple, Monthly Activity", color=TEXT, fontsize=11, fontweight="bold")

    # Trade duration
    ax1 = axes[0, 0]
    if durations:
        ax1.hist(durations, bins=30, color=BLUE, alpha=0.85, edgecolor=BORDER)
        ax1.axvline(np.median(durations), color=AMBER, linewidth=1.5, linestyle="--",
                    label=f"Median {np.median(durations):.1f}h")
        ax1.set_xlabel("Duration (hours)")
        ax1.set_ylabel("Count")
        ax1.set_title("Trade Duration Distribution", color=TEXT)
        ax1.legend(fontsize=7.5)
        ax1.grid(True, axis="y", alpha=0.3)

    # R-multiple distribution
    ax2 = axes[0, 1]
    if r_mults:
        bins = np.linspace(-3, max(r_mults) * 1.05, 40)
        ax2.hist(winner_r, bins=bins, color=GREEN, alpha=0.75, label=f"Winners (n={len(winner_r)})", edgecolor=BORDER)
        ax2.hist(loser_r,  bins=bins, color=RED,   alpha=0.75, label=f"Losers (n={len(loser_r)})",  edgecolor=BORDER)
        ax2.axvline(0, color=MUTED, linewidth=1.0, alpha=0.5)
        ax2.axvline(np.mean(winner_r) if winner_r else 0, color=GREEN, linewidth=1.5, linestyle="--",
                    label=f"Avg win {np.mean(winner_r):.2f}R" if winner_r else "")
        ax2.axvline(np.mean(loser_r)  if loser_r  else 0, color=RED,   linewidth=1.5, linestyle="--",
                    label=f"Avg loss {np.mean(loser_r):.2f}R"  if loser_r  else "")
        ax2.set_xlabel("R-Multiple (P&L / 10% notional stop)")
        ax2.set_ylabel("Count")
        ax2.set_title("R-Multiple Distribution", color=TEXT)
        ax2.legend(fontsize=7.5)
        ax2.grid(True, axis="y", alpha=0.3)

    # Monthly trade count
    ax3 = axes[1, 0]
    if monthly:
        months   = sorted(monthly.keys())
        longs    = [monthly[m]["long"]  for m in months]
        shorts   = [monthly[m]["short"] for m in months]
        x = range(len(months))
        ax3.bar(x, longs,  label="Long",  color=GREEN, alpha=0.85, edgecolor=BORDER)
        ax3.bar(x, shorts, label="Short", color=RED,   alpha=0.85, edgecolor=BORDER,
                bottom=longs)
        step = max(1, len(months) // 12)
        ax3.set_xticks(list(range(0, len(months), step)))
        ax3.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right", fontsize=7)
        ax3.set_ylabel("Trades")
        ax3.set_title("Monthly Trade Count (Long / Short)", color=TEXT)
        ax3.legend(fontsize=7.5)
        ax3.grid(True, axis="y", alpha=0.3)

    # Monthly win rate
    ax4 = axes[1, 1]
    if monthly:
        months = sorted(monthly.keys())
        wrs    = [monthly[m]["wins"] / monthly[m]["total"] * 100 if monthly[m]["total"] > 0 else 0 for m in months]
        colors = [GREEN if w >= 50 else AMBER if w >= 35 else RED for w in wrs]
        ax4.bar(range(len(months)), wrs, color=colors, alpha=0.85, edgecolor=BORDER)
        ax4.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
        step = max(1, len(months) // 12)
        ax4.set_xticks(list(range(0, len(months), step)))
        ax4.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right", fontsize=7)
        ax4.set_ylabel("Win Rate %")
        ax4.set_ylim(0, 100)
        ax4.set_title("Monthly Win Rate", color=TEXT)
        ax4.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, OUT_DIR / "06_trade_dist.png")


# ── Text summary ────────────────────────────────────────────────────────────────
def print_summary(summary: dict, trades: list[dict]) -> None:
    s = summary
    total_cands  = int(s["candidate_trades"])
    executed     = int(s["executed_trades"])
    skip_ticker  = int(s["skipped_same_ticker"])
    skip_cap     = int(s["skipped_no_capacity"])
    above_ema    = int(s["entries_above_ema"])
    below_ema    = int(s["entries_below_ema"])
    risk_on      = int(s["entries_ext_hours_risk_on_micro"])
    risk_off     = int(s["entries_ext_hours_risk_off_micro"])
    neutral_m    = int(s["entries_ext_hours_neutral_micro"])
    ext_scaled   = int(s["entries_ext_hours_proxy_scaled"])

    winners = [t for t in trades if (t["net_pnl"] or 0) > 0]
    losers  = [t for t in trades if (t["net_pnl"] or 0) <= 0]
    avg_win  = np.mean([float(t["net_pnl"]) for t in winners]) if winners else 0
    avg_loss = np.mean([float(t["net_pnl"]) for t in losers])  if losers  else 0
    expectancy = float(s["win_rate_pct"]) / 100 * avg_win + (1 - float(s["win_rate_pct"]) / 100) * avg_loss

    print()
    print("=" * 65)
    print("  SESSION TURTLE CORE x3 -- ema_hard_fg_half  REPORT SUMMARY")
    print("=" * 65)
    print(f"  Period     : {s['start_date'][:10]} to {s['end_date'][:10]}")
    print(f"  Universe   : {s['candidate_universe_size']} assets  (core basket)")
    print()
    print("  PERFORMANCE")
    print(f"    Total Return : {s['total_return_pct']:.2f}%  (${s['initial_capital']:,.0f} -> ${s['final_equity']:,.0f})")
    print(f"    CAGR         : {s['cagr_pct']:.2f}%")
    print(f"    Max Drawdown : {s['max_realized_drawdown_pct']:.2f}%")
    print(f"    Profit Factor: {s['profit_factor']:.2f}")
    print(f"    Win Rate     : {s['win_rate_pct']:.1f}%  ({len(winners)} wins / {len(losers)} losses)")
    print(f"    Avg Win      : ${avg_win:+,.2f}")
    print(f"    Avg Loss     : ${avg_loss:+,.2f}")
    print(f"    Expectancy   : ${expectancy:+,.2f} per trade")
    print()
    print("  CANDIDATE FUNNEL")
    print(f"    Raw candidates      : {total_cands:>5}")
    print(f"    Skip (same ticker)  : {skip_ticker:>5}  ({skip_ticker/total_cands*100:.1f}%)")
    print(f"    Skip (no capacity)  : {skip_cap:>5}  ({skip_cap/total_cands*100:.1f}%)")
    print(f"    Executed            : {executed:>5}  ({executed/total_cands*100:.1f}%)")
    print()
    print("  OVERLAY IMPACT")
    print(f"    EMA above (with-trend long active)  : {above_ema:>4} trades")
    print(f"    EMA below (with-trend short active) : {below_ema:>4} trades")
    print(f"    EMA hard-stop blocked               : crypto counter-trend entries suppressed")
    print(f"    Macro risk-on micro (shorts ~0x)    : {risk_on:>4} entries")
    print(f"    Macro neutral micro (both full)     : {neutral_m:>4} entries")
    print(f"    Macro risk-off micro (longs 0.5x)  : {risk_off:>4} entries")
    print(f"    Actually size-reduced (EH mult<1)  : {ext_scaled:>4} entries  (avg mult {s['avg_ext_hours_proxy_mult']:.3f})")
    print()
    print("  P&L BY BUCKET")
    for b in ["equity", "crypto", "gold", "metals"]:
        pnl    = float(s.get(f"{b}_pnl", 0))
        ntrades = int(s.get(f"{b}_trades", 0))
        bt     = [t for t in trades if t.get("asset_bucket") == b]
        wr     = (sum(1 for t in bt if (t["net_pnl"] or 0) > 0) / len(bt) * 100) if bt else 0
        print(f"    {b.capitalize():<8} : ${pnl:>10,.0f}   {ntrades:>3} trades   WR {wr:.0f}%")
    print()
    print("  FORWARD-LOOK BIAS GUARDS")
    print(f"    extended_hours_proxy_lag_days = {s['extended_hours_proxy_lag_days']}  (prior-day VIX/F&G)")
    print(f"    per_asset_ema_lag_days        = 1  (prior-day EMA)")
    print(f"    EMA warmup                    = 300 days before backtest start")
    print()


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not SUMMARY_FILE.exists() or not TRADES_FILE.exists():
        print("ERROR: Run run_ema_hard_fg_half_backtest.py first to generate report data.")
        sys.exit(1)

    print("\nLoading data...")
    summary, trades = _load()
    print(f"  {len(trades)} trades loaded")

    print_summary(summary, trades)

    print("Generating charts...")
    chart_equity(trades, summary)
    chart_candidate_funnel(summary)
    chart_overlay_regimes(trades)
    chart_winner_vs_loser(trades)
    chart_pnl_by_bucket(trades, summary)
    chart_trade_distributions(trades)

    print(f"\nAll charts saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
