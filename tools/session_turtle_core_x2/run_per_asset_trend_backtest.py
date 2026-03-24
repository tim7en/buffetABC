"""
Per-asset standalone trend backtest — 4H EMA(55/200) filter + Donchian(20) channel.

For every (ticker, session) combo in the core universe, runs three variants:
  - Long only   (trend filter: only when fast > slow)
  - Short only  (trend filter: only when fast < slow)
  - Combined    (both directions)

Shows which assets have genuine trending runs in each direction.
"""
from __future__ import annotations

import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")

import django
django.setup()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from edgar.services.session_turtle_portfolio import CORE_SESSION_TURTLE_UNIVERSE
from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest

OUTPUT_DIR = Path("reports/per_asset_trend_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
LOOKBACK_YEARS  = 4.1

BUCKET_MAP = {
    "BTC-USD": "crypto",  "ETH-USD": "crypto",  "SOL-USD": "crypto",
    "PAXG-USD": "gold",   "GLD": "gold",
    "AMZN": "equity",     "COIN": "equity",      "CRCL": "equity",
    "HOOD": "equity",     "INTC": "equity",       "MSTR": "equity",
    "PLTR": "equity",     "TSLA": "equity",
    "COPPER": "metals",   "PPLT": "metals",       "SLV": "metals",
}
BUCKET_COLORS = {
    "equity": "#1565C0", "crypto": "#6A1B9A",
    "gold":   "#F9A825", "metals": "#558B2F",
}

BASE_KWARGS = dict(
    initial_capital=INITIAL_CAPITAL,
    lookback_years=LOOKBACK_YEARS,
    interval="5m",
    channel_period=20,
    atr_period=20,
    fixed_stop_pct=0.10,
    base_risk_pct=0.05,
    directional_volume_risk_pct=0.07,
    use_4h_trend_filter=True,
    trend_fast_period=55,
    trend_slow_period=200,
    slippage_bps=2.0,
    commission_bps=1.0,
)


def _safe_run(ticker, source, session, allow_longs, allow_shorts):
    try:
        return run_session_turtle_trend_backtest(
            ticker=ticker,
            market_data_source=source,
            session_open=session,
            allow_longs=allow_longs,
            allow_shorts=allow_shorts,
            **BASE_KWARGS,
        )
    except Exception as e:
        return {"error": str(e)}


def _metrics(result: dict) -> dict:
    if "error" in result:
        return {"error": result["error"]}
    trades = result.get("trades", [])
    n = len(trades)
    if n == 0:
        return {"trades": 0, "total_return_pct": 0, "cagr_pct": 0,
                "max_dd_pct": 0, "win_rate": 0, "payoff_ratio": 0,
                "expectancy": 0, "profit_factor": 0}

    pnls       = [t.pnl for t in trades]
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p < 0]
    final_cap  = result.get("final_capital", INITIAL_CAPITAL)
    total_ret  = (final_cap / INITIAL_CAPITAL - 1) * 100
    eq_curve   = result.get("equity_curve", [])
    max_dd     = result.get("max_drawdown_pct", 0) * 100

    # CAGR
    years = LOOKBACK_YEARS
    cagr  = ((final_cap / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    win_rate    = len(wins) / n * 100 if n else 0
    avg_win     = np.mean(wins)   if wins   else 0
    avg_loss    = abs(np.mean(losses)) if losses else 0
    payoff      = avg_win / avg_loss if avg_loss > 0 else 999
    expectancy  = np.mean(pnls)
    pf          = sum(wins) / abs(sum(losses)) if losses else 999

    # Sharpe from equity curve
    sharpe = 0.0
    if eq_curve and len(eq_curve) > 2:
        eq_vals = [e["equity"] for e in eq_curve]
        rets    = np.diff(eq_vals) / np.array(eq_vals[:-1])
        rfr     = (1.05 ** (1/252)) - 1
        excess  = rets - rfr
        if excess.std() > 1e-9:
            sharpe = float(excess.mean() / excess.std() * np.sqrt(252))

    return dict(
        trades=n,
        total_return_pct=round(total_ret, 2),
        cagr_pct=round(cagr, 2),
        max_dd_pct=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        payoff_ratio=round(payoff, 2),
        expectancy=round(expectancy, 2),
        profit_factor=round(pf, 3),
        sharpe=round(sharpe, 3),
    )


def main():
    # deduplicate by (ticker, session) — each combo runs once combined, long-only, short-only
    combos = list(dict.fromkeys(
        (ticker, source, session)
        for ticker, source, session in CORE_SESSION_TURTLE_UNIVERSE
    ))

    rows = []
    print(f"\nRunning {len(combos)} asset-session combos x3 variants...\n")

    for ticker, source, session in combos:
        label = f"{ticker} / {session.replace('_open','').replace('_',' ')}"
        bucket = BUCKET_MAP.get(ticker, "equity")
        print(f"  {label} ...", end=" ", flush=True)

        r_long  = _safe_run(ticker, source, session, allow_longs=True,  allow_shorts=False)
        r_short = _safe_run(ticker, source, session, allow_longs=False, allow_shorts=True)
        r_both  = _safe_run(ticker, source, session, allow_longs=True,  allow_shorts=True)

        m_long  = _metrics(r_long)
        m_short = _metrics(r_short)
        m_both  = _metrics(r_both)

        rows.append(dict(
            ticker=ticker, session=session, bucket=bucket,
            label=label,
            long_trades=m_long.get("trades", 0),
            long_return=m_long.get("total_return_pct", 0),
            long_cagr=m_long.get("cagr_pct", 0),
            long_dd=m_long.get("max_dd_pct", 0),
            long_wr=m_long.get("win_rate", 0),
            long_payoff=m_long.get("payoff_ratio", 0),
            long_sharpe=m_long.get("sharpe", 0),
            short_trades=m_short.get("trades", 0),
            short_return=m_short.get("total_return_pct", 0),
            short_cagr=m_short.get("cagr_pct", 0),
            short_dd=m_short.get("max_dd_pct", 0),
            short_wr=m_short.get("win_rate", 0),
            short_payoff=m_short.get("payoff_ratio", 0),
            short_sharpe=m_short.get("sharpe", 0),
            both_trades=m_both.get("trades", 0),
            both_return=m_both.get("total_return_pct", 0),
            both_cagr=m_both.get("cagr_pct", 0),
            both_dd=m_both.get("max_dd_pct", 0),
            both_wr=m_both.get("win_rate", 0),
            both_sharpe=m_both.get("sharpe", 0),
            long_error=r_long.get("error", ""),
            short_error=r_short.get("error", ""),
        ))
        print(f"L:{m_long.get('cagr_pct',0):+.0f}%  S:{m_short.get('cagr_pct',0):+.0f}%  B:{m_both.get('cagr_pct',0):+.0f}%")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "per_asset_trend_results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n{'Asset / Session':<38}  {'--- LONG ---':^28}  {'--- SHORT ---':^28}  {'--- BOTH ---':^20}")
    print(f"{'':38}  {'CAGR%':>7} {'MaxDD%':>7} {'Payoff':>6} {'WR%':>5}  "
          f"{'CAGR%':>7} {'MaxDD%':>7} {'Payoff':>6} {'WR%':>5}  "
          f"{'CAGR%':>7} {'MaxDD%':>7}")
    print("-" * 118)
    for r in sorted(rows, key=lambda x: x["both_cagr"], reverse=True):
        print(
            f"{r['label']:<38}  "
            f"{r['long_cagr']:>7.1f} {r['long_dd']:>7.1f} {r['long_payoff']:>6.2f} {r['long_wr']:>5.1f}  "
            f"{r['short_cagr']:>7.1f} {r['short_dd']:>7.1f} {r['short_payoff']:>6.2f} {r['short_wr']:>5.1f}  "
            f"{r['both_cagr']:>7.1f} {r['both_dd']:>7.1f}"
        )

    # ── plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Per-Asset Trend Backtest: Long vs Short Run Performance\n"
                 "4H EMA(55/200) filter + Donchian(20) + ATR stop, 4.1yr window",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_cagr  = fig.add_subplot(gs[0, :])
    ax_pay   = fig.add_subplot(gs[1, 0])
    ax_dd    = fig.add_subplot(gs[1, 1])

    # sort by combined CAGR
    sorted_rows = sorted(rows, key=lambda x: x["both_cagr"], reverse=True)
    labels  = [r["label"] for r in sorted_rows]
    bcolors = [BUCKET_COLORS[r["bucket"]] for r in sorted_rows]
    x = np.arange(len(labels))
    w = 0.28

    # 1. CAGR grouped bars
    ax_cagr.bar(x - w,   [r["long_cagr"]  for r in sorted_rows], w,
                label="Long only",  color="#42A5F5", alpha=0.85)
    ax_cagr.bar(x,       [r["short_cagr"] for r in sorted_rows], w,
                label="Short only", color="#EF5350", alpha=0.85)
    ax_cagr.bar(x + w,   [r["both_cagr"]  for r in sorted_rows], w,
                label="Combined",   color="#66BB6A", alpha=0.85)
    ax_cagr.axhline(0, color="black", linewidth=0.8)
    ax_cagr.set_xticks(x)
    ax_cagr.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_cagr.set_ylabel("CAGR %")
    ax_cagr.set_title("CAGR by Asset & Session — Long vs Short vs Combined", fontsize=12)
    ax_cagr.legend(fontsize=10)
    ax_cagr.grid(alpha=0.3, axis="y")

    # 2. Payoff ratio scatter: long vs short
    for r in rows:
        c = BUCKET_COLORS[r["bucket"]]
        lp = min(r["long_payoff"],  15)
        sp = min(r["short_payoff"], 15)
        ax_pay.scatter(lp, sp, color=c, s=80, alpha=0.8, zorder=3)
        ax_pay.annotate(r["ticker"], (lp, sp),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax_pay.axline((0, 0), slope=1, color="grey", linewidth=0.8, linestyle="--", label="Long = Short")
    ax_pay.axhline(2, color="red",   linewidth=0.8, linestyle=":", alpha=0.6)
    ax_pay.axvline(2, color="blue",  linewidth=0.8, linestyle=":", alpha=0.6)
    ax_pay.set_xlabel("Long payoff ratio")
    ax_pay.set_ylabel("Short payoff ratio")
    ax_pay.set_title("Payoff Ratio: Long vs Short\n(dashed = parity, lines = 2x threshold)", fontsize=11)
    ax_pay.grid(alpha=0.3)
    from matplotlib.patches import Patch
    ax_pay.legend(handles=[Patch(fc=c, label=b) for b, c in BUCKET_COLORS.items()] +
                  [plt.Line2D([0],[0], color="grey", linestyle="--", label="parity")],
                  fontsize=8)

    # 3. Max DD: long vs short bars
    x2 = np.arange(len(sorted_rows))
    ax_dd.bar(x2 - w/2, [-r["long_dd"]  for r in sorted_rows], w,
              label="Long DD",  color="#42A5F5", alpha=0.85)
    ax_dd.bar(x2 + w/2, [-r["short_dd"] for r in sorted_rows], w,
              label="Short DD", color="#EF5350", alpha=0.85)
    ax_dd.set_xticks(x2)
    ax_dd.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_dd.set_ylabel("Max Drawdown %")
    ax_dd.set_title("Max Drawdown — Long vs Short", fontsize=11)
    ax_dd.axhline(-35, color="orange", linewidth=1, linestyle="--", label="35% DD threshold")
    ax_dd.legend(fontsize=9)
    ax_dd.grid(alpha=0.3, axis="y")

    fig.savefig(OUTPUT_DIR / "per_asset_trend_backtest.png", dpi=150, bbox_inches="tight")
    print(f"Plot -> {OUTPUT_DIR / 'per_asset_trend_backtest.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
