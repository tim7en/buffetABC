"""Asset volatility & performance audit.

Groups all universe assets by volatility tier and theme, then cross-references
actual backtest trade performance from the latest run.

Output:
  reports/session_turtle_x3_document_review_20260403/asset_volatility_audit.json
  reports/session_turtle_x3_document_review_20260403/asset_volatility_audit.md
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "buffet.settings")
import django
django.setup()

# ── asset metadata ─────────────────────────────────────────────────────────────
# (requested_ticker, engine_ticker, source, bucket, theme, proxy_note)
ASSET_REGISTRY = [
    # crypto
    ("BTC-USD",   "BTC-USD",   "binance", "crypto",  "crypto_l1",    None),
    ("ETH-USD",   "ETH-USD",   "binance", "crypto",  "crypto_l1",    None),
    ("SOL-USD",   "SOL-USD",   "binance", "crypto",  "crypto_l1",    None),
    ("PAXG-USD",  "PAXG-USD",  "binance", "gold",    "crypto_gold",  None),
    # metals
    ("XAG-USD",   "SLV",       "tiingo",  "metals",  "precious_metals", "proxy=SLV"),
    ("XPD-USD",   "XPD-USD",   "tiingo",  "metals",  "precious_metals", None),
    ("XPT-USD",   "PPLT",      "tiingo",  "metals",  "precious_metals", "proxy=PPLT"),
    ("COPPER-USD","COPPER-USD","tiingo",  "metals",  "industrial_metals",None),
    # energy
    ("BZ-USD",    "BRENT",     "tiingo",  "energy",  "energy",       "proxy=BRENT"),
    ("NATGAS-USD","NATGAS-USD","tiingo",  "energy",  "energy",       None),
    # equities — high-beta / crypto-adjacent
    ("MSTR",      "MSTR",      "tiingo",  "equity",  "crypto_proxy", None),
    ("COIN",      "COIN",      "tiingo",  "equity",  "crypto_proxy", None),
    ("CRCL",      "CRCL",      "tiingo",  "equity",  "crypto_proxy", None),
    ("HOOD",      "HOOD",      "tiingo",  "equity",  "fintech",      None),
    ("PLTR",      "PLTR",      "tiingo",  "equity",  "ai_tech",      None),
    ("TSLA",      "TSLA",      "tiingo",  "equity",  "ev_disruptive",None),
    # equities — mega-cap tech
    ("NVDA",      "NVDA",      "tiingo",  "equity",  "mega_cap_tech",None),
    ("META",      "META",      "tiingo",  "equity",  "mega_cap_tech",None),
    ("GOOGL",     "GOOGL",     "tiingo",  "equity",  "mega_cap_tech",None),
    ("AMZN",      "AMZN",      "tiingo",  "equity",  "mega_cap_tech",None),
    ("INTC",      "INTC",      "tiingo",  "equity",  "mega_cap_tech",None),
    # equities — international ETFs
    ("EWJ",       "EWJ",       "tiingo",  "equity",  "intl_etf",     None),
    ("EWY",       "EWY",       "tiingo",  "equity",  "intl_etf",     None),
    # ── recently added (what-if only) ─────────────────────────────────────────
    ("QQQ",       "QQQ",       "tiingo",  "equity",  "broad_market_etf", None),
    ("SPY",       "SPY",       "tiingo",  "equity",  "broad_market_etf", None),
    ("TSM",       "TSM",       "tiingo",  "equity",  "semi_intl",    None),
    ("AAPL",      "AAPL",      "tiingo",  "equity",  "mega_cap_tech",None),
]

THEME_LABEL = {
    "crypto_l1":         "Crypto L1",
    "crypto_gold":       "Crypto-Gold Hybrid",
    "crypto_proxy":      "Crypto-Proxy Equity",
    "precious_metals":   "Precious Metals",
    "industrial_metals": "Industrial Metals",
    "energy":            "Energy / Commodities",
    "fintech":           "Fintech High-Beta",
    "ai_tech":           "AI / Disruptive Tech",
    "ev_disruptive":     "EV / Disruptive",
    "mega_cap_tech":     "Mega-Cap Tech",
    "intl_etf":          "International ETF",
    "semi_intl":         "International Semi",
    "broad_market_etf":  "Broad Market ETF",
}


def _annualised_vol(engine_ticker: str, source: str) -> float | None:
    """Return annualised daily-return volatility (%) from cached files."""
    try:
        if source == "binance":
            # binance data is gzipped CSV
            binance_ticker_map = {
                "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
                "SOL-USD": "SOLUSDT", "PAXG-USD": "PAXGUSDT",
            }
            bt = binance_ticker_map.get(engine_ticker)
            if bt is None:
                return None
            folder = ROOT / "cache" / "binance_asia_orb"
            gz_files = list(folder.glob(f"{bt}_*.csv.gz"))
            if not gz_files:
                return None
            df = pd.concat([pd.read_csv(f, compression="gzip") for f in gz_files])
            # columns vary — find timestamp col
            time_col = next((c for c in df.columns if "time" in c.lower() or "open_time" in c.lower()), df.columns[0])
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), "close")
            df["_dt"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
            df = df.set_index("_dt").sort_index()
            close = df[close_col].astype(float)
        else:
            # tiingo parquet — has a 'time' column, RangeIndex
            stem = engine_ticker.replace("-USD", "")
            pq_path = ROOT / "cache" / "cache" / "tiingo" / f"{stem}_5m.parquet"
            if not pq_path.exists():
                return None
            df = pd.read_parquet(pq_path)
            time_col = next((c for c in df.columns if c.lower() in ("time", "ts")), None)
            close_col = next((c for c in df.columns if c.lower() in ("close", "c")), None)
            if time_col is None or close_col is None:
                return None
            df["_dt"] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index("_dt").sort_index()
            close = df[close_col].astype(float)

        daily = close.resample("1D").last().dropna()
        ret = np.log(daily / daily.shift(1)).dropna()
        return round(float(ret.std()) * math.sqrt(252) * 100, 1)
    except Exception:
        return None


def _per_asset_trade_stats(trades: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: dict(
        n=0, wins=0, gross_win=0.0, gross_loss=0.0,
        total_pnl=0.0, total_notional=0.0, bucket="?",
    ))
    for t in trades:
        k = t["ticker"]
        pnl = float(t["net_pnl"])
        notional = abs(float(t["notional"]))
        stats[k]["n"] += 1
        stats[k]["total_pnl"] += pnl
        stats[k]["total_notional"] += notional
        stats[k]["bucket"] = t["asset_bucket"]
        if pnl > 0:
            stats[k]["wins"] += 1
            stats[k]["gross_win"] += pnl
        else:
            stats[k]["gross_loss"] += abs(pnl)
    return dict(stats)


def _vol_tier(vol: float | None) -> str:
    if vol is None:
        return "unknown"
    if vol >= 80:
        return "extreme (80%+)"
    if vol >= 50:
        return "very high (50-80%)"
    if vol >= 30:
        return "high (30-50%)"
    if vol >= 15:
        return "medium (15-30%)"
    return "low (<15%)"


def main():
    # load trades
    trades_path = ROOT / "reports" / "session_turtle_x3_document_review_20260403" / "trades.csv"
    trades = list(csv.DictReader(open(trades_path, encoding="utf-8")))
    trade_stats = _per_asset_trade_stats(trades)

    rows = []
    for req_ticker, eng_ticker, source, bucket, theme, proxy in ASSET_REGISTRY:
        vol = _annualised_vol(eng_ticker, source)
        tier = _vol_tier(vol)
        s = trade_stats.get(eng_ticker, {})
        n = s.get("n", 0)
        wr = s["wins"] / n * 100 if n > 0 else None
        pf = (s["gross_win"] / s["gross_loss"]
              if s.get("gross_loss", 0) > 0
              else (float("inf") if s.get("gross_win", 0) > 0 else None))
        avg_not = s["total_notional"] / n if n > 0 else None
        in_core = req_ticker not in {"QQQ", "SPY", "TSM", "AAPL"}

        rows.append({
            "ticker":         req_ticker,
            "engine_ticker":  eng_ticker,
            "source":         source,
            "bucket":         bucket,
            "theme":          theme,
            "theme_label":    THEME_LABEL[theme],
            "proxy":          proxy,
            "in_core_universe": in_core,
            "ann_vol_pct":    vol,
            "vol_tier":       tier,
            "trades":         n,
            "win_rate_pct":   round(wr, 1) if wr is not None else None,
            "profit_factor":  round(pf, 2) if pf is not None and pf != float("inf") else pf,
            "total_pnl":      round(s.get("total_pnl", 0), 2),
            "avg_notional":   round(avg_not, 2) if avg_not is not None else None,
        })

    # ── console output ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  ASSET UNIVERSE — VOLATILITY & THEME ANALYSIS")
    print(f"  Session Turtle X3  |  {len(rows)} assets  |  {len(trades)} trades")
    print(f"{'='*100}\n")

    # group by vol tier
    tier_order = [
        "extreme (80%+)", "very high (50-80%)", "high (30-50%)",
        "medium (15-30%)", "low (<15%)", "unknown",
    ]
    by_tier: dict[str, list] = defaultdict(list)
    for r in rows:
        by_tier[r["vol_tier"]].append(r)

    hdr = (f"  {'Ticker':<12} {'Theme':<22} {'Bucket':<9} {'Vol%':>6} "
           f"{'Trades':>6} {'WR%':>6} {'PF':>6} {'TotPnL':>10} {'Core':>5}")
    sep = "  " + "-" * 90

    for tier in tier_order:
        if tier not in by_tier:
            continue
        tier_rows = sorted(by_tier[tier], key=lambda r: -(r["total_pnl"] or 0))
        tier_pnl = sum(r["total_pnl"] or 0 for r in tier_rows)
        tier_trades = sum(r["trades"] for r in tier_rows)
        print(f"  ── {tier.upper()}  (total PnL: {tier_pnl:+,.2f}  |  trades: {tier_trades}) ──")
        print(hdr)
        print(sep)
        for r in tier_rows:
            vol_s = f"{r['ann_vol_pct']:.0f}%" if r["ann_vol_pct"] else "  n/a"
            wr_s  = f"{r['win_rate_pct']:.0f}%" if r["win_rate_pct"] is not None else "  —"
            pf_s  = f"{r['profit_factor']:.2f}" if r["profit_factor"] is not None and r["profit_factor"] != float("inf") else ("inf" if r["profit_factor"] == float("inf") else "  —")
            core_s = "YES" if r["in_core_universe"] else " no"
            proxy_s = f" [{r['proxy']}]" if r["proxy"] else ""
            print(f"  {r['ticker']:<12} {r['theme_label']:<22} {r['bucket']:<9} {vol_s:>6} "
                  f"{r['trades']:>6} {wr_s:>6} {pf_s:>6} {r['total_pnl']:>+10.2f} {core_s:>5}{proxy_s}")
        print()

    # ── theme-level summary ────────────────────────────────────────────────────
    print(f"  {'='*100}")
    print(f"  THEME-LEVEL SUMMARY (core universe only)")
    print(f"  {'='*100}")
    theme_stats: dict[str, dict] = defaultdict(lambda: dict(n_assets=0, trades=0, total_pnl=0.0, wins=0, losses=0))
    for r in rows:
        if not r["in_core_universe"]:
            continue
        th = r["theme_label"]
        theme_stats[th]["n_assets"] += 1
        theme_stats[th]["trades"] += r["trades"]
        theme_stats[th]["total_pnl"] += r["total_pnl"] or 0

    print(f"\n  {'Theme':<24} {'Assets':>6} {'Trades':>7} {'Total PnL':>12}")
    print("  " + "-" * 55)
    for th, s in sorted(theme_stats.items(), key=lambda x: -x[1]["total_pnl"]):
        print(f"  {th:<24} {s['n_assets']:>6} {s['trades']:>7} {s['total_pnl']:>+12.2f}")
    print()

    # ── save outputs ───────────────────────────────────────────────────────────
    out_dir = ROOT / "reports" / "session_turtle_x3_document_review_20260403"

    json_path = out_dir / "asset_volatility_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"assets": rows}, f, indent=2, default=str)
    print(f"  JSON saved  -> {json_path}")

    # markdown
    md_lines = [
        "# Asset Universe — Volatility & Theme Audit",
        "",
        "**Strategy:** Session Turtle X3  |  **Run date:** 2026-04-03",
        "",
        "## Per-Asset Table",
        "",
        "| Ticker | Theme | Bucket | Ann.Vol% | Vol Tier | Trades | WR% | PF | Total PnL | Core |",
        "|---|---|---|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(rows, key=lambda x: -(x["ann_vol_pct"] or 0)):
        vol_s = f"{r['ann_vol_pct']:.0f}%" if r["ann_vol_pct"] else "n/a"
        wr_s  = f"{r['win_rate_pct']:.0f}%" if r["win_rate_pct"] is not None else "—"
        pf_s  = f"{r['profit_factor']:.2f}" if (r["profit_factor"] is not None and r["profit_factor"] != float("inf")) else ("inf" if r["profit_factor"] == float("inf") else "—")
        core_s = "YES" if r["in_core_universe"] else "—"
        ticker_s = f"{r['ticker']}*" if r["proxy"] else r["ticker"]
        md_lines.append(
            f"| {ticker_s} | {r['theme_label']} | {r['bucket']} | {vol_s} "
            f"| {r['vol_tier']} | {r['trades']} | {wr_s} | {pf_s} "
            f"| {r['total_pnl']:+,.2f} | {core_s} |"
        )

    md_lines += [
        "",
        "> \\* Proxy ticker used (see engine_ticker in JSON for mapping)",
        "",
        "## Theme Summary (Core Universe)",
        "",
        "| Theme | Assets | Trades | Total PnL |",
        "|---|---:|---:|---:|",
    ]
    for th, s in sorted(theme_stats.items(), key=lambda x: -x[1]["total_pnl"]):
        md_lines.append(f"| {th} | {s['n_assets']} | {s['trades']} | {s['total_pnl']:+,.2f} |")

    md_lines += [
        "",
        "## Key Hypotheses",
        "",
        "| # | Hypothesis | Evidence |",
        "|---|---|---|",
        "| 1 | High-beta/crypto assets drive most of the PnL | Check extreme + very-high vol tier PnL share |",
        "| 2 | Mega-cap ETFs (SPY/QQQ) dilute returns | Expanded what-if: -33% to -97% CAGR across all scenarios |",
        "| 3 | Crypto-proxy equities (MSTR, COIN, CRCL) add alpha beyond raw crypto | Compare theme PnL vs crypto L1 |",
        "| 4 | International ETFs (EWJ, EWY) are low-vol drag | Check medium vol tier contribution |",
    ]

    md_path = out_dir / "asset_volatility_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  Markdown saved -> {md_path}")


if __name__ == "__main__":
    main()
