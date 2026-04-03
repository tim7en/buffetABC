"""Compute long vs short breakdown per asset for the investor report."""
import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
trades = list(csv.DictReader(open(
    ROOT / 'reports/session_turtle_x3_document_review_20260403/trades.csv',
    encoding='utf-8'
)))

VOL = {
    "BTC-USD": 43, "ETH-USD": 58, "SOL-USD": 81,
    "PAXG-USD": 15, "SLV": 39, "XPD-USD": 42,
    "PPLT": 33, "COPPER-USD": 27, "BRENT": 33,
    "NATGAS-USD": 89, "AMZN": 31, "COIN": 81,
    "CRCL": 113, "GOOGL": 136, "HOOD": 62,
    "INTC": 54, "META": 43, "MSTR": 150,
    "NVDA": 128, "PLTR": 64, "TSLA": 58,
    "EWJ": 18, "EWY": 26,
    "QQQ": 22, "SPY": 16, "TSM": 36, "AAPL": 27,
}

THEME = {
    "BTC-USD": "Crypto L1", "ETH-USD": "Crypto L1", "SOL-USD": "Crypto L1",
    "PAXG-USD": "Crypto-Gold", "SLV": "Precious Metals",
    "XPD-USD": "Precious Metals", "PPLT": "Precious Metals",
    "COPPER-USD": "Industrial Metals", "BRENT": "Energy",
    "NATGAS-USD": "Energy", "COIN": "Crypto-Proxy",
    "CRCL": "Crypto-Proxy", "MSTR": "Crypto-Proxy",
    "HOOD": "Fintech HiBeta", "PLTR": "AI/Disruptive",
    "TSLA": "EV/Disruptive", "NVDA": "Mega-Cap Tech",
    "META": "Mega-Cap Tech", "GOOGL": "Mega-Cap Tech",
    "AMZN": "Mega-Cap Tech", "INTC": "Mega-Cap Tech",
    "EWJ": "Intl ETF", "EWY": "Intl ETF",
    "QQQ": "Broad ETF", "SPY": "Broad ETF",
    "TSM": "Intl Semi", "AAPL": "Mega-Cap Tech",
}

stats = defaultdict(lambda: dict(
    long_n=0, short_n=0,
    long_wins=0, short_wins=0,
    long_pnl=0.0, short_pnl=0.0,
    long_gw=0.0, long_gl=0.0,
    short_gw=0.0, short_gl=0.0,
    bucket='?'
))

for t in trades:
    k = t['ticker']
    pnl = float(t['net_pnl'])
    stats[k]['bucket'] = t['asset_bucket']
    if t['direction'] == 'long':
        stats[k]['long_n'] += 1
        stats[k]['long_pnl'] += pnl
        if pnl > 0:
            stats[k]['wins_long'] = stats[k].get('wins_long', 0) + 1
            stats[k]['long_wins'] += 1
            stats[k]['long_gw'] += pnl
        else:
            stats[k]['long_gl'] += abs(pnl)
    else:
        stats[k]['short_n'] += 1
        stats[k]['short_pnl'] += pnl
        if pnl > 0:
            stats[k]['short_wins'] += 1
            stats[k]['short_gw'] += pnl
        else:
            stats[k]['short_gl'] += abs(pnl)

rows = []
for k, s in sorted(stats.items(), key=lambda x: -(x[1]['long_pnl'] + x[1]['short_pnl'])):
    lwr = s['long_wins'] / s['long_n'] * 100 if s['long_n'] else None
    swr = s['short_wins'] / s['short_n'] * 100 if s['short_n'] else None
    lpf = s['long_gw'] / s['long_gl'] if s['long_gl'] > 0 else (float('inf') if s['long_gw'] > 0 else None)
    spf = s['short_gw'] / s['short_gl'] if s['short_gl'] > 0 else (float('inf') if s['short_gw'] > 0 else None)

    total_pnl = round(s['long_pnl'] + s['short_pnl'], 2)
    long_share = round(s['long_pnl'] / total_pnl * 100, 1) if total_pnl != 0 else None

    rows.append({
        'ticker': k,
        'bucket': s['bucket'],
        'theme': THEME.get(k, '?'),
        'vol_pct': VOL.get(k),
        'total_pnl': total_pnl,
        'long_n': s['long_n'],
        'long_pnl': round(s['long_pnl'], 2),
        'long_wr_pct': round(lwr, 1) if lwr is not None else None,
        'long_pf': round(lpf, 2) if lpf is not None and lpf != float('inf') else lpf,
        'short_n': s['short_n'],
        'short_pnl': round(s['short_pnl'], 2),
        'short_wr_pct': round(swr, 1) if swr is not None else None,
        'short_pf': round(spf, 2) if spf is not None and spf != float('inf') else spf,
        'long_pnl_share_pct': long_share,
        'short_role': (
            'earning' if s['short_pnl'] > 200 else
            'protecting' if s['short_pnl'] > -50 else
            'drag'
        ),
    })

out = ROOT / 'reports/session_turtle_x3_document_review_20260403/longshort_analysis.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(rows, f, indent=2, default=str)
print(f'Saved -> {out}')

# print table
print(f"\n{'Ticker':<13} {'Theme':<18} {'Vol':>4} {'TotPnL':>9}  |  {'LN':>3} {'LWR':>5} {'LPnL':>9} {'LPF':>5}  |  {'SN':>3} {'SWR':>5} {'SPnL':>9} {'SPF':>5}  Role")
print('-' * 115)
for r in rows:
    lwr_s = f"{r['long_wr_pct']:.0f}%" if r['long_wr_pct'] is not None else '  —'
    swr_s = f"{r['short_wr_pct']:.0f}%" if r['short_wr_pct'] is not None else '  —'
    lpf_s = f"{r['long_pf']:.2f}" if r['long_pf'] is not None and r['long_pf'] != float('inf') else ('inf' if r['long_pf'] == float('inf') else '  —')
    spf_s = f"{r['short_pf']:.2f}" if r['short_pf'] is not None and r['short_pf'] != float('inf') else ('inf' if r['short_pf'] == float('inf') else '  —')
    print(f"{r['ticker']:<13} {r['theme']:<18} {str(r['vol_pct'] or '?'):>4} {r['total_pnl']:>+9.2f}  |  {r['long_n']:>3} {lwr_s:>5} {r['long_pnl']:>+9.2f} {lpf_s:>5}  |  {r['short_n']:>3} {swr_s:>5} {r['short_pnl']:>+9.2f} {spf_s:>5}  {r['short_role']}")
