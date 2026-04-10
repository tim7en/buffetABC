"""Build an SP500-target macro-regime dataset with QQQ as a growth variable.

This is a compatibility wrapper around the existing QQQ macro regime feature
engine. Internally, the reused engine still expects ``qqq_*`` target columns, so
this script feeds SPY prices into that target slot and then adds actual QQQ
variables under ``growth_qqq_*`` names.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

import qqq_macro_ml_regime_analysis as regime_analysis


ROOT = regime_analysis.ROOT
DEFAULT_SP500_PATH = ROOT / "cache" / "cache" / "cache" / "SPY_daily.parquet"
DEFAULT_QQQ_PATH = regime_analysis.DEFAULT_QQQ_PATH
DEFAULT_MACRO_PATH = regime_analysis.DEFAULT_MACRO_PATH
DEFAULT_FRED_CACHE_DIR = regime_analysis.DEFAULT_FRED_CACHE_DIR
DEFAULT_OUT_DIR = ROOT / "reports" / "sp500_macro_regime_with_qqq_20260410" / "analysis"
DEFAULT_START = "1999-03-10"
DEFAULT_SP500_TICKER = "SPY"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

QQQ_GROWTH_FEATURES = [
    "growth_qqq_21d_return",
    "growth_qqq_63d_return",
    "growth_qqq_126d_return",
    "growth_qqq_252d_return",
    "growth_qqq_vs_sma200",
    "growth_qqq_realized_vol_21d",
    "growth_qqq_drawdown_252d",
    "growth_qqq_minus_sp500_63d_return",
    "growth_qqq_beta_ratio_63d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500-ticker", default=DEFAULT_SP500_TICKER)
    parser.add_argument("--sp500-path", type=Path, default=DEFAULT_SP500_PATH)
    parser.add_argument("--qqq-path", type=Path, default=DEFAULT_QQQ_PATH)
    parser.add_argument("--macro-path", type=Path, default=DEFAULT_MACRO_PATH)
    parser.add_argument("--fred-cache-dir", type=Path, default=DEFAULT_FRED_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh-sp500", action="store_true")
    parser.add_argument("--target-horizon", type=int, default=63)
    parser.add_argument("--monthly-release-lag-days", type=int, default=45)
    parser.add_argument("--quarterly-release-lag-days", type=int, default=45)
    parser.add_argument("--annual-release-lag-days", type=int, default=365)
    return parser.parse_args()


def request_json(session: requests.Session, url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(
                url,
                params=params,
                timeout=45,
                headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-macro-regime/1.0)"},
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2.0 * (attempt + 1), 20.0))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def download_yahoo_daily(ticker: str, start: str, end: str | None, output_path: Path) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    if end_date < start_date:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    yahoo_end = end_date + timedelta(days=1)
    end_dt = datetime(yahoo_end.year, yahoo_end.month, yahoo_end.day, tzinfo=timezone.utc)
    quoted_ticker = quote(ticker, safe="")
    url = YAHOO_CHART_URL.format(ticker=quoted_ticker)
    payload = request_json(
        requests.Session(),
        url,
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "div,splits,capitalGains",
            "includeAdjustedClose": "true",
        },
    )
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {ticker}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo chart returned no data for {ticker}")

    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    frame = pd.DataFrame({"timestamp": result.get("timestamp") or []})
    for column, output in [
        ("open", "o"),
        ("high", "h"),
        ("low", "l"),
        ("close", "c"),
        ("volume", "v"),
    ]:
        values = quote_data.get(column)
        frame[output] = values if values is not None else pd.NA
    frame["adj_c"] = adjclose if adjclose is not None else pd.NA
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame["date"] = frame["time"].dt.date
    frame["ts"] = frame["timestamp"].astype("int64") * 1000
    for column in ["o", "h", "l", "c", "adj_c"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["v"] = pd.to_numeric(frame["v"], errors="coerce").fillna(0.0).astype("int64")
    frame = frame.dropna(subset=["c"]).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[["date", "time", "ts", "o", "h", "l", "c", "adj_c", "v"]].to_parquet(output_path, index=False)
    metadata = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "source": "Yahoo Finance chart API",
        "source_url": url,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "output_parquet": str(output_path),
        "rows": int(len(frame)),
        "first_date": frame["date"].min().isoformat() if not frame.empty else None,
        "last_date": frame["date"].max().isoformat() if not frame.empty else None,
    }
    output_path.with_name(f"{output_path.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_market(path: Path, start: str | None, end: str | None, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing daily parquet: {path}")
    raw = pd.read_parquet(path).copy()
    if "date" in raw.columns:
        index = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    elif "time" in raw.columns:
        index = pd.to_datetime(raw["time"]).dt.tz_localize(None)
    else:
        raise ValueError(f"{path} must contain 'date' or 'time'.")
    close_col = "adj_c" if "adj_c" in raw.columns else "c"
    volume_col = "v" if "v" in raw.columns else "volume"
    frame = pd.DataFrame(
        {
            f"{prefix}_close": pd.to_numeric(raw[close_col], errors="coerce").to_numpy(),
            f"{prefix}_volume": pd.to_numeric(raw[volume_col], errors="coerce").to_numpy(),
        },
        index=index,
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna(subset=[f"{prefix}_close"])
    if start:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame[frame.index <= pd.Timestamp(end)]
    if len(frame) < 756:
        raise ValueError(f"Only {len(frame)} rows after filtering {path}; need about 3 years.")
    return frame


def add_sp500_aliases(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    for column in list(out.columns):
        if column.startswith("qqq_"):
            out[f"sp500_{column.removeprefix('qqq_')}"] = out[column]
    return out


def add_qqq_growth_features(dataset: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    close = regime_analysis._align_daily(qqq["qqq_close"], out.index)
    volume = regime_analysis._align_daily(qqq["qqq_volume"], out.index)
    returns = close.pct_change()
    out["growth_qqq_close"] = close
    out["growth_qqq_volume"] = volume
    out["growth_qqq_21d_return"] = regime_analysis._pct_change(close, 21)
    out["growth_qqq_63d_return"] = regime_analysis._pct_change(close, 63)
    out["growth_qqq_126d_return"] = regime_analysis._pct_change(close, 126)
    out["growth_qqq_252d_return"] = regime_analysis._pct_change(close, 252)
    qqq_sma200 = close.rolling(200, min_periods=200).mean()
    out["growth_qqq_vs_sma200"] = close / qqq_sma200 - 1.0
    out["growth_qqq_realized_vol_21d"] = returns.rolling(21, min_periods=21).std() * (
        regime_analysis.TRADING_DAYS_PER_YEAR ** 0.5
    )
    out["growth_qqq_drawdown_252d"] = close / close.rolling(252, min_periods=63).max() - 1.0
    out["growth_qqq_minus_sp500_63d_return"] = out["growth_qqq_63d_return"] - out["sp500_63d_return"]
    out["growth_qqq_beta_ratio_63d"] = out["growth_qqq_63d_return"] / out["sp500_63d_return"].replace(0.0, pd.NA)
    return out


def write_feature_manifest(out_dir: Path, dataset: pd.DataFrame) -> None:
    base_features = regime_analysis.available_features(dataset, regime_analysis.MODEL_FEATURES, min_non_na=96)
    qqq_features = regime_analysis.available_features(dataset, QQQ_GROWTH_FEATURES, min_non_na=96)
    rows = []
    for feature in base_features:
        rows.append({"feature": feature, "role": "sp500_target_compatible_feature"})
    for feature in qqq_features:
        rows.append({"feature": feature, "role": "qqq_growth_variable"})
    pd.DataFrame(rows).to_csv(out_dir / "sp500_supervised_feature_manifest.csv", index=False)
    (out_dir / "qqq_growth_feature_args.txt").write_text(" ".join(qqq_features), encoding="utf-8")


def write_report(out_dir: Path, dataset: pd.DataFrame, sp500_ticker: str) -> None:
    latest = dataset.iloc[-1]
    lines = [
        "# SP500 Macro Regime Dataset With QQQ Variables",
        "",
        f"- Target/backtest proxy: `{sp500_ticker}` adjusted close, saved in `sp500_close`.",
        "- Compatibility note: `qqq_close` and `qqq_fwd_*` are SP500 target columns in this folder only, so existing walk-forward code can run without a broad refactor.",
        "- Actual QQQ inputs are stored separately as `growth_qqq_*` columns and can be passed through `--extra-model-features`.",
        "- This keeps QQQ as a higher-beta growth expression rather than the primary regime target.",
        "",
        "## Latest Row",
        "",
        f"- Date: `{dataset.index[-1].date()}`",
        f"- SP500 proxy close: `{latest['sp500_close']:.2f}`",
        f"- QQQ close: `{latest['growth_qqq_close']:.2f}`",
        f"- SP500 63d return: `{latest['sp500_63d_return'] * 100:.2f}%`",
        f"- QQQ 63d return: `{latest['growth_qqq_63d_return'] * 100:.2f}%`",
        "",
        "## Output Files",
        "",
        "- `aligned_daily_dataset.csv`: SP500 target dataset plus QQQ growth variables.",
        "- `sp500_supervised_feature_manifest.csv`: feature roles for the supervised models.",
        "- `qqq_growth_feature_args.txt`: space-separated QQQ features for the compare script.",
    ]
    (out_dir / "sp500_macro_regime_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_sp500 or not args.sp500_path.exists():
        download_yahoo_daily(args.sp500_ticker, args.start, args.end, args.sp500_path)

    sp500 = load_market(args.sp500_path, args.start, args.end, "sp500")
    qqq = regime_analysis.load_qqq(args.qqq_path, args.start, args.end)
    target_frame = pd.DataFrame(
        {
            "qqq_close": sp500["sp500_close"],
            "qqq_volume": sp500["sp500_volume"],
        },
        index=sp500.index,
    )
    macro = regime_analysis.load_macro(
        args.macro_path,
        target_frame.index,
        args.monthly_release_lag_days,
        args.quarterly_release_lag_days,
        args.annual_release_lag_days,
    )
    stress, fred_status = regime_analysis.load_stress_proxies(target_frame.index, args.fred_cache_dir, refresh=False)
    dataset = regime_analysis.build_dataset(target_frame, macro, stress, args.target_horizon)
    dataset = add_sp500_aliases(dataset)
    dataset = add_qqq_growth_features(dataset, qqq)

    dataset.to_csv(args.out_dir / "aligned_daily_dataset.csv", index_label="date")
    pd.DataFrame(
        [{"series": key, "status": value} for key, value in fred_status.items()]
    ).to_csv(args.out_dir / "fred_status.csv", index=False)
    write_feature_manifest(args.out_dir, dataset)
    write_report(args.out_dir, dataset, args.sp500_ticker)
    metadata = {
        "target_proxy": args.sp500_ticker,
        "sp500_path": str(args.sp500_path),
        "qqq_path": str(args.qqq_path),
        "macro_path": str(args.macro_path),
        "start": args.start,
        "end": args.end,
        "target_horizon": args.target_horizon,
        "rows": int(len(dataset)),
        "first_date": dataset.index.min().date().isoformat(),
        "last_date": dataset.index.max().date().isoformat(),
        "qqq_growth_features": QQQ_GROWTH_FEATURES,
        "compatibility_note": "qqq_* target columns represent SP500/SPY in this folder; growth_qqq_* columns represent actual QQQ.",
    }
    (args.out_dir / "sp500_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote SP500 macro dataset to {args.out_dir}")
    print(f"Rows: {len(dataset):,} | {dataset.index.min().date()} -> {dataset.index.max().date()}")
    print(f"Latest {args.sp500_ticker} close: {dataset['sp500_close'].iloc[-1]:.2f}")
    print(f"Latest QQQ close: {dataset['growth_qqq_close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
