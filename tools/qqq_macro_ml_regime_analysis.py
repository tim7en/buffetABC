"""Industry-grade QQQ macro regime and DCA research workflow.

This script is research tooling, not investment advice. It is designed to make
the analysis reproducible and harder to fool with obvious look-ahead mistakes:

- QQQ prices and daily macro data are aligned on QQQ trading days.
- Sparse CPI / unemployment prints are lagged before forward filling because
  their observation month is not the same as the public release date.
- Predictive tests use month-end samples and Newey-West standard errors to
  reduce the false precision caused by overlapping forward returns.
- ML validation is chronological and purged so a training row whose forward
  return window overlaps the test start is excluded.
- DCA regime signals are generated in walk-forward form before the backtest.

The "sentiment" variable is intentionally treated as a latent black box. We do
not claim to observe sentiment directly; instead we test a transparent proxy
driven by QQQ feedback, volatility/credit/financial-condition shocks, dollar,
rates, oil, CPI, and unemployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QQQ_PATH = ROOT / "cache" / "cache" / "cache" / "QQQ_daily.parquet"
DEFAULT_MACRO_PATH = ROOT / "cache" / "cache" / "macro_daily_1999.parquet"
DEFAULT_OUT_DIR = ROOT / "reports" / "qqq_macro_ml_regime_analysis"
DEFAULT_FRED_CACHE_DIR = ROOT / "cache" / "fred"
TRADING_DAYS_PER_YEAR = 252
MONTHLY_TRADING_DAYS = 21
RANDOM_STATE = 42

FRED_STRESS_SERIES = {
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
    "nfci": "NFCI",
    "t10y3m": "T10Y3M",
}

DAILY_MACRO_RENAMES = {
    "dxy_close": "dxy",
    "us_2y_yield": "us2y",
    "us_10y_yield": "us10y",
    "us_30y_yield": "us30y",
    "wti_usd_per_bbl": "wti",
}

MONTHLY_MACRO_RENAMES = {
    "cpi_all_items_index": "cpi_index",
    "cpi_mom_pct": "cpi_mom_pct",
    "cpi_yoy_pct": "cpi_yoy_pct",
    "unemployment_rate_pct": "unemployment_rate_pct",
}

MODEL_FEATURES = [
    "latent_sentiment_index",
    "sentiment_black_box_pc1",
    "external_shock_score",
    "qqq_feedback_score",
    "qqq_21d_return",
    "qqq_63d_return",
    "qqq_vs_sma200",
    "qqq_realized_vol_21d",
    "qqq_drawdown_252d",
    "dxy_63d_return",
    "us10y_level",
    "us10y_63d_change_pp",
    "curve_10y2y_level",
    "wti_63d_return",
    "cpi_yoy_pct",
    "cpi_yoy_3m_change_pp",
    "unemployment_rate_pct",
    "unemployment_6m_change_pp",
    "vix_level",
    "vix_21d_change",
    "hy_oas_level",
    "hy_oas_63d_change_pp",
    "nfci_level",
    "nfci_63d_change",
    "t10y3m_level",
]

GMM_FEATURES = [
    "latent_sentiment_index",
    "external_shock_score",
    "qqq_63d_return",
    "qqq_vs_sma200",
    "qqq_realized_vol_21d",
    "dxy_63d_return",
    "us10y_63d_change_pp",
    "curve_10y2y_level",
    "wti_63d_return",
    "cpi_yoy_pct",
    "unemployment_rate_pct",
    "vix_level",
    "hy_oas_level",
    "nfci_level",
]


@dataclass
class DcaResult:
    name: str
    equity: pd.Series
    allocation: pd.Series
    cashflows: list[tuple[pd.Timestamp, float]]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _normal_two_sided_pvalue(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return float(math.erfc(abs(float(z_value)) / math.sqrt(2.0)))


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return q
    ranked = valid.sort_values()
    m = float(len(ranked))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    q.loc[adjusted.index] = adjusted
    return q


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 12:
        return np.nan
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def _safe_zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def _rolling_zscore(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series / series.shift(periods) - 1.0


def _forward_min_return(close: pd.Series, horizon: int) -> pd.Series:
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(0, len(values) - horizon):
        window = values[i + 1 : i + horizon + 1]
        if np.isfinite(values[i]) and values[i] > 0.0 and np.isfinite(window).any():
            out[i] = np.nanmin(window) / values[i] - 1.0
    return pd.Series(out, index=close.index, name=f"qqq_fwd_{horizon}d_min_return")


def _align_daily(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").sort_index()
    return series.reindex(series.index.union(index)).sort_index().ffill().reindex(index)


def _align_sparse_release_lag(series: pd.Series, index: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.empty:
        return pd.Series(np.nan, index=index)
    released = pd.Series(series.to_numpy(), index=series.index + pd.to_timedelta(lag_days, unit="D"))
    released = released[~released.index.duplicated(keep="last")].sort_index()
    return released.reindex(released.index.union(index)).sort_index().ffill().reindex(index)


def load_qqq(path: Path, start: str | None, end: str | None) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing QQQ parquet: {path}")
    raw = pd.read_parquet(path).copy()
    if "date" in raw.columns:
        index = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    elif "time" in raw.columns:
        index = pd.to_datetime(raw["time"]).dt.tz_localize(None)
    else:
        raise ValueError("QQQ parquet must contain 'date' or 'time'.")
    close_col = "adj_c" if "adj_c" in raw.columns else "c"
    close = pd.Series(pd.to_numeric(raw[close_col], errors="coerce").to_numpy(), index=index, name="qqq_close")
    close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
    if start:
        close = close[close.index >= pd.Timestamp(start)]
    if end:
        close = close[close.index <= pd.Timestamp(end)]
    if len(close) < 756:
        raise ValueError(f"Only {len(close)} QQQ rows after filtering; need at least about 3 years.")
    return close


def load_macro(path: Path, qqq_index: pd.DatetimeIndex, monthly_release_lag_days: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing macro parquet: {path}")
    raw = pd.read_parquet(path).copy()
    if "date" not in raw.columns:
        raise ValueError("Macro parquet must contain a 'date' column.")
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw = raw.set_index("date").sort_index()
    out = pd.DataFrame(index=qqq_index)

    for source, dest in DAILY_MACRO_RENAMES.items():
        if source in raw.columns:
            out[dest] = _align_daily(raw[source], qqq_index)
        else:
            out[dest] = np.nan

    for source, dest in MONTHLY_MACRO_RENAMES.items():
        if source in raw.columns:
            out[dest] = _align_sparse_release_lag(raw[source], qqq_index, monthly_release_lag_days)
        else:
            out[dest] = np.nan

    return out


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        raw_date = (row.get("DATE") or row.get("observation_date") or "").strip()
        raw_value = (row.get(series_id) or "").strip()
        if not raw_date or not raw_value or raw_value == ".":
            continue
        try:
            rows.append((pd.Timestamp(raw_date), float(raw_value)))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No usable values returned for FRED series {series_id}.")
    idx, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(idx), name=series_id).sort_index()


def fetch_fred_series(
    series_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.csv"
    if cache_path.exists() and not refresh:
        text = cache_path.read_text(encoding="utf-8")
        cached = _parse_fred_csv(text, series_id)
        if cached.index.max() >= end_date - pd.Timedelta(days=14):
            return cached

    params = urllib.parse.urlencode(
        {
            "id": series_id,
            "cosd": start_date.date().isoformat(),
            "coed": end_date.date().isoformat(),
        }
    )
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return _parse_fred_csv(text, series_id)


def load_stress_proxies(
    qqq_index: pd.DatetimeIndex,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = pd.DataFrame(index=qqq_index)
    status: dict[str, str] = {}
    for label, series_id in FRED_STRESS_SERIES.items():
        try:
            series = fetch_fred_series(series_id, qqq_index[0], qqq_index[-1], cache_dir, refresh)
            out[label] = _align_daily(series, qqq_index)
            status[label] = "loaded"
        except Exception as exc:  # pragma: no cover - runtime data-source fallback
            out[label] = np.nan
            status[label] = f"missing: {type(exc).__name__}: {exc}"
    return out, status


def add_black_box_pca(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pca_cols = [
        "qqq_63d_return",
        "qqq_vs_sma200",
        "qqq_drawdown_252d",
        "qqq_realized_vol_21d",
        "vix_level",
        "hy_oas_level",
        "nfci_level",
        "dxy_63d_return",
        "us10y_63d_change_pp",
        "wti_21d_return",
        "cpi_yoy_3m_change_pp",
        "unemployment_6m_change_pp",
    ]
    available = [col for col in pca_cols if col in out.columns and out[col].notna().sum() >= 252]
    if len(available) < 4:
        out["sentiment_black_box_pc1"] = np.nan
        out["sentiment_black_box_pc2"] = np.nan
        out["sentiment_black_box_pc1_explained_var"] = np.nan
        return out

    raw = out[available].replace([np.inf, -np.inf], np.nan)
    valid = raw.dropna()
    if len(valid) < 252:
        out["sentiment_black_box_pc1"] = np.nan
        out["sentiment_black_box_pc2"] = np.nan
        out["sentiment_black_box_pc1_explained_var"] = np.nan
        return out

    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=2))
    scores = pipeline.fit_transform(raw)
    pc1 = pd.Series(scores[:, 0], index=out.index)
    pc2 = pd.Series(scores[:, 1], index=out.index)
    sign_anchor = pc1.corr(out["qqq_63d_return"])
    if np.isfinite(sign_anchor) and sign_anchor < 0.0:
        pc1 = -pc1
    pca = pipeline.named_steps["pca"]
    out["sentiment_black_box_pc1"] = pc1
    out["sentiment_black_box_pc2"] = pc2
    out["sentiment_black_box_pc1_explained_var"] = float(pca.explained_variance_ratio_[0])
    return out


def build_dataset(
    qqq_close: pd.Series,
    macro: pd.DataFrame,
    stress: pd.DataFrame,
    target_horizon: int,
) -> pd.DataFrame:
    close = qqq_close.astype(float)
    daily_return = close.pct_change()
    df = pd.DataFrame(index=close.index)
    df.index.name = "date"
    df["qqq_close"] = close
    df["qqq_1d_return"] = daily_return
    df["qqq_21d_return"] = _pct_change(close, 21)
    df["qqq_63d_return"] = _pct_change(close, 63)
    df["qqq_126d_return"] = _pct_change(close, 126)
    df["qqq_252d_return"] = _pct_change(close, 252)
    df["qqq_realized_vol_21d"] = daily_return.rolling(21, min_periods=21).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    df["qqq_realized_vol_63d"] = daily_return.rolling(63, min_periods=63).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    df["qqq_sma50"] = close.rolling(50, min_periods=50).mean()
    df["qqq_sma200"] = close.rolling(200, min_periods=200).mean()
    df["qqq_vs_sma200"] = close / df["qqq_sma200"] - 1.0
    df["qqq_drawdown_252d"] = close / close.rolling(252, min_periods=63).max() - 1.0

    df["dxy_level"] = macro["dxy"]
    df["dxy_21d_return"] = _pct_change(macro["dxy"], 21)
    df["dxy_63d_return"] = _pct_change(macro["dxy"], 63)
    df["dxy_252d_return"] = _pct_change(macro["dxy"], 252)
    df["wti_level"] = macro["wti"]
    df["wti_21d_return"] = _pct_change(macro["wti"], 21)
    df["wti_63d_return"] = _pct_change(macro["wti"], 63)
    df["wti_252d_return"] = _pct_change(macro["wti"], 252)

    for tenor in ["us2y", "us10y", "us30y"]:
        df[f"{tenor}_level"] = macro[tenor]
        df[f"{tenor}_21d_change_pp"] = macro[tenor].diff(21)
        df[f"{tenor}_63d_change_pp"] = macro[tenor].diff(63)
        df[f"{tenor}_252d_change_pp"] = macro[tenor].diff(252)
    df["curve_10y2y_level"] = macro["us10y"] - macro["us2y"]
    df["curve_30y2y_level"] = macro["us30y"] - macro["us2y"]
    df["curve_10y2y_63d_change_pp"] = df["curve_10y2y_level"].diff(63)
    df["curve_10y2y_252d_change_pp"] = df["curve_10y2y_level"].diff(252)

    df["cpi_yoy_pct"] = macro["cpi_yoy_pct"]
    df["cpi_mom_pct"] = macro["cpi_mom_pct"]
    df["cpi_yoy_3m_change_pp"] = macro["cpi_yoy_pct"].diff(63)
    df["cpi_yoy_6m_change_pp"] = macro["cpi_yoy_pct"].diff(126)
    df["unemployment_rate_pct"] = macro["unemployment_rate_pct"]
    df["unemployment_3m_change_pp"] = macro["unemployment_rate_pct"].diff(63)
    df["unemployment_6m_change_pp"] = macro["unemployment_rate_pct"].diff(126)

    df["vix_level"] = stress["vix"]
    df["vix_21d_change"] = stress["vix"].diff(21)
    df["vix_63d_change"] = stress["vix"].diff(63)
    df["hy_oas_level"] = stress["hy_oas"]
    df["hy_oas_21d_change_pp"] = stress["hy_oas"].diff(21)
    df["hy_oas_63d_change_pp"] = stress["hy_oas"].diff(63)
    df["nfci_level"] = stress["nfci"]
    df["nfci_21d_change"] = stress["nfci"].diff(21)
    df["nfci_63d_change"] = stress["nfci"].diff(63)
    df["t10y3m_level"] = stress["t10y3m"]
    df["t10y3m_63d_change_pp"] = stress["t10y3m"].diff(63)

    shock_parts = [
        _rolling_zscore(df["vix_21d_change"]).clip(lower=0.0),
        _rolling_zscore(df["hy_oas_63d_change_pp"]).clip(lower=0.0),
        _rolling_zscore(df["nfci_63d_change"]).clip(lower=0.0),
        _rolling_zscore(df["wti_21d_return"].abs()).clip(lower=0.0),
        _rolling_zscore(df["dxy_21d_return"]).clip(lower=0.0),
        _rolling_zscore(df["us10y_21d_change_pp"]).clip(lower=0.0),
        _rolling_zscore((-df["qqq_21d_return"]).clip(lower=0.0)).clip(lower=0.0),
    ]
    df["external_shock_score"] = pd.concat(shock_parts, axis=1).mean(axis=1)
    df["qqq_feedback_score"] = _rolling_zscore(df["qqq_63d_return"])

    sentiment_parts = [
        _rolling_zscore(df["qqq_63d_return"]),
        _rolling_zscore(df["qqq_vs_sma200"]),
        _rolling_zscore(df["qqq_drawdown_252d"]),
        -_rolling_zscore(df["qqq_realized_vol_21d"]),
        -_rolling_zscore(df["vix_level"]),
        -_rolling_zscore(df["hy_oas_level"]),
        -_rolling_zscore(df["nfci_level"]),
        -_rolling_zscore(df["dxy_63d_return"]),
        -_rolling_zscore(df["us10y_63d_change_pp"]),
        -_rolling_zscore(df["wti_21d_return"].abs()),
        -_rolling_zscore(df["cpi_yoy_3m_change_pp"]),
        -_rolling_zscore(df["unemployment_6m_change_pp"]),
    ]
    df["latent_sentiment_index"] = pd.concat(sentiment_parts, axis=1).mean(axis=1)
    df["latent_sentiment_index"] = _rolling_zscore(df["latent_sentiment_index"], window=504, min_periods=126)
    df = add_black_box_pca(df)

    for horizon in [21, 63, 126, 252]:
        df[f"qqq_fwd_{horizon}d_return"] = close.shift(-horizon) / close - 1.0
        df[f"qqq_fwd_{horizon}d_min_return"] = _forward_min_return(close, horizon)
        df[f"qqq_fwd_{horizon}d_end_date"] = pd.Series(close.index, index=close.index).shift(-horizon)

    df["risk_off_target"] = (
        (df[f"qqq_fwd_{target_horizon}d_return"] <= -0.05)
        | (df[f"qqq_fwd_{target_horizon}d_min_return"] <= -0.10)
    ).astype(float)
    df.loc[df[f"qqq_fwd_{target_horizon}d_return"].isna(), "risk_off_target"] = np.nan
    df["jump_in_target"] = (
        (df[f"qqq_fwd_{target_horizon}d_return"] >= 0.07)
        & (df[f"qqq_fwd_{target_horizon}d_min_return"] > -0.08)
    ).astype(float)
    df.loc[df[f"qqq_fwd_{target_horizon}d_return"].isna(), "jump_in_target"] = np.nan

    df["volatility_shock"] = (df["vix_level"] >= 25.0) | (df["vix_21d_change"] >= 5.0)
    df["credit_spread_shock"] = (df["hy_oas_level"] >= 5.0) | (df["hy_oas_63d_change_pp"] >= 1.0)
    df["financial_conditions_shock"] = (df["nfci_level"] >= 0.0) | (df["nfci_63d_change"] >= 0.35)
    df["equity_drawdown_shock"] = df["qqq_drawdown_252d"] <= -0.15
    df["oil_up_shock"] = df["wti_21d_return"] >= 0.15
    df["dollar_up_shock"] = df["dxy_63d_return"] >= 0.05
    df["rate_up_shock"] = df["us10y_63d_change_pp"] >= 0.50
    shock_cols = [col for col in df.columns if col.endswith("_shock")]
    df["shock_count"] = df[shock_cols].sum(axis=1, min_count=1)
    df["major_shock"] = df["shock_count"] >= 2
    return df


def month_end_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.copy()
    sample["month"] = sample.index.to_period("M")
    sample = sample.groupby("month", sort=True).tail(1).drop(columns=["month"])
    return sample


def available_features(df: pd.DataFrame, features: list[str], min_non_na: int = 60) -> list[str]:
    return [feature for feature in features if feature in df.columns and df[feature].notna().sum() >= min_non_na]


def newey_west_ols(y: pd.Series, x: pd.DataFrame, lags: int) -> pd.DataFrame:
    aligned = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) <= len(x.columns) + 12:
        return pd.DataFrame()

    y_values = aligned["y"].to_numpy(dtype=float)
    x_values = aligned.drop(columns=["y"]).to_numpy(dtype=float)
    x_mean = x_values.mean(axis=0)
    x_std = x_values.std(axis=0, ddof=0)
    keep = np.isfinite(x_std) & (x_std > 0.0)
    if not keep.any():
        return pd.DataFrame()
    kept_names = list(aligned.drop(columns=["y"]).columns[keep])
    x_values = (x_values[:, keep] - x_mean[keep]) / x_std[keep]
    x_design = np.column_stack([np.ones(len(x_values)), x_values])
    names = ["intercept", *kept_names]

    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    beta = xtx_inv @ x_design.T @ y_values
    residuals = y_values - x_design @ beta
    n_obs, n_params = x_design.shape
    lags = int(max(0, min(lags, n_obs - 2)))

    meat = np.zeros((n_params, n_params), dtype=float)
    for t in range(n_obs):
        xt = x_design[t : t + 1].T
        meat += residuals[t] ** 2 * (xt @ xt.T)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = np.zeros((n_params, n_params), dtype=float)
        for t in range(lag, n_obs):
            xt = x_design[t : t + 1].T
            xlag = x_design[t - lag : t - lag + 1].T
            gamma += residuals[t] * residuals[t - lag] * (xt @ xlag.T)
        meat += weight * (gamma + gamma.T)

    scale = n_obs / max(n_obs - n_params, 1)
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stats = beta / se
    p_values = [_normal_two_sided_pvalue(value) for value in t_stats]
    result = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "std_error": se,
            "t_stat": t_stats,
            "p_value": p_values,
            "n_obs": n_obs,
            "nw_lags": lags,
            "r2": 1.0 - float(np.sum(residuals**2)) / float(np.sum((y_values - y_values.mean()) ** 2)),
        }
    )
    return result


def run_impact_tests(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for horizon in [21, 63, 126, 252]:
        outcome = f"qqq_fwd_{horizon}d_return"
        if outcome not in sample.columns:
            continue
        lags = max(1, int(math.ceil(horizon / MONTHLY_TRADING_DAYS)))
        table = newey_west_ols(sample[outcome], sample[features], lags=lags)
        if table.empty:
            continue
        table = table[table["term"] != "intercept"].copy()
        table["horizon_days"] = horizon
        table["outcome"] = outcome
        table["coef_pct_points_per_1sd"] = table["coef"] * 100.0
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["q_value_bh_fdr"] = out.groupby("horizon_days", group_keys=False)["p_value"].apply(_bh_fdr)
    return out.sort_values(["horizon_days", "p_value", "term"])


def run_mediation_style_tests(sample: pd.DataFrame, features: list[str], target_horizon: int) -> pd.DataFrame:
    controls = [
        col
        for col in [
            "dxy_63d_return",
            "us10y_63d_change_pp",
            "curve_10y2y_level",
            "wti_63d_return",
            "cpi_yoy_pct",
            "unemployment_rate_pct",
            "vix_level",
            "hy_oas_level",
            "nfci_level",
        ]
        if col in features
    ]
    rows = []
    tests = [
        (
            "sentiment_driver_feedback_and_shocks",
            "latent_sentiment_index",
            ["qqq_feedback_score", "external_shock_score", *controls],
        ),
        (
            "forward_return_with_sentiment",
            f"qqq_fwd_{target_horizon}d_return",
            ["latent_sentiment_index", "qqq_feedback_score", "external_shock_score", *controls],
        ),
        (
            "forward_return_without_sentiment",
            f"qqq_fwd_{target_horizon}d_return",
            ["qqq_feedback_score", "external_shock_score", *controls],
        ),
    ]
    for test_name, outcome, test_features in tests:
        use_features = [feature for feature in test_features if feature in sample.columns]
        if outcome not in sample.columns or not use_features:
            continue
        table = newey_west_ols(sample[outcome], sample[use_features], lags=max(1, target_horizon // 21))
        if table.empty:
            continue
        table = table[table["term"] != "intercept"].copy()
        table["test"] = test_name
        table["outcome"] = outcome
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["q_value_bh_fdr"] = out.groupby("test", group_keys=False)["p_value"].apply(_bh_fdr)
    return out.sort_values(["test", "p_value"])


def compute_vif(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    raw = sample[features].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    if len(raw) < len(features) + 12:
        return pd.DataFrame()
    z = pd.DataFrame(StandardScaler().fit_transform(raw), index=raw.index, columns=raw.columns)
    for feature in z.columns:
        y = z[feature].to_numpy(dtype=float)
        others = [col for col in z.columns if col != feature]
        x = np.column_stack([np.ones(len(z)), z[others].to_numpy(dtype=float)])
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        fitted = x @ beta
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
        vif = 1.0 / max(1.0 - r2, 1e-9) if np.isfinite(r2) else np.nan
        rows.append({"feature": feature, "auxiliary_r2": r2, "vif": vif, "n_obs": len(z)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def classify_gmm_regimes(df: pd.DataFrame, features: list[str], random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    use_features = available_features(out, features, min_non_na=756)
    if len(use_features) < 4:
        out["gmm_regime"] = "unknown"
        return out, pd.DataFrame()

    valid = out[use_features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 756:
        out["gmm_regime"] = "unknown"
        return out, pd.DataFrame()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(valid[use_features])
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=random_state)
    labels = gmm.fit_predict(x_scaled)
    probabilities = gmm.predict_proba(x_scaled)

    score_frame = pd.DataFrame({"cluster": labels}, index=valid.index)
    score_frame["latent_sentiment_index"] = out.loc[valid.index, "latent_sentiment_index"]
    score_frame["external_shock_score"] = out.loc[valid.index, "external_shock_score"]
    score_frame["qqq_63d_return"] = out.loc[valid.index, "qqq_63d_return"]
    score_frame["vix_level"] = out.loc[valid.index, "vix_level"]
    cluster_score = (
        score_frame.groupby("cluster")["latent_sentiment_index"].mean()
        + score_frame.groupby("cluster")["qqq_63d_return"].mean().fillna(0.0)
        - score_frame.groupby("cluster")["external_shock_score"].mean().fillna(0.0)
        - _safe_zscore(score_frame.groupby("cluster")["vix_level"].mean()).fillna(0.0)
    )
    sorted_clusters = list(cluster_score.sort_values().index)
    cluster_to_regime = {
        sorted_clusters[0]: "risk_off",
        sorted_clusters[1]: "neutral",
        sorted_clusters[2]: "risk_on",
    }

    out["gmm_regime"] = "unknown"
    out.loc[valid.index, "gmm_regime"] = pd.Series(labels, index=valid.index).map(cluster_to_regime)
    for cluster, regime in cluster_to_regime.items():
        out.loc[valid.index, f"gmm_prob_{regime}"] = probabilities[:, cluster]

    summary_rows = []
    for regime, group in out.loc[valid.index].groupby("gmm_regime"):
        summary_rows.append(
            {
                "regime": regime,
                "count": int(len(group)),
                "avg_latent_sentiment": float(group["latent_sentiment_index"].mean()),
                "avg_external_shock_score": float(group["external_shock_score"].mean()),
                "avg_vix": float(group["vix_level"].mean()) if group["vix_level"].notna().any() else np.nan,
                "avg_forward_63d_return": float(group["qqq_fwd_63d_return"].mean()),
                "positive_63d_rate": float((group["qqq_fwd_63d_return"] > 0).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("avg_latent_sentiment")
    return out, summary


def shock_return_tests(sample: pd.DataFrame, target_horizon: int) -> pd.DataFrame:
    outcome = f"qqq_fwd_{target_horizon}d_return"
    rows = []
    for shock in [col for col in sample.columns if col.endswith("_shock") or col == "major_shock"]:
        valid = sample[[shock, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        valid[shock] = valid[shock].astype(bool)
        on = valid.loc[valid[shock], outcome]
        off = valid.loc[~valid[shock], outcome]
        if len(on) < 8 or len(off) < 8:
            continue
        diff = float(on.mean() - off.mean())
        se = math.sqrt(float(on.var(ddof=1)) / len(on) + float(off.var(ddof=1)) / len(off))
        t_stat = diff / se if se > 0.0 else np.nan
        rows.append(
            {
                "shock": shock,
                "horizon_days": target_horizon,
                "shock_count": int(len(on)),
                "no_shock_count": int(len(off)),
                "shock_avg_return": float(on.mean()),
                "no_shock_avg_return": float(off.mean()),
                "shock_minus_no_shock": diff,
                "t_stat_normal_approx": t_stat,
                "p_value_normal_approx": _normal_two_sided_pvalue(t_stat),
                "shock_positive_rate": float((on > 0).mean()),
                "no_shock_positive_rate": float((off > 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_fdr"] = _bh_fdr(out["p_value_normal_approx"])
    return out.sort_values("p_value_normal_approx")


def purged_train_test(sample: pd.DataFrame, target: str, horizon: int, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = sample.dropna(subset=[target]).copy()
    if valid.empty:
        return valid, valid
    split = max(int(len(valid) * (1.0 - test_size)), 1)
    test = valid.iloc[split:].copy()
    if test.empty:
        return valid.iloc[:split].copy(), test
    end_col = f"qqq_fwd_{horizon}d_end_date"
    if end_col in valid.columns:
        train = valid.iloc[:split].loc[valid.iloc[:split][end_col] < test.index[0]].copy()
    else:
        train = valid.iloc[: max(split - horizon // 21, 1)].copy()
    return train, test


def evaluate_models(
    sample: pd.DataFrame,
    features: list[str],
    target_horizon: int,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    return_target = f"qqq_fwd_{target_horizon}d_return"

    train, test = purged_train_test(sample, return_target, target_horizon, test_size)
    train = train.dropna(subset=[return_target])
    test = test.dropna(subset=[return_target])
    if len(train) >= 60 and len(test) >= 24:
        x_train, y_train = train[features], train[return_target]
        x_test, y_test = test[features], test[return_target]
        regressors = {
            "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        }
        for model_name, model in regressors.items():
            model.fit(x_train, y_train)
            pred = pd.Series(model.predict(x_test), index=x_test.index)
            rows.append(
                {
                    "target": return_target,
                    "model": model_name,
                    "train_start": train.index.min(),
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "train_n": len(train),
                    "test_n": len(test),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "r2": float(r2_score(y_test, pred)),
                    "spearman_pred_actual": _safe_spearman(pred, y_test),
                }
            )
            if model_name == "random_forest":
                perm = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring="neg_mean_absolute_error",
                )
                for feature, mean_imp, std_imp in zip(features, perm.importances_mean, perm.importances_std):
                    importance_rows.append(
                        {
                            "target": return_target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(mean_imp),
                            "importance_std": float(std_imp),
                        }
                    )

    for target in ["risk_off_target", "jump_in_target"]:
        train, test = purged_train_test(sample, target, target_horizon, test_size)
        train = train.dropna(subset=[target])
        test = test.dropna(subset=[target])
        if len(train) < 60 or len(test) < 24 or train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        x_train, y_train = train[features], train[target].astype(int)
        x_test, y_test = test[features], test[target].astype(int)
        classifiers = {
            "logistic": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
            ),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        }
        for model_name, model in classifiers.items():
            model.fit(x_train, y_train)
            proba = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index)
            pred = (proba >= 0.50).astype(int)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "train_start": train.index.min(),
                    "train_end": train.index.max(),
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    "train_n": len(train),
                    "test_n": len(test),
                    "event_rate_train": float(y_train.mean()),
                    "event_rate_test": float(y_test.mean()),
                    "auc": float(roc_auc_score(y_test, proba)),
                    "average_precision": float(average_precision_score(y_test, proba)),
                    "brier": float(brier_score_loss(y_test, proba)),
                    "balanced_accuracy_at_50pct": float(balanced_accuracy_score(y_test, pred)),
                    "precision_at_50pct": float(precision_score(y_test, pred, zero_division=0)),
                    "recall_at_50pct": float(recall_score(y_test, pred, zero_division=0)),
                }
            )
            if model_name == "logistic":
                clf = model.named_steps["logisticregression"]
                for feature, coef in zip(features, clf.coef_[0]):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(coef),
                            "importance_std": np.nan,
                        }
                    )
            if model_name == "random_forest":
                perm = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=random_state,
                    n_jobs=-1,
                    scoring="roc_auc",
                )
                for feature, mean_imp, std_imp in zip(features, perm.importances_mean, perm.importances_std):
                    importance_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance_mean": float(mean_imp),
                            "importance_std": float(std_imp),
                        }
                    )

        latest = sample.iloc[[-1]]
        all_train = sample.dropna(subset=[target]).copy()
        if len(all_train) >= 60 and all_train[target].nunique() >= 2:
            current_model = classifiers["logistic"]
            current_model.fit(all_train[features], all_train[target].astype(int))
            current[f"current_{target}_probability"] = float(current_model.predict_proba(latest[features])[:, 1][0])

    return pd.DataFrame(rows), pd.DataFrame(importance_rows), current


def walkforward_signal_probabilities(
    sample: pd.DataFrame,
    features: list[str],
    target: str,
    target_horizon: int,
    min_train_months: int,
    random_state: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=sample.index, name=f"{target}_walkforward_probability")
    end_col = f"qqq_fwd_{target_horizon}d_end_date"
    for i, date in enumerate(sample.index):
        train = sample.iloc[:i].copy()
        if end_col in train.columns:
            train = train[train[end_col] < date]
        train = train.dropna(subset=[target])
        if len(train) < min_train_months or train[target].nunique() < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
        )
        model.fit(train[features], train[target].astype(int))
        out.loc[date] = float(model.predict_proba(sample.loc[[date], features])[:, 1][0])
    return out


def build_allocation_signal(
    sample: pd.DataFrame,
    risk_off_prob: pd.Series,
    jump_in_prob: pd.Series,
    risk_off_threshold: float,
    jump_in_threshold: float,
) -> pd.DataFrame:
    signals = pd.DataFrame(index=sample.index)
    signals["risk_off_probability"] = risk_off_prob
    signals["jump_in_probability"] = jump_in_prob
    signals["latent_sentiment_index"] = sample["latent_sentiment_index"]
    signals["gmm_regime"] = sample.get("gmm_regime", pd.Series("unknown", index=sample.index))
    risk_off = (
        (signals["risk_off_probability"] >= risk_off_threshold)
        | (signals["latent_sentiment_index"] <= -1.0)
        | (signals["gmm_regime"] == "risk_off")
    )
    jump_in = (
        (signals["jump_in_probability"] >= jump_in_threshold)
        & (signals["risk_off_probability"] < risk_off_threshold)
        & (signals["latent_sentiment_index"] > 0.0)
    )
    allocation = pd.Series(0.70, index=signals.index)
    allocation.loc[risk_off] = 0.25
    allocation.loc[jump_in] = 1.00
    allocation.loc[signals["risk_off_probability"].isna() | signals["jump_in_probability"].isna()] = np.nan
    labels = pd.Series("neutral_dca", index=signals.index, dtype=object)
    labels.loc[risk_off] = "risk_off_reserve_cash"
    labels.loc[jump_in] = "jump_in_full_allocation"
    labels.loc[allocation.isna()] = "insufficient_walkforward_history"
    signals["target_equity_allocation"] = allocation
    signals["signal"] = labels
    return signals


def _xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    flows = [(pd.Timestamp(date), float(amount)) for date, amount in cashflows if np.isfinite(amount)]
    if not flows or not any(amount < 0 for _, amount in flows) or not any(amount > 0 for _, amount in flows):
        return np.nan
    start = flows[0][0]

    def npv(rate: float) -> float:
        total = 0.0
        for date, amount in flows:
            years = max((date - start).days / 365.25, 0.0)
            total += amount / ((1.0 + rate) ** years)
        return total

    low, high = -0.999, 10.0
    low_npv, high_npv = npv(low), npv(high)
    if not np.isfinite(low_npv) or not np.isfinite(high_npv) or low_npv * high_npv > 0:
        return np.nan
    for _ in range(120):
        mid = (low + high) / 2.0
        mid_npv = npv(mid)
        if abs(mid_npv) < 1e-7:
            return mid
        if low_npv * mid_npv <= 0:
            high = mid
            high_npv = mid_npv
        else:
            low = mid
            low_npv = mid_npv
    return (low + high) / 2.0


def simulate_dca(
    close: pd.Series,
    target_allocation: pd.Series,
    *,
    name: str,
    start_date: pd.Timestamp,
    initial_capital: float,
    monthly_contribution: float,
    trading_cost_bps: float,
) -> DcaResult:
    price = close[close.index >= start_date].copy()
    signal = target_allocation.reindex(target_allocation.index.union(price.index)).sort_index().ffill().reindex(price.index)
    signal = signal.shift(1).ffill().fillna(1.0).clip(lower=0.0, upper=1.0)
    if price.empty:
        raise ValueError("No price rows available for the DCA backtest.")

    cost_rate = trading_cost_bps / 10_000.0
    cash = float(initial_capital)
    shares = 0.0
    equity_values = []
    allocations = []
    cashflows: list[tuple[pd.Timestamp, float]] = [(price.index[0], -float(initial_capital))]
    prior_month = None

    for date, px in price.items():
        month = date.to_period("M")
        contribution = 0.0
        if prior_month is not None and month != prior_month:
            contribution = float(monthly_contribution)
            cash += contribution
            cashflows.append((date, -contribution))
        prior_month = month

        account_value = cash + shares * float(px)
        rebalance_day = contribution > 0.0 or len(equity_values) == 0
        if rebalance_day and account_value > 0.0 and np.isfinite(signal.loc[date]):
            target_value = float(signal.loc[date]) * account_value
            current_value = shares * float(px)
            trade_value = target_value - current_value
            if trade_value > 0.0:
                available = max(cash, 0.0)
                total_needed = trade_value * (1.0 + cost_rate)
                actual_trade = min(trade_value, available / (1.0 + cost_rate)) if total_needed > available else trade_value
                cash -= actual_trade * (1.0 + cost_rate)
                shares += actual_trade / float(px)
            elif trade_value < 0.0:
                sell_value = min(-trade_value, shares * float(px))
                shares -= sell_value / float(px)
                cash += sell_value * (1.0 - cost_rate)

        account_value = cash + shares * float(px)
        equity_values.append(account_value)
        allocations.append((shares * float(px) / account_value) if account_value > 0.0 else 0.0)

    equity = pd.Series(equity_values, index=price.index, name=name)
    allocation = pd.Series(allocations, index=price.index, name=name)
    cashflows.append((price.index[-1], float(equity.iloc[-1])))
    return DcaResult(name=name, equity=equity, allocation=allocation, cashflows=cashflows)


def dca_metrics(results: list[DcaResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        equity = result.equity.dropna()
        if equity.empty:
            continue
        total_contributed = -sum(amount for _, amount in result.cashflows[:-1] if amount < 0.0)
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "strategy": result.name,
                "start": equity.index[0],
                "end": equity.index[-1],
                "final_value": float(equity.iloc[-1]),
                "total_contributed": float(total_contributed),
                "profit": float(equity.iloc[-1] - total_contributed),
                "profit_on_contributed": float(equity.iloc[-1] / total_contributed - 1.0) if total_contributed > 0 else np.nan,
                "xirr": _xirr(result.cashflows),
                "max_drawdown_on_account_value": float(drawdown.min()),
                "avg_equity_allocation": float(result.allocation.mean()),
                "min_equity_allocation": float(result.allocation.min()),
                "max_equity_allocation": float(result.allocation.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("final_value", ascending=False)


def plot_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coefficients(impact: pd.DataFrame, path: Path, horizon: int) -> None:
    if impact.empty:
        return
    data = impact[(impact["horizon_days"] == horizon) & (impact["term"] != "intercept")].copy()
    if data.empty:
        return
    data["abs_coef"] = data["coef_pct_points_per_1sd"].abs()
    data = data.sort_values("abs_coef", ascending=False).head(15).sort_values("coef_pct_points_per_1sd")
    colors = np.where(data["q_value_bh_fdr"] <= 0.10, "#1f77b4", "#8c8c8c")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(data["term"], data["coef_pct_points_per_1sd"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Newey-West OLS impact on {horizon}D forward QQQ return")
    ax.set_xlabel("Percentage points of forward return per 1-sd feature move")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: Path, target: str) -> None:
    if importance.empty:
        return
    data = importance[(importance["target"] == target) & (importance["model"] == "random_forest")].copy()
    if data.empty:
        data = importance[importance["target"] == target].copy()
    if data.empty:
        return
    data["abs_importance"] = data["importance_mean"].abs()
    data = data.sort_values("abs_importance", ascending=False).head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(data["feature"], data["importance_mean"], color="#2ca02c")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Walk-forward holdout feature importance: {target}")
    ax.set_xlabel("Permutation importance or standardized coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sentiment(df: pd.DataFrame, path: Path) -> None:
    data = df[["qqq_close", "latent_sentiment_index", "external_shock_score"]].dropna(how="all").copy()
    if data.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(data.index, data["qqq_close"], color="black", linewidth=1.2, label="QQQ adjusted close")
    ax1.set_yscale("log")
    ax1.set_ylabel("QQQ adjusted close, log scale")
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["latent_sentiment_index"], color="#1f77b4", linewidth=1.0, label="Latent sentiment")
    ax2.plot(data.index, data["external_shock_score"], color="#d62728", linewidth=0.9, alpha=0.7, label="External shock score")
    ax2.axhline(0.0, color="#777777", linewidth=0.8)
    ax2.set_ylabel("Z-score style indices")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("QQQ, latent sentiment black box, and external shock proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_regimes(df: pd.DataFrame, path: Path) -> None:
    data = df[["qqq_close", "gmm_regime"]].dropna(subset=["qqq_close"]).copy()
    if data.empty or "gmm_regime" not in data:
        return
    colors = {"risk_on": "#dff0d8", "neutral": "#fcf8e3", "risk_off": "#f2dede", "unknown": "#eeeeee"}
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data["qqq_close"], color="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("QQQ adjusted close, log scale")
    ax.set_title("Unsupervised macro/sentiment regimes over QQQ")
    regimes = data["gmm_regime"].fillna("unknown")
    start = data.index[0]
    current = regimes.iloc[0]
    for date, regime in regimes.iloc[1:].items():
        if regime != current:
            ax.axvspan(start, date, color=colors.get(str(current), "#eeeeee"), alpha=0.30, linewidth=0)
            start = date
            current = regime
    ax.axvspan(start, data.index[-1], color=colors.get(str(current), "#eeeeee"), alpha=0.30, linewidth=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.30) for color in colors.values()]
    ax.legend(handles, list(colors.keys()), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dca(results: list[DcaResult], path: Path) -> None:
    if not results:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    for result in results:
        ax.plot(result.equity.index, result.equity, linewidth=1.2, label=result.name)
    ax.set_title("DCA backtest equity curves")
    ax.set_ylabel("Account value, USD")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_allocation(signals: pd.DataFrame, path: Path) -> None:
    if signals.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(signals.index, signals["target_equity_allocation"], color="#1f77b4", linewidth=1.2, label="Target equity allocation")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Target equity allocation")
    ax2 = ax1.twinx()
    ax2.plot(signals.index, signals["risk_off_probability"], color="#d62728", alpha=0.7, label="Risk-off probability")
    ax2.plot(signals.index, signals["jump_in_probability"], color="#2ca02c", alpha=0.7, label="Jump-in probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Walk-forward probability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Walk-forward regime allocation signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.1f}%"


def fmt_num(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"
