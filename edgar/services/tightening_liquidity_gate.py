from __future__ import annotations

import bisect
import csv
from datetime import date, datetime, timedelta
from pathlib import Path


_DEFAULT_FRED_DIR = Path("cache/fred")


def _simple_moving_average(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    window: list[float] = []
    rolling_sum = 0.0
    for idx, value in enumerate(values):
        if value is None:
            window.clear()
            rolling_sum = 0.0
            continue
        window.append(float(value))
        rolling_sum += float(value)
        if len(window) > period:
            rolling_sum -= window.pop(0)
        if len(window) == period:
            out[idx] = rolling_sum / period
    return out


def load_local_macro_series(
    csv_path: str | Path,
    value_column: str | None = None,
) -> dict[str, float]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Macro series not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "observation_date" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required observation_date column")
        value_key = value_column
        if value_key is None:
            value_fields = [field for field in reader.fieldnames if field != "observation_date"]
            if len(value_fields) != 1:
                raise ValueError(f"Could not infer value column for {path}")
            value_key = value_fields[0]

        series: dict[str, float] = {}
        for row in reader:
            day = str(row.get("observation_date") or "").strip()
            raw = str(row.get(value_key) or "").strip()
            if not day or raw in {"", ".", "nan", "NaN", "None"}:
                continue
            try:
                series[day] = float(raw)
            except ValueError:
                continue
    if not series:
        raise ValueError(f"No usable data found in {path}")
    return series


def load_default_tightening_liquidity_inputs(
    fred_dir: str | Path = _DEFAULT_FRED_DIR,
) -> dict[str, dict[str, float]]:
    base_dir = Path(fred_dir)
    return {
        "curve": load_local_macro_series(base_dir / "T10Y3M.csv"),
        "credit_spread": load_local_macro_series(base_dir / "BAMLH0A0HYM2.csv"),
        "nfci": load_local_macro_series(base_dir / "NFCI.csv"),
        "vix": load_local_macro_series(base_dir / "VIXCLS.csv"),
    }


def _align_series(
    series_map: dict[str, dict[str, float]],
) -> tuple[list[date], dict[str, list[float | None]]]:
    all_dates = sorted(
        {
            date.fromisoformat(day)
            for series in series_map.values()
            for day in series.keys()
        }
    )
    aligned: dict[str, list[float | None]] = {}
    for name, series in series_map.items():
        lookup = {date.fromisoformat(day): float(value) for day, value in series.items()}
        values: list[float | None] = []
        last_value: float | None = None
        for day in all_dates:
            if day in lookup:
                last_value = lookup[day]
            values.append(last_value)
        aligned[name] = values
    return all_dates, aligned


def build_tightening_liquidity_state(
    *,
    curve_series: dict[str, float] | None = None,
    credit_spread_series: dict[str, float] | None = None,
    nfci_series: dict[str, float] | None = None,
    vix_series: dict[str, float] | None = None,
    ma_days: int = 60,
    tight_score_threshold: int = 2,
    fred_dir: str | Path = _DEFAULT_FRED_DIR,
) -> dict:
    if ma_days <= 0:
        raise ValueError("ma_days must be positive")
    if tight_score_threshold <= 0:
        raise ValueError("tight_score_threshold must be positive")

    if (
        curve_series is None
        and credit_spread_series is None
        and nfci_series is None
        and vix_series is None
    ):
        defaults = load_default_tightening_liquidity_inputs(fred_dir=fred_dir)
        curve_series = defaults["curve"]
        credit_spread_series = defaults["credit_spread"]
        nfci_series = defaults["nfci"]
        vix_series = defaults["vix"]

    series_map = {
        "curve": curve_series or {},
        "credit_spread": credit_spread_series or {},
        "nfci": nfci_series or {},
        "vix": vix_series or {},
    }
    populated = {name: series for name, series in series_map.items() if series}
    if not populated:
        raise ValueError("At least one macro series is required to build tightening/liquidity state")

    dates, aligned = _align_series(populated)
    components: dict[str, dict[str, list[float | None] | list[bool | None]]] = {}
    for name, values in aligned.items():
        sma = _simple_moving_average(values, ma_days)
        flags: list[bool | None] = [None] * len(values)
        for idx, value in enumerate(values):
            ma_value = sma[idx]
            if value is None or ma_value is None:
                continue
            if name == "curve":
                flags[idx] = bool(float(value) <= 0.0 or float(value) < float(ma_value))
            else:
                flags[idx] = bool(float(value) > float(ma_value))
        components[name] = {
            "values": values,
            "sma": sma,
            "tight_flag": flags,
        }

    scores: list[int | None] = [None] * len(dates)
    regimes: list[str | None] = [None] * len(dates)
    component_count = len(components)
    for idx in range(len(dates)):
        ready_flags = [
            bool(component["tight_flag"][idx])
            for component in components.values()
            if component["tight_flag"][idx] is not None
        ]
        if len(ready_flags) < component_count:
            continue
        score = sum(1 for flag in ready_flags if flag)
        scores[idx] = score
        regimes[idx] = (
            "tightening_liquidity_tight"
            if score >= tight_score_threshold
            else "tightening_liquidity_neutral"
        )

    return {
        "dates": dates,
        "scores": scores,
        "regimes": regimes,
        "ma_days": ma_days,
        "tight_score_threshold": tight_score_threshold,
        "components": components,
    }


def lookup_tightening_liquidity_signal(
    *,
    entry_ts: datetime,
    asset_bucket: str,
    direction: str,
    state: dict | None,
    lag_days: int = 1,
    gated_buckets: frozenset[str] | None = None,
    mode: str = "size",
    tight_long_mult: float = 0.5,
    neutral_long_mult: float = 1.0,
    tight_short_mult: float = 1.0,
    neutral_short_mult: float = 1.0,
) -> tuple[str | None, float, int | None, bool]:
    if (
        not state
        or direction not in {"long", "short"}
        or (gated_buckets is not None and asset_bucket not in gated_buckets)
    ):
        return None, 1.0, None, False
    if mode not in {"size", "entry"}:
        raise ValueError("mode must be one of {'size', 'entry'}")

    target_day = entry_ts.date() - timedelta(days=max(lag_days, 0))
    dates = state.get("dates", [])
    idx = bisect.bisect_right(dates, target_day) - 1
    if idx < 0:
        return None, 1.0, None, False

    regimes = state.get("regimes", [])
    scores = state.get("scores", [])
    regime = regimes[idx] if idx < len(regimes) else None
    score = scores[idx] if idx < len(scores) else None
    if regime is None:
        return None, 1.0, None, False

    if direction == "long":
        if regime == "tightening_liquidity_tight":
            if mode == "entry":
                return regime, 0.0, score, True
            return regime, float(tight_long_mult), score, False
        return regime, float(neutral_long_mult), score, False

    if regime == "tightening_liquidity_tight":
        return regime, float(tight_short_mult), score, False
    return regime, float(neutral_short_mult), score, False
