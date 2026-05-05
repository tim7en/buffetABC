from __future__ import annotations

import bisect
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from edgar.services.tightening_liquidity_gate import load_local_macro_series


_DEFAULT_COMBINED_MACRO_PATH = Path("cache/cache/macro_daily_1999.csv")
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
        numeric = float(value)
        window.append(numeric)
        rolling_sum += numeric
        if len(window) > period:
            rolling_sum -= window.pop(0)
        if len(window) == period:
            out[idx] = rolling_sum / period
    return out


def _lookback_delta(values: list[float | None], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if lookback <= 0:
        return out
    for idx in range(lookback, len(values)):
        current = values[idx]
        prior = values[idx - lookback]
        if current is None or prior is None:
            continue
        out[idx] = float(current) - float(prior)
    return out


def _lookback_return(values: list[float | None], lookback: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if lookback <= 0:
        return out
    for idx in range(lookback, len(values)):
        current = values[idx]
        prior = values[idx - lookback]
        if current is None or prior is None or abs(float(prior)) <= 1e-9:
            continue
        out[idx] = float(current) / float(prior) - 1.0
    return out


def _series_spread(
    lhs: list[float | None],
    rhs: list[float | None],
) -> list[float | None]:
    out: list[float | None] = [None] * min(len(lhs), len(rhs))
    for idx, (left, right) in enumerate(zip(lhs, rhs)):
        if left is None or right is None:
            continue
        out[idx] = float(left) - float(right)
    return out


def _series_ratio(
    numerator: list[float | None],
    denominator: list[float | None],
) -> list[float | None]:
    out: list[float | None] = [None] * min(len(numerator), len(denominator))
    for idx, (top, bottom) in enumerate(zip(numerator, denominator)):
        if top is None or bottom is None or abs(float(bottom)) <= 1e-9:
            continue
        out[idx] = float(top) / float(bottom)
    return out


def _load_combined_macro_series(
    csv_path: str | Path = _DEFAULT_COMBINED_MACRO_PATH,
) -> dict[str, dict[str, float]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Combined macro dataset not found: {path}")
    wanted = {"dxy_close", "us_2y_yield", "us_10y_yield", "vix3m_level"}
    series_map = {column: {} for column in wanted}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required date column")
        for row in reader:
            day = str(row.get("date") or "").strip()
            if not day:
                continue
            for column in wanted:
                raw = str(row.get(column) or "").strip()
                if raw in {"", ".", "nan", "NaN", "None"}:
                    continue
                try:
                    series_map[column][day] = float(raw)
                except ValueError:
                    continue
    return series_map


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


def build_macro_regime_score_state(
    *,
    combined_macro_path: str | Path = _DEFAULT_COMBINED_MACRO_PATH,
    fred_dir: str | Path = _DEFAULT_FRED_DIR,
    version: str = "v1",
    dxy_fast_ma_days: int = 100,
    dxy_slow_ma_days: int = 200,
    rates_ma_days: int = 60,
    stress_ma_days: int = 60,
    lookback_days: int = 21,
    rates_change_threshold_bps: float = 20.0,
    front_end_rates_change_threshold_bps: float = 15.0,
    vix_term_contango_ratio: float = 0.98,
    vix_term_backwardation_ratio: float = 1.02,
    score_cap: int = 3,
) -> dict:
    if version not in {"v1", "v2_front_end"}:
        raise ValueError("version must be 'v1' or 'v2_front_end'")
    if dxy_fast_ma_days <= 0 or dxy_slow_ma_days <= 0 or rates_ma_days <= 0 or stress_ma_days <= 0:
        raise ValueError("MA windows must be positive")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if score_cap <= 0:
        raise ValueError("score_cap must be positive")
    if front_end_rates_change_threshold_bps < 0.0:
        raise ValueError("front_end_rates_change_threshold_bps must be non-negative")
    if (
        vix_term_contango_ratio <= 0.0
        or vix_term_backwardation_ratio <= 0.0
        or vix_term_contango_ratio >= vix_term_backwardation_ratio
    ):
        raise ValueError("VIX term ratios must be positive and ordered contango < backwardation")

    combined = _load_combined_macro_series(csv_path=combined_macro_path)
    fred_base = Path(fred_dir)
    series_map = {
        "dxy": combined["dxy_close"],
        "us2y": combined["us_2y_yield"],
        "us10y": combined["us_10y_yield"],
        "vix3m": combined["vix3m_level"],
        "curve": load_local_macro_series(fred_base / "T10Y3M.csv"),
        "hy": load_local_macro_series(fred_base / "BAMLH0A0HYM2.csv"),
        "nfci": load_local_macro_series(fred_base / "NFCI.csv"),
        "vix": load_local_macro_series(fred_base / "VIXCLS.csv"),
    }
    dates, aligned = _align_series(series_map)

    dxy_vals = aligned["dxy"]
    us2y_vals = aligned["us2y"]
    us10y_vals = aligned["us10y"]
    vix3m_vals = aligned["vix3m"]
    curve_vals = aligned["curve"]
    hy_vals = aligned["hy"]
    nfci_vals = aligned["nfci"]
    vix_vals = aligned["vix"]

    dxy_ma_fast = _simple_moving_average(dxy_vals, dxy_fast_ma_days)
    dxy_ma_slow = _simple_moving_average(dxy_vals, dxy_slow_ma_days)
    dxy_return = _lookback_return(dxy_vals, lookback_days)
    us2y_ma = _simple_moving_average(us2y_vals, rates_ma_days)
    us2y_change = _lookback_delta(us2y_vals, lookback_days)
    us10y_ma = _simple_moving_average(us10y_vals, rates_ma_days)
    us10y_change = _lookback_delta(us10y_vals, lookback_days)
    two_ten_vals = _series_spread(us10y_vals, us2y_vals)
    two_ten_ma = _simple_moving_average(two_ten_vals, stress_ma_days)
    two_ten_change = _lookback_delta(two_ten_vals, lookback_days)
    curve_ma = _simple_moving_average(curve_vals, stress_ma_days)
    curve_change = _lookback_delta(curve_vals, lookback_days)
    hy_ma = _simple_moving_average(hy_vals, stress_ma_days)
    hy_change = _lookback_delta(hy_vals, lookback_days)
    nfci_ma = _simple_moving_average(nfci_vals, stress_ma_days)
    nfci_change = _lookback_delta(nfci_vals, lookback_days)
    vix_ma = _simple_moving_average(vix_vals, stress_ma_days)
    vix_change = _lookback_delta(vix_vals, lookback_days)
    vix_term_ratio = _series_ratio(vix_vals, vix3m_vals)

    component_scores: dict[str, list[int | None]] = {
        "dollar": [None] * len(dates),
        "rates": [None] * len(dates),
        "stress": [None] * len(dates),
        "liquidity": [None] * len(dates),
    }
    raw_scores: list[int | None] = [None] * len(dates)
    scores: list[int | None] = [None] * len(dates)
    labels: list[str | None] = [None] * len(dates)

    rates_threshold = float(rates_change_threshold_bps) / 100.0
    front_end_rates_threshold = float(front_end_rates_change_threshold_bps) / 100.0
    for idx in range(len(dates)):
        dxy_score: int | None = None
        if (
            dxy_vals[idx] is not None
            and dxy_ma_fast[idx] is not None
            and dxy_ma_slow[idx] is not None
        ):
            if float(dxy_vals[idx]) < float(dxy_ma_fast[idx]) and float(dxy_vals[idx]) < float(dxy_ma_slow[idx]):
                dxy_score = 1
            elif (
                float(dxy_vals[idx]) > float(dxy_ma_fast[idx])
                and float(dxy_vals[idx]) > float(dxy_ma_slow[idx])
                and dxy_return[idx] is not None
                and float(dxy_return[idx]) > 0.0
            ):
                dxy_score = -1
            else:
                dxy_score = 0
        component_scores["dollar"][idx] = dxy_score

        rates_score: int | None = None
        if version == "v1":
            if us10y_vals[idx] is not None and us10y_change[idx] is not None:
                if (
                    float(us10y_change[idx]) <= -rates_threshold
                    or (
                        us10y_ma[idx] is not None
                        and float(us10y_vals[idx]) < float(us10y_ma[idx])
                        and float(us10y_change[idx]) < 0.0
                    )
                ):
                    rates_score = 1
                elif (
                    float(us10y_change[idx]) >= rates_threshold
                    or (
                        us10y_ma[idx] is not None
                        and float(us10y_vals[idx]) > float(us10y_ma[idx])
                        and float(us10y_change[idx]) > 0.0
                    )
                ):
                    rates_score = -1
                else:
                    rates_score = 0
        else:
            two_y_supportive = False
            two_y_hostile = False
            ten_y_supportive = False
            ten_y_hostile = False
            curve_supportive = False
            curve_hostile = False

            if us2y_vals[idx] is not None and us2y_change[idx] is not None:
                two_y_supportive = (
                    float(us2y_change[idx]) <= -front_end_rates_threshold
                    or (
                        us2y_ma[idx] is not None
                        and float(us2y_vals[idx]) < float(us2y_ma[idx])
                        and float(us2y_change[idx]) < 0.0
                    )
                )
                two_y_hostile = (
                    float(us2y_change[idx]) >= front_end_rates_threshold
                    or (
                        us2y_ma[idx] is not None
                        and float(us2y_vals[idx]) > float(us2y_ma[idx])
                        and float(us2y_change[idx]) > 0.0
                    )
                )
            if us10y_vals[idx] is not None and us10y_change[idx] is not None:
                ten_y_supportive = (
                    float(us10y_change[idx]) <= -rates_threshold
                    or (
                        us10y_ma[idx] is not None
                        and float(us10y_vals[idx]) < float(us10y_ma[idx])
                        and float(us10y_change[idx]) < 0.0
                    )
                )
                ten_y_hostile = (
                    float(us10y_change[idx]) >= rates_threshold
                    or (
                        us10y_ma[idx] is not None
                        and float(us10y_vals[idx]) > float(us10y_ma[idx])
                        and float(us10y_change[idx]) > 0.0
                    )
                )
            if two_ten_vals[idx] is not None and two_ten_change[idx] is not None:
                curve_supportive = (
                    (
                        two_ten_ma[idx] is not None
                        and float(two_ten_vals[idx]) > float(two_ten_ma[idx])
                        and float(two_ten_change[idx]) >= 0.0
                    )
                    or (
                        float(two_ten_vals[idx]) >= 0.0
                        and float(two_ten_change[idx]) > 0.0
                    )
                )
                curve_hostile = (
                    (
                        two_ten_ma[idx] is not None
                        and float(two_ten_vals[idx]) < float(two_ten_ma[idx])
                        and float(two_ten_change[idx]) < 0.0
                    )
                    or (
                        float(two_ten_vals[idx]) < 0.0
                        and float(two_ten_change[idx]) < 0.0
                    )
                )

            if two_y_supportive and not curve_hostile and (ten_y_supportive or curve_supportive):
                rates_score = 1
            elif two_y_hostile and (ten_y_hostile or curve_hostile or (two_ten_vals[idx] is not None and float(two_ten_vals[idx]) < 0.0)):
                rates_score = -1
            elif two_y_supportive and not curve_hostile:
                rates_score = 0
            elif two_y_hostile:
                rates_score = 0
            elif (
                us2y_vals[idx] is not None
                and us2y_change[idx] is not None
                and us10y_vals[idx] is not None
                and us10y_change[idx] is not None
            ):
                rates_score = 0
        component_scores["rates"][idx] = rates_score

        easing_votes = 0
        stress_votes = 0
        available_votes = 0
        for values, ma_vals, delta_vals in (
            (vix_vals, vix_ma, vix_change),
            (hy_vals, hy_ma, hy_change),
            (nfci_vals, nfci_ma, nfci_change),
        ):
            if values[idx] is None or ma_vals[idx] is None or delta_vals[idx] is None:
                continue
            available_votes += 1
            value = float(values[idx])
            ma_value = float(ma_vals[idx])
            delta_value = float(delta_vals[idx])
            if value < ma_value and delta_value <= 0.0:
                easing_votes += 1
            elif value > ma_value and delta_value > 0.0:
                stress_votes += 1
        if version == "v2_front_end" and vix_term_ratio[idx] is not None:
            available_votes += 1
            if float(vix_term_ratio[idx]) <= float(vix_term_contango_ratio):
                easing_votes += 1
            elif float(vix_term_ratio[idx]) >= float(vix_term_backwardation_ratio):
                stress_votes += 1
        stress_score: int | None = None
        if available_votes >= 2:
            if easing_votes >= 2:
                stress_score = 1
            elif stress_votes >= 2:
                stress_score = -1
            else:
                stress_score = 0
        component_scores["stress"][idx] = stress_score

        liquidity_score: int | None = None
        if version == "v1":
            if (
                curve_vals[idx] is not None
                and curve_ma[idx] is not None
                and curve_change[idx] is not None
                and nfci_change[idx] is not None
            ):
                curve_value = float(curve_vals[idx])
                curve_ma_value = float(curve_ma[idx])
                curve_delta = float(curve_change[idx])
                nfci_delta = float(nfci_change[idx])
                if curve_value > curve_ma_value and curve_delta > 0.0 and nfci_delta <= 0.0:
                    liquidity_score = 1
                elif curve_value < 0.0 and curve_delta < 0.0 and nfci_delta > 0.0:
                    liquidity_score = -1
                else:
                    liquidity_score = 0
        else:
            supportive_votes = 0
            restrictive_votes = 0
            liquidity_votes = 0

            if curve_vals[idx] is not None and curve_change[idx] is not None:
                liquidity_votes += 1
                if (
                    curve_ma[idx] is not None
                    and float(curve_vals[idx]) > float(curve_ma[idx])
                    and float(curve_change[idx]) > 0.0
                ):
                    supportive_votes += 1
                elif float(curve_vals[idx]) < 0.0 and float(curve_change[idx]) < 0.0:
                    restrictive_votes += 1

            if two_ten_vals[idx] is not None and two_ten_change[idx] is not None:
                liquidity_votes += 1
                if (
                    (
                        two_ten_ma[idx] is not None
                        and float(two_ten_vals[idx]) > float(two_ten_ma[idx])
                        and float(two_ten_change[idx]) >= 0.0
                    )
                    or (
                        float(two_ten_vals[idx]) >= 0.0
                        and float(two_ten_change[idx]) > 0.0
                    )
                ):
                    supportive_votes += 1
                elif float(two_ten_vals[idx]) < 0.0 and float(two_ten_change[idx]) < 0.0:
                    restrictive_votes += 1

            if nfci_vals[idx] is not None and nfci_change[idx] is not None:
                liquidity_votes += 1
                if (
                    nfci_ma[idx] is not None
                    and float(nfci_vals[idx]) < float(nfci_ma[idx])
                    and float(nfci_change[idx]) <= 0.0
                ):
                    supportive_votes += 1
                elif (
                    nfci_ma[idx] is not None
                    and float(nfci_vals[idx]) > float(nfci_ma[idx])
                    and float(nfci_change[idx]) > 0.0
                ):
                    restrictive_votes += 1

            if liquidity_votes >= 2:
                if supportive_votes >= 2:
                    liquidity_score = 1
                elif restrictive_votes >= 2:
                    liquidity_score = -1
                else:
                    liquidity_score = 0
        component_scores["liquidity"][idx] = liquidity_score

        row_components = [component_scores[name][idx] for name in ("dollar", "rates", "stress", "liquidity")]
        if any(component is None for component in row_components):
            continue
        raw_score = sum(int(component) for component in row_components if component is not None)
        score = max(-score_cap, min(score_cap, raw_score))
        raw_scores[idx] = raw_score
        scores[idx] = score
        if score >= 2:
            labels[idx] = "macro_tailwind"
        elif score >= 0:
            labels[idx] = "macro_neutral"
        elif score <= -2:
            labels[idx] = "macro_headwind"
        else:
            labels[idx] = "macro_negative"

    return {
        "dates": dates,
        "scores": scores,
        "raw_scores": raw_scores,
        "labels": labels,
        "version": version,
        "lookback_days": lookback_days,
        "dxy_fast_ma_days": dxy_fast_ma_days,
        "dxy_slow_ma_days": dxy_slow_ma_days,
        "rates_ma_days": rates_ma_days,
        "stress_ma_days": stress_ma_days,
        "rates_change_threshold_bps": rates_change_threshold_bps,
        "front_end_rates_change_threshold_bps": front_end_rates_change_threshold_bps,
        "vix_term_contango_ratio": vix_term_contango_ratio,
        "vix_term_backwardation_ratio": vix_term_backwardation_ratio,
        "score_cap": score_cap,
        "component_scores": component_scores,
    }


def lookup_macro_regime_signal(
    *,
    entry_ts: datetime,
    asset_bucket: str,
    direction: str,
    state: dict | None,
    lag_days: int = 1,
    gated_buckets: frozenset[str] | None = frozenset({"crypto", "equity", "etf"}),
    long_full_threshold: int = 2,
    long_half_threshold: int = 0,
    long_half_mult: float = 0.5,
    negative_long_mult: float = 0.0,
    short_full_threshold: int = -2,
    short_half_threshold: int = 0,
    short_half_mult: float = 0.5,
    positive_short_mult: float = 0.0,
) -> dict[str, object]:
    if not state or direction not in {"long", "short"}:
        return {
            "score_date": None,
            "score": None,
            "raw_score": None,
            "label": None,
            "mult": 1.0,
            "blocked": False,
            "action": "no_state",
            "components": {},
        }

    target_day = entry_ts.date() - timedelta(days=max(lag_days, 0))
    dates = state.get("dates", [])
    idx = bisect.bisect_right(dates, target_day) - 1
    if idx < 0:
        return {
            "score_date": None,
            "score": None,
            "raw_score": None,
            "label": None,
            "mult": 1.0,
            "blocked": False,
            "action": "pre_history",
            "components": {},
        }

    score = state["scores"][idx]
    raw_score = state["raw_scores"][idx]
    label = state["labels"][idx]
    component_scores = {
        name: values[idx]
        for name, values in state.get("component_scores", {}).items()
    }
    if score is None:
        return {
            "score_date": dates[idx].isoformat(),
            "score": None,
            "raw_score": raw_score,
            "label": "macro_unavailable",
            "mult": 1.0,
            "blocked": False,
            "action": "unavailable",
            "components": component_scores,
        }

    if gated_buckets is not None and asset_bucket not in gated_buckets:
        return {
            "score_date": dates[idx].isoformat(),
            "score": int(score),
            "raw_score": raw_score,
            "label": label,
            "mult": 1.0,
            "blocked": False,
            "action": "ungated_real_asset",
            "components": component_scores,
        }

    if direction == "long":
        if int(score) >= long_full_threshold:
            mult = 1.0
            blocked = False
            action = "full_long"
        elif int(score) >= long_half_threshold:
            mult = float(long_half_mult)
            blocked = mult <= 1e-9
            action = "half_long"
        else:
            mult = float(negative_long_mult)
            blocked = mult <= 1e-9
            action = "blocked_long" if blocked else "reduced_long"
    else:
        if int(score) <= short_full_threshold:
            mult = 1.0
            blocked = False
            action = "full_short"
        elif int(score) <= short_half_threshold:
            mult = float(short_half_mult)
            blocked = mult <= 1e-9
            action = "half_short"
        else:
            mult = float(positive_short_mult)
            blocked = mult <= 1e-9
            action = "blocked_short" if blocked else "reduced_short"

    return {
        "score_date": dates[idx].isoformat(),
        "score": int(score),
        "raw_score": int(raw_score) if raw_score is not None else None,
        "label": label,
        "mult": float(mult),
        "blocked": bool(blocked),
        "action": action,
        "components": component_scores,
    }
