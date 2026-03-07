"""Buffett-style stock assessment module.

Evaluates companies based on Warren Buffett's investment principles:
1. Consistent earnings growth (predictable earnings)
2. High return on equity (ROE > 15%)
3. Low debt-to-equity ratio
4. Good profit margins
5. Strong free cash flow generation
6. Margin of safety (intrinsic value vs current price)

Each criterion scores 0-100, and the overall score is a weighted average.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from edgar.models import (
    BuffettScore,
    EdgarCompany,
    EdgarFundamental,
    StockPrice,
)
from edgar.services.metric_mapping import (
    STANDARD_METRICS,
    resolve_and_store_metric_mapping,
)

logger = logging.getLogger(__name__)

WEIGHTS = {
    "roe": 0.20,
    "debt": 0.15,
    "margin": 0.15,
    "earnings_growth": 0.20,
    "fcf": 0.15,
    "valuation": 0.15,
}


def _get_annual_metric_series(company: EdgarCompany, mapping: dict, metric_key: str) -> list[tuple[int, float]]:
    """Return sorted list of (calendar_year, value) for a mapped metric.

    SEC XBRL 10-K filings contain both quarterly sub-period values and the
    annual total, all labelled fp=FY. At the fiscal year-end date (e.g.
    2020-12-31) there are typically two values: the Q4 quarter (~$800M) and
    the full-year total (~$3B).

    Strategy: group by end_date. At each end_date, when multiple values
    exist, the annual total is the largest absolute value. We only keep
    end_dates that fall at a year boundary (month 12 for Dec fiscal year-end
    companies, or the latest end_date per calendar year). This gives one
    clean annual data point per year.
    """
    info = mapping.get(metric_key)
    if not info:
        return []

    points = (
        EdgarFundamental.objects.filter(
            company=company,
            taxonomy=info["taxonomy"],
            tag=info["tag"],
            form__in=["10-K", "20-F", "40-F"],
        )
        .order_by("end_date")
        .values_list("end_date", "value", "fiscal_year")
    )

    # Collect candidates: at each end_date there may be multiple values
    # (Q4 quarterly vs full-year total). The annual total has the largest
    # absolute magnitude.  We only keep points where end_date falls at the
    # fiscal year end (end_date.year matches the calendar year of the data).
    from collections import defaultdict as _dd

    # Group: (end_date_year) -> list of (end_date, |value|, value, fiscal_year)
    by_ed_year: dict[int, list[tuple[str, float, float, int | None]]] = _dd(list)
    for end_date, val, fy in points:
        if val is None or end_date is None:
            continue
        by_ed_year[end_date.year].append(
            (end_date.isoformat(), abs(val), val, fy)
        )

    by_year: dict[int, float] = {}
    for ed_year, entries in by_ed_year.items():
        # Sort: latest end_date first, then largest |value| first.
        # The annual total at the fiscal year-end date is what we want.
        entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = entries[0]
        year = ed_year  # use end_date's year as the canonical year
        by_year[year] = best[2]

    return sorted(by_year.items())


def _score_roe(series_equity: list, series_net_income: list) -> tuple[float | None, float]:
    """Score ROE: average ROE across years. Target: >15% = 100, <5% = 0."""
    if len(series_equity) < 2 or len(series_net_income) < 2:
        return None, 0

    eq_dict = dict(series_equity)
    ni_dict = dict(series_net_income)
    common_years = sorted(set(eq_dict) & set(ni_dict))

    if not common_years:
        return None, 0

    roes = []
    for y in common_years:
        eq = eq_dict[y]
        ni = ni_dict[y]
        if eq and eq > 0:
            roes.append(ni / eq)

    if not roes:
        return None, 0

    avg_roe = sum(roes) / len(roes)
    score = min(100, max(0, (avg_roe - 0.05) / 0.10 * 100))
    return avg_roe, score


def _score_debt(series_liabilities: list, series_equity: list) -> tuple[float | None, float]:
    """Score D/E ratio: latest year. Target: <0.5 = 100, >2.0 = 0."""
    if not series_liabilities or not series_equity:
        return None, 0

    latest_year = max(dict(series_liabilities).keys() & dict(series_equity).keys(), default=None)
    if latest_year is None:
        return None, 0

    liab = dict(series_liabilities)[latest_year]
    eq = dict(series_equity)[latest_year]

    if not eq or eq <= 0:
        return None, 0

    de = liab / eq
    score = min(100, max(0, (2.0 - de) / 1.5 * 100))
    return de, score


def _score_margin(series_revenue: list, series_net_income: list) -> tuple[float | None, float]:
    """Score net profit margin: average across years. Target: >20% = 100, <5% = 0."""
    rev_dict = dict(series_revenue)
    ni_dict = dict(series_net_income)
    common = sorted(set(rev_dict) & set(ni_dict))

    if not common:
        return None, 0

    margins = []
    for y in common:
        rev = rev_dict[y]
        ni = ni_dict[y]
        if rev and rev > 0:
            margins.append(ni / rev)

    if not margins:
        return None, 0

    avg = sum(margins) / len(margins)
    score = min(100, max(0, (avg - 0.05) / 0.15 * 100))
    return avg, score


def _score_earnings_growth(series_net_income: list) -> tuple[float | None, float]:
    """Score earnings CAGR over available years. Target: >10% = 100, <0% = 0."""
    if len(series_net_income) < 3:
        return None, 0

    first_val = series_net_income[0][1]
    last_val = series_net_income[-1][1]
    n_years = series_net_income[-1][0] - series_net_income[0][0]

    if n_years <= 0 or first_val <= 0 or last_val <= 0:
        return None, 0

    cagr = (last_val / first_val) ** (1 / n_years) - 1
    score = min(100, max(0, cagr / 0.10 * 100))
    return cagr, score


def _score_fcf(series_ocf: list, series_capex: list) -> tuple[float | None, float]:
    """Score free cash flow consistency. Positive FCF in most years = good.

    When capex data is unavailable, falls back to operating cash flow as a
    proxy (OCF > 0 is still a strong Buffett signal).
    """
    ocf_dict = dict(series_ocf)
    capex_dict = dict(series_capex)

    if capex_dict:
        common = sorted(set(ocf_dict) & set(capex_dict))
    else:
        # Fallback: use OCF alone (treat capex as 0)
        common = sorted(ocf_dict.keys())

    if not common:
        return None, 0

    fcfs = []
    for y in common:
        ocf = ocf_dict.get(y) or 0
        capex = abs(capex_dict.get(y) or 0)
        fcfs.append(ocf - capex)

    if not fcfs:
        return None, 0

    avg_fcf = sum(fcfs) / len(fcfs)
    positive_pct = sum(1 for f in fcfs if f > 0) / len(fcfs)
    score = min(100, positive_pct * 100)
    return avg_fcf, score


def _estimate_intrinsic_value(
    series_fcf_values: list[float],
    growth_rate: float | None,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    shares_outstanding: float | None = None,
) -> float | None:
    """Simple DCF intrinsic value estimate per share."""
    if not series_fcf_values or growth_rate is None:
        return None

    latest_fcf = series_fcf_values[-1]
    if latest_fcf <= 0:
        return None

    g = min(max(growth_rate, 0.0), 0.20)
    projection_years = 10
    total_pv = 0
    for i in range(1, projection_years + 1):
        projected = latest_fcf * (1 + g) ** i
        total_pv += projected / (1 + discount_rate) ** i

    terminal_value = (latest_fcf * (1 + g) ** projection_years * (1 + terminal_growth)) / (
        discount_rate - terminal_growth
    )
    total_pv += terminal_value / (1 + discount_rate) ** projection_years

    if shares_outstanding and shares_outstanding > 0:
        return total_pv / shares_outstanding
    return total_pv


def _score_valuation(intrinsic: float | None, current_price: float | None) -> tuple[float | None, float]:
    """Score margin of safety. Target: price < 70% of intrinsic = 100."""
    if intrinsic is None or current_price is None or intrinsic <= 0:
        return None, 0

    margin_of_safety = (intrinsic - current_price) / intrinsic
    score = min(100, max(0, margin_of_safety / 0.30 * 100))
    return margin_of_safety, score


def compute_buffett_score(
    company: EdgarCompany,
    force_refresh: bool = False,
) -> BuffettScore:
    """Compute and persist a Buffett-style assessment for a company."""
    if not force_refresh:
        existing = BuffettScore.objects.filter(company=company).first()
        if existing:
            return existing

    mapping = resolve_and_store_metric_mapping(company=company, use_ai=False, persist=False)

    revenue = _get_annual_metric_series(company, mapping, "revenue")
    net_income = _get_annual_metric_series(company, mapping, "net_income")
    equity = _get_annual_metric_series(company, mapping, "equity")
    liabilities = _get_annual_metric_series(company, mapping, "liabilities")
    ocf = _get_annual_metric_series(company, mapping, "operating_cash_flow")
    capex = _get_annual_metric_series(company, mapping, "capex")

    roe_avg, roe_score = _score_roe(equity, net_income)
    de_ratio, debt_score = _score_debt(liabilities, equity)
    margin_avg, margin_score = _score_margin(revenue, net_income)
    earnings_cagr, eg_score = _score_earnings_growth(net_income)
    fcf_avg, fcf_score = _score_fcf(ocf, capex)

    latest_price_obj = StockPrice.objects.filter(company=company).order_by("-date").first()
    current_price = latest_price_obj.close if latest_price_obj else None

    # Build FCF series; fall back to OCF if capex unmapped
    ocf_dict = dict(ocf)
    capex_dict = dict(capex)
    if capex_dict:
        fcf_years = sorted(set(ocf_dict) & set(capex_dict))
    else:
        fcf_years = sorted(ocf_dict.keys())
    fcf_values = [(ocf_dict[y] or 0) - abs(capex_dict.get(y) or 0) for y in fcf_years]

    # Get shares outstanding for per-share intrinsic value
    shares_series = _get_annual_metric_series(company, mapping, "shares_outstanding")
    shares = dict(shares_series).get(max(dict(shares_series).keys())) if shares_series else None

    intrinsic = _estimate_intrinsic_value(
        fcf_values,
        growth_rate=earnings_cagr,
        shares_outstanding=shares,
    )
    margin_of_safety, val_score = _score_valuation(intrinsic, current_price)

    overall = (
        WEIGHTS["roe"] * roe_score
        + WEIGHTS["debt"] * debt_score
        + WEIGHTS["margin"] * margin_score
        + WEIGHTS["earnings_growth"] * eg_score
        + WEIGHTS["fcf"] * fcf_score
        + WEIGHTS["valuation"] * val_score
    )

    score, _ = BuffettScore.objects.update_or_create(
        company=company,
        defaults={
            "overall_score": round(overall, 2),
            "roe_avg": roe_avg,
            "roe_score": round(roe_score, 2),
            "debt_to_equity": de_ratio,
            "debt_score": round(debt_score, 2),
            "margin_avg": margin_avg,
            "margin_score": round(margin_score, 2),
            "earnings_growth": earnings_cagr,
            "earnings_growth_score": round(eg_score, 2),
            "fcf_avg": fcf_avg,
            "fcf_score": round(fcf_score, 2),
            "intrinsic_value": intrinsic,
            "current_price": current_price,
            "margin_of_safety": margin_of_safety,
            "valuation_score": round(val_score, 2),
            "detail": {
                "years_of_data": len(net_income),
                "revenue_years": len(revenue),
                "mapping_keys": list(mapping.keys()),
            },
        },
    )
    return score


def bulk_compute_scores(
    tickers: list[str] | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Compute Buffett scores for multiple companies."""
    if tickers:
        companies = EdgarCompany.objects.filter(ticker__in=[t.upper() for t in tickers])
    else:
        companies = EdgarCompany.objects.filter(fundamentals__isnull=False).distinct()

    results = []
    for company in companies:
        try:
            score = compute_buffett_score(company, force_refresh=force_refresh)
            results.append({
                "ticker": company.ticker,
                "overall_score": score.overall_score,
                "roe_score": score.roe_score,
                "debt_score": score.debt_score,
                "margin_score": score.margin_score,
                "earnings_growth_score": score.earnings_growth_score,
                "fcf_score": score.fcf_score,
                "valuation_score": score.valuation_score,
                "current_price": score.current_price,
                "intrinsic_value": score.intrinsic_value,
                "margin_of_safety": score.margin_of_safety,
            })
        except Exception as exc:
            logger.warning("Score computation failed for %s: %s", company.ticker, exc)
            results.append({"ticker": company.ticker, "error": str(exc)})
    return results
