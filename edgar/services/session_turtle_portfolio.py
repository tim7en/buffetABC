from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest


CORE_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("BTC-USD", "binance", "hong_kong_open"),
    ("BTC-USD", "binance", "new_york_equity_open"),
    ("ETH-USD", "binance", "hong_kong_open"),
    ("ETH-USD", "binance", "new_york_equity_open"),
    ("SOL-USD", "binance", "hong_kong_open"),
    ("SOL-USD", "binance", "new_york_equity_open"),
    ("PAXG-USD", "binance", "hong_kong_open"),
    ("PAXG-USD", "binance", "new_york_equity_open"),
    ("AMZN", "tiingo", "new_york_equity_open"),
    ("COIN", "tiingo", "new_york_equity_open"),
    ("COPPER", "tiingo", "new_york_equity_open"),
    ("CRCL", "tiingo", "new_york_equity_open"),
    ("GLD", "tiingo", "new_york_equity_open"),
    ("HOOD", "tiingo", "new_york_equity_open"),
    ("INTC", "tiingo", "new_york_equity_open"),
    ("MSTR", "tiingo", "new_york_equity_open"),
    ("PLTR", "tiingo", "new_york_equity_open"),
    ("PPLT", "tiingo", "new_york_equity_open"),
    ("SLV", "tiingo", "new_york_equity_open"),
    ("TSLA", "tiingo", "new_york_equity_open"),
)

INDEX_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    ("QQQ", "tiingo", "new_york_equity_open"),
    ("SPY", "tiingo", "new_york_equity_open"),
)

EXPANDED_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
    *CORE_SESSION_TURTLE_UNIVERSE,
    *INDEX_SESSION_TURTLE_UNIVERSE,
)

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}
GOLD_TICKERS = {"PAXG-USD", "GLD"}
EQUITY_TICKERS = {"AMZN", "COIN", "CRCL", "HOOD", "INTC", "MSTR", "PLTR", "QQQ", "SPY", "TSLA"}
METAL_TICKERS = {"COPPER", "PPLT", "SLV"}


def _resolve_universe(basket: str) -> tuple[tuple[str, str, str], ...]:
    key = (basket or "expanded").strip().lower()
    if key == "core":
        return CORE_SESSION_TURTLE_UNIVERSE
    if key == "index":
        return INDEX_SESSION_TURTLE_UNIVERSE
    if key == "expanded":
        return EXPANDED_SESSION_TURTLE_UNIVERSE
    raise ValueError("basket must be one of {'core', 'index', 'expanded'}")


@dataclass
class _OpenTrade:
    combo_idx: int
    trade_idx: int
    ticker: str
    source: str
    session_open: str
    direction: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    shares: float
    position_size: float
    pnl: float
    risk_model: str
    entry_rel_volume: float
    asset_bucket: str
    entry_exposure_mult: float
    scale: float
    scaled_position_size: float
    scaled_shares: float
    scaled_pnl: float
    performance_risk_mult: float
    performance_score: float | None
    performance_rank_pct: float | None
    performance_peer_count: int


def _asset_bucket(ticker: str) -> str:
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in GOLD_TICKERS:
        return "gold"
    if ticker in EQUITY_TICKERS:
        return "equity"
    if ticker in METAL_TICKERS:
        return "metals"
    return "other"


def _realized_drawdown_pct(capital: float, peak_capital: float) -> float:
    if peak_capital <= 0:
        return 0.0
    return (peak_capital - capital) / peak_capital * 100.0


def _candidate_return_pct(candidate: dict) -> float:
    position_size = float(candidate.get("position_size", 0.0) or 0.0)
    if position_size <= 0:
        return 0.0
    return float(candidate.get("pnl", 0.0) or 0.0) / position_size


def _decayed_trade_return_score(returns: list[float], lookback_trades: int, decay: float) -> float:
    if lookback_trades <= 0 or decay <= 0:
        return 0.0
    recent = returns[-lookback_trades:]
    if not recent:
        return 0.0
    total = 0.0
    weights = 0.0
    for age, value in enumerate(reversed(recent)):
        weight = decay**age
        total += float(value) * weight
        weights += weight
    return total / weights if weights > 0 else 0.0


def _performance_leadership_scale(
    *,
    ticker: str,
    closed_trade_returns_by_ticker: dict[str, list[float]],
    lookback_trades: int,
    decay: float,
    floor_mult: float,
    cap_mult: float,
    min_history: int,
) -> tuple[float, float | None, float | None, int]:
    if lookback_trades <= 0 or min_history <= 0:
        return 1.0, None, None, 0

    scores: dict[str, float] = {}
    for asset_ticker, returns in closed_trade_returns_by_ticker.items():
        if len(returns) < min_history:
            continue
        scores[str(asset_ticker)] = _decayed_trade_return_score(
            returns=returns,
            lookback_trades=lookback_trades,
            decay=decay,
        )
    if ticker not in scores or len(scores) < 2:
        return 1.0, scores.get(ticker), None, len(scores)

    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    ticker_idx = next(idx for idx, item in enumerate(ordered) if item[0] == ticker)
    rank_pct = ticker_idx / (len(ordered) - 1) if len(ordered) > 1 else 0.5
    mult = floor_mult + (rank_pct * (cap_mult - floor_mult))
    return mult, scores[ticker], rank_pct, len(scores)


def _active_exposure_mult(
    *,
    base_exposure_mult: float,
    capital: float,
    peak_capital: float,
    use_drawdown_governor: bool,
    drawdown_trigger_1_pct: float,
    drawdown_exposure_mult_1: float,
    drawdown_trigger_2_pct: float,
    drawdown_exposure_mult_2: float,
) -> float:
    if not use_drawdown_governor:
        return base_exposure_mult
    drawdown_pct = _realized_drawdown_pct(capital=capital, peak_capital=peak_capital)
    if drawdown_pct >= drawdown_trigger_2_pct:
        return drawdown_exposure_mult_2
    if drawdown_pct >= drawdown_trigger_1_pct:
        return drawdown_exposure_mult_1
    return base_exposure_mult


def _build_yearly_rows(executed_trades: list[dict], initial_capital: float) -> list[dict]:
    grouped: dict[int, dict] = defaultdict(
        lambda: {
            "pnl": 0.0,
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "winning_trades": 0,
        }
    )
    for trade in executed_trades:
        year = datetime.fromisoformat(trade["exit_ts"]).year
        row = grouped[year]
        row["pnl"] += float(trade["net_pnl"])
        row["trades"] += 1
        if trade["direction"] == "long":
            row["long_trades"] += 1
        else:
            row["short_trades"] += 1
        if float(trade["net_pnl"]) > 0:
            row["winning_trades"] += 1

    start_equity = float(initial_capital)
    yearly_rows: list[dict] = []
    for year in sorted(grouped):
        row = grouped[year]
        pnl = float(row["pnl"])
        end_equity = start_equity + pnl
        yearly_rows.append(
            {
                "year": year,
                "start_equity": round(start_equity, 4),
                "end_equity": round(end_equity, 4),
                "pnl": round(pnl, 4),
                "return_pct": round((pnl / start_equity * 100.0) if start_equity > 0 else 0.0, 2),
                "trades": row["trades"],
                "long_trades": row["long_trades"],
                "short_trades": row["short_trades"],
                "win_rate_pct": round((row["winning_trades"] / row["trades"] * 100.0) if row["trades"] else 0.0, 2),
            }
        )
        start_equity = end_equity
    return yearly_rows


def _build_asset_rows(executed_trades: list[dict], total_pnl: float) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "source": "",
            "asset_bucket": "",
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "pnl": 0.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        }
    )
    for trade in executed_trades:
        ticker = trade["ticker"]
        row = grouped[ticker]
        row["source"] = trade["source"]
        row["asset_bucket"] = trade["asset_bucket"]
        row["trades"] += 1
        row["pnl"] += float(trade["net_pnl"])
        if trade["direction"] == "long":
            row["long_trades"] += 1
            row["long_pnl"] += float(trade["net_pnl"])
        else:
            row["short_trades"] += 1
            row["short_pnl"] += float(trade["net_pnl"])

    rows: list[dict] = []
    for ticker, row in sorted(grouped.items(), key=lambda item: item[1]["pnl"], reverse=True):
        rows.append(
            {
                "ticker": ticker,
                "source": row["source"],
                "asset_bucket": row["asset_bucket"],
                "trades": row["trades"],
                "long_trades": row["long_trades"],
                "short_trades": row["short_trades"],
                "pnl": round(row["pnl"], 4),
                "pnl_share_pct": round((row["pnl"] / total_pnl * 100.0) if abs(total_pnl) > 1e-9 else 0.0, 2),
                "long_pnl": round(row["long_pnl"], 4),
                "short_pnl": round(row["short_pnl"], 4),
            }
        )
    return rows


def generate_session_turtle_shared_account_report(
    *,
    basket: str = "expanded",
    exposure_mult: float = 2.0,
    use_drawdown_governor: bool = False,
    drawdown_trigger_1_pct: float = 10.0,
    drawdown_exposure_mult_1: float = 1.5,
    drawdown_trigger_2_pct: float = 20.0,
    drawdown_exposure_mult_2: float = 1.0,
    crypto_cap_mult: float | None = None,
    gold_cap_mult: float | None = None,
    metals_cap_mult: float | None = None,
    equity_cap_mult: float | None = None,
    initial_capital: float = 1_000.0,
    lookback_years: float = 4.1,
    channel_period: int = 20,
    base_risk_pct: float = 0.05,
    fixed_stop_pct: float = 0.10,
    directional_volume_risk_pct: float = 0.07,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    base_portfolio_cap_pct: float = 0.90,
    use_performance_leadership_overlay: bool = False,
    performance_lookback_trades: int = 6,
    performance_decay: float = 0.75,
    performance_floor_mult: float = 0.75,
    performance_cap_mult: float = 1.25,
    performance_min_history: int = 3,
) -> dict:
    if exposure_mult <= 0:
        raise ValueError("exposure_mult must be positive")
    if drawdown_trigger_1_pct < 0 or drawdown_trigger_2_pct < 0:
        raise ValueError("drawdown triggers must be non-negative")
    if drawdown_trigger_2_pct <= drawdown_trigger_1_pct:
        raise ValueError("drawdown_trigger_2_pct must be greater than drawdown_trigger_1_pct")
    if drawdown_exposure_mult_1 <= 0 or drawdown_exposure_mult_2 <= 0:
        raise ValueError("drawdown exposure multipliers must be positive")
    if drawdown_exposure_mult_1 > exposure_mult:
        raise ValueError("drawdown_exposure_mult_1 must be <= exposure_mult")
    if drawdown_exposure_mult_2 > drawdown_exposure_mult_1:
        raise ValueError("drawdown_exposure_mult_2 must be <= drawdown_exposure_mult_1")
    for label, cap_mult in (
        ("crypto_cap_mult", crypto_cap_mult),
        ("gold_cap_mult", gold_cap_mult),
        ("metals_cap_mult", metals_cap_mult),
        ("equity_cap_mult", equity_cap_mult),
    ):
        if cap_mult is not None and cap_mult <= 0:
            raise ValueError(f"{label} must be positive when provided")
    if performance_lookback_trades <= 0:
        raise ValueError("performance_lookback_trades must be positive")
    if not 0 < performance_decay <= 1.0:
        raise ValueError("performance_decay must be between 0 and 1")
    if performance_floor_mult <= 0 or performance_cap_mult <= 0:
        raise ValueError("performance floor/cap multipliers must be positive")
    if performance_floor_mult > performance_cap_mult:
        raise ValueError("performance_floor_mult must be <= performance_cap_mult")
    if not performance_floor_mult <= 1.0 <= performance_cap_mult:
        raise ValueError("performance overlay must bracket neutral sizing at 1.0")
    if performance_min_history <= 0:
        raise ValueError("performance_min_history must be positive")
    universe = _resolve_universe(basket)

    candidates: list[dict] = []
    for combo_idx, (ticker, source, session_open) in enumerate(universe):
        payload = run_session_turtle_trend_backtest(
            ticker=ticker,
            initial_capital=initial_capital,
            interval="5m",
            lookback_years=lookback_years,
            market_data_source=source,
            session_open=session_open,
            channel_period=channel_period,
            base_risk_pct=base_risk_pct,
            max_position_pct=0.90,
            fixed_stop_pct=fixed_stop_pct,
            use_4h_trend_filter=True,
            trend_fast_period=trend_fast_period,
            trend_slow_period=trend_slow_period,
            use_directional_volume_risk_boost=True,
            directional_volume_min_rel_volume=1.25,
            directional_volume_close_location_threshold=0.65,
            directional_volume_risk_pct=directional_volume_risk_pct,
            enable_pyramiding=False,
            use_break_even_stop=False,
            use_chandelier_exit=False,
        )
        for trade_idx, trade in enumerate(payload["trades"]):
            candidates.append(
                {
                    "combo_idx": combo_idx,
                    "trade_idx": trade_idx,
                    "ticker": ticker,
                    "source": source,
                    "session_open": session_open,
                    "direction": trade["direction"],
                    "entry_ts": datetime.fromisoformat(trade["entry_date"]),
                    "exit_ts": datetime.fromisoformat(trade["exit_date"]),
                    "entry_price": float(trade["entry_price"]),
                    "exit_price": float(trade["exit_price"]),
                    "shares": float(trade["shares"]),
                    "position_size": float(trade["position_size"]),
                    "pnl": float(trade["pnl"]),
                    "risk_model": str(trade["risk_model"]),
                    "entry_rel_volume": float(trade["entry_rel_volume"]),
                    "asset_bucket": _asset_bucket(ticker),
                }
            )

    candidates.sort(key=lambda row: (row["entry_ts"], row["combo_idx"], row["trade_idx"]))
    closed_candidate_results = sorted(candidates, key=lambda row: (row["exit_ts"], row["combo_idx"], row["trade_idx"]))
    capital = float(initial_capital)
    peak_capital = capital
    max_drawdown = 0.0
    skipped_same_ticker = 0
    skipped_no_capacity = 0
    open_positions: list[_OpenTrade] = []
    executed_trades: list[dict] = []
    equity_curve: list[dict] = []
    asset_class_caps = {
        "crypto": crypto_cap_mult,
        "gold": gold_cap_mult,
        "metals": metals_cap_mult,
        "equity": equity_cap_mult,
    }
    closed_candidate_idx = 0
    closed_trade_returns_by_ticker: dict[str, list[float]] = defaultdict(list)

    def close_candidate_history_up_to(timestamp: datetime) -> None:
        nonlocal closed_candidate_idx
        while (
            closed_candidate_idx < len(closed_candidate_results)
            and closed_candidate_results[closed_candidate_idx]["exit_ts"] <= timestamp
        ):
            closed_candidate = closed_candidate_results[closed_candidate_idx]
            closed_trade_returns_by_ticker[str(closed_candidate["ticker"])].append(
                _candidate_return_pct(closed_candidate)
            )
            closed_candidate_idx += 1

    def close_positions_up_to(timestamp: datetime) -> None:
        nonlocal capital, peak_capital, max_drawdown, open_positions
        still_open: list[_OpenTrade] = []
        closing: list[_OpenTrade] = []
        for position in open_positions:
            if position.exit_ts <= timestamp:
                closing.append(position)
            else:
                still_open.append(position)
        open_positions = still_open

        for position in sorted(closing, key=lambda row: (row.exit_ts, row.combo_idx, row.trade_idx)):
            capital += position.scaled_pnl
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            equity_curve.append({"date": position.exit_ts.isoformat(), "equity": round(capital, 4)})
            executed_trades.append(
                {
                    "ticker": position.ticker,
                    "source": position.source,
                    "session_open": position.session_open,
                    "asset_bucket": position.asset_bucket,
                    "direction": position.direction,
                    "entry_ts": position.entry_ts.isoformat(),
                    "exit_ts": position.exit_ts.isoformat(),
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(position.exit_price, 4),
                    "shares": round(position.scaled_shares, 6),
                    "notional": round(position.scaled_position_size, 4),
                    "entry_exposure_mult": round(position.entry_exposure_mult, 4),
                    "scale": round(position.scale, 6),
                    "entry_rel_volume": round(position.entry_rel_volume, 4),
                    "risk_model": position.risk_model,
                    "performance_risk_mult": round(position.performance_risk_mult, 4),
                    "performance_score": (
                        round(position.performance_score, 6)
                        if position.performance_score is not None
                        else None
                    ),
                    "performance_rank_pct": (
                        round(position.performance_rank_pct, 4)
                        if position.performance_rank_pct is not None
                        else None
                    ),
                    "performance_peer_count": position.performance_peer_count,
                    "net_pnl": round(position.scaled_pnl, 4),
                    "equity_after_exit": round(capital, 4),
                }
            )

    for candidate in candidates:
        close_positions_up_to(candidate["entry_ts"])
        close_candidate_history_up_to(candidate["entry_ts"])
        if any(position.ticker == candidate["ticker"] for position in open_positions):
            skipped_same_ticker += 1
            continue

        active_exposure_mult = _active_exposure_mult(
            base_exposure_mult=exposure_mult,
            capital=capital,
            peak_capital=peak_capital,
            use_drawdown_governor=use_drawdown_governor,
            drawdown_trigger_1_pct=drawdown_trigger_1_pct,
            drawdown_exposure_mult_1=drawdown_exposure_mult_1,
            drawdown_trigger_2_pct=drawdown_trigger_2_pct,
            drawdown_exposure_mult_2=drawdown_exposure_mult_2,
        )
        portfolio_cap = capital * base_portfolio_cap_pct * active_exposure_mult
        used_notional = sum(position.scaled_position_size for position in open_positions)
        available_notional = max(portfolio_cap - used_notional, 0.0)
        asset_bucket = str(candidate["asset_bucket"])
        asset_class_cap_mult = asset_class_caps.get(asset_bucket)
        if asset_class_cap_mult is not None:
            class_cap = capital * base_portfolio_cap_pct * asset_class_cap_mult
            used_class_notional = sum(
                position.scaled_position_size
                for position in open_positions
                if position.asset_bucket == asset_bucket
            )
            available_notional = min(available_notional, max(class_cap - used_class_notional, 0.0))
        if available_notional <= 1e-9:
            skipped_no_capacity += 1
            continue

        performance_risk_mult = 1.0
        performance_score: float | None = None
        performance_rank_pct: float | None = None
        performance_peer_count = 0
        if use_performance_leadership_overlay:
            (
                performance_risk_mult,
                performance_score,
                performance_rank_pct,
                performance_peer_count,
            ) = _performance_leadership_scale(
                ticker=str(candidate["ticker"]),
                closed_trade_returns_by_ticker=closed_trade_returns_by_ticker,
                lookback_trades=performance_lookback_trades,
                decay=performance_decay,
                floor_mult=performance_floor_mult,
                cap_mult=performance_cap_mult,
                min_history=performance_min_history,
            )

        target_position_size = float(candidate["position_size"]) * performance_risk_mult
        scaled_position_size = min(target_position_size, available_notional)
        if scaled_position_size <= 1e-9:
            skipped_no_capacity += 1
            continue

        scale = scaled_position_size / float(candidate["position_size"]) if float(candidate["position_size"]) > 0 else 0.0
        open_positions.append(
            _OpenTrade(
                combo_idx=int(candidate["combo_idx"]),
                trade_idx=int(candidate["trade_idx"]),
                ticker=str(candidate["ticker"]),
                source=str(candidate["source"]),
                session_open=str(candidate["session_open"]),
                direction=str(candidate["direction"]),
                entry_ts=candidate["entry_ts"],
                exit_ts=candidate["exit_ts"],
                entry_price=float(candidate["entry_price"]),
                exit_price=float(candidate["exit_price"]),
                shares=float(candidate["shares"]),
                position_size=float(candidate["position_size"]),
                pnl=float(candidate["pnl"]),
                risk_model=str(candidate["risk_model"]),
                entry_rel_volume=float(candidate["entry_rel_volume"]),
                asset_bucket=asset_bucket,
                entry_exposure_mult=active_exposure_mult,
                scale=scale,
                scaled_position_size=scaled_position_size,
                scaled_shares=float(candidate["shares"]) * scale,
                scaled_pnl=float(candidate["pnl"]) * scale,
                performance_risk_mult=performance_risk_mult,
                performance_score=performance_score,
                performance_rank_pct=performance_rank_pct,
                performance_peer_count=performance_peer_count,
            )
        )

    close_positions_up_to(datetime.max)

    if executed_trades:
        start_date = executed_trades[0]["entry_ts"]
        end_date = executed_trades[-1]["exit_ts"]
        equity_curve.insert(0, {"date": start_date, "equity": round(initial_capital, 4)})
    else:
        start_date = None
        end_date = None

    total_pnl = capital - initial_capital
    years = (
        max((datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
        if start_date and end_date
        else 1 / 365.25
    )
    winning_trades = [trade for trade in executed_trades if float(trade["net_pnl"]) > 0]
    losing_trades = [trade for trade in executed_trades if float(trade["net_pnl"]) <= 0]
    gross_profit = sum(float(trade["net_pnl"]) for trade in winning_trades)
    gross_loss_abs = abs(sum(float(trade["net_pnl"]) for trade in losing_trades))
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    long_trades = [trade for trade in executed_trades if trade["direction"] == "long"]
    short_trades = [trade for trade in executed_trades if trade["direction"] == "short"]

    yearly_rows = _build_yearly_rows(executed_trades, initial_capital)
    asset_rows = _build_asset_rows(executed_trades, total_pnl)
    bucket_counter = Counter(trade["asset_bucket"] for trade in executed_trades)
    bucket_pnl = Counter()
    exposure_counter = Counter()
    for trade in executed_trades:
        bucket_pnl[trade["asset_bucket"]] += float(trade["net_pnl"])
        exposure_counter[float(trade["entry_exposure_mult"])] += 1
    performance_mults = [float(trade.get("performance_risk_mult", 1.0) or 1.0) for trade in executed_trades]

    label = f"Session Turtle Trend {basket.capitalize()} x{exposure_mult:g}"
    if use_drawdown_governor:
        label += " With DD Governor"
    if use_performance_leadership_overlay:
        label += " With Leadership Overlay"
    if any(cap is not None for cap in asset_class_caps.values()):
        label += " With Asset Class Caps"

    summary = {
        "strategy_variant": "session_turtle_trend_shared_account",
        "label": label,
        "basket": basket,
        "start_date": start_date,
        "end_date": end_date,
        "candidate_trades": len(candidates),
        "executed_trades": len(executed_trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "skipped_same_ticker": skipped_same_ticker,
        "skipped_no_capacity": skipped_no_capacity,
        "initial_capital": round(initial_capital, 4),
        "final_equity": round(capital, 4),
        "total_return_pct": round((capital / initial_capital - 1.0) * 100.0, 2) if initial_capital > 0 else 0.0,
        "cagr_pct": round(((capital / initial_capital) ** (1 / years) - 1.0) * 100.0, 2) if initial_capital > 0 else 0.0,
        "max_realized_drawdown_pct": round(max_drawdown * 100.0, 2),
        "win_rate_pct": round((len(winning_trades) / len(executed_trades) * 100.0) if executed_trades else 0.0, 2),
        "profit_factor": round(profit_factor, 2),
        "exposure_mult": exposure_mult,
        "use_drawdown_governor": use_drawdown_governor,
        "drawdown_trigger_1_pct": round(drawdown_trigger_1_pct, 2),
        "drawdown_exposure_mult_1": round(drawdown_exposure_mult_1, 4),
        "drawdown_trigger_2_pct": round(drawdown_trigger_2_pct, 2),
        "drawdown_exposure_mult_2": round(drawdown_exposure_mult_2, 4),
        "crypto_cap_mult": crypto_cap_mult,
        "gold_cap_mult": gold_cap_mult,
        "metals_cap_mult": metals_cap_mult,
        "equity_cap_mult": equity_cap_mult,
        "entries_at_base_exposure": exposure_counter[float(exposure_mult)],
        "entries_at_drawdown_exposure_1": exposure_counter[float(drawdown_exposure_mult_1)],
        "entries_at_drawdown_exposure_2": exposure_counter[float(drawdown_exposure_mult_2)],
        "channel_period": channel_period,
        "lookback_years": lookback_years,
        "base_risk_pct": base_risk_pct,
        "directional_volume_risk_pct": directional_volume_risk_pct,
        "use_performance_leadership_overlay": use_performance_leadership_overlay,
        "performance_lookback_trades": performance_lookback_trades,
        "performance_decay": performance_decay,
        "performance_floor_mult": performance_floor_mult,
        "performance_cap_mult": performance_cap_mult,
        "performance_min_history": performance_min_history,
        "avg_performance_risk_mult": (
            round(sum(performance_mults) / len(performance_mults), 4) if performance_mults else 1.0
        ),
        "entries_performance_upscaled": sum(1 for mult in performance_mults if mult > 1.000001),
        "entries_performance_downscaled": sum(1 for mult in performance_mults if mult < 0.999999),
        "trend_fast_period": trend_fast_period,
        "trend_slow_period": trend_slow_period,
        "crypto_trades": bucket_counter["crypto"],
        "gold_trades": bucket_counter["gold"],
        "equity_trades": bucket_counter["equity"],
        "metals_trades": bucket_counter["metals"],
        "crypto_pnl": round(bucket_pnl["crypto"], 4),
        "gold_pnl": round(bucket_pnl["gold"], 4),
        "equity_pnl": round(bucket_pnl["equity"], 4),
        "metals_pnl": round(bucket_pnl["metals"], 4),
    }

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": executed_trades,
        "yearly_returns": yearly_rows,
        "asset_summary": asset_rows,
    }
