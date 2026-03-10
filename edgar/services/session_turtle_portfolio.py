from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from edgar.services.session_turtle_trend_strategy import run_session_turtle_trend_backtest


DEFAULT_SESSION_TURTLE_UNIVERSE: tuple[tuple[str, str, str], ...] = (
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

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"}
EQUITY_TICKERS = {"AMZN", "COIN", "CRCL", "HOOD", "INTC", "MSTR", "PLTR", "TSLA"}
METAL_TICKERS = {"GLD", "PPLT", "SLV"}


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
    scale: float
    scaled_position_size: float
    scaled_shares: float
    scaled_pnl: float


def _asset_bucket(ticker: str) -> str:
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in EQUITY_TICKERS:
        return "equity"
    if ticker in METAL_TICKERS:
        return "metals"
    return "other"


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
    exposure_mult: float = 2.0,
    initial_capital: float = 1_000.0,
    lookback_years: float = 4.1,
    channel_period: int = 20,
    base_risk_pct: float = 0.05,
    fixed_stop_pct: float = 0.10,
    directional_volume_risk_pct: float = 0.07,
    trend_fast_period: int = 55,
    trend_slow_period: int = 200,
    base_portfolio_cap_pct: float = 0.90,
) -> dict:
    candidates: list[dict] = []
    for combo_idx, (ticker, source, session_open) in enumerate(DEFAULT_SESSION_TURTLE_UNIVERSE):
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
    capital = float(initial_capital)
    peak_capital = capital
    max_drawdown = 0.0
    skipped_same_ticker = 0
    skipped_no_capacity = 0
    open_positions: list[_OpenTrade] = []
    executed_trades: list[dict] = []
    equity_curve: list[dict] = []

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
                    "scale": round(position.scale, 6),
                    "entry_rel_volume": round(position.entry_rel_volume, 4),
                    "risk_model": position.risk_model,
                    "net_pnl": round(position.scaled_pnl, 4),
                    "equity_after_exit": round(capital, 4),
                }
            )

    for candidate in candidates:
        close_positions_up_to(candidate["entry_ts"])
        if any(position.ticker == candidate["ticker"] for position in open_positions):
            skipped_same_ticker += 1
            continue

        portfolio_cap = capital * base_portfolio_cap_pct * exposure_mult
        used_notional = sum(position.scaled_position_size for position in open_positions)
        available_notional = max(portfolio_cap - used_notional, 0.0)
        if available_notional <= 1e-9:
            skipped_no_capacity += 1
            continue

        scaled_position_size = min(float(candidate["position_size"]), available_notional)
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
                asset_bucket=str(candidate["asset_bucket"]),
                scale=scale,
                scaled_position_size=scaled_position_size,
                scaled_shares=float(candidate["shares"]) * scale,
                scaled_pnl=float(candidate["pnl"]) * scale,
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
    for trade in executed_trades:
        bucket_pnl[trade["asset_bucket"]] += float(trade["net_pnl"])

    summary = {
        "strategy_variant": "session_turtle_trend_shared_account",
        "label": f"Session Turtle Trend x{exposure_mult:g}",
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
        "channel_period": channel_period,
        "lookback_years": lookback_years,
        "base_risk_pct": base_risk_pct,
        "directional_volume_risk_pct": directional_volume_risk_pct,
        "trend_fast_period": trend_fast_period,
        "trend_slow_period": trend_slow_period,
        "crypto_trades": bucket_counter["crypto"],
        "equity_trades": bucket_counter["equity"],
        "metals_trades": bucket_counter["metals"],
        "crypto_pnl": round(bucket_pnl["crypto"], 4),
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
