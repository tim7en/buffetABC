from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_equity_and_drawdown(
    equity_curve: list[dict],
    output_path: str | Path,
    *,
    title: str,
    initial_capital: float,
) -> Path:
    output = _prepare_output(output_path)
    if not equity_curve:
        raise ValueError("equity_curve is required to plot equity and drawdown")

    dates = [row["date"] for row in equity_curve]
    equities = [float(row["equity"]) for row in equity_curve]
    running_peak = []
    peak = float(initial_capital)
    for equity in equities:
        peak = max(peak, equity)
        running_peak.append(peak)
    drawdowns = [((peak_now - equity) / peak_now * 100.0) if peak_now > 0 else 0.0 for equity, peak_now in zip(equities, running_peak)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(dates, equities, color="#0f766e", linewidth=2.0, label="Equity")
    axes[0].plot(dates, running_peak, color="#94a3b8", linewidth=1.2, linestyle="--", label="Peak")
    axes[0].set_ylabel("Equity")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(dates, drawdowns, 0, color="#dc2626", alpha=0.85)
    axes[1].set_ylabel("DD %")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_yearly_pnl(yearly_rows: list[dict], output_path: str | Path, *, title: str) -> Path:
    output = _prepare_output(output_path)
    if not yearly_rows:
        raise ValueError("yearly_rows are required to plot yearly pnl")

    years = [str(row["year"]) for row in yearly_rows]
    pnl_values = [float(row["pnl"]) for row in yearly_rows]
    colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in pnl_values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(years, pnl_values, color=colors)
    ax.axhline(0, color="#1f2937", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("PnL")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, pnl_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_asset_pnl(asset_rows: list[dict], output_path: str | Path, *, title: str, top_n: int = 12) -> Path:
    output = _prepare_output(output_path)
    if not asset_rows:
        raise ValueError("asset_rows are required to plot asset pnl")

    ordered = sorted(asset_rows, key=lambda row: float(row["pnl"]), reverse=True)[:top_n]
    labels = [str(row["ticker"]) for row in ordered]
    pnl_values = [float(row["pnl"]) for row in ordered]
    colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in pnl_values]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, pnl_values, color=colors)
    ax.axvline(0, color="#1f2937", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("PnL")
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()
    for bar, value in zip(bars, pnl_values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2.0,
            f" {value:.0f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
