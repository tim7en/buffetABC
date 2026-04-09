"""Build a consensus regime chart from lagged daily GMM and logistic signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS_CSV = ROOT / "reports" / "qqq_macro_walkforward_model_compare_20260409_monthly_equal" / "walkforward_model_signals_daily.csv"
DEFAULT_DATASET_CSV = ROOT / "reports" / "qqq_macro_ml_regime_analysis" / "aligned_daily_dataset.csv"
DEFAULT_OUT_PATH = ROOT / "reports" / "macro_regime_edge_audit_20260409" / "consensus_regimes_full_common_window.png"
DEFAULT_EXPORT_CSV = ROOT / "reports" / "macro_regime_edge_audit_20260409" / "consensus_signal_lag1_daily.csv"

REGIME_FILL_COLORS = {
    "risk_on": "#dff3e4",
    "neutral": "#f8f3d9",
    "risk_off": "#f8d7da",
}


def build_consensus_signal(signals: pd.DataFrame) -> pd.Series:
    gmm = signals["gmm_signal_lag1"].astype("object")
    logistic = signals["logistic_signal_lag1"].astype("object")
    both_known = gmm.notna() & logistic.notna()

    consensus = pd.Series(np.nan, index=signals.index, dtype="object")
    consensus.loc[both_known & ((gmm == "risk_off") | (logistic == "risk_off"))] = "risk_off"
    consensus.loc[both_known & (gmm == "risk_on") & (logistic == "risk_on")] = "risk_on"
    consensus.loc[both_known & consensus.isna()] = "neutral"
    return consensus


def regime_spans(signal: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current_regime: str | None = None
    start_date: pd.Timestamp | None = None
    previous_date: pd.Timestamp | None = None
    for date, regime in signal.dropna().astype(str).items():
        if regime != current_regime:
            if current_regime is not None and start_date is not None and previous_date is not None:
                spans.append((start_date, previous_date, current_regime))
            current_regime = regime
            start_date = pd.Timestamp(date)
        previous_date = pd.Timestamp(date)
    if current_regime is not None and start_date is not None and previous_date is not None:
        spans.append((start_date, previous_date, current_regime))
    return spans


def plot_consensus_chart(close: pd.Series, signal: pd.Series, out_path: Path) -> None:
    signal = signal.reindex(close.index)
    signal = signal.dropna()
    if signal.empty:
        raise ValueError("Consensus signal is empty; nothing to plot.")

    start_date = signal.index.min()
    price = close.loc[close.index >= start_date].copy()
    signal = signal.reindex(price.index).ffill()
    regime_level = signal.map({"risk_off": 0.0, "neutral": 1.0, "risk_on": 2.0})

    fig, (ax_price, ax_regime) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15]},
    )

    for span_start, span_end, regime in regime_spans(signal):
        color = REGIME_FILL_COLORS.get(regime, "#e5e7eb")
        ax_price.axvspan(span_start, span_end, color=color, alpha=0.55, linewidth=0)
        ax_regime.axvspan(span_start, span_end, color=color, alpha=0.55, linewidth=0)

    ax_price.plot(price.index, price.astype(float), color="#0f172a", linewidth=1.7, label="QQQ close")
    ax_price.set_title("Consensus traded regimes on QQQ")
    ax_price.set_ylabel("QQQ close")
    ax_price.set_yscale("log")
    ax_price.grid(alpha=0.25)
    ax_price.legend(
        handles=[
            Patch(facecolor=REGIME_FILL_COLORS["risk_on"], edgecolor="none", label="Risk on"),
            Patch(facecolor=REGIME_FILL_COLORS["neutral"], edgecolor="none", label="Neutral"),
            Patch(facecolor=REGIME_FILL_COLORS["risk_off"], edgecolor="none", label="Risk off"),
        ],
        loc="upper left",
        ncol=3,
    )

    ax_regime.step(regime_level.index, regime_level.astype(float), where="post", color="#111827", linewidth=1.4)
    ax_regime.set_ylim(-0.5, 2.5)
    ax_regime.set_yticks([0.0, 1.0, 2.0], labels=["Risk off", "Neutral", "Risk on"])
    ax_regime.set_ylabel("Regime")
    ax_regime.grid(alpha=0.15)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS_CSV)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--export-csv", type=Path, default=DEFAULT_EXPORT_CSV)
    args = parser.parse_args()

    signals = pd.read_csv(args.signals_csv, parse_dates=["date"]).set_index("date").sort_index()
    dataset = pd.read_csv(args.dataset_csv, parse_dates=["date"]).set_index("date").sort_index()

    consensus = build_consensus_signal(signals)
    export = signals[["gmm_signal_lag1", "logistic_signal_lag1"]].copy()
    export["consensus_signal_lag1"] = consensus
    args.export_csv.parent.mkdir(parents=True, exist_ok=True)
    export.reset_index().to_csv(args.export_csv, index=False)

    plot_consensus_chart(dataset["qqq_close"], consensus, args.out_path)
    print(f"Wrote {args.out_path}")
    print(f"Wrote {args.export_csv}")


if __name__ == "__main__":
    main()
