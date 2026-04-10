#!/usr/bin/env python3
"""
Train the SPY-gated QQQ 2x DCA model and save fitted models to models.pkl.

This script runs the full walk-forward pipeline exactly once using locally
cached data, saves the final-month logistic + random-forest models to a .pkl
file, and prints today's signal.

After running this script, copy models.pkl into the Docker container's /data
volume.  The container will then use fast inference (< 2 min) on daily runs
instead of the full ~15-min walk-forward retraining.

Usage (from the standalone project root):
    python train_and_save_models.py

Options:
    --qqq-parquet PATH   QQQ daily parquet (default: auto-detect from cache)
    --spy-parquet PATH   SPY daily parquet (default: auto-detect from cache)
    --macro-parquet PATH macro parquet    (default: auto-detect from cache)
    --fred-dir PATH      FRED stress CSV dir (default: cache/fred)
    --output PATH        where to save models.pkl (default: ./models.pkl)
    --data-root PATH     repo root override (parent of cache/)

Requirements:
    pip install scikit-learn pandas pyarrow joblib requests numpy scipy
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------ #
# Path setup — make sure tools/ is importable
# ------------------------------------------------------------------ #
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR / "tools"
CACHE_SCRIPTS_DIR = SCRIPT_DIR / "cache_scripts"
for _p in [str(TOOLS_DIR), str(CACHE_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import joblib
    import numpy as np
    import pandas as pd
    import qqq_macro_ml_regime_analysis as ra
    import qqq_macro_walkforward_model_compare as mc
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("        Run:  pip install scikit-learn pandas pyarrow joblib requests numpy scipy")
    sys.exit(1)

# ------------------------------------------------------------------ #
# Default data paths (mirrors pipeline.py / repo layout)
# ------------------------------------------------------------------ #
# Repo root is 1 level up from standalone folder by default; override
# with --data-root if the data lives elsewhere.
_REPO_ROOT = SCRIPT_DIR.parent
_CACHE_CACHE = _REPO_ROOT / "cache" / "cache"       # QQQ/SPY live here in the repo
_CACHE_MACRO = _REPO_ROOT / "cache" / "cache"       # macro_daily_1999.parquet
_FRED_DIR    = _REPO_ROOT / "cache" / "fred"

DEFAULT_QQQ_PARQUET   = _CACHE_CACHE / "QQQ_daily.parquet"
DEFAULT_SPY_PARQUET   = _CACHE_CACHE / "SPY_daily.parquet"
DEFAULT_MACRO_PARQUET = _CACHE_MACRO / "macro_daily_1999.parquet"

# ------------------------------------------------------------------ #
# Hyper-parameters (identical to container pipeline.py)
# ------------------------------------------------------------------ #
RISK_OFF_THRESHOLD  = 0.45
JUMP_IN_THRESHOLD   = 0.55
MIN_TRAIN_MONTHS    = 96
RF_ESTIMATORS       = 300
TARGET_HORIZON      = 63
PRICE_START         = "1999-03-10"
MONTHLY_LAG_DAYS    = 45
QUARTERLY_LAG_DAYS  = 45
ANNUAL_LAG_DAYS     = 365


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _check_path(p: Path, label: str) -> None:
    if not p.exists():
        print(f"[ERROR] {label} not found: {p}")
        print("        Run the download scripts first, or pass --{label.lower().replace(' ', '-')}")
        sys.exit(1)


def _banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qqq-parquet",   type=Path, default=DEFAULT_QQQ_PARQUET)
    parser.add_argument("--spy-parquet",   type=Path, default=DEFAULT_SPY_PARQUET)
    parser.add_argument("--macro-parquet", type=Path, default=DEFAULT_MACRO_PARQUET)
    parser.add_argument("--fred-dir",      type=Path, default=_FRED_DIR)
    parser.add_argument("--output",        type=Path, default=SCRIPT_DIR / "models.pkl")
    args = parser.parse_args()

    for path, label in [
        (args.qqq_parquet,   "QQQ parquet"),
        (args.spy_parquet,   "SPY parquet"),
        (args.macro_parquet, "macro parquet"),
        (args.fred_dir,      "FRED dir"),
    ]:
        _check_path(path, label)

    print(f"QQQ  : {args.qqq_parquet}")
    print(f"SPY  : {args.spy_parquet}")
    print(f"macro: {args.macro_parquet}")
    print(f"FRED : {args.fred_dir}")
    print(f"out  : {args.output}")

    # ---------------------------------------------------------------- #
    # Load price data
    # ---------------------------------------------------------------- #
    _banner("Loading price data …")
    spy_data = ra.load_qqq(args.spy_parquet, start=PRICE_START, end=None)
    qqq_data = ra.load_qqq(args.qqq_parquet, start=PRICE_START, end=None)
    print(f"  SPY : {len(spy_data)} trading days  ({spy_data.index[0].date()} … {spy_data.index[-1].date()})")
    print(f"  QQQ : {len(qqq_data)} trading days  ({qqq_data.index[0].date()} … {qqq_data.index[-1].date()})")

    # ---------------------------------------------------------------- #
    # Load macro + stress proxies
    # ---------------------------------------------------------------- #
    _banner("Loading macro & FRED stress data …")
    spy_macro = ra.load_macro(
        args.macro_parquet, spy_data.index,
        monthly_release_lag_days=MONTHLY_LAG_DAYS,
        quarterly_release_lag_days=QUARTERLY_LAG_DAYS,
        annual_release_lag_days=ANNUAL_LAG_DAYS,
    )
    qqq_macro = ra.load_macro(
        args.macro_parquet, qqq_data.index,
        monthly_release_lag_days=MONTHLY_LAG_DAYS,
        quarterly_release_lag_days=QUARTERLY_LAG_DAYS,
        annual_release_lag_days=ANNUAL_LAG_DAYS,
    )

    spy_stress, spy_stress_status = ra.load_stress_proxies(
        spy_data.index, args.fred_dir, refresh=False
    )
    qqq_stress, qqq_stress_status = ra.load_stress_proxies(
        qqq_data.index, args.fred_dir, refresh=False
    )
    for series, status in {**spy_stress_status, **qqq_stress_status}.items():
        if status != "loaded":
            print(f"  WARNING: FRED series '{series}': {status}")

    # ---------------------------------------------------------------- #
    # Feature engineering + month-end sampling
    # ---------------------------------------------------------------- #
    _banner("Building feature datasets …")
    spy_dataset = ra.build_dataset(spy_data, spy_macro, spy_stress, target_horizon=TARGET_HORIZON)
    qqq_dataset = ra.build_dataset(qqq_data, qqq_macro, qqq_stress, target_horizon=TARGET_HORIZON)

    spy_monthly = ra.month_end_sample(spy_dataset)
    qqq_monthly = ra.month_end_sample(qqq_dataset)
    print(f"  SPY monthly sample : {len(spy_monthly)} rows  (first valid signal after ~{MIN_TRAIN_MONTHS} months)")
    print(f"  QQQ monthly sample : {len(qqq_monthly)} rows")

    # ---------------------------------------------------------------- #
    # Fit final-month models (no full walk-forward loop)
    # ---------------------------------------------------------------- #
    _banner("Fitting SPY final-month models (logistic + random forest) …")
    spy_models = mc.fit_final_models(
        spy_monthly, ra.MODEL_FEATURES,
        target_horizon=TARGET_HORIZON,
        min_train_months=MIN_TRAIN_MONTHS,
        risk_off_threshold=RISK_OFF_THRESHOLD,
        jump_in_threshold=JUMP_IN_THRESHOLD,
        random_state=ra.RANDOM_STATE,
        rf_estimators=RF_ESTIMATORS,
    )
    print(f"  trained_through : {spy_models['trained_through']}")
    print(f"  train_n         : {spy_models['train_n']} month-end rows")
    print(f"  regime          : {spy_models['regime']}")
    print(f"  risk_off_prob   : {spy_models['risk_off_prob']:.3f}")
    print(f"  jump_in_prob    : {spy_models['jump_in_prob']:.3f}")

    _banner("Fitting QQQ final-month models (logistic + random forest) …")
    qqq_models = mc.fit_final_models(
        qqq_monthly, ra.MODEL_FEATURES,
        target_horizon=TARGET_HORIZON,
        min_train_months=MIN_TRAIN_MONTHS,
        risk_off_threshold=RISK_OFF_THRESHOLD,
        jump_in_threshold=JUMP_IN_THRESHOLD,
        random_state=ra.RANDOM_STATE,
        rf_estimators=RF_ESTIMATORS,
    )
    print(f"  trained_through : {qqq_models['trained_through']}")
    print(f"  train_n         : {qqq_models['train_n']} month-end rows")
    print(f"  regime          : {qqq_models['regime']}")
    print(f"  risk_off_prob   : {qqq_models['risk_off_prob']:.3f}")
    print(f"  jump_in_prob    : {qqq_models['jump_in_prob']:.3f}")

    # ---------------------------------------------------------------- #
    # Apply SPY-gate policy
    # ---------------------------------------------------------------- #
    spy_regime = spy_models["regime"]
    qqq_regime = qqq_models["regime"]
    if spy_regime == "risk_off":
        policy_signal = "risk_off"
    elif spy_regime == "risk_on" and qqq_regime != "risk_off":
        policy_signal = "risk_on"
    else:
        policy_signal = "neutral"

    _banner("TODAY'S SIGNAL")
    notice_map = {
        "risk_on":  "ENTER  — deploy at 2× leverage (QLD / 2× QQQ)",
        "neutral":  "HOLD   — 1× unlevered QQQ, no new leverage",
        "risk_off": "REDUCE — park new contributions, reduce 2× QQQ exposure",
    }
    print(f"  policy_signal : {policy_signal}")
    print(f"  notice        : {notice_map[policy_signal]}")
    print(f"  SPY gate      : {spy_regime}")
    print(f"  QQQ signal    : {qqq_regime}")

    # ---------------------------------------------------------------- #
    # Save pkl
    # ---------------------------------------------------------------- #
    _banner(f"Saving models to {args.output} …")
    payload = {
        "spy": spy_models,
        "qqq": qqq_models,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "spy_trained_through": spy_models["trained_through"],
        "qqq_trained_through": qqq_models["trained_through"],
        "policy_signal": policy_signal,
        "strategy": "spy_gate_qqq_ensemble_blend_2x",
        "risk_off_threshold": RISK_OFF_THRESHOLD,
        "jump_in_threshold": JUMP_IN_THRESHOLD,
        "target_horizon": TARGET_HORIZON,
        "min_train_months": MIN_TRAIN_MONTHS,
        "rf_estimators": RF_ESTIMATORS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    size_kb = args.output.stat().st_size // 1024
    print(f"  Saved  {args.output}  ({size_kb} KB)")
    print()
    print("  Copy models.pkl into the Docker volume to enable fast inference:")
    print("    docker cp models.pkl spy_qqq_signal_api:/data/models.pkl")
    print("    docker restart spy_qqq_signal_api")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
