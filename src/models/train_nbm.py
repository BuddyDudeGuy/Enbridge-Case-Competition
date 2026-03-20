"""
Train Normal Behavior Models (NBMs) for a given wind farm.

For each target sensor defined in nbm_config, trains a LightGBM regressor
that predicts expected temperature from operating conditions. Saves trained
models (joblib) and a JSON training report with per-model metrics.

Usage:
    py src/models/train_nbm.py --farm A
    py src/models/train_nbm.py --farm B
    py src/models/train_nbm.py --farm C
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_farm_training_data
from src.models.nbm_config import get_nbm_config, get_target_info

# ---------------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def train_nbm_for_farm(farm_letter: str) -> dict:
    """Train all NBMs for a single farm and return the training report.

    Parameters
    ----------
    farm_letter : str
        Single letter: "A", "B", or "C".

    Returns
    -------
    dict
        Training report with per-model metrics.
    """
    farm_letter = farm_letter.upper()
    farm_name = f"Wind Farm {farm_letter}"

    print(f"\n{'='*60}")
    print(f"  Training NBMs for {farm_name}")
    print(f"{'='*60}\n")

    # --- Load config ---
    config = get_nbm_config(farm_name)
    input_features = config["inputs"]
    targets = config["targets"]  # {subsystem: column_name}

    print(f"Input features ({len(input_features)}): {input_features}")
    print(f"Target subsystems ({len(targets)}): {list(targets.keys())}\n")

    # --- Load training data ---
    print("Loading training data...")
    t0 = time.perf_counter()
    df = load_farm_training_data(farm_letter)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns in {load_time:.1f}s\n")

    # --- Output directories ---
    model_dir = PROJECT_ROOT / "data" / "processed" / "models" / f"farm_{farm_letter.lower()}"
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- Train one model per target ---
    report = {
        "farm": farm_name,
        "farm_letter": farm_letter,
        "input_features": input_features,
        "lgbm_params": LGBM_PARAMS,
        "models": [],
    }

    for subsystem, target_col in targets.items():
        print(f"--- {subsystem} ({target_col}) ---")

        # Get target metadata
        target_info = get_target_info(farm_name, subsystem)

        # Extract X and y, drop NaN rows
        cols_needed = input_features + [target_col]
        df_sub = df[cols_needed].dropna()
        X = df_sub[input_features]
        y = df_sub[target_col]

        n_dropped = len(df) - len(df_sub)
        print(f"  Training rows: {len(df_sub):,} (dropped {n_dropped:,} NaN rows)")

        # Train
        model = LGBMRegressor(**LGBM_PARAMS)
        t_start = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - t_start

        # Compute training metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        print(f"  MAE:  {mae:.4f} °C")
        print(f"  RMSE: {rmse:.4f} °C")
        print(f"  R²:   {r2:.6f}")
        print(f"  Time: {train_time:.1f}s")

        # Save model
        model_path = model_dir / f"{subsystem}_nbm.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}\n")

        # Append to report
        report["models"].append({
            "subsystem": subsystem,
            "target_sensor": target_col,
            "target_description": target_info["description"],
            "n_training_rows": len(df_sub),
            "n_dropped_nan": n_dropped,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "training_time_seconds": round(train_time, 2),
        })

    # --- Save report ---
    report_path = report_dir / f"nbm_training_farm_{farm_letter.lower()}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved: {report_path}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"  Summary — {farm_name}")
    print(f"{'='*60}")
    print(f"{'Subsystem':<25} {'MAE':>8} {'RMSE':>8} {'R²':>10} {'Rows':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for m in report["models"]:
        print(
            f"{m['subsystem']:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
            f"{m['r2']:>10.6f} {m['n_training_rows']:>10,}"
        )
    print()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train Normal Behavior Models (LightGBM) for a wind farm."
    )
    parser.add_argument(
        "--farm",
        type=str,
        required=True,
        choices=["A", "B", "C", "a", "b", "c"],
        help="Farm letter: A, B, or C",
    )
    args = parser.parse_args()

    train_nbm_for_farm(args.farm)


if __name__ == "__main__":
    main()
