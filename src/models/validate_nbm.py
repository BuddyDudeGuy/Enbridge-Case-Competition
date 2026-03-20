"""
Validate trained NBM models on held-out prediction windows of normal events.

Models were trained on the TRAINING portion of normal events. This script
evaluates them on the PREDICTION portion to check generalization. If a model
generalises well, residuals on unseen normal data should remain small
(MAE < 2 C, R^2 > 0.85).

Usage:
    py src/models/validate_nbm.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_event_prediction, get_event_ids
from src.models.nbm_config import get_nbm_config, get_target_info

# Minimum normal-operation rows required in a prediction window to include
# that event in validation metrics.
MIN_NORMAL_ROWS = 10

# Targets for acceptable generalisation
TARGET_MAE = 2.0       # degrees C
TARGET_R2 = 0.85

FARMS = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(farm_letter: str, subsystem: str):
    """Load a trained NBM model from disk."""
    model_path = (
        PROJECT_ROOT / "data" / "processed" / "models"
        / f"farm_{farm_letter.lower()}" / f"{subsystem}_nbm.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib_load(model_path)


def validate_farm(farm_letter: str) -> dict:
    """Validate all NBMs for a single farm on prediction windows of normal events.

    Returns
    -------
    dict
        Per-model validation metrics aggregated across normal events.
    """
    farm_letter = farm_letter.upper()
    farm_name = f"Wind Farm {farm_letter}"

    config = get_nbm_config(farm_name)
    input_features = config["inputs"]
    targets = config["targets"]  # {subsystem: column_name}

    normal_ids = get_event_ids(farm_letter, label="normal")

    print(f"\n{'='*70}")
    print(f"  Validating NBMs for {farm_name}")
    print(f"  Normal events: {len(normal_ids)}  |  Models: {len(targets)}")
    print(f"{'='*70}")

    farm_results = {
        "farm": farm_name,
        "farm_letter": farm_letter,
        "n_normal_events": len(normal_ids),
        "models": [],
    }

    for subsystem, target_col in targets.items():
        target_info = get_target_info(farm_name, subsystem)
        model = _load_model(farm_letter, subsystem)

        print(f"\n--- {subsystem} ({target_col}: {target_info['description']}) ---")

        event_maes = []
        event_rmses = []
        event_r2s = []
        events_used = 0
        events_skipped = 0

        for eid in normal_ids:
            # Load prediction window (all rows, including fault statuses)
            try:
                df_pred = load_event_prediction(farm_letter, eid)
            except FileNotFoundError:
                print(f"  Event {eid}: file not found, skipping")
                events_skipped += 1
                continue

            # Filter to normal operation rows only
            df_normal = df_pred[df_pred["status_type_id"].isin([0, 2])].copy()

            if len(df_normal) < MIN_NORMAL_ROWS:
                events_skipped += 1
                continue

            # Select relevant columns and drop NaN rows
            cols_needed = input_features + [target_col]
            missing_cols = [c for c in cols_needed if c not in df_normal.columns]
            if missing_cols:
                print(f"  Event {eid}: missing columns {missing_cols}, skipping")
                events_skipped += 1
                continue

            df_sub = df_normal[cols_needed].dropna()

            if len(df_sub) < MIN_NORMAL_ROWS:
                events_skipped += 1
                continue

            X = df_sub[input_features].values
            y_actual = df_sub[target_col].values

            # Predict
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = model.predict(X)

            # Metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)

            event_maes.append(mae)
            event_rmses.append(rmse)
            event_r2s.append(r2)
            events_used += 1

        # Aggregate across events
        if events_used == 0:
            print(f"  No usable events (all skipped). Cannot compute metrics.")
            farm_results["models"].append({
                "subsystem": subsystem,
                "target_sensor": target_col,
                "target_description": target_info["description"],
                "events_used": 0,
                "events_skipped": events_skipped,
                "avg_mae": None,
                "avg_rmse": None,
                "avg_r2": None,
                "min_r2": None,
                "max_mae": None,
                "flag": "NO_USABLE_EVENTS",
            })
            continue

        avg_mae = float(np.mean(event_maes))
        avg_rmse = float(np.mean(event_rmses))
        avg_r2 = float(np.mean(event_r2s))
        min_r2 = float(np.min(event_r2s))
        max_mae = float(np.max(event_maes))

        # Flag check
        flags = []
        if avg_mae > TARGET_MAE:
            flags.append(f"AVG_MAE>{TARGET_MAE}")
        if avg_r2 < TARGET_R2:
            flags.append(f"AVG_R2<{TARGET_R2}")
        flag_str = " | ".join(flags) if flags else "PASS"

        print(f"  Events used: {events_used}, skipped: {events_skipped}")
        print(f"  Avg MAE:  {avg_mae:.4f} C")
        print(f"  Avg RMSE: {avg_rmse:.4f} C")
        print(f"  Avg R2:   {avg_r2:.6f}")
        print(f"  Min R2:   {min_r2:.6f}")
        print(f"  Max MAE:  {max_mae:.4f} C")
        print(f"  Status:   {flag_str}")

        farm_results["models"].append({
            "subsystem": subsystem,
            "target_sensor": target_col,
            "target_description": target_info["description"],
            "events_used": events_used,
            "events_skipped": events_skipped,
            "avg_mae": round(avg_mae, 6),
            "avg_rmse": round(avg_rmse, 6),
            "avg_r2": round(avg_r2, 6),
            "min_r2": round(min_r2, 6),
            "max_mae": round(max_mae, 6),
            "flag": flag_str,
        })

    return farm_results


def print_summary_table(all_results: list[dict]):
    """Print a formatted summary table across all farms."""
    print(f"\n{'='*90}")
    print(f"  NBM VALIDATION SUMMARY — Prediction Windows of Normal Events")
    print(f"{'='*90}")
    print(
        f"{'Farm':<8} {'Subsystem':<22} {'Avg MAE':>9} {'Avg RMSE':>10} "
        f"{'Avg R2':>9} {'Min R2':>9} {'Events':>7} {'Status':<20}"
    )
    print(f"{'-'*8} {'-'*22} {'-'*9} {'-'*10} {'-'*9} {'-'*9} {'-'*7} {'-'*20}")

    for farm_res in all_results:
        farm_letter = farm_res["farm_letter"]
        for m in farm_res["models"]:
            if m["avg_mae"] is None:
                print(
                    f"{farm_letter:<8} {m['subsystem']:<22} {'N/A':>9} {'N/A':>10} "
                    f"{'N/A':>9} {'N/A':>9} {m['events_used']:>7} {m['flag']:<20}"
                )
            else:
                print(
                    f"{farm_letter:<8} {m['subsystem']:<22} {m['avg_mae']:>9.4f} "
                    f"{m['avg_rmse']:>10.4f} {m['avg_r2']:>9.4f} {m['min_r2']:>9.4f} "
                    f"{m['events_used']:>7} {m['flag']:<20}"
                )

    print(f"\nTargets: MAE < {TARGET_MAE} C, R2 > {TARGET_R2}")
    print(f"Min normal rows per event: {MIN_NORMAL_ROWS}")


def main():
    all_results = []

    for farm_letter in FARMS:
        result = validate_farm(farm_letter)
        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Save combined report
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "nbm_validation_results.json"

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {report_path}")

    # Count flags
    n_pass = sum(
        1 for r in all_results for m in r["models"] if m["flag"] == "PASS"
    )
    n_flag = sum(
        1 for r in all_results for m in r["models"] if m["flag"] != "PASS"
    )
    total = n_pass + n_flag
    print(f"\n{n_pass}/{total} models PASS targets, {n_flag}/{total} flagged.")


if __name__ == "__main__":
    main()
