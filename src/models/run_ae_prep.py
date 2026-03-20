"""
Run autoencoder data preparation for all three farms.

Usage:
    py src/models/run_ae_prep.py

Fits scalers on normal-operation training data, creates sliding-window
sequences, and saves everything for downstream LSTM-Autoencoder training.
"""

import json
import time
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.autoencoder_data import (
    AUTOENCODER_SENSORS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
    prepare_training_data,
)


def main():
    print("=" * 60)
    print("AUTOENCODER DATA PREPARATION")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Window size:  {DEFAULT_WINDOW_SIZE} steps (6 hours)")
    print(f"Step size:    {DEFAULT_STEP_SIZE} steps (1 hour)")
    print()

    summary = {
        "window_size": DEFAULT_WINDOW_SIZE,
        "step_size": DEFAULT_STEP_SIZE,
        "farms": {},
    }

    total_start = time.time()

    for farm in ["a", "b", "c"]:
        farm_start = time.time()
        sensor_cols = AUTOENCODER_SENSORS[f"farm_{farm}"]

        X, scaler = prepare_training_data(farm, PROJECT_ROOT)

        farm_elapsed = time.time() - farm_start

        farm_summary = {
            "n_sensors": len(sensor_cols),
            "sensors": sensor_cols,
            "n_sequences": int(X.shape[0]),
            "sequence_shape": list(X.shape),
            "dtype": str(X.dtype),
            "elapsed_sec": round(farm_elapsed, 1),
        }
        summary["farms"][f"farm_{farm}"] = farm_summary

        print(f"  Time: {farm_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    summary["total_elapsed_sec"] = round(total_elapsed, 1)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for farm_key, info in summary["farms"].items():
        print(f"  {farm_key.upper()}: {info['n_sequences']:>7,} sequences  "
              f"| shape {info['sequence_shape']}  "
              f"| {info['n_sensors']} sensors  "
              f"| {info['elapsed_sec']:.1f}s")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Save summary JSON
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ae_data_prep_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {report_path}")


if __name__ == "__main__":
    main()
