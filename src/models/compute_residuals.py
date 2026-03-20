"""
Compute temperature residuals (actual - predicted) for all events across all farms.

For each event, loads the full dataset, runs each trained NBM to predict expected
temperature, and computes residuals. Saves per-event parquet files and a summary JSON.

Usage:
    py src/models/compute_residuals.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_event, get_event_ids, load_event_info, clear_cache
from src.models.nbm_config import get_nbm_config, NBM_TARGETS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FARMS = ["A", "B", "C"]


def load_models(farm_letter: str) -> dict:
    """Load all trained NBM models for a farm.

    Parameters
    ----------
    farm_letter : str
        "A", "B", or "C".

    Returns
    -------
    dict
        {subsystem: fitted LGBMRegressor}
    """
    model_dir = PROJECT_ROOT / "data" / "processed" / "models" / f"farm_{farm_letter.lower()}"
    farm_name = f"Wind Farm {farm_letter.upper()}"
    config = get_nbm_config(farm_name)
    subsystems = list(config["targets"].keys())

    models = {}
    for subsystem in subsystems:
        model_path = model_dir / f"{subsystem}_nbm.joblib"
        if not model_path.exists():
            print(f"  WARNING: model not found: {model_path}, skipping {subsystem}")
            continue
        models[subsystem] = joblib.load(model_path)
    return models


def compute_residuals_for_event(
    farm_letter: str,
    event_id: int,
    models: dict,
    config: dict,
) -> pd.DataFrame:
    """Compute residuals for a single event across all subsystem models.

    Parameters
    ----------
    farm_letter : str
        "A", "B", or "C".
    event_id : int
        Event ID.
    models : dict
        {subsystem: fitted LGBMRegressor}
    config : dict
        NBM config from get_nbm_config().

    Returns
    -------
    pd.DataFrame
        Original event data plus {subsystem}_actual, _predicted, _residual columns.
    """
    # Load full event (both train + prediction portions)
    df = load_event(farm_letter, event_id, cache=False)

    input_features = config["inputs"]
    targets = config["targets"]  # {subsystem: target_column}

    # Build all new columns in a dict first, then concat once (avoids fragmentation)
    new_cols = {}

    for subsystem, model in models.items():
        target_col = targets[subsystem]

        # Actual values
        if target_col in df.columns:
            actual = df[target_col].values
            new_cols[f"{subsystem}_actual"] = actual
        else:
            # Target column missing from this event's data
            new_cols[f"{subsystem}_actual"] = np.full(len(df), np.nan)
            new_cols[f"{subsystem}_predicted"] = np.full(len(df), np.nan)
            new_cols[f"{subsystem}_residual"] = np.full(len(df), np.nan)
            continue

        # Build input matrix — use NaN for any missing input feature columns
        X = pd.DataFrame(
            {feat: df[feat].values if feat in df.columns else np.nan
             for feat in input_features},
            index=df.index,
        )

        # Predict — LightGBM handles NaN inputs natively
        predicted = model.predict(X)
        new_cols[f"{subsystem}_predicted"] = predicted
        new_cols[f"{subsystem}_residual"] = actual - predicted

    # Attach all residual columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def compute_prediction_summary(df: pd.DataFrame, subsystems: list[str]) -> dict:
    """Compute summary statistics on residuals in the prediction window only.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame with residual columns.
    subsystems : list[str]
        List of subsystem names.

    Returns
    -------
    dict
        {subsystem: {mean_residual, max_residual, std_residual, n_rows}}
    """
    pred_mask = df["train_test"] == "prediction"
    df_pred = df.loc[pred_mask]
    n_pred = len(df_pred)

    summary = {}
    for subsystem in subsystems:
        residual_col = f"{subsystem}_residual"
        if residual_col not in df_pred.columns:
            continue

        residuals = df_pred[residual_col].dropna()
        if len(residuals) == 0:
            summary[subsystem] = {
                "mean_residual": None,
                "max_residual": None,
                "std_residual": None,
                "n_rows": 0,
            }
        else:
            summary[subsystem] = {
                "mean_residual": round(float(residuals.mean()), 6),
                "max_residual": round(float(residuals.max()), 6),
                "std_residual": round(float(residuals.std()), 6),
                "n_rows": int(len(residuals)),
            }

    return summary


def select_output_columns(df: pd.DataFrame, subsystems: list[str]) -> list[str]:
    """Select columns to keep in the output parquet.

    Keeps: time_stamp, train_test, status_type_id, plus all
    {subsystem}_actual, _predicted, _residual columns.
    """
    base_cols = []
    for col in ["time_stamp", "train_test", "status_type_id"]:
        if col in df.columns:
            base_cols.append(col)

    residual_cols = []
    for subsystem in subsystems:
        for suffix in ["_actual", "_predicted", "_residual"]:
            col = f"{subsystem}{suffix}"
            if col in df.columns:
                residual_cols.append(col)

    return base_cols + residual_cols


def main():
    t_global = time.perf_counter()

    # Load event info for label lookup
    events_info = load_event_info()

    # Build a lookup: (farm, event_id) -> event_label
    label_lookup = {}
    for _, row in events_info.iterrows():
        label_lookup[(row["farm"], row["event_id"])] = row["event_label"]

    # Output dirs
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Master summary dict: {farm: {event_id: {subsystem: stats}}}
    all_summaries = {}

    total_events = 0
    for farm_letter in FARMS:
        event_ids = get_event_ids(farm_letter)
        total_events += len(event_ids)
    print(f"Total events to process: {total_events}\n")

    processed = 0
    for farm_letter in FARMS:
        farm_name = f"Wind Farm {farm_letter}"
        print(f"\n{'='*60}")
        print(f"  Processing {farm_name}")
        print(f"{'='*60}")

        # Load models
        print("  Loading models...")
        models = load_models(farm_letter)
        config = get_nbm_config(farm_name)
        subsystems = list(config["targets"].keys())
        print(f"  Loaded {len(models)} models: {list(models.keys())}")

        # Get all event IDs
        event_ids = get_event_ids(farm_letter)
        print(f"  Events to process: {len(event_ids)}\n")

        # Output dir for this farm's residuals
        residual_dir = PROJECT_ROOT / "data" / "processed" / "residuals" / f"farm_{farm_letter.lower()}"
        residual_dir.mkdir(parents=True, exist_ok=True)

        farm_summaries = {}

        for i, event_id in enumerate(event_ids):
            t_event = time.perf_counter()
            label = label_lookup.get((farm_letter, event_id), "unknown")

            # Compute residuals
            df_residuals = compute_residuals_for_event(
                farm_letter, event_id, models, config
            )

            # Compute prediction-window summary
            summary = compute_prediction_summary(df_residuals, subsystems)
            farm_summaries[str(event_id)] = summary

            # Select columns and save parquet
            output_cols = select_output_columns(df_residuals, subsystems)
            df_out = df_residuals[output_cols]
            out_path = residual_dir / f"event_{event_id}.parquet"
            df_out.to_parquet(out_path, index=False)

            elapsed = time.perf_counter() - t_event
            processed += 1

            # Clear cache to save memory
            clear_cache()

            print(
                f"  [{processed:3d}/{total_events}] Farm {farm_letter} | "
                f"Event {event_id:3d} ({label:>7s}) | "
                f"{len(df_residuals):6,} rows | "
                f"{elapsed:.1f}s"
            )

        all_summaries[farm_letter] = farm_summaries

    # Save summary JSON
    summary_path = report_dir / "residuals_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    total_time = time.perf_counter() - t_global
    print(f"\nTotal processing time: {total_time:.1f}s")

    # -----------------------------------------------------------------------
    # Print summary table: normal vs anomaly average residuals
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  RESIDUAL SUMMARY: Normal vs Anomaly (prediction window)")
    print(f"{'='*70}")

    for farm_letter in FARMS:
        farm_name = f"Wind Farm {farm_letter}"
        config = get_nbm_config(farm_name)
        subsystems = list(config["targets"].keys())

        # Collect mean residuals for normal vs anomaly events
        normal_residuals = {s: [] for s in subsystems}
        anomaly_residuals = {s: [] for s in subsystems}

        farm_summaries = all_summaries[farm_letter]
        for eid_str, sub_summary in farm_summaries.items():
            eid = int(eid_str)
            label = label_lookup.get((farm_letter, eid), "unknown")
            for subsystem in subsystems:
                if subsystem in sub_summary and sub_summary[subsystem]["mean_residual"] is not None:
                    val = sub_summary[subsystem]["mean_residual"]
                    if label == "normal":
                        normal_residuals[subsystem].append(val)
                    elif label == "anomaly":
                        anomaly_residuals[subsystem].append(val)

        print(f"\n  {farm_name}")
        print(f"  {'Subsystem':<25} {'Normal Mean':>12} {'Anomaly Mean':>13} {'Normal Std':>11} {'Anomaly Std':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*13} {'-'*11} {'-'*12}")

        for subsystem in subsystems:
            n_vals = normal_residuals[subsystem]
            a_vals = anomaly_residuals[subsystem]

            n_mean = f"{np.mean(n_vals):.4f}" if n_vals else "N/A"
            a_mean = f"{np.mean(a_vals):.4f}" if a_vals else "N/A"
            n_std = f"{np.std(n_vals):.4f}" if n_vals else "N/A"
            a_std = f"{np.std(a_vals):.4f}" if a_vals else "N/A"

            print(f"  {subsystem:<25} {n_mean:>12} {a_mean:>13} {n_std:>11} {a_std:>12}")

    print()


if __name__ == "__main__":
    main()
