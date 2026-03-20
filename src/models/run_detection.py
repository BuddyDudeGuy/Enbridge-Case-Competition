"""
Run CUSUM + EWMA anomaly detection across all 95 events.

For each event:
  1. Load the residual parquet (actual, predicted, residual per subsystem)
  2. Compute reference_mean and reference_std from the TRAINING portion
  3. Apply CUSUM (k=0.5*sigma, h=5*sigma) and EWMA (span=144, L=2.0) over the full series
  4. Save detection results to data/processed/detections/farm_{x}/event_{id}.parquet
  5. Aggregate metrics into outputs/reports/detection_summary.json

Usage:
    py src/models/run_detection.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.anomaly_detection import detect_anomalies

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FARMS = ["A", "B", "C"]

# CUSUM parameters (in units of training std)
CUSUM_K_SIGMA = 0.5   # Allowance = 0.5 * training_std — tolerates half-sigma noise
CUSUM_H_SIGMA = 5.0   # Alarm when cumulative shift reaches 5 * training_std

# EWMA parameters
EWMA_SPAN = 144  # 24 hours of 10-min data
EWMA_L = 2.0     # 2-sigma control limits (on raw std, applied to smoothed signal)

# Sustained alarm filter
SUSTAINED_MIN_RUN = 6  # Minimum 6 consecutive alarm timesteps (1 hour)

# Farm-to-subsystem mapping (matches what's in residual parquets)
FARM_SUBSYSTEMS = {
    "A": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
    "B": ["gearbox", "generator_bearings", "transformer"],
    "C": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
}


def load_unified_events() -> pd.DataFrame:
    """Load the unified events table."""
    path = PROJECT_ROOT / "data" / "processed" / "unified_events.csv"
    return pd.read_csv(path)


def get_residual_path(farm_letter: str, event_id: int) -> Path:
    """Return the path to a residual parquet file."""
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "residuals"
        / f"farm_{farm_letter.lower()}"
        / f"event_{event_id}.parquet"
    )


def get_detection_path(farm_letter: str, event_id: int) -> Path:
    """Return the output path for a detection parquet file."""
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "detections"
        / f"farm_{farm_letter.lower()}"
        / f"event_{event_id}.parquet"
    )


def process_event(
    farm_letter: str,
    event_id: int,
    subsystems: list,
    event_info: dict,
) -> dict:
    """Run detection on a single event and save results.

    Parameters
    ----------
    farm_letter : str
        "A", "B", or "C".
    event_id : int
        Event ID.
    subsystems : list of str
        Subsystem names available for this farm.
    event_info : dict
        Row from unified_events with event metadata.

    Returns
    -------
    dict
        Per-event detection summary.
    """
    # Load residual parquet
    residual_path = get_residual_path(farm_letter, event_id)
    df = pd.read_parquet(residual_path)

    # Verify subsystems are present
    available_subsystems = [
        s for s in subsystems
        if f"{s}_residual" in df.columns
    ]

    # Run detection (CUSUM + EWMA)
    detection_df = detect_anomalies(
        df,
        subsystems=available_subsystems,
        method="both",
        cusum_k_sigma=CUSUM_K_SIGMA,
        cusum_h_sigma=CUSUM_H_SIGMA,
        ewma_span=EWMA_SPAN,
        ewma_L=EWMA_L,
        sustained_min_run=SUSTAINED_MIN_RUN,
    )

    # Save detection parquet
    out_path = get_detection_path(farm_letter, event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detection_df.to_parquet(out_path, index=False)

    # --- Build per-event summary ---
    pred_mask = detection_df["train_test"] == "prediction"
    pred_df = detection_df[pred_mask]
    n_pred = len(pred_df)

    label = event_info.get("event_label", "unknown")
    event_start = event_info.get("event_start", None)

    subsystem_summaries = {}
    any_alarm = False
    first_alarm_idx = None  # Index within prediction window

    for subsystem in available_subsystems:
        combined_col = f"{subsystem}_combined_alarm"
        cusum_col = f"{subsystem}_cusum_alarm"
        ewma_col = f"{subsystem}_ewma_alarm"

        sub_summary = {}

        if combined_col in pred_df.columns:
            combined_alarms = pred_df[combined_col].values
            n_combined = int(combined_alarms.sum())
            sub_summary["combined_alarm_count"] = n_combined

            if n_combined > 0:
                any_alarm = True
                # First alarm index in the prediction window
                alarm_indices = np.where(combined_alarms)[0]
                first_idx = int(alarm_indices[0])
                sub_summary["first_alarm_pred_idx"] = first_idx

                # Track overall first alarm
                if first_alarm_idx is None or first_idx < first_alarm_idx:
                    first_alarm_idx = first_idx
            else:
                sub_summary["first_alarm_pred_idx"] = None

        if cusum_col in pred_df.columns:
            sub_summary["cusum_alarm_count"] = int(pred_df[cusum_col].sum())

        if ewma_col in pred_df.columns:
            sub_summary["ewma_alarm_count"] = int(pred_df[ewma_col].sum())

        subsystem_summaries[subsystem] = sub_summary

    # First alarm timestamp
    first_alarm_timestamp = None
    if first_alarm_idx is not None and "time_stamp" in pred_df.columns:
        first_alarm_timestamp = str(pred_df["time_stamp"].iloc[first_alarm_idx])

    summary = {
        "event_id": int(event_id),
        "farm": farm_letter,
        "label": label,
        "event_start": str(event_start) if event_start else None,
        "n_prediction_rows": n_pred,
        "any_alarm": bool(any_alarm),
        "first_alarm_pred_idx": first_alarm_idx,
        "first_alarm_timestamp": first_alarm_timestamp,
        "subsystems": subsystem_summaries,
    }

    return summary


def compute_earliness(summary: dict) -> int | None:
    """Compute how many timesteps before the end of the prediction window the alarm fired.

    For anomaly events, earliness = n_prediction_rows - first_alarm_pred_idx.
    Higher = earlier detection = better.

    Returns None if no alarm or not an anomaly.
    """
    if summary["label"] != "anomaly" or not summary["any_alarm"]:
        return None
    if summary["first_alarm_pred_idx"] is None:
        return None
    return summary["n_prediction_rows"] - summary["first_alarm_pred_idx"]


def main():
    t_start = time.perf_counter()

    # Load event metadata
    events_df = load_unified_events()

    # Build lookup
    event_lookup = {}
    for _, row in events_df.iterrows():
        key = (row["farm"], row["event_id"])
        event_lookup[key] = row.to_dict()

    print(f"Loaded {len(events_df)} events from unified_events.csv")
    print(f"CUSUM params: k={CUSUM_K_SIGMA}*sigma, h={CUSUM_H_SIGMA}*sigma")
    print(f"EWMA params:  span={EWMA_SPAN}, L={EWMA_L}")
    print()

    # Process all events
    all_summaries = []
    total_events = len(events_df)
    processed = 0

    for farm_letter in FARMS:
        farm_events = events_df[events_df["farm"] == farm_letter]
        subsystems = FARM_SUBSYSTEMS[farm_letter]

        print(f"{'='*65}")
        print(f"  Farm {farm_letter} — {len(farm_events)} events, "
              f"{len(subsystems)} subsystems: {subsystems}")
        print(f"{'='*65}")

        for _, row in farm_events.iterrows():
            event_id = int(row["event_id"])
            label = row["event_label"]
            t_event = time.perf_counter()

            # Check residual file exists
            residual_path = get_residual_path(farm_letter, event_id)
            if not residual_path.exists():
                print(f"  SKIP: Farm {farm_letter} event {event_id} — "
                      f"residual file not found")
                continue

            event_info = event_lookup[(farm_letter, event_id)]
            summary = process_event(farm_letter, event_id, subsystems, event_info)
            all_summaries.append(summary)

            elapsed = time.perf_counter() - t_event
            processed += 1

            alarm_str = "ALARM" if summary["any_alarm"] else "clean"
            print(
                f"  [{processed:3d}/{total_events}] Event {event_id:3d} "
                f"({label:>7s}) | {alarm_str:>5s} | "
                f"{summary['n_prediction_rows']:5d} pred rows | "
                f"{elapsed:.2f}s"
            )

    # --- Save detection summary ---
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "detection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nDetection summary saved: {summary_path}")

    # --- Compute and print aggregate metrics ---
    print_aggregate_metrics(all_summaries)

    total_time = time.perf_counter() - t_start
    print(f"\nTotal detection time: {total_time:.1f}s")


def print_aggregate_metrics(summaries: list[dict]):
    """Print detection rate, false alarm rate, and earliness metrics."""

    print(f"\n{'='*65}")
    print("  DETECTION SUMMARY")
    print(f"{'='*65}")

    # Separate anomaly vs normal events
    anomaly_events = [s for s in summaries if s["label"] == "anomaly"]
    normal_events = [s for s in summaries if s["label"] == "normal"]

    n_anomaly = len(anomaly_events)
    n_normal = len(normal_events)

    # Detection rate: fraction of anomaly events with at least one alarm
    detected = [s for s in anomaly_events if s["any_alarm"]]
    n_detected = len(detected)
    detection_rate = n_detected / n_anomaly if n_anomaly > 0 else 0.0

    print(f"\n  Anomaly events:     {n_anomaly}")
    print(f"  Detected (alarm):   {n_detected}")
    print(f"  Detection rate:     {detection_rate:.1%}")

    # False alarm rate: fraction of normal events with an alarm
    false_alarmed = [s for s in normal_events if s["any_alarm"]]
    n_false = len(false_alarmed)
    false_alarm_rate = n_false / n_normal if n_normal > 0 else 0.0

    print(f"\n  Normal events:      {n_normal}")
    print(f"  False alarms:       {n_false}")
    print(f"  False alarm rate:   {false_alarm_rate:.1%}")

    # Earliness: for detected anomalies, how early was the first alarm?
    earliness_values = []
    for s in detected:
        e = compute_earliness(s)
        if e is not None:
            earliness_values.append(e)

    if earliness_values:
        avg_earliness = np.mean(earliness_values)
        median_earliness = np.median(earliness_values)
        min_earliness = np.min(earliness_values)
        max_earliness = np.max(earliness_values)

        # Convert to hours (10-min intervals)
        avg_hours = avg_earliness * 10 / 60
        median_hours = median_earliness * 10 / 60

        print(f"\n  Earliness (detected anomalies):")
        print(f"    Avg timesteps before end:    {avg_earliness:.0f} "
              f"({avg_hours:.1f} hours)")
        print(f"    Median timesteps before end: {median_earliness:.0f} "
              f"({median_hours:.1f} hours)")
        print(f"    Range: {min_earliness} — {max_earliness} timesteps")
    else:
        print(f"\n  Earliness: N/A (no anomalies detected)")

    # Per-subsystem breakdown
    print(f"\n  Per-subsystem detection breakdown (anomaly events):")
    print(f"  {'Subsystem':<25} {'CUSUM':>6} {'EWMA':>6} {'Combined':>9}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*9}")

    # Collect all subsystems
    all_subsystems = set()
    for s in summaries:
        all_subsystems.update(s["subsystems"].keys())

    for subsystem in sorted(all_subsystems):
        cusum_detected = 0
        ewma_detected = 0
        combined_detected = 0

        for s in anomaly_events:
            if subsystem in s["subsystems"]:
                sub = s["subsystems"][subsystem]
                if sub.get("cusum_alarm_count", 0) > 0:
                    cusum_detected += 1
                if sub.get("ewma_alarm_count", 0) > 0:
                    ewma_detected += 1
                if sub.get("combined_alarm_count", 0) > 0:
                    combined_detected += 1

        # Count how many anomaly events have this subsystem
        n_with_subsystem = sum(
            1 for s in anomaly_events if subsystem in s["subsystems"]
        )

        print(f"  {subsystem:<25} {cusum_detected:>3}/{n_with_subsystem:<2} "
              f"{ewma_detected:>3}/{n_with_subsystem:<2} "
              f"{combined_detected:>3}/{n_with_subsystem:<2}")

    # Missed anomalies
    missed = [s for s in anomaly_events if not s["any_alarm"]]
    if missed:
        print(f"\n  Missed anomaly events ({len(missed)}):")
        for s in missed:
            print(f"    Farm {s['farm']} event {s['event_id']}: "
                  f"{s['n_prediction_rows']} pred rows")

    # False alarm details
    if false_alarmed:
        print(f"\n  False alarm events ({len(false_alarmed)}):")
        for s in false_alarmed:
            alarm_subsystems = [
                sub for sub, info in s["subsystems"].items()
                if info.get("combined_alarm_count", 0) > 0
            ]
            print(f"    Farm {s['farm']} event {s['event_id']}: "
                  f"alarms in {alarm_subsystems}")


if __name__ == "__main__":
    main()
