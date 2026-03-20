"""
Phase 5.5 & 5.6: Run trained LSTM-Autoencoders on all 95 events.

For each event, produce:
  - Per-sequence reconstruction errors  (anomaly signal)
  - Per-sequence 32-dim bottleneck embeddings  (Similar Fault Finder)

Outputs are saved under  data/processed/ae_outputs/farm_{x}/
Summary statistics go to  outputs/reports/ae_reconstruction_summary.json
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_event, get_event_ids, load_event_info
from src.models.lstm_autoencoder import (
    LSTMAutoencoder,
    compute_reconstruction_error,
    extract_embeddings,
)
from src.models.autoencoder_data import (
    AUTOENCODER_SENSORS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
    create_sequences,
)

FARMS = ["a", "b", "c"]
WINDOW_SIZE = DEFAULT_WINDOW_SIZE   # 36
STEP_SIZE = DEFAULT_STEP_SIZE       # 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_scaler(farm: str):
    """Load a trained LSTM-AE and the pre-fitted scaler for *farm*."""
    farm = farm.lower()
    model_dir = PROJECT_ROOT / "data" / "processed" / "models" / f"farm_{farm}"

    # Config
    with open(model_dir / "lstm_ae_config.json") as f:
        config = json.load(f)

    # Model
    model = LSTMAutoencoder(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        bottleneck_size=config["bottleneck_size"],
        seq_len=config["seq_len"],
    )
    state_dict = torch.load(model_dir / "lstm_ae.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Scaler
    scaler = joblib.load(model_dir / "ae_scaler.joblib")

    return model, scaler, config


def process_event(farm: str, event_id: int, model, scaler, config, event_info_df):
    """Run AE inference on a single event.

    Returns a dict with summary statistics, or None if the event is skipped.
    """
    farm_letter = farm.upper()
    farm_key = f"farm_{farm.lower()}"
    sensor_cols = AUTOENCODER_SENSORS[farm_key]
    n_features = len(sensor_cols)

    # Output directory
    out_dir = PROJECT_ROOT / "data" / "processed" / "ae_outputs" / f"farm_{farm.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the full event
    try:
        df = load_event(farm_letter, event_id, cache=False)
    except FileNotFoundError:
        print(f"    WARNING: event {event_id} CSV not found, skipping")
        return None

    # Sort by time
    df = df.sort_values("time_stamp").reset_index(drop=True)

    # Check sensor availability and fill missing with NaN
    available = [c for c in sensor_cols if c in df.columns]
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan

    # NaN handling
    sensor_data = df[sensor_cols].copy()
    sensor_data = sensor_data.ffill().bfill().fillna(0.0)

    # Normalize with pre-fitted scaler (no leakage)
    sensor_data_scaled = pd.DataFrame(
        scaler.transform(sensor_data.values),
        columns=sensor_cols,
    )

    # -------------------------------------------------------------------
    # Create sequences & track train/prediction membership per row
    # -------------------------------------------------------------------
    data = sensor_data_scaled.values.astype(np.float32)
    n_rows = len(data)

    if n_rows < WINDOW_SIZE:
        print(f"    WARNING: event {event_id} has only {n_rows} rows (< {WINDOW_SIZE}), skipping")
        return None

    # Determine per-row train/prediction status
    train_test_col = df["train_test"].values  # array of "train" / "prediction"

    sequences = []
    meta_rows = []

    for i, start in enumerate(range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE)):
        end = start + WINDOW_SIZE
        sequences.append(data[start:end])

        # A sequence is "prediction" if any of its rows fall in the prediction window
        tt_slice = train_test_col[start:end]
        n_pred_rows = (tt_slice == "prediction").sum()
        portion = "prediction" if n_pred_rows > 0 else "train"

        meta_rows.append({
            "seq_idx": i,
            "start_row": int(start),
            "end_row": int(end - 1),
            "start_time": str(df["time_stamp"].iloc[start]),
            "end_time": str(df["time_stamp"].iloc[end - 1]),
            "portion": portion,
            "n_pred_rows_in_window": int(n_pred_rows),
        })

    X = np.array(sequences, dtype=np.float32)
    n_sequences = X.shape[0]

    # -------------------------------------------------------------------
    # Inference: reconstruction errors + embeddings
    # -------------------------------------------------------------------
    recon_errors = compute_reconstruction_error(model, X, batch_size=512, device="cpu")
    embeddings = extract_embeddings(model, X, batch_size=512, device="cpu")

    # Save numpy arrays
    np.save(out_dir / f"event_{event_id}_recon_error.npy", recon_errors.astype(np.float32))
    np.save(out_dir / f"event_{event_id}_embeddings.npy", embeddings.astype(np.float32))

    # -------------------------------------------------------------------
    # Metadata JSON
    # -------------------------------------------------------------------
    # Determine event label
    evt_row = event_info_df[
        (event_info_df["farm"] == farm_letter) & (event_info_df["event_id"] == event_id)
    ]
    event_label = evt_row["event_label"].iloc[0] if len(evt_row) > 0 else "unknown"
    event_desc = evt_row["event_description"].iloc[0] if len(evt_row) > 0 else ""
    if pd.isna(event_desc):
        event_desc = ""

    # Sequence indices for train vs prediction
    train_seq_idxs = [m["seq_idx"] for m in meta_rows if m["portion"] == "train"]
    pred_seq_idxs = [m["seq_idx"] for m in meta_rows if m["portion"] == "prediction"]

    meta = {
        "farm": farm_letter,
        "event_id": int(event_id),
        "event_label": event_label,
        "event_description": event_desc,
        "n_sequences": n_sequences,
        "n_features": n_features,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "n_rows": n_rows,
        "train_sequence_indices": train_seq_idxs,
        "prediction_sequence_indices": pred_seq_idxs,
        "n_train_sequences": len(train_seq_idxs),
        "n_prediction_sequences": len(pred_seq_idxs),
        "sequence_details": meta_rows,
    }

    with open(out_dir / f"event_{event_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------
    summary = {
        "farm": farm_letter,
        "event_id": int(event_id),
        "event_label": event_label,
        "event_description": event_desc,
        "n_sequences": n_sequences,
        "n_train_sequences": len(train_seq_idxs),
        "n_prediction_sequences": len(pred_seq_idxs),
        "overall_mean_error": float(recon_errors.mean()),
        "overall_max_error": float(recon_errors.max()),
        "overall_std_error": float(recon_errors.std()),
    }

    # Prediction-window stats (the part that matters for anomaly detection)
    if len(pred_seq_idxs) > 0:
        pred_errors = recon_errors[pred_seq_idxs]
        summary["pred_mean_error"] = float(pred_errors.mean())
        summary["pred_max_error"] = float(pred_errors.max())
        summary["pred_std_error"] = float(pred_errors.std())
    else:
        summary["pred_mean_error"] = None
        summary["pred_max_error"] = None
        summary["pred_std_error"] = None

    # Train-window stats (baseline)
    if len(train_seq_idxs) > 0:
        train_errors = recon_errors[train_seq_idxs]
        summary["train_mean_error"] = float(train_errors.mean())
        summary["train_max_error"] = float(train_errors.max())
        summary["train_std_error"] = float(train_errors.std())
    else:
        summary["train_mean_error"] = None
        summary["train_max_error"] = None
        summary["train_std_error"] = None

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("LSTM-Autoencoder Inference — Phase 5.5 & 5.6")
    print("=" * 70)

    event_info_df = load_event_info()
    all_summaries = []
    total_sequences = 0
    skipped = 0

    for farm in FARMS:
        farm_letter = farm.upper()
        print(f"\n{'-' * 60}")
        print(f"Farm {farm_letter}")
        print(f"{'-' * 60}")

        # Load model and scaler
        model, scaler, config = load_model_and_scaler(farm)
        print(f"  Model loaded: {config['n_features']} features, "
              f"hidden={config['hidden_size']}, bottleneck={config['bottleneck_size']}, "
              f"seq_len={config['seq_len']}")

        # Get all event IDs for this farm
        event_ids = get_event_ids(farm_letter)
        print(f"  Events to process: {len(event_ids)}")

        for idx, eid in enumerate(event_ids):
            label_row = event_info_df[
                (event_info_df["farm"] == farm_letter) &
                (event_info_df["event_id"] == eid)
            ]
            label = label_row["event_label"].iloc[0] if len(label_row) > 0 else "?"

            summary = process_event(farm, eid, model, scaler, config, event_info_df)

            if summary is None:
                skipped += 1
                continue

            all_summaries.append(summary)
            total_sequences += summary["n_sequences"]

            pred_err = summary["pred_mean_error"]
            pred_str = f"{pred_err:.6f}" if pred_err is not None else "n/a"
            print(f"  [{idx+1:2d}/{len(event_ids)}] event {eid:3d}  "
                  f"({label:7s})  seqs={summary['n_sequences']:5d}  "
                  f"pred_mean_err={pred_str}  "
                  f"max_err={summary['overall_max_error']:.6f}")

    # -------------------------------------------------------------------
    # Save reconstruction summary
    # -------------------------------------------------------------------
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "ae_reconstruction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute aggregate stats
    normal_summaries = [s for s in all_summaries if s["event_label"] == "normal"]
    anomaly_summaries = [s for s in all_summaries if s["event_label"] == "anomaly"]

    # Per-farm breakdown
    farm_stats = {}
    for farm in FARMS:
        fl = farm.upper()
        farm_normal = [s for s in normal_summaries if s["farm"] == fl]
        farm_anomaly = [s for s in anomaly_summaries if s["farm"] == fl]

        def safe_mean(lst, key):
            vals = [s[key] for s in lst if s[key] is not None]
            return float(np.mean(vals)) if vals else None

        farm_stats[fl] = {
            "n_normal": len(farm_normal),
            "n_anomaly": len(farm_anomaly),
            "normal_pred_mean_error": safe_mean(farm_normal, "pred_mean_error"),
            "anomaly_pred_mean_error": safe_mean(farm_anomaly, "pred_mean_error"),
            "normal_pred_max_error": safe_mean(farm_normal, "pred_max_error"),
            "anomaly_pred_max_error": safe_mean(farm_anomaly, "pred_max_error"),
        }

        # Separation ratio (anomaly / normal) -- higher is better
        if (farm_stats[fl]["normal_pred_mean_error"] is not None and
                farm_stats[fl]["anomaly_pred_mean_error"] is not None and
                farm_stats[fl]["normal_pred_mean_error"] > 0):
            farm_stats[fl]["separation_ratio_mean"] = (
                farm_stats[fl]["anomaly_pred_mean_error"] /
                farm_stats[fl]["normal_pred_mean_error"]
            )
        else:
            farm_stats[fl]["separation_ratio_mean"] = None

        if (farm_stats[fl]["normal_pred_max_error"] is not None and
                farm_stats[fl]["anomaly_pred_max_error"] is not None and
                farm_stats[fl]["normal_pred_max_error"] > 0):
            farm_stats[fl]["separation_ratio_max"] = (
                farm_stats[fl]["anomaly_pred_max_error"] /
                farm_stats[fl]["normal_pred_max_error"]
            )
        else:
            farm_stats[fl]["separation_ratio_max"] = None

    # Overall normal vs anomaly
    def safe_overall_mean(lst, key):
        vals = [s[key] for s in lst if s[key] is not None]
        return float(np.mean(vals)) if vals else None

    overall = {
        "total_events_processed": len(all_summaries),
        "total_events_skipped": skipped,
        "total_sequences": total_sequences,
        "n_normal_events": len(normal_summaries),
        "n_anomaly_events": len(anomaly_summaries),
        "normal_avg_pred_mean_error": safe_overall_mean(normal_summaries, "pred_mean_error"),
        "anomaly_avg_pred_mean_error": safe_overall_mean(anomaly_summaries, "pred_mean_error"),
        "normal_avg_pred_max_error": safe_overall_mean(normal_summaries, "pred_max_error"),
        "anomaly_avg_pred_max_error": safe_overall_mean(anomaly_summaries, "pred_max_error"),
    }

    output = {
        "overall": overall,
        "per_farm": farm_stats,
        "per_event": all_summaries,
    }

    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # -------------------------------------------------------------------
    # Print final report
    # -------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL REPORT")
    print(f"{'=' * 70}")
    print(f"Events processed:  {len(all_summaries)}")
    print(f"Events skipped:    {skipped}")
    print(f"Total sequences:   {total_sequences:,}")
    print(f"Time elapsed:      {elapsed:.1f}s")

    print(f"\n--- Reconstruction Error: Normal vs Anomaly ---")
    print(f"  Normal events  ({overall['n_normal_events']:2d}):  "
          f"avg pred mean error = {overall['normal_avg_pred_mean_error']:.6f}" if overall['normal_avg_pred_mean_error'] else "  Normal: n/a")
    print(f"  Anomaly events ({overall['n_anomaly_events']:2d}):  "
          f"avg pred mean error = {overall['anomaly_avg_pred_mean_error']:.6f}" if overall['anomaly_avg_pred_mean_error'] else "  Anomaly: n/a")

    if (overall['normal_avg_pred_mean_error'] and
            overall['anomaly_avg_pred_mean_error'] and
            overall['normal_avg_pred_mean_error'] > 0):
        ratio = overall['anomaly_avg_pred_mean_error'] / overall['normal_avg_pred_mean_error']
        print(f"  Overall separation ratio: {ratio:.2f}x")

    print(f"\n--- Per-Farm Separation ---")
    best_farm = None
    best_ratio = 0.0
    best_metric = None
    for fl, fs in farm_stats.items():
        print(f"  Farm {fl}: normal_pred_mean={fs['normal_pred_mean_error']:.6f}  "
              f"anomaly_pred_mean={fs['anomaly_pred_mean_error']:.6f}  "
              f"ratio_mean={fs['separation_ratio_mean']:.2f}x"
              if fs['separation_ratio_mean'] else f"  Farm {fl}: insufficient data")

        for metric_name, ratio_key in [("mean", "separation_ratio_mean"), ("max", "separation_ratio_max")]:
            if fs[ratio_key] is not None and fs[ratio_key] > best_ratio:
                best_ratio = fs[ratio_key]
                best_farm = fl
                best_metric = metric_name

    if best_farm:
        print(f"\n  Best separation: Farm {best_farm} ({best_metric} error) — {best_ratio:.2f}x ratio")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
