"""
Thermal Degradation Index (TDI) — Phase 6

Combines NBM residual z-scores (weight 0.7) and LSTM-Autoencoder reconstruction
errors (weight 0.3) into a single 0-100 health score per event.

Thresholds:
  Green  (0-30):  Healthy / normal operation
  Yellow (30-60): Watch / early degradation
  Red    (60+):   Alert / significant anomaly
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NBM_WEIGHT = 0.70
AE_WEIGHT = 0.30

THRESHOLDS = {"green": (0, 30), "yellow": (30, 60), "red": (60, 100)}


def get_tdi_thresholds():
    """Return threshold dict."""
    return THRESHOLDS


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_tdi(project_root):
    """
    Compute the Thermal Degradation Index for all 95 events.

    Returns a DataFrame with columns:
        farm, event_id, event_label, event_description,
        nbm_score, ae_score, combined_raw, tdi_score, tdi_status
    """
    root = Path(project_root)

    # ----- Load NBM event scores -----
    nbm_df = pd.read_parquet(root / "data" / "processed" / "event_scores.parquet")

    # ----- Load AE reconstruction summary -----
    with open(root / "outputs" / "reports" / "ae_reconstruction_summary.json") as f:
        ae_data = json.load(f)

    ae_events = pd.DataFrame(ae_data["per_event"])
    # We use pred_mean_error as the AE signal (prediction-window reconstruction error)
    ae_events = ae_events[["farm", "event_id", "event_label", "event_description",
                            "pred_mean_error"]].copy()

    # ----- Merge -----
    merged = nbm_df.merge(
        ae_events[["farm", "event_id", "pred_mean_error"]],
        on=["farm", "event_id"],
        how="left",
    )

    # ----- Z-score AE errors using normal events' distribution -----
    normal_mask = merged["event_label"] == "normal"
    ae_normal_mean = merged.loc[normal_mask, "pred_mean_error"].mean()
    ae_normal_std = merged.loc[normal_mask, "pred_mean_error"].std()

    # z-score
    merged["ae_zscore"] = (merged["pred_mean_error"] - ae_normal_mean) / ae_normal_std

    # Clip extreme z-scores to avoid outlier domination (e.g., Farm C events 11, 33)
    merged["ae_zscore"] = merged["ae_zscore"].clip(-5, 10)

    # ----- Combine -----
    merged["nbm_score"] = merged["aggregated_score"]
    merged["ae_score"] = merged["ae_zscore"]
    merged["combined_raw"] = NBM_WEIGHT * merged["nbm_score"] + AE_WEIGHT * merged["ae_score"]

    # ----- Sigmoid normalization to 0-100 -----
    # Calibrate so normal events cluster around TDI 10-20, anomalies spread 40-100
    normal_raw = merged.loc[normal_mask, "combined_raw"]
    raw_mean = normal_raw.mean()
    raw_std = normal_raw.std()

    # Scale factor: we want normal events ~1 std above mean to map to ~TDI 30
    # sigmoid(x) = 0.30 when x ≈ -0.847
    # So (raw_score - raw_mean) / raw_std * scale = -0.847 when raw_score = raw_mean + 1*raw_std
    # => scale * 1.0 = -0.847 ... that doesn't work.
    # Instead: we want the center of normal distribution (raw_mean) to map to ~TDI 15
    # sigmoid(x) = 0.15 => x ≈ -1.735
    # For raw_score = raw_mean: (0) * scale + offset = -1.735
    # For raw_score = raw_mean + 3*raw_std (strong anomaly): (3) * scale + offset ≈ 1.1 => TDI=75
    # offset = -1.735, scale * 3 + (-1.735) = 1.1 => scale = 0.945
    # Let's try: scale_factor = 0.95, offset so that sigmoid(0*0.95 + offset) = 0.15
    # Actually, let's just parametrize directly:
    # z = (raw - raw_mean) / raw_std
    # tdi = 100 * sigmoid(scale * z + shift)
    # We want: z=0 (normal mean) => TDI ~ 15 => sigmoid(shift) = 0.15 => shift = -1.735
    # z=3 (3 std above normal) => TDI ~ 75 => sigmoid(3*scale - 1.735) = 0.75 => 3*scale - 1.735 = 1.0986 => scale = 0.945
    # z=5 (extreme anomaly) => TDI ~ 90 => sigmoid(5*0.945 - 1.735) = sigmoid(2.99) = 0.952 => TDI ~ 95 ✓

    scale_factor = 0.95
    shift = -1.735

    z = (merged["combined_raw"] - raw_mean) / raw_std
    merged["tdi_score"] = 100.0 * expit(scale_factor * z + shift)

    # ----- Assign status -----
    def _status(tdi):
        if tdi < 30:
            return "Green"
        elif tdi < 60:
            return "Yellow"
        else:
            return "Red"

    merged["tdi_status"] = merged["tdi_score"].apply(_status)

    # ----- Add event description from ae_events -----
    desc_map = ae_events.set_index(["farm", "event_id"])["event_description"].to_dict()
    merged["event_description"] = merged.apply(
        lambda r: desc_map.get((r["farm"], r["event_id"]), ""), axis=1
    )

    # ----- Select output columns -----
    result = merged[[
        "farm", "event_id", "event_label", "event_description",
        "nbm_score", "ae_score", "combined_raw", "tdi_score", "tdi_status"
    ]].copy()

    result = result.sort_values("tdi_score", ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def tdi_summary_stats(tdi_df):
    """
    Compute summary statistics for TDI scores.

    Returns a dict with:
      - mean_tdi_normal, mean_tdi_anomaly
      - detection_rate (anomalies that are Yellow or Red)
      - false_alarm_rate (normals that are Yellow or Red)
      - per_farm breakdown
      - threshold_analysis at Green/Yellow/Red boundaries
    """
    normal = tdi_df[tdi_df["event_label"] == "normal"]
    anomaly = tdi_df[tdi_df["event_label"] == "anomaly"]

    stats = {
        "n_events": len(tdi_df),
        "n_normal": len(normal),
        "n_anomaly": len(anomaly),
        "mean_tdi_normal": float(normal["tdi_score"].mean()),
        "mean_tdi_anomaly": float(anomaly["tdi_score"].mean()),
        "median_tdi_normal": float(normal["tdi_score"].median()),
        "median_tdi_anomaly": float(anomaly["tdi_score"].median()),
        "std_tdi_normal": float(normal["tdi_score"].std()),
        "std_tdi_anomaly": float(anomaly["tdi_score"].std()),
    }

    # Detection at each threshold
    for name, (lo, hi) in THRESHOLDS.items():
        n_anom_above = int((anomaly["tdi_score"] >= lo).sum())
        n_norm_above = int((normal["tdi_score"] >= lo).sum())
        stats[f"anomalies_above_{name}_threshold_{lo}"] = n_anom_above
        stats[f"normals_above_{name}_threshold_{lo}"] = n_norm_above

    # Yellow-or-Red detection rate
    detected = (anomaly["tdi_score"] >= 30).sum()
    stats["detection_rate_yellow_plus"] = float(detected / len(anomaly)) if len(anomaly) > 0 else 0
    false_alarms = (normal["tdi_score"] >= 30).sum()
    stats["false_alarm_rate_yellow_plus"] = float(false_alarms / len(normal)) if len(normal) > 0 else 0

    # Red-only detection rate
    detected_red = (anomaly["tdi_score"] >= 60).sum()
    stats["detection_rate_red"] = float(detected_red / len(anomaly)) if len(anomaly) > 0 else 0
    false_alarms_red = (normal["tdi_score"] >= 60).sum()
    stats["false_alarm_rate_red"] = float(false_alarms_red / len(normal)) if len(normal) > 0 else 0

    # Per-farm breakdown
    per_farm = {}
    for farm in sorted(tdi_df["farm"].unique()):
        farm_df = tdi_df[tdi_df["farm"] == farm]
        farm_normal = farm_df[farm_df["event_label"] == "normal"]
        farm_anomaly = farm_df[farm_df["event_label"] == "anomaly"]

        farm_stats = {
            "n_normal": len(farm_normal),
            "n_anomaly": len(farm_anomaly),
            "mean_tdi_normal": float(farm_normal["tdi_score"].mean()) if len(farm_normal) > 0 else None,
            "mean_tdi_anomaly": float(farm_anomaly["tdi_score"].mean()) if len(farm_anomaly) > 0 else None,
        }

        if len(farm_anomaly) > 0:
            farm_stats["detection_rate_yellow_plus"] = float(
                (farm_anomaly["tdi_score"] >= 30).sum() / len(farm_anomaly)
            )
            farm_stats["detection_rate_red"] = float(
                (farm_anomaly["tdi_score"] >= 60).sum() / len(farm_anomaly)
            )
        if len(farm_normal) > 0:
            farm_stats["false_alarm_rate_yellow_plus"] = float(
                (farm_normal["tdi_score"] >= 30).sum() / len(farm_normal)
            )

        # Status distribution
        for status in ["Green", "Yellow", "Red"]:
            farm_stats[f"n_{status.lower()}"] = int(
                (farm_df["tdi_status"] == status).sum()
            )

        per_farm[farm] = farm_stats

    stats["per_farm"] = per_farm

    # Status distribution overall
    for status in ["Green", "Yellow", "Red"]:
        stats[f"n_{status.lower()}_total"] = int((tdi_df["tdi_status"] == status).sum())

    return stats
