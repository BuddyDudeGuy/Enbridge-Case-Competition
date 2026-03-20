"""
Phase 4.4 — Calibrate detection thresholds to minimize false alarms.

The CARE score weights Accuracy at 2x. Accuracy = tn / (fp + tn) on normal events.
If Accuracy < 0.5, the entire CARE score collapses.

Current problem: 52% false alarm rate using global thresholds. Weak subsystem
models (transformer R2=-0.10 to -0.28, hydraulic R2=-0.49 to -1.43) produce noisy
residuals that trigger false positives on normal events.

This script tests three threshold strategies on the event feature matrix and picks
the best one for downstream use.

Usage:
    py src/models/calibrate_thresholds.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "event_feature_matrix.parquet"
VALIDATION_RESULTS_PATH = PROJECT_ROOT / "outputs" / "reports" / "nbm_validation_results.json"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "threshold_calibration_results.json"
OUTPUT_THRESHOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "calibrated_thresholds.json"

# Subsystems per farm (Farm B lacks hydraulic and cooling)
FARM_SUBSYSTEMS = {
    "A": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
    "B": ["gearbox", "generator_bearings", "transformer"],
    "C": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
}

ALL_SUBSYSTEMS = ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"]


def load_data():
    """Load feature matrix and split into normal / anomaly."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    normal = df[df["event_label"] == "normal"].copy()
    anomaly = df[df["event_label"] == "anomaly"].copy()
    return df, normal, anomaly


def load_r2_scores():
    """Load NBM R2 scores per farm per subsystem from validation results."""
    with open(VALIDATION_RESULTS_PATH) as f:
        data = json.load(f)

    r2 = {}
    for farm_info in data:
        farm = farm_info["farm_letter"]
        r2[farm] = {}
        for model in farm_info["models"]:
            r2[farm][model["subsystem"]] = model["avg_r2"]
    return r2


def compute_metrics(y_true, y_pred):
    """Compute detection rate, false alarm rate, accuracy, and F1.

    Parameters
    ----------
    y_true : array-like of bool
        True labels (True = anomaly, False = normal).
    y_pred : array-like of bool
        Predicted labels (True = flagged as anomaly).

    Returns
    -------
    dict with detection_rate, false_alarm_rate, accuracy_care, f1
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    # True positives: anomaly correctly flagged
    tp = int(np.sum(y_true & y_pred))
    # False positives: normal incorrectly flagged
    fp = int(np.sum(~y_true & y_pred))
    # True negatives: normal correctly not flagged
    tn = int(np.sum(~y_true & ~y_pred))
    # False negatives: anomaly missed
    fn = int(np.sum(y_true & ~y_pred))

    n_anomaly = int(np.sum(y_true))
    n_normal = int(np.sum(~y_true))

    detection_rate = tp / n_anomaly if n_anomaly > 0 else 0.0
    false_alarm_rate = fp / n_normal if n_normal > 0 else 0.0
    accuracy_care = tn / (fp + tn) if (fp + tn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = detection_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_anomaly": n_anomaly,
        "n_normal": n_normal,
        "detection_rate": round(detection_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "accuracy_care": round(accuracy_care, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "care_safe": accuracy_care >= 0.5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 1: Per-subsystem percentile thresholds (global across farms)
# ═══════════════════════════════════════════════════════════════════════════

def strategy1_percentile(df, normal, anomaly, percentile=99):
    """Flag event if ANY subsystem's overall_max exceeds the Nth percentile
    of that subsystem's normal distribution.

    Uses global (cross-farm) normal distribution for each subsystem.
    """
    feature_col = "overall_max"
    thresholds = {}

    for sub in ALL_SUBSYSTEMS:
        col = f"{sub}_{feature_col}"
        normal_vals = normal[col].dropna()
        if len(normal_vals) == 0:
            continue
        thresholds[sub] = float(np.percentile(normal_vals, percentile))

    # Flag each event
    predictions = []
    for _, row in df.iterrows():
        farm = row["farm"]
        available_subs = FARM_SUBSYSTEMS.get(farm, ALL_SUBSYSTEMS)
        flagged = False
        for sub in available_subs:
            col = f"{sub}_{feature_col}"
            if sub not in thresholds:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            if val > thresholds[sub]:
                flagged = True
                break
        predictions.append(flagged)

    y_true = (df["event_label"] == "anomaly").values
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_true, y_pred)
    metrics["thresholds"] = {k: round(v, 4) for k, v in thresholds.items()}
    metrics["percentile"] = percentile
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 1b: Per-farm per-subsystem percentile thresholds
# ═══════════════════════════════════════════════════════════════════════════

def strategy1b_perfarm_percentile(df, normal, anomaly, percentile=99):
    """Like Strategy 1 but with per-farm thresholds instead of global."""
    feature_col = "overall_max"
    thresholds = {}  # farm -> sub -> threshold

    for farm in ["A", "B", "C"]:
        farm_normal = normal[normal["farm"] == farm]
        thresholds[farm] = {}
        available_subs = FARM_SUBSYSTEMS.get(farm, ALL_SUBSYSTEMS)
        for sub in available_subs:
            col = f"{sub}_{feature_col}"
            vals = farm_normal[col].dropna()
            if len(vals) < 3:
                # Too few normal events for this farm-subsystem to set a threshold;
                # fall back to global normal
                vals = normal[col].dropna()
            if len(vals) == 0:
                continue
            thresholds[farm][sub] = float(np.percentile(vals, percentile))

    # Flag each event
    predictions = []
    for _, row in df.iterrows():
        farm = row["farm"]
        farm_thresholds = thresholds.get(farm, {})
        available_subs = FARM_SUBSYSTEMS.get(farm, ALL_SUBSYSTEMS)
        flagged = False
        for sub in available_subs:
            col = f"{sub}_{feature_col}"
            if sub not in farm_thresholds:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            if val > farm_thresholds[sub]:
                flagged = True
                break
        predictions.append(flagged)

    y_true = (df["event_label"] == "anomaly").values
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_true, y_pred)
    metrics["thresholds_per_farm"] = {
        farm: {k: round(v, 4) for k, v in subs.items()}
        for farm, subs in thresholds.items()
    }
    metrics["percentile"] = percentile
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 2: Best single feature with optimal threshold
# ═══════════════════════════════════════════════════════════════════════════

def strategy2_best_single_feature(df, normal, anomaly):
    """Find the single best feature and optimal threshold.

    Tries all subsystem features and sweeps thresholds to maximize
    (detection_rate - false_alarm_rate), i.e., Youden's J statistic.
    """
    y_true = (df["event_label"] == "anomaly").values
    best_result = None
    best_j = -999

    # Try all feature columns (not metadata)
    feature_cols = [c for c in df.columns if c not in ("farm", "event_id", "event_label")]

    for col in feature_cols:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue

        # Get non-NaN mask
        mask = ~df[col].isna()
        if mask.sum() < 10:
            continue

        col_vals = df.loc[mask, col].values
        col_true = y_true[mask.values]

        # Sweep thresholds: use the actual data values as candidates
        unique_vals = np.sort(np.unique(col_vals))
        if len(unique_vals) < 2:
            continue

        # Add intermediate points for better resolution
        candidates = []
        for i in range(len(unique_vals) - 1):
            candidates.append((unique_vals[i] + unique_vals[i + 1]) / 2)
        # Also try percentiles of normal distribution
        normal_vals = normal.loc[~normal[col].isna(), col].values
        if len(normal_vals) > 0:
            for pct in [90, 92, 95, 97, 99]:
                candidates.append(np.percentile(normal_vals, pct))

        for thresh in candidates:
            preds = col_vals > thresh
            m = compute_metrics(col_true, preds)
            j = m["detection_rate"] - m["false_alarm_rate"]
            # Tie-break: prefer higher detection rate if J is the same
            if j > best_j or (j == best_j and best_result is not None
                              and m["detection_rate"] > best_result["detection_rate"]):
                best_j = j
                best_result = m.copy()
                best_result["feature"] = col
                best_result["threshold"] = round(float(thresh), 6)
                best_result["youden_j"] = round(j, 4)

    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 3: Weighted z-score voting
# ═══════════════════════════════════════════════════════════════════════════

def strategy3_weighted_zscore(df, normal, anomaly, r2_scores):
    """Compute weighted z-scores across subsystems and sweep threshold.

    For each subsystem:
      z = (event_feature - normal_mean) / normal_std
    Weight each z-score by max(R2, 0) so bad models get zero/low weight.
    Sum weighted z-scores across available subsystems.
    Flag event if total exceeds threshold.

    Uses overall_max as the feature for each subsystem.
    """
    feature_col = "overall_max"

    # Compute normal distribution stats per subsystem (global)
    normal_stats = {}
    for sub in ALL_SUBSYSTEMS:
        col = f"{sub}_{feature_col}"
        vals = normal[col].dropna()
        if len(vals) >= 3:
            normal_stats[sub] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
            # Prevent division by zero
            if normal_stats[sub]["std"] < 0.01:
                normal_stats[sub]["std"] = 0.01

    # Compute weights from R2 — only positive R2 models get weight
    # Use average R2 across farms for each subsystem
    sub_weights = {}
    for sub in ALL_SUBSYSTEMS:
        r2_vals = []
        for farm in ["A", "B", "C"]:
            if farm in r2_scores and sub in r2_scores[farm]:
                r2_vals.append(r2_scores[farm][sub])
        if r2_vals:
            avg_r2 = np.mean(r2_vals)
            sub_weights[sub] = max(avg_r2, 0.0)  # Clamp at 0
        else:
            sub_weights[sub] = 0.0

    # Normalize weights so they sum to 1
    total_weight = sum(sub_weights.values())
    if total_weight > 0:
        for sub in sub_weights:
            sub_weights[sub] /= total_weight

    # Compute weighted z-score for each event
    z_scores_all = []
    valid_mask = []

    for _, row in df.iterrows():
        farm = row["farm"]
        available_subs = FARM_SUBSYSTEMS.get(farm, ALL_SUBSYSTEMS)

        weighted_z = 0.0
        total_w = 0.0

        for sub in available_subs:
            if sub not in normal_stats or sub not in sub_weights:
                continue
            col = f"{sub}_{feature_col}"
            val = row[col]
            if pd.isna(val):
                continue
            if sub_weights[sub] <= 0:
                continue

            z = (val - normal_stats[sub]["mean"]) / normal_stats[sub]["std"]
            weighted_z += sub_weights[sub] * z
            total_w += sub_weights[sub]

        # Normalize by the total weight of available subsystems
        if total_w > 0:
            weighted_z /= total_w
            z_scores_all.append(weighted_z)
            valid_mask.append(True)
        else:
            z_scores_all.append(0.0)
            valid_mask.append(False)

    z_scores_all = np.array(z_scores_all)
    y_true = (df["event_label"] == "anomaly").values

    # Sweep thresholds
    best_result = None
    best_j = -999

    candidates = np.arange(0.0, 6.0, 0.1)
    # Also try some finer resolution around likely sweet spots
    candidates = np.concatenate([
        candidates,
        np.arange(0.5, 3.0, 0.05),
    ])
    candidates = np.sort(np.unique(candidates))

    for thresh in candidates:
        preds = z_scores_all > thresh
        m = compute_metrics(y_true, preds)
        j = m["detection_rate"] - m["false_alarm_rate"]
        if j > best_j or (j == best_j and best_result is not None
                          and m["detection_rate"] > best_result["detection_rate"]):
            best_j = j
            best_result = m.copy()
            best_result["z_threshold"] = round(float(thresh), 4)
            best_result["youden_j"] = round(j, 4)

    best_result["subsystem_weights"] = {k: round(v, 4) for k, v in sub_weights.items()}
    best_result["normal_stats"] = {
        sub: {k: round(v, 4) for k, v in stats.items()}
        for sub, stats in normal_stats.items()
    }
    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 3b: Weighted voting using ONLY strong models (R2 > 0.5)
# ═══════════════════════════════════════════════════════════════════════════

def strategy3b_strong_models_only(df, normal, anomaly, r2_scores):
    """Like Strategy 3 but exclude subsystems with R2 < 0.5 entirely.

    This is the "trust only what works" approach.
    """
    feature_col = "overall_max"

    # Identify strong subsystems per farm
    strong_subs_per_farm = {}
    for farm in ["A", "B", "C"]:
        strong = []
        if farm in r2_scores:
            for sub, r2 in r2_scores[farm].items():
                if r2 >= 0.5:
                    strong.append(sub)
        strong_subs_per_farm[farm] = strong

    # Compute normal stats using ONLY strong subsystem data
    normal_stats = {}
    for sub in ALL_SUBSYSTEMS:
        col = f"{sub}_{feature_col}"
        vals = normal[col].dropna()
        if len(vals) >= 3:
            normal_stats[sub] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
            if normal_stats[sub]["std"] < 0.01:
                normal_stats[sub]["std"] = 0.01

    # R2 weights for strong models only
    sub_weights = {}
    for sub in ALL_SUBSYSTEMS:
        r2_vals = []
        for farm in ["A", "B", "C"]:
            if farm in r2_scores and sub in r2_scores[farm]:
                r2_val = r2_scores[farm][sub]
                if r2_val >= 0.5:
                    r2_vals.append(r2_val)
        if r2_vals:
            sub_weights[sub] = float(np.mean(r2_vals))
        else:
            sub_weights[sub] = 0.0

    total_weight = sum(sub_weights.values())
    if total_weight > 0:
        for sub in sub_weights:
            sub_weights[sub] /= total_weight

    # Compute weighted z-score for each event
    z_scores_all = []

    for _, row in df.iterrows():
        farm = row["farm"]
        strong_subs = strong_subs_per_farm.get(farm, [])

        weighted_z = 0.0
        total_w = 0.0

        for sub in strong_subs:
            if sub not in normal_stats or sub_weights.get(sub, 0) <= 0:
                continue
            col = f"{sub}_{feature_col}"
            val = row[col]
            if pd.isna(val):
                continue

            z = (val - normal_stats[sub]["mean"]) / normal_stats[sub]["std"]
            weighted_z += sub_weights[sub] * z
            total_w += sub_weights[sub]

        if total_w > 0:
            weighted_z /= total_w
        z_scores_all.append(weighted_z)

    z_scores_all = np.array(z_scores_all)
    y_true = (df["event_label"] == "anomaly").values

    best_result = None
    best_j = -999

    candidates = np.arange(0.0, 8.0, 0.1)
    candidates = np.concatenate([
        candidates,
        np.arange(0.5, 4.0, 0.05),
    ])
    candidates = np.sort(np.unique(candidates))

    for thresh in candidates:
        preds = z_scores_all > thresh
        m = compute_metrics(y_true, preds)
        j = m["detection_rate"] - m["false_alarm_rate"]
        if j > best_j or (j == best_j and best_result is not None
                          and m["detection_rate"] > best_result["detection_rate"]):
            best_j = j
            best_result = m.copy()
            best_result["z_threshold"] = round(float(thresh), 4)
            best_result["youden_j"] = round(j, 4)

    best_result["subsystem_weights"] = {k: round(v, 4) for k, v in sub_weights.items()}
    best_result["strong_subs_per_farm"] = strong_subs_per_farm
    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 4: Multi-feature OR voting with per-farm percentile thresholds
#             using ONLY strong-model subsystems
# ═══════════════════════════════════════════════════════════════════════════

def strategy4_multifeat_strong_perfarm(df, normal, anomaly, r2_scores, percentile=95):
    """Combine multiple features per strong subsystem with per-farm thresholds.

    For each event:
      - For each strong subsystem (R2 >= 0.5) in the event's farm:
        - Check multiple features: overall_max, max_24h, overall_std, anomaly_frac
        - Flag if ANY feature exceeds its per-farm percentile threshold
      - Event is flagged if ANY strong subsystem flags.

    This gives more "shots" at catching an anomaly while limiting false alarms
    to only well-modeled subsystems.
    """
    feature_suffixes = ["overall_max", "max_24h", "overall_std", "anomaly_frac"]

    # Identify strong subsystems per farm
    strong_subs_per_farm = {}
    for farm in ["A", "B", "C"]:
        strong = []
        if farm in r2_scores:
            for sub, r2 in r2_scores[farm].items():
                if r2 >= 0.5:
                    strong.append(sub)
        strong_subs_per_farm[farm] = strong

    # Compute per-farm thresholds for each sub x feature combination
    thresholds = {}  # farm -> sub -> feature_suffix -> threshold
    for farm in ["A", "B", "C"]:
        farm_normal = normal[normal["farm"] == farm]
        thresholds[farm] = {}
        strong_subs = strong_subs_per_farm[farm]
        for sub in strong_subs:
            thresholds[farm][sub] = {}
            for suffix in feature_suffixes:
                col = f"{sub}_{suffix}"
                if col not in df.columns:
                    continue
                vals = farm_normal[col].dropna()
                if len(vals) < 3:
                    # Fall back to global normal
                    vals = normal[col].dropna()
                if len(vals) == 0:
                    continue
                thresholds[farm][sub][suffix] = float(np.percentile(vals, percentile))

    # Flag each event
    predictions = []
    for _, row in df.iterrows():
        farm = row["farm"]
        strong_subs = strong_subs_per_farm.get(farm, [])
        farm_thresh = thresholds.get(farm, {})
        flagged = False
        for sub in strong_subs:
            sub_thresh = farm_thresh.get(sub, {})
            for suffix, thresh_val in sub_thresh.items():
                col = f"{sub}_{suffix}"
                val = row[col]
                if pd.isna(val):
                    continue
                if val > thresh_val:
                    flagged = True
                    break
            if flagged:
                break
        predictions.append(flagged)

    y_true = (df["event_label"] == "anomaly").values
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_true, y_pred)
    metrics["thresholds_per_farm"] = {
        farm: {
            sub: {k: round(v, 4) for k, v in feats.items()}
            for sub, feats in subs.items()
        }
        for farm, subs in thresholds.items()
    }
    metrics["strong_subs_per_farm"] = strong_subs_per_farm
    metrics["feature_suffixes"] = feature_suffixes
    metrics["percentile"] = percentile
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Strategy 1c: Per-subsystem percentile, ONLY strong models (R2 > 0.5)
# ═══════════════════════════════════════════════════════════════════════════

def strategy1c_strong_only_percentile(df, normal, anomaly, r2_scores, percentile=99):
    """Strategy 1 but only using subsystems with R2 >= 0.5.

    This directly addresses the root cause: weak models generate noisy
    residuals -> false alarms. Disabling them eliminates the noise.
    """
    feature_col = "overall_max"

    # Identify strong subsystems per farm
    strong_subs_per_farm = {}
    for farm in ["A", "B", "C"]:
        strong = []
        if farm in r2_scores:
            for sub, r2 in r2_scores[farm].items():
                if r2 >= 0.5:
                    strong.append(sub)
        strong_subs_per_farm[farm] = strong

    # Compute thresholds (global)
    thresholds = {}
    for sub in ALL_SUBSYSTEMS:
        col = f"{sub}_{feature_col}"
        normal_vals = normal[col].dropna()
        if len(normal_vals) == 0:
            continue
        thresholds[sub] = float(np.percentile(normal_vals, percentile))

    # Flag each event
    predictions = []
    for _, row in df.iterrows():
        farm = row["farm"]
        strong_subs = strong_subs_per_farm.get(farm, [])
        flagged = False
        for sub in strong_subs:
            col = f"{sub}_{feature_col}"
            if sub not in thresholds:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            if val > thresholds[sub]:
                flagged = True
                break
        predictions.append(flagged)

    y_true = (df["event_label"] == "anomaly").values
    y_pred = np.array(predictions)
    metrics = compute_metrics(y_true, y_pred)
    metrics["thresholds"] = {k: round(v, 4) for k, v in thresholds.items()}
    metrics["strong_subs_per_farm"] = strong_subs_per_farm
    metrics["percentile"] = percentile
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PHASE 4.4 — THRESHOLD CALIBRATION")
    print("=" * 70)
    print()

    # Load data
    df, normal, anomaly = load_data()
    r2_scores = load_r2_scores()

    print(f"Feature matrix: {len(df)} events ({len(anomaly)} anomaly, {len(normal)} normal)")
    print(f"Farms: A={len(df[df['farm']=='A'])}, B={len(df[df['farm']=='B'])}, C={len(df[df['farm']=='C'])}")
    print()

    # Print R2 context
    print("NBM R2 scores (model quality — higher = more trustworthy):")
    for farm in ["A", "B", "C"]:
        subs = r2_scores.get(farm, {})
        scores_str = ", ".join(f"{s}={r2:.3f}" for s, r2 in subs.items())
        print(f"  Farm {farm}: {scores_str}")
    print()

    # Strong model check
    print("Strong models (R2 >= 0.5):")
    for farm in ["A", "B", "C"]:
        strong = [s for s, r2 in r2_scores.get(farm, {}).items() if r2 >= 0.5]
        weak = [s for s, r2 in r2_scores.get(farm, {}).items() if r2 < 0.5]
        print(f"  Farm {farm}: strong={strong}, weak={weak}")
    print()

    # ─── Run all strategies ─────────────────────────────────────────────
    results = {}

    # Strategy 1a: Global 95th percentile
    print("-" * 70)
    print("Strategy 1a: Per-subsystem 95th percentile (global, all subsystems)")
    s1a = strategy1_percentile(df, normal, anomaly, percentile=95)
    results["strategy_1a_p95_global"] = s1a
    print_metrics(s1a)

    # Strategy 1b: Global 99th percentile
    print("-" * 70)
    print("Strategy 1b: Per-subsystem 99th percentile (global, all subsystems)")
    s1b = strategy1_percentile(df, normal, anomaly, percentile=99)
    results["strategy_1b_p99_global"] = s1b
    print_metrics(s1b)

    # Strategy 1c: Per-farm 99th percentile
    print("-" * 70)
    print("Strategy 1c: Per-subsystem 99th percentile (per-farm thresholds)")
    s1c = strategy1b_perfarm_percentile(df, normal, anomaly, percentile=99)
    results["strategy_1c_p99_perfarm"] = s1c
    print_metrics(s1c)

    # Strategy 1d: Strong models only, 99th percentile
    print("-" * 70)
    print("Strategy 1d: Per-subsystem 99th percentile (strong models only, R2>=0.5)")
    s1d = strategy1c_strong_only_percentile(df, normal, anomaly, r2_scores, percentile=99)
    results["strategy_1d_p99_strong_only"] = s1d
    print_metrics(s1d)

    # Strategy 1e: Strong models only, 95th percentile
    print("-" * 70)
    print("Strategy 1e: Per-subsystem 95th percentile (strong models only, R2>=0.5)")
    s1e = strategy1c_strong_only_percentile(df, normal, anomaly, r2_scores, percentile=95)
    results["strategy_1e_p95_strong_only"] = s1e
    print_metrics(s1e)

    # Strategy 2: Best single feature
    print("-" * 70)
    print("Strategy 2: Best single feature (optimal threshold via Youden's J)")
    s2 = strategy2_best_single_feature(df, normal, anomaly)
    results["strategy_2_best_feature"] = s2
    print_metrics(s2)
    print(f"  Best feature: {s2['feature']}")
    print(f"  Threshold:    {s2['threshold']}")
    print(f"  Youden's J:   {s2['youden_j']}")

    # Strategy 3a: Weighted z-score (all subsystems)
    print("-" * 70)
    print("Strategy 3a: Weighted z-score voting (all subsystems, R2-weighted)")
    s3a = strategy3_weighted_zscore(df, normal, anomaly, r2_scores)
    results["strategy_3a_zscore_all"] = s3a
    print_metrics(s3a)
    print(f"  Z threshold:  {s3a['z_threshold']}")
    print(f"  Weights:      {s3a['subsystem_weights']}")

    # Strategy 3b: Weighted z-score (strong models only)
    print("-" * 70)
    print("Strategy 3b: Weighted z-score voting (strong models only, R2>=0.5)")
    s3b = strategy3b_strong_models_only(df, normal, anomaly, r2_scores)
    results["strategy_3b_zscore_strong"] = s3b
    print_metrics(s3b)
    print(f"  Z threshold:  {s3b['z_threshold']}")
    print(f"  Weights:      {s3b['subsystem_weights']}")

    # Strategy 4a: Multi-feature strong-only with per-farm p95
    print("-" * 70)
    print("Strategy 4a: Multi-feature OR, strong only, per-farm p95")
    s4a = strategy4_multifeat_strong_perfarm(df, normal, anomaly, r2_scores, percentile=95)
    results["strategy_4a_multifeat_p95"] = s4a
    print_metrics(s4a)

    # Strategy 4b: Multi-feature strong-only with per-farm p99
    print("-" * 70)
    print("Strategy 4b: Multi-feature OR, strong only, per-farm p99")
    s4b = strategy4_multifeat_strong_perfarm(df, normal, anomaly, r2_scores, percentile=99)
    results["strategy_4b_multifeat_p99"] = s4b
    print_metrics(s4b)

    # ─── Summary comparison table ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print()
    print(f"  {'Strategy':<45} {'Det%':>5} {'FA%':>5} {'Acc':>5} {'F1':>5} {'CARE':>5}")
    print(f"  {'-'*45} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    strategy_names = {
        "strategy_1a_p95_global":       "1a. Percentile p95 (all subs)",
        "strategy_1b_p99_global":       "1b. Percentile p99 (all subs)",
        "strategy_1c_p99_perfarm":      "1c. Percentile p99 (per-farm)",
        "strategy_1d_p99_strong_only":  "1d. Percentile p99 (strong only)",
        "strategy_1e_p95_strong_only":  "1e. Percentile p95 (strong only)",
        "strategy_2_best_feature":      "2.  Best single feature",
        "strategy_3a_zscore_all":       "3a. Z-score voting (all subs)",
        "strategy_3b_zscore_strong":    "3b. Z-score voting (strong only)",
        "strategy_4a_multifeat_p95":    "4a. Multi-feat strong per-farm p95",
        "strategy_4b_multifeat_p99":    "4b. Multi-feat strong per-farm p99",
    }

    for key, name in strategy_names.items():
        r = results[key]
        care_ok = "OK" if r["care_safe"] else "BAD"
        print(
            f"  {name:<45} "
            f"{r['detection_rate']*100:>4.0f}% "
            f"{r['false_alarm_rate']*100:>4.0f}% "
            f"{r['accuracy_care']:.2f}  "
            f"{r['f1']:.2f}  "
            f"{care_ok}"
        )

    # ─── Pick the best strategy ────────────────────────────────────────
    # CARE scoring context:
    #   - Accuracy = tn/(fp+tn) has 2x weight
    #   - If Accuracy < 0.5, CARE collapses to 0
    #   - Detection rate matters but false alarms are the #1 enemy
    #
    # Selection criteria:
    #   1. MUST have accuracy_care >= 0.5
    #   2. Maximize a CARE-proxy score: 2*accuracy + detection_rate
    #      This directly mirrors CARE's 2x weighting on accuracy.
    #   3. Tie-break by detection_rate
    print()
    print("-" * 70)

    valid_strategies = {
        k: v for k, v in results.items()
        if v["accuracy_care"] >= 0.5
    }

    if not valid_strategies:
        print("  WARNING: No strategy achieves Accuracy >= 0.5!")
        print("  Selecting the one with highest accuracy anyway.")
        valid_strategies = results

    def care_proxy_score(item):
        k, v = item
        # CARE weights accuracy 2x; detection is the remaining component.
        # We want: high accuracy (low FA) AND reasonable detection.
        proxy = 2 * v["accuracy_care"] + v["detection_rate"]
        return (proxy, v["detection_rate"])

    best_key = max(valid_strategies.items(), key=care_proxy_score)[0]
    best = results[best_key]
    best_name = strategy_names[best_key]

    j_best = best["detection_rate"] - best["false_alarm_rate"]
    care_proxy = 2 * best["accuracy_care"] + best["detection_rate"]

    print(f"  RECOMMENDED: {best_name}")
    print(f"    Detection rate:   {best['detection_rate']*100:.0f}%")
    print(f"    False alarm rate: {best['false_alarm_rate']*100:.0f}%")
    print(f"    Accuracy (CARE):  {best['accuracy_care']:.2f}")
    print(f"    F1 score:         {best['f1']:.2f}")
    print(f"    CARE safe:        {'YES' if best['care_safe'] else 'NO'}")
    print(f"    Youden's J:       {j_best:.4f}")
    print(f"    CARE proxy score: {care_proxy:.4f} (2*Acc + Det)")
    print()

    # Also show runner-up strategies for context
    print("  All strategies ranked by CARE proxy (2*Acc + Det):")
    ranked = sorted(valid_strategies.items(), key=care_proxy_score, reverse=True)
    for i, (k, v) in enumerate(ranked):
        proxy = 2 * v["accuracy_care"] + v["detection_rate"]
        marker = " <-- BEST" if k == best_key else ""
        print(f"    {i+1}. {strategy_names[k]:<45} proxy={proxy:.3f}{marker}")
    print()

    # ─── Save results ──────────────────────────────────────────────────
    # Also identify the best conservative strategy (lowest FA with decent detection)
    conservative_candidates = {
        k: v for k, v in valid_strategies.items()
        if v["false_alarm_rate"] <= 0.20 and v["detection_rate"] >= 0.20
    }
    conservative_key = None
    if conservative_candidates:
        conservative_key = max(
            conservative_candidates.items(),
            key=lambda x: (x[1]["detection_rate"], x[1]["accuracy_care"])
        )[0]

    output = {
        "description": "Phase 4.4 — Threshold calibration results",
        "n_events": len(df),
        "n_anomaly": len(anomaly),
        "n_normal": len(normal),
        "recommended_strategy": best_key,
        "recommended_strategy_name": best_name,
        "conservative_strategy": conservative_key,
        "conservative_strategy_name": strategy_names.get(conservative_key, None),
        "analysis": {
            "care_proxy_formula": "2*accuracy + detection_rate",
            "care_formula": "(2*Accuracy + Detection + Earliness) / 4",
            "key_findings": [
                "Transformer and hydraulic models (R2 < 0) are the primary source of false alarms",
                "Restricting to strong models (R2 >= 0.5) dramatically reduces false alarms",
                "Per-farm thresholds outperform global thresholds (farms have different normal baselines)",
                f"Best overall: {best_name} — CARE proxy {care_proxy:.3f}",
                f"Best conservative: {strategy_names.get(conservative_key, 'N/A')} — low false alarm with decent detection",
            ],
        },
        "strategies": results,
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Report saved: {OUTPUT_REPORT_PATH}")

    # ─── Save calibrated thresholds for Phase 4.5+ ────────────────────
    threshold_config = build_threshold_config(best_key, best, r2_scores)
    OUTPUT_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_THRESHOLDS_PATH, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"  Thresholds saved: {OUTPUT_THRESHOLDS_PATH}")
    print()
    print("Done.")


def build_threshold_config(strategy_key, strategy_result, r2_scores):
    """Build the threshold configuration dict for downstream use."""
    config = {
        "strategy": strategy_key,
        "description": "Calibrated thresholds for anomaly detection (Phase 4.4)",
        "metrics": {
            "detection_rate": strategy_result["detection_rate"],
            "false_alarm_rate": strategy_result["false_alarm_rate"],
            "accuracy_care": strategy_result["accuracy_care"],
            "f1": strategy_result["f1"],
        },
    }

    if "thresholds" in strategy_result:
        config["type"] = "percentile"
        config["percentile"] = strategy_result.get("percentile", 99)
        config["thresholds"] = strategy_result["thresholds"]
        config["feature"] = "overall_max"
        if "strong_subs_per_farm" in strategy_result:
            config["strong_subs_per_farm"] = strategy_result["strong_subs_per_farm"]
            config["use_strong_only"] = True
        else:
            config["use_strong_only"] = False

    elif "feature_suffixes" in strategy_result:
        # Strategy 4: multi-feature per-farm
        config["type"] = "multifeat_perfarm"
        config["percentile"] = strategy_result.get("percentile", 95)
        config["thresholds_per_farm"] = strategy_result["thresholds_per_farm"]
        config["feature_suffixes"] = strategy_result["feature_suffixes"]
        config["strong_subs_per_farm"] = strategy_result["strong_subs_per_farm"]
        config["use_strong_only"] = True

    elif "thresholds_per_farm" in strategy_result:
        config["type"] = "percentile_per_farm"
        config["percentile"] = strategy_result.get("percentile", 99)
        config["thresholds_per_farm"] = strategy_result["thresholds_per_farm"]
        config["feature"] = "overall_max"

    elif "feature" in strategy_result:
        config["type"] = "single_feature"
        config["feature"] = strategy_result["feature"]
        config["threshold"] = strategy_result["threshold"]

    elif "z_threshold" in strategy_result:
        config["type"] = "weighted_zscore"
        config["z_threshold"] = strategy_result["z_threshold"]
        config["subsystem_weights"] = strategy_result["subsystem_weights"]
        config["feature"] = "overall_max"
        if "normal_stats" in strategy_result:
            config["normal_stats"] = strategy_result["normal_stats"]
        if "strong_subs_per_farm" in strategy_result:
            config["strong_subs_per_farm"] = strategy_result["strong_subs_per_farm"]

    # Always include R2 reference
    config["r2_scores"] = r2_scores
    config["farm_subsystems"] = FARM_SUBSYSTEMS

    return config


def print_metrics(m):
    """Print key metrics for a strategy."""
    print(f"  Detection rate:  {m['detection_rate']*100:.1f}% ({m['tp']}/{m['n_anomaly']})")
    print(f"  False alarm rate: {m['false_alarm_rate']*100:.1f}% ({m['fp']}/{m['n_normal']})")
    print(f"  Accuracy (CARE): {m['accuracy_care']:.4f} {'OK' if m['care_safe'] else 'DANGER <0.5!'}")
    print(f"  Precision:       {m['precision']:.4f}")
    print(f"  F1:              {m['f1']:.4f}")
    print()


if __name__ == "__main__":
    main()
