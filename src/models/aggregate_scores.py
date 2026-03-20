"""
Phase 4.5 — Aggregated anomaly scoring via weighted voting.

Combines per-subsystem anomaly signals into a single score per event.
Stronger NBM models (higher R²) get more weight in the vote.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# Subsystem TDI weights (from thermal_config.py)
SUBSYSTEM_WEIGHTS = {
    "gearbox": 0.25,
    "generator_bearings": 0.20,
    "transformer": 0.20,
    "hydraulic": 0.15,
    "cooling": 0.10,
}

# Key features used for z-score computation
KEY_FEATURES = ["overall_max", "overall_mean", "trend_slope"]

# Subsystems present in the feature matrix (nacelle_ambient has no features)
ALL_SUBSYSTEMS = ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"]


def _load_r2_scores(project_root: Path) -> dict:
    """Load R² scores from NBM validation results, keyed by farm letter -> subsystem."""
    val_path = project_root / "outputs" / "reports" / "nbm_validation_results.json"
    with open(val_path) as f:
        results = json.load(f)

    r2_map = {}
    for farm_entry in results:
        letter = farm_entry["farm_letter"]
        r2_map[letter] = {}
        for model in farm_entry["models"]:
            r2_map[letter][model["subsystem"]] = model["avg_r2"]
    return r2_map


def _load_calibrated_r2(project_root: Path) -> dict:
    """Load R² scores from calibrated_thresholds.json as fallback."""
    cal_path = project_root / "data" / "processed" / "calibrated_thresholds.json"
    with open(cal_path) as f:
        cal = json.load(f)
    return cal.get("r2_scores", {})


def compute_normal_stats(df: pd.DataFrame) -> dict:
    """Compute mean and std of key features from normal events only, per farm.

    Parameters
    ----------
    df : pd.DataFrame
        Event feature matrix with 'farm', 'event_label', and subsystem feature columns.

    Returns
    -------
    dict
        Nested: {farm: {subsystem: {feature: {mean, std}}}}
    """
    normal = df[df["event_label"] == "normal"]
    stats = {}

    for farm in df["farm"].unique():
        farm_normal = normal[normal["farm"] == farm]
        stats[farm] = {}

        for subsystem in ALL_SUBSYSTEMS:
            stats[farm][subsystem] = {}
            for feat in KEY_FEATURES:
                col = f"{subsystem}_{feat}"
                if col not in df.columns:
                    continue
                values = farm_normal[col].dropna()
                if len(values) < 2:
                    continue
                stats[farm][subsystem][feat] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }

    return stats


def compute_subsystem_score(
    event_features: dict,
    subsystem: str,
    normal_stats: dict,
) -> float:
    """Compute anomaly score for a single subsystem using z-scores.

    Parameters
    ----------
    event_features : dict
        Feature values for a single event (column_name -> value).
    subsystem : str
        Subsystem name (e.g. 'gearbox').
    normal_stats : dict
        Normal statistics for this farm+subsystem: {feature: {mean, std}}.

    Returns
    -------
    float
        Subsystem anomaly score: average |z-score| across key features,
        clipped to [0, 10]. Returns NaN if no features are available.
    """
    z_scores = []

    for feat in KEY_FEATURES:
        col = f"{subsystem}_{feat}"
        value = event_features.get(col)

        # Skip if value is missing or NaN
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        # Skip if no normal stats for this feature
        if feat not in normal_stats or not normal_stats[feat]:
            continue

        mean = normal_stats[feat]["mean"]
        std = normal_stats[feat]["std"]

        # Guard against zero/tiny std
        if std < 1e-8:
            z = abs(value - mean)
        else:
            z = abs(value - mean) / std

        z_scores.append(z)

    if not z_scores:
        return float("nan")

    score = np.mean(z_scores)
    return float(np.clip(score, 0.0, 10.0))


def compute_aggregated_score(
    event_features: dict,
    farm: str,
    normal_stats: dict,
    model_quality: dict,
) -> tuple:
    """Compute weighted aggregated anomaly score across subsystems.

    Parameters
    ----------
    event_features : dict
        Feature values for a single event.
    farm : str
        Farm letter (e.g. 'A', 'B', 'C').
    normal_stats : dict
        Normal statistics for this farm: {subsystem: {feature: {mean, std}}}.
    model_quality : dict
        R² scores for this farm: {subsystem: r2_value}.

    Returns
    -------
    tuple of (float, dict)
        - aggregated_score: weighted average of subsystem scores
        - subsystem_scores: {subsystem: score} for each scored subsystem
    """
    subsystem_scores = {}
    weights = {}

    for subsystem in ALL_SUBSYSTEMS:
        # Skip if no normal stats for this subsystem
        if subsystem not in normal_stats or not normal_stats[subsystem]:
            continue

        score = compute_subsystem_score(
            event_features, subsystem, normal_stats[subsystem]
        )

        if np.isnan(score):
            continue

        subsystem_scores[subsystem] = score

        # Combined weight = TDI subsystem weight * max(R², 0.1)
        tdi_weight = SUBSYSTEM_WEIGHTS.get(subsystem, 0.1)
        r2 = model_quality.get(subsystem, 0.1)
        quality_weight = max(r2, 0.1)
        weights[subsystem] = tdi_weight * quality_weight

    if not subsystem_scores:
        return 0.0, {}

    # Weighted average (renormalize weights to sum to 1)
    total_weight = sum(weights.values())
    if total_weight < 1e-8:
        # Fallback to equal weights
        aggregated = np.mean(list(subsystem_scores.values()))
    else:
        aggregated = sum(
            subsystem_scores[s] * weights[s] / total_weight
            for s in subsystem_scores
        )

    return float(aggregated), subsystem_scores


def score_all_events(project_root: str | Path) -> pd.DataFrame:
    """Score every event in the feature matrix.

    Parameters
    ----------
    project_root : str or Path
        Project root directory.

    Returns
    -------
    pd.DataFrame
        Columns: farm, event_id, event_label, aggregated_score,
        plus per-subsystem score columns (e.g. gearbox_score).
    """
    project_root = Path(project_root)

    # Load feature matrix
    fm_path = project_root / "data" / "processed" / "event_feature_matrix.parquet"
    df = pd.read_parquet(fm_path)
    print(f"Loaded feature matrix: {df.shape[0]} events, {df.shape[1]} columns")

    # Compute normal stats from normal events
    normal_stats = compute_normal_stats(df)

    # Load model quality (R² scores)
    r2_map = _load_r2_scores(project_root)
    print(f"Loaded R² scores for farms: {list(r2_map.keys())}")

    # Score each event
    rows = []
    for _, event in df.iterrows():
        farm = event["farm"]
        event_features = event.to_dict()
        farm_stats = normal_stats.get(farm, {})
        farm_r2 = r2_map.get(farm, {})

        agg_score, sub_scores = compute_aggregated_score(
            event_features, farm, farm_stats, farm_r2
        )

        row = {
            "farm": farm,
            "event_id": int(event["event_id"]),
            "event_label": event["event_label"],
            "aggregated_score": agg_score,
        }
        for subsystem in ALL_SUBSYSTEMS:
            row[f"{subsystem}_score"] = sub_scores.get(subsystem, float("nan"))

        rows.append(row)

    result = pd.DataFrame(rows)
    print(f"Scored {len(result)} events")
    return result
