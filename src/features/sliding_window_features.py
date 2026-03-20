"""
Phase 4.3 — Sliding window feature extraction from NBM residual time-series.

Extracts rolling-window summary statistics and global prediction-window features
from each subsystem's residual series. These features characterize residual
behavior over different time horizons (1h, 6h, 24h) and will feed into
event-level classification (Phase 4.4).

Feature naming convention:
    {subsystem}_mean_1h       — rolling 1h mean (last value)
    {subsystem}_std_6h        — rolling 6h std (last value)
    {subsystem}_max_24h       — rolling 24h max absolute (last value)
    {subsystem}_overall_mean  — prediction window mean
    {subsystem}_overall_std   — prediction window std
    {subsystem}_overall_max   — prediction window max absolute
    {subsystem}_trend_slope   — linear regression slope over prediction window
    {subsystem}_anomaly_frac  — fraction of timesteps with |residual| > 2*std
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Window sizes in 10-minute intervals
WINDOW_MAP = {
    6: "1h",    # 6 * 10min = 1 hour
    36: "6h",   # 36 * 10min = 6 hours
    144: "24h", # 144 * 10min = 24 hours
}


def extract_window_features(
    residual_series: pd.Series,
    windows: list[int] | None = None,
) -> dict[str, float]:
    """Extract rolling-window and global features from a single residual series.

    Parameters
    ----------
    residual_series : pd.Series
        Residual values (actual - predicted) for one subsystem in the
        prediction window.  Index is assumed to be sequential (10-min ticks).
    windows : list[int], optional
        Rolling window sizes in number of 10-min timesteps.
        Default: [6, 36, 144] for 1h, 6h, 24h.

    Returns
    -------
    dict[str, float]
        Feature name -> value. Names are suffixed with the time horizon
        (e.g. 'mean_1h', 'std_6h', etc.) plus global features.
    """
    if windows is None:
        windows = [6, 36, 144]

    features: dict[str, float] = {}
    series = residual_series.dropna()

    # If entirely empty / all NaN, return NaN for everything
    if len(series) == 0:
        for w in windows:
            label = WINDOW_MAP.get(w, f"{w}step")
            features[f"mean_{label}"] = np.nan
            features[f"std_{label}"] = np.nan
            features[f"max_{label}"] = np.nan
        features["overall_mean"] = np.nan
        features["overall_std"] = np.nan
        features["overall_max"] = np.nan
        features["trend_slope"] = np.nan
        features["anomaly_frac"] = np.nan
        return features

    # ---- Rolling window features (take the LAST value) ----
    for w in windows:
        label = WINDOW_MAP.get(w, f"{w}step")

        rolling_mean = series.rolling(window=w, min_periods=1).mean()
        features[f"mean_{label}"] = float(rolling_mean.iloc[-1])

        rolling_std = series.rolling(window=w, min_periods=1).std()
        features[f"std_{label}"] = float(rolling_std.iloc[-1])

        rolling_max_abs = series.abs().rolling(window=w, min_periods=1).max()
        features[f"max_{label}"] = float(rolling_max_abs.iloc[-1])

    # ---- Global prediction-window features ----
    features["overall_mean"] = float(series.mean())
    features["overall_std"] = float(series.std())
    features["overall_max"] = float(series.abs().max())

    # Trend slope via numpy polyfit (degree 1)
    x = np.arange(len(series), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slope, _ = np.polyfit(x, series.values.astype(np.float64), 1)
            features["trend_slope"] = float(slope)
        except (np.linalg.LinAlgError, ValueError):
            features["trend_slope"] = np.nan

    # Anomaly fraction: percentage of timesteps where |residual| > 2*std
    overall_std = features["overall_std"]
    if overall_std > 0 and not np.isnan(overall_std):
        threshold = 2.0 * overall_std
        features["anomaly_frac"] = float((series.abs() > threshold).mean())
    else:
        features["anomaly_frac"] = 0.0

    return features


def extract_event_features(
    event_parquet_path: str | Path,
    subsystems: list[str],
) -> dict[str, float]:
    """Extract sliding-window features for all subsystems in one event.

    Parameters
    ----------
    event_parquet_path : str or Path
        Path to the event parquet file (e.g., .../farm_a/event_0.parquet).
    subsystems : list[str]
        Subsystem names to look for (e.g., ['gearbox', 'generator_bearings', ...]).
        The residual column is expected to be named '{subsystem}_residual'.

    Returns
    -------
    dict[str, float]
        Flat dict: '{subsystem}_{feature_name}' -> value.
        Subsystems not present in the parquet get NaN-filled features.
    """
    df = pd.read_parquet(event_parquet_path)

    # Filter to prediction window only
    pred = df[df["train_test"] == "prediction"].copy()

    all_features: dict[str, float] = {}

    for sub in subsystems:
        col = f"{sub}_residual"
        if col in pred.columns:
            residual_series = pred[col].reset_index(drop=True)
            sub_features = extract_window_features(residual_series)
        else:
            # Subsystem not modeled for this farm — NaN fill
            sub_features = extract_window_features(pd.Series(dtype=float))

        # Prefix with subsystem name
        for feat_name, value in sub_features.items():
            all_features[f"{sub}_{feat_name}"] = value

    return all_features


def extract_all_events(project_root: str | Path) -> pd.DataFrame:
    """Extract sliding-window features for all 95 events across all farms.

    Parameters
    ----------
    project_root : str or Path
        Root directory of the project (contains data/, src/, etc.).

    Returns
    -------
    pd.DataFrame
        One row per event, columns = features + metadata (farm, event_id, event_label).
    """
    project_root = Path(project_root)
    residual_dir = project_root / "data" / "processed" / "residuals"
    events_csv = project_root / "data" / "processed" / "unified_events.csv"

    # Load unified events
    events_df = pd.read_csv(events_csv)

    # All possible subsystems (union across farms)
    all_subsystems = [
        "gearbox",
        "generator_bearings",
        "transformer",
        "hydraulic",
        "cooling",
    ]

    rows = []
    for _, event_row in events_df.iterrows():
        event_id = int(event_row["event_id"])
        farm = event_row["farm"].strip()
        farm_key = f"farm_{farm.lower()}"
        event_label = event_row["event_label"].strip()

        parquet_path = residual_dir / farm_key / f"event_{event_id}.parquet"

        if not parquet_path.exists():
            print(f"  [WARN] Missing parquet: {parquet_path}")
            continue

        features = extract_event_features(parquet_path, all_subsystems)

        # Add metadata
        features["farm"] = farm
        features["event_id"] = event_id
        features["event_label"] = event_label

        rows.append(features)

    result_df = pd.DataFrame(rows)

    # Reorder: metadata first, then feature columns
    meta_cols = ["farm", "event_id", "event_label"]
    feat_cols = [c for c in result_df.columns if c not in meta_cols]
    feat_cols.sort()
    result_df = result_df[meta_cols + feat_cols]

    return result_df
