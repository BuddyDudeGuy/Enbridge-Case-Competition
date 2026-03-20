"""
Autoencoder data preparation for LSTM-based anomaly detection.

Phase 5.1 & 5.2: Convert raw thermal sensor data into sliding-window
sequences normalized using only normal-operation training statistics.

One autoencoder per farm — each uses a curated set of ~15-20 thermal
sensors spanning all available subsystems.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Sensor selection per farm
# ---------------------------------------------------------------------------
# Prioritizes NBM target sensors, then adds 1-2 extras per subsystem for
# richer multivariate signal.  Farm C is heavily pruned from 58 → 20.

AUTOENCODER_SENSORS = {
    "farm_a": [
        # gearbox (2/2) — both available
        "sensor_11_avg",   # gearbox bearing temp
        "sensor_12_avg",   # gearbox oil temp  [NBM target]
        # generator_bearings (5/5) — all available
        "sensor_13_avg",   # generator bearing 2 DE  [NBM target]
        "sensor_14_avg",   # generator bearing 1
        "sensor_15_avg",   # generator bearing 3
        "sensor_16_avg",   # generator bearing 4
        "sensor_17_avg",   # generator winding
        # transformer (3/3)
        "sensor_38_avg",   # HV transformer L1  [NBM target]
        "sensor_39_avg",   # HV transformer L2
        "sensor_40_avg",   # HV transformer L3
        # hydraulic (1/1)
        "sensor_41_avg",   # hydraulic oil  [NBM target]
        # cooling (3/3)
        "sensor_8_avg",    # cooling water 1
        "sensor_9_avg",    # cooling water 2
        "sensor_10_avg",   # VCS cooling water  [NBM target]
        # nacelle_ambient (5/5)
        "sensor_0_avg",    # nacelle temp
        "sensor_6_avg",    # ambient temp
        "sensor_7_avg",    # nacelle interior
        "sensor_43_avg",   # nacelle misc
        "sensor_53_avg",   # nacelle misc 2
    ],  # 19 total — all Farm A thermal sensors

    "farm_b": [
        # gearbox (6/6)
        "sensor_34_avg",   # gearbox bearing 1
        "sensor_35_avg",   # gearbox bearing 2
        "sensor_36_avg",   # gearbox bearing 3
        "sensor_37_avg",   # gearbox oil inlet
        "sensor_38_avg",   # gearbox oil outlet
        "sensor_39_avg",   # gearbox oil tank  [NBM target]
        # generator_bearings (5/5)
        "sensor_32_avg",   # generator bearing 1  [NBM target]
        "sensor_33_avg",   # generator bearing 2
        "sensor_51_avg",   # generator winding 1
        "sensor_52_avg",   # generator winding 2
        "sensor_53_avg",   # generator winding 3
        # transformer (9/9)
        "sensor_31_avg",   # transformer temp 1
        "sensor_40_avg",   # transformer core  [NBM target]
        "sensor_41_avg",   # transformer phase 1
        "sensor_42_avg",   # transformer phase 2
        "sensor_43_avg",   # transformer phase 3
        "sensor_44_avg",   # transformer oil 1
        "sensor_45_avg",   # transformer oil 2
        "sensor_46_avg",   # transformer oil 3
        "sensor_47_avg",   # transformer oil 4
        # nacelle_ambient (2/2)
        "sensor_8_avg",    # nacelle temp
        "sensor_22_avg",   # ambient temp
    ],  # 22 total — all Farm B thermal sensors

    "farm_c": [
        # gearbox (5/20 selected) — NBM target + key bearings & oil
        "sensor_151_avg",  # gearbox bearing 1
        "sensor_155_avg",  # gearbox bearing 5
        "sensor_160_avg",  # gearbox bearing 10
        "sensor_186_avg",  # gearbox oil temp 1  [NBM target]
        "sensor_189_avg",  # gearbox oil temp 3
        # generator_bearings (5/19 selected) — NBM target + spread
        "sensor_18_avg",   # generator winding 1
        "sensor_168_avg",  # generator bearing DE
        "sensor_173_avg",  # generator bearing NDE
        "sensor_196_avg",  # rotor bearing 1  [NBM target]
        "sensor_200_avg",  # rotor bearing 5
        # transformer (3/4 selected)
        "sensor_167_avg",  # transformer temp
        "sensor_191_avg",  # main transformer oil 1  [NBM target]
        "sensor_192_avg",  # main transformer oil 2
        # hydraulic (2/2)
        "sensor_178_avg",  # hydraulic oil tank 1  [NBM target]
        "sensor_179_avg",  # hydraulic oil tank 2
        # cooling (3/9 selected) — NBM target + key temps
        "sensor_175_avg",  # cooling water 1
        "sensor_228_avg",  # cooling water generator inlet 1  [NBM target]
        "sensor_233_avg",  # cooling water 5
        # nacelle_ambient (2/4 selected)
        "sensor_7_avg",    # nacelle temp
        "sensor_177_avg",  # ambient temp
    ],  # 20 total — pruned from 58
}

# Sequence parameters
DEFAULT_WINDOW_SIZE = 36   # 36 x 10 min = 6 hours
DEFAULT_STEP_SIZE = 6      # 6 x 10 min = 1 hour stride


# ---------------------------------------------------------------------------
# Scaler fitting
# ---------------------------------------------------------------------------

def get_scaler(farm: str, project_root: str | Path) -> StandardScaler:
    """Fit a StandardScaler on normal-operation training data for a farm.

    Parameters
    ----------
    farm : str
        One of 'a', 'b', 'c' (case-insensitive).
    project_root : str or Path
        Project root directory.

    Returns
    -------
    StandardScaler
        Fitted scaler for the farm's autoencoder sensors.
    """
    farm = farm.lower()
    project_root = Path(project_root)
    sensor_cols = AUTOENCODER_SENSORS[f"farm_{farm}"]

    # Load training parquet (already filtered to normal operation)
    parquet_path = project_root / "data" / "processed" / "training" / f"farm_{farm}_train.parquet"
    df = pd.read_parquet(parquet_path, columns=sensor_cols)

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df[sensor_cols].values)

    # Save scaler
    model_dir = project_root / "data" / "processed" / "models" / f"farm_{farm}"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "ae_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    return scaler


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

def create_sequences(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> np.ndarray:
    """Create sliding-window sequences from a DataFrame.

    NaN handling: forward-fill then backward-fill within the provided
    DataFrame.  Any remaining NaNs are filled with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (should be a single contiguous segment).
    sensor_columns : list[str]
        Columns to extract.
    window_size : int
        Number of timesteps per sequence.
    step_size : int
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Shape ``(n_sequences, window_size, n_features)``.
        Returns empty array with shape ``(0, window_size, len(sensor_columns))``
        if the input is too short.
    """
    n_features = len(sensor_columns)

    # Extract and fill NaNs
    values = df[sensor_columns].copy()
    values = values.ffill().bfill().fillna(0.0)
    data = values.values  # shape (n_rows, n_features)

    n_rows = len(data)
    if n_rows < window_size:
        return np.empty((0, window_size, n_features), dtype=np.float32)

    # Build sliding windows
    sequences = []
    for start in range(0, n_rows - window_size + 1, step_size):
        sequences.append(data[start : start + window_size])

    return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

_MAX_TRAINING_ROWS = 200_000  # subsample cap per farm for memory

def prepare_training_data(
    farm: str,
    project_root: str | Path,
    max_rows: int = _MAX_TRAINING_ROWS,
) -> tuple[np.ndarray, StandardScaler]:
    """Load training data, fit scaler, normalize, create sequences, save.

    Sequences are created per-asset (turbine) to avoid crossing turbine
    boundaries.  Within each asset's block the data is assumed to be time-
    sorted and contiguous.

    Parameters
    ----------
    farm : str
        One of 'a', 'b', 'c'.
    project_root : str or Path
        Project root directory.
    max_rows : int
        Maximum rows to use (subsampled if exceeded).

    Returns
    -------
    tuple[np.ndarray, StandardScaler]
        (sequences array of shape (N, 36, F), fitted scaler)
    """
    farm = farm.lower()
    project_root = Path(project_root)
    sensor_cols = AUTOENCODER_SENSORS[f"farm_{farm}"]

    print(f"\n{'='*60}")
    print(f"Preparing autoencoder training data for Farm {farm.upper()}")
    print(f"{'='*60}")
    print(f"  Sensors: {len(sensor_cols)}")

    # Load training data
    parquet_path = project_root / "data" / "processed" / "training" / f"farm_{farm}_train.parquet"
    needed_cols = ["asset_id", "time_stamp"] + sensor_cols
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    print(f"  Raw training rows: {len(df):,}")

    # Sort by asset_id then time_stamp for contiguous segments
    df = df.sort_values(["asset_id", "time_stamp"]).reset_index(drop=True)

    # Subsample if too large (sample complete asset blocks)
    if len(df) > max_rows:
        # Random sample of assets, keeping their blocks intact
        rng = np.random.RandomState(42)
        all_assets = df["asset_id"].unique()
        rng.shuffle(all_assets)

        sampled_assets = []
        total = 0
        for aid in all_assets:
            n = (df["asset_id"] == aid).sum()
            if total + n > max_rows:
                break
            sampled_assets.append(aid)
            total += n

        df = df[df["asset_id"].isin(sampled_assets)].reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} rows ({len(sampled_assets)} assets)")

    # Fit scaler on the (potentially subsampled) training data
    scaler = get_scaler(farm, project_root)

    # Normalize sensor columns
    df[sensor_cols] = scaler.transform(df[sensor_cols].values)

    # Create sequences per asset to respect turbine boundaries
    all_sequences = []
    for asset_id, group in df.groupby("asset_id", sort=True):
        group = group.sort_values("time_stamp").reset_index(drop=True)
        seqs = create_sequences(group, sensor_cols)
        if len(seqs) > 0:
            all_sequences.append(seqs)

    if all_sequences:
        X = np.concatenate(all_sequences, axis=0)
    else:
        X = np.empty((0, DEFAULT_WINDOW_SIZE, len(sensor_cols)), dtype=np.float32)

    print(f"  Sequences: {X.shape[0]:,}  |  Shape: {X.shape}")

    # Save sequences
    save_dir = project_root / "data" / "processed" / "ae_sequences"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"farm_{farm}_train_X.npy"
    np.save(save_path, X)
    print(f"  Saved to {save_path}")

    return X, scaler


# ---------------------------------------------------------------------------
# Event data preparation (inference time)
# ---------------------------------------------------------------------------

def prepare_event_data(
    farm: str,
    event_id: int,
    scaler: StandardScaler,
    project_root: str | Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load a single event, normalize with pre-fitted scaler, create sequences.

    Parameters
    ----------
    farm : str
        Farm letter — 'A', 'B', or 'C'.
    event_id : int
        Event ID.
    scaler : StandardScaler
        Pre-fitted scaler from training data.
    project_root : str or Path
        Project root directory.
    window_size : int
        Timesteps per sequence.
    step_size : int
        Stride between windows.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        - sequences: shape (N, window_size, n_features)
        - metadata: DataFrame with columns [seq_idx, start_row, end_row,
          start_time, end_time] mapping each sequence to its source rows.
    """
    import sys
    sys.path.insert(0, str(Path(project_root)))
    from src.data.load_data import load_event

    farm_letter = farm.upper()
    farm_key = f"farm_{farm.lower()}"
    sensor_cols = AUTOENCODER_SENSORS[farm_key]

    # Load full event (train + prediction portions)
    df = load_event(farm_letter, event_id, cache=False)

    # Check which sensor columns are available
    available = [c for c in sensor_cols if c in df.columns]
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} sensors missing for event {event_id}: {missing}")
        # Add missing columns as NaN — they'll be filled with 0 after ffill/bfill
        for col in missing:
            df[col] = np.nan

    # Sort by time
    df = df.sort_values("time_stamp").reset_index(drop=True)

    # NaN handling before scaling
    sensor_data = df[sensor_cols].copy()
    sensor_data = sensor_data.ffill().bfill().fillna(0.0)

    # Normalize using pre-fitted scaler (no data leakage)
    sensor_data[sensor_cols] = scaler.transform(sensor_data[sensor_cols].values)

    # Build a working df with normalized sensors
    df_work = sensor_data.copy()
    df_work["time_stamp"] = df["time_stamp"].values

    # Create sequences
    data = df_work[sensor_cols].values.astype(np.float32)
    n_rows = len(data)

    sequences = []
    meta_rows = []

    if n_rows >= window_size:
        for i, start in enumerate(range(0, n_rows - window_size + 1, step_size)):
            end = start + window_size
            sequences.append(data[start:end])
            meta_rows.append({
                "seq_idx": i,
                "start_row": start,
                "end_row": end - 1,
                "start_time": df["time_stamp"].iloc[start],
                "end_time": df["time_stamp"].iloc[end - 1],
            })

    if sequences:
        X = np.array(sequences, dtype=np.float32)
    else:
        X = np.empty((0, window_size, len(sensor_cols)), dtype=np.float32)

    metadata = pd.DataFrame(meta_rows)

    return X, metadata
