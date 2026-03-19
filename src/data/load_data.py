"""
Reusable data loading module for the Enbridge Wind Turbine project.

All downstream phases (EDA, modeling, dashboard) should import from here
instead of manually constructing paths and reading CSVs.

Usage:
    from src.data.load_data import load_event, load_event_train, load_event_prediction
"""

from pathlib import Path
import pandas as pd
import time

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def get_data_root() -> Path:
    """Return the project data root (the `data/` directory).

    Walks up from this file's location until it finds a directory that
    contains a `data/` subdirectory.  This avoids hard-coding absolute
    paths and works regardless of how the module is imported.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety limit
        candidate = current / "data"
        # Look for data/raw to distinguish project data/ from src/data/
        if candidate.is_dir() and (candidate / "raw").is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "Could not locate the project data/ directory. "
        "Make sure the repo structure is intact."
    )


_DATA_ROOT: Path = get_data_root()

# ---------------------------------------------------------------------------
# Module-level cache  (dict, not lru_cache — DataFrames aren't hashable)
# ---------------------------------------------------------------------------

_event_cache: dict[tuple[str, int], pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_event(farm: str, event_id: int, cache: bool = True) -> pd.DataFrame:
    """Load a single event dataset CSV.

    Parameters
    ----------
    farm : str
        Farm letter — "A", "B", or "C".
    event_id : int
        Integer event ID matching the CSV filename (e.g. 0 → 0.csv).
    cache : bool, default True
        If True, cache the DataFrame in memory so repeated calls are free.

    Returns
    -------
    pd.DataFrame
    """
    farm = farm.upper()
    key = (farm, event_id)

    if cache and key in _event_cache:
        return _event_cache[key]

    csv_path = (
        _DATA_ROOT / "raw" / "CARE_To_Compare"
        / f"Wind Farm {farm}" / "datasets" / f"{event_id}.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    if cache:
        _event_cache[key] = df

    return df


def load_event_train(farm: str, event_id: int) -> pd.DataFrame:
    """Load the *training* portion of an event.

    Filters to rows where ``train_test == "train"`` AND
    ``status_type_id`` is 0 (Normal) or 2 (Idling).
    """
    df = load_event(farm, event_id)
    mask = (df["train_test"] == "train") & (df["status_type_id"].isin([0, 2]))
    return df.loc[mask].reset_index(drop=True)


def load_event_prediction(farm: str, event_id: int) -> pd.DataFrame:
    """Load the *prediction* portion of an event.

    Filters to rows where ``train_test == "prediction"``.
    """
    df = load_event(farm, event_id)
    return df.loc[df["train_test"] == "prediction"].reset_index(drop=True)


def load_farm_training_data(farm: str) -> pd.DataFrame:
    """Load the combined training parquet for a farm.

    Reads from ``data/processed/training/farm_{farm}_train.parquet``.
    """
    farm = farm.upper()
    parquet_path = _DATA_ROOT / "processed" / "training" / f"farm_{farm.lower()}_train.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_event_info() -> pd.DataFrame:
    """Load the unified event catalog.

    Reads from ``data/processed/unified_events.csv``.
    """
    csv_path = _DATA_ROOT / "processed" / "unified_events.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Unified events file not found: {csv_path}")
    return pd.read_csv(csv_path)


def get_event_ids(farm: str, label: str | None = None) -> list[int]:
    """Return a list of event IDs for a given farm.

    Parameters
    ----------
    farm : str
        Farm letter — "A", "B", or "C".
    label : str or None
        Optional filter: "anomaly", "normal", or None (all events).

    Returns
    -------
    list[int]
    """
    farm = farm.upper()
    events = load_event_info()
    mask = events["farm"] == farm
    if label is not None:
        mask = mask & (events["event_label"] == label.lower())
    return sorted(events.loc[mask, "event_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Drop all cached DataFrames."""
    _event_cache.clear()


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== load_data.py smoke test ===\n")

    # 1. Load event 0 from Farm A
    df_full = load_event("A", 0)
    print(f"1) load_event('A', 0)        -> shape: {df_full.shape}")

    # 2. Training portion
    df_train = load_event_train("A", 0)
    print(f"2) load_event_train('A', 0)  -> shape: {df_train.shape}")

    # 3. Prediction portion
    df_pred = load_event_prediction("A", 0)
    print(f"3) load_event_prediction('A', 0) -> shape: {df_pred.shape}")

    # 4. Combined Farm A training parquet
    df_farm = load_farm_training_data("A")
    print(f"4) load_farm_training_data('A')  -> shape: {df_farm.shape}")

    # 5. Anomaly event IDs for Farm A
    anomaly_ids = get_event_ids("A", label="anomaly")
    print(f"5) get_event_ids('A', 'anomaly') -> {anomaly_ids}")

    # 6. Cache verification
    t0 = time.perf_counter()
    _ = load_event("A", 0)
    elapsed = time.perf_counter() - t0
    print(f"6) Second load_event('A', 0)     -> {elapsed*1000:.2f} ms (cached)")

    print("\nAll checks passed.")
