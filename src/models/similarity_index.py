"""
Vector similarity index over autoencoder embeddings.

Enables "Similar Fault Finder": given a new turbine event state,
find the most similar historical events across all farms.

Uses cosine similarity on 32-dim mean embeddings (one per event,
computed from prediction-window sequences only).
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def build_event_embeddings(project_root: str) -> pd.DataFrame:
    """
    Load all event embeddings and compute mean prediction-window embedding per event.

    For each event across farms A/B/C:
      - Load the full embedding matrix (n_sequences, 32)
      - Load metadata to get prediction_sequence_indices
      - Slice to prediction-window sequences only
      - Compute the mean across those sequences -> one 32-dim vector

    Returns:
        DataFrame with columns: farm, event_id, event_label, event_description,
        emb_0..emb_31 (95 rows, one per event)
    """
    project_root = Path(project_root)
    ae_dir = project_root / "data" / "processed" / "ae_outputs"

    rows = []
    farm_map = {"A": "farm_a", "B": "farm_b", "C": "farm_c"}

    # Load unified events for descriptions
    events_df = pd.read_csv(project_root / "data" / "processed" / "unified_events.csv")
    events_lookup = {}
    for _, row in events_df.iterrows():
        events_lookup[(row["farm"], row["event_id"])] = {
            "event_label": row["event_label"],
            "event_description": row["event_description"],
        }

    for farm_letter, farm_folder in farm_map.items():
        farm_dir = ae_dir / farm_folder

        if not farm_dir.exists():
            print(f"  [WARN] Farm directory not found: {farm_dir}")
            continue

        # Find all embedding files in this farm directory
        emb_files = sorted(farm_dir.glob("event_*_embeddings.npy"))

        for emb_path in emb_files:
            # Extract event_id from filename: event_{id}_embeddings.npy
            event_id = int(emb_path.stem.split("_")[1])
            meta_path = farm_dir / f"event_{event_id}_meta.json"

            # Load embeddings
            embeddings = np.load(emb_path)  # (n_sequences, 32)

            # Load metadata for train/prediction split indices
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                pred_indices = meta.get("prediction_sequence_indices", [])
            else:
                print(f"  [WARN] No meta for farm {farm_letter} event {event_id}, "
                      f"using all sequences")
                pred_indices = []

            # Compute mean embedding over prediction window only
            if len(pred_indices) > 0:
                # Ensure indices are within bounds
                valid_indices = [i for i in pred_indices if i < len(embeddings)]
                if len(valid_indices) > 0:
                    mean_emb = embeddings[valid_indices].mean(axis=0)
                else:
                    print(f"  [WARN] No valid pred indices for farm {farm_letter} "
                          f"event {event_id}, using overall mean")
                    mean_emb = embeddings.mean(axis=0)
            else:
                # Fallback: use overall mean if no prediction indices
                print(f"  [WARN] No prediction indices for farm {farm_letter} "
                      f"event {event_id}, using overall mean")
                mean_emb = embeddings.mean(axis=0)

            # Look up event info
            info = events_lookup.get(
                (farm_letter, event_id),
                {"event_label": "unknown", "event_description": "unknown"},
            )

            row = {
                "farm": farm_letter,
                "event_id": event_id,
                "event_label": info["event_label"],
                "event_description": info["event_description"],
            }
            # Add embedding dimensions as separate columns
            for dim in range(len(mean_emb)):
                row[f"emb_{dim}"] = float(mean_emb[dim])

            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by farm and event_id for consistent ordering
    df = df.sort_values(["farm", "event_id"]).reset_index(drop=True)

    return df


def build_similarity_index(event_embeddings_df: pd.DataFrame):
    """
    Fit a NearestNeighbors model using cosine distance on event embeddings.

    Args:
        event_embeddings_df: DataFrame from build_event_embeddings()

    Returns:
        Tuple of (fitted NearestNeighbors model, embeddings numpy array)
    """
    emb_cols = [c for c in event_embeddings_df.columns if c.startswith("emb_")]
    embeddings_matrix = event_embeddings_df[emb_cols].values.astype(np.float32)

    # Cosine distance with brute force (exact) for 95 vectors
    index = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(len(embeddings_matrix), 10),
    )
    index.fit(embeddings_matrix)

    return index, embeddings_matrix


def find_similar_events(
    query_embedding: np.ndarray,
    index: NearestNeighbors,
    event_embeddings_df: pd.DataFrame,
    k: int = 5,
    exclude_event: tuple = None,
) -> list[dict]:
    """
    Find the k most similar events to a query embedding.

    Args:
        query_embedding: 32-dim vector (the query event's mean embedding)
        index: fitted NearestNeighbors model
        event_embeddings_df: DataFrame with event metadata + embeddings
        k: number of similar events to return
        exclude_event: optional (farm, event_id) tuple to exclude from results
            (e.g. the query event itself)

    Returns:
        List of dicts with: farm, event_id, event_label, event_description,
        similarity_score, distance
    """
    query = np.array(query_embedding).reshape(1, -1).astype(np.float32)

    # Fetch k+1 neighbors in case we need to exclude the query itself
    n_fetch = min(k + 1, len(event_embeddings_df))
    distances, indices = index.kneighbors(query, n_neighbors=n_fetch)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = event_embeddings_df.iloc[idx]

        # Skip the query event itself
        if exclude_event is not None:
            if (row["farm"], row["event_id"]) == exclude_event:
                continue

        similarity = 1.0 - float(dist)  # cosine_similarity = 1 - cosine_distance

        results.append({
            "farm": row["farm"],
            "event_id": int(row["event_id"]),
            "event_label": row["event_label"],
            "event_description": row["event_description"],
            "similarity_score": round(similarity, 4),
            "distance": round(float(dist), 4),
        })

        if len(results) >= k:
            break

    return results


def save_index(
    event_embeddings_df: pd.DataFrame,
    index: NearestNeighbors,
    path: str,
) -> None:
    """
    Save the event embeddings DataFrame and fitted index to disk.

    Args:
        event_embeddings_df: DataFrame with embeddings + metadata
        index: fitted NearestNeighbors model
        path: directory to save files to
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save embeddings as parquet (efficient columnar storage)
    emb_path = path / "event_embeddings.parquet"
    event_embeddings_df.to_parquet(emb_path, index=False)
    print(f"  Saved embeddings: {emb_path}")

    # Save fitted index with joblib
    idx_path = path / "similarity_index.joblib"
    joblib.dump(index, idx_path)
    print(f"  Saved index: {idx_path}")


def load_index(path: str):
    """
    Load a previously saved similarity index and embeddings.

    Args:
        path: directory containing saved files

    Returns:
        Tuple of (event_embeddings_df, fitted NearestNeighbors index)
    """
    path = Path(path)

    event_embeddings_df = pd.read_parquet(path / "event_embeddings.parquet")
    index = joblib.load(path / "similarity_index.joblib")

    return event_embeddings_df, index
