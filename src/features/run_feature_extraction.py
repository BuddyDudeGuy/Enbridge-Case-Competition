"""
Phase 4.3 — Run sliding window feature extraction for all events.

Produces:
  - data/processed/event_feature_matrix.parquet
  - outputs/reports/event_feature_matrix.csv

Usage:
    py src/features/run_feature_extraction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.sliding_window_features import extract_all_events


def main():
    print("=" * 60)
    print("Phase 4.3 — Sliding Window Feature Extraction")
    print("=" * 60)
    print()

    # ---- Extract features for all events ----
    print("Extracting sliding-window features from residuals...")
    feature_df = extract_all_events(PROJECT_ROOT)
    print(f"  Done. Shape: {feature_df.shape}")
    print()

    # ---- Save outputs ----
    parquet_path = PROJECT_ROOT / "data" / "processed" / "event_feature_matrix.parquet"
    csv_path = PROJECT_ROOT / "outputs" / "reports" / "event_feature_matrix.csv"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet: {parquet_path}")

    feature_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV:     {csv_path}")
    print()

    # ---- Summary statistics ----
    meta_cols = ["farm", "event_id", "event_label"]
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    n_events = len(feature_df)
    n_features = len(feat_cols)

    print(f"Total events:           {n_events}")
    print(f"Features per event:     {n_features}")
    print(f"Feature matrix shape:   {feature_df.shape}")
    print()

    # ---- Label distribution ----
    label_counts = feature_df["event_label"].value_counts()
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print()

    # ---- NaN report ----
    nan_pct = feature_df[feat_cols].isna().mean() * 100
    nan_cols = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if len(nan_cols) > 0:
        print(f"Features with NaN ({len(nan_cols)} / {n_features}):")
        for col, pct in nan_cols.head(10).items():
            print(f"  {col}: {pct:.1f}% NaN")
        if len(nan_cols) > 10:
            print(f"  ... and {len(nan_cols) - 10} more")
    else:
        print("No NaN values in any features.")
    print()

    # ---- Top 5 features by normal vs anomaly difference ----
    print("Top 5 features with largest normal-vs-anomaly difference:")
    print("(by absolute difference of means)")
    print()

    anomaly = feature_df[feature_df["event_label"] == "anomaly"]
    normal = feature_df[feature_df["event_label"] == "normal"]

    diffs = {}
    for col in feat_cols:
        anomaly_mean = anomaly[col].mean()
        normal_mean = normal[col].mean()
        if not (np.isnan(anomaly_mean) or np.isnan(normal_mean)):
            diffs[col] = abs(anomaly_mean - normal_mean)

    top_features = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (col, diff) in enumerate(top_features, 1):
        a_mean = anomaly[col].mean()
        n_mean = normal[col].mean()
        print(f"  {i}. {col}")
        print(f"     anomaly mean: {a_mean:.4f} | normal mean: {n_mean:.4f} | diff: {diff:.4f}")

    print()

    # ---- Sample rows ----
    print("Sample rows (first 5):")
    print(feature_df.head().to_string())


if __name__ == "__main__":
    main()
