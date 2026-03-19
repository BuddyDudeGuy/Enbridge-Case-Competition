"""
Build unified event catalog from all three wind farms.
Harmonizes column names and adds farm labels.
"""

import pandas as pd
import json
from pathlib import Path

BASE = Path("D:/Personal Projects/Enbridge Case Compettion")
RAW = BASE / "data" / "raw" / "CARE_To_Compare"

FARMS = {
    "A": RAW / "Wind Farm A" / "event_info.csv",
    "B": RAW / "Wind Farm B" / "event_info.csv",
    "C": RAW / "Wind Farm C" / "event_info.csv",
}

# Load and harmonize
frames = []
for farm, path in FARMS.items():
    df = pd.read_csv(path, sep=";")
    # Farm A uses 'asset', B and C use 'asset_id'
    if "asset" in df.columns and "asset_id" not in df.columns:
        df = df.rename(columns={"asset": "asset_id"})
    df["farm"] = farm
    frames.append(df)

unified = pd.concat(frames, ignore_index=True)

# Save unified CSV (comma-separated for processed files)
out_csv = BASE / "data" / "processed" / "unified_events.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
unified.to_csv(out_csv, index=False)
print(f"Saved unified events to {out_csv}")
print(f"Shape: {unified.shape}")
print(f"\nColumns: {list(unified.columns)}")
print(f"\nEvent labels:\n{unified['event_label'].value_counts()}")
print(f"\nEvents per farm:\n{unified.groupby('farm')['event_id'].count()}")
print(f"\nSample rows:")
print(unified.head(10).to_string())

# Build summary
anomaly_count = int((unified["event_label"] == "anomaly").sum())
normal_count = int((unified["event_label"] == "normal").sum())

per_farm = {}
for farm in ["A", "B", "C"]:
    fdf = unified[unified["farm"] == farm]
    per_farm[farm] = {
        "events": int(len(fdf)),
        "anomalies": int((fdf["event_label"] == "anomaly").sum()),
        "normals": int((fdf["event_label"] == "normal").sum()),
        "unique_assets": int(fdf["asset_id"].nunique()),
    }

fault_dist = unified["event_description"].value_counts().to_dict()
# Convert any numpy int keys/values to plain Python types
fault_dist = {str(k): int(v) for k, v in fault_dist.items()}

summary = {
    "total_events": int(len(unified)),
    "anomaly_count": anomaly_count,
    "normal_count": normal_count,
    "per_farm": per_farm,
    "fault_type_distribution": fault_dist,
    "unique_assets_per_farm": {farm: per_farm[farm]["unique_assets"] for farm in ["A", "B", "C"]},
}

out_json = BASE / "outputs" / "reports" / "unified_events_summary.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {out_json}")
print(json.dumps(summary, indent=2))
