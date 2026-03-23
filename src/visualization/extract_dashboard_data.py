"""Extract chart data from parquet residual files for the Next.js dashboard."""

import json
import pandas as pd

BASE = "D:/Personal Projects/Enbridge Case Compettion"

events_config = [
    {
        "farm": "A",
        "event_id": 72,
        "parquet": f"{BASE}/data/processed/residuals/farm_a/event_72.parquet",
        "cols": ("gearbox_actual", "gearbox_predicted", "gearbox_residual"),
        "sample_every": 20,
        "subsystem": "Gearbox",
        "r2": 0.804,
        "warningDays": 7,
    },
    {
        "farm": "B",
        "event_id": 53,
        "parquet": f"{BASE}/data/processed/residuals/farm_b/event_53.parquet",
        "cols": ("generator_bearings_actual", "generator_bearings_predicted", "generator_bearings_residual"),
        "sample_every": 80,
        "subsystem": "Generator Bearings",
        "r2": 0.50,
        "warningDays": 42,
    },
    {
        "farm": "C",
        "event_id": 44,
        "parquet": f"{BASE}/data/processed/residuals/farm_c/event_44.parquet",
        "cols": ("cooling_actual", "cooling_predicted", "cooling_residual"),
        "sample_every": 80,
        "subsystem": "Cooling",
        "r2": 0.761,
        "warningDays": 65,
    },
]

# Load fault start times
unified = pd.read_csv(f"{BASE}/data/processed/unified_events.csv", sep=",")

result = {}

for cfg in events_config:
    df = pd.read_parquet(cfg["parquet"])
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    # Filter to prediction rows only
    pred = df[df["train_test"] == "prediction"].copy()
    pred = pred.sort_values("time_stamp").reset_index(drop=True)

    first_ts = pred["time_stamp"].iloc[0]

    # Calculate relative days
    pred["day"] = (pred["time_stamp"] - first_ts).dt.total_seconds() / 86400.0

    total_days = round(pred["day"].iloc[-1], 1)

    # Sample every Nth row
    sampled = pred.iloc[:: cfg["sample_every"]].copy()

    actual_col, predicted_col, residual_col = cfg["cols"]

    data_points = []
    for _, row in sampled.iterrows():
        data_points.append({
            "day": round(row["day"], 1),
            "actual": round(row[actual_col], 1),
            "predicted": round(row[predicted_col], 1),
            "residual": round(row[residual_col], 1),
        })

    # Get fault start relative day
    event_row = unified[unified["event_id"] == cfg["event_id"]]
    fault_ts = pd.to_datetime(event_row["event_start"].iloc[0])
    fault_start_day = round((fault_ts - first_ts).total_seconds() / 86400.0, 1)

    result[cfg["farm"]] = {
        "title": f"Farm {cfg['farm']} \u2014 Event {cfg['event_id']}",
        "subtitle": f"{cfg['subsystem']} Temperature Anomaly",
        "subsystem": cfg["subsystem"],
        "r2": cfg["r2"],
        "warningDays": cfg["warningDays"],
        "data": data_points,
        "faultStart": fault_start_day,
        "totalDays": total_days,
    }

    print(f"Farm {cfg['farm']}: {len(data_points)} points, faultStart={fault_start_day}, totalDays={total_days}")

output_path = f"{BASE}/dashboard/data/eventChartData.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nWrote {output_path}")
