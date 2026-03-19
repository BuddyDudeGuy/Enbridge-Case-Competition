"""
Phase 1.7 — Filter SCADA datasets to normal operation rows for NBM training.

Keeps only rows where:
  - train_test == "train"
  - status_type_id in [0, 2]
  - event_label == "normal" (from unified events catalog)

Outputs:
  - Per-event parquets:  data/processed/training/{farm}_{event_id}.parquet
  - Per-farm combined:   data/processed/training/farm_{a,b,c}_train.parquet
  - Summary report:      outputs/reports/training_data_summary.json
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path("D:/Personal Projects/Enbridge Case Compettion")
EVENTS_CSV = ROOT / "data" / "processed" / "unified_events.csv"
RAW_BASE = ROOT / "data" / "raw" / "CARE_To_Compare"
OUT_DIR = ROOT / "data" / "processed" / "training"
REPORT_PATH = ROOT / "outputs" / "reports" / "training_data_summary.json"

FARM_DIRS = {"A": "Wind Farm A", "B": "Wind Farm B", "C": "Wind Farm C"}
NORMAL_STATUS = [0, 2]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── load unified events catalog ──────────────────────────────────────
    events = pd.read_csv(EVENTS_CSV)
    normal_events = events[events["event_label"] == "normal"]
    print(f"Unified events catalog: {len(events)} total, {len(normal_events)} normal\n")

    report = {}
    overall = {"total_events_used": 0, "total_rows_raw": 0,
               "total_rows_filtered": 0}

    for farm in ["A", "B", "C"]:
        farm_events = normal_events[normal_events["farm"] == farm]
        event_ids = sorted(farm_events["event_id"].tolist())
        print(f"=== Farm {farm}: {len(event_ids)} normal events ===")

        farm_frames = []
        farm_raw_total = 0
        farm_filtered_total = 0
        status_counts_total = {}

        for eid in event_ids:
            csv_path = RAW_BASE / FARM_DIRS[farm] / "datasets" / f"{eid}.csv"
            if not csv_path.exists():
                print(f"  [WARN] {csv_path.name} not found, skipping")
                continue

            df = pd.read_csv(csv_path, sep=";")
            raw_rows = len(df)
            farm_raw_total += raw_rows

            # status distribution before filtering
            for st, cnt in df["status_type_id"].value_counts().items():
                status_counts_total[int(st)] = status_counts_total.get(int(st), 0) + int(cnt)

            # filter: train split + normal status
            mask = (df["train_test"] == "train") & (df["status_type_id"].isin(NORMAL_STATUS))
            df_filt = df[mask].copy()
            filt_rows = len(df_filt)
            farm_filtered_total += filt_rows

            # save per-event parquet
            out_path = OUT_DIR / f"{farm}_{eid}.parquet"
            df_filt.to_parquet(out_path, index=False)

            pct = filt_rows / raw_rows * 100 if raw_rows else 0
            print(f"  event {eid:>3d}: {raw_rows:>7,} raw -> {filt_rows:>7,} filtered ({pct:5.1f}%)")

            farm_frames.append(df_filt)

        # combined per-farm parquet
        if farm_frames:
            farm_df = pd.concat(farm_frames, ignore_index=True)
            farm_out = OUT_DIR / f"farm_{farm.lower()}_train.parquet"
            farm_df.to_parquet(farm_out, index=False)
            print(f"  Combined: {farm_out.name} — {len(farm_df):,} rows")
        else:
            print("  [WARN] No data for this farm")

        pct_ret = farm_filtered_total / farm_raw_total * 100 if farm_raw_total else 0
        print(f"  Summary: {farm_raw_total:,} raw -> {farm_filtered_total:,} filtered ({pct_ret:.1f}%)\n")

        report[f"farm_{farm.lower()}"] = {
            "total_events_used": len(event_ids),
            "total_rows_raw": farm_raw_total,
            "total_rows_filtered": farm_filtered_total,
            "pct_retained": round(pct_ret, 2),
            "status_type_distribution": {str(k): v for k, v in sorted(status_counts_total.items())},
        }
        overall["total_events_used"] += len(event_ids)
        overall["total_rows_raw"] += farm_raw_total
        overall["total_rows_filtered"] += farm_filtered_total

    overall["pct_retained"] = round(
        overall["total_rows_filtered"] / overall["total_rows_raw"] * 100
        if overall["total_rows_raw"] else 0, 2
    )
    report["overall"] = overall

    # ── save report ──────────────────────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {REPORT_PATH}")
    print(f"\n{'='*50}")
    print(f"OVERALL: {overall['total_rows_raw']:,} raw -> {overall['total_rows_filtered']:,} filtered ({overall['pct_retained']}%)")
    print(f"Events used: {overall['total_events_used']}")


if __name__ == "__main__":
    main()
