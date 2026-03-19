"""
Phase 1.1 — Validate all CARE to Compare datasets across Wind Farms A, B, C.
Checks column counts, data types, missing values, train_test values, status_type_id range.
Saves per-farm results to outputs/reports/farm_{a,b,c}_validation.json

Usage: py src/data/validate_datasets.py
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(BASE_DIR, "data", "raw", "CARE_To_Compare")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

FARMS = {
    "A": {"path": os.path.join(DATA_ROOT, "Wind Farm A", "datasets"), "expected_cols": 86},
    "B": {"path": os.path.join(DATA_ROOT, "Wind Farm B", "datasets"), "expected_cols": 257},
    "C": {"path": os.path.join(DATA_ROOT, "Wind Farm C", "datasets"), "expected_cols": 957},
}

METADATA_COLS = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
VALID_STATUS_TYPES = {0, 1, 2, 3, 4, 5}
VALID_TRAIN_TEST = {"train", "prediction"}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def validate_file(filepath, expected_cols):
    filename = os.path.basename(filepath)
    result = {"filename": filename, "valid": True, "issues": []}

    try:
        df = pd.read_csv(filepath, sep=";")
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Failed to load: {e}")
        result.update({"rows": 0, "columns": 0, "missing_values_total": 0,
                        "status_types_found": [], "train_test_values": []})
        return result

    result["rows"] = len(df)
    result["columns"] = len(df.columns)

    # column count
    if len(df.columns) != expected_cols:
        result["valid"] = False
        result["issues"].append(f"Expected {expected_cols} columns, got {len(df.columns)}")

    # data types
    non_numeric = [c for c in df.columns if c not in METADATA_COLS and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        result["valid"] = False
        result["issues"].append(f"{len(non_numeric)} non-numeric data columns: {non_numeric[:5]}")

    # missing values
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    result["missing_values_total"] = total_missing
    if total_missing > 0:
        top = missing[missing > 0].sort_values(ascending=False).head(5)
        result["top_missing_columns"] = {col: int(v) for col, v in top.items()}

    # train_test
    tt_vals = set(df["train_test"].dropna().unique()) if "train_test" in df.columns else set()
    result["train_test_values"] = sorted(list(tt_vals))
    if not tt_vals.issubset(VALID_TRAIN_TEST):
        result["valid"] = False
        result["issues"].append(f"Unexpected train_test values: {tt_vals - VALID_TRAIN_TEST}")

    # status_type_id
    if "status_type_id" in df.columns:
        st_vals = set(df["status_type_id"].dropna().unique().astype(int))
        result["status_types_found"] = sorted(list(st_vals))
        if not st_vals.issubset(VALID_STATUS_TYPES):
            result["valid"] = False
            result["issues"].append(f"Unexpected status_type_id: {st_vals - VALID_STATUS_TYPES}")
    else:
        result["status_types_found"] = []
        result["valid"] = False
        result["issues"].append("Missing status_type_id column")

    return result


def validate_farm(farm_name, config):
    data_dir = config["path"]
    expected_cols = config["expected_cols"]

    csv_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    print(f"\n{'='*60}")
    print(f"WIND FARM {farm_name} — {len(csv_files)} datasets, expecting {expected_cols} columns")
    print(f"{'='*60}")

    results = []
    for i, fname in enumerate(csv_files, 1):
        fpath = os.path.join(data_dir, fname)
        print(f"  [{i:>2}/{len(csv_files)}] {fname}...", end=" ")
        res = validate_file(fpath, expected_cols)
        results.append(res)

        status = "OK" if res["valid"] else "ISSUES"
        missing_str = f", {res['missing_values_total']} missing" if res["missing_values_total"] > 0 else ""
        print(f"{status}  ({res['rows']} rows{missing_str})")

        if res["issues"]:
            for issue in res["issues"]:
                print(f"           -> {issue}")

    all_valid = all(r["valid"] for r in results)
    total_rows = sum(r["rows"] for r in results)
    total_missing = sum(r["missing_values_total"] for r in results)

    summary = (f"All {len(csv_files)} files valid." if all_valid
               else f"{sum(1 for r in results if not r['valid'])}/{len(csv_files)} files had issues.")

    output = {
        "farm": farm_name,
        "total_files": len(csv_files),
        "total_rows": total_rows,
        "total_missing_values": total_missing,
        "expected_columns": expected_cols,
        "all_valid": all_valid,
        "summary": summary,
        "files": results,
    }

    out_path = os.path.join(REPORTS_DIR, f"farm_{farm_name.lower()}_validation.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)

    print(f"\n  RESULT: {summary}")
    print(f"  Total rows: {total_rows:,} | Missing values: {total_missing:,}")
    print(f"  Saved: {out_path}")

    return output


if __name__ == "__main__":
    print("Phase 1.1 — Dataset Validation")
    print("=" * 60)

    all_results = {}
    for farm_name, config in FARMS.items():
        all_results[farm_name] = validate_farm(farm_name, config)

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for farm, res in all_results.items():
        status = "PASS" if res["all_valid"] else "FAIL"
        print(f"  Farm {farm}: {res['total_files']} files, {res['total_rows']:,} rows, "
              f"{res['total_missing_values']:,} missing — [{status}]")
