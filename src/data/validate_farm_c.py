"""
Phase 1.1 — Validate all Wind Farm C datasets.
Checks column count, data types, missing values, train_test values, status_type_id range.
Saves results to outputs/reports/farm_c_validation.json
"""

import os
import json
import numpy as np
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

DATA_DIR = r"D:\Personal Projects\Enbridge Case Compettion\data\raw\CARE_To_Compare\Wind Farm C\datasets"
OUTPUT_PATH = r"D:\Personal Projects\Enbridge Case Compettion\outputs\reports\farm_c_validation.json"

EXPECTED_COLS = 957
METADATA_COLS = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
VALID_STATUS_TYPES = {0, 1, 2, 3, 4, 5}
VALID_TRAIN_TEST = {"train", "prediction"}


def validate_file(filepath):
    filename = os.path.basename(filepath)
    result = {"filename": filename, "valid": True, "issues": []}

    try:
        df = pd.read_csv(filepath, sep=";")
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Failed to load: {e}")
        result["rows"] = 0
        result["columns"] = 0
        result["missing_values_total"] = 0
        result["status_types_found"] = []
        result["train_test_values"] = []
        return result

    result["rows"] = len(df)
    result["columns"] = len(df.columns)

    # Check column count
    if len(df.columns) != EXPECTED_COLS:
        result["valid"] = False
        result["issues"].append(f"Expected {EXPECTED_COLS} columns, got {len(df.columns)}")

    # Check data types — non-metadata columns should be numeric
    non_numeric_cols = []
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
    if non_numeric_cols:
        result["valid"] = False
        result["issues"].append(f"{len(non_numeric_cols)} non-numeric data columns: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}")

    # Missing values
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    result["missing_values_total"] = total_missing
    if total_missing > 0:
        cols_with_missing = int((missing > 0).sum())
        result["missing_values_by_column_count"] = cols_with_missing
        # top 10 columns with most missing
        top_missing = missing[missing > 0].sort_values(ascending=False).head(10)
        result["top_missing_columns"] = {col: int(val) for col, val in top_missing.items()}

    # Check train_test values
    if "train_test" in df.columns:
        tt_vals = set(df["train_test"].dropna().unique())
        result["train_test_values"] = sorted(list(tt_vals))
        if not tt_vals.issubset(VALID_TRAIN_TEST):
            result["valid"] = False
            result["issues"].append(f"Unexpected train_test values: {tt_vals - VALID_TRAIN_TEST}")
    else:
        result["train_test_values"] = []
        result["valid"] = False
        result["issues"].append("Missing train_test column")

    # Check status_type_id values
    if "status_type_id" in df.columns:
        st_vals = set(df["status_type_id"].dropna().unique().astype(int))
        result["status_types_found"] = sorted(list(st_vals))
        if not st_vals.issubset(VALID_STATUS_TYPES):
            result["valid"] = False
            result["issues"].append(f"Unexpected status_type_id values: {st_vals - VALID_STATUS_TYPES}")
    else:
        result["status_types_found"] = []
        result["valid"] = False
        result["issues"].append("Missing status_type_id column")

    return result


def main():
    csv_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    print(f"Found {len(csv_files)} CSV files in Wind Farm C datasets/\n")

    files_results = []
    all_issues = []

    for i, fname in enumerate(csv_files, 1):
        fpath = os.path.join(DATA_DIR, fname)
        print(f"[{i}/{len(csv_files)}] Validating {fname}...", end=" ")
        res = validate_file(fpath)
        files_results.append(res)

        if res["valid"]:
            print(f"OK  ({res['rows']} rows, {res['missing_values_total']} missing)")
        else:
            print(f"ISSUES  ({res['rows']} rows, issues: {res['issues']})")
            for issue in res["issues"]:
                all_issues.append(f"{fname}: {issue}")

    all_valid = all(f["valid"] for f in files_results)

    if all_valid:
        summary = f"All {len(csv_files)} files passed validation."
    else:
        invalid_count = sum(1 for f in files_results if not f["valid"])
        summary = f"{invalid_count}/{len(csv_files)} files had issues. Details: " + "; ".join(all_issues)

    output = {
        "farm": "C",
        "total_files": len(csv_files),
        "files": files_results,
        "all_valid": all_valid,
        "summary": summary,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {summary}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
