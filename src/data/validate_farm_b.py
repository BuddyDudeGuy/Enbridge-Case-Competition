import pandas as pd
import json
import os
import glob

DATA_DIR = "D:/Personal Projects/Enbridge Case Compettion/data/raw/CARE_To_Compare/Wind Farm B/datasets"
OUTPUT_PATH = "D:/Personal Projects/Enbridge Case Compettion/outputs/reports/farm_b_validation.json"
METADATA_COLS = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=lambda f: int(os.path.basename(f).replace(".csv", "")))

files_results = []
issues = []

for filepath in csv_files:
    fname = os.path.basename(filepath)
    print(f"Validating {fname}...")
    df = pd.read_csv(filepath, sep=';')

    n_rows, n_cols = df.shape
    file_valid = True
    file_issues = []

    # Check column count
    if n_cols != 257:
        file_issues.append(f"Expected 257 columns, got {n_cols}")
        file_valid = False

    # Check data types - non-metadata columns should be numeric
    non_numeric = []
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        file_issues.append(f"Non-numeric columns (excluding metadata): {non_numeric}")
        file_valid = False

    # Missing values
    missing = df.isnull().sum()
    missing_total = int(missing.sum())
    missing_by_col = {col: int(v) for col, v in missing.items() if v > 0}

    # train_test values
    tt_values = sorted(df["train_test"].dropna().unique().tolist())
    valid_tt = set(tt_values).issubset({"train", "prediction"})
    if not valid_tt:
        file_issues.append(f"Unexpected train_test values: {tt_values}")
        file_valid = False

    # status_type_id values
    st_values = sorted(df["status_type_id"].dropna().unique().tolist())
    valid_st = all(0 <= v <= 5 for v in st_values)
    if not valid_st:
        file_issues.append(f"status_type_id out of range 0-5: {st_values}")
        file_valid = False

    if file_issues:
        issues.append(f"{fname}: " + "; ".join(file_issues))

    files_results.append({
        "filename": fname,
        "rows": n_rows,
        "columns": n_cols,
        "missing_values_total": missing_total,
        "missing_values_by_column": missing_by_col,
        "non_numeric_non_metadata_cols": non_numeric,
        "status_types_found": st_values,
        "train_test_values": tt_values,
        "issues": file_issues,
        "valid": file_valid
    })

all_valid = all(f["valid"] for f in files_results)

if issues:
    summary = "Issues found:\n" + "\n".join(f"  - {i}" for i in issues)
else:
    summary = f"All {len(csv_files)} files passed validation. 257 columns each, valid status_type_id and train_test values."

result = {
    "farm": "B",
    "total_files": len(csv_files),
    "all_valid": all_valid,
    "summary": summary,
    "files": files_results
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

# Print summary
print(f"\n{'='*60}")
print(f"FARM B VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Total files: {len(csv_files)}")
print(f"All valid: {all_valid}")
print()
for fr in files_results:
    status = "OK" if fr["valid"] else "FAIL"
    print(f"  {fr['filename']:>8s}  rows={fr['rows']:>6d}  cols={fr['columns']}  missing={fr['missing_values_total']:>6d}  status_ids={fr['status_types_found']}  tt={fr['train_test_values']}  [{status}]")
    if fr["issues"]:
        for iss in fr["issues"]:
            print(f"           ISSUE: {iss}")
    if fr["missing_values_by_column"]:
        top_missing = sorted(fr["missing_values_by_column"].items(), key=lambda x: -x[1])[:5]
        print(f"           Top missing cols: {top_missing}")
print()
print(summary)
print(f"\nResults saved to: {OUTPUT_PATH}")
