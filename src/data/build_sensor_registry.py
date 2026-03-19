"""
Phase 1.3: Build sensor registry from feature_description.csv files.

Parses all 3 farms' feature descriptions, categorizes sensors by type,
and outputs per-farm registries + a combined summary.
"""

import json
import os
import pandas as pd

# Paths
BASE_DIR = "D:/Personal Projects/Enbridge Case Compettion"
DATA_DIR = os.path.join(BASE_DIR, "data/raw/CARE_To_Compare")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/reports")

FARMS = {
    "A": os.path.join(DATA_DIR, "Wind Farm A/feature_description.csv"),
    "B": os.path.join(DATA_DIR, "Wind Farm B/feature_description.csv"),
    "C": os.path.join(DATA_DIR, "Wind Farm C/feature_description.csv"),
}


def categorize_sensor(description: str, is_angle: bool) -> str:
    """Categorize a sensor based on its description and is_angle flag."""
    desc = description.lower().strip()

    # Order matters — check more specific categories first
    if is_angle or any(kw in desc for kw in ["angle", "direction", "position"]):
        return "Position/Angle"
    if any(kw in desc for kw in ["vibration"]):
        return "Vibration"
    if any(kw in desc for kw in ["temperature", "temp"]):
        return "Temperature"
    if any(kw in desc for kw in ["pressure"]):
        return "Pressure"
    if any(kw in desc for kw in ["current"]):
        return "Current"
    if any(kw in desc for kw in ["voltage", "volt"]):
        return "Voltage"
    if any(kw in desc for kw in ["speed", "rpm", "rotor", "rotational"]):
        return "Speed/RPM"
    if any(kw in desc for kw in ["wind"]):
        return "Wind"
    if "power" in desc and "reactive" not in desc:
        return "Power"
    if any(kw in desc for kw in ["flow", "level"]):
        return "Flow/Level"
    return "Other"


def parse_farm(farm_name: str, filepath: str) -> dict:
    """Parse a single farm's feature_description.csv into a sensor registry."""
    df = pd.read_csv(filepath, sep=";")

    # Normalize column name (some files use statistics_type vs statistic_type)
    cols = df.columns.tolist()
    stat_col = [c for c in cols if "statistic" in c.lower()][0]

    sensors = []
    for _, row in df.iterrows():
        name = str(row["sensor_name"]).strip()
        description = str(row["description"]).strip()
        unit = str(row["unit"]).strip() if pd.notna(row["unit"]) else ""
        is_angle = str(row["is_angle"]).strip().lower() == "true"
        is_counter = str(row["is_counter"]).strip().lower() == "true"
        stat_types = [s.strip() for s in str(row[stat_col]).split(",")]
        category = categorize_sensor(description, is_angle)

        sensors.append({
            "name": name,
            "description": description,
            "unit": unit,
            "statistic_types": stat_types,
            "category": category,
            "is_angle": is_angle,
            "is_counter": is_counter,
        })

    # Count by category
    category_counts = {}
    for s in sensors:
        cat = s["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    registry = {
        "farm": farm_name,
        "total_sensors": len(sensors),
        "category_counts": category_counts,
        "sensors": sensors,
    }
    return registry


def build_summary(registries: dict) -> dict:
    """Build a cross-farm summary from individual registries."""
    all_categories = set()
    for reg in registries.values():
        all_categories.update(reg["category_counts"].keys())
    all_categories = sorted(all_categories)

    per_farm = {}
    for farm, reg in registries.items():
        per_farm[farm] = {
            "total_sensors": reg["total_sensors"],
            "category_counts": {
                cat: reg["category_counts"].get(cat, 0) for cat in all_categories
            },
        }

    # Cross-farm: which categories exist in all farms vs only some
    cross_farm = {}
    for cat in all_categories:
        farms_with = [f for f, reg in registries.items() if reg["category_counts"].get(cat, 0) > 0]
        cross_farm[cat] = {
            "present_in_farms": farms_with,
            "counts": {f: registries[f]["category_counts"].get(cat, 0) for f in registries},
        }

    return {
        "all_categories": all_categories,
        "per_farm": per_farm,
        "cross_farm_comparison": cross_farm,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    registries = {}
    for farm_name, filepath in FARMS.items():
        print(f"Parsing Farm {farm_name}: {filepath}")
        reg = parse_farm(farm_name, filepath)
        registries[farm_name] = reg

        # Save per-farm registry
        out_path = os.path.join(OUTPUT_DIR, f"sensor_registry_farm_{farm_name.lower()}.json")
        with open(out_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"  -> Saved {out_path}")

    # Build and save summary
    summary = build_summary(registries)
    summary_path = os.path.join(OUTPUT_DIR, "sensor_registry_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SENSOR REGISTRY SUMMARY")
    print("=" * 70)

    categories = summary["all_categories"]
    header = f"{'Category':<20} {'Farm A':>8} {'Farm B':>8} {'Farm C':>8}"
    print(header)
    print("-" * len(header))

    for cat in categories:
        counts = summary["cross_farm_comparison"][cat]["counts"]
        print(f"{cat:<20} {counts.get('A', 0):>8} {counts.get('B', 0):>8} {counts.get('C', 0):>8}")

    print("-" * len(header))
    totals = summary["per_farm"]
    print(f"{'TOTAL':<20} {totals['A']['total_sensors']:>8} {totals['B']['total_sensors']:>8} {totals['C']['total_sensors']:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
