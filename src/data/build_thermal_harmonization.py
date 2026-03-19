"""
Phase 1.4: Cross-Farm Thermal Sensor Harmonization
Maps semantically equivalent temperature sensors across Wind Farms A, B, and C
into 6 thermal subsystems for the Thermal Degradation Index (TDI).
"""

import json
import os
import pandas as pd

# Paths
BASE = "D:/Personal Projects/Enbridge Case Compettion"
FARM_A_DESC = f"{BASE}/data/raw/CARE_To_Compare/Wind Farm A/feature_description.csv"
FARM_B_DESC = f"{BASE}/data/raw/CARE_To_Compare/Wind Farm B/feature_description.csv"
FARM_C_DESC = f"{BASE}/data/raw/CARE_To_Compare/Wind Farm C/feature_description.csv"
OUTPUT_MAP = f"{BASE}/data/processed/thermal_harmonization_map.json"
OUTPUT_SUMMARY = f"{BASE}/outputs/reports/thermal_harmonization_summary.json"


def load_feature_descriptions(path):
    """Load feature descriptions CSV (semicolon-separated)."""
    df = pd.read_csv(path, sep=';')
    return df


def get_temp_sensors(df):
    """Filter to temperature-related sensors based on description and unit."""
    desc_lower = df['description'].str.lower()
    unit_lower = df['unit'].str.lower().fillna('')

    mask = (
        desc_lower.str.contains('temp', na=False) |
        unit_lower.isin(['°c', 'celsius', '�c'])
    )
    temp_df = df[mask].copy()
    return temp_df


def get_avg_sensor_name(sensor_name, statistics_type):
    """Return the _avg variant of a sensor name."""
    stats = [s.strip() for s in statistics_type.split(',')]
    if 'average' in stats:
        return f"{sensor_name}_avg"
    return f"{sensor_name}_avg"  # default to avg even if not listed


def classify_sensor(description):
    """Classify a temperature sensor into a thermal subsystem based on description."""
    desc = description.lower().strip()

    # --- Gearbox ---
    if any(kw in desc for kw in ['gearbox', 'gear box', 'gear oil', 'planetary bearing']):
        return 'gearbox'

    # --- Generator/Bearings ---
    if any(kw in desc for kw in ['generator bearing', 'stator winding', 'generator temperature',
                                   'generator cooling', 'slip ring', 'rotor bearing']):
        return 'generator_bearings'

    # --- Transformer ---
    if any(kw in desc for kw in ['transformer', 'transformator', 'hv transformer']):
        return 'transformer'

    # --- Hydraulic ---
    if any(kw in desc for kw in ['hydraulic']):
        return 'hydraulic'

    # --- Nacelle/Ambient ---
    # Exclude "electrical cabinet ambient" — that's cabinet, not nacelle
    if 'electrical cabinet' in desc:
        return 'unmatched'
    if any(kw in desc for kw in ['ambient', 'nacelle temp', 'outside temp',
                                   'nacelle outside', 'hub temp']):
        return 'nacelle_ambient'

    # --- Cooling ---
    if any(kw in desc for kw in ['cooling', 'vcs', 'water temp', 'cooler',
                                   'water cooler']):
        return 'cooling'

    # --- Additional patterns for edge cases ---
    # Axial bearing -> generator/bearings
    if 'axial bearing' in desc:
        return 'generator_bearings'

    # Motor temperature (pitch motors) -> unmatched
    if 'motor temp' in desc:
        return 'unmatched'

    # Converter/inverter -> unmatched (electrical, not thermal subsystem)
    if any(kw in desc for kw in ['converter', 'inverter', 'igbt']):
        return 'unmatched'

    # Electrical cabinet -> unmatched
    if any(kw in desc for kw in ['electrical cabinet', 'control box', 'board temp',
                                   'busbar', 'split ring chamber', 'platform']):
        return 'unmatched'

    # Hub controller / top nacelle controller -> nacelle_ambient
    if any(kw in desc for kw in ['hub controller', 'top nacelle controller', 'nose cone']):
        return 'nacelle_ambient'

    # VCP-board, choke coils -> cooling (VCS related)
    if any(kw in desc for kw in ['vcp-board', 'choke coil']):
        return 'cooling'

    # Transformer oil temps in Farm C
    if 'oil temp' in desc and ('transformer' in desc or 'eb transformer' in desc):
        return 'transformer'

    return 'unmatched'


def build_harmonization():
    """Build the full thermal harmonization map."""
    # Load all feature descriptions
    df_a = load_feature_descriptions(FARM_A_DESC)
    df_b = load_feature_descriptions(FARM_B_DESC)
    df_c = load_feature_descriptions(FARM_C_DESC)

    # Filter to temperature sensors
    temp_a = get_temp_sensors(df_a)
    temp_b = get_temp_sensors(df_b)
    temp_c = get_temp_sensors(df_c)

    print(f"Temperature sensors found:")
    print(f"  Farm A: {len(temp_a)}")
    print(f"  Farm B: {len(temp_b)}")
    print(f"  Farm C: {len(temp_c)}")
    print()

    # Classify each sensor
    subsystems = ['gearbox', 'generator_bearings', 'transformer', 'hydraulic',
                  'nacelle_ambient', 'cooling', 'unmatched']

    subsystem_labels = {
        'gearbox': 'Gearbox',
        'generator_bearings': 'Generator/Bearings',
        'transformer': 'Transformer',
        'hydraulic': 'Hydraulic',
        'nacelle_ambient': 'Nacelle/Ambient',
        'cooling': 'Cooling',
        'unmatched': 'Unmatched'
    }

    harmonization_map = {}
    summary = {}

    for subsystem in subsystems:
        harmonization_map[subsystem] = {
            'subsystem_name': subsystem_labels[subsystem],
            'sensors_farm_a': [],
            'sensors_farm_b': [],
            'sensors_farm_c': []
        }
        summary[subsystem] = {
            'subsystem_name': subsystem_labels[subsystem],
            'sensors_farm_a': [],
            'sensors_farm_b': [],
            'sensors_farm_c': [],
            'count_farm_a': 0,
            'count_farm_b': 0,
            'count_farm_c': 0
        }

    # Process Farm A
    print("=" * 80)
    print("FARM A - Temperature Sensors")
    print("=" * 80)
    for _, row in temp_a.iterrows():
        sensor = row['sensor_name']
        desc = row['description']
        avg_name = get_avg_sensor_name(sensor, row['statistics_type'])
        subsystem = classify_sensor(desc)

        harmonization_map[subsystem]['sensors_farm_a'].append(avg_name)
        summary[subsystem]['sensors_farm_a'].append({
            'sensor': avg_name,
            'description': desc
        })
        print(f"  [{subsystem_labels[subsystem]:20s}] {avg_name:25s} -> {desc}")

    print()

    # Process Farm B
    print("=" * 80)
    print("FARM B - Temperature Sensors")
    print("=" * 80)
    for _, row in temp_b.iterrows():
        sensor = row['sensor_name']
        desc = row['description']
        avg_name = get_avg_sensor_name(sensor, row['statistics_type'])
        subsystem = classify_sensor(desc)

        harmonization_map[subsystem]['sensors_farm_b'].append(avg_name)
        summary[subsystem]['sensors_farm_b'].append({
            'sensor': avg_name,
            'description': desc
        })
        print(f"  [{subsystem_labels[subsystem]:20s}] {avg_name:25s} -> {desc}")

    print()

    # Process Farm C
    print("=" * 80)
    print("FARM C - Temperature Sensors")
    print("=" * 80)
    for _, row in temp_c.iterrows():
        sensor = row['sensor_name']
        desc = row['description']
        avg_name = get_avg_sensor_name(sensor, row['statistics_type'])
        subsystem = classify_sensor(desc)

        harmonization_map[subsystem]['sensors_farm_c'].append(avg_name)
        summary[subsystem]['sensors_farm_c'].append({
            'sensor': avg_name,
            'description': desc
        })
        print(f"  [{subsystem_labels[subsystem]:20s}] {avg_name:25s} -> {desc}")

    print()

    # Update counts in summary
    for subsystem in subsystems:
        summary[subsystem]['count_farm_a'] = len(summary[subsystem]['sensors_farm_a'])
        summary[subsystem]['count_farm_b'] = len(summary[subsystem]['sensors_farm_b'])
        summary[subsystem]['count_farm_c'] = len(summary[subsystem]['sensors_farm_c'])

    # Print summary table
    print("=" * 80)
    print("HARMONIZATION SUMMARY")
    print("=" * 80)
    print(f"{'Subsystem':<25s} {'Farm A':>8s} {'Farm B':>8s} {'Farm C':>8s} {'Total':>8s}")
    print("-" * 57)
    total_a = total_b = total_c = 0
    for subsystem in subsystems:
        ca = summary[subsystem]['count_farm_a']
        cb = summary[subsystem]['count_farm_b']
        cc = summary[subsystem]['count_farm_c']
        total_a += ca
        total_b += cb
        total_c += cc
        print(f"{subsystem_labels[subsystem]:<25s} {ca:>8d} {cb:>8d} {cc:>8d} {ca+cb+cc:>8d}")
    print("-" * 57)
    print(f"{'TOTAL':<25s} {total_a:>8d} {total_b:>8d} {total_c:>8d} {total_a+total_b+total_c:>8d}")
    print()

    # Print detailed cross-farm mapping for modeled subsystems
    print("=" * 80)
    print("CROSS-FARM SENSOR MAPPING (for TDI modeling)")
    print("=" * 80)
    for subsystem in subsystems:
        if subsystem == 'unmatched':
            continue
        info = harmonization_map[subsystem]
        print(f"\n--- {subsystem_labels[subsystem]} ---")
        for farm_key, farm_label in [('sensors_farm_a', 'Farm A'),
                                      ('sensors_farm_b', 'Farm B'),
                                      ('sensors_farm_c', 'Farm C')]:
            sensors = info[farm_key]
            if sensors:
                # Get descriptions
                sub_summary = summary[subsystem][farm_key]
                for s in sub_summary:
                    print(f"  {farm_label}: {s['sensor']:30s} | {s['description']}")
            else:
                print(f"  {farm_label}: (none)")

    # Print unmatched
    print(f"\n--- Unmatched ---")
    for farm_key, farm_label in [('sensors_farm_a', 'Farm A'),
                                  ('sensors_farm_b', 'Farm B'),
                                  ('sensors_farm_c', 'Farm C')]:
        sub_summary = summary['unmatched'][farm_key]
        if sub_summary:
            for s in sub_summary:
                print(f"  {farm_label}: {s['sensor']:30s} | {s['description']}")

    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_MAP), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)

    with open(OUTPUT_MAP, 'w') as f:
        json.dump(harmonization_map, f, indent=2)
    print(f"\nSaved harmonization map to: {OUTPUT_MAP}")

    with open(OUTPUT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {OUTPUT_SUMMARY}")


if __name__ == '__main__':
    build_harmonization()
