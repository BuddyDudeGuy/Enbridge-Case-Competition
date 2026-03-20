"""
Batch 1 chart fixes — regenerates 4 figures with corrections.

1. thermal_boxplots_farm_c.png     — clip temps to [-50, 200] to remove corrupt -3000 gearbox data
2. summary_normal_vs_anomaly_grid.png — bigger figsize, dpi, font sizes for readability
3. farm_a_generator_bearing_normal_vs_anomaly.png — remove raw sensor ID from title/y-axis, enlarge
4. thermal_cross_farm_ranges.png   — add "°C" to temperature axis labels
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

# Project root on sys.path so imports work
PROJECT_ROOT = "D:/Personal Projects/Enbridge Case Compettion"
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data.load_data import load_farm_training_data, load_event, load_event_info
from src.features.thermal_config import (
    get_all_thermal_sensors,
    get_sensors,
    THERMAL_SUBSYSTEMS,
    SUBSYSTEM_SENSORS,
)

FIGURES_DIR = Path(PROJECT_ROOT) / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

# Load event catalog (needed for charts 2 and 3)
events = load_event_info()


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — thermal_boxplots_farm_c.png
# Problem: Gearbox panel has corrupt sentinel values going to -3276.7
# Fix: clip all temperature values to physical range [-50, 200] before plotting
# ═══════════════════════════════════════════════════════════════════════════════

def fix_thermal_boxplots_farm_c():
    print("=" * 60)
    print("FIX 1: thermal_boxplots_farm_c.png")
    print("  Clipping all temps to [-50, 200] to remove corrupt data")
    print("=" * 60)

    # Load Farm C thermal data
    thermal_c_all = get_all_thermal_sensors("farm_c")
    df_c = load_farm_training_data("C")[thermal_c_all]
    print(f"  Loaded Farm C: {df_c.shape[0]:,} rows x {df_c.shape[1]} cols")

    # Build subsystem panel list (representative sensors, max 3 per subsystem)
    subsystems_with_sensors_c = []
    for sub_key, sub_info in THERMAL_SUBSYSTEMS.items():
        s_list = get_sensors("farm_c", sub_key)[:3]  # max 3 per subsystem
        s_list = [s for s in s_list if s in df_c.columns]
        if s_list:
            subsystems_with_sensors_c.append((sub_info["name"], s_list))

    n_subs_c = len(subsystems_with_sensors_c)
    fig, axes = plt.subplots(n_subs_c, 1, figsize=(14, 3.5 * n_subs_c), constrained_layout=True)
    if n_subs_c == 1:
        axes = [axes]

    for ax, (sub_name, s_list) in zip(axes, subsystems_with_sensors_c):
        # Sample for speed, then CLIP to physical range
        sample = df_c[s_list].dropna().sample(n=min(50_000, len(df_c)), random_state=42)
        sample = sample.clip(lower=-50, upper=200)  # <-- THE FIX
        melted = sample.melt(var_name="sensor", value_name="temperature")
        sns.boxplot(data=melted, x="sensor", y="temperature", ax=ax,
                    palette="Set2", fliersize=1, linewidth=0.8)
        ax.set_title(f"{sub_name} (representative sensors)", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Temperature")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Farm C — Thermal Sensor Distributions by Subsystem (representatives)",
                 fontsize=15, fontweight="bold", y=1.01)
    out_path = FIGURES_DIR / "thermal_boxplots_farm_c.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — summary_normal_vs_anomaly_grid.png
# Problem: currently unreadable — too small, tiny fonts
# Fix: figsize=(18,12), dpi=200, larger font sizes for titles and labels
# ═══════════════════════════════════════════════════════════════════════════════

def fix_summary_grid():
    print("\n" + "=" * 60)
    print("FIX 2: summary_normal_vs_anomaly_grid.png")
    print("  Re-rendering at figsize=(18,12), dpi=200, larger fonts")
    print("=" * 60)

    summary_cases = [
        ("A", 72,  13, "sensor_12_avg",  "Farm A — Gearbox Failure",               "Gearbox Oil Temp"),
        ("A", 40,   3, "sensor_14_avg",  "Farm A — Generator Bearing Failure",     "Gen. Bearing Temp"),
        ("B", 34,  52, "sensor_41_avg",  "Farm B — Transformer High Temp",         "Transformer Cell Temp"),
        ("B", 53,  23, "sensor_32_avg",  "Farm B — Rotor Bearing Damage",          "Rotor Bearing Temp"),
        ("C", 47,  62, "sensor_178_avg", "Farm C — Hydraulic Pump Failure",        "Hydraulic Oil Temp"),
        ("C", 44,  75, "sensor_175_avg", "Farm C — Cooling Valve Misposition",     "Cooling System Temp"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, (farm, anom_id, norm_id, sensor, fault, subsys) in enumerate(summary_cases):
        ax = axes_flat[idx]

        df_anom = load_event(farm, anom_id)
        df_norm = load_event(farm, norm_id)

        meta_anom = events[(events["farm"] == farm) & (events["event_id"] == anom_id)].iloc[0]
        meta_norm = events[(events["farm"] == farm) & (events["event_id"] == norm_id)].iloc[0]

        anom_train_end = (df_anom["train_test"] == "train").sum()
        norm_train_end = (df_norm["train_test"] == "train").sum()

        anom_event_start_idx = int(meta_anom["event_start_id"])
        anom_event_end_idx   = int(meta_anom["event_end_id"])

        anom_vals = df_anom[sensor].values
        norm_vals = df_norm[sensor].values

        pred_len_anom = len(df_anom) - anom_train_end
        context_rows = min(int(pred_len_anom * 2.5), anom_train_end)
        anom_start = anom_train_end - context_rows

        anom_slice = anom_vals[anom_start:]
        anom_x = np.arange(len(anom_slice))

        context_rows_norm = min(context_rows, norm_train_end)
        norm_start = norm_train_end - context_rows_norm
        norm_slice = norm_vals[norm_start:]
        norm_x = np.arange(len(norm_slice))

        train_boundary_rel = context_rows
        event_start_rel = max(0, min(anom_event_start_idx - anom_start, len(anom_slice) - 1))
        event_end_rel   = max(0, min(anom_event_end_idx - anom_start, len(anom_slice) - 1))

        # Plot with thicker lines for readability
        ax.plot(norm_x, norm_slice, color="steelblue", alpha=0.7, linewidth=0.8,
                label=f"Normal (evt {norm_id})")
        ax.plot(anom_x, anom_slice, color="crimson", alpha=0.85, linewidth=0.8,
                label=f"Anomaly (evt {anom_id})")
        ax.axvline(train_boundary_rel, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvspan(event_start_rel, event_end_rel, color="red", alpha=0.08)

        # Larger fonts for subplot titles and labels
        ax.set_title(fault, fontsize=14, fontweight="bold")
        ax.set_ylabel(subsys, fontsize=12)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.8)
        ax.tick_params(labelsize=10)

        if idx >= 4:
            ax.set_xlabel("Row index (10-min intervals)", fontsize=12)

    fig.suptitle(
        "Temperature Deviates Before Failure — Normal (blue) vs. Anomaly (red) Across All Farms",
        fontsize=17, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "summary_normal_vs_anomaly_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — farm_a_generator_bearing_normal_vs_anomaly.png
# Problem: raw sensor ID "sensor_14_avg" shown in title and y-axis
# Fix: use only human-readable name, enlarge to match hydraulic chart size
# ═══════════════════════════════════════════════════════════════════════════════

def fix_generator_bearing_chart():
    print("\n" + "=" * 60)
    print("FIX 3: farm_a_generator_bearing_normal_vs_anomaly.png")
    print("  Removing raw sensor ID, enlarging figure")
    print("=" * 60)

    farm = "A"
    anomaly_event_id = 40
    normal_event_id = 3
    sensor_col = "sensor_14_avg"
    fault_label = "Generator Bearing Failure (31-day event)"
    subsystem_label = "Generator Bearing Temp"

    df_anom = load_event(farm, anomaly_event_id)
    df_norm = load_event(farm, normal_event_id)

    meta_anom = events[(events["farm"] == farm) & (events["event_id"] == anomaly_event_id)].iloc[0]
    meta_norm = events[(events["farm"] == farm) & (events["event_id"] == normal_event_id)].iloc[0]

    anom_train_end = (df_anom["train_test"] == "train").sum()
    norm_train_end = (df_norm["train_test"] == "train").sum()

    anom_event_start_idx = int(meta_anom["event_start_id"])
    anom_event_end_idx = int(meta_anom["event_end_id"])

    anom_vals = df_anom[sensor_col].values
    norm_vals = df_norm[sensor_col].values

    pred_len_anom = len(df_anom) - anom_train_end
    context_rows = min(int(pred_len_anom * 2.5), anom_train_end)
    anom_start = anom_train_end - context_rows

    anom_slice = anom_vals[anom_start:]
    anom_x = np.arange(len(anom_slice))

    context_rows_norm = min(context_rows, norm_train_end)
    norm_start = norm_train_end - context_rows_norm
    norm_slice = norm_vals[norm_start:]
    norm_x = np.arange(len(norm_slice))

    train_boundary_rel = context_rows
    event_start_rel = max(0, min(anom_event_start_idx - anom_start, len(anom_slice) - 1))
    event_end_rel = max(0, min(anom_event_end_idx - anom_start, len(anom_slice) - 1))

    # Enlarged figure to match hydraulic chart (14, 6)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(norm_x, norm_slice, color="steelblue", alpha=0.7, linewidth=0.8,
            label=f"Normal (event {normal_event_id}, asset {int(meta_norm['asset_id'])})")
    ax.plot(anom_x, anom_slice, color="crimson", alpha=0.85, linewidth=0.8,
            label=f"Anomaly (event {anomaly_event_id}, asset {int(meta_anom['asset_id'])})")
    ax.axvline(train_boundary_rel, color="black", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Train -> Prediction")
    ax.axvspan(event_start_rel, event_end_rel, color="red", alpha=0.10, label="Fault window")

    ax.set_xlabel("Row index (10-min intervals)")
    # FIX: only human-readable name on y-axis (no sensor ID)
    ax.set_ylabel(f"{subsystem_label}")
    # FIX: only human-readable name in title (no sensor ID)
    ax.set_title(f"Farm {farm} — {fault_label}\n{subsystem_label}: Normal vs. Anomaly",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "farm_a_generator_bearing_normal_vs_anomaly.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4 — thermal_cross_farm_ranges.png
# Problem: temperature axis labels just say "Temperature" with no units
# Fix: add "°C" to the xlabel
# ═══════════════════════════════════════════════════════════════════════════════

def fix_cross_farm_ranges():
    print("\n" + "=" * 60)
    print("FIX 4: thermal_cross_farm_ranges.png")
    print("  Adding degree-C units to temperature axis labels")
    print("=" * 60)

    # Load training data for all 3 farms
    df_a = load_farm_training_data("A")
    df_b = load_farm_training_data("B")
    thermal_c_all = get_all_thermal_sensors("farm_c")
    df_c = load_farm_training_data("C")[thermal_c_all]

    farm_data = {"A": df_a, "B": df_b, "C": df_c}
    farm_keys = {"A": "farm_a", "B": "farm_b", "C": "farm_c"}

    cross_farm_rows = []
    for sub_key, sub_info in THERMAL_SUBSYSTEMS.items():
        sub_name = sub_info["name"]
        for farm_letter in ["A", "B", "C"]:
            fk = farm_keys[farm_letter]
            sensors = get_sensors(fk, sub_key)
            df = farm_data[farm_letter]
            present = [s for s in sensors if s in df.columns]
            if not present:
                continue
            vals = df[present].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            cross_farm_rows.append({
                "subsystem": sub_name,
                "farm": f"Farm {farm_letter}",
                "p01": np.percentile(vals, 1),
                "p25": np.percentile(vals, 25),
                "median": np.median(vals),
                "p75": np.percentile(vals, 75),
                "p99": np.percentile(vals, 99),
            })

    cross_df = pd.DataFrame(cross_farm_rows)

    subsystems_to_plot = list(cross_df["subsystem"].unique())
    farm_colors = {"Farm A": "#1f77b4", "Farm B": "#ff7f0e", "Farm C": "#2ca02c"}

    fig, axes = plt.subplots(len(subsystems_to_plot), 1,
                              figsize=(12, 2.5 * len(subsystems_to_plot)),
                              constrained_layout=True)
    if len(subsystems_to_plot) == 1:
        axes = [axes]

    for ax, sub_name in zip(axes, subsystems_to_plot):
        sub_data = cross_df[cross_df["subsystem"] == sub_name].reset_index(drop=True)
        for i, (_, row) in enumerate(sub_data.iterrows()):
            color = farm_colors.get(row["farm"], "gray")
            ax.barh(i, row["p99"] - row["p01"], left=row["p01"], height=0.5,
                    color=color, alpha=0.3, edgecolor=color, linewidth=1.2)
            ax.barh(i, row["p75"] - row["p25"], left=row["p25"], height=0.5,
                    color=color, alpha=0.7, edgecolor=color, linewidth=1.2)
            ax.plot(row["median"], i, marker="|", color="black", markersize=15, markeredgewidth=2)
        ax.set_yticks(range(len(sub_data)))
        ax.set_yticklabels(sub_data["farm"].values)
        ax.set_title(sub_name, fontweight="bold")
        # FIX: add °C to the x-axis label
        ax.set_xlabel("Temperature (\u00b0C)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="gray", alpha=0.3, edgecolor="gray", label="1st-99th percentile"),
        Patch(facecolor="gray", alpha=0.7, edgecolor="gray", label="IQR (25th-75th)"),
        Line2D([0], [0], marker="|", color="black", linestyle="None",
               markersize=12, markeredgewidth=2, label="Median"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=10)

    fig.suptitle("Cross-Farm Temperature Ranges by Subsystem", fontsize=15, fontweight="bold")
    out_path = FIGURES_DIR / "thermal_cross_farm_ranges.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all fixes
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting batch 1 chart fixes...\n")

    fix_thermal_boxplots_farm_c()
    fix_summary_grid()
    fix_generator_bearing_chart()
    fix_cross_farm_ranges()

    print("\n" + "=" * 60)
    print("All 4 charts fixed and saved to outputs/figures/")
    print("=" * 60)
