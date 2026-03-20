"""
Batch 2 chart fixes — re-render charts that were saved at incorrect sizes.

Fixes:
  1. power_curve_farm_a/b/c.png — figsize=(14,6), proper layout
  2. temp_vs_ops_gearbox_oil_farm_a.png — figsize=(14,6)
  3. temp_vs_ops_gen_bearing_farm_a.png — figsize=(14,6)
  4. temp_vs_ops_cross_farm_gearbox.png — figsize=(18,6)
  5. temp_vs_ops_correlation_heatmaps.png — figsize=(20,8), readable labels
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/Personal Projects/Enbridge Case Compettion")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.data.load_data import (
    load_event,
    load_event_info,
    get_event_ids,
    load_farm_training_data,
)
from src.features.operating_conditions import (
    get_operating_features,
    FEATURE_DESCRIPTIONS,
    FEATURE_CATEGORIES,
)
from src.features.thermal_config import get_all_thermal_sensors, THERMAL_SUBSYSTEMS

# ── Plotting defaults ──────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 9,
})

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants from the notebooks ───────────────────────────────────────────
POWER_CURVE_COLS = {
    "A": {"wind_speed": "wind_speed_3_avg", "power": "power_29_avg"},
    "B": {"wind_speed": "wind_speed_61_avg", "power": "power_62_avg"},
    "C": {"wind_speed": "wind_speed_235_avg", "power": "power_6_avg"},
}

STATUS_LABELS = {
    0: "Normal",
    1: "Derated",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}
STATUS_COLORS = {
    0: "#2196F3",
    1: "#FF9800",
    2: "#4CAF50",
    3: "#9C27B0",
    4: "#F44336",
    5: "#795548",
}

MAX_POINTS = 50_000
SAMPLE_N = 30_000

SENSOR_LABELS = {
    "sensor_12_avg": "Gearbox Oil Temp",
    "sensor_11_avg": "Gearbox Bearing Temp",
    "sensor_13_avg": "Generator Bearing DE Temp",
    "sensor_14_avg": "Generator Bearing NDE Temp",
    "sensor_15_avg": "Generator Winding U Temp",
    "sensor_16_avg": "Generator Winding V Temp",
    "sensor_17_avg": "Generator Winding W Temp",
    "sensor_38_avg": "Transformer Phase 1 Temp",
    "sensor_39_avg": "Transformer Phase 2 Temp",
    "sensor_40_avg": "Transformer Phase 3 Temp",
    "sensor_41_avg": "Hydraulic Oil Temp",
    "sensor_8_avg":  "Cooling Liquid Temp",
    "sensor_0_avg":  "Ambient Temp",
    "sensor_34_avg": "Gearbox Oil Sump Temp (B)",
    "sensor_35_avg": "Gearbox HS Bearing Temp (B)",
    "sensor_32_avg": "Generator Bearing 1 Temp (B)",
    "sensor_33_avg": "Generator Bearing 2 Temp (B)",
    "sensor_151_avg": "Gearbox Oil Sump Temp (C)",
    "sensor_152_avg": "Gearbox Bearing Temp (C)",
    "sensor_18_avg":  "Generator Bearing Temp (C)",
}

GEARBOX_OIL_SENSOR = {
    "A": "sensor_12_avg",
    "B": "sensor_34_avg",
    "C": "sensor_151_avg",
}

POWER_COL = {
    "A": "power_29_avg",
    "B": "power_62_avg",
    "C": "power_6_avg",
}

WIND_COL = {
    "A": "wind_speed_3_avg",
    "B": "wind_speed_61_avg",
    "C": "wind_speed_235_avg",
}

AMBIENT_COL = {
    "A": "sensor_0_avg",
    "B": "sensor_8_avg",
    "C": "sensor_7_avg",
}


# ── Helpers ────────────────────────────────────────────────────────────────

events_df = load_event_info()


def get_event_description(farm_letter, event_id):
    row = events_df[
        (events_df["farm"] == farm_letter) & (events_df["event_id"] == event_id)
    ]
    if len(row) == 0:
        return "N/A"
    desc = row.iloc[0]["event_description"]
    if pd.isna(desc):
        return "Normal operation"
    return str(desc)


def plot_power_curve(ax, df, farm_letter, title, sample_max=MAX_POINTS):
    cols = POWER_CURVE_COLS[farm_letter]
    ws_col = cols["wind_speed"]
    pw_col = cols["power"]

    plot_df = df[[ws_col, pw_col, "status_type_id"]].dropna()
    if len(plot_df) > sample_max:
        plot_df = plot_df.sample(n=sample_max, random_state=42)

    status_counts = plot_df["status_type_id"].value_counts()
    plot_df = plot_df.copy()
    plot_df["_sort_order"] = plot_df["status_type_id"].map(
        lambda s: status_counts.get(s, 0)
    )
    plot_df = plot_df.sort_values("_sort_order", ascending=False)

    present_statuses = sorted(plot_df["status_type_id"].unique())
    for status in present_statuses:
        mask = plot_df["status_type_id"] == status
        subset = plot_df[mask]
        ax.scatter(
            subset[ws_col],
            subset[pw_col],
            c=STATUS_COLORS.get(status, "#999999"),
            label=f"{STATUS_LABELS.get(status, f'Status {status}')} ({len(subset):,})",
            s=4,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (normalized)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, markerscale=3, framealpha=0.9)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.05)


def make_corr_label(col, farm_name):
    desc_map = FEATURE_DESCRIPTIONS.get(farm_name, {})
    if col in desc_map:
        name, unit = desc_map[col]
        return f"{name} ({unit})"
    if col in SENSOR_LABELS:
        return SENSOR_LABELS[col]
    return col


# =========================================================================
# FIX 1 — Power curve charts (farms A, B, C)
# =========================================================================
print("=" * 60)
print("FIX 1: Re-rendering power curve charts...")
print("=" * 60)

power_curve_configs = {
    "A": {"normal_id": 3, "anomaly_id": 0},
    "B": {"normal_id": 2, "anomaly_id": 7},
    "C": {"normal_id": 1, "anomaly_id": 4},
}

for farm, cfg in power_curve_configs.items():
    normal_id = cfg["normal_id"]
    anomaly_id = cfg["anomaly_id"]

    df_norm = load_event(farm, normal_id)
    df_anom = load_event(farm, anomaly_id)

    norm_desc = get_event_description(farm, normal_id)
    anom_desc = get_event_description(farm, anomaly_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    plot_power_curve(
        axes[0], df_norm, farm,
        f"Farm {farm} -- Event {normal_id} ({norm_desc})"
    )
    plot_power_curve(
        axes[1], df_anom, farm,
        f"Farm {farm} -- Event {anomaly_id} ({anom_desc})"
    )

    fig.suptitle(
        f"Farm {farm} -- Normal vs Anomaly Power Curves",
        fontsize=15, fontweight="bold",
    )

    out_path = FIGURES_DIR / f"power_curve_farm_{farm.lower()}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


# =========================================================================
# FIX 2 — Temp vs Ops scatter plots (Farm A gearbox oil + gen bearing)
# =========================================================================
print("\n" + "=" * 60)
print("FIX 2: Re-rendering temp vs ops scatter plots (Farm A)...")
print("=" * 60)

print("  Loading Farm A training data...")
df_a = load_farm_training_data("A")
print(f"  Farm A: {df_a.shape[0]:,} rows")

# --- 2a: Gearbox Oil Temp ---
temp_col = "sensor_12_avg"
ws_col = WIND_COL["A"]
pw_col = POWER_COL["A"]

plot_df = df_a[[ws_col, pw_col, temp_col]].dropna()
if len(plot_df) > SAMPLE_N:
    plot_df = plot_df.sample(n=SAMPLE_N, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Left panel: vs Wind Speed, colored by Power
sc1 = axes[0].scatter(
    plot_df[ws_col], plot_df[temp_col],
    c=plot_df[pw_col], cmap="viridis", s=4, alpha=0.3,
    edgecolors="none", rasterized=True,
)
cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.85, pad=0.02)
cb1.set_label("Power Output (kW)", fontsize=10)

sorted_idx = plot_df[ws_col].argsort()
ws_sorted = plot_df[ws_col].values[sorted_idx]
temp_sorted = plot_df[temp_col].values[sorted_idx]
lw = lowess(temp_sorted, ws_sorted, frac=0.15)
axes[0].plot(lw[:, 0], lw[:, 1], color="red", linewidth=2.5, label="LOWESS trend")
axes[0].set_xlabel("Wind Speed (m/s)")
axes[0].set_ylabel("Gearbox Oil Temperature (deg C)")
axes[0].set_title("Gearbox Oil Temp vs Wind Speed")
axes[0].legend(loc="lower right", fontsize=9)

# Right panel: vs Power, colored by Wind Speed
sc2 = axes[1].scatter(
    plot_df[pw_col], plot_df[temp_col],
    c=plot_df[ws_col], cmap="plasma", s=4, alpha=0.3,
    edgecolors="none", rasterized=True,
)
cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.85, pad=0.02)
cb2.set_label("Wind Speed (m/s)", fontsize=10)

sorted_idx2 = plot_df[pw_col].argsort()
pw_sorted = plot_df[pw_col].values[sorted_idx2]
temp_sorted2 = plot_df[temp_col].values[sorted_idx2]
lw2 = lowess(temp_sorted2, pw_sorted, frac=0.15)
axes[1].plot(lw2[:, 0], lw2[:, 1], color="red", linewidth=2.5, label="LOWESS trend")
axes[1].set_xlabel("Power Output (kW)")
axes[1].set_ylabel("Gearbox Oil Temperature (deg C)")
axes[1].set_title("Gearbox Oil Temp vs Power Output")
axes[1].legend(loc="lower right", fontsize=9)

fig.suptitle(
    "Farm A -- Gearbox Oil Temperature vs Operating Conditions",
    fontsize=14, fontweight="bold",
)

out_path = FIGURES_DIR / "temp_vs_ops_gearbox_oil_farm_a.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")

# --- 2b: Generator Bearing DE Temp ---
temp_col = "sensor_13_avg"

plot_df = df_a[[ws_col, pw_col, temp_col]].dropna()
if len(plot_df) > SAMPLE_N:
    plot_df = plot_df.sample(n=SAMPLE_N, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Left: vs Wind Speed, colored by Power
sc1 = axes[0].scatter(
    plot_df[ws_col], plot_df[temp_col],
    c=plot_df[pw_col], cmap="viridis", s=4, alpha=0.3,
    edgecolors="none", rasterized=True,
)
cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.85, pad=0.02)
cb1.set_label("Power Output (kW)", fontsize=10)

sorted_idx = plot_df[ws_col].argsort()
lw = lowess(plot_df[temp_col].values[sorted_idx], plot_df[ws_col].values[sorted_idx], frac=0.15)
axes[0].plot(lw[:, 0], lw[:, 1], color="red", linewidth=2.5, label="LOWESS trend")
axes[0].set_xlabel("Wind Speed (m/s)")
axes[0].set_ylabel("Generator Bearing DE Temperature (deg C)")
axes[0].set_title("Gen. Bearing Temp vs Wind Speed")
axes[0].legend(loc="lower right", fontsize=9)

# Right: vs Power, colored by Wind Speed
sc2 = axes[1].scatter(
    plot_df[pw_col], plot_df[temp_col],
    c=plot_df[ws_col], cmap="plasma", s=4, alpha=0.3,
    edgecolors="none", rasterized=True,
)
cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.85, pad=0.02)
cb2.set_label("Wind Speed (m/s)", fontsize=10)

sorted_idx2 = plot_df[pw_col].argsort()
lw2 = lowess(plot_df[temp_col].values[sorted_idx2], plot_df[pw_col].values[sorted_idx2], frac=0.15)
axes[1].plot(lw2[:, 0], lw2[:, 1], color="red", linewidth=2.5, label="LOWESS trend")
axes[1].set_xlabel("Power Output (kW)")
axes[1].set_ylabel("Generator Bearing DE Temperature (deg C)")
axes[1].set_title("Gen. Bearing Temp vs Power Output")
axes[1].legend(loc="lower right", fontsize=9)

fig.suptitle(
    "Farm A -- Generator Bearing (DE) Temperature vs Operating Conditions",
    fontsize=14, fontweight="bold",
)

out_path = FIGURES_DIR / "temp_vs_ops_gen_bearing_farm_a.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


# =========================================================================
# FIX 3 — Cross-farm gearbox oil temp comparison
# =========================================================================
print("\n" + "=" * 60)
print("FIX 3: Re-rendering cross-farm gearbox comparison...")
print("=" * 60)

print("  Loading Farm B and C training data...")
df_b = load_farm_training_data("B")
df_c = load_farm_training_data("C")
print(f"  Farm B: {df_b.shape[0]:,} rows")
print(f"  Farm C: {df_c.shape[0]:,} rows")

farm_data = {"A": df_a, "B": df_b, "C": df_c}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for i, (farm_letter, df_farm) in enumerate(farm_data.items()):
    ax = axes[i]
    gb_col = GEARBOX_OIL_SENSOR[farm_letter]
    pw_col = POWER_COL[farm_letter]
    ws_col = WIND_COL[farm_letter]

    plot_df = df_farm[[pw_col, gb_col, ws_col]].dropna()
    if len(plot_df) > SAMPLE_N:
        plot_df = plot_df.sample(n=SAMPLE_N, random_state=42)

    sc = ax.scatter(
        plot_df[pw_col], plot_df[gb_col],
        c=plot_df[ws_col], cmap="plasma", s=4, alpha=0.3,
        edgecolors="none", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Wind Speed (m/s)", fontsize=9)

    sorted_idx = plot_df[pw_col].argsort()
    lw = lowess(
        plot_df[gb_col].values[sorted_idx],
        plot_df[pw_col].values[sorted_idx],
        frac=0.15,
    )
    ax.plot(lw[:, 0], lw[:, 1], color="red", linewidth=2.5, label="LOWESS trend")

    sensor_label = SENSOR_LABELS.get(gb_col, gb_col)
    ax.set_xlabel("Power Output (kW)")
    ax.set_ylabel(f"{sensor_label} (deg C)")
    ax.set_title(f"Farm {farm_letter}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "Cross-Farm -- Gearbox Oil Temperature vs Power Output",
    fontsize=15, fontweight="bold",
)

out_path = FIGURES_DIR / "temp_vs_ops_cross_farm_gearbox.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


# =========================================================================
# FIX 4 — Correlation heatmaps (one combined figure at larger size)
# =========================================================================
print("\n" + "=" * 60)
print("FIX 4: Re-rendering correlation heatmaps...")
print("=" * 60)

farm_configs = {
    "Wind Farm A": (df_a, "farm_a"),
    "Wind Farm B": (df_b, "farm_b"),
    "Wind Farm C": (df_c, "farm_c"),
}

fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

for idx, (farm_name, (df_farm, farm_key)) in enumerate(farm_configs.items()):
    ax = axes[idx]

    op_cols = get_operating_features(farm_name)
    th_cols = get_all_thermal_sensors(farm_key)

    # Filter to columns present in data
    op_cols = [c for c in op_cols if c in df_farm.columns]
    th_cols = [c for c in th_cols if c in df_farm.columns]

    # Limit thermal sensors for readability
    if len(th_cols) > 10:
        th_cols = th_cols[:10]

    combined = df_farm[op_cols + th_cols].dropna()
    corr = combined.corr().loc[op_cols, th_cols]

    # Human-readable labels
    y_labels = [make_corr_label(c, farm_name) for c in op_cols]
    x_labels = [SENSOR_LABELS.get(c, c) for c in th_cols]

    sns.heatmap(
        corr,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(f"{farm_name}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

fig.suptitle(
    "Correlation: Operating Conditions vs Thermal Sensors",
    fontsize=15, fontweight="bold",
)

out_path = FIGURES_DIR / "temp_vs_ops_correlation_heatmaps.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
print("All fixes applied. Updated files:")
print("=" * 60)
fixed_files = [
    "power_curve_farm_a.png",
    "power_curve_farm_b.png",
    "power_curve_farm_c.png",
    "temp_vs_ops_gearbox_oil_farm_a.png",
    "temp_vs_ops_gen_bearing_farm_a.png",
    "temp_vs_ops_cross_farm_gearbox.png",
    "temp_vs_ops_correlation_heatmaps.png",
]
for f in fixed_files:
    p = FIGURES_DIR / f
    if p.exists():
        print(f"  {f:50s} ({p.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"  {f:50s} MISSING!")
