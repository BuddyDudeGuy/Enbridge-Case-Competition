"""
Regenerate the three slide 6 charts with professional presentation annotations.

Charts:
  1. Farm A Gearbox Zoomed (Event 72) — "3-5 deg C above baseline"
  2. Farm C Cooling Normal vs Anomaly (Event 44 vs 75) — "66 days undetected"
  3. Farm A Generator Bearing Normal vs Anomaly (Event 40 vs 3) — "Spike to 200 deg C"

Each annotation uses a curved arrow with bold text in a white box with red border.

Output: overwrites existing PNGs in outputs/figures/
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "D:/Personal Projects/Enbridge Case Compettion"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from src.data.load_data import load_event, load_event_info

FIGURES_DIR = Path(PROJECT_ROOT) / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load event catalog once
events = load_event_info()

# Consistent annotation style
ANNOTATION_STYLE = dict(
    arrowprops=dict(
        arrowstyle="->",
        lw=2,
        color="#E74C3C",
        connectionstyle="arc3,rad=0.3",
    ),
    fontsize=11,
    fontweight="bold",
    color="#333333",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="#E74C3C",
        linewidth=2,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Farm A Gearbox Zoomed (same logic as eda_zoomed_charts.py)
# ─────────────────────────────────────────────────────────────────────────────

def chart_gearbox_zoomed():
    print("=" * 60)
    print("Chart 1: farm_a_gearbox_zoomed.png (annotated)")
    print("  Event 72 — Gearbox Oil Temperature (sensor_12_avg)")
    print("=" * 60)

    sns.set_style("whitegrid")

    df = pd.read_csv(
        f"{PROJECT_ROOT}/data/raw/CARE_To_Compare/Wind Farm A/datasets/72.csv",
        sep=";",
    )
    print(f"  Loaded event 72: {df.shape[0]:,} rows x {df.shape[1]} cols")

    sensor = "sensor_12_avg"
    train_mask = df["train_test"] == "train"
    pred_mask = df["train_test"] == "prediction"

    pred_start_idx = pred_mask.idxmax()

    evt_start = 52497
    evt_end = 53505

    # Healthy baseline
    normal_train = df[train_mask & df["status_type_id"].isin([0, 2])]
    healthy_mean = normal_train[sensor].mean()
    print(f"  Healthy baseline: {healthy_mean:.2f} C")

    # Zoom window
    buffer = 500
    zoom_start = max(0, pred_start_idx - buffer)
    zoom_df = df.iloc[zoom_start:].copy()
    zoom_df["row"] = np.arange(len(zoom_df))

    zoom_df["rolling_24h"] = (
        zoom_df[sensor].rolling(window=144, center=False, min_periods=1).mean()
    )

    pred_row = pred_start_idx - zoom_start
    evt_start_row = max(evt_start - zoom_start, 0)
    evt_end_row = min(evt_end - zoom_start, len(zoom_df) - 1)

    rows_per_day = 6 * 24
    zoom_df["days"] = (zoom_df["row"] - pred_row) / rows_per_day

    pred_day = 0.0
    evt_start_day = (evt_start_row - pred_row) / rows_per_day
    evt_end_day = (evt_end_row - pred_row) / rows_per_day

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        zoom_df["days"], zoom_df[sensor],
        color="#7BAFD4", alpha=0.35, linewidth=0.7,
        label="Raw sensor (10-min)",
    )
    ax.plot(
        zoom_df["days"], zoom_df["rolling_24h"],
        color="#1B4F72", linewidth=2.2,
        label="Rolling 24-hr average",
    )
    ax.axhline(
        y=healthy_mean, color="#2ECC71", linewidth=1.8, linestyle="--",
        label=f"Healthy baseline ({healthy_mean:.1f} \u00b0C)",
    )
    ax.axvspan(evt_start_day, evt_end_day, color="#E74C3C", alpha=0.12, label="Fault window")
    ax.axvline(x=pred_day, color="#888888", linewidth=1.0, linestyle=":", label="Prediction start")

    ax.set_title(
        "Farm A \u2014 Gearbox Oil Temperature (Zoomed to Fault Window)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Days relative to prediction start", fontsize=11)
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ── Annotation: find the rolling average peak inside the fault window ──
    fault_mask = (zoom_df["days"] >= evt_start_day) & (zoom_df["days"] <= evt_end_day)
    fault_data = zoom_df[fault_mask]

    if len(fault_data) > 0:
        peak_idx = fault_data["rolling_24h"].idxmax()
        peak_day = fault_data.loc[peak_idx, "days"]
        peak_temp = fault_data.loc[peak_idx, "rolling_24h"]
        print(f"  Rolling avg peak in fault window: day={peak_day:.1f}, temp={peak_temp:.1f} C")

        # Place text box above and to the left of the arrow target
        text_x = peak_day - 2.5
        text_y = peak_temp + 6

        # Clamp text position to be within axes
        xlim = ax.get_xlim()
        if text_x < xlim[0] + 0.5:
            text_x = peak_day + 1.5

        ax.annotate(
            "3-5\u00b0C above\nbaseline",
            xy=(peak_day, peak_temp),
            xytext=(text_x, text_y),
            **ANNOTATION_STYLE,
        )

    plt.tight_layout()

    out_path = FIGURES_DIR / "farm_a_gearbox_zoomed.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: replicate the notebook's plot_normal_vs_anomaly logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_data(farm, anomaly_event_id, normal_event_id, sensor_col):
    """Load and slice data for a normal-vs-anomaly overlay.

    Returns a dict with all the arrays and positions needed for plotting.
    """
    df_anom = load_event(farm, anomaly_event_id)
    df_norm = load_event(farm, normal_event_id)

    meta_anom = events[
        (events["farm"] == farm) & (events["event_id"] == anomaly_event_id)
    ].iloc[0]
    meta_norm = events[
        (events["farm"] == farm) & (events["event_id"] == normal_event_id)
    ].iloc[0]

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

    return dict(
        anom_slice=anom_slice,
        anom_x=anom_x,
        norm_slice=norm_slice,
        norm_x=norm_x,
        train_boundary_rel=train_boundary_rel,
        event_start_rel=event_start_rel,
        event_end_rel=event_end_rel,
        meta_anom=meta_anom,
        meta_norm=meta_norm,
        anomaly_event_id=anomaly_event_id,
        normal_event_id=normal_event_id,
    )


def _plot_comparison(ax, d, sensor_col, fault_label, subsystem_label, farm):
    """Draw the standard normal-vs-anomaly chart on the given axes."""
    ax.plot(
        d["norm_x"], d["norm_slice"],
        color="steelblue", alpha=0.7, linewidth=0.8,
        label=f"Normal (event {d['normal_event_id']}, asset {int(d['meta_norm']['asset_id'])})",
    )
    ax.plot(
        d["anom_x"], d["anom_slice"],
        color="crimson", alpha=0.85, linewidth=0.8,
        label=f"Anomaly (event {d['anomaly_event_id']}, asset {int(d['meta_anom']['asset_id'])})",
    )
    ax.axvline(
        d["train_boundary_rel"], color="black", linestyle="--", linewidth=1.2,
        alpha=0.7, label="Train \u2192 Prediction",
    )
    ax.axvspan(
        d["event_start_rel"], d["event_end_rel"],
        color="red", alpha=0.10, label="Fault window",
    )

    ax.set_xlabel("Row index (10-min intervals)")
    ax.set_ylabel(f"{subsystem_label} ({sensor_col})")
    ax.set_title(
        f"Farm {farm} \u2014 {fault_label}\n{sensor_col}: Normal vs. Anomaly",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Farm C Cooling Normal vs Anomaly (Event 44 vs 75)
# ─────────────────────────────────────────────────────────────────────────────

def chart_cooling_normal_vs_anomaly():
    print("=" * 60)
    print("Chart 2: farm_c_cooling_normal_vs_anomaly.png (annotated)")
    print("  Event 44 vs 75 — Cooling System Temp (sensor_175_avg)")
    print("=" * 60)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    d = _build_comparison_data(
        farm="C",
        anomaly_event_id=44,
        normal_event_id=75,
        sensor_col="sensor_175_avg",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_comparison(
        ax, d,
        sensor_col="sensor_175_avg",
        fault_label="Cooling Valve Left in Wrong Position (66-day event)",
        subsystem_label="Cooling System Temp",
        farm="C",
    )

    # ── Annotation: arrow pointing to middle of fault window where red diverges ──
    fault_mid = (d["event_start_rel"] + d["event_end_rel"]) // 2

    # Get the anomaly temperature at the midpoint of the fault window
    if fault_mid < len(d["anom_slice"]):
        # Use a small window around the midpoint for a smoother value
        window_start = max(0, fault_mid - 72)
        window_end = min(len(d["anom_slice"]), fault_mid + 72)
        mid_temp = np.nanmean(d["anom_slice"][window_start:window_end])
        print(f"  Fault window midpoint: x={fault_mid}, temp={mid_temp:.1f}")

        # Place text box to upper-left
        text_x = fault_mid - (d["event_end_rel"] - d["event_start_rel"]) * 0.6
        text_y = mid_temp + (np.nanmax(d["anom_slice"]) - np.nanmin(d["anom_slice"])) * 0.25

        ax.annotate(
            "66 days\nundetected",
            xy=(fault_mid, mid_temp),
            xytext=(text_x, text_y),
            **ANNOTATION_STYLE,
        )

    plt.tight_layout()

    out_path = FIGURES_DIR / "farm_c_cooling_normal_vs_anomaly.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Farm A Generator Bearing Normal vs Anomaly (Event 40 vs 3)
# ─────────────────────────────────────────────────────────────────────────────

def chart_generator_bearing_normal_vs_anomaly():
    print("=" * 60)
    print("Chart 3: farm_a_generator_bearing_normal_vs_anomaly.png (annotated)")
    print("  Event 40 vs 3 — Generator Bearing Temp (sensor_14_avg)")
    print("=" * 60)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    d = _build_comparison_data(
        farm="A",
        anomaly_event_id=40,
        normal_event_id=3,
        sensor_col="sensor_14_avg",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_comparison(
        ax, d,
        sensor_col="sensor_14_avg",
        fault_label="Generator Bearing Failure (31-day event)",
        subsystem_label="Generator Bearing Temp",
        farm="A",
    )

    # ── Annotation: arrow pointing to the ~200 C spike near the end ──
    # Find the index of the maximum temperature in the anomaly slice
    spike_idx = np.nanargmax(d["anom_slice"])
    spike_temp = d["anom_slice"][spike_idx]
    print(f"  Max spike: x={spike_idx}, temp={spike_temp:.1f} C")

    # Place text box to the left and slightly below the spike
    text_x = spike_idx - len(d["anom_slice"]) * 0.2
    text_y = spike_temp * 0.75

    # Slightly larger text for the dramatic spike
    spike_style = dict(ANNOTATION_STYLE)
    spike_style["fontsize"] = 12

    ax.annotate(
        "Spike to\n200\u00b0C",
        xy=(spike_idx, spike_temp),
        xytext=(text_x, text_y),
        **spike_style,
    )

    plt.tight_layout()

    out_path = FIGURES_DIR / "farm_a_generator_bearing_normal_vs_anomaly.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating annotated slide 6 charts...\n")
    chart_gearbox_zoomed()
    chart_cooling_normal_vs_anomaly()
    chart_generator_bearing_normal_vs_anomaly()
    print("All 3 annotated charts saved.")
