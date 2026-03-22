"""
Zoomed-in EDA chart for Farm A Gearbox Oil Temperature (Event 72).

Shows the prediction window with a buffer of training context,
rolling 24-hour average, healthy baseline reference, and fault shading.

Output: outputs/figures/farm_a_gearbox_zoomed.png
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "D:/Personal Projects/Enbridge Case Compettion"
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIGURES_DIR = Path(PROJECT_ROOT) / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def generate_farm_a_gearbox_zoomed():
    print("=" * 60)
    print("Generating: farm_a_gearbox_zoomed.png")
    print("  Event 72 — Gearbox Oil Temperature (sensor_12_avg)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(
        f"{PROJECT_ROOT}/data/raw/CARE_To_Compare/Wind Farm A/datasets/72.csv",
        sep=";"
    )
    print(f"  Loaded event 72: {df.shape[0]:,} rows x {df.shape[1]} cols")

    sensor = "sensor_12_avg"
    train_mask = df["train_test"] == "train"
    pred_mask = df["train_test"] == "prediction"

    pred_start_idx = pred_mask.idxmax()  # first prediction row index
    print(f"  Prediction starts at row {pred_start_idx}")

    # Event window from event_info.csv: event_start_id=52497, event_end_id=53505
    evt_start = 52497
    evt_end = 53505

    # ------------------------------------------------------------------
    # Compute healthy baseline (normal-operation training mean)
    # ------------------------------------------------------------------
    normal_train = df[train_mask & df["status_type_id"].isin([0, 2])]
    healthy_mean = normal_train[sensor].mean()
    print(f"  Healthy baseline (training mean, normal ops): {healthy_mean:.2f} °C")

    # ------------------------------------------------------------------
    # Zoom window: 500 training rows before prediction + all prediction
    # ------------------------------------------------------------------
    buffer = 500
    zoom_start = max(0, pred_start_idx - buffer)
    zoom_df = df.iloc[zoom_start:].copy()
    zoom_df["row"] = np.arange(len(zoom_df))

    # Rolling 24-hour average: 144 rows at 10-min intervals
    zoom_df["rolling_24h"] = zoom_df[sensor].rolling(window=144, center=False, min_periods=1).mean()

    # Convert indices to zoom_df row-space for shading
    pred_row = pred_start_idx - zoom_start
    evt_start_row = max(evt_start - zoom_start, 0)
    evt_end_row = min(evt_end - zoom_start, len(zoom_df) - 1)

    print(f"  Zoom window: rows {zoom_start}–{zoom_start + len(zoom_df) - 1} ({len(zoom_df):,} rows)")
    print(f"  Event shading: zoom rows {evt_start_row}–{evt_end_row}")

    # ------------------------------------------------------------------
    # X-axis: convert row index to days relative to prediction start
    # ------------------------------------------------------------------
    rows_per_day = 6 * 24  # 144 rows per day
    zoom_df["days"] = (zoom_df["row"] - pred_row) / rows_per_day

    pred_day = 0.0
    evt_start_day = (evt_start_row - pred_row) / rows_per_day
    evt_end_day = (evt_end_row - pred_row) / rows_per_day

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw sensor — light/transparent
    ax.plot(
        zoom_df["days"], zoom_df[sensor],
        color="#7BAFD4", alpha=0.35, linewidth=0.7,
        label="Raw sensor (10-min)"
    )

    # Rolling 24h average — bold, prominent
    ax.plot(
        zoom_df["days"], zoom_df["rolling_24h"],
        color="#1B4F72", linewidth=2.2,
        label="Rolling 24-hr average"
    )

    # Healthy baseline
    ax.axhline(
        y=healthy_mean, color="#2ECC71", linewidth=1.8, linestyle="--",
        label=f"Healthy baseline ({healthy_mean:.1f} °C)"
    )

    # Shade event/fault window
    ax.axvspan(
        evt_start_day, evt_end_day,
        color="#E74C3C", alpha=0.12, label="Fault window"
    )

    # Vertical line at prediction start
    ax.axvline(
        x=pred_day, color="#888888", linewidth=1.0, linestyle=":",
        label="Prediction start"
    )

    # ------------------------------------------------------------------
    # Labels, legend, formatting
    # ------------------------------------------------------------------
    ax.set_title(
        "Farm A — Gearbox Oil Temperature (Zoomed to Fault Window)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.set_xlabel("Days relative to prediction start", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = FIGURES_DIR / "farm_a_gearbox_zoomed.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    print()


if __name__ == "__main__":
    generate_farm_a_gearbox_zoomed()
    print("Done.")
