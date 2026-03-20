"""
Batch 3 chart fixes — re-generate specific EDA figures with corrections.

Fixes:
1. cross_farm_hourly_gearbox_temp.png — remove "(from time_stamp)" from x-axis
2. event_duration_barchart.png — top 15 longest events only, bigger y-label font
3. timeline_farm_c_event44_cooling.png — fix duration discrepancy (actual: 65 days)
4. All 4 Phase 2.6 correlation breakdown charts — higher DPI, larger figsize,
   bigger legend font, shortened sensor names
"""

import sys
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("D:/Personal Projects/Enbridge Case Compettion")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.data.load_data import (
    load_farm_training_data, load_event, load_event_info, get_event_ids, clear_cache,
)
from src.features.thermal_config import THERMAL_SUBSYSTEMS

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "CARE_To_Compare"

# Consistent farm colors
FARM_COLORS = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
FARM_PALETTE = [FARM_COLORS["A"], FARM_COLORS["B"], FARM_COLORS["C"]]

# Load event info once
event_info = load_event_info()

# Representative gearbox sensors per farm
GEARBOX_SENSORS = {
    "A": "sensor_12_avg",
    "B": "sensor_39_avg",
    "C": "sensor_186_avg",
}


# ===================================================================
# FIX 1: cross_farm_hourly_gearbox_temp.png
# Remove "(from time_stamp)" from x-axis label
# ===================================================================
def fix_hourly_gearbox_temp():
    print("\n[1/4] Fixing cross_farm_hourly_gearbox_temp.png ...")

    print("  Loading full training data for all farms...")
    df_a_full = load_farm_training_data("A")
    df_b_full = load_farm_training_data("B")
    df_c_full = load_farm_training_data("C")
    farm_full_dfs = {"A": df_a_full, "B": df_b_full, "C": df_c_full}

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    for farm_letter in ["A", "B", "C"]:
        df_full = farm_full_dfs[farm_letter]
        temp_col = GEARBOX_SENSORS[farm_letter]

        # Try to parse time_stamp if it exists
        has_timestamp = False
        if "time_stamp" in df_full.columns:
            try:
                ts = pd.to_datetime(df_full["time_stamp"], errors="coerce")
                if ts.notna().sum() > 1000:
                    hour = ts.dt.hour
                    has_timestamp = True
            except Exception:
                pass

        if has_timestamp:
            hourly_df = pd.DataFrame({"hour": hour, "temp": df_full[temp_col]})
        else:
            proxy_hour = (df_full.index % 144) // 6
            hourly_df = pd.DataFrame({"hour": proxy_hour, "temp": df_full[temp_col]})

        hourly_avg = hourly_df.groupby("hour")["temp"].agg(["mean", "std"]).reset_index()

        ax.plot(
            hourly_avg["hour"], hourly_avg["mean"],
            color=FARM_COLORS[farm_letter], linewidth=2.5,
            label=f"Farm {farm_letter}", marker="o", markersize=4,
        )
        ax.fill_between(
            hourly_avg["hour"],
            hourly_avg["mean"] - hourly_avg["std"],
            hourly_avg["mean"] + hourly_avg["std"],
            color=FARM_COLORS[farm_letter], alpha=0.15,
        )

    # FIX: just "Hour of Day" — no parenthetical note
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Gearbox Oil Temperature (\u00b0C)")
    ax.set_title("Gearbox Oil Temperature \u2014 Hourly Patterns Across Farms",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out = FIGURES_DIR / "cross_farm_hourly_gearbox_temp.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")

    # Free memory
    del df_a_full, df_b_full, df_c_full, farm_full_dfs
    clear_cache()


# ===================================================================
# FIX 2: event_duration_barchart.png
# Filter to top 15 longest events, increase y-axis label font size
# ===================================================================
def fix_event_duration_barchart():
    print("\n[2/4] Fixing event_duration_barchart.png ...")

    anomaly_events = event_info[event_info["event_label"] == "anomaly"].copy()
    anomaly_events["event_start_dt"] = pd.to_datetime(anomaly_events["event_start"])
    anomaly_events["event_end_dt"] = pd.to_datetime(anomaly_events["event_end"])
    anomaly_events["duration_days"] = (
        (anomaly_events["event_end_dt"] - anomaly_events["event_start_dt"])
        .dt.total_seconds() / 86400
    )

    # Top 15 longest only
    top15 = anomaly_events.nlargest(15, "duration_days").copy()
    top15 = top15.sort_values("duration_days", ascending=True).reset_index(drop=True)

    def short_desc(desc, max_len=50):
        if len(desc) <= max_len:
            return desc
        return desc[:max_len - 3] + "..."

    top15["short_desc"] = top15["event_description"].apply(short_desc)
    top15["bar_label"] = (
        "Farm " + top15["farm"] + " #" + top15["event_id"].astype(str)
        + ": " + top15["short_desc"]
    )

    farm_colors = {"A": "#3498db", "B": "#e67e22", "C": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(
        range(len(top15)),
        top15["duration_days"],
        color=[farm_colors[f] for f in top15["farm"]],
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # Duration labels on bars
    for i, (_, row) in enumerate(top15.iterrows()):
        duration = row["duration_days"]
        ax.text(duration + 0.5, i, f"{duration:.1f}d",
                va="center", fontsize=9, color="#2c3e50", fontweight="bold")

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["bar_label"], fontsize=10)  # bigger font
    ax.set_xlabel("Event Duration (days)", fontsize=12)
    ax.set_title("Top 15 Longest Anomaly Events Across Fleet",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Farm legend
    farm_patches = [mpatches.Patch(color=farm_colors[f], label=f"Farm {f}")
                    for f in ["A", "B", "C"]]
    ax.legend(handles=farm_patches, loc="lower right", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "event_duration_barchart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")


# ===================================================================
# FIX 3: timeline_farm_c_event44_cooling.png
# Fix duration discrepancy — actual duration is 65 days
# ===================================================================

# Status type color map (same as notebook 08)
STATUS_COLORS = {
    0: "#2ecc71", 1: "#f1c40f", 2: "#00bcd4",
    3: "#9b59b6", 4: "#e74c3c", 5: "#95a5a6",
}
STATUS_LABELS = {
    0: "Normal", 1: "Derated", 2: "Idling",
    3: "Service", 4: "Downtime", 5: "Other",
}
SENSOR_COLORS_TIMELINE = [
    "#2980b9", "#e67e22", "#27ae60", "#e74c3c", "#8e44ad", "#1abc9c",
]
SENSOR_LABELS = {
    "sensor_46_avg": "Water Temp Flow TT1",
    "sensor_175_avg": "Axis 3 Cooling Element",
    "sensor_176_avg": "Cooling Air Exit Temp",
    "sensor_208_avg": "Axis 1 Cooling Element",
}


def get_sensor_label(sensor_col):
    if sensor_col in SENSOR_LABELS:
        return SENSOR_LABELS[sensor_col]
    return sensor_col.replace("_avg", "").replace("_", " ").title()


def fix_timeline_event44():
    print("\n[3/4] Fixing timeline_farm_c_event44_cooling.png ...")

    farm = "C"
    event_id = 44
    sensors = ["sensor_46_avg", "sensor_175_avg", "sensor_176_avg", "sensor_208_avg"]

    df = load_event(farm, event_id)
    meta = event_info[
        (event_info["farm"] == farm.upper()) & (event_info["event_id"] == event_id)
    ].iloc[0]

    event_start_id = int(meta["event_start_id"])
    event_end_id = int(meta["event_end_id"])
    event_desc = meta["event_description"]

    # Compute correct duration
    start_ts = pd.Timestamp(meta["event_start"])
    end_ts = pd.Timestamp(meta["event_end"])
    duration_days = (end_ts - start_ts).days  # 65

    # Filter to valid sensors
    valid_sensors = [s for s in sensors if s in df.columns]
    primary_sensor = valid_sensors[0]
    rolling_std = df[primary_sensor].rolling(window=144, min_periods=72).std()

    x = np.arange(len(df))
    train_mask = df["train_test"] == "train"
    boundary_idx = train_mask.sum()

    # ---- Figure ----
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [4, 1, 2]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # Use consistent duration in title
    title = f"Farm {farm} \u2014 Event {event_id}: Cooling Valve Misposition ({duration_days} days)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # -- TOP PANEL --
    ax_top = axes[0]
    for i, sensor in enumerate(valid_sensors):
        color = SENSOR_COLORS_TIMELINE[i % len(SENSOR_COLORS_TIMELINE)]
        label = get_sensor_label(sensor)
        ax_top.plot(x, df[sensor], color=color, linewidth=0.6, alpha=0.85, label=label)

    ax_top.set_ylabel("Temperature (\u00b0C)", fontsize=10)
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax_top.grid(True, alpha=0.3, linewidth=0.5)
    ax_top.set_title("Thermal Sensors", fontsize=10, loc="left", style="italic", pad=4)

    # -- MIDDLE PANEL --
    ax_mid = axes[1]
    status = df["status_type_id"].values
    segments = []
    seg_start = 0
    for i in range(1, len(status)):
        if status[i] != status[seg_start]:
            segments.append((seg_start, i - seg_start, status[seg_start]))
            seg_start = i
    segments.append((seg_start, len(status) - seg_start, status[seg_start]))

    for seg_x, seg_w, seg_status in segments:
        color = STATUS_COLORS.get(int(seg_status), "#95a5a6")
        ax_mid.barh(0, seg_w, left=seg_x, height=1.0, color=color, edgecolor="none")

    ax_mid.set_ylim(-0.5, 0.5)
    ax_mid.set_yticks([])
    ax_mid.set_ylabel("Status", fontsize=10)

    present_statuses = sorted(set(int(s) for s in df["status_type_id"].unique()))
    status_patches = [
        mpatches.Patch(
            color=STATUS_COLORS.get(s, "#95a5a6"),
            label=STATUS_LABELS.get(s, f"Unknown ({s})"),
        )
        for s in present_statuses
    ]
    ax_mid.legend(
        handles=status_patches, loc="upper left", fontsize=7,
        framealpha=0.9, ncol=len(present_statuses),
    )

    # -- BOTTOM PANEL --
    ax_bot = axes[2]
    ax_bot.fill_between(x, 0, rolling_std, color="#3498db", alpha=0.4)
    ax_bot.plot(x, rolling_std, color="#2980b9", linewidth=0.7, alpha=0.8)
    ax_bot.set_ylabel(f"Rolling Std\n({get_sensor_label(primary_sensor)})", fontsize=9)
    ax_bot.set_xlabel("Row Index", fontsize=10)
    ax_bot.grid(True, alpha=0.3, linewidth=0.5)
    ax_bot.set_title("Volatility (1-day rolling std)", fontsize=10, loc="left",
                     style="italic", pad=4)

    # -- ANNOTATIONS --
    for ax in axes:
        ax.axvline(boundary_idx, color="#2c3e50", linestyle="--", linewidth=1.2, alpha=0.8)
        if "id" in df.columns:
            start_rows = df.index[df["id"] == event_start_id]
            end_rows = df.index[df["id"] == event_end_id]
            if len(start_rows) > 0 and len(end_rows) > 0:
                ev_start_row = start_rows[0]
                ev_end_row = end_rows[0]
            else:
                ev_start_row = max(0, event_start_id - df["id"].iloc[0])
                ev_end_row = min(len(df) - 1, event_end_id - df["id"].iloc[0])
        else:
            ev_start_row = event_start_id
            ev_end_row = min(event_end_id, len(df) - 1)
        ax.axvspan(ev_start_row, ev_end_row, color="red", alpha=0.12, zorder=0)

    # Train/prediction label
    ax_top.text(
        boundary_idx, ax_top.get_ylim()[1], "  Train | Prediction ",
        fontsize=8, color="#2c3e50", fontweight="bold",
        verticalalignment="top", horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#2c3e50", alpha=0.85),
    )

    # Event description — USE CORRECT duration_days (65)
    ev_mid = (ev_start_row + ev_end_row) / 2
    ypos = ax_top.get_ylim()[0] + 0.75 * (ax_top.get_ylim()[1] - ax_top.get_ylim()[0])
    event_text = f"Cooling Valve Misposition\n({duration_days} days)"
    ax_top.annotate(
        event_text, xy=(ev_mid, ypos),
        fontsize=8, color="#c0392b", fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8",
                  edgecolor="#e74c3c", alpha=0.9),
    )

    # Secondary x-axis: approximate days
    total_days = len(df) / 144.0
    day_ticks = np.arange(0, total_days, max(1, int(total_days // 10)))
    ax_sec = ax_bot.secondary_xaxis("bottom")
    ax_sec.set_xticks(day_ticks * 144)
    ax_sec.set_xticklabels([f"Day {int(d)}" for d in day_ticks], fontsize=7, color="#7f8c8d")
    ax_sec.spines["bottom"].set_position(("outward", 25))

    ax_bot.set_xlim(0, len(df))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = FIGURES_DIR / "timeline_farm_c_event44_cooling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  Duration in title AND annotation: {duration_days} days (consistent)")

    clear_cache()


# ===================================================================
# FIX 4: Four Phase 2.6 correlation breakdown charts
# Higher DPI (200), bigger figsize (~20%), larger legend font,
# shortened sensor names in legends
# ===================================================================

# Human-friendly abbreviations for correlation chart legends
CORR_SENSOR_ABBREVS = {
    # Farm A gearbox
    "sensor_11_avg": "Gearbox HSS Bearing",
    "sensor_12_avg": "Gearbox Oil",
    # Farm A generator bearings
    "sensor_13_avg": "Gen Bearing DE",
    "sensor_14_avg": "Gen Bearing NDE",
    "sensor_15_avg": "Stator Winding P1",
    "sensor_16_avg": "Stator Winding P2",
    # Farm B rotor bearings
    "sensor_51_avg": "Rotor Bearing 1",
    "sensor_52_avg": "Rotor Bearing 2",
    "sensor_32_avg": "Gen Bearing 1",
    "sensor_33_avg": "Gen Bearing 2",
}

WINDOW = 144
CORR_THRESHOLD = 0.7


def compute_rolling_corr(df, col_a, col_b, window=WINDOW):
    return df[col_a].rolling(window=window, min_periods=window // 2).corr(df[col_b])


def get_corr_label(sensor_col):
    """Short human-friendly sensor label for correlation charts."""
    return CORR_SENSOR_ABBREVS.get(sensor_col, sensor_col.replace("_avg", ""))


def plot_correlation_breakdown_fixed(
    df_anomaly, df_normal, sensor_a, sensor_b,
    label_a, label_b,
    anomaly_title, normal_title,
    event_info_anomaly, event_info_normal,
    suptitle, window=WINDOW,
):
    """Same as notebook but with bigger figsize, DPI=200, larger legend fonts."""

    # ~20% larger than original 16x9
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8), constrained_layout=True)

    datasets = [
        (df_normal, normal_title, event_info_normal, 0),
        (df_anomaly, anomaly_title, event_info_anomaly, 1),
    ]

    for df, title, ev_info, col_idx in datasets:
        ax_top = axes[0][col_idx]
        ax_bot = axes[1][col_idx]

        pred_mask = df["train_test"] == "prediction"
        if pred_mask.any():
            pred_start_idx = pred_mask.idxmax()
            context_start = max(0, pred_start_idx - 1000)
            plot_df = df.iloc[context_start:].copy().reset_index(drop=True)
            boundary_idx = min(1000, pred_start_idx - context_start)
        else:
            plot_df = df.copy().reset_index(drop=True)
            boundary_idx = None

        x = np.arange(len(plot_df))
        rolling_corr = compute_rolling_corr(plot_df, sensor_a, sensor_b, window)

        # -- Top: raw sensor values --
        ax_top.plot(x, plot_df[sensor_a], color="#2196F3", linewidth=0.8,
                    alpha=0.85, label=label_a)
        ax_top.plot(x, plot_df[sensor_b], color="#FF9800", linewidth=0.8,
                    alpha=0.85, label=label_b)

        if boundary_idx is not None:
            ax_top.axvline(boundary_idx, color="black", linestyle="--",
                           linewidth=1.2, alpha=0.7, label="Train/Pred split")

        # Event shading
        event_start_id = ev_info.get("event_start_id", None)
        event_end_id = ev_info.get("event_end_id", None)
        if event_start_id is not None and boundary_idx is not None:
            pred_start_original = pred_mask.idxmax() if pred_mask.any() else 0
            ev_start_local = int(event_start_id) - (pred_start_original - boundary_idx)
            ev_end_local = int(event_end_id) - (pred_start_original - boundary_idx)
            ev_start_local = max(0, ev_start_local - context_start if context_start > 0 else ev_start_local)
            ev_end_local = max(0, ev_end_local - context_start if context_start > 0 else ev_end_local)
            if 0 < ev_start_local < len(plot_df) or 0 < ev_end_local < len(plot_df):
                ax_top.axvspan(max(0, ev_start_local), min(len(plot_df), ev_end_local),
                               color="red", alpha=0.08, label="Event window")

        ax_top.set_title(title, fontweight="bold", fontsize=12)
        ax_top.set_ylabel("Temperature", fontsize=11)
        ax_top.legend(loc="upper left", fontsize=10, ncol=2)  # bigger legend
        ax_top.grid(True, alpha=0.3)

        # -- Bottom: rolling correlation --
        corr_vals = rolling_corr.values
        corr_healthy = np.where(corr_vals >= CORR_THRESHOLD, corr_vals, np.nan)
        corr_broken = np.where(corr_vals < CORR_THRESHOLD, corr_vals, np.nan)

        ax_bot.plot(x, corr_healthy, color="#2196F3", linewidth=1.0, alpha=0.9)
        ax_bot.plot(x, corr_broken, color="#F44336", linewidth=1.2, alpha=0.9)

        ax_bot.axhline(0.95, color="gray", linestyle=":", linewidth=1.0,
                       alpha=0.6, label="Normal baseline (0.95)")
        ax_bot.axhline(CORR_THRESHOLD, color="#F44336", linestyle="--",
                       linewidth=1.0, alpha=0.5, label=f"Threshold ({CORR_THRESHOLD})")

        if boundary_idx is not None:
            ax_bot.axvline(boundary_idx, color="black", linestyle="--",
                           linewidth=1.2, alpha=0.7)

        ax_bot.set_ylim(-0.5, 1.05)
        ax_bot.set_ylabel("Rolling Correlation (r)", fontsize=11)
        ax_bot.set_xlabel("Time index (10-min intervals)", fontsize=11)
        ax_bot.legend(loc="lower left", fontsize=10)  # bigger legend
        ax_bot.grid(True, alpha=0.3)
        ax_bot.fill_between(x, -0.5, CORR_THRESHOLD, color="#F44336", alpha=0.03)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    return fig


def fix_correlation_charts():
    print("\n[4/4] Fixing 4 correlation breakdown charts ...")

    # ------------------------------------------------------------------
    # Chart A: Farm A Gearbox — Event 72 (anomaly) vs Event 13 (normal)
    # ------------------------------------------------------------------
    print("  Loading Farm A gearbox events (72 anomaly, 13 normal)...")
    df_a72 = load_event("A", 72)
    df_a13 = load_event("A", 13)

    ev72 = event_info[(event_info["farm"] == "A") & (event_info["event_id"] == 72)].iloc[0]
    ev13 = event_info[(event_info["farm"] == "A") & (event_info["event_id"] == 13)].iloc[0]

    fig = plot_correlation_breakdown_fixed(
        df_anomaly=df_a72, df_normal=df_a13,
        sensor_a="sensor_11_avg", sensor_b="sensor_12_avg",
        label_a=get_corr_label("sensor_11_avg"),
        label_b=get_corr_label("sensor_12_avg"),
        anomaly_title="Event 72 \u2014 Gearbox Failure (Asset 21)",
        normal_title="Event 13 \u2014 Normal Operation (Asset 21)",
        event_info_anomaly=ev72, event_info_normal=ev13,
        suptitle="Farm A \u2014 Gearbox: Rolling Correlation Breakdown During Failure",
    )
    out = FIGURES_DIR / "corr_breakdown_farm_a_gearbox.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")

    clear_cache()

    # ------------------------------------------------------------------
    # Chart B: Farm B Bearing — Event 53 (anomaly) vs Event 23 (normal)
    # ------------------------------------------------------------------
    print("  Loading Farm B bearing events (53 anomaly, 23 normal)...")
    df_b53 = load_event("B", 53)
    df_b23 = load_event("B", 23)

    ev53 = event_info[(event_info["farm"] == "B") & (event_info["event_id"] == 53)].iloc[0]
    ev23 = event_info[(event_info["farm"] == "B") & (event_info["event_id"] == 23)].iloc[0]

    fig = plot_correlation_breakdown_fixed(
        df_anomaly=df_b53, df_normal=df_b23,
        sensor_a="sensor_51_avg", sensor_b="sensor_52_avg",
        label_a=get_corr_label("sensor_51_avg"),
        label_b=get_corr_label("sensor_52_avg"),
        anomaly_title="Event 53 \u2014 Rotor Bearing 2 Damage (Asset 6)",
        normal_title="Event 23 \u2014 Normal Operation (Asset 6)",
        event_info_anomaly=ev53, event_info_normal=ev23,
        suptitle="Farm B \u2014 Rotor Bearings: Gradual Correlation Decline During Degradation",
    )
    out = FIGURES_DIR / "corr_breakdown_farm_b_bearing.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")

    clear_cache()

    # ------------------------------------------------------------------
    # Chart C: Farm A Gen Bearing — Event 40 (anomaly) vs Event 3 (normal)
    # ------------------------------------------------------------------
    print("  Loading Farm A gen bearing events (40 anomaly, 3 normal)...")
    df_a40 = load_event("A", 40)
    df_a03 = load_event("A", 3)

    ev40 = event_info[(event_info["farm"] == "A") & (event_info["event_id"] == 40)].iloc[0]
    ev03 = event_info[(event_info["farm"] == "A") & (event_info["event_id"] == 3)].iloc[0]

    fig = plot_correlation_breakdown_fixed(
        df_anomaly=df_a40, df_normal=df_a03,
        sensor_a="sensor_13_avg", sensor_b="sensor_14_avg",
        label_a=get_corr_label("sensor_13_avg"),
        label_b=get_corr_label("sensor_14_avg"),
        anomaly_title="Event 40 \u2014 Generator Bearing Failure (Asset 10)",
        normal_title="Event 3 \u2014 Normal Operation (Asset 10)",
        event_info_anomaly=ev40, event_info_normal=ev03,
        suptitle="Farm A \u2014 Generator Bearings: Rolling Correlation During Bearing Failure",
    )
    out = FIGURES_DIR / "corr_breakdown_farm_a_gen_bearing.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")

    clear_cache()

    # ------------------------------------------------------------------
    # Chart D: Boxplots — anomaly vs normal correlation stats (Farm A)
    # ------------------------------------------------------------------
    print("  Computing correlation stats for all Farm A events...")
    s_gb_a, s_gb_b = "sensor_11_avg", "sensor_12_avg"
    s_gen_a, s_gen_b = "sensor_13_avg", "sensor_14_avg"

    anomaly_ids = get_event_ids("A", "anomaly")
    normal_ids = get_event_ids("A", "normal")

    results = []
    for label_type, event_ids in [("anomaly", anomaly_ids), ("normal", normal_ids)]:
        for eid in event_ids:
            try:
                df = load_event("A", eid)
                pred = df[df["train_test"] == "prediction"]
                if len(pred) < WINDOW:
                    continue

                rc_gb = compute_rolling_corr(pred, s_gb_a, s_gb_b)
                rc_gb_clean = rc_gb.replace([np.inf, -np.inf], np.nan).dropna()
                rc_gen = compute_rolling_corr(pred, s_gen_a, s_gen_b)
                rc_gen_clean = rc_gen.replace([np.inf, -np.inf], np.nan).dropna()

                results.append({
                    "event_id": eid,
                    "label": label_type,
                    "gb_mean_r": rc_gb_clean.mean() if len(rc_gb_clean) > 0 else np.nan,
                    "gb_min_r": rc_gb_clean.min() if len(rc_gb_clean) > 0 else np.nan,
                    "gb_std_r": rc_gb_clean.std() if len(rc_gb_clean) > 0 else np.nan,
                    "gen_mean_r": rc_gen_clean.mean() if len(rc_gen_clean) > 0 else np.nan,
                    "gen_min_r": rc_gen_clean.min() if len(rc_gen_clean) > 0 else np.nan,
                    "gen_std_r": rc_gen_clean.std() if len(rc_gen_clean) > 0 else np.nan,
                })
            except Exception as e:
                print(f"    Skipping event {eid}: {e}")

    results_df = pd.DataFrame(results)

    # ~20% bigger than original 16x9
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), constrained_layout=True)

    # Shortened metric titles
    metrics = [
        ("gb_mean_r", "Gearbox \u2014 Mean Corr"),
        ("gb_min_r", "Gearbox \u2014 Min Corr"),
        ("gb_std_r", "Gearbox \u2014 Corr Std Dev"),
        ("gen_mean_r", "Gen Bearing \u2014 Mean Corr"),
        ("gen_min_r", "Gen Bearing \u2014 Min Corr"),
        ("gen_std_r", "Gen Bearing \u2014 Corr Std Dev"),
    ]
    colors = {"normal": "#4CAF50", "anomaly": "#F44336"}

    for idx, (col, title) in enumerate(metrics):
        row, c = divmod(idx, 3)
        ax = axes[row][c]

        data_normal = results_df[results_df["label"] == "normal"][col].dropna()
        data_anomaly = results_df[results_df["label"] == "anomaly"][col].dropna()

        bp = ax.boxplot(
            [data_normal, data_anomaly],
            tick_labels=["Normal", "Anomaly"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], [colors["normal"], colors["anomaly"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (data, label) in enumerate(
            [(data_normal, "normal"), (data_anomaly, "anomaly")]
        ):
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(data))
            ax.scatter(
                np.full(len(data), i + 1) + jitter, data,
                color=colors[label], edgecolor="white", s=45, alpha=0.8, zorder=5,
            )

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel(
            col.split("_")[-1].replace("r", "Correlation").replace("std", "Std Dev"),
            fontsize=11,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=10)

    fig.suptitle(
        "Farm A \u2014 Correlation Statistics: Normal vs Anomaly Events (Prediction Window)",
        fontsize=14, fontweight="bold",
    )

    out = FIGURES_DIR / "corr_breakdown_boxplots_farm_a.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")

    clear_cache()


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Batch 3 Chart Fixes")
    print("=" * 65)

    fix_hourly_gearbox_temp()
    fix_event_duration_barchart()
    fix_timeline_event44()
    fix_correlation_charts()

    print("\n" + "=" * 65)
    print("  All fixes applied. Summary of updated files:")
    print("=" * 65)
    targets = [
        "cross_farm_hourly_gearbox_temp.png",
        "event_duration_barchart.png",
        "timeline_farm_c_event44_cooling.png",
        "corr_breakdown_farm_a_gearbox.png",
        "corr_breakdown_farm_b_bearing.png",
        "corr_breakdown_farm_a_gen_bearing.png",
        "corr_breakdown_boxplots_farm_a.png",
    ]
    for name in targets:
        p = FIGURES_DIR / name
        if p.exists():
            print(f"  {name:50s} ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  {name:50s} MISSING!")
    print()
