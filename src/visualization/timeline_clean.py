"""
Clean timeline chart for Farm B Event 53 — Rotor Bearing Degradation.

Plots only the two relevant rotor bearing sensors with a status strip
and fault window shading. No rolling volatility panel.

Usage:
    py src/visualization/timeline_clean.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "CARE_To_Compare", "Wind Farm B", "datasets", "53.csv"
)
EVENT_INFO_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "CARE_To_Compare", "Wind Farm B", "event_info.csv"
)
SAVE_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "figures", "timeline_farm_b_event53_bearing_clean.png"
)

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep=";")
event_info = pd.read_csv(EVENT_INFO_PATH, sep=";")

# Get event 53 metadata
meta = event_info[event_info["event_id"] == 53].iloc[0]
event_start_id = int(meta["event_start_id"])
event_end_id = int(meta["event_end_id"])

# Map IDs to row indices
ev_start_row = df.index[df["id"] == event_start_id][0]
ev_end_row = df.index[df["id"] == event_end_id][0]

# X-axis in days (each row = 10 min, 144 rows per day)
days = np.arange(len(df)) / 144.0

# ── Status color mapping (all values actually present in data) ───────────
STATUS_COLORS = {
    0: "#2ecc71",  # Normal  — green
    1: "#f1c40f",  # Service — yellow
    2: "#3498db",  # Idle    — blue
    3: "#e74c3c",  # Downtime — red
    4: "#e74c3c",  # Downtime — red  (actual value in data)
    5: "#95a5a6",  # Other   — gray  (actual value in data)
}
STATUS_LABELS = {
    0: "Normal",
    1: "Service / Derated",
    2: "Idle",
    3: "Downtime",
    4: "Downtime",
    5: "Other",
}

# ── Figure setup: 2 panels (temp + status) ───────────────────────────────
fig, (ax_top, ax_mid) = plt.subplots(
    2, 1,
    figsize=(14, 6),
    gridspec_kw={"height_ratios": [5, 1]},
    sharex=True,
)
fig.subplots_adjust(hspace=0.06)

fig.suptitle(
    "Farm B \u2014 Rotor Bearing Degradation (42 days)",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

# ── Top panel: temperature time series ───────────────────────────────────
# Fault sensor — bold prominent line
ax_top.plot(
    days,
    df["sensor_52_avg"],
    color="#c0392b",
    linewidth=1.4,
    alpha=0.9,
    label="Rotor Bearing 2 (fault)",
    zorder=3,
)

# Reference sensor — thin gray line
ax_top.plot(
    days,
    df["sensor_51_avg"],
    color="#7f8c8d",
    linewidth=0.7,
    alpha=0.6,
    label="Rotor Bearing 1 (reference)",
    zorder=2,
)

# Fault window shading
ev_start_day = ev_start_row / 144.0
ev_end_day = ev_end_row / 144.0
ax_top.axvspan(ev_start_day, ev_end_day, color="#e74c3c", alpha=0.10, zorder=0)

ax_top.set_ylabel("Temperature (\u00b0C)", fontsize=11)
ax_top.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_top.grid(True, alpha=0.25, linewidth=0.5)

# ── Middle panel: status strip ───────────────────────────────────────────
status = df["status_type_id"].values

# Build contiguous segments
segments = []
seg_start = 0
for i in range(1, len(status)):
    if status[i] != status[seg_start]:
        segments.append((seg_start, i - seg_start, int(status[seg_start])))
        seg_start = i
segments.append((seg_start, len(status) - seg_start, int(status[seg_start])))

for seg_x, seg_w, seg_status in segments:
    x_day = seg_x / 144.0
    w_day = seg_w / 144.0
    color = STATUS_COLORS.get(seg_status, "#95a5a6")
    ax_mid.barh(0, w_day, left=x_day, height=1.0, color=color, edgecolor="none")

# Fault window shading on status panel too
ax_mid.axvspan(ev_start_day, ev_end_day, color="#e74c3c", alpha=0.10, zorder=0)

ax_mid.set_ylim(-0.5, 0.5)
ax_mid.set_yticks([])
ax_mid.set_ylabel("Status", fontsize=10)
ax_mid.set_xlabel("Days", fontsize=11)

# Status legend — only statuses actually present
present = sorted(set(int(s) for s in df["status_type_id"].unique()))
# De-duplicate labels (4 and 3 both map to "Downtime")
seen_labels = set()
status_patches = []
for s in present:
    lbl = STATUS_LABELS.get(s, f"Unknown ({s})")
    if lbl not in seen_labels:
        seen_labels.add(lbl)
        status_patches.append(
            mpatches.Patch(color=STATUS_COLORS.get(s, "#95a5a6"), label=lbl)
        )
ax_mid.legend(
    handles=status_patches,
    loc="upper left",
    fontsize=7,
    framealpha=0.9,
    ncol=len(status_patches),
)

# X-axis limit in days
ax_mid.set_xlim(0, len(df) / 144.0)

# ── Save ─────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {SAVE_PATH}")
plt.close(fig)
