"""
Generate a zoomed-in TDI bar chart showing only the interesting events:
all Red, all Yellow, and the top ~8 Green events for contrast.

Saves to: outputs/figures/tdi_all_events_bar_zoomed.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    # Load TDI scores
    csv_path = PROJECT_ROOT / "outputs" / "reports" / "tdi_scores.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events from {csv_path}")

    # Split by status
    red = df[df["tdi_status"] == "Red"]
    yellow = df[df["tdi_status"] == "Yellow"]
    green = df[df["tdi_status"] == "Green"]

    print(f"  Red: {len(red)}, Yellow: {len(yellow)}, Green: {len(green)}")

    # Take top 8 green events (highest TDI among greens) for contrast
    green_top = green.sort_values("tdi_score", ascending=False).head(8)

    # Combine and sort ascending for horizontal bar layout (highest at top)
    subset = pd.concat([red, yellow, green_top], ignore_index=True)
    subset = subset.sort_values("tdi_score", ascending=True).reset_index(drop=True)

    print(f"  Showing {len(subset)} events ({len(red)} red + {len(yellow)} yellow + {len(green_top)} green)")

    # Build chart — same style as original
    n_bars = len(subset)
    fig, ax = plt.subplots(figsize=(12, max(8, n_bars * 0.28)))

    color_map = {"Green": "#2ecc71", "Yellow": "#f39c12", "Red": "#e74c3c"}
    colors = [color_map[s] for s in subset["tdi_status"]]

    # Create labels
    labels = []
    for _, row in subset.iterrows():
        marker = " *" if row["event_label"] == "anomaly" else ""
        labels.append(f"Farm {row['farm']} | Event {int(row['event_id'])}{marker}")

    bars = ax.barh(range(n_bars), subset["tdi_score"], color=colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("TDI Score", fontsize=12)
    ax.set_title("Top Events by Thermal Degradation Index (Red + Yellow + Top Green)",
                  fontsize=13, fontweight="bold")

    # Threshold lines
    ax.axvline(30, color="orange", linestyle="--", alpha=0.7)
    ax.axvline(60, color="red", linestyle="--", alpha=0.7)

    # Legend
    patches = [
        mpatches.Patch(color="#2ecc71", label="Green (0-30)"),
        mpatches.Patch(color="#f39c12", label="Yellow (30-60)"),
        mpatches.Patch(color="#e74c3c", label="Red (60+)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)

    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    # Footnotes
    ax.text(0.99, 0.01, "* = anomaly event", transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", style="italic", alpha=0.6)
    ax.text(0.01, 0.01,
            f"Showing {n_bars} of 95 events (68 green events omitted for clarity)",
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
            style="italic", alpha=0.6)

    plt.tight_layout()

    out_path = PROJECT_ROOT / "outputs" / "figures" / "tdi_all_events_bar_zoomed.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
