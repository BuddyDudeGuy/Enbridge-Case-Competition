"""
Run TDI computation for all 95 events.

Produces:
  - data/processed/tdi_scores.parquet
  - outputs/reports/tdi_scores.csv
  - outputs/reports/tdi_results.json
  - outputs/figures/tdi_distribution_by_label.png
  - outputs/figures/tdi_all_events_bar.png
  - outputs/figures/tdi_per_farm.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tdi_score import compute_tdi, tdi_summary_stats, get_tdi_thresholds


def make_distribution_plot(tdi_df, out_path):
    """Violin + strip plot of TDI by label."""
    fig, ax = plt.subplots(figsize=(10, 6))

    normal = tdi_df[tdi_df["event_label"] == "normal"]["tdi_score"].values
    anomaly = tdi_df[tdi_df["event_label"] == "anomaly"]["tdi_score"].values

    parts = ax.violinplot([normal, anomaly], positions=[0, 1], showmedians=True,
                          showextrema=True)

    # Color violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#2ecc71", "#e74c3c"][i])
        pc.set_alpha(0.4)

    # Overlay individual points with jitter
    for i, (data, color) in enumerate([(normal, "#27ae60"), (anomaly, "#c0392b")]):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter(np.full_like(data, i) + jitter, data, c=color, alpha=0.6, s=30, zorder=5)

    # Threshold lines
    ax.axhline(30, color="orange", linestyle="--", alpha=0.7, label="Yellow threshold (30)")
    ax.axhline(60, color="red", linestyle="--", alpha=0.7, label="Red threshold (60)")

    # Shade regions
    ax.axhspan(0, 30, alpha=0.05, color="green")
    ax.axhspan(30, 60, alpha=0.05, color="orange")
    ax.axhspan(60, 100, alpha=0.05, color="red")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal Events", "Anomaly Events"], fontsize=12)
    ax.set_ylabel("TDI Score (0-100)", fontsize=12)
    ax.set_title("Thermal Degradation Index — Normal vs Anomaly", fontsize=14, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_bar_chart(tdi_df, out_path):
    """Horizontal bar chart of all events sorted by TDI, colored by status."""
    df = tdi_df.sort_values("tdi_score", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(16, len(df) * 0.22)))

    color_map = {"Green": "#2ecc71", "Yellow": "#f39c12", "Red": "#e74c3c"}
    colors = [color_map[s] for s in df["tdi_status"]]

    # Create labels
    labels = []
    for _, row in df.iterrows():
        marker = " *" if row["event_label"] == "anomaly" else ""
        labels.append(f"Farm {row['farm']} | Event {row['event_id']}{marker}")

    bars = ax.barh(range(len(df)), df["tdi_score"], color=colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("TDI Score", fontsize=12)
    ax.set_title("All Events Ranked by Thermal Degradation Index", fontsize=14, fontweight="bold")

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

    # Add note about asterisks
    ax.text(0.99, 0.01, "* = anomaly event", transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", style="italic", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_per_farm_plot(tdi_df, out_path):
    """Box plot of TDI per farm, split by label."""
    farms = sorted(tdi_df["farm"].unique())
    fig, axes = plt.subplots(1, len(farms), figsize=(5 * len(farms), 6), sharey=True)

    if len(farms) == 1:
        axes = [axes]

    for ax, farm in zip(axes, farms):
        farm_df = tdi_df[tdi_df["farm"] == farm]
        normal = farm_df[farm_df["event_label"] == "normal"]["tdi_score"].values
        anomaly = farm_df[farm_df["event_label"] == "anomaly"]["tdi_score"].values

        bp = ax.boxplot([normal, anomaly], tick_labels=["Normal", "Anomaly"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][0].set_alpha(0.4)
        bp["boxes"][1].set_facecolor("#e74c3c")
        bp["boxes"][1].set_alpha(0.4)

        # Overlay points
        for i, (data, color) in enumerate([(normal, "#27ae60"), (anomaly, "#c0392b")]):
            jitter = np.random.normal(0, 0.05, size=len(data))
            ax.scatter(np.full_like(data, i + 1) + jitter, data, c=color, alpha=0.6, s=30, zorder=5)

        ax.axhline(30, color="orange", linestyle="--", alpha=0.5)
        ax.axhline(60, color="red", linestyle="--", alpha=0.5)
        ax.axhspan(0, 30, alpha=0.04, color="green")
        ax.axhspan(30, 60, alpha=0.04, color="orange")
        ax.axhspan(60, 100, alpha=0.04, color="red")

        ax.set_title(f"Farm {farm}", fontsize=13, fontweight="bold")
        ax.set_ylim(-2, 102)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("TDI Score (0-100)", fontsize=12)
    fig.suptitle("TDI Distribution by Farm", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("Phase 6: Thermal Degradation Index (TDI)")
    print("=" * 60)

    # Compute TDI
    print("\nComputing TDI scores...")
    tdi_df = compute_tdi(PROJECT_ROOT)
    print(f"  Computed TDI for {len(tdi_df)} events")

    # Save outputs
    out_data = PROJECT_ROOT / "data" / "processed"
    out_reports = PROJECT_ROOT / "outputs" / "reports"
    out_figures = PROJECT_ROOT / "outputs" / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    tdi_df.to_parquet(out_data / "tdi_scores.parquet", index=False)
    print(f"  Saved: {out_data / 'tdi_scores.parquet'}")

    tdi_df.to_csv(out_reports / "tdi_scores.csv", index=False)
    print(f"  Saved: {out_reports / 'tdi_scores.csv'}")

    # Summary stats
    print("\n" + "-" * 60)
    print("TDI SUMMARY")
    print("-" * 60)
    stats = tdi_summary_stats(tdi_df)

    print(f"\n  Events: {stats['n_events']} total ({stats['n_normal']} normal, {stats['n_anomaly']} anomaly)")
    print(f"\n  Mean TDI — Normal:  {stats['mean_tdi_normal']:.1f}  |  Anomaly: {stats['mean_tdi_anomaly']:.1f}")
    print(f"  Median TDI — Normal: {stats['median_tdi_normal']:.1f}  |  Anomaly: {stats['median_tdi_anomaly']:.1f}")

    print(f"\n  Detection rate (Yellow+Red, TDI>=30): {stats['detection_rate_yellow_plus']:.1%}")
    print(f"  False alarm rate (Yellow+Red, TDI>=30): {stats['false_alarm_rate_yellow_plus']:.1%}")
    print(f"  Detection rate (Red only, TDI>=60):     {stats['detection_rate_red']:.1%}")
    print(f"  False alarm rate (Red only, TDI>=60):   {stats['false_alarm_rate_red']:.1%}")

    print(f"\n  Status distribution:")
    print(f"    Green:  {stats['n_green_total']}")
    print(f"    Yellow: {stats['n_yellow_total']}")
    print(f"    Red:    {stats['n_red_total']}")

    print("\n  Per-farm breakdown:")
    for farm, fs in stats["per_farm"].items():
        det = fs.get("detection_rate_yellow_plus", "N/A")
        det_str = f"{det:.0%}" if isinstance(det, float) else det
        fa = fs.get("false_alarm_rate_yellow_plus", "N/A")
        fa_str = f"{fa:.0%}" if isinstance(fa, float) else fa
        print(f"    Farm {farm}: "
              f"normal={fs['n_normal']}, anomaly={fs['n_anomaly']} | "
              f"mean TDI normal={fs['mean_tdi_normal']:.1f}, anomaly={fs['mean_tdi_anomaly']:.1f} | "
              f"detect={det_str}, FA={fa_str} | "
              f"G={fs['n_green']}/Y={fs['n_yellow']}/R={fs['n_red']}")

    # Print top events
    print("\n  Top 15 events by TDI:")
    print("  " + "-" * 85)
    print(f"  {'Farm':>4} {'Event':>6} {'Label':>8} {'NBM':>6} {'AE':>6} {'Raw':>7} {'TDI':>6} {'Status':>8}")
    print("  " + "-" * 85)
    for _, row in tdi_df.head(15).iterrows():
        print(f"  {row['farm']:>4} {row['event_id']:>6} {row['event_label']:>8} "
              f"{row['nbm_score']:>6.2f} {row['ae_score']:>6.2f} {row['combined_raw']:>7.2f} "
              f"{row['tdi_score']:>6.1f} {row['tdi_status']:>8}")

    # Save report
    report = {
        "phase": "6.1-6.4",
        "description": "Thermal Degradation Index (TDI) — combined NBM + AE health score",
        "weights": {"nbm": 0.7, "ae": 0.3},
        "thresholds": get_tdi_thresholds(),
        "sigmoid_params": {"scale_factor": 0.95, "shift": -1.735},
        "summary": stats,
        "top_10_events": tdi_df.head(10).to_dict(orient="records"),
    }

    with open(out_reports / "tdi_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {out_reports / 'tdi_results.json'}")

    # Visualizations
    print("\nGenerating visualizations...")
    make_distribution_plot(tdi_df, out_figures / "tdi_distribution_by_label.png")
    make_bar_chart(tdi_df, out_figures / "tdi_all_events_bar.png")
    make_per_farm_plot(tdi_df, out_figures / "tdi_per_farm.png")

    print("\n" + "=" * 60)
    print("Phase 6 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
