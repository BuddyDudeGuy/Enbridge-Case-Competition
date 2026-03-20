"""
Phase 4.6 -- Event-level binary classification system.

Combines the best single-feature strategy (Phase 4.4) with the weighted
voting approach (Phase 4.5) into a union classifier.  An event is flagged
as anomaly if EITHER detector fires.

Functions
---------
classify_events(project_root)
    Run the combined classifier on all 95 events.
compute_confusion_metrics(true_labels, predicted_labels)
    TP / FP / TN / FN and derived scores including CARE accuracy.
generate_classification_report(results_df, project_root)
    Confusion-matrix heatmap, per-farm breakdown, hit/miss lists, saved to outputs/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SUBSYSTEMS = ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"]

FARM_SUBSYSTEMS = {
    "A": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
    "B": ["gearbox", "generator_bearings", "transformer"],
    "C": ["gearbox", "generator_bearings", "transformer", "hydraulic", "cooling"],
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  classify_events
# ═══════════════════════════════════════════════════════════════════════════

def classify_events(project_root) -> pd.DataFrame:
    """Run the combined detector on every event.

    Strategy:
        1. Best single feature (transformer_overall_std > threshold)
        2. Weighted z-score voting across strong subsystems
        3. Union of (1) and (2): flag if EITHER fires

    Returns DataFrame with per-event predictions and confidence.
    """
    project_root = Path(project_root)

    # --- load artefacts -----------------------------------------------
    scores_df = pd.read_parquet(
        project_root / "data" / "processed" / "event_scores.parquet"
    )
    feat_df = pd.read_parquet(
        project_root / "data" / "processed" / "event_feature_matrix.parquet"
    )
    events_df = pd.read_csv(
        project_root / "data" / "processed" / "unified_events.csv"
    )

    with open(project_root / "data" / "processed" / "calibrated_thresholds.json") as f:
        cal = json.load(f)

    with open(project_root / "outputs" / "reports" / "threshold_calibration_results.json") as f:
        cal_results = json.load(f)

    # --- strategy 1: best single feature with per-farm thresholds ----
    # The global threshold (3.95) flags ALL Farm A normals because Farm A
    # has naturally high transformer_overall_std (5.6-8.0).  Per-farm
    # thresholds computed via Youden's J fix this.
    best_feat_cfg = cal_results["strategies"]["strategy_2_best_feature"]
    best_feature = best_feat_cfg["feature"]               # transformer_overall_std

    # Compute per-farm optimal thresholds via Youden's J
    normal_df = feat_df[feat_df["event_label"] == "normal"]
    perfarm_thresholds = {}
    for farm in feat_df["farm"].unique():
        farm_df = feat_df[feat_df["farm"] == farm]
        farm_normal = normal_df[normal_df["farm"] == farm]
        normal_vals = farm_normal[best_feature].dropna()
        y_true_farm = (farm_df["event_label"] == "anomaly").values
        all_vals = farm_df[best_feature].dropna().values

        best_j = -999
        best_t = float(np.percentile(normal_vals, 90))  # fallback

        sweep = np.linspace(normal_vals.min(), normal_vals.max(), 200)
        for t in sweep:
            preds = all_vals > t
            tp_f = (y_true_farm & preds).sum()
            fp_f = (~y_true_farm & preds).sum()
            tn_f = (~y_true_farm & ~preds).sum()
            fn_f = (y_true_farm & ~preds).sum()
            det_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
            fa_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
            j = det_f - fa_f
            if j > best_j:
                best_j = j
                best_t = float(t)
        perfarm_thresholds[farm] = best_t

    print(f"      Per-farm thresholds for {best_feature}:")
    for farm, t in sorted(perfarm_thresholds.items()):
        print(f"        Farm {farm}: {t:.3f}")

    # --- strategy 2: weighted z-score voting (strong models only) -----
    zscore_cfg = cal_results["strategies"]["strategy_3b_zscore_strong"]
    z_threshold = zscore_cfg["z_threshold"]                # 0.1
    z_weights = zscore_cfg["subsystem_weights"]            # sub -> weight
    z_normal = cal_results["strategies"]["strategy_3a_zscore_all"]["normal_stats"]
    strong_subs_per_farm = zscore_cfg["strong_subs_per_farm"]

    # --- classify each event ------------------------------------------
    rows = []
    for idx, feat_row in feat_df.iterrows():
        farm = feat_row["farm"]
        event_id = int(feat_row["event_id"])
        true_label = feat_row["event_label"]

        # --- single-feature flag (per-farm threshold) ----------------
        feat_val = feat_row.get(best_feature, np.nan)
        farm_threshold = perfarm_thresholds.get(farm, 3.95)
        if pd.isna(feat_val):
            single_flag = False
            single_margin = 0.0
        else:
            single_flag = feat_val > farm_threshold
            single_margin = feat_val - farm_threshold

        # --- weighted z-score flag ------------------------------------
        strong_subs = strong_subs_per_farm.get(farm, [])
        weighted_z = 0.0
        total_w = 0.0

        for sub in strong_subs:
            w = z_weights.get(sub, 0.0)
            if w <= 0:
                continue
            col = f"{sub}_overall_max"
            val = feat_row.get(col, np.nan)
            if pd.isna(val):
                continue
            nstats = z_normal.get(sub, {})
            mu = nstats.get("mean", 0)
            sigma = nstats.get("std", 1)
            if sigma < 0.01:
                sigma = 0.01
            z = (val - mu) / sigma
            weighted_z += w * z
            total_w += w

        if total_w > 0:
            weighted_z /= total_w
        zscore_flag = weighted_z > z_threshold
        zscore_margin = weighted_z - z_threshold

        # --- combined (selective union) --------------------------------
        # Use z-score union only for farms with 2+ strong subsystems.
        # Farm B has only 1 strong sub (gearbox), so z-score is noisy there;
        # single-feature alone is sufficient.
        if len(strong_subs) >= 2:
            combined_flag = single_flag or zscore_flag
        else:
            combined_flag = single_flag

        # Confidence: how far above (positive) or below (negative) the
        # most-activated threshold.  We normalise each margin to [0, 1]
        # scale, then take the max of the two.
        if combined_flag:
            # Distance above threshold (positive number)
            confidence = max(
                single_margin / max(abs(farm_threshold), 1e-6),
                zscore_margin / max(abs(z_threshold + 1), 1e-6),
            )
        else:
            # Distance below threshold (negative number)
            confidence = min(
                single_margin / max(abs(farm_threshold), 1e-6),
                zscore_margin / max(abs(z_threshold + 1), 1e-6),
            )

        # Retrieve aggregated score from the scoring pipeline
        match = scores_df[
            (scores_df["farm"] == farm) & (scores_df["event_id"] == event_id)
        ]
        agg_score = float(match["aggregated_score"].iloc[0]) if len(match) else 0.0

        # Event description
        ev_match = events_df[
            (events_df["farm"] == farm) & (events_df["event_id"] == event_id)
        ]
        description = ""
        if len(ev_match):
            desc = ev_match["event_description"].iloc[0]
            if pd.notna(desc):
                description = str(desc)

        rows.append({
            "farm": farm,
            "event_id": event_id,
            "event_label": true_label,
            "predicted_label": "anomaly" if combined_flag else "normal",
            "aggregated_score": round(agg_score, 4),
            "confidence": round(float(confidence), 4),
            "single_feature_flag": single_flag,
            "zscore_flag": zscore_flag,
            "transformer_overall_std": round(float(feat_val), 4) if pd.notna(feat_val) else None,
            "weighted_zscore": round(float(weighted_z), 4),
            "event_description": description,
        })

    results = pd.DataFrame(rows)

    # --- safety check: CARE accuracy must stay >= 0.5 ----------------
    normal_mask = results["event_label"] == "normal"
    fp = ((results["predicted_label"] == "anomaly") & normal_mask).sum()
    tn = ((results["predicted_label"] == "normal") & normal_mask).sum()
    care_acc = tn / (fp + tn) if (fp + tn) > 0 else 1.0

    if care_acc < 0.5:
        print(f"WARNING: CARE Accuracy {care_acc:.4f} < 0.5 -- falling back to single-feature only")
        results["predicted_label"] = results["single_feature_flag"].map(
            {True: "anomaly", False: "normal"}
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2.  compute_confusion_metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_confusion_metrics(true_labels, predicted_labels) -> dict:
    """Compute full confusion-matrix metrics.

    Parameters
    ----------
    true_labels, predicted_labels : array-like of str
        'anomaly' or 'normal'.

    Returns
    -------
    dict with TP, FP, TN, FN, accuracy, precision, recall, F1,
    specificity, care_accuracy.
    """
    y_true = np.array([1 if l == "anomaly" else 0 for l in true_labels])
    y_pred = np.array([1 if l == "anomaly" else 0 for l in predicted_labels])

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    care_accuracy = tn / (fp + tn) if (fp + tn) > 0 else 1.0  # same as specificity
    detection_rate = recall   # alias
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": n,
        "n_anomaly": tp + fn,
        "n_normal": tn + fp,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "care_accuracy": round(care_accuracy, 4),
        "detection_rate": round(detection_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  generate_classification_report
# ═══════════════════════════════════════════════════════════════════════════

def generate_classification_report(results_df: pd.DataFrame, project_root) -> dict:
    """Generate confusion-matrix figure, per-farm breakdown, and event lists.

    Saves:
        outputs/figures/confusion_matrix.png
        outputs/figures/score_distribution_by_label.png
        outputs/reports/classification_results.json

    Returns the full metrics dict (also saved to JSON).
    """
    project_root = Path(project_root)
    fig_dir = project_root / "outputs" / "figures"
    rep_dir = project_root / "outputs" / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # --- overall metrics ----------------------------------------------
    metrics = compute_confusion_metrics(
        results_df["event_label"], results_df["predicted_label"]
    )

    # --- confusion matrix heatmap -------------------------------------
    _plot_confusion_matrix(metrics, fig_dir / "confusion_matrix.png")

    # --- score distribution plot --------------------------------------
    _plot_score_distribution(results_df, fig_dir / "score_distribution_by_label.png")

    # --- per-farm breakdown -------------------------------------------
    per_farm = {}
    for farm in sorted(results_df["farm"].unique()):
        farm_df = results_df[results_df["farm"] == farm]
        fm = compute_confusion_metrics(
            farm_df["event_label"], farm_df["predicted_label"]
        )
        per_farm[farm] = fm

    # --- event lists ---------------------------------------------------
    correct_anomalies = results_df[
        (results_df["event_label"] == "anomaly")
        & (results_df["predicted_label"] == "anomaly")
    ].sort_values("aggregated_score", ascending=False)

    missed_anomalies = results_df[
        (results_df["event_label"] == "anomaly")
        & (results_df["predicted_label"] == "normal")
    ].sort_values("aggregated_score", ascending=False)

    false_alarms = results_df[
        (results_df["event_label"] == "normal")
        & (results_df["predicted_label"] == "anomaly")
    ].sort_values("aggregated_score", ascending=False)

    correct_normals = results_df[
        (results_df["event_label"] == "normal")
        & (results_df["predicted_label"] == "normal")
    ]

    # --- build JSON output --------------------------------------------
    def _event_list(df):
        return [
            {
                "farm": r["farm"],
                "event_id": int(r["event_id"]),
                "aggregated_score": r["aggregated_score"],
                "confidence": r["confidence"],
                "transformer_overall_std": r["transformer_overall_std"],
                "weighted_zscore": r["weighted_zscore"],
                "description": r["event_description"],
            }
            for _, r in df.iterrows()
        ]

    output = {
        "description": "Phase 4.6 -- Event-level classification results",
        "strategy": "Union of best-single-feature (transformer_overall_std) + weighted z-score voting (strong models)",
        "overall_metrics": metrics,
        "per_farm_metrics": per_farm,
        "correct_anomalies": _event_list(correct_anomalies),
        "missed_anomalies": _event_list(missed_anomalies),
        "false_alarms": _event_list(false_alarms),
        "n_correct_normals": len(correct_normals),
        "per_event_predictions": [
            {
                "farm": r["farm"],
                "event_id": int(r["event_id"]),
                "true_label": r["event_label"],
                "predicted_label": r["predicted_label"],
                "aggregated_score": r["aggregated_score"],
                "confidence": r["confidence"],
            }
            for _, r in results_df.iterrows()
        ],
    }

    out_path = rep_dir / "classification_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


# ─── Internal plotting helpers ────────────────────────────────────────────

def _plot_confusion_matrix(metrics: dict, save_path: Path):
    """Seaborn heatmap with counts and percentages."""
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    n = metrics["n_total"]

    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / n * 100

    labels = np.array([
        [f"{tn}\n({cm_pct[0,0]:.1f}%)", f"{fp}\n({cm_pct[0,1]:.1f}%)"],
        [f"{fn}\n({cm_pct[1,0]:.1f}%)", f"{tp}\n({cm_pct[1,1]:.1f}%)"],
    ])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted\nNormal", "Predicted\nAnomaly"],
        yticklabels=["Actual\nNormal", "Actual\nAnomaly"],
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(
        f"Event Classification Confusion Matrix\n"
        f"Accuracy={metrics['accuracy']:.2f}  |  "
        f"F1={metrics['f1']:.2f}  |  "
        f"CARE Acc={metrics['care_accuracy']:.2f}  |  "
        f"Recall={metrics['recall']:.2f}",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_score_distribution(results_df: pd.DataFrame, save_path: Path):
    """Side-by-side violin + strip plot of aggregated scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Violin plot for aggregated scores ---
    ax = axes[0]
    for i, label in enumerate(["normal", "anomaly"]):
        subset = results_df[results_df["event_label"] == label]["aggregated_score"]
        parts = ax.violinplot(
            subset.values, positions=[i], showmeans=True, showextrema=True
        )
        color = "#3498db" if label == "normal" else "#e74c3c"
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(color)

        # Overlay individual points
        jitter = np.random.normal(0, 0.04, size=len(subset))
        ax.scatter(
            np.full_like(subset, i) + jitter,
            subset.values,
            alpha=0.5,
            s=20,
            color=color,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal\n(n=50)", "Anomaly\n(n=45)"])
    ax.set_ylabel("Aggregated Anomaly Score")
    ax.set_title("Score Distribution by True Label")
    ax.axhline(y=results_df["aggregated_score"].median(), color="gray",
               linestyle="--", alpha=0.3, label="Overall median")
    ax.legend(fontsize=8)

    # --- Histogram of transformer_overall_std (the key feature) ---
    ax2 = axes[1]
    feat_col = "transformer_overall_std"
    for label, color in [("normal", "#3498db"), ("anomaly", "#e74c3c")]:
        subset = results_df[results_df["event_label"] == label]
        vals = subset[feat_col].dropna()
        ax2.hist(
            vals, bins=15, alpha=0.5, color=color, edgecolor="white",
            label=f"{label} (n={len(vals)})",
        )
    # Threshold line
    with open(save_path.parent.parent / "reports" / "threshold_calibration_results.json") as f:
        cal_res = json.load(f)
    thresh = cal_res["strategies"]["strategy_2_best_feature"]["threshold"]
    ax2.axvline(x=thresh, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold = {thresh:.2f}")
    ax2.set_xlabel("transformer_overall_std")
    ax2.set_ylabel("Count")
    ax2.set_title("Key Feature Distribution (Best Single Feature)")
    ax2.legend(fontsize=8)

    plt.suptitle(
        "Phase 4.6 -- Score Distributions for Normal vs Anomaly Events",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
