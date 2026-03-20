"""
Phase 4.5 — Run aggregated anomaly scoring and save results.

Usage:
    py src/models/run_scoring.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aggregate_scores import score_all_events


def find_optimal_threshold(scores_normal, scores_anomaly):
    """Find threshold that maximizes separation (Youden's J = TPR - FPR).

    Returns
    -------
    dict with threshold, detection_rate (TPR), false_alarm_rate (FPR), j_statistic
    """
    all_scores = np.concatenate([scores_normal, scores_anomaly])
    candidates = np.linspace(all_scores.min(), all_scores.max(), 500)

    best = {"threshold": 0, "detection_rate": 0, "false_alarm_rate": 1, "j_statistic": -1}

    for t in candidates:
        tpr = np.mean(scores_anomaly >= t)  # detection rate
        fpr = np.mean(scores_normal >= t)   # false alarm rate
        j = tpr - fpr
        if j > best["j_statistic"]:
            best = {
                "threshold": float(t),
                "detection_rate": float(tpr),
                "false_alarm_rate": float(fpr),
                "j_statistic": float(j),
            }

    return best


def main():
    print("=" * 65)
    print("Phase 4.5 — Aggregated Anomaly Scoring (Weighted Voting)")
    print("=" * 65)
    print()

    # Score all events
    scored = score_all_events(PROJECT_ROOT)

    # --- Save outputs ---
    out_parquet = PROJECT_ROOT / "data" / "processed" / "event_scores.parquet"
    out_csv = PROJECT_ROOT / "outputs" / "reports" / "event_scores.csv"

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scored.to_parquet(out_parquet, index=False)
    scored.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_csv}")

    # --- Summary statistics ---
    normal = scored[scored["event_label"] == "normal"]
    anomaly = scored[scored["event_label"] == "anomaly"]

    print("\n" + "=" * 65)
    print("SCORING SUMMARY")
    print("=" * 65)

    print(f"\nTotal events:   {len(scored)}")
    print(f"  Normal:       {len(normal)}")
    print(f"  Anomaly:      {len(anomaly)}")

    print(f"\nAggregated Score Statistics:")
    print(f"  Normal  — mean: {normal['aggregated_score'].mean():.4f}, "
          f"std: {normal['aggregated_score'].std():.4f}, "
          f"median: {normal['aggregated_score'].median():.4f}, "
          f"range: [{normal['aggregated_score'].min():.4f}, {normal['aggregated_score'].max():.4f}]")
    print(f"  Anomaly — mean: {anomaly['aggregated_score'].mean():.4f}, "
          f"std: {anomaly['aggregated_score'].std():.4f}, "
          f"median: {anomaly['aggregated_score'].median():.4f}, "
          f"range: [{anomaly['aggregated_score'].min():.4f}, {anomaly['aggregated_score'].max():.4f}]")

    separation = anomaly["aggregated_score"].mean() - normal["aggregated_score"].mean()
    pooled_std = np.sqrt(
        (normal["aggregated_score"].std() ** 2 + anomaly["aggregated_score"].std() ** 2) / 2
    )
    effect_size = separation / pooled_std if pooled_std > 0 else 0
    print(f"\n  Mean separation:    {separation:.4f}")
    print(f"  Effect size (d):    {effect_size:.4f}")

    # --- Per-subsystem breakdown ---
    subsystem_cols = [c for c in scored.columns if c.endswith("_score") and c != "aggregated_score"]
    print(f"\nPer-subsystem average scores (normal vs anomaly):")
    print(f"  {'Subsystem':<25} {'Normal':>8} {'Anomaly':>8} {'Diff':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for col in sorted(subsystem_cols):
        sub_name = col.replace("_score", "")
        n_mean = normal[col].mean()
        a_mean = anomaly[col].mean()
        # Handle NaN
        n_str = f"{n_mean:.4f}" if not np.isnan(n_mean) else "   N/A"
        a_str = f"{a_mean:.4f}" if not np.isnan(a_mean) else "   N/A"
        d_str = f"{a_mean - n_mean:.4f}" if not (np.isnan(n_mean) or np.isnan(a_mean)) else "   N/A"
        print(f"  {sub_name:<25} {n_str:>8} {a_str:>8} {d_str:>8}")

    # --- Per-farm breakdown ---
    print(f"\nPer-farm average aggregated scores:")
    print(f"  {'Farm':<8} {'Normal':>8} {'Anomaly':>8} {'Diff':>8} {'N(norm)':>8} {'N(anom)':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for farm in sorted(scored["farm"].unique()):
        fn = scored[(scored["farm"] == farm) & (scored["event_label"] == "normal")]
        fa = scored[(scored["farm"] == farm) & (scored["event_label"] == "anomaly")]
        fn_mean = fn["aggregated_score"].mean() if len(fn) > 0 else float("nan")
        fa_mean = fa["aggregated_score"].mean() if len(fa) > 0 else float("nan")
        diff = fa_mean - fn_mean if not (np.isnan(fn_mean) or np.isnan(fa_mean)) else float("nan")
        print(f"  {farm:<8} {fn_mean:>8.4f} {fa_mean:>8.4f} {diff:>8.4f} {len(fn):>8} {len(fa):>8}")

    # --- Optimal threshold ---
    scores_normal = normal["aggregated_score"].values
    scores_anomaly = anomaly["aggregated_score"].values

    opt = find_optimal_threshold(scores_normal, scores_anomaly)
    print(f"\nOptimal threshold (max Youden's J):")
    print(f"  Threshold:        {opt['threshold']:.4f}")
    print(f"  Detection rate:   {opt['detection_rate']:.4f} ({int(opt['detection_rate'] * len(anomaly))}/{len(anomaly)} anomalies caught)")
    print(f"  False alarm rate: {opt['false_alarm_rate']:.4f} ({int(opt['false_alarm_rate'] * len(normal))}/{len(normal)} normals misclassified)")
    print(f"  Youden's J:       {opt['j_statistic']:.4f}")

    accuracy = (
        opt["detection_rate"] * len(anomaly) +
        (1 - opt["false_alarm_rate"]) * len(normal)
    ) / len(scored)
    print(f"  Accuracy:         {accuracy:.4f}")

    # --- Score distribution ---
    print(f"\nScore distribution (deciles):")
    for label, group in [("Normal", normal), ("Anomaly", anomaly)]:
        deciles = np.percentile(group["aggregated_score"], [10, 25, 50, 75, 90])
        print(f"  {label:>7}: P10={deciles[0]:.3f}  P25={deciles[1]:.3f}  "
              f"P50={deciles[2]:.3f}  P75={deciles[3]:.3f}  P90={deciles[4]:.3f}")

    # --- Save report ---
    report = {
        "phase": "4.5",
        "description": "Aggregated anomaly scoring via weighted voting",
        "n_events": len(scored),
        "n_normal": len(normal),
        "n_anomaly": len(anomaly),
        "normal_score_stats": {
            "mean": float(normal["aggregated_score"].mean()),
            "std": float(normal["aggregated_score"].std()),
            "median": float(normal["aggregated_score"].median()),
            "min": float(normal["aggregated_score"].min()),
            "max": float(normal["aggregated_score"].max()),
        },
        "anomaly_score_stats": {
            "mean": float(anomaly["aggregated_score"].mean()),
            "std": float(anomaly["aggregated_score"].std()),
            "median": float(anomaly["aggregated_score"].median()),
            "min": float(anomaly["aggregated_score"].min()),
            "max": float(anomaly["aggregated_score"].max()),
        },
        "mean_separation": float(separation),
        "effect_size_cohens_d": float(effect_size),
        "optimal_threshold": opt,
        "accuracy_at_optimal": float(accuracy),
        "per_farm_scores": {},
        "weighting": {
            "subsystem_weights": {
                "gearbox": 0.25,
                "generator_bearings": 0.20,
                "transformer": 0.20,
                "hydraulic": 0.15,
                "cooling": 0.10,
            },
            "method": "combined_weight = subsystem_tdi_weight * max(R2, 0.1)",
            "key_features": ["overall_max", "overall_mean", "trend_slope"],
        },
    }

    for farm in sorted(scored["farm"].unique()):
        fn = scored[(scored["farm"] == farm) & (scored["event_label"] == "normal")]
        fa = scored[(scored["farm"] == farm) & (scored["event_label"] == "anomaly")]
        report["per_farm_scores"][farm] = {
            "n_normal": len(fn),
            "n_anomaly": len(fa),
            "normal_mean": float(fn["aggregated_score"].mean()) if len(fn) > 0 else None,
            "anomaly_mean": float(fa["aggregated_score"].mean()) if len(fa) > 0 else None,
        }

    report_path = PROJECT_ROOT / "outputs" / "reports" / "aggregated_scoring_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
