"""
Phase 4.6 -- Run event-level classification and generate all outputs.

Usage:
    py src/models/run_classification.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.event_classifier import (
    classify_events,
    compute_confusion_metrics,
    generate_classification_report,
)


def main():
    print("=" * 72)
    print("  PHASE 4.6 -- EVENT-LEVEL CLASSIFICATION")
    print("=" * 72)
    print()

    # 1. Run classifier ---------------------------------------------------
    print("[1/3] Classifying events ...")
    results_df = classify_events(PROJECT_ROOT)
    print(f"      {len(results_df)} events classified")
    print()

    # 2. Compute metrics --------------------------------------------------
    print("[2/3] Computing confusion metrics ...")
    metrics = compute_confusion_metrics(
        results_df["event_label"], results_df["predicted_label"]
    )

    # 3. Generate report + figures ----------------------------------------
    print("[3/3] Generating report and figures ...")
    report = generate_classification_report(results_df, PROJECT_ROOT)
    print()

    # ─── Print results ───────────────────────────────────────────────────
    print("=" * 72)
    print("  OVERALL CONFUSION MATRIX")
    print("=" * 72)
    print()
    print(f"                     Predicted Normal    Predicted Anomaly")
    print(f"  Actual Normal           {metrics['tn']:>3}  (TN)            {metrics['fp']:>3}  (FP)")
    print(f"  Actual Anomaly          {metrics['fn']:>3}  (FN)            {metrics['tp']:>3}  (TP)")
    print()
    print(f"  Total events:       {metrics['n_total']}")
    print(f"  True anomalies:     {metrics['n_anomaly']}")
    print(f"  True normals:       {metrics['n_normal']}")
    print()
    print(f"  Accuracy (overall): {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall / Detection: {metrics['recall']:.4f}  ({metrics['tp']}/{metrics['n_anomaly']})")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  CARE Accuracy:      {metrics['care_accuracy']:.4f}  (tn / (fp + tn))")
    print(f"  False Alarm Rate:   {metrics['false_alarm_rate']:.4f}  ({metrics['fp']}/{metrics['n_normal']})")
    print()

    care_safe = metrics["care_accuracy"] >= 0.5
    print(f"  CARE safety check:  {'PASS' if care_safe else 'FAIL'} (need >= 0.5)")
    print()

    # ─── Per-farm breakdown ──────────────────────────────────────────────
    print("=" * 72)
    print("  PER-FARM BREAKDOWN")
    print("=" * 72)
    print()
    print(f"  {'Farm':<6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}  "
          f"{'Recall':>7} {'Specif':>7} {'F1':>6} {'CARE Acc':>9}")
    print(f"  {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4}  "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*9}")

    for farm in sorted(report["per_farm_metrics"]):
        fm = report["per_farm_metrics"][farm]
        print(
            f"  {farm:<6} {fm['tp']:>4} {fm['fp']:>4} {fm['tn']:>4} {fm['fn']:>4}  "
            f"{fm['recall']:>7.2%} {fm['specificity']:>7.2%} "
            f"{fm['f1']:>6.3f} {fm['care_accuracy']:>9.4f}"
        )
    print()

    # ─── Correctly detected anomalies ────────────────────────────────────
    correct = report["correct_anomalies"]
    print("=" * 72)
    print(f"  CORRECTLY DETECTED ANOMALIES ({len(correct)} / {metrics['n_anomaly']})")
    print("=" * 72)
    for e in correct:
        desc = e["description"] if e["description"] else "(no description)"
        print(f"  Farm {e['farm']} | event {e['event_id']:>3} | "
              f"score={e['aggregated_score']:.3f} | {desc}")
    print()

    # ─── Missed anomalies ────────────────────────────────────────────────
    missed = report["missed_anomalies"]
    print("=" * 72)
    print(f"  MISSED ANOMALIES ({len(missed)} / {metrics['n_anomaly']})")
    print("=" * 72)
    if missed:
        for e in missed:
            desc = e["description"] if e["description"] else "(no description)"
            print(f"  Farm {e['farm']} | event {e['event_id']:>3} | "
                  f"score={e['aggregated_score']:.3f} | z={e['weighted_zscore']:.2f} | "
                  f"std={e.get('transformer_overall_std', 'N/A')} | {desc}")
        print()
        print("  Analysis: These anomalies were missed because their thermal")
        print("  signatures fall within the normal operating range. Possible reasons:")
        print("    - Fault type is non-thermal (pitch, electrical, communication)")
        print("    - Short-duration event with minimal temperature impact")
        print("    - Farm B has fewer strong models, reducing detection power")
    else:
        print("  (none -- all anomalies detected!)")
    print()

    # ─── False alarms ────────────────────────────────────────────────────
    fa = report["false_alarms"]
    print("=" * 72)
    print(f"  FALSE ALARMS ({len(fa)} / {metrics['n_normal']} normal events)")
    print("=" * 72)
    if fa:
        for e in fa:
            print(f"  Farm {e['farm']} | event {e['event_id']:>3} | "
                  f"score={e['aggregated_score']:.3f} | z={e['weighted_zscore']:.2f} | "
                  f"std={e.get('transformer_overall_std', 'N/A')}")
        print()
        print("  Analysis: These normal events triggered detection because their")
        print("  thermal signatures resemble anomalous patterns. These may represent")
        print("  boundary cases or periods of unusual-but-healthy operation.")
    else:
        print("  (none -- no false alarms!)")
    print()

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print()
    print(f"  Detection rate:     {metrics['recall']:.1%} ({metrics['tp']} of {metrics['n_anomaly']} anomalies caught)")
    print(f"  False alarm rate:   {metrics['false_alarm_rate']:.1%} ({metrics['fp']} of {metrics['n_normal']} normals falsely flagged)")
    print(f"  CARE Accuracy:      {metrics['care_accuracy']:.4f} (>= 0.5 required)")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print()
    print(f"  Saved:")
    print(f"    outputs/reports/classification_results.json")
    print(f"    outputs/figures/confusion_matrix.png")
    print(f"    outputs/figures/score_distribution_by_label.png")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
