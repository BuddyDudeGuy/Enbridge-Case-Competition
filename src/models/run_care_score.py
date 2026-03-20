"""
Run CARE Score computation and save results.

Usage:
    py src/models/run_care_score.py
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.care_score import compute_care


def main():
    print("=" * 70)
    print("CARE Score Evaluation -- DSMLC x Enbridge Competition")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Compute CARE score
    results = compute_care(str(PROJECT_ROOT))

    care = results['care_score']
    sub = results['sub_scores']

    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  CARE Score:          {care:.4f}")
    print(f"  Coverage (F_bar):    {sub['F_bar']:.4f}")
    print(f"  Accuracy (Acc_bar):  {sub['Acc_bar']:.4f}")
    print(f"  Reliability (EF):    {sub['EF_beta']:.4f}")
    print(f"  Earliness (WS_bar):  {sub['WS_bar']:.4f}")

    if results['special_case']:
        print(f"  Special case:        {results['special_case']}")

    # Baseline comparison
    baseline_care = 0.66
    print(f"\n{'='*70}")
    print("COMPARISON TO BASELINE")
    print(f"{'='*70}")
    print(f"  Baseline (autoencoder NBM): {baseline_care:.2f}")
    print(f"  Our CARE score:             {care:.4f}")
    diff = care - baseline_care
    pct = (diff / baseline_care) * 100 if baseline_care != 0 else 0
    if diff > 0:
        print(f"  Improvement:                +{diff:.4f} (+{pct:.1f}%)")
    elif diff < 0:
        print(f"  Gap:                        {diff:.4f} ({pct:.1f}%)")
    else:
        print(f"  Matches baseline exactly")

    # Reliability breakdown
    rel = results['reliability']
    print(f"\n{'='*70}")
    print("RELIABILITY DETAIL (criticality counter)")
    print(f"{'='*70}")
    print(f"  Criticality threshold: {72} (12 hours @ 10-min intervals)")
    print(f"  Event-level TP: {rel['event_level_tp']} / {rel['event_level_tp'] + rel['event_level_fn']} anomaly events detected")
    print(f"  Event-level FP: {rel['event_level_fp']} / {rel['event_level_fp'] + rel['event_level_tn']} normal events falsely flagged")
    print(f"  Event-level F0.5: {rel['EF_beta']:.4f}")

    # Per-event criticality details
    anomaly_events = [e for e in rel['per_event'] if e['true_label'] == 'anomaly']
    normal_events = [e for e in rel['per_event'] if e['true_label'] == 'normal']

    detected_anomalies = [e for e in anomaly_events if e['crit_predicted'] == 'anomaly']
    missed_anomalies = [e for e in anomaly_events if e['crit_predicted'] == 'normal']
    false_alarms = [e for e in normal_events if e['crit_predicted'] == 'anomaly']

    if missed_anomalies:
        print(f"\n  Missed anomalies ({len(missed_anomalies)}):")
        for e in sorted(missed_anomalies, key=lambda x: x['max_crit'], reverse=True):
            print(f"    Farm {e['farm']} event {e['event_id']}: max_crit = {e['max_crit']}")

    if false_alarms:
        print(f"\n  False alarms on normal events ({len(false_alarms)}):")
        for e in sorted(false_alarms, key=lambda x: x['max_crit'], reverse=True):
            print(f"    Farm {e['farm']} event {e['event_id']}: max_crit = {e['max_crit']}")

    # Coverage detail
    cov = results['coverage']
    print(f"\n{'='*70}")
    print("COVERAGE DETAIL (point-wise F0.5 per anomaly event)")
    print(f"{'='*70}")
    sorted_cov = sorted(cov['per_event'], key=lambda x: x['f05'])
    low_coverage = [e for e in sorted_cov if e['f05'] < 0.3]
    if low_coverage:
        print(f"  Low-coverage events (F0.5 < 0.3): {len(low_coverage)}")
        for e in low_coverage[:10]:
            print(f"    Farm {e['farm']} event {e['event_id']}: F0.5={e['f05']:.4f} "
                  f"(TP={e['tp']}, FN={e['fn']}, rows={e['n_normal_rows']})")

    high_coverage = [e for e in sorted_cov if e['f05'] >= 0.8]
    print(f"  High-coverage events (F0.5 >= 0.8): {len(high_coverage)}")

    # Accuracy detail
    acc = results['accuracy']
    print(f"\n{'='*70}")
    print("ACCURACY DETAIL (per normal event)")
    print(f"{'='*70}")
    sorted_acc = sorted(acc['per_event'], key=lambda x: x['accuracy'])
    low_acc = [e for e in sorted_acc if e['accuracy'] < 0.8]
    if low_acc:
        print(f"  Low-accuracy events (< 0.8): {len(low_acc)}")
        for e in low_acc[:10]:
            print(f"    Farm {e['farm']} event {e['event_id']}: Acc={e['accuracy']:.4f} "
                  f"(FP={e['fp']}, TN={e['tn']})")

    # Earliness detail
    earl = results['earliness']
    print(f"\n{'='*70}")
    print("EARLINESS DETAIL (weighted score per anomaly event)")
    print(f"{'='*70}")
    sorted_earl = sorted(earl['per_event'], key=lambda x: x['ws'])
    late_events = [e for e in sorted_earl if e['ws'] < 0.3]
    if late_events:
        print(f"  Late/missed detection events (WS < 0.3): {len(late_events)}")
        for e in late_events[:10]:
            print(f"    Farm {e['farm']} event {e['event_id']}: WS={e['ws']:.4f}")

    early_events = [e for e in sorted_earl if e['ws'] >= 0.7]
    print(f"  Early detection events (WS >= 0.7): {len(early_events)}")

    # Save results
    output_dir = PROJECT_ROOT / 'outputs' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'care_score_results.json'

    # Prepare JSON-serializable output
    output = {
        'timestamp': datetime.now().isoformat(),
        'care_score': results['care_score'],
        'sub_scores': results['sub_scores'],
        'weights': results['weights'],
        'special_case': results['special_case'],
        'baseline_comparison': {
            'baseline_care': baseline_care,
            'our_care': care,
            'difference': round(diff, 6),
            'pct_change': round(pct, 2)
        },
        'coverage_detail': {
            'F_bar': cov['F_bar'],
            'n_anomaly_events': cov['n_anomaly_events'],
            'per_event': cov['per_event']
        },
        'accuracy_detail': {
            'Acc_bar': acc['Acc_bar'],
            'n_normal_events': acc['n_normal_events'],
            'per_event': acc['per_event']
        },
        'reliability_detail': {
            'EF_beta': rel['EF_beta'],
            'event_level_tp': rel['event_level_tp'],
            'event_level_fp': rel['event_level_fp'],
            'event_level_fn': rel['event_level_fn'],
            'event_level_tn': rel['event_level_tn'],
            'crit_threshold': 72,
            'per_event': rel['per_event']
        },
        'earliness_detail': {
            'WS_bar': earl['WS_bar'],
            'n_anomaly_events': earl['n_anomaly_events'],
            'per_event': earl['per_event']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
