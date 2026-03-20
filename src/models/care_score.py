"""
CARE Score Implementation (Coverage, Accuracy, Reliability, Earliness)

Implements the exact formulas from arXiv:2404.10320 (CARE to Compare paper).
This is the primary evaluation metric for the DSMLC x Enbridge competition.

CARE = 0                                                    if no anomalies detected
CARE = Acc_bar                                              if Acc_bar < 0.5
CARE = (w1*F_bar + w2*WS_bar + w3*EF_beta + w4*Acc_bar) / sum(wi)   otherwise
       where w1=w2=w3=1, w4=2 (Accuracy has DOUBLE weight)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA = 0.5                       # F-beta parameter (F0.5 emphasizes precision)
BETA_SQ = BETA ** 2              # 0.25
CRIT_THRESHOLD = 72              # 12 hours at 10-min intervals
NORMAL_STATUS = {0, 2}           # status_type_id values for normal/idling
W1, W2, W3, W4 = 1, 1, 1, 2     # Accuracy gets double weight


def _farm_letter(farm_str: str) -> str:
    """Normalize farm identifier to lowercase letter."""
    farm_str = str(farm_str).strip().upper()
    if farm_str in ('A', 'B', 'C'):
        return farm_str.lower()
    if 'FARM A' in farm_str.upper():
        return 'a'
    if 'FARM B' in farm_str.upper():
        return 'b'
    if 'FARM C' in farm_str.upper():
        return 'c'
    return farm_str.lower()


def _load_detection(detection_dir: str, farm: str, event_id: int) -> pd.DataFrame:
    """Load a detection parquet for a specific farm/event."""
    farm_dir = f"farm_{_farm_letter(farm)}"
    path = os.path.join(detection_dir, farm_dir, f"event_{event_id}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detection parquet not found: {path}")
    return pd.read_parquet(path)


def _get_prediction_window(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the prediction window rows."""
    return df[df['train_test'] == 'prediction'].copy()


def _get_normal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with normal status (status_type_id 0 or 2)."""
    return df[df['status_type_id'].isin(NORMAL_STATUS)].copy()


def _get_any_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series: True if ANY subsystem combined_alarm is True.
    This is the point-wise detector prediction.
    """
    alarm_cols = [c for c in df.columns if c.endswith('_combined_alarm')]
    if not alarm_cols:
        raise ValueError("No combined_alarm columns found in detection data")
    return df[alarm_cols].any(axis=1)


def _fbeta_score(tp: int, fp: int, fn: int, beta: float = BETA) -> float:
    """
    Compute F-beta score.
    F_beta = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    Returns 0.0 if denominator is 0.
    """
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ===========================================================================
# Component 1: Coverage (F_bar) -- Point-wise F0.5 on anomaly events
# ===========================================================================
def compute_coverage(events_df: pd.DataFrame, detection_dir: str) -> dict:
    """
    Compute Coverage (F_bar): point-wise F0.5 on anomaly event datasets.

    For each anomaly event:
      - Take the prediction window
      - Keep only rows with normal status (0/2)
      - In anomaly events, all normal-status prediction rows are "should be detected"
      - TP = timestep is anomalous AND detector flags it  (flagged correctly)
      - FN = timestep is anomalous AND detector misses it (missed detection)
      - FP = timestep is normal AND detector flags it     (false alarm)
      NOTE: For anomaly events, every normal-status row in prediction window
            is considered "truly anomalous" (the event IS anomalous).

    Returns dict with F_bar and per-event breakdown.
    """
    anomaly_events = events_df[events_df['event_label'] == 'anomaly']
    per_event = []

    for _, row in anomaly_events.iterrows():
        farm = row['farm']
        event_id = row['event_id']

        try:
            df = _load_detection(detection_dir, farm, event_id)
        except FileNotFoundError:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'f05': 0.0, 'tp': 0, 'fp': 0, 'fn': 0,
                'n_normal_rows': 0, 'error': 'file_not_found'
            })
            continue

        pred = _get_prediction_window(df)
        normal_pred = _get_normal_rows(pred)

        if len(normal_pred) == 0:
            # No normal-status rows in prediction window
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'f05': 0.0, 'tp': 0, 'fp': 0, 'fn': 0,
                'n_normal_rows': 0, 'note': 'no_normal_rows'
            })
            continue

        detected = _get_any_alarm(normal_pred)

        # All normal-status rows in anomaly event prediction window = truly anomalous
        tp = int(detected.sum())         # flagged (correct, it IS anomalous)
        fn = int((~detected).sum())      # missed (should have been flagged)
        fp = 0                           # no FP within an anomaly event's own rows

        f05 = _fbeta_score(tp, fp, fn)

        per_event.append({
            'farm': farm, 'event_id': int(event_id),
            'f05': round(f05, 6),
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_normal_rows': len(normal_pred)
        })

    f_scores = [e['f05'] for e in per_event]
    f_bar = float(np.mean(f_scores)) if f_scores else 0.0

    return {
        'F_bar': round(f_bar, 6),
        'n_anomaly_events': len(per_event),
        'per_event': per_event
    }


# ===========================================================================
# Component 2: Accuracy (Acc_bar) -- On normal events ONLY
# ===========================================================================
def compute_accuracy(events_df: pd.DataFrame, detection_dir: str) -> dict:
    """
    Compute Accuracy (Acc_bar): point-wise accuracy on normal event datasets.

    For each normal event:
      - Take the prediction window
      - Keep only rows with normal status (0/2)
      - TN = timestep is normal AND detector says normal (correct)
      - FP = timestep is normal AND detector says anomaly (false alarm)
      - Acc = TN / (FP + TN)

    Returns dict with Acc_bar and per-event breakdown.
    """
    normal_events = events_df[events_df['event_label'] == 'normal']
    per_event = []

    for _, row in normal_events.iterrows():
        farm = row['farm']
        event_id = row['event_id']

        try:
            df = _load_detection(detection_dir, farm, event_id)
        except FileNotFoundError:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'accuracy': 1.0, 'tn': 0, 'fp': 0,
                'n_normal_rows': 0, 'error': 'file_not_found'
            })
            continue

        pred = _get_prediction_window(df)
        normal_pred = _get_normal_rows(pred)

        if len(normal_pred) == 0:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'accuracy': 1.0, 'tn': 0, 'fp': 0,
                'n_normal_rows': 0, 'note': 'no_normal_rows'
            })
            continue

        detected = _get_any_alarm(normal_pred)

        fp = int(detected.sum())         # false alarms on normal event
        tn = int((~detected).sum())      # correctly identified as normal

        total = fp + tn
        acc = tn / total if total > 0 else 1.0

        per_event.append({
            'farm': farm, 'event_id': int(event_id),
            'accuracy': round(acc, 6),
            'tn': tn, 'fp': fp,
            'n_normal_rows': len(normal_pred)
        })

    accuracies = [e['accuracy'] for e in per_event]
    acc_bar = float(np.mean(accuracies)) if accuracies else 1.0

    return {
        'Acc_bar': round(acc_bar, 6),
        'n_normal_events': len(per_event),
        'per_event': per_event
    }


# ===========================================================================
# Component 3: Reliability (EF_beta) -- Event-level via criticality counter
# ===========================================================================
def compute_reliability(events_df: pd.DataFrame, detection_dir: str) -> dict:
    """
    Compute Reliability (EF_beta): event-level F0.5 using criticality counter.

    For each event (all 95):
      - Walk through prediction window timesteps
      - Criticality counter algorithm:
        - If status is abnormal (not 0/2): skip (don't update counter)
        - If status is normal AND detector flags anomaly: crit[t] = crit[t-1] + 1
        - If status is normal AND detector says normal:   crit[t] = max(crit[t-1] - 1, 0)
      - If max(crit) >= 72: predict this event as anomaly

    Then compute F0.5 at event level comparing true labels to crit-based predictions.

    Returns dict with EF_beta and per-event breakdown.
    """
    per_event = []

    for _, row in events_df.iterrows():
        farm = row['farm']
        event_id = row['event_id']
        true_label = row['event_label']

        try:
            df = _load_detection(detection_dir, farm, event_id)
        except FileNotFoundError:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'true_label': true_label,
                'crit_predicted': 'normal',
                'max_crit': 0,
                'error': 'file_not_found'
            })
            continue

        pred = _get_prediction_window(df)

        if len(pred) == 0:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'true_label': true_label,
                'crit_predicted': 'normal',
                'max_crit': 0,
                'note': 'no_prediction_rows'
            })
            continue

        # Criticality counter walk
        status = pred['status_type_id'].values
        alarms = _get_any_alarm(pred).values
        crit = 0
        max_crit = 0

        for t in range(len(pred)):
            if status[t] not in NORMAL_STATUS:
                # Abnormal status: skip, don't update counter
                continue
            if alarms[t]:
                crit += 1
            else:
                crit = max(crit - 1, 0)
            max_crit = max(max_crit, crit)

        crit_predicted = 'anomaly' if max_crit >= CRIT_THRESHOLD else 'normal'

        per_event.append({
            'farm': farm, 'event_id': int(event_id),
            'true_label': true_label,
            'crit_predicted': crit_predicted,
            'max_crit': int(max_crit)
        })

    # Compute event-level F0.5
    tp = sum(1 for e in per_event if e['true_label'] == 'anomaly' and e['crit_predicted'] == 'anomaly')
    fp = sum(1 for e in per_event if e['true_label'] == 'normal' and e['crit_predicted'] == 'anomaly')
    fn = sum(1 for e in per_event if e['true_label'] == 'anomaly' and e['crit_predicted'] == 'normal')
    tn = sum(1 for e in per_event if e['true_label'] == 'normal' and e['crit_predicted'] == 'normal')

    ef_beta = _fbeta_score(tp, fp, fn)

    return {
        'EF_beta': round(ef_beta, 6),
        'event_level_tp': tp,
        'event_level_fp': fp,
        'event_level_fn': fn,
        'event_level_tn': tn,
        'n_events': len(per_event),
        'per_event': per_event
    }


# ===========================================================================
# Component 4: Earliness (WS_bar) -- Weighted score on anomaly events
# ===========================================================================
def compute_earliness(events_df: pd.DataFrame, detection_dir: str) -> dict:
    """
    Compute Earliness (WS_bar): weighted detection score on anomaly events.

    For each anomaly event:
      - The event window is event_start_id to event_end_id (row indices in the parquet)
      - Assign position weights:
        - First half of event window: weight = 1.0
        - Second half: weight decreases linearly from 1.0 to 0.0
      - WS = sum(weight[t] * detected[t]) / sum(weight[t])
      - detected[t] is 1 if ANY subsystem combined_alarm is True

    WS_bar = mean WS across all anomaly events.
    """
    anomaly_events = events_df[events_df['event_label'] == 'anomaly']
    per_event = []

    for _, row in anomaly_events.iterrows():
        farm = row['farm']
        event_id = row['event_id']
        event_start_id = int(row['event_start_id'])
        event_end_id = int(row['event_end_id'])

        try:
            df = _load_detection(detection_dir, farm, event_id)
        except FileNotFoundError:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'ws': 0.0, 'event_window_len': 0,
                'error': 'file_not_found'
            })
            continue

        # Event window: rows from event_start_id to event_end_id (inclusive)
        event_window = df.loc[event_start_id:event_end_id]

        if len(event_window) == 0:
            per_event.append({
                'farm': farm, 'event_id': int(event_id),
                'ws': 0.0, 'event_window_len': 0,
                'note': 'empty_event_window'
            })
            continue

        n = len(event_window)
        half = n // 2

        # Build weights array
        weights = np.ones(n)
        if n > 1:
            # Second half: linear decrease from 1.0 to 0.0
            second_half_len = n - half
            weights[half:] = np.linspace(1.0, 0.0, second_half_len + 1)[1:]  # exclude first 1.0

        # Get detection flags for the event window
        detected = _get_any_alarm(event_window).values.astype(float)

        # WS = sum(weight * detected) / sum(weight)
        weight_sum = weights.sum()
        if weight_sum == 0:
            ws = 0.0
        else:
            ws = float((weights * detected).sum() / weight_sum)

        per_event.append({
            'farm': farm, 'event_id': int(event_id),
            'ws': round(ws, 6),
            'event_window_len': n,
            'n_detected': int(detected.sum()),
            'weighted_detections': round(float((weights * detected).sum()), 4),
            'total_weight': round(float(weight_sum), 4)
        })

    ws_scores = [e['ws'] for e in per_event]
    ws_bar = float(np.mean(ws_scores)) if ws_scores else 0.0

    return {
        'WS_bar': round(ws_bar, 6),
        'n_anomaly_events': len(per_event),
        'per_event': per_event
    }


# ===========================================================================
# Final CARE Score
# ===========================================================================
def compute_care(project_root: str) -> dict:
    """
    Compute the complete CARE score.

    CARE = 0                               if no anomalies detected at all
    CARE = Acc_bar                         if Acc_bar < 0.5
    CARE = (w1*F + w2*WS + w3*EF + w4*Acc) / (w1+w2+w3+w4)    otherwise

    Returns dict with final score, all sub-scores, per-event breakdowns, and
    which special case (if any) triggered.
    """
    project_root = Path(project_root)
    detection_dir = str(project_root / 'data' / 'processed' / 'detections')
    events_path = str(project_root / 'data' / 'processed' / 'unified_events.csv')

    # Load event info
    events_df = pd.read_csv(events_path)
    print(f"Loaded {len(events_df)} events "
          f"({(events_df['event_label']=='anomaly').sum()} anomaly, "
          f"{(events_df['event_label']=='normal').sum()} normal)")

    # Compute all four components
    print("\n[1/4] Computing Coverage (F_bar)...")
    coverage = compute_coverage(events_df, detection_dir)
    print(f"  F_bar = {coverage['F_bar']:.4f}")

    print("\n[2/4] Computing Accuracy (Acc_bar)...")
    accuracy = compute_accuracy(events_df, detection_dir)
    print(f"  Acc_bar = {accuracy['Acc_bar']:.4f}")

    print("\n[3/4] Computing Reliability (EF_beta)...")
    reliability = compute_reliability(events_df, detection_dir)
    print(f"  EF_beta = {reliability['EF_beta']:.4f}")
    print(f"  Event-level: TP={reliability['event_level_tp']}, "
          f"FP={reliability['event_level_fp']}, "
          f"FN={reliability['event_level_fn']}, "
          f"TN={reliability['event_level_tn']}")

    print("\n[4/4] Computing Earliness (WS_bar)...")
    earliness = compute_earliness(events_df, detection_dir)
    print(f"  WS_bar = {earliness['WS_bar']:.4f}")

    F_bar = coverage['F_bar']
    Acc_bar = accuracy['Acc_bar']
    EF_beta = reliability['EF_beta']
    WS_bar = earliness['WS_bar']

    # Determine special cases
    # Check if ANY anomalies were detected at all
    # (using reliability's criticality-based predictions)
    any_detected = any(
        e['crit_predicted'] == 'anomaly'
        for e in reliability['per_event']
    )
    # Also check point-wise: if no alarms fired at all across all events
    any_pointwise = any(
        e.get('tp', 0) > 0
        for e in coverage['per_event']
    )

    special_case = None
    if not any_detected and not any_pointwise:
        special_case = 'no_anomalies_detected'
        care_score = 0.0
    elif Acc_bar < 0.5:
        special_case = 'accuracy_penalty'
        care_score = Acc_bar
    else:
        # Normal case: weighted average
        w_sum = W1 + W2 + W3 + W4
        care_score = (W1 * F_bar + W2 * WS_bar + W3 * EF_beta + W4 * Acc_bar) / w_sum

    care_score = round(care_score, 6)

    print(f"\n{'='*60}")
    print(f"CARE Score Calculation:")
    print(f"  F_bar (Coverage):    {F_bar:.4f}  (w={W1})")
    print(f"  WS_bar (Earliness):  {WS_bar:.4f}  (w={W2})")
    print(f"  EF_beta (Reliability): {EF_beta:.4f}  (w={W3})")
    print(f"  Acc_bar (Accuracy):  {Acc_bar:.4f}  (w={W4})")
    if special_case:
        print(f"  Special case: {special_case}")
    else:
        print(f"  Formula: ({W1}*{F_bar:.4f} + {W2}*{WS_bar:.4f} + "
              f"{W3}*{EF_beta:.4f} + {W4}*{Acc_bar:.4f}) / {W1+W2+W3+W4}")
    print(f"  CARE = {care_score:.4f}")
    print(f"{'='*60}")

    return {
        'care_score': care_score,
        'sub_scores': {
            'F_bar': F_bar,
            'Acc_bar': Acc_bar,
            'EF_beta': EF_beta,
            'WS_bar': WS_bar
        },
        'weights': {
            'w1_coverage': W1,
            'w2_earliness': W2,
            'w3_reliability': W3,
            'w4_accuracy': W4
        },
        'special_case': special_case,
        'coverage': coverage,
        'accuracy': accuracy,
        'reliability': reliability,
        'earliness': earliness
    }
