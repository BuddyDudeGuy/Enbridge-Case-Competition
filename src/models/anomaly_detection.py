"""
CUSUM and EWMA anomaly detection on NBM residuals.

Applies statistical process control methods to residual time-series to detect
sustained shifts (degradation) vs. random noise. These methods turn noisy
residuals into clean detection signals.

Methods:
  - CUSUM (Cumulative Sum): Accumulates deviations from a reference mean.
    Random fluctuations cancel out, but sustained shifts cause the statistic
    to climb. Good for slow thermal degradation.
  - EWMA (Exponentially Weighted Moving Average): Smooths residuals with
    exponential weighting. Recent values matter more. Good for trend detection.
    Control limits use the raw residual std (not the EWMA-reduced std) to
    provide physically meaningful thresholds.

Usage:
    from src.models.anomaly_detection import compute_cusum, compute_ewma, detect_anomalies
"""

import numpy as np
import pandas as pd


def compute_cusum(
    residuals: np.ndarray,
    reference_mean: float = 0.0,
    k: float = 1.0,
    h: float = 5.0,
) -> dict:
    """Compute two-sided CUSUM on a residual time-series.

    The upper CUSUM detects positive shifts (overheating), the lower CUSUM
    detects negative shifts (sensor drift or undercooling).

    Parameters
    ----------
    residuals : array-like
        Residual values (actual - predicted). NaNs are handled by carrying
        forward the previous CUSUM value (no accumulation on NaN steps).
    reference_mean : float
        Baseline mean of residuals under normal operation. Typically computed
        from the training portion of each event.
    k : float
        Allowance (slack) parameter. Deviations smaller than k don't
        accumulate. Set to half the shift magnitude you want to detect.
    h : float
        Decision threshold. An alarm fires when CUSUM exceeds h.

    Returns
    -------
    dict
        'upper_cusum' : np.ndarray -- upper CUSUM statistic at each timestep
        'lower_cusum' : np.ndarray -- lower CUSUM statistic at each timestep
        'alarm_upper' : np.ndarray (bool) -- True where upper CUSUM > h
        'alarm_lower' : np.ndarray (bool) -- True where lower CUSUM > h
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    n = len(residuals)

    upper_cusum = np.zeros(n, dtype=np.float64)
    lower_cusum = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isnan(residuals[i]):
            # Carry forward previous value (no accumulation on NaN)
            upper_cusum[i] = upper_cusum[i - 1] if i > 0 else 0.0
            lower_cusum[i] = lower_cusum[i - 1] if i > 0 else 0.0
        else:
            prev_upper = upper_cusum[i - 1] if i > 0 else 0.0
            prev_lower = lower_cusum[i - 1] if i > 0 else 0.0

            # Upper CUSUM: detects positive shifts
            upper_cusum[i] = max(0.0, prev_upper + (residuals[i] - reference_mean) - k)

            # Lower CUSUM: detects negative shifts
            lower_cusum[i] = max(0.0, prev_lower - (residuals[i] - reference_mean) - k)

    alarm_upper = upper_cusum > h
    alarm_lower = lower_cusum > h

    return {
        "upper_cusum": upper_cusum,
        "lower_cusum": lower_cusum,
        "alarm_upper": alarm_upper,
        "alarm_lower": alarm_lower,
    }


def compute_ewma(
    residuals: np.ndarray,
    span: int = 144,
    reference_mean: float = 0.0,
    reference_std: float = 1.0,
    L: float = 3.0,
) -> dict:
    """Compute EWMA control chart on a residual time-series.

    Control limits are set at L * reference_std from the reference mean,
    using the raw residual std. This means L=3.0 corresponds to a 3-sigma
    deviation in the smoothed signal -- a physically meaningful threshold
    for temperature anomalies.

    Parameters
    ----------
    residuals : array-like
        Residual values (actual - predicted).
    span : int
        EWMA span in timesteps. 144 = 24 hours of 10-min data.
        Corresponds to lambda = 2 / (span + 1).
    reference_mean : float
        Baseline mean of residuals under normal operation.
    reference_std : float
        Baseline standard deviation of residuals under normal operation.
        Used to compute control limits. Must be > 0.
    L : float
        Control limit multiplier (number of raw std deviations).
        Default 3.0 (3-sigma limits on smoothed signal).

    Returns
    -------
    dict
        'ewma'  : np.ndarray -- EWMA statistic at each timestep
        'ucl'   : np.ndarray -- upper control limit at each timestep
        'lcl'   : np.ndarray -- lower control limit at each timestep
        'alarm' : np.ndarray (bool) -- True where EWMA exceeds control limits
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    n = len(residuals)

    # Smoothing factor from span
    lam = 2.0 / (span + 1.0)

    # Ensure reference_std is positive
    if reference_std <= 0:
        reference_std = 1.0

    # Compute EWMA
    ewma = np.zeros(n, dtype=np.float64)
    ewma[0] = reference_mean  # Initialize at reference mean

    for i in range(n):
        if i == 0:
            if np.isnan(residuals[i]):
                ewma[i] = reference_mean
            else:
                ewma[i] = lam * residuals[i] + (1.0 - lam) * reference_mean
        else:
            if np.isnan(residuals[i]):
                ewma[i] = ewma[i - 1]  # Carry forward on NaN
            else:
                ewma[i] = lam * residuals[i] + (1.0 - lam) * ewma[i - 1]

    # Control limits based on raw residual std
    # The EWMA smooths the signal, so when the EWMA deviates by L * raw_sigma,
    # it means the underlying process has shifted substantially.
    ucl = np.full(n, reference_mean + L * reference_std, dtype=np.float64)
    lcl = np.full(n, reference_mean - L * reference_std, dtype=np.float64)

    alarm = (ewma > ucl) | (ewma < lcl)

    return {
        "ewma": ewma,
        "ucl": ucl,
        "lcl": lcl,
        "alarm": alarm,
    }


def _sustained_alarm(alarm_array: np.ndarray, min_run: int = 6) -> np.ndarray:
    """Filter alarm array to keep only sustained runs of min_run consecutive alarms.

    This removes sporadic single-point false alarms. A sustained alarm means
    the process has shifted for at least min_run consecutive timesteps (default
    6 = 1 hour at 10-min intervals).

    Parameters
    ----------
    alarm_array : np.ndarray (bool)
        Raw alarm boolean array.
    min_run : int
        Minimum consecutive True values to count as a real alarm.

    Returns
    -------
    np.ndarray (bool)
        Filtered alarm array where only sustained runs are True.
    """
    alarm = np.asarray(alarm_array, dtype=bool)
    n = len(alarm)
    result = np.zeros(n, dtype=bool)

    run_start = None
    for i in range(n):
        if alarm[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_run:
                    result[run_start:i] = True
                run_start = None

    # Handle run that extends to the end
    if run_start is not None:
        run_length = n - run_start
        if run_length >= min_run:
            result[run_start:n] = True

    return result


def detect_anomalies(
    residual_df: pd.DataFrame,
    subsystems: list,
    method: str = "both",
    cusum_k_sigma: float = 0.5,
    cusum_h_sigma: float = 5.0,
    ewma_span: int = 144,
    ewma_L: float = 3.0,
    sustained_min_run: int = 6,
) -> pd.DataFrame:
    """Apply CUSUM and/or EWMA detection to each subsystem's residuals.

    Computes reference_mean and reference_std from the training portion
    of the event data (rows where train_test == "train"), then applies
    the detection methods across the full time-series.

    CUSUM k and h are specified in units of the training standard deviation,
    so they auto-scale to each subsystem's noise level:
      k_actual = cusum_k_sigma * reference_std
      h_actual = cusum_h_sigma * reference_std

    Alarms require sustained runs of min_run consecutive timesteps to filter
    out sporadic false alarms.

    Parameters
    ----------
    residual_df : pd.DataFrame
        DataFrame from a residual parquet file. Must contain 'train_test'
        column and '{subsystem}_residual' columns.
    subsystems : list of str
        Subsystem names to analyze (e.g. ['gearbox', 'generator_bearings']).
    method : str
        'cusum', 'ewma', or 'both' (default).
    cusum_k_sigma : float
        CUSUM allowance in units of training std. 0.5 = half a std deviation
        of slack before accumulation begins.
    cusum_h_sigma : float
        CUSUM decision threshold in units of training std. 5.0 = alarm fires
        after accumulating 5 std-deviations worth of shift.
    ewma_span : int
        EWMA span in timesteps (144 = 24 hours of 10-min data).
    ewma_L : float
        EWMA control limit multiplier (number of raw std deviations on the
        smoothed signal).
    sustained_min_run : int
        Minimum consecutive alarm timesteps to count as a real alarm.
        Default 6 (1 hour at 10-min intervals).

    Returns
    -------
    pd.DataFrame
        Original metadata columns (time_stamp, train_test, status_type_id)
        plus detection columns for each subsystem:
          {subsystem}_cusum_upper, {subsystem}_cusum_lower,
          {subsystem}_cusum_alarm (if method includes cusum)
          {subsystem}_ewma, {subsystem}_ewma_alarm (if method includes ewma)
          {subsystem}_combined_alarm (True if either method fires)
    """
    # Start with metadata columns
    result_cols = {}
    for col in ["time_stamp", "train_test", "status_type_id"]:
        if col in residual_df.columns:
            result_cols[col] = residual_df[col].values

    # Training mask for computing reference stats
    train_mask = residual_df["train_test"] == "train"

    for subsystem in subsystems:
        residual_col = f"{subsystem}_residual"

        if residual_col not in residual_df.columns:
            continue

        residuals = residual_df[residual_col].values.astype(np.float64)
        train_residuals = residuals[train_mask]

        # Compute reference stats from training portion (ignoring NaN)
        valid_train = train_residuals[~np.isnan(train_residuals)]
        if len(valid_train) > 0:
            reference_mean = float(np.mean(valid_train))
            reference_std = float(np.std(valid_train))
        else:
            reference_mean = 0.0
            reference_std = 1.0

        # Guard against zero std
        if reference_std < 1e-8:
            reference_std = 1.0

        # Scale CUSUM parameters by training std
        k_actual = cusum_k_sigma * reference_std
        h_actual = cusum_h_sigma * reference_std

        cusum_alarm = np.zeros(len(residuals), dtype=bool)
        ewma_alarm = np.zeros(len(residuals), dtype=bool)

        # --- CUSUM ---
        if method in ("cusum", "both"):
            cusum_result = compute_cusum(
                residuals,
                reference_mean=reference_mean,
                k=k_actual,
                h=h_actual,
            )
            result_cols[f"{subsystem}_cusum_upper"] = cusum_result["upper_cusum"]
            result_cols[f"{subsystem}_cusum_lower"] = cusum_result["lower_cusum"]
            # Apply sustained-run filter to CUSUM alarms
            raw_cusum_alarm = cusum_result["alarm_upper"] | cusum_result["alarm_lower"]
            cusum_alarm = _sustained_alarm(raw_cusum_alarm, min_run=sustained_min_run)
            result_cols[f"{subsystem}_cusum_alarm"] = cusum_alarm

        # --- EWMA ---
        if method in ("ewma", "both"):
            ewma_result = compute_ewma(
                residuals,
                span=ewma_span,
                reference_mean=reference_mean,
                reference_std=reference_std,
                L=ewma_L,
            )
            result_cols[f"{subsystem}_ewma"] = ewma_result["ewma"]
            result_cols[f"{subsystem}_ewma_ucl"] = ewma_result["ucl"]
            result_cols[f"{subsystem}_ewma_lcl"] = ewma_result["lcl"]
            # Apply sustained-run filter to EWMA alarms
            ewma_alarm = _sustained_alarm(ewma_result["alarm"], min_run=sustained_min_run)
            result_cols[f"{subsystem}_ewma_alarm"] = ewma_alarm

        # --- Combined alarm (AND logic: both methods must agree) ---
        # Individual method alarms are preserved separately for Phase 4.3
        # threshold calibration. The combined alarm is conservative: only fires
        # when both CUSUM and EWMA detect a sustained shift simultaneously.
        result_cols[f"{subsystem}_combined_alarm"] = cusum_alarm & ewma_alarm

    return pd.DataFrame(result_cols)
