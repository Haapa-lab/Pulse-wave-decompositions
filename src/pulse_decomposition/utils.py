from enum import Enum

import numpy as np


class ModelChoice(Enum):
    """Enumeration of supported model families for pulse decomposition.

    Members:
        GAUSSIAN ('gaussian'): Fit a sum of Gaussian components.
        LOGNORMAL ('lognormal'): Fit a sum of Lognormal components.
        GAUSSIAN_PLUS_LOGNORMAL ('gaussian_lognormal'): Fit a composite model with 1 Gaussian + N Lognormals.
    """

    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GAUSSIAN_PLUS_LOGNORMAL = "gaussian_lognormal"


def detrend_signal(time, signal, degree=1):
    """
    Remove baseline trends from a 1-D signal.

    Supports mean removal (degree 0), endpoint-based linear detrending (degree 1),
    or general polynomial detrending using "numpy.polyfit" for higher degrees.

    For "degree=1", the function computes a linear trend determined **only**
    by the first and last sample values (endpoint-based), i.e., it finds a line
    "k * n + b" passing through both endpoints and subtracts it from the signal.
    This method is more robust for short pulses than a full least-squares fit.

    Args:
        time (array-like): Time samples of shape "(n,)".
        signal (array-like): Signal samples of shape "(n,)".
        degree (int, optional): Degree of polynomial trend to remove.
            - "0" → subtract the mean
            - "1" → linear detrending using endpoints
            - ">=2" → polynomial detrending via "numpy.polyfit"
            Defaults to "1".

    Returns:
        numpy.ndarray: The detrended signal, same length as input.

    Raises:
        ValueError: If "time" and "signal" do not have the same length.

    Notes:
        - For "degree=1":
            * A small sample count (fewer than 2 points) falls back to mean removal.
            * Zero-based indices are internally mapped to a 1-based domain when
              constructing the endpoint line.
        - For "degree>=2":
            * Uses "numpy.polyfit(time, signal, degree)".
            * Suitable for baseline curvature correction.
        - The returned array is **not normalized**; only detrending is performed.
    """
    t = np.array(time, dtype=float)
    y = np.array(signal, dtype=float)
    if t.size != y.size:
        raise ValueError("time and signal must have same length for detrending.")
    if degree == 0:
        return y - np.mean(y)
    if degree == 1:
        # Endpoint-based linear detrending
        pulse = np.asarray(y).reshape(-1)  # ensure 1-D
        mm = pulse.size
        if mm < 2:
            # fallback to mean removal
            return y - np.mean(y)

        rng = np.arange(1, mm + 1)  # [1, 2, ..., mm]
        # Build system to solve for line through endpoints (k*s + b)
        A = np.array([[1, 1], [mm, 1]], dtype=float)
        # Right-hand side: endpoint values (1-based -> 0-based index conversion)
        y_end = np.array([pulse[1 - 1], pulse[mm - 1]], dtype=float)
        try:
            k, b = np.linalg.solve(A, y_end)
        except np.linalg.LinAlgError:
            return y - np.mean(y)
        # Compute trend at indices 1..mm
        trend = k * rng + b
        # Subtract trend from the corresponding pulse samples (map back to 0-based)
        detrended = pulse[rng - 1] - trend
        return detrended
    # degree == 2 or other: use polynomial fit on (time, y)
    coeffs = np.polyfit(t, y, degree)
    trend = np.polyval(coeffs, t)
    return y - trend
