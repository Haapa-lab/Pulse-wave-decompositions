import numpy as np

from .gaussian_lognormal_model import GaussianLognormalModel
from .gaussian_model import GaussianModel
from .lognormal_model import LognormalModel
from .plot_results import plot_results
from .utils import detrend_signal, ModelChoice


def run_decomposition(y, t, detrend_arg, model, components, plot_each):
    """Run signal decomposition using Gaussian, Lognormal, or hybrid
    (Gaussian+Lognormal) models with optional polynomial detrending
    and automatic normalization.

    This function preprocesses an input signal—optionally applying
    polynomial detrending and normalization to the [0, 1] range—then
    fits a selected decomposition model. It returns fitted parameters,
    the reconstructed signal, and diagnostic values such as the
    normalization scaling factor and RSS.

    Args:
        y (array-like of float): Input signal values.
        t (array-like of float): Corresponding x-axis values (e.g., time or sample indices).
        detrend_arg (bool or int or None): Controls detrending:
            - None or False → no detrending
            - True → polynomial detrend with degree 1
            - int → polynomial detrend with the given degree
            Any invalid value defaults to degree 1.
        model (str): Decomposition model type (case-insensitive). 
            Supported values:
                - "gaussian"
                - "lognormal"
                - "gaussian_lognormal" (or any unrecognized value defaults to hybrid)
        components (int): Number of component functions to fit.
        plot_each (bool): If True, plot the fitted components and total reconstructed signal.

    Returns:
        params (dict or None): Fitted model parameters, or None if fitting failed.
        fitted (array-like or None): Reconstructed fitted signal, or None if fitting failed.
        scaling_factor (float or None): The multiplicative factor used to normalize the signal.
            None if normalization was skipped (e.g., constant signal).
        rss_val (float): Residual sum of squares (RSS) between normalized data and
            fitted signal. Returns np.nan if RSS cannot be computed.

    Notes:
        - Detrending is polynomial-based and falls back to the original signal
            if any error occurs.
        - Normalization is skipped if the signal is constant after detrending.
        - Model selection is mapped via "ModelChoice(model.lower())".
        - All internal exceptions are caught; the function attempts to return
            gracefully without raising errors.
    """

    try:
        y = np.array(y, dtype=float)
        # --- Optional detrending ---
        if detrend_arg is None or detrend_arg is False:
            do_detrend = False
            detrend_degree = None
        else:
            do_detrend = True
            if isinstance(detrend_arg, bool):
                detrend_degree = 1
            else:
                try:
                    detrend_degree = int(detrend_arg)
                except Exception:
                    detrend_degree = 1

        if do_detrend:
            y_orig = y.copy()
            try:
                y = detrend_signal(t, y_orig, degree=detrend_degree)
                print(f"Signal detrended (degree={detrend_degree}).")
            except Exception as e:
                print(f"Detrending failed: {e}; using original signal.")
                y = y_orig

        y_min = np.min(y)
        y_max = np.max(y)
        scaling_factor = None
        if np.isclose(y_min, y_max):
            print("Warning: signal is constant after preprocessing; cannot normalize.")
        else:
            if not (np.isclose(y_min, 0.0) and np.isclose(y_max, 1.0)):
                scale = y_max - y_min
                scaling_factor = 1.0 / scale
                y = (y - y_min) / scale
                print(
                    f"Signal normalized to range [0, 1] (min={y_min:.6g}, max={y_max:.6g}). Scaling factor applied: {scaling_factor:.6g}"
                )
    except Exception as e:
        print(f"Preprocessing skipped due to error: {e}")
        return None, None, t, None, np.nan

    # choose model
    model_choice = ModelChoice(model.lower())
    nc = components

    if model_choice == ModelChoice.GAUSSIAN:
        gaussian_model = GaussianModel(nc)
        params, fitted = gaussian_model.fit(t, y)
    elif model_choice == ModelChoice.LOGNORMAL:
        lognormal_model = LognormalModel(nc)
        params, fitted = lognormal_model.fit(t, y)
    else:
        gauss_logn_model = GaussianLognormalModel(nc)
        params, fitted = gauss_logn_model.fit(t, y)

    rss_val = np.nan
    if fitted is None:
        print("Fitting failed.")
    else:
        print("Fitted parameters:")
        print(params)
        try:
            rss_val = float(np.sum((np.array(y) - np.array(fitted)) ** 2))
            print(f"Residual Sum of Squares (RSS): {rss_val:.6e}")
        except Exception:
            print("Could not compute RSS.")

        # optionally plot
        if plot_each:
            plot_results(t, y, fitted, params, model_choice)

    return params, fitted, scaling_factor, rss_val
