import numpy as np
import matplotlib.pyplot as plt

from .gaussian_model import gaussian
from .lognormal_model import lognormal_function
from .utils import ModelChoice


def plot_results(time, signal, fitted, params, model_choice):
    """Plot the original signal, fitted sum, and component curves.

    Computes the Residual Sum of Squares (RSS) between 'signal' and 'fitted'
    and displays it in the legend. Component curves are labeled per model:
    - Gaussian model: G1, G2, ...
    - Lognormal model: L1, L2, ...
    - Mixed model: first Gaussian (G1), followed by Lognormals (L1, L2, ...).

    Args:
        time (array-like): Time values (seconds), shape "(n,)".
        signal (array-like): Original (preprocessed) signal values, shape "(n,)".
        fitted (array-like): Summed fitted model evaluated on 'time', shape "(n,)".
        params (array-like | None): Flat parameter vector returned by fitting.
            Interpretation depends on 'model_choice':
                - GAUSSIAN: 3 params per component (e.g., amplitude, mean, sigma)
                - LOGNORMAL: 4 params per component (e.g., A, s, loc, scale)
                - GAUSSIAN_PLUS_LOGNORMAL: 3 Gaussian params first, then 4 per Lognormal
            If None, only original and fitted curves are plotted.
        model_choice (ModelChoice): Which model family was fitted.

    Returns:
        None: This function only produces a plot.

    Notes:
        - If RSS cannot be computed, it is shown as NaN.
        - The function is visualization-only and does not modify inputs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, "b-", label="Original", alpha=0.7)
    # compute RSS and include in legend
    try:
        rss = float(np.sum((np.array(signal) - np.array(fitted)) ** 2))
    except Exception:
        rss = np.nan
    plt.plot(time, fitted, "r--", label=f"Fitted sum (RSS={rss:.4e})", linewidth=1.3)
    if params is not None:
        if model_choice == ModelChoice.GAUSSIAN:
            for i in range(0, len(params), 3):
                comp = gaussian(time, *params[i : i + 3])
                plt.plot(time, comp, ":", label=f"G{i // 3 + 1}")
        elif model_choice == ModelChoice.GAUSSIAN_PLUS_LOGNORMAL:
            # first component is gaussian
            if len(params) >= 3:
                gcomp = gaussian(time, *params[0:3])
                plt.plot(time, gcomp, ":", label="G1")
            # remaining are lognormals
            for i in range(3, len(params), 4):
                comp = lognormal_function(time, *params[i : i + 4])
                plt.plot(time, comp, ":", label=f"L{(i - 3) // 4 + 1}")
        else:
            for i in range(0, len(params), 4):
                comp = lognormal_function(time, *params[i : i + 4])
                plt.plot(time, comp, ":", label=f"L{i // 4 + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # Select human-friendly title depending on model_choice
    if model_choice == ModelChoice.GAUSSIAN:
        title_str = "Gaussian"
    elif model_choice == ModelChoice.LOGNORMAL:
        title_str = "Lognormal"
    elif model_choice == ModelChoice.GAUSSIAN_PLUS_LOGNORMAL:
        title_str = "Combination of Gaussian and Lognormals"
    else:
        title_str = "Pulse decomposition"
    plt.title(f"Pulse decomposition ({title_str})")
    plt.legend()
    plt.grid(True)
    plt.show()
