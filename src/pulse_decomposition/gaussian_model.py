import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, amplitude, mean, sigma):
    """Compute a Gaussian function.

    The Gaussian is defined as:

        "f(x) = amplitude * exp(-((x - mean) / sigma)**2)"

    No normalization factor is applied; this is a *scaled* Gaussian suitable
    for curve-fitting applications.

    Args:
        x (array-like): Input values where the Gaussian is evaluated.
        amplitude (float): Peak amplitude of the Gaussian.
        mean (float): Center position of the Gaussian.
        sigma (float): Spread (standard deviation-like term). Should be nonzero.
            A zero value will typically cause numerical problems and should be
            avoided by the caller.

    Returns:
        numpy.ndarray: Gaussian values evaluated at "x".

    Notes:
        - This definition uses "exp(-((x - mean)/sigma)**2)" instead of the
          standard "exp(-(x - mean)**2 / (2 * sigma**2))"; the effect is a
          slightly different scaling of "sigma".
        - Caller is responsible for ensuring "sigma" is positive.
    """
    return amplitude * np.exp(-(((x - mean) / sigma) ** 2))


class GaussianModel:
    """Gaussian mixture model composed of 'num_components' Gaussian functions.

    This class provides both a model evaluator and a bounded non-linear
    least-squares fitting routine using "scipy.optimize.curve_fit".
    """

    def __init__(self, num_components):
        """Initialize a Gaussian mixture model.

        Args:
            num_components (int): Number of Gaussian components. Must be >= 1.

        Raises:
            ValueError: If "num_components < 1".
        """
        if num_components < 1:
            raise ValueError("num_components must be >= 1")
        self.num_components = num_components

    def __model(self, x, *params):
        """Evaluate the Gaussian mixture model at points 'x'.

        Each Gaussian component uses a 3-parameter tuple:
        "[amp_i, mean_i, sigma_i]".
        Negative or zero sigma values are coerced to a small positive epsilon
        ("1e-6") to avoid numerical instability.

        Args:
            x (array-like): Input values of shape "(n,)".
            *params (float): Flattened Gaussian parameters. Must have length
                "3 * num_components" in the order:
                "[amp1, mean1, sigma1, amp2, mean2, sigma2, ...]".

        Returns:
            numpy.ndarray: Model values evaluated at "x" with shape "(n,)".

        Notes:
            - 'gaussian' is imported from ".signal_utils".
            - A zero "sigma" is replaced with "1e-6".
        """
        x = np.array(x, dtype=float)
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, mean, sigma = params[i : i + 3]
            y += gaussian(x, amp, mean, abs(sigma) if sigma != 0 else 1e-6)
        return y

    def fit(self, time, signal):
        """Fit the Gaussian mixture model to data.

        Uses bounded non-linear least squares to estimate:

        "[amp1, mean1, sigma1, amp2, mean2, sigma2, ..., ampN, meanN, sigmaN]"

        Initial guesses:
        - Component amplitudes are initialized to an equal fraction of the
          maximum signal.
        - Means are spaced evenly across the data range.
        - Sigma values are based on a fraction of the total time span.

        Parameter bounds:
        - "amp >= 0"
        - "mean" within slightly expanded data limits
        - "sigma" in "[1e-6, time_range]"

        Args:
            time (array-like): Time values of shape "(n,)".
            signal (array-like): Signal values of shape "(n,)".

        Returns:
            tuple[numpy.ndarray | None, numpy.ndarray | None]:
                "(popt, fitted)", where:

                - "popt" is the optimized parameter vector, or "None" on failure.
                - "fitted" is the model evaluated at "time" using "popt",
                  or "None" on failure.

        Raises:
            ValueError: If "time" and "signal" have different lengths.

        Notes:
            Any runtime fitting exception is caught, printed, and results in
            "(None, None)" being returned.
        """
        if len(time) != len(signal):
            raise ValueError("time and signal must have same length.")

        tmin, tmax = float(np.min(time)), float(np.max(time))
        trange = max(tmax - tmin, 1e-6)
        peak = np.max(signal) if signal.size else 1.0

        p0 = []
        lb = []
        ub = []
        for i in range(self.num_components):
            amp_guess = peak * (1.0 / self.num_components)
            mean_guess = tmin + (trange / (self.num_components + 1)) * (i + 1)
            sigma_guess = trange / (self.num_components * 4 + 1)
            p0.extend([amp_guess, mean_guess, sigma_guess])
            lb.extend([0.0, tmin - 0.01 * trange, 1e-6])
            ub.extend([np.inf, tmax + 0.01 * trange, trange])

        try:
            popt, _ = curve_fit(
                self.__model, time, signal, p0=p0, bounds=(lb, ub), maxfev=20000
            )
            fitted = self.__model(time, *popt)
            return popt, fitted
        except Exception as e:
            print(f"Gaussian fit error: {e}")
            return None, None
