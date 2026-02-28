import numpy as np
from scipy.optimize import curve_fit

from .gaussian_model import gaussian
from .lognormal_model import lognormal_function


class GaussianLognormalModel:
    """Fit a model consisting of one Gaussian + N lognormal components."""

    def __init__(self, num_logn):
        """Initialize with number of lognormal components.
        Args:
            num_logn (int): Number of Lognormal components. Must be >= 1.
        Raises:
            ValueError: If "num_logn < 1".
        """
        if num_logn < 1:
            raise ValueError("num_logn must be >= 1")
        self.num_logn = num_logn

    def __model(self, t, amp_g, mean_g, sigma_g, *log_params):
        """Evaluate the model on a time grid.

        The parameter layout is:

        - Gaussian (3 params): "amp_g, mean_g, sigma_g"
        - Lognormals (4 params each): "A_i, s_i, loc_i, scale_i" for
          "i = 1..num_logn"

        For numerical stability, "sigma_g", each "s_i", and each
        "scale_i" are coerced to positive values with a small epsilon
        ("1e-6") if zero is provided.

        Args:
            t (array-like): Time vector of shape "(n,)".
            amp_g (float): Amplitude of the Gaussian component.
            mean_g (float): Mean (center) of the Gaussian component.
            sigma_g (float): Standard deviation of the Gaussian component.
            *log_params (float): Flattened Lognormal parameters in groups of 4:
                "[A1, s1, loc1, scale1, A2, s2, loc2, scale2, ...]" with
                length "4 * num_logn".

        Returns:
            numpy.ndarray: Model values evaluated at "t" with shape "(n,)".

        Notes:
            - Gaussian and Lognormal kernels are provided by
              "gaussian" and "lognormal_function" respectively.
            - Absolute values are applied to "sigma_g", "s_i", and
              "scale_i" and a small epsilon is substituted if 0 to avoid
              numerical issues.
        """
        t = np.array(t, dtype=float)
        y = gaussian(t, amp_g, mean_g, abs(sigma_g) if sigma_g != 0 else 1e-6)
        for i in range(self.num_logn):
            A, s, loc, scale = log_params[i * 4 : (i + 1) * 4]
            y += lognormal_function(
                t,
                A,
                abs(s) if s != 0 else 1e-6,
                loc,
                abs(scale) if scale != 0 else 1e-6,
            )
        return y

    def fit(self, time, signal):
        """Fit the composite model to data.

        Uses bounded non-linear least squares to estimate:

        "[amp_g, mean_g, sigma_g, A1, s1, loc1, scale1, ..., A_N, s_N, loc_N, scale_N]"

        Initial guesses:
        - Gaussian: centered at the global maximum of "signal" with amplitude
          ~60% of the peak and "sigma" ~2-5% of the time span.
        - Lognormals: placed after the Gaussian mean, spaced across the remaining
          window with decreasing amplitudes.

        Bounds:
        - Gaussian: "amp >= 0", "mean in [tmin - 1% range, tmax + 1% range]",
          "sigma in [1e-6, range]".
        - Lognormal: "A >= 0", "s in [1e-6, 1]", "loc in [tmin - 1% range, tmax + 1% range]",
          "scale in [1e-6, +inf)".

        Args:
            time (array-like): Time vector of shape "(n,)" (monotonic recommended).
            signal (array-like): Signal values of shape "(n,)". Prefer normalized
                inputs for robustness.

        Returns:
            tuple[numpy.ndarray | None, numpy.ndarray | None]:
                "(popt, fitted)" where "popt" is the optimized parameter vector and
                "fitted" is the model evaluated on "time". Returns "(None, None)"
                if fitting fails.

        Raises:
            ValueError: If "time" and "signal" do not have the same length.

        Notes:
            The routine catches any fitting exception, prints a message, and returns
            "(None, None)" instead of raising.
        """
        if len(time) != len(signal):
            raise ValueError("time and signal must have same length.")

        tmin, tmax = float(np.min(time)), float(np.max(time))
        trange = max(tmax - tmin, 1e-6)
        peak = np.max(signal) if signal.size else 1.0

        # Gaussian initial guess - place at global maximum
        amp_g = float(peak * 0.6)
        mean_g = float(time[np.argmax(signal)])
        sigma_g = float(max(trange * 0.05, trange * 0.02))

        p0 = [amp_g, mean_g, sigma_g]
        lb = [0.0, tmin - 0.01 * trange, 1e-6]
        ub = [np.inf, tmax + 0.01 * trange, trange]

        # Lognormal initial guesses - place after gaussian peak spaced across remaining window
        for i in range(self.num_logn):
            A_guess = float(peak * 0.4 / (i + 1))
            s_guess = 0.5
            loc_guess = mean_g + (trange / (self.num_logn + 1)) * (i + 1) * 0.6
            scale_guess = trange / (self.num_logn * 4 + 1)
            p0.extend([A_guess, s_guess, loc_guess, scale_guess])
            lb.extend([0.0, 1e-6, tmin - 0.01 * trange, 1e-6])
            ub.extend([np.inf, 1, tmax + 0.01 * trange, np.inf])

        try:
            popt, _ = curve_fit(
                self.__model, time, signal, p0=p0, bounds=(lb, ub), maxfev=40000
            )
            fitted = self.__model(time, *popt)
            return popt, fitted
        except Exception as e:
            print(f"Gauss+Lognormal fit error: {e}")
            return None, None
