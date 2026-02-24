import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import lognorm


def lognormal_function(t, A, s, loc, scale):
    """Compute a scaled Lognormal probability density function.

    The function evaluates:

        "f(t) = A * lognorm.pdf(t, s, loc=loc, scale=scale)"

    The Lognormal PDF is defined only for "t > loc". Values outside this
    domain are set to zero.

    Args:
        t (array-like): Time or input vector of shape "(n,)".
        A (float): Amplitude multiplier applied to the PDF.
        s (float): Shape parameter "sigma" of the underlying normal distribution.
            Should be positive; caller should avoid "s = 0".
        loc (float): Location shift of the distribution.
        scale (float): Scale parameter (must be positive).

    Returns:
        numpy.ndarray: Scaled lognormal PDF evaluated at "t" with zeros where
        "t <= loc".

    Notes:
        - Invalid ranges ("t <= loc") are explicitly zeroed out.
        - Parameterization follows "scipy.stats.lognorm" conventions.
        - Caller is responsible for enforcing positivity of "scale" and "s".
    """
    t_arr = np.array(t, dtype=float)
    # pdf is defined for t>loc; set invalid to 0
    pdf = np.zeros_like(t_arr)
    mask = t_arr > loc
    if np.any(mask):
        pdf[mask] = lognorm.pdf(t_arr[mask], s, loc=loc, scale=scale)
    return A * pdf


class LognormalModel:
    """Model composed of multiple Lognormal components.

    This class provides a multi-component Lognormal mixture model and a bounded
    non-linear least-squares fitting routine using "scipy.optimize.curve_fit".
    """

    def __init__(self, num_components):
        """Initialize a Lognormal mixture model.

        Args:
            num_components (int): Number of Lognormal components. Must be >= 1.

        Raises:
            ValueError: If "num_components < 1".
        """
        if num_components < 1:
            raise ValueError("num_components must be >= 1")
        self.num_components = num_components

    def model(self, t, *params):
        """Evaluate the Lognormal mixture model at sample points 't'.

        Each Lognormal component is parameterized by a group of four values:

        - "A_i": amplitude
        - "s_i": shape parameter (absolute value applied; 1e-6 substituted if 0)
        - "loc_i": location shift
        - "scale_i": scale parameter (absolute value applied; 1e-6 if 0)

        Args:
            t (array-like): Input time vector of shape "(n,)".
            *params (float): Flattened list of model parameters in the order
                "[A1, s1, loc1, scale1, A2, s2, loc2, scale2, ...]"
                with length "4 * num_components".

        Returns:
            numpy.ndarray: Model output evaluated on "t" with shape "(n,)".

        Notes:
            - "lognormal_function" performs the per-component computation.
            - Parameters "s" and "scale" are coerced to positive values to
              ensure numerical stability.
        """
        t = np.array(t, dtype=float)
        y = np.zeros_like(t)
        for i in range(self.num_components):
            A, s, loc, scale = params[i * 4 : (i + 1) * 4]
            y += lognormal_function(
                t,
                A,
                abs(s) if s != 0 else 1e-6,
                loc,
                abs(scale) if scale != 0 else 1e-6,
            )
        return y

    def fit(self, time, signal):
        """Fit the multi-component Lognormal model to data.

        The optimizer estimates:

        "[A1, s1, loc1, scale1, A2, s2, loc2, scale2, ..., A_N, s_N, loc_N, scale_N]"

        Initialization strategy:
        - Amplitudes decrease with component index.
        - Shape parameters start at 0.5.
        - Component centers are placed increasingly across the time span.
        - Scale values depend on the total signal duration.

        Parameter bounds:
        - "A >= 0"
        - "s in [1e-6, 1]"
        - "loc in [tmin - 1% range, tmax + 1% range]"
        - "scale in [1e-6, +inf)"

        Args:
            time (array-like): Time vector of shape "(n,)".
            signal (array-like): Signal values of shape "(n,)" to be fitted.

        Returns:
            tuple[numpy.ndarray | None, numpy.ndarray | None]:
                "(popt, fitted)" where:

                - "popt" is the optimized parameter vector
                - "fitted" is the evaluated model curve
                - On fitting failure, returns "(None, None)"

        Raises:
            ValueError: If "time" and "signal" have different lengths.

        Notes:
            - Uses "curve_fit" with bounds and a maximum evaluation count of 20,000.
            - Any exception during fitting is printed and results in "(None, None)".
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
            A_guess = peak * (1.0 / (i + 1)) * 0.6
            s_guess = 0.5
            loc_guess = tmin + (trange / (self.num_components + 1)) * (i + 1) * 0.8
            scale_guess = trange / (self.num_components * 4 + 1)
            p0.extend([A_guess, s_guess, loc_guess, scale_guess])
            lb.extend([0.0, 1e-6, tmin - 0.01 * trange, 1e-6])
            ub.extend([np.inf, 1, tmax + 0.01 * trange, np.inf])

        try:
            popt, pcov = curve_fit(
                self.model, time, signal, p0=p0, bounds=(lb, ub), maxfev=20000
            )
            fitted = self.model(time, *popt)
            return popt, fitted
        except Exception as e:
            print(f"Lognormal fit error: {e}")
            return None, None
