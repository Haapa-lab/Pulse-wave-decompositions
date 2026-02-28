# Decompositions package


## Overview

This repository implements pulse decomposition methods for analyzing arterial pulse waves (for example, blood volume pulse measured with PPG). It provides model implementations (Gaussian, lognormal, and a mixed Gaussian/lognormal model), utilities for loading and preprocessing data, fitting decompositions, and plotting results. The library is usable as a Python package from the `src/` modules and also exposes a small console script entrypoint for quick runs and experiments. You can use either the console script by answering the questions or use `decomposition.py` as part of your script.

Key components:

- Model implementations: `gaussian_model.py`, `lognormal_model.py`, `gaussian_lognormal_model.py`.
- Orchestration and API: `decomposition.py`, `example.py`.
- I/O and helpers: `load_data.py`, `utils.py`, `plot_results.py`.
- Packaging: `pyproject.toml` and `requirements.txt` for dependencies and build.

## Install and build

Option 1. Use `uv`

    Step 1. Install `uv` (if you don't have it):

    ```bash
    pip install uv
    ```

    Step 2. Build a wheel:

    ```bash
    uv build
    ```

    Step 3. Install the built wheel (or you can install directly from source):

    ```bash
    pip install dist/*.whl
    ```

Option 2. Install directly with pip
```bash
pip install -r requirements.txt
pip install .
```

## Run the console script

After installation the package exposes a console script `pulse-decomposition`

```bash
pulse-decomposition
```

## Python API

This project provides API under the `pulse_decomposition` package. For details, see the module docstrings in source files under `src/pulse_decomposition/`. Below are the main entry points, their parameters and return values so you can use the library programmatically.

### Primary functions and classes

- `run_decomposition(y, t, detrend_arg, model, components, plot_each)`
    - Purpose: Preprocess (optional detrend + normalize) and fit a decomposition.
    - Args:
        - `y` (array-like[float]): Signal amplitude samples.
        - `t` (array-like[float]): Corresponding time/sample axis.
        - `detrend_arg` (bool | int | None): Detrending control — `False`/`None` no detrend,
            `True` uses degree 1, or an `int` specifies polynomial degree.
        - `model` (str): One of `'gaussian'`, `'lognormal'`, or `'gaussian_lognormal'`.
        - `components` (int): Number of components to fit. For `'gaussian_lognormal'`
            this is the number of *lognormal* components (the model also fits one Gaussian).
        - `plot_each` (bool): If `True`, calls the plotting helper to show components.
    - Returns: `(params, fitted, scaling_factor, rss_val)` where
        - `params` (array-like | dict | None): Fitted parameter vector (flattened) or
            `None` on failure.
        - `fitted` (array-like | None): Reconstructed fitted signal or `None`.
        - `scaling_factor` (float | None): Multiplicative factor applied during
            normalization, or `None` if normalization skipped.
        - `rss_val` (float): Residual sum of squares (RSS) between processed data and fit.

- `load_pulse_csv(filepath)`
    - Purpose: Load pulse data stored as CSV where row 0 = amplitudes and row 1 = times.
    - Args: `filepath` (str)
    - Returns: `(time, amplitude)` as 1D numpy arrays.

- `detrend_signal(time, signal, degree=1)`
    - Purpose: Remove baseline trends (mean removal, endpoint linear, or polynomial).
    - Args:
        - `time` (array-like), `signal` (array-like), `degree` (int, default `1`).
    - Returns: detrended signal (numpy.ndarray).
    - Notes:
        - If degree is "0", detrend subtracts the mean.
        - If degree is "1", detrend uses linear detrending using endpoints.
        - If degree is ">=2", detrend performs a polynomial detrending via "numpy.polyfit". 

- `gaussian(x, amplitude, mean, sigma)`
    - Purpose: Compute a scaled Gaussian kernel used by `GaussianModel` and `GaussianLognormalModel`.
    - Args:
        - `x` (array-like): Points where the Gaussian is evaluated.
        - `amplitude` (float): Peak amplitude.
        - `mean` (float): Center position.
        - `sigma` (float): Spread parameter (non-zero recommended).
    - Returns: `numpy.ndarray` of evaluated Gaussian values.

- `lognormal_function(t, A, s, loc, scale)`
    - Purpose: Compute a scaled lognormal probability density function.
    - Args:
        - `t` (array-like): Input time vector.
        - `A` (float): Amplitude multiplier.
        - `s` (float): Shape parameter (sigma-like), positive.
        - `loc` (float): Location shift.
        - `scale` (float): Scale parameter, positive.
    - Returns: `numpy.ndarray` of evaluated scaled lognormal values.

- `GaussianModel`, `LognormalModel`, `GaussianLognormalModel`
    - Initialization
        - `GaussianModel(num_components)`, where `num_components` is the number of kernels
        - `LognormalModel(num_components)`,  where `num_components` is the number of kernels
        - `GaussianLognormalModel(num_logn)`,  where `num_logn` is the number of lognormal kernels
    - Provides `fit(time, signal)` methods that return `(popt, fitted)`
      where `popt` is a flattened parameter vector and `fitted` is the evaluated curve on `time`.
        - `popt` content:
            - `GaussianModel`: 3 parameters per component — `[amp, mean, sigma]`
            - `LognormalModel`: 4 parameters per component — `[A, s, loc, scale]`
            - `GaussianLognormalModel`: first 3 params for Gaussian (`amp_g, mean_g, sigma_g`), then 4 per lognormal component as above

- `plot_results(time, signal, fitted, params, model_choice)`
    - Purpose: Plot original, summed fit and individual components; shows RSS.
    - Args: `time`, `signal`, `fitted`, `params`, `model_choice` (see `ModelChoice` enum).
    - Returns: `None` (displays a matplotlib figure).

Notes
- All `fit(...)` methods use `scipy.optimize.curve_fit` with bounds and
    sensible initial guesses; on failure they print an error and return `(None, None)`.
- `ModelChoice` (enum) is available in `utils.py`.

### Example usage

```python
from pulse_decomposition.decomposition import run_decomposition

# y, t = ...  # load or generate arrays
params, fitted, scaling, rss = run_decomposition(y, t, detrend_arg=1, model='gaussian', components=3, plot_each=False)
```

## Data
The accepted pulse‑wave data format is a .csv file in which row 0 contains amplitude samples and row 1 contains the corresponding time samples.

Three example pulse waves are provided in `data/examples`

## Notes

- The package exposes the modules under the package namespace so functions and classes from `src/pulse_decomposition/` are available when importing `pulse_decomposition`.
- The project uses `pyproject.toml` with `uv` as the build backend.

## License

This project is licensed under the [MIT License](LICENSE).

## Authors

- Mira Haapatikka
- Antti Vehkaoja
