Decompositions package
======================

Overview
--------

This repository implements pulse decomposition methods for analyzing arterial pulse waves (for example, blood volume pulse measured with PPG). It provides model implementations (Gaussian, lognormal, and a mixed Gaussian/lognormal model), utilities for loading and preprocessing data, fitting decompositions, and plotting results. The library is usable as a Python package from the `src/` modules and also exposes a small console script entrypoint for quick runs and experiments. You can use either the console script by answering the questions or use `decomposition.py` as part of your script.

Key components:

- Model implementations: `gaussian_model.py`, `lognormal_model.py`, `gaussian_lognormal_model.py`.
- Orchestration and API: `decomposition.py`, `example.py`.
- I/O and helpers: `load_data.py`, `utils.py`, `plot_results.py`.
- Packaging: `pyproject.toml` and `requirements.txt` for dependencies and build.

## Install and build
-----------------
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
----------------------

After installation the package exposes a console script `pulse-decomposition`

```bash
pulse-decomposition
```

Python API
----------

Import the package to access the functions and classes from the `src/pulse_decomposition/` modules:

```python
import pulse_decomposition

pulse_decomposition.run_decomposition(...)
```

## Data
The accepted pulse‑wave data format is a .csv file in which row 0 contains amplitude samples and row 1 contains the corresponding time samples.

Three example pulse waves are provided in `data/examples`

## Notes
-----

- The package exposes the modules under the package namespace so functions and classes from `src/pulse_decomposition/` are available when importing `pulse_decomposition`.
- The project uses `pyproject.toml` with `uv` as the build backend.

## License
This project is licensed under the [MIT License](LICENSE).

## Authors
- Mira Haapatikka
- Antti Vehkaoja
