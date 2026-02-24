"""
Pulse decomposition CLI for fitting Gaussian, Lognormal, or mixed (1 Gaussian + N Lognormals)
models to a pulse-like signal loaded from a CSV file.

This script:
- Loads a (time, amplitude) pulse from a CSV file (no header) via 'load_pulse_csv'.
- Optionally detrends the signal with an endpoint-based linear baseline removal.
- Normalizes the amplitude to [0, 1] if not already normalized.
- Fits one of the following models:
    - GaussianModel ('gaussian'): sum of Gaussians
    - LognormalModel ('lognormal'): sum of Lognormals
    - GaussianLognormalModel ('gaussian_lognormal'): 1 Gaussian + N Lognormals
- Prints fitted parameters and RSS, and plots the original signal,
  the summed fit, and individual component curves.

Command-line Usage:
    Interactive (prompts):
        $ python pulse_decomposition.py

    Non-interactive:
        $ python pulse_decomposition.py --path data/pulse.csv --detrend y --model gaussian_lognormal --components 3

Notes:
    - The CSV format must be compatible with 'load_pulse_csv', which returns
      'time' and 'amplitude' as 1D arrays (row 0 = amplitude, row 1 = time).
    - Detrending uses endpoint-based linear fit when '--detrend y' is passed.
    - RSS (Residual Sum of Squares) is printed and shown in the plot legend.
"""

import json
import os
from pathlib import Path

import click
import pandas as pd

from pulse_decomposition.decomposition import run_decomposition
from pulse_decomposition.load_data import load_pulse_csv


@click.command()
@click.option(
    "--path",
    prompt="CSV path or directory",
    help="Path to CSV file or directory containing .csv files",
)
@click.option(
    "--detrend",
    is_flag=True,
    prompt="Detrend signal (linear baseline removal)?",
    help="Apply detrending",
)
@click.option(
    "--detrend-degree",
    type=int,
    default=None,
    help="Degree for detrending: 0 mean removal, 1 linear endpoints, >=2 polynomial fit",
)
@click.option(
    "--model",
    type=click.Choice(["gaussian", "lognormal", "gaussian_lognormal"], case_sensitive=False),
    prompt="Choose model - Press (gaussian) for Gaussian, (lognormal) for Lognormal, or (gaussian_lognormal) for 1 Gaussian + N Lognormals",
    help="Model choice",
)
@click.option(
    "--components",
    type=int,
    prompt='Number of components to fit (positive integer). For "gaussian_lognormal" enter number of LOGNORMAL components (N>=1)',
    help="Number of components",
)
@click.option(
    "--no-plot", is_flag=True, default=False, help="Suppress plotting for each file"
)
@click.option(
    "--out",
    type=str,
    default=None,
    help="Optional output CSV file to save DataFrame with params, scaling_factor, and fitted arrays",
)
def run(path, detrend, detrend_degree, model, components, no_plot, out):
    """Run the pulse decomposition pipeline (CLI entry point).

    Loads a pulse from CSV, optionally detrends and normalizes it, fits the
    selected model, prints the fitted parameters and RSS, and plots the results.

    If a directory path is supplied, all '*.csv' files inside will be processed
    sequentially and the fitting results (params and fitted arrays) will be
    collected into a pandas DataFrame and optionally saved to disk.

    Args:
        path (str): Path to the input CSV file or directory containing CSVs.
        detrend (bool): Whether to apply linear endpoint-based detrending.
        model (str): Model family to fit:
            - 'gaussian' → GaussianModel (sum of Gaussians)
            - 'lognormal' → LognormalModel (sum of Lognormals)
            - 'gaussian_lognormal' → GaussianLognormalModel (1 Gaussian + N Lognormals)
        components (int): Number of components. If 'model == 'gaussian_lognormal'', this is the
            number of Lognormal components (N >= 1) in addition to the single Gaussian.
        no_plot (bool): If True, skip plotting for each file.
        out (str|None): Optional path to write a CSV file containing the results DataFrame
            with params, scaling_factor, and other metadata.

    Returns:
        None: Exits the program after processing and optional saving.
    """
    if not path:
        print("CSV path is required.")
        raise SystemExit(1)

    # If path is a directory, gather all CSVs, otherwise just the single file
    if os.path.isdir(path):
        p = Path(path)
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in directory: {path}")
            raise SystemExit(1)
        files = csv_files
        dir_mode = True
    else:
        files = [Path(path)]
        dir_mode = False

    # Ask whether to plot figures for each successful fit (if plotting not suppressed)
    if not no_plot:
        plot_each = click.confirm("Plot figures for each successful fit?", default=True)
    else:
        plot_each = False

    # Prompt for detrending degree only if detrending requested and degree not provided
    if detrend:
        if detrend_degree is None:
            detrend_degree = click.prompt(
                "Detrending degree (0 mean removal, 1 linear endpoints, >=2 polynomial fit)?",
                type=int,
                default=1,
            )
    else:
        detrend_degree = None

    results = []

    for file in files:
        print(f"\nProcessing '{file}'...")
        try:
            t, y = load_pulse_csv(str(file))
            print(f"Loaded {t.size} samples.")
        except Exception as e:
            print(f"Failed to load CSV '{file}': {e}")
            continue

        # Build a single detrend argument (bool False or int degree)
        if detrend:
            detrend_arg = detrend_degree if detrend_degree is not None else 1
        else:
            detrend_arg = False

        params, fitted, scaling_factor, rss_val = run_decomposition(
            y, t, detrend_arg, model, components, plot_each
        )

        # store results (convert arrays to lists for DataFrame compatibility)
        row = {
            "filename": str(file),
            "time": t.tolist() if hasattr(t, "tolist") else t,
            "params": params.tolist() if hasattr(params, "tolist") else params,
            "fitted": fitted.tolist() if hasattr(fitted, "tolist") else fitted,
            "rss": rss_val,
            "n_samples": int(t.size) if hasattr(t, "size") else None,
            "scaling_factor": scaling_factor,
        }
        results.append(row)

    # If multiple files processed or user asked to save, create DataFrame and persist
    if results:
        df = pd.DataFrame(results)
        print("\nFitting results collected for the following files:")
        print(df[["filename", "rss", "n_samples"]].to_string(index=False))
        # determine whether and where to save results
        if out is None:
            save_results = click.confirm("Save results to CSV file?", default=True)
            if save_results:
                default_csv = click.prompt(
                    "Output CSV filename", default="fitting_results.csv"
                )
            else:
                default_csv = None
        else:
            default_csv = out

        if default_csv is not None:
            # save DataFrame to CSV with time, params, fitted arrays, and scaling_factor
            try:
                summary = df.copy()
                summary["time"] = summary["time"].apply(
                    lambda x: json.dumps(x) if x is not None else ""
                )
                summary["params"] = summary["params"].apply(
                    lambda x: json.dumps(x) if x is not None else ""
                )
                summary["fitted"] = summary["fitted"].apply(
                    lambda x: json.dumps(x) if x is not None else ""
                )
                summary.to_csv(default_csv, index=False)
                print(f"Results saved to CSV: {default_csv}")
            except Exception as e:
                print(f"Failed to save CSV '{default_csv}': {e}")
        else:
            print("Results were not saved.")
    else:
        print("No successful fits to save.")


def go():
    """Script entrypoint for pulse decomposition CLI."""
    run()


if __name__ == "__main__":
    go()
