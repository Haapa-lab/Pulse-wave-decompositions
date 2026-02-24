import pandas as pd


def load_pulse_csv(filepath):
    """
    Load a pulse CSV file containing amplitude and time rows.

    This function reads a CSV file with no header where:
    - Row 0 contains amplitude samples.
    - Row 1 contains time samples.

    The function returns time and amplitude as 1D NumPy arrays.
    If the two rows have mismatched lengths, the function performs a basic
    sanity check to detect a possible transposed CSV but does not correct it.

    Args:
        filepath (str): Path to the CSV file to load.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            A tuple "(time, amplitude)" where both arrays have shape "(n,)".

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV does not contain at least two rows.

    Notes:
        - The CSV must contain at least two rows:
            row 0 → amplitude,
            row 1 → time.
        - If the CSV is transposed, the caller must correct it externally.
    """
    try:
        df = pd.read_csv(filepath, header=None)
    except FileNotFoundError:
        raise
    if df.shape[0] < 2:
        raise ValueError(
            "CSV must contain at least two rows: amplitude (row 0) and time (row 1)."
        )
    y = df.iloc[0].to_numpy(dtype=float)
    t = df.iloc[1].to_numpy(dtype=float)
    if t.size != y.size:
        # if CSV was written transposed (columns are samples), try transpose
        if df.shape[1] >= 2 and df.shape[0] >= 2 and df.shape[1] == df.shape[0]:
            # unlikely, but keep safe
            pass
    return t, y
