
import numpy as np
import pandas as pd

## -----------------------------------------------------------------------------------------------------------------------------#

def clean_forecast_data(df, threshold=1, verbose=True):
    """
    Cleans forecast data by removing rows where the normalized absolute mismatch 
    exceeds a given threshold and adds useful derived columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame. Must contain the columns 'mismatch', 'load', and 'forecast'.
    threshold : float, optional (default=1)
        The threshold for normalized absolute mismatch beyond which rows are removed.
    verbose : bool, optional (default=True)
        If True, prints the number of rows removed.

    Returns:
    --------
    pd.DataFrame
        A cleaned DataFrame with additional columns:
        - 'mismatch-abs': Absolute value of 'mismatch'
        - 'mismatch-abs-normalized': 'mismatch-abs' normalized by 'load'
        - 'position': 'shortage' if mismatch < 0, otherwise 'surplus'

    Raises:
    -------
    ValueError
        If required columns are missing from the input DataFrame.
    """
    required_columns = {'mismatch', 'load', 'forecast'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"The input DataFrame is missing required columns: {missing_columns}")

    df = df.copy()

    df['mismatch-abs'] = df['mismatch'].abs()
    df['mismatch-abs-normalized'] = df['mismatch-abs'] / df['load']

    initial_len = len(df)
    df.loc[df['mismatch-abs-normalized'] > threshold, 'forecast'] = np.nan
    df = df.dropna(subset=['forecast'])
    removed_rows = initial_len - len(df)

    df['position'] = df['mismatch'].apply(lambda x: 'shortage' if x < 0 else 'surplus')

    if verbose:
        print(f"Removed {removed_rows} rows.")

    return df

## -----------------------------------------------------------------------------------------------------------------------------#

