
import pandas as pd
import os
import re
from functools import reduce
from typing import Callable, Optional

## -----------------------------------------------------------------------------------------------------------------------------#

def set_tz(df):
    """
    Ensure the DataFrame index is timezone-aware and converted to UTC.

    If the index is naive (no timezone), it is localized to UTC.
    If the index is timezone-aware, it is converted to UTC.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the index in UTC.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('utc')
    else:
        df.index = df.index.tz_convert('utc')
    return df

## -----------------------------------------------------------------------------------------------------------------------------#

def process_forecasts_file(file_path, tAhead=39, quantiles_to_drop=None):
    """
    Load and process forecast data from a Parquet file.

    Filters the data for a specific forecast horizon (`tAhead`) and retains only 
    columns related to quantiles. Optionally drops specified quantile columns.

    Parameters:
        file_path (str): Path to the Parquet file containing forecast data.
        tAhead (int, optional): Forecast horizon to filter by. Defaults to 39.
        quantiles_to_drop (list of str, optional): List of quantile column names to drop 
                                                   (e.g., ['quantile_P10', 'quantile_P30']). 
                                                   Defaults to None (no quantiles dropped).

    Returns:
        pd.DataFrame: Processed DataFrame with quantile columns, optionally excluding some.
    """
    df = pd.read_parquet(file_path).copy()
    df = df[df.tAhead == tAhead]

    # Keep only columns that include "quantile" in their name
    df = df[[col for col in df.columns if 'quantile' in col]]

    if quantiles_to_drop:
        df = df.drop(columns=[col for col in quantiles_to_drop if col in df.columns])

    return df


## -----------------------------------------------------------------------------------------------------------------------------#

def load_testbed_forecasts(testbed_run_dir, additional_str:str = ''):
    """
    Load and process forecast files from a testbed directory.

    Parameters:
        forecasts_dir (str): Path to the main forecasts directory.
        additional_str (str): String to append to each forecast column name.

    Returns:
        pd.DataFrame: Concatenated and formatted forecast DataFrame.
    """
    run_names = [d for d in os.listdir(testbed_run_dir) if os.path.isdir(os.path.join(testbed_run_dir, d))]

    forecasts_dict = {}
    for run_name in run_names:
        file_path = os.path.join(testbed_run_dir, run_name, 'forecasts.parquet')
        run_results = process_forecasts_file(file_path)
        run_results.columns = [c.replace('quantile_', '') + '_' + run_name + additional_str for c in run_results.columns]
        forecasts_dict[run_name] = run_results

    testebed_forecasts = pd.concat(forecasts_dict, axis=1).droplevel(0, axis=1).round(3)
    return testebed_forecasts

## -----------------------------------------------------------------------------------------------------------------------------#


def collect_and_concat_forecasts(
    run_dir_base: str,
    max_iter: Optional[int] = None,
    loader_func: Optional[Callable[[str, str], pd.DataFrame]] = None,
    iter_dir_pattern: str = 'load_training_period_90d_iter_{}',
    sub_dir: str = 'CONTINUON_BEST',
    suffix_prefix: str = ''
) -> pd.DataFrame:
    """
    Load and concatenate forecast data from multiple experiment iterations.

    Parameters:
        run_dir_base (str): Base directory where experiment subdirectories are located.
        max_iter (int, optional): Number of iterations to collect (starts from 1 to max_iter - 1).
                                  If None, automatically detect the maximum iteration number.
        loader_func (Callable[[str, str], pd.DataFrame], optional): Function to load forecasts 
                                  from a given path and optional suffix.
        iter_dir_pattern (str): Format string to construct iteration folder name. 
                                Must include one placeholder '{}'.
        sub_dir (str): Subdirectory within each iteration folder where forecast files are stored.
        suffix_prefix (str): Prefix to add before iteration number in the additional string 
                             passed to the loader.

    Returns:
        pd.DataFrame: Concatenated DataFrame of forecasts from all iterations (columns aligned).
    """
    forecasts_all_dict: dict[str, pd.DataFrame] = {}

    if max_iter is None:
        # Auto-detect iteration numbers based on folder names
        pattern_prefix = iter_dir_pattern.split('{}')[0]
        iter_dirs = [d for d in os.listdir(run_dir_base) if d.startswith(pattern_prefix)]
        iter_nums = [int(re.search(r'(\d+)', d).group(1)) for d in iter_dirs if re.search(r'(\d+)', d)]
        if not iter_nums:
            raise ValueError("No iteration directories found.")
        max_iter = max(iter_nums) + 1

    for iter_num in range(1, max_iter):
        try:
            iter_dir_name = iter_dir_pattern.format(iter_num)
            iter_path = os.path.join(run_dir_base, iter_dir_name, sub_dir)
            additional_str = f'{suffix_prefix}{iter_num}'
            forecasts = loader_func(iter_path, additional_str) if loader_func else pd.DataFrame()
        except Exception as e:
            print(e)
        forecasts_all_dict[f'iter_{iter_num}'] = forecasts

    forecasts_all = pd.concat(forecasts_all_dict, axis=1).droplevel(0, axis=1)
    return forecasts_all


## -----------------------------------------------------------------------------------------------------------------------------#

def evaluate_forecasts_consistency(X):
    """
    Evaluate the consistency of forecasts by calculating the relative variability of each forecast.

    Parameters:
        X (pd.DataFrame): DataFrame containing forecast columns.

    Returns:
        pd.Series: Series containing the relative variability of each forecast.
    """
    X = X.copy()
    X['forecasts_mean'] = X.mean(axis=1)
    X['forecasts_std'] = X.std(axis=1)
    X['forecasts_relative_variability'] = X['forecasts_std'] / X['forecasts_mean']
    return X

## -----------------------------------------------------------------------------------------------------------------------------#

def merge_dataframes(dataframes_list):
    """
    Merges a list of pandas DataFrames on their indices using an inner join.

    This function takes a list of DataFrames and merges them sequentially on their indices.
    All DataFrames must have compatible indices for the merge to succeed.

    Parameters:
        dataframes_list (list of pd.DataFrame): List of pandas DataFrames to merge.
            The merge is performed in the order of the list.

    Returns:
        pd.DataFrame: A single DataFrame resulting from merging all input DataFrames on their indices.
    
    Raises:
        ValueError: If the input list is empty.
    """
    return reduce(lambda left, right: left.merge(right, left_index=True, right_index=True), dataframes_list)

## -----------------------------------------------------------------------------------------------------------------------------#

def monthly_yearly_format(df, value_col, aggfunc):
    """
    Transforms a time-series DataFrame to show the monthly average of a specified column 
    for each available year.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a datetime index.
    value_col (str): Name of the column to aggregate.

    Returns:
    pd.DataFrame: Pivoted DataFrame with months as index and years as columns,
                  containing average monthly values of the specified column.
    """
    df = df.copy()

    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be of datetime type.")

    # Ensure the specified column exists
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")

    # Extract year and month from the index
    df['year'] = df.index.year
    df['month'] = df.index.month

    # Pivot to get average value by month and year
    df = df.pivot_table(
        index='month',
        columns='year',
        values=value_col,
        aggfunc=aggfunc
    )

    return df

## -----------------------------------------------------------------------------------------------------------------------------#

def clip_dataframe_percentiles(df, lower_percentile=0.001, upper_percentile=None):
    """
    Clips the values in each column of a DataFrame between the specified percentiles.

    Parameters:
    - df: pandas.DataFrame
    - lower_percentile: float, default 0.001 (0.1st percentile)
    - upper_percentile: float, optional. If None, set to (1 - lower_percentile).

    Returns:
    - A new DataFrame with clipped values.
    """
    df = df.copy()
    
    if upper_percentile is None:
        upper_percentile = 1 - lower_percentile

    lower = df.quantile(lower_percentile)
    upper = df.quantile(upper_percentile)
    return df.clip(lower=lower, upper=upper, axis=1)

## -----------------------------------------------------------------------------------------------------------------------------#

def add_percentile_means(df, percentiles=('P50', 'P70', 'P90'), suffix_template=''):
    """
    For each percentile code in `percentiles`, find all columns whose names
    contain that code (e.g. 'P50') and add a new column using the
    `suffix_template`, which supports formatting with {P}.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input forecast dataframe.
    percentiles : tuple[str], default ('P50','P70','P90')
        Percentile tags to look for.
    suffix_template : str, default '{P}_mean_all_iter'
        Format string for the new column name. Use '{P}' as a placeholder.
    
    Returns
    -------
    pandas.DataFrame
        The dataframe with new mean columns added (same object, not a copy).
    """

    df = df.copy()
    for P in percentiles:
        cols = df.filter(like=P).columns
        if cols.size:
            new_col_name = f'{P}_mean_all_{suffix_template}'
            df[new_col_name] = df[cols].mean(axis=1)
    return df

## -----------------------------------------------------------------------------------------------------------------------------#




