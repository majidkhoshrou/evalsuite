
import os
import re
import logging
from functools import reduce
from datetime import tzinfo
from typing import Callable, Optional, Union, List, Literal

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

## -----------------------------------------------------------------------------------------------------------------------------#

def set_tz(df: DataFrame, target_tz: Union[str, tzinfo] = 'UTC') -> DataFrame:
    """
    Ensure the DataFrame index is timezone-aware and converted to the specified timezone (default: UTC).

    If the index is naive (no timezone), it is localized to the target timezone.
    If the index is timezone-aware, it is converted to the target timezone.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        target_tz (str or tzinfo): Timezone to convert to (default is 'UTC').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the index in the specified timezone.
    """
    df = df.copy()

    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError("DataFrame index could not be converted to datetime.") from e

    if df.empty:
        df.index = df.index.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
    elif df.index.tz is None:
        df.index = df.index.tz_localize(target_tz)
    else:
        df.index = df.index.tz_convert(target_tz)

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

# Configure basic logging (you can customize this as needed in your main script)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_and_concat_forecasts(
    run_dir_base: str,
    loader_func: Optional[Callable[[str, str], pd.DataFrame]] = None,
    iter_dir_pattern: str = 'load_training_period_90d_iter_{}',
    sub_dir: str = 'CONTINUON_BEST',
    suffix_prefix: str = ''
) -> pd.DataFrame:
    """
    Load and concatenate forecast data from multiple experiment iterations.

    Parameters:
        run_dir_base (str): Base directory containing iteration subdirectories.
        loader_func (Optional[Callable[[str, str], pd.DataFrame]]): 
            Function to load forecast data. It should take a path and a string suffix as arguments.
            If not provided, empty DataFrames will be used.
        iter_dir_pattern (str): Format string pattern for iteration directories. Must contain '{}' placeholder.
        sub_dir (str): Subdirectory inside each iteration folder where forecast data resides.
        suffix_prefix (str): Prefix to append before iteration number in the suffix passed to the loader.

    Returns:
        pd.DataFrame: Concatenated forecast DataFrame from all iterations.

    Raises:
        ValueError: If no matching directories are found or no data is successfully loaded.
    """
    if '{}' not in iter_dir_pattern:
        raise ValueError("iter_dir_pattern must contain '{}' as a placeholder for iteration number.")

    pattern_prefix = iter_dir_pattern.split('{}')[0]
    all_dirs = os.listdir(run_dir_base)
    all_pattern_prefix_dirs = [d for d in all_dirs if d.startswith(pattern_prefix)]

    if not all_pattern_prefix_dirs:
        raise ValueError('No directory matching given pattern was found!')

    logger.info(f'{len(all_pattern_prefix_dirs)} matching directories found: {all_pattern_prefix_dirs}')

    forecasts_all_dict: dict[str, pd.DataFrame] = {}

    for d in all_pattern_prefix_dirs:
        try:
            iter_path = os.path.join(run_dir_base, d, sub_dir)
            iter_num = d.split('_')[-1]
            additional_str = f'{suffix_prefix}{iter_num}'
            forecasts = loader_func(iter_path, additional_str) if loader_func else pd.DataFrame()
            forecasts_all_dict[f'iter_{iter_num}'] = forecasts
        except Exception as e:
            logger.warning(f"Error loading iteration {d}: {e}")

    if not forecasts_all_dict:
        raise ValueError("No forecast data was successfully loaded.")

    forecasts_all = pd.concat(forecasts_all_dict, axis=1)
    forecasts_all.columns = forecasts_all.columns.droplevel(0) if isinstance(forecasts_all.columns, pd.MultiIndex) else forecasts_all.columns
    return forecasts_all

## -----------------------------------------------------------------------------------------------------------------------------#

def add_percentile_means(
    df: pd.DataFrame,
    on_invalid: Literal["raise", "ignore"] = "raise",
    suffix_template: str = ''
) -> pd.DataFrame:
    """
    Adds mean columns for each percentile group (e.g., P50, P90) based on column name prefixes.

    Percentile columns must start with 'Pxx_' (e.g., 'P50_', 'P90_'). For each group, a new column
    '<Pxx>_mean_<suffix_template>_all' is added with the row-wise mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with percentile benchmark columns.
    on_invalid : {'raise', 'ignore'}, default='raise'
        How to handle columns not matching 'Pxx_' pattern.
    suffix_template : str, default=''
        Optional string to include in the new column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with added mean columns.

    Raises
    ------
    ValueError
        If invalid columns are found and `on_invalid='raise'`.
    """
    valid_pattern = re.compile(r'^P\d+_')
    valid_cols = [c for c in df.columns if valid_pattern.match(c)]
    invalid_cols = [c for c in df.columns if not valid_pattern.match(c)]

    if invalid_cols:
        if on_invalid == "raise":
            raise ValueError(
                f"The following columns do not start with a valid percentile pattern 'Pxx_': {invalid_cols}"
            )
        elif on_invalid != "ignore":
            raise ValueError("`on_invalid` must be 'raise' or 'ignore'.")

    percentiles = list(set([p.split('_')[0] for p in valid_cols]))

    df = df.copy()
    for P in percentiles:
        cols = df.filter(like=P).columns
        if cols.size:
            new_col_name = f'{P}_mean_{suffix_template}_all'
            df[new_col_name] = df[cols].mean(axis=1)

    return df

## -----------------------------------------------------------------------------------------------------------------------------#

def evaluate_forecasts_consistency(X: DataFrame) -> DataFrame:
    """
    Evaluate the consistency of forecasts by calculating the relative variability
    (coefficient of variation) across forecasts for each row, and add it as a new column.

    Parameters:
        X (pd.DataFrame): DataFrame where each column is a forecast.

    Returns:
        pd.DataFrame: The original DataFrame with an added column
                      'forecasts_relative_variability' containing std / mean per row.
    """
    X = X.copy()
    X['forecasts_mean'] = X.mean(axis=1)
    X['forecasts_std'] = X.std(axis=1)
    X['forecasts_relative_variability'] = X['forecasts_std'] / X['forecasts_mean']
    return X

## -----------------------------------------------------------------------------------------------------------------------------#

def merge_dataframes(dataframes_list: List[DataFrame], how: str = 'inner') -> DataFrame:
    """
    Merges a list of pandas DataFrames on their indices using the specified join method.

    Parameters:
        dataframes_list (List[pd.DataFrame]): List of pandas DataFrames to merge.
        how (str): Type of merge to be performed – 'left', 'right', 'outer', 'inner'. Default is 'inner'.

    Returns:
        pd.DataFrame: A single DataFrame resulting from merging all input DataFrames on their indices.
    
    Raises:
        ValueError: If the input list is empty.
    """
    if not dataframes_list:
        raise ValueError("The input list of DataFrames is empty.")

    return reduce(
        lambda left, right: left.merge(right, left_index=True, right_index=True, how=how),
        dataframes_list
    )

## -----------------------------------------------------------------------------------------------------------------------------#

def time_aggregate_format(
    df: Union[pd.DataFrame, pd.Series],
    value_col: Optional[str] = None,
    aggfunc: Union[str, Callable] = 'mean',
    time_units: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregates a time-series DataFrame or Series by specified datetime components,
    applying an aggregation function on a given column.

    Supports special units:
      - 'quarter_hour': 15-minute intervals (0-3 per hour)
      - 'time': exact time of day as datetime.time objects (hour:minute:second)

    Parameters:
    -----------
    df : pd.DataFrame or pd.Series
        Input with a datetime index.
    value_col : str, optional
        Name of the column to aggregate (ignored if df is a Series).
    aggfunc : function or str, default='mean'
        Aggregation function to apply (e.g., 'mean', 'sum', np.mean).
    time_units : list of str, optional
        List of datetime components to aggregate by, in order.
        Supported units: 'year', 'month', 'day', 'hour', 'minute', 'second',
                         'quarter_hour', 'time'.
        Default is ['month', 'year'].

    Returns:
    --------
    pd.DataFrame
        Pivoted DataFrame with the first time unit as index and subsequent
        units as columns, containing aggregated values.

    Raises:
    -------
    ValueError:
        If the index is not datetime,
        or if value_col is required but not found,
        or if unsupported time units are given.
    """
    # Convert Series to DataFrame
    if isinstance(df, pd.Series):
        if df.name is None:
            df = df.to_frame(name='value')
        else:
            df = df.to_frame()
        value_col = df.columns[0]
    else:
        df = df.copy()
        if value_col is None:
            raise ValueError("value_col must be provided when using a DataFrame.")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in DataFrame.")

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime.")

    if time_units is None:
        time_units = ['month', 'year']

    supported_units = {
        'year': df.index.year,
        'month': df.index.month,
        'day': df.index.day,
        'date': df.index.date,
        'hour': df.index.hour,
        'minute': df.index.minute,
        'second': df.index.second,
        'time': df.index.time,
    }

    for unit in time_units:
        if unit not in supported_units and unit != 'quarter_hour':
            raise ValueError(f"Unsupported time unit '{unit}'. Supported units: "
                             f"{list(supported_units.keys()) + ['quarter_hour']}")

    for unit in time_units:
        if unit == 'quarter_hour':
            df[unit] = df.index.minute // 15
        else:
            df[unit] = supported_units[unit]

    index_unit = time_units[0]
    columns_units = time_units[1:] if len(time_units) > 1 else None

    if columns_units:
        result = df.pivot_table(
            index=index_unit,
            columns=columns_units,
            values=value_col,
            aggfunc=aggfunc
        )
    else:
        result = df.groupby(index_unit)[value_col].agg(aggfunc)

    return result

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

def flatten_multiindex_to_timestamp(obj: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
    """
    Converts a Series or DataFrame with a MultiIndex (time, date)
    into one with a single datetime index.

    Parameters:
    obj (Union[pd.Series, pd.DataFrame]): Object with MultiIndex (time, date)

    Returns:
    Union[pd.Series, pd.DataFrame]: Same type as input, with timestamp index
    """
    if not isinstance(obj, (Series, DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")
    
    if not isinstance(obj.index, pd.MultiIndex) or obj.index.nlevels != 2:
        raise ValueError("Index must be a MultiIndex with exactly two levels: (time, date).")
    
    # Try to access named levels 'time' and 'date' if available
    level_names = obj.index.names
    try:
        time_level = obj.index.get_level_values('time')
        date_level = obj.index.get_level_values('date')
    except (KeyError, ValueError):
        # Fallback to positional index
        time_level = obj.index.get_level_values(0)
        date_level = obj.index.get_level_values(1)

    # Combine into datetime
    timestamps = pd.to_datetime(date_level.astype(str) + ' ' + time_level.astype(str))

    # Replace index
    obj_new = obj.copy()
    obj_new.index = timestamps
    obj_new.index.name = 'timestamp'
    obj_new.sort_index(inplace=True)

    return obj_new






