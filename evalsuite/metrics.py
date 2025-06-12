import pandas as pd
import numpy as np
from typing import Callable, Dict, Iterable, Mapping, Tuple

## -----------------------------------------------------------------------------------------------------------------------------#

def mape(realised: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between actual and forecasted values.

    MAPE is calculated as the mean of the absolute differences between realised and forecast values,
    divided by the realised values. It measures prediction accuracy as a percentage. NaNs are dropped
    before calculation.

    Parameters
    ----------
    realised : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.

    Returns
    -------
    float
        The MAPE value rounded to 3 decimal places. Returns NaN if input is invalid or causes an error.

    Notes
    -----
    - The calculation excludes any rows where either `realised` or `forecast` is NaN.
    - If `realised` contains zero values, this may cause division by zero and result in inf or NaN.
    - A try-except block is used to return NaN in case of any errors during the calculation.
    """
    try: 
        df = pd.concat([realised, forecast], axis=1).dropna()
        realised = df.iloc[:, 0]
        forecast = df.iloc[:, 1]
        abs_diff = np.abs(realised - forecast)
        abs_percentage_error = np.abs(abs_diff / realised)
        return np.round(np.mean(abs_percentage_error), 4)
    except:
        return np.nan


## -----------------------------------------------------------------------------------------------------------------------------#

def mae(realised: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between actual and forecasted values.

    MAE is the average of the absolute differences between realised and forecast values,
    providing a measure of prediction accuracy in the same units as the data.

    Parameters
    ----------
    realised : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.

    Returns
    -------
    float
        The MAE value. Returns NaN if input is invalid or causes an error.

    Notes
    -----
    - The calculation excludes any rows where either `realised` or `forecast` is NaN.
    - A try-except block is used to return NaN in case of any errors during the calculation.
    """
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()
        realised = df.iloc[:, 0]
        forecast = df.iloc[:, 1]
        return np.mean(np.abs(forecast - realised))
    except:
        return np.nan

## -----------------------------------------------------------------------------------------------------------------------------#

def r_mae(realised: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Relative Mean Absolute Error (R-MAE) between actual and forecasted values.

    R-MAE is the Mean Absolute Error normalized by the range of the realised values,
    where the range is computed as the difference between the maximum and minimum realised
    values over the entire period (e.g., previous two weeks). This normalization makes
    the error relative to the variability of the data.

    Parameters
    ----------
    realised : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.

    Returns
    -------
    float
        The relative mean absolute error. Returns NaN if the range is zero or if any error occurs.

    Notes
    -----
    - The calculation excludes any rows where either `realised` or `forecast` is NaN.
    - If the realised values have zero range (max equals min), the function returns NaN to avoid division by zero.
    - This function depends on the `mae` function for calculating the mean absolute error.
    - A try-except block is used to safely return NaN in case of any errors.
    """
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()
        realised = df.iloc[:, 0]
        forecast = df.iloc[:, 1]
        range_ = (realised.max() - realised.min() if (realised.max() - realised.min()) != 0 else np.nan)
        return mae(realised, forecast) / range_
    except:
        return np.nan

## -----------------------------------------------------------------------------------------------------------------------------#

def r_mae_percentiles(
    realised: pd.Series, 
    forecast: pd.Series, 
    low_percentile: float = 1, 
    high_percentile: float = 99
) -> float:
    """
    Calculate the Relative Mean Absolute Error (R-MAE) normalized by the inter-percentile range of realised values.

    This function computes the MAE between realised and forecast values, then normalizes it
    by the range between the specified lower and upper percentiles of the realised values.
    By default, it uses the 1st and 99th percentiles to reduce the influence of extreme outliers.

    Parameters
    ----------
    realised : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.
    low_percentile : float, optional
        The lower percentile to use for range calculation (default is 1).
    high_percentile : float, optional
        The upper percentile to use for range calculation (default is 99).

    Returns
    -------
    float
        The relative mean absolute error normalized by the specified percentile range.
        Returns NaN if the percentile range is zero or if an error occurs.

    Notes
    -----
    - The calculation excludes any rows where either `realised` or `forecast` is NaN.
    - If the percentile range is zero, the function returns NaN to avoid division by zero.
    - This function relies on the `mae` function for calculating the mean absolute error.
    - A try-except block ensures any exceptions during calculation result in NaN being returned.
    """
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()
        realised = df.iloc[:, 0]
        forecast = df.iloc[:, 1]
        p_low = np.percentile(realised, low_percentile)
        p_high = np.percentile(realised, high_percentile)
        range_ = p_high - p_low if (p_high - p_low) != 0 else np.nan
        return mae(realised, forecast) / range_
    except:
        return np.nan

## -----------------------------------------------------------------------------------------------------------------------------#

def smape(realized: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between realized and forecast values.

    SMAPE is a normalized measure of prediction accuracy that treats over- and under-forecasts symmetrically.
    It is calculated as the mean of the absolute differences divided by the average absolute values of
    the realized and forecast values.

    Parameters
    ----------
    realized : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.

    Returns
    -------
    float
        The SMAPE value rounded to 3 decimal places. Returns NaN if an error occurs.

    Notes
    -----
    - The calculation excludes any rows where either `realized` or `forecast` is NaN.
    - To avoid division by zero, any zero denominators are replaced with NaN.
    - A try-except block catches and prints exceptions, returning NaN on failure.
    """
    try:
        df = pd.concat([realized, forecast], axis=1).dropna()
        realized = df.iloc[:, 0]
        forecast = df.iloc[:, 1]
        denominator = (np.abs(realized) + np.abs(forecast)) / 2
        smape_val = np.mean(np.abs(realized - forecast) / denominator.replace(0, np.nan))
        return np.round(smape_val, 3)
    except Exception as e:
        print(e)
        return np.nan

    
## -----------------------------------------------------------------------------------------------------------------------------#

def wmape(realized: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE) between realized and forecast values.

    WMAPE is computed as the sum of absolute errors divided by the sum of the absolute realized values,
    providing a normalized error metric weighted by the magnitude of the actual values.

    Parameters
    ----------
    realized : pd.Series
        The actual observed values.
    forecast : pd.Series
        The predicted or forecasted values.

    Returns
    -------
    float
        The WMAPE value rounded to 3 decimal places. Returns NaN if an error occurs.

    Notes
    -----
    - The calculation excludes any rows where either `realized` or `forecast` is NaN.
    - If the sum of absolute realized values is zero, this may result in division by zero.
    - A try-except block catches and prints exceptions, returning NaN on failure.
    """
    try:
        df = pd.concat([realized, forecast], axis=1).dropna()
        numerator = np.sum(np.abs(df.iloc[:, 0] - df.iloc[:, 1]))
        denominator = np.sum(np.abs(df.iloc[:, 0]))
        return np.round(numerator / denominator, 3)
    except Exception as e:
        print(e)
        return np.nan

    
## -----------------------------------------------------------------------------------------------------------------------------#

def calculate_metrics(
    df: pd.DataFrame,
    target_variable: str,
    criterion: str | Callable[[pd.Series, pd.Series], float] = "mape",
    *,
    frequencies: Iterable[str] = ("D", "QE")   # pandas offset aliases: D=daily, Q=quarterly
) -> Tuple[
    Dict[str, Dict[str, pd.Series]],   # {frequency -> {column -> Series}}
    Dict[str, float]                   # overall {column -> value}
]:
    """
    Compute evaluation metrics overall and at one or more temporal resolutions.

    Parameters
    ----------
    df : pd.DataFrame
        Must be indexed by a DatetimeIndex and contain the target and prediction columns.
    target_variable : str
        Column name of the observed / ground-truth series.
    criterion : str | Callable, default "mape"
        • If str  - name of a metric function available in ``globals()``  
        • If callable - function with signature ``f(y_true, y_pred) -> float``
    frequencies : Iterable[str], default ("D", "Q")
        Any pandas offset alias(es) to group by, e.g. "H" for hourly, "M" for monthly,
        "Q" for calendar quarter, etc.

    Returns
    -------
    tuple
        1. ``freq_metrics`` - mapping *frequency → {column → Series}*  
           (each Series is indexed by the period start date)
        2. ``overall_metrics`` - mapping *column → single float*.

    Raises
    ------
    ValueError
        • If *target_variable* not found  
        • If *criterion* (when str) not available.
    """
    # ---------------- validation ----------------
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in DataFrame.")

    if isinstance(criterion, str):
        try:
            metric_fn: Callable[[pd.Series, pd.Series], float] = globals()[criterion]
        except KeyError as exc:
            raise ValueError(
                f"Metric function '{criterion}' is not defined in the global scope."
            ) from exc
    elif callable(criterion):
        metric_fn = criterion
    else:
        raise TypeError("criterion must be a str or a callable")

    # we won't mutate the caller's DataFrame
    df = df.copy()

    # ---------------- calculations ----------------
    pred_columns = df.columns.difference([target_variable])
    freq_metrics: Dict[str, Dict[str, pd.Series]] = {f: {} for f in frequencies}
    overall_metrics: Dict[str, float] = {}

    for col in pred_columns:
        # overall metric (whole DataFrame)
        overall_metrics[col] = metric_fn(df[target_variable], df[col])

        # metrics per required frequency
        for freq in frequencies:
            # use Grouper so the index stays a DatetimeIndex
            grouped = df.groupby(pd.Grouper(freq=freq, label="left", closed="left"))
            freq_metrics[freq][col] = grouped.apply(
                lambda g: metric_fn(g[target_variable], g[col])
            )

    return freq_metrics, overall_metrics
## -----------------------------------------------------------------------------------------------------------------------------#

def evaluate_calendar_types(
    df: pd.DataFrame,
    calendar_data_type_list: list,
    criterion: str = 'mape',
    area: str = 'aggregated'
) -> pd.DataFrame:
    """
    Evaluates forecast accuracy across different calendar types.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'load', 'forecast', and calendar-type boolean columns.
    - calendar_data_type_list (list): List of column names representing calendar filters (boolean masks).
    - criterion (str): Evaluation metric to use (e.g., 'mape', 'rmse'). Default is 'mape'.
    - area (str): Area name to be included in the result column name. Default is 'aggregated'.

    Returns:
    - pd.DataFrame: A DataFrame with calendar_type and corresponding evaluation metric.
    """
    df = df.copy()
    calendar_type_eval_dict = {}

    for calendar_type in calendar_data_type_list:
        filtered_df = df[df[calendar_type]][['load', 'forecast']].copy()
        calendar_type_eval_dict[calendar_type] = calculate_metrics(
            filtered_df,
            target_variable='load',
            criterion=criterion,
            frequencies=()
        )[1]['forecast']

    calendar_type_eval_df = pd.DataFrame(
        calendar_type_eval_dict.items(),
        columns=['calendar_type', f'{criterion} {area}']
    )

    return calendar_type_eval_df

## -----------------------------------------------------------------------------------------------------------------------------#


