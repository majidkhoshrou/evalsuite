import pandas as pd
import numpy as np
from typing import Dict, Tuple, Callable

## -----------------------------------------------------------------------------------------------------------------------------#

def mape(realised: pd.Series, forecast: pd.Series) -> float:
    """
    Function that calculates the relative mean absolute error based on the true and prediction.

    The range is based on the entire period of observation.
    """
    try: 
        df = pd.concat([realised, forecast], axis=1).dropna()  # it should not contain nan
        realised = df.iloc[:,0]
        forecast = df.iloc[:,1]
        abs_diff = np.abs(realised - forecast)
        abs_percentage_error = np.abs(abs_diff / realised)
        return np.round(np.mean(abs_percentage_error), 3)  # Result as percentage
    except:
        return np.nan 

## -----------------------------------------------------------------------------------------------------------------------------#

def mae(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the mean absolute error based on the true and prediction."""
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()  # it should not contain nan
        realised = df.iloc[:,0]
        forecast = df.iloc[:,1]
        return np.mean(np.abs(forecast - realised))
    except:
        return np.nan

## -----------------------------------------------------------------------------------------------------------------------------#

def r_mae(realised: pd.Series, forecast: pd.Series) -> float:
    """Function that calculates the relative mean absolute error based on the true and prediction.

    The range is based on the load range of the previous two weeks

    """
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()  # it should not contain nan
        realised = df.iloc[:,0]
        forecast = df.iloc[:,1]
        # Determine load range on entire dataset
        range_ = (realised.max() - realised.min() if (realised.max() - realised.min()) != 0  else np.nan)
        return mae(realised, forecast) / range_
    except:
        return np.nan

## -----------------------------------------------------------------------------------------------------------------------------#

def r_mae_percentiles(realised: pd.Series, forecast: pd.Series) -> float:
    """Calculates the relative mean absolute error using the 1st to 99th percentile range of the realised values."""
    try:
        df = pd.concat([realised, forecast], axis=1).dropna()  # it should not contain nan
        realised = df.iloc[:,0]
        forecast = df.iloc[:,1]
        # Compute the 1st and 99th percentiles
        p1 = np.percentile(realised, 1)
        p99 = np.percentile(realised, 99)
        # Avoid division by zero
        range_ = p99 - p1 if (p99 - p1) != 0 else np.nan
        return mae(realised, forecast) / range_
    except:
        return np.nan
    
## -----------------------------------------------------------------------------------------------------------------------------#

def smape(realized: pd.Series, forecast: pd.Series) -> float:
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
    try:
        df = pd.concat([realized, forecast], axis=1).dropna()
        numerator = np.sum(np.abs(df.iloc[:,0] - df.iloc[:,1]))
        denominator = np.sum(np.abs(df.iloc[:,0]))
        return np.round(numerator / denominator, 3)
    except Exception as e:
        print(e)
        return np.nan
    
## -----------------------------------------------------------------------------------------------------------------------------#
from typing import Callable, Dict, Iterable, Mapping, Tuple
import pandas as pd


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


