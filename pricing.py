
import os
import pandas as pd
import numpy as np
import logging
from typing import List
from typing import Union

from evaluation_utils.utils import set_tz, merge_dataframes

logger = logging.getLogger(__name__)

## -----------------------------------------------------------------------------------------------------------------------------#

def get_settlement_apx_prices(data_dir):
    """
    Load and merge settlement prices and APX prices into a single DataFrame.

    Args:
        data_dir (str): Directory path containing the 'predictors_CONTINUON_all.csv' file.
    Returns:
        pd.DataFrame: Merged DataFrame containing settlement prices and APX prices.
    """

    settlement_prices_dir = os.path.join(data_dir, 'settlement_prices')
    logger.info(f'Settlement price directory: {settlement_prices_dir}')
    
    settlement_prices_files = [
        f for f in os.listdir(settlement_prices_dir) if f.endswith('.csv')
    ]
    logger.info(f'There are {len(settlement_prices_files)} settlement price files in the directory: {settlement_prices_dir}')

    settlement_prices = pd.concat(
        pd.read_csv(os.path.join(settlement_prices_dir, f), delimiter=';')[
            ['Timeinterval End Loc', 'Price Shortage', 'Price Surplus']
        ]
        for f in settlement_prices_files
    )
    settlement_prices = settlement_prices.rename(columns={'Timeinterval End Loc': 'datetimeFC'})
    settlement_prices = settlement_prices.set_index('datetimeFC')
    settlement_prices = set_tz(settlement_prices)

    logger.info(f'Settlement price timestamps: {settlement_prices.index.min()} to {settlement_prices.index.max()}')

    # CONTINUON is an example, the price is the same for all areas!
    apx_prices = pd.read_csv(
        os.path.join(data_dir, 'predictors_CONTINUON_all.csv'), index_col=0
    )[['APX']].dropna()
    apx_prices = set_tz(apx_prices)

    logger.info(f'APX prices timestamps: {apx_prices.index.min()} to {apx_prices.index.max()}')

    settlement_apx_prices = merge_dataframes([settlement_prices, apx_prices])
    return settlement_apx_prices
## -----------------------------------------------------------------------------------------------------------------------------#

def calculate_mismatch(
    measured_forecasts_df: pd.DataFrame,
    target_variable: str,
) -> pd.DataFrame:
    """
    Calculate the mismatch between each forecast column and the target variable.

    Parameters:
    - measured_forecasts_df (DataFrame): DataFrame containing forecast and target data
    - target_variable (str): Name of the column to be treated as the target variable

    Returns:
    - DataFrame: DataFrame containing the mismatch for each forecast column
    """
    
    if target_variable not in measured_forecasts_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in DataFrame columns.")
    if measured_forecasts_df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(measured_forecasts_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(target_variable, str):
        raise TypeError("Target variable must be a string.")

    mismatch_measured_forecasts_df = pd.DataFrame()
    for c in measured_forecasts_df.columns.difference([target_variable]):
        mismatch_measured_forecasts_df[c] = (
            measured_forecasts_df[c] - measured_forecasts_df[target_variable]
        )
    return mismatch_measured_forecasts_df
## -----------------------------------------------------------------------------------------------------------------------------#

def calculate_mismatch_costs(
    mismatch_measured_forecasts_df: pd.DataFrame,
    settlement_apx_prices: pd.DataFrame,
    energy_scaling_factor: Union[int, float] = 1000,
    gridloss_price_factor: Union[int, float] = 1
) -> pd.DataFrame:
    """
    Calculate the mismatch costs between measured and forecasted values using APX settlement prices.

    Args:
        mismatch_measured_forecasts_df (pd.DataFrame): DataFrame with mismatch values.
        settlement_apx_prices (pd.DataFrame): DataFrame with 'APX', 'Price Surplus', and 'Price Shortage'.
        energy_scaling_factor (int | float): Factor to scale energy values (default: 1000).
        gridloss_price_factor (int | float): Multiplier for adjusting the final mismatch costs (default: 1).

    Returns:
        pd.DataFrame: DataFrame with calculated mismatch costs for each relevant column.
    """
    mismatch_measured_forecasts_df = mismatch_measured_forecasts_df.copy()
    mismatch_measured_forecasts_df /= energy_scaling_factor

    mismatch_measured_forecasts_df = merge_dataframes([
        mismatch_measured_forecasts_df, 
        settlement_apx_prices
    ])

    mismatch_costs_df = pd.DataFrame()

    for c in mismatch_measured_forecasts_df.columns.difference(set(settlement_apx_prices.columns)):
        mismatch_costs_df[c] = mismatch_measured_forecasts_df.apply(
            lambda row: row[c] * (-row['Price Surplus'] + row['APX']) if row[c] > 0 
            else abs(row[c]) * (row['Price Shortage'] - row['APX']), 
            axis=1
        )

    mismatch_costs_df *= gridloss_price_factor

    return mismatch_costs_df

## -----------------------------------------------------------------------------------------------------------------------------#




