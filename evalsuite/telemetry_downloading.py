
import requests
import pandas as pd
from typing import List, Dict, Any, Optional


## -----------------------------------------------------------------------------------------------------------------------------#

def get_measured_grid_area_data(
    url: str,
    cookie_str: str,
    start_date_str: str,
    end_date_str: str,
    terminals: List[str],
    minimal_known_after: str = "P3D"
) -> requests.Response:
    """
    Sends a POST request to get measured grid area data.

    Args:
        url: The endpoint URL.
        cookie_str: The cookie string for authentication.
        start_date_str: Start date as ISO8601 string.
        end_date_str: End date as ISO8601 string.
        terminals: List of terminal IDs.
        minimal_known_after: Minimal known after value (default "P3D").

    Returns:
        The response object from the requests library.
    """
    headers: Dict[str, str] = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Cookie": cookie_str
    }

    payload: Dict[str, Any] = {
        "terminals": terminals,
        "range": {
            "start": start_date_str,
            "end": end_date_str,
            "sort": "Asc"
        },
        "analogs": [
            {
                "measurementValueSource": "Measured",
                "measurementType": "ActivePower",
                "measurementSource": "DSO",
                "unitMultiplier": "k",
                "measuringPeriod": "FifteenMinute",
                "phaseCode": "ABC",
                "aggregation": "Average",
                "minimalKnownAfter": minimal_known_after
            },
            {
                "measurementValueSource": "Measured",
                "measurementType": "ActiveEnergy",
                "measurementSource": "DSO",
                "unitMultiplier": "k",
                "measuringPeriod": "FifteenMinute",
                "phaseCode": "ABC",
                "aggregation": "None",
                "minimalKnownAfter": "P3D"
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers)

    return response

## -----------------------------------------------------------------------------------------------------------------------------#

def get_forecasted_grid_area_data(
    url: str,
    cookie_str: str,
    start_date_str: str,
    end_date_str: str,
    terminals: List[str],
    minimal_known_after: str = "P3D",
    percentile: int = 50
) -> requests.Response:
    """
    Sends a POST request to get forecasted grid area data.

    Args:
        url: The endpoint URL.
        cookie_str: Raw cookie string for authentication.
        start_date_str: ISO start date string.
        end_date_str: ISO end date string.
        terminals: List of terminal IDs.
        minimal_known_after: ISO duration string (default "P3D").
        percentile: Forecast percentile (default 50).

    Returns:
        The HTTP response object from the requests library.
    """
    headers: Dict[str, str] = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Cookie": cookie_str
    }

    payload: Dict[str, Any] = {
        "terminals": terminals,
        "range": {
            "start": start_date_str,
            "end": end_date_str,
            "sort": "Asc"
        },
        "analogs": [
            {
                "measurementValueSource": "Forecasted",
                "measurementType": "ActivePower",
                "measurementSource": "DSO",
                "unitMultiplier": "k",
                "measuringPeriod": "FifteenMinute",
                "phaseCode": "ABC",
                "aggregation": "Average",
                "minimalKnownAfter": minimal_known_after,
                "percentile": percentile
            },
            {
                "measurementValueSource": "Forecasted",
                "measurementType": "ActiveEnergy",
                "measurementSource": "DSO",
                "unitMultiplier": "k",
                "measuringPeriod": "FifteenMinute",
                "phaseCode": "ABC",
                "aggregation": "Average",
                "minimalKnownAfter": minimal_known_after,
                "percentile": percentile
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers)

    return response

## -----------------------------------------------------------------------------------------------------------------------------#

def process_measured_response(
    response: Any,
    area_mrid_dict: Dict[str, str]
) -> Optional[pd.DataFrame]:
    """
    Processes the response from get_measured_grid_area_data and returns a DataFrame.

    Args:
        response: The response object from requests.post.
        area_mrid_dict: Mapping of terminal IDs to area names.

    Returns:
        A pandas DataFrame with timestamps as index and measured analog values as columns.
        Returns None if processing fails.
    """
    try:
        data = response.json()
        if "analogs" not in data:
            raise KeyError("'analogs' key not found in response JSON.")

        df = pd.DataFrame(data["analogs"])
        # Select only the needed columns
        df = df[["terminal", "measurementType", "aggregation", "minimalKnownAfter", "analogValues"]].copy()

        # Map area names
        df["area"] = df["terminal"].map(area_mrid_dict)

        # Create analog type identifier
        df["analog_type"] = (
            df["area"] + "__" +
            df["measurementType"] + "__" +
            df["aggregation"] + "__" +
            df["minimalKnownAfter"]
        )

        df = df[["analog_type", "analogValues"]].set_index("analog_type")

        # Build time series DataFrame
        analog_df = pd.DataFrame()

        for analog_type in df.index:
            analog_values = pd.DataFrame(df.loc[analog_type, "analogValues"])
            analog_values = (
                analog_values[["value", "timestamp"]]
                .rename(columns={"value": analog_type})
                .set_index("timestamp")
            )
            analog_df = pd.concat([analog_df, analog_values], axis=1)

        # Add suffix to columns
        analog_df.columns = [col + "__measured" for col in analog_df.columns]

        return analog_df

    except Exception as e:
        print(f"Error processing measured response: {e}")
        return None

## -----------------------------------------------------------------------------------------------------------------------------#

def process_forecasts_response(
    response: Any,
    area_mrid_dict: Dict[str, str]
) -> Optional[pd.DataFrame]:
    """
    Processes the forecasted grid area data response and returns a DataFrame.

    Args:
        response: The response object from requests.post.
        area_mrid_dict: Mapping of terminal IDs to area names.

    Returns:
        A pandas DataFrame with timestamps as index and forecasted analog values as columns.
        Returns None if processing fails.
    """
    try:
        data = response.json()
        if "analogs" not in data:
            raise KeyError("'analogs' key not found in response JSON.")

        df = pd.DataFrame(data["analogs"])

        df = df[
            ["terminal", "measurementType", "percentile", "aggregation", "minimalKnownAfter", "analogValues"]
        ].copy()

        df["area"] = df["terminal"].map(area_mrid_dict)

        df["analog_type"] = (
            df["area"] + "__" +
            df["measurementType"] + "__" +
            "P" + df["percentile"].astype(str) + "__" +
            df["aggregation"] + "__" +
            df["minimalKnownAfter"]
        )

        df = df[["analog_type", "analogValues"]].set_index("analog_type")

        analog_df = pd.DataFrame()

        for analog_type in df.index:
            analog_values = pd.DataFrame(df.loc[analog_type, "analogValues"])
            analog_values = (
                analog_values[["value", "timestamp"]]
                .rename(columns={"value": analog_type})
                .set_index("timestamp")
            )
            analog_df = pd.concat([analog_df, analog_values], axis=1)

        analog_df.columns = [col + "__forecasts" for col in analog_df.columns]

        return analog_df

    except Exception as e:
        print(f"Error processing forecasts response: {e}")
        return None

## -----------------------------------------------------------------------------------------------------------------------------#
