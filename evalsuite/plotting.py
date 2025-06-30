
import os
import pandas as pd

from typing import List, Optional
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from evalsuite.metrics import *

## -----------------------------------------------------------------------------------------------------------------------------#

def plot_timeseries(
    timeseries_df: pd.DataFrame,
    title: str = 'Overview of data',
    output_filename: str = 'timeseries',
    output_dir: str = './',
    color_mapping: dict = None  # New optional color palette
) -> None:
    """
    Plot a time series DataFrame and save the plot as an interactive HTML file.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        A pandas DataFrame containing the time series data to be plotted.
    title : str, optional
        The title of the plot. Default is 'Overview of data'.
    output_filename : str, optional
        The base name of the output HTML file (without extension). Default is 'timeseries'.
    output_dir : str, optional
        The directory where the HTML file will be saved. Default is the current directory ('./').
    color_mapping : dict, optional
        A dict of colors to use for the lines. If None, the default Plotly colors are used.

    Returns
    -------
    None
        This function does not return anything. It saves the plot to an HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timeseries_df = timeseries_df.copy()

    # Build the plot
    fig = px.line(
        timeseries_df,
        title=title,
        color_discrete_map=color_mapping  # Use custom color palette if provided
    )

    # Save outputs
    html_path = os.path.join(output_dir, f'{output_filename}.html')
    img_path = os.path.join(output_dir, f'{output_filename}.png')

    fig.write_html(html_path)
    fig.write_image(img_path)

    fig.write_html(html_path, auto_open=False)
    fig.update_layout(width=1500, height=600)
    fig.write_image(img_path, format='png', scale=3)

## -----------------------------------------------------------------------------------------------------------------------------#

def barplot(eval_overall_df: pd.DataFrame, x: str, y: str,
            color: str = '', title: str = '', barmode: str = 'group',
            output_dir: str = './outputs', output_filename: str = 'barplot',
            sort_by_y: bool = True) -> None:
    """
    Plots a bar chart from the evaluation DataFrame with optional color grouping.

    Parameters:
        eval_overall_df (pd.DataFrame): The input data.
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        color (str): Optional column name for color grouping.
        title (str): Title of the plot.
        output_dir (str): Directory to save outputs (not used in current function).
        output_filename (str): Name of the output file (without extension).
        sort_by_y (bool): Whether to sort the bars by y-values if x-values are unique.
    """

    # Determine whether x-axis has repeated categories
    x_value_counts = eval_overall_df[x].value_counts()
    x_is_unique = all(x_value_counts == 1)

    if x_is_unique and sort_by_y:
        # Sort x by y-values (descending)
        sorted_df = eval_overall_df.sort_values(by=y, ascending=True)
        ordered_x = sorted_df[x].tolist()
    else:
        # Use unique categories in their existing order (or fallback to sorted)
        ordered_x = eval_overall_df[x].unique().tolist()

    # Apply the ordered categorical x
    eval_overall_df[x] = pd.Categorical(eval_overall_df[x], categories=ordered_x, ordered=True)

    # Handle color
    color_map = None
    if color and color in eval_overall_df.columns:
        unique_categories = sorted(eval_overall_df[color].dropna().unique())
        if len(unique_categories) <= 10:
            palette = px.colors.qualitative.Plotly
        else:
            palette = sns.color_palette("hsv", len(unique_categories)).as_hex()
        color_map = dict(zip(unique_categories, palette))
    else:
        color = None  # Disable color if column doesn't exist or not provided

    # Create bar chart
    fig = px.bar(
        eval_overall_df,
        x=x,
        y=y,
        title=title,
        color=color,
        color_discrete_map=color_map if color_map else None,
        category_orders={x: eval_overall_df[x].cat.categories.tolist()},
        barmode=barmode,
        opacity=0.9

    )

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f'{output_filename}.html')
    img_path = os.path.join(output_dir, f'{output_filename}.png')

    fig.write_html(html_path, auto_open=False)
    fig.update_layout(width=1500, height=600)
    fig.write_image(img_path, format='png', scale=3)

## -----------------------------------------------------------------------------------------------------------------------------#

def boxplot(
    eval_daily_df: pd.DataFrame, 
    x: str, 
    y: str,   
    color: str = '', 
    title: str = '', 
    output_dir: str = './outputs', 
    output_filename: str = 'boxplot'
):
    """
    Plots a boxplot of daily evaluation metrics.

    Parameters:
        eval_daily_df (pd.DataFrame): The input DataFrame.
        x (str): Column name for the x-axis (categorical).
        y (str): Column name for the y-axis (numeric).
        color (str): Optional column name for color grouping.
        title (str): Title for the plot.
        output_dir (str): Directory to save outputs.
        output_filename (str): Base filename for saved plot files (HTML and PNG).
    """

    eval_daily_df = eval_daily_df.copy()

    # Order x by mean of y
    order = (
        eval_daily_df.groupby(x)[y]
        .mean()
        .sort_values()
        .index.tolist()
    )
    eval_daily_df[x] = pd.Categorical(eval_daily_df[x], categories=order, ordered=True)

    # Handle color palette
    color_map = None
    if color and color in eval_daily_df.columns:
        unique_categories = sorted(eval_daily_df[color].dropna().unique())
        if len(unique_categories) <= 10:
            palette = px.colors.qualitative.Plotly
        else:
            palette = sns.color_palette("hsv", len(unique_categories)).as_hex()
        color_map = dict(zip(unique_categories, palette))
    else:
        color = None  # Disable color if column is missing or not provided

    # Create the boxplot
    fig = px.box(
        eval_daily_df,
        x=x,
        y=y,
        points='outliers',
        color=color,
        color_discrete_map=color_map if color_map else None,
        category_orders={x: order},
        title=title + f' (sorted by mean {y})'
    )

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f'{output_filename}_boxplot.html')
    img_path = os.path.join(output_dir, f'{output_filename}_boxplot.png')

    fig.write_html(html_path, auto_open=False)
    fig.update_layout(width=1500, height=600)
    fig.write_image(img_path, format='png', scale=3)

## -----------------------------------------------------------------------------------------------------------------------------#

def save_dataframe_as_png(df, output_path):

    """Save a pandas DataFrame as a PNG image."""
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2.9, len(df) * 0.55))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

## -----------------------------------------------------------------------------------------------------------------------------#

def process_and_plot_calendar_types(
    df: pd.DataFrame,
    calendar_data_type_list: List[str],
    output_dir: str = "./",
    area: Optional[str] = None,
    add_day_name: bool = True,
) -> None:
    """
    Processes calendar types in a DataFrame and plots time series data.
    """

    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Strip time from date
    df["date"] = df["date"].dt.date

    for calendar_type in calendar_data_type_list:
        if calendar_type not in df.columns:
            print(f"Warning: Column '{calendar_type}' not found. Skipping.")
            continue

        filtered_df = df[df[calendar_type]].copy()

        if add_day_name:
            filtered_df["day_name"] = pd.Series(pd.to_datetime(filtered_df["date"])).dt.day_name()
            filtered_df["label"] = filtered_df["date"].astype(str) + "_" + filtered_df["day_name"]
        else:
            filtered_df["label"] = filtered_df["date"].astype(str)
        
        unique_days = list(filtered_df["label"].unique())

        pivot_df = (
            filtered_df.pivot_table(index="time", columns="label", values=["load", 'forecast'])
            .sort_index(axis=1)
        )
        
        for a_given_day in unique_days:

            a_given_date_data = pivot_df.xs(a_given_day, axis=1, level=1)

            overall_err_val = calculate_metrics(a_given_date_data, target_variable='load', frequencies=())[1]['forecast']

            title_stub = f"<b>{calendar_type}</b>"
            title = (
                f"Overview of load and forecast on {a_given_day}, categorized as {title_stub}, aggregated over 3 areas - <b>{overall_err_val}</b> MAPE"
                if area is None
                else f"Overview of load and forecast on {a_given_day}, categorized as {title_stub}, in the {area} area - <b>{overall_err_val}</b> MAPE"
            )

            filename = f"{calendar_type}_{a_given_day}_{'agg_3_areas' if area is None else area}"

            os.makedirs(output_dir, exist_ok=True)

            plot_timeseries(
                a_given_date_data,
                title=title,
                output_dir=output_dir,
                output_filename=filename,
                color_mapping={'load':'blue', 'forecast':'red'}
            )

## -----------------------------------------------------------------------------------------------------------------------------#

def compare_monthly_metrics(
    monthly_results_dict: dict,
    output_dir: str = '.',
    criterion: str = 'MAPE',
    area: Optional[str] = None,
    title: str = 'Monthly MAPE Comparison',
    output_filename: str = 'mape_results'
) -> None:
    """
    Combine experiment DataFrames from a dictionary, reshape them, and plot a monthly comparison of specified metrics.

    This function takes a dictionary of DataFrames containing performance metrics (e.g., P50, P90) for different experiments.
    It reshapes the data into long format and generates a bar plot comparing metrics across months and experiments.

    Parameters
    ----------
    monthly_results_dict : dict
        Dictionary where keys are experiment names and values are pandas DataFrames.
        Each DataFrame must have:
          - A datetime index representing months.
          - An 'Experiment' column identifying the experiment.
          - 'P50' and/or 'P90' columns (or other metrics).
    output_dir : str, optional
        Directory to save the output plot. Default is the current directory ('.').
    criterion : str, optional
        Name of the metric to plot (e.g., 'MAPE', 'RMSE'). Default is 'MAPE'.
        This is used as the y-axis label.
    area : Optional[str], optional
        (Optional) Name of an area or group; not currently used but reserved for future extensions.
    title : str, optional
        Title of the plot. Default is 'Monthly MAPE Comparison'.
    output_filename : str, optional
        Base name for the saved plot file (without extension). Default is 'mape_results'.

    Raises
    ------
    ValueError
        If any input DataFrame does not include an 'Experiment' column.

    Notes
    -----
    This function expects that a `barplot` function is defined elsewhere
    which generates and saves the visualization.
    """
    df_list = []

    # Prepare each DataFrame
    for df in monthly_results_dict.values():
        temp_df = df.copy()
        temp_df.index.name = 'Month'
        temp_df = temp_df.reset_index()

        if 'Experiment' not in temp_df.columns:
            raise ValueError("Each DataFrame must include an 'Experiment' column.")

        df_list.append(temp_df)

    # Combine all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)

    # Reshape to long format (only P50 shown, adjust if needed)
    melted_df = pd.melt(
        combined_df,
        id_vars=["Month", "Experiment"],
        value_vars=["P50"],
        var_name="Percentile",
        value_name=criterion
    )

    # Create a combined label for coloring
    melted_df['Experiment-Percentile'] = (
        melted_df['Experiment'] + '-' + melted_df['Percentile']
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate bar plot
    barplot(
        melted_df,
        x='Month',
        y=criterion,
        color='Experiment-Percentile',
        sort_by_y=False,
        output_dir=output_dir,
        title=title,
        output_filename=output_filename
    )

## -----------------------------------------------------------------------------------------------------------------------------#


