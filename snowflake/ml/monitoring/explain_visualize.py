from typing import Any, Union, cast, overload

import altair as alt
import numpy as np
import pandas as pd

import snowflake.snowpark.dataframe as sp_df
from snowflake import snowpark
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._signatures import snowpark_handler

DEFAULT_FIGSIZE = (1400, 500)
DEFAULT_VIOLIN_FIGSIZE = (1400, 100)
MAX_ANNOTATION_LENGTH = 20
MIN_DISTANCE = 10  # Increase minimum distance between labels for more spreading in plot_force


@overload
def plot_force(
    shap_row: snowpark.Row,
    features_row: snowpark.Row,
    base_value: float = 0.0,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    contribution_threshold: float = 0.05,
) -> alt.LayerChart:
    ...


@overload
def plot_force(
    shap_row: pd.Series,
    features_row: pd.Series,
    base_value: float = 0.0,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    contribution_threshold: float = 0.05,
) -> alt.LayerChart:
    ...


def plot_force(
    shap_row: Union[pd.Series, snowpark.Row],
    features_row: Union[pd.Series, snowpark.Row],
    base_value: float = 0.0,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    contribution_threshold: float = 0.05,
) -> alt.LayerChart:
    """
    Create a force plot for SHAP values with stacked bars based on influence direction.

    Args:
        shap_row: pandas Series or snowpark Row containing SHAP values for a specific instance
        features_row: pandas Series or snowpark Row containing the feature values for the same instance
        base_value: base value of the predictions. Defaults to 0, but is usually the model's average prediction
        figsize: tuple of (width, height) for the plot
        contribution_threshold:
            Only features with magnitude greater than contribution_threshold as a percentage of the
            total absolute SHAP values will be plotted. Defaults to 0.05 (5%)

    Returns:
        Altair chart object

    Raises:
        SnowflakeMLException: If the contribution threshold is not between 0 and 1,
            or if no features with significant contributions are found.
    """
    if not (0 < contribution_threshold and contribution_threshold < 1):
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("contribution_threshold must be between 0 and 1."),
        )

    if isinstance(shap_row, snowpark.Row):
        shap_row = pd.Series(shap_row.as_dict())
    if isinstance(features_row, snowpark.Row):
        features_row = pd.Series(features_row.as_dict())

    # Create a dataframe for plotting
    positive_label = "Positive"
    negative_label = "Negative"
    plot_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_value": features_row.iloc[index],
                "feature_annotated": f"{feature}: {features_row.iloc[index]}"[:MAX_ANNOTATION_LENGTH],
                "influence_value": shap_row.iloc[index],
                "bar_direction": positive_label if shap_row.iloc[index] >= 0 else negative_label,
            }
            for index, feature in enumerate(features_row.index)
        ]
    )

    # Calculate cumulative positions for the stacked bars
    shap_sum = np.sum(shap_row)
    current_position_pos = shap_sum
    current_position_neg = shap_sum
    positions = []

    total_abs_value_sum = np.sum(plot_df["influence_value"].abs())
    max_abs_value = plot_df["influence_value"].abs().max()
    spacing = max_abs_value * 0.07  # Use 2% of max value as spacing between bars

    # Sort by absolute value to have largest impacts first
    plot_df = plot_df.reindex(plot_df["influence_value"].abs().sort_values(ascending=False).index)
    for _, row in plot_df.iterrows():
        # Skip features with small contributions
        row_influence_value = row["influence_value"]
        if abs(row_influence_value) / total_abs_value_sum < contribution_threshold:
            continue

        if row_influence_value >= 0:
            start = current_position_pos - spacing
            end = current_position_pos - row_influence_value - spacing
            current_position_pos = end
        else:
            start = current_position_neg + spacing
            end = current_position_neg + abs(row_influence_value) + spacing
            current_position_neg = end

        positions.append(
            {
                "start": start,
                "end": end,
                "avg": (start + end) / 2,
                "influence_value": row_influence_value,
                "feature_value": row["feature_value"],
                "feature_annotated": row["feature_annotated"],
                "bar_direction": row["bar_direction"],
                "bar_y": 0,
                "feature": row["feature"],
            }
        )

    if len(positions) == 0:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "No features with significant contributions found. Try lowering the contribution_threshold,"
                "and verify the input is non-empty."
            ),
        )

    position_df = pd.DataFrame(positions)

    # Create force plot using Altair
    blue_color = "#1f77b4"
    red_color = "#d62728"
    width, height = figsize
    bars: alt.Chart = (
        alt.Chart(position_df)
        .mark_bar(size=10)
        .encode(
            x=alt.X("start:Q", title="Feature Impact"),
            x2=alt.X2("end:Q"),
            y=alt.Y("bar_y:Q", axis=None),
            color=alt.Color(
                "bar_direction:N",
                scale=alt.Scale(domain=[positive_label, negative_label], range=[red_color, blue_color]),
                legend=alt.Legend(title="Influence Direction"),
            ),
            tooltip=["feature", "influence_value", "feature_value"],
        )
        .properties(title="Feature Influence (SHAP values)", width=width, height=height)
    ).interactive()

    arrow: alt.Chart = (
        alt.Chart(position_df)
        .mark_point(shape="triangle", filled=True, fillOpacity=1)
        .encode(
            x=alt.X("start:Q"),
            y=alt.Y("bar_y:Q", axis=None),
            angle=alt.Angle("bar_direction:N", scale=alt.Scale(domain=["Positive", "Negative"], range=[90, -90])),
            color=alt.Color(
                "bar_direction:N", scale=alt.Scale(domain=["Positive", "Negative"], range=["#1f77b4", "#d62728"])
            ),
            size=alt.SizeValue(300),
            tooltip=alt.value(None),
        )
    )

    # Add a vertical line at the base value
    zero_line: alt.Chart = alt.Chart(pd.DataFrame({"x": [base_value]})).mark_rule(strokeDash=[3, 3]).encode(x="x:Q")

    # Calculate label positions to avoid overlap and ensure labels are spread apart horizontally

    # Sort by bar center (avg) for label placement
    sorted_positions = sorted(positions, key=lambda x: x["avg"])

    # Improved label spreading algorithm:
    # Calculate the minimum and maximum x positions (avg) for the bars
    min_x = min(pos["avg"] for pos in sorted_positions)
    max_x = max(pos["avg"] for pos in sorted_positions)
    n_labels = len(sorted_positions)
    # Calculate the minimum required distance between labels
    spread_width = max_x - min_x
    if n_labels > 1:
        space_per_label = spread_width / (n_labels - 1)
        # If space_per_label is less than min_distance, use min_distance instead
        effective_distance = max(space_per_label, MIN_DISTANCE)
    else:
        effective_distance = 0

    # Start from min_x - offset, and assign label_x for each label from left to right
    offset = -effective_distance  # Start a bit to the left
    label_positions = []
    label_lines = []
    placed_label_xs: list[float] = []
    for i, pos in enumerate(sorted_positions):
        if i == 0:
            label_x = min_x + offset
        else:
            label_x = placed_label_xs[-1] + effective_distance
        placed_label_xs.append(label_x)
        label_positions.append(
            {
                "label_x": label_x,
                "label_y": 1,  # Place labels below the bars
                "feature_annotated": pos["feature_annotated"],
                "feature_value": pos["feature_value"],
            }
        )
        # Draw a diagonal line from the bar to the label
        label_lines.append(
            {
                "x": pos["avg"],
                "x2": label_x,
                "y": 0,
                "y2": 1,
            }
        )

    label_positions_df = pd.DataFrame(label_positions)
    label_lines_df = pd.DataFrame(label_lines)

    # Draw diagonal lines from bar to label
    label_connectors = (
        alt.Chart(label_lines_df)
        .mark_rule(strokeDash=[2, 2], color="grey")
        .encode(
            x="x:Q",
            x2="x2:Q",
            y=alt.Y("y:Q", axis=None),
            y2="y2:Q",
        )
    )

    # Place labels at adjusted positions
    feature_labels = (
        alt.Chart(label_positions_df)
        .mark_text(align="center", baseline="line-bottom", dy=0, fontSize=11)
        .encode(
            x=alt.X("label_x:Q"),
            y=alt.Y("label_y:Q", axis=None),
            text=alt.Text("feature_annotated:N"),
            color=alt.value("grey"),
            tooltip=["feature_value"],
        )
    )

    return cast(alt.LayerChart, bars + feature_labels + zero_line + arrow + label_connectors)


def plot_influence_sensitivity(
    shap_values: type_hints.SupportedDataType,
    feature_values: type_hints.SupportedDataType,
    infer_is_categorical: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Any:
    """
    Create a SHAP dependence scatter plot for a specific feature. If a DataFrame is provided, a select box
    will be displayed to select the feature. This is only supported in Snowflake notebooks.
    If Streamlit is not available and a DataFrame is passed in, an ImportError will be raised.

    Args:
        shap_values: pandas Series or 2D array containing the SHAP values for a specific feature
        feature_values: pandas Series or 2D array containing the feature values for the same feature
        infer_is_categorical: If True, the function will infer if the feature is categorical
            based on the number of unique values.
        figsize: tuple of (width, height) for the plot

    Returns:
        Altair chart object

    Raises:
        ValueError: If the types of feature_values and shap_values are not the same

    """

    use_streamlit = False
    feature_values_df = _convert_to_pandas_df(feature_values)
    shap_values_df = _convert_to_pandas_df(shap_values)

    if len(shap_values_df.shape) > 1:
        feature_values, shap_values, st = _prepare_feature_values_for_streamlit(feature_values_df, shap_values_df)
        use_streamlit = True
    elif feature_values_df.shape[0] != shap_values_df.shape[0]:
        raise ValueError("Feature values and SHAP values must have the same number of rows.")

    scatter = _create_scatter_plot(feature_values, shap_values, infer_is_categorical, figsize)
    return st.altair_chart(scatter) if use_streamlit else scatter


def _prepare_feature_values_for_streamlit(
    feature_values_df: pd.DataFrame, shap_values: pd.DataFrame
) -> tuple[pd.Series, pd.Series, Any]:
    try:
        from IPython import get_ipython
        from snowbook.executor.python_transformer import IPythonProxy

        assert isinstance(
            get_ipython(), IPythonProxy
        ), "Influence sensitivity plots for a DataFrame are not supported outside of Snowflake notebooks."
    except ImportError:
        raise RuntimeError(
            "Influence sensitivity plots for a DataFrame are not supported outside of Snowflake notebooks."
        )

    import streamlit as st

    feature_columns = feature_values_df.columns
    chosen_ft: str = st.selectbox("Feature:", feature_columns)
    feature_values = feature_values_df[chosen_ft]
    shap_values = shap_values.iloc[:, feature_columns.get_loc(chosen_ft)]
    return feature_values, shap_values, st


def _create_scatter_plot(
    feature_values: pd.Series, shap_values: pd.Series, infer_is_categorical: bool, figsize: tuple[float, float]
) -> alt.Chart:
    unique_vals = np.sort(np.unique(feature_values.values))
    max_points_per_unique_value = float(np.max(np.bincount(np.searchsorted(unique_vals, feature_values.values))))
    points_per_value = len(feature_values.values) / len(unique_vals)
    is_categorical = float(max(max_points_per_unique_value, points_per_value)) > 10 if infer_is_categorical else False

    kwargs = (
        {
            "x": alt.X("feature_value:N", title="Feature Value"),
            "color": alt.Color("feature_value:N").legend(None),
            "xOffset": "jitter:Q",
        }
        if is_categorical
        else {"x": alt.X("feature_value:Q", title="Feature Value")}
    )

    # Create a dataframe for plotting
    plot_df = pd.DataFrame({"feature_value": feature_values, "shap_value": shap_values})

    width, height = figsize

    # Create scatter plot
    scatter = (
        alt.Chart(plot_df)
        .transform_calculate(jitter="random()")
        .mark_circle(size=60, opacity=0.7)
        .encode(
            y=alt.Y("shap_value:Q", title="SHAP Value"),
            tooltip=["feature_value", "shap_value"],
            **kwargs,
        )
        .properties(title="SHAP Dependence Scatter Plot", width=width, height=height)
    )

    return cast(alt.Chart, scatter)


def plot_violin(
    shap_df: type_hints.SupportedDataType,
    feature_df: type_hints.SupportedDataType,
    figsize: tuple[float, float] = DEFAULT_VIOLIN_FIGSIZE,
) -> alt.Chart:
    """
    Create a violin plot per feature showing the distribution of SHAP values.

    Args:
        shap_df: 2D array containing SHAP values for multiple features
        feature_df: 2D array containing the corresponding feature values
        figsize: tuple of (width, height) for the plot

    Returns:
        Altair chart object
    """

    shap_df_pd = _convert_to_pandas_df(shap_df)
    feature_df_pd = _convert_to_pandas_df(feature_df)

    # Assert that the input dataframes are 2D
    assert len(shap_df_pd.shape) == 2, f"shap_df must be 2D, but got shape {shap_df_pd.shape}"
    assert len(feature_df_pd.shape) == 2, f"feature_df must be 2D, but got shape {feature_df_pd.shape}"

    # Prepare data for plotting
    plot_data = pd.DataFrame(
        {
            "feature_name": feature_df_pd.columns.repeat(shap_df_pd.shape[0]),
            "shap_value": shap_df_pd.transpose().values.flatten(),
        }
    )

    # Order the rows by the absolute sum of SHAP values per feature
    feature_abs_sum = shap_df_pd.abs().sum(axis=0)
    sorted_features = feature_abs_sum.sort_values(ascending=False).index
    column_sort_order = [feature_df_pd.columns[shap_df_pd.columns.get_loc(col)] for col in sorted_features]

    # Create the violin plot
    width, height = figsize
    violin = (
        alt.Chart(plot_data)
        .transform_density(density="shap_value", groupby=["feature_name"], as_=["shap_value", "density"])
        .mark_area(orient="vertical")
        .encode(
            y=alt.Y("density:Q", title=None).stack("center").impute(None).axis(labels=False, grid=False, ticks=False),
            x=alt.X("shap_value:Q", title="SHAP Value"),
            row=alt.Row(
                "feature_name:N", sort=column_sort_order, header=alt.Header(labelAngle=0, labelAlign="left")
            ).spacing(0),
            color=alt.Color("feature_name:N", legend=None),
            tooltip=["feature_name", "shap_value"],
        )
        .properties(width=width, height=height)
    ).interactive()

    return cast(alt.Chart, violin)


def _convert_to_pandas_df(
    data: type_hints.SupportedDataType,
) -> pd.DataFrame:
    if isinstance(data, sp_df.DataFrame):
        return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(data)

    return model_signature._convert_local_data_to_df(data)
