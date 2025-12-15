from dataclasses import dataclass
from typing import Optional

from snowflake.ml.model._client.model import model_version_impl


@dataclass
class ModelMonitorSourceConfig:
    """Configuration for the source of data to be monitored."""

    source: str
    """Name of table or view containing monitoring data."""

    timestamp_column: str
    """Name of column in the source containing timestamp."""

    id_columns: list[str]
    """List of columns in the source containing unique identifiers."""

    prediction_score_columns: Optional[list[str]] = None
    """List of columns in the source containing prediction scores.
    Can be regression scores for regression models and probability scores for classification models."""

    prediction_class_columns: Optional[list[str]] = None
    """List of columns in the source containing prediction classes for classification models."""

    actual_score_columns: Optional[list[str]] = None
    """List of columns in the source containing actual scores."""

    actual_class_columns: Optional[list[str]] = None
    """List of columns in the source containing actual classes for classification models."""

    baseline: Optional[str] = None
    """Name of table containing the baseline data."""

    segment_columns: Optional[list[str]] = None
    """List of columns in the source containing segment information for grouped monitoring."""

    custom_metric_columns: Optional[list[str]] = None
    """List of columns in the source containing custom metrics."""


@dataclass
class ModelMonitorConfig:
    """Configuration for the Model Monitor."""

    model_version: model_version_impl.ModelVersion
    """Model version to monitor."""

    model_function_name: str
    """Function name in the model to monitor."""

    background_compute_warehouse_name: str
    """Name of the warehouse to use for background compute."""

    refresh_interval: str = "1 hour"
    """Interval at which to refresh the monitoring data."""

    aggregation_window: str = "1 day"
    """Window for aggregating monitoring data."""
