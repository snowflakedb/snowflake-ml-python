from dataclasses import dataclass

from snowflake.ml.model._client.model import model_version_impl


@dataclass
class ModelMonitorSourceConfig:
    """Configuration for the source of data to be monitored.

    Pass ``source`` and ``timestamp_column`` to monitor a user table. Omit both to monitor the model inference table.
    ``ground_truth`` is only valid when monitoring the model inference table, and must be passed with ``id_columns``.
    """

    source: str | None = None
    """Name of table or view containing monitoring data. Omit this to monitor the model inference table."""

    timestamp_column: str | None = None
    """Name of column in the source containing timestamp. Required with ``source``. Omit this for the model inference
    table.
    """

    id_columns: list[str] | None = None
    """List of columns containing unique identifiers. Required with ``source``. Pass with ``ground_truth``."""

    prediction_score_columns: list[str] | None = None
    """List of columns in the source containing prediction scores.
    Can be regression scores for regression models and probability scores for classification models.
    Inferred when omitted for the model inference table.
    """

    prediction_class_columns: list[str] | None = None
    """List of columns in the source containing prediction classes for classification models.
    Inferred when omitted for the model inference table.
    """

    actual_score_columns: list[str] | None = None
    """List of columns in the source containing actual scores. Inferred when omitted for the model inference table."""

    actual_class_columns: list[str] | None = None
    """List of columns in the source containing actual classes for classification models.
    Inferred when omitted for the model inference table.
    """

    baseline: str | None = None
    """Name of table containing the baseline data."""

    segment_columns: list[str] | None = None
    """List of columns in the source containing segment information for grouped monitoring."""

    custom_metric_columns: list[str] | None = None
    """List of columns in the source containing custom metrics. Not accepted for the model inference table."""

    timestamp_custom_metric_table: str | None = None
    """Optional name of a table containing timestamp-based custom metrics.
    Can be specified unqualified or fully qualified as database.schema.table.
    Not accepted for the model inference table.
    """

    ground_truth: str | None = None
    """Name of the ground truth table. Model inference table monitors only. Pass this with ``id_columns``."""

    def __post_init__(self) -> None:
        if self._uses_inference_table():
            self._validate_inference_table()
        else:
            self._validate_user_table()

    def _uses_inference_table(self) -> bool:
        """Return whether this config monitors the model inference table."""
        return self.source is None

    def _validate_user_table(self) -> None:
        if self.timestamp_column is None:
            raise ValueError("A user-table monitor requires a timestamp column.")
        if self.id_columns is None:
            raise ValueError("A user-table monitor requires id columns.")
        if self.ground_truth is not None:
            raise ValueError("Ground truth is only available for a model inference table monitor.")

    def _validate_inference_table(self) -> None:
        if self.timestamp_column is not None:
            raise ValueError("A model inference table monitor does not take a timestamp column.")
        if self.custom_metric_columns is not None:
            raise ValueError("A model inference table monitor does not accept custom metric columns.")
        if self.timestamp_custom_metric_table is not None:
            raise ValueError("A model inference table monitor does not accept a timestamp custom metric table.")
        has_ground_truth = self.ground_truth is not None
        has_id_columns = bool(self.id_columns)
        if has_ground_truth != has_id_columns:
            raise ValueError("Pass ground_truth and id_columns together.")


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
    """Window for aggregating monitoring data. A model version monitor accepts a number of days only."""
