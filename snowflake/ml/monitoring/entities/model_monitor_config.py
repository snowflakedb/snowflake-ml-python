from dataclasses import dataclass

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

    prediction_score_columns: list[str] | None = None
    """List of columns in the source containing prediction scores.
    Can be regression scores for regression models and probability scores for classification models.
    """

    prediction_class_columns: list[str] | None = None
    """List of columns in the source containing prediction classes for classification models."""

    actual_score_columns: list[str] | None = None
    """List of columns in the source containing actual scores."""

    actual_class_columns: list[str] | None = None
    """List of columns in the source containing actual classes for classification models."""

    baseline: str | None = None
    """Name of table containing the baseline data."""

    segment_columns: list[str] | None = None
    """List of columns in the source containing segment information for grouped monitoring."""

    custom_metric_columns: list[str] | None = None
    """List of columns in the source containing custom metrics."""

    timestamp_custom_metric_table: str | None = None
    """Optional name of a table containing timestamp-based custom metrics.
    Can be specified unqualified or fully qualified as database.schema.table.
    """

    _ground_truth: str | None = None
    _inference_table_source: bool = False

    def __init__(
        self,
        source: str,
        timestamp_column: str,
        id_columns: list[str],
        prediction_score_columns: list[str] | None = None,
        prediction_class_columns: list[str] | None = None,
        actual_score_columns: list[str] | None = None,
        actual_class_columns: list[str] | None = None,
        baseline: str | None = None,
        segment_columns: list[str] | None = None,
        custom_metric_columns: list[str] | None = None,
        timestamp_custom_metric_table: str | None = None,
    ) -> None:
        self.source = source
        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        self.prediction_score_columns = prediction_score_columns
        self.prediction_class_columns = prediction_class_columns
        self.actual_score_columns = actual_score_columns
        self.actual_class_columns = actual_class_columns
        self.baseline = baseline
        self.segment_columns = segment_columns
        self.custom_metric_columns = custom_metric_columns
        self.timestamp_custom_metric_table = timestamp_custom_metric_table
        self._ground_truth = None
        self._inference_table_source = False

    @classmethod
    def _from_inference_table(
        cls,
        *,
        baseline: str | None = None,
        segment_columns: list[str] | None = None,
        ground_truth: str | None = None,
        id_columns: list[str] | None = None,
        prediction_score_columns: list[str] | None = None,
        prediction_class_columns: list[str] | None = None,
        actual_score_columns: list[str] | None = None,
        actual_class_columns: list[str] | None = None,
    ) -> "ModelMonitorSourceConfig":
        """Build a source config for the model inference table.

        This entry point is in development. ``ground_truth`` and ``id_columns`` are passed together. Prediction and
        actual columns are inferred when omitted.

        Args:
            baseline: Name of the baseline table.
            segment_columns: Columns used to segment monitoring.
            ground_truth: Name of the ground truth table. Pass this with ``id_columns``.
            id_columns: Columns that join the inference table to ``ground_truth``.
            prediction_score_columns: Prediction score columns. Inferred when omitted.
            prediction_class_columns: Prediction class columns. Inferred when omitted.
            actual_score_columns: Actual score columns. Inferred when omitted.
            actual_class_columns: Actual class columns. Inferred when omitted.

        Returns:
            A source config that monitors the model inference table.

        Raises:
            ValueError: If ``ground_truth`` and ``id_columns`` are not passed together.
        """
        config = cls.__new__(cls)
        config.source = None  # type: ignore[assignment]
        config.timestamp_column = None  # type: ignore[assignment]
        config.id_columns = id_columns  # type: ignore[assignment]
        config.prediction_score_columns = prediction_score_columns
        config.prediction_class_columns = prediction_class_columns
        config.actual_score_columns = actual_score_columns
        config.actual_class_columns = actual_class_columns
        config.baseline = baseline
        config.segment_columns = segment_columns
        config.custom_metric_columns = None
        config.timestamp_custom_metric_table = None
        config._ground_truth = ground_truth
        config._inference_table_source = True
        has_ground_truth = ground_truth is not None
        has_id_columns = bool(id_columns)
        if has_ground_truth != has_id_columns:
            raise ValueError("Pass ground_truth and id_columns together.")
        return config

    def _uses_inference_table(self) -> bool:
        """Return whether this config monitors the model inference table."""
        return self._inference_table_source


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
