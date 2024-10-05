from dataclasses import dataclass
from typing import List

from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring.entities import model_monitor_interval


@dataclass
class ModelMonitorTableConfig:
    source_table: str
    timestamp_column: str
    prediction_columns: List[str]
    label_columns: List[str]
    id_columns: List[str]


@dataclass
class ModelMonitorConfig:
    model_version: model_version_impl.ModelVersion

    # Python model function name
    model_function_name: str
    background_compute_warehouse_name: str
    # TODO: Add support for pythonic notion of time.
    refresh_interval: str = model_monitor_interval.ModelMonitorRefreshInterval.DAILY
    aggregation_window: model_monitor_interval.ModelMonitorAggregationWindow = (
        model_monitor_interval.ModelMonitorAggregationWindow.WINDOW_1_DAY
    )
