from dataclasses import dataclass
from typing import List, Optional

from snowflake.ml.model._client.model import model_version_impl


@dataclass
class ModelMonitorSourceConfig:
    source: str
    timestamp_column: str
    id_columns: List[str]
    prediction_score_columns: Optional[List[str]] = None
    prediction_class_columns: Optional[List[str]] = None
    actual_score_columns: Optional[List[str]] = None
    actual_class_columns: Optional[List[str]] = None
    baseline: Optional[str] = None


@dataclass
class ModelMonitorConfig:
    model_version: model_version_impl.ModelVersion

    # Python model function name
    model_function_name: str
    background_compute_warehouse_name: str
    # TODO: Add support for pythonic notion of time.
    refresh_interval: str = "1 hour"
    aggregation_window: str = "1 day"
