"""Enum definitions for the unified Feature View spec schema.

These enums lock down string literal choices to prevent typos and ensure
type safety when constructing the JSON payload for the Go backend.
All enums inherit (str, Enum) so they serialize naturally to their string
values in JSON.
"""

from enum import Enum


class FeatureViewKind(str, Enum):
    """The kind of feature view being defined."""

    StreamingFeatureView = "StreamingFeatureView"
    RealtimeFeatureView = "RealtimeFeatureView"
    BatchFeatureView = "BatchFeatureView"


class StoreType(str, Enum):
    """Backend storage type for offline/online tables."""

    SNOWFLAKE = "snowflake"
    POSTGRES = "postgres"


class TableType(str, Enum):
    """The type of offline table configuration."""

    UDF_TRANSFORMED = "UDFTransformed"
    TILED = "Tiled"
    BATCH_SOURCE = "BatchSource"


class SourceType(str, Enum):
    """The type of data source for a feature view."""

    STREAM = "Stream"
    REQUEST = "Request"
    FEATURES = "Features"
    BATCH = "Batch"


class FeatureAggregationMethod(str, Enum):
    """The aggregation method for feature computation."""

    TILES = "tiles"
    CONTINUOUS = "continuous"
