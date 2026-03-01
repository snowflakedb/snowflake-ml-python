"""Unified Feature View Spec Generator.

Re-exports the public API for constructing validated JSON payloads for the
Go backend's unified FeatureView schema.
"""

from snowflake.ml.feature_store.spec.builder import (
    BatchSource,
    FeatureViewSpecBuilder,
    SnowflakeTableInfo,
)
from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    StoreType,
    TableType,
)
from snowflake.ml.feature_store.spec.models import FeatureViewSpec

__all__ = [
    "BatchSource",
    "FeatureViewSpec",
    "FeatureViewSpecBuilder",
    "SnowflakeTableInfo",
    "FeatureViewKind",
    "StoreType",
    "TableType",
    "FeatureAggregationMethod",
]
