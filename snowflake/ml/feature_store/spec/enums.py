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
    FeatureGroup = "FeatureGroup"


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


class FSBaseType(str, Enum):
    """Canonical feature store type vocabulary.

    Values mirror Snowpark Python SDK class names. These are the
    string-based types used in declarative authoring YAML, JSON,
    and compiled output. They are intentionally distinct from
    :class:`snowflake.snowpark.types.DataType` instances — the
    spec-internal models in :mod:`snowflake.ml.feature_store.spec.models`
    use Snowpark types, while the declarative authoring layer uses these
    string values so it stays installable in environments without the
    Snowpark SDK.
    """

    StringType = "StringType"
    LongType = "LongType"
    DoubleType = "DoubleType"
    DecimalType = "DecimalType"
    BooleanType = "BooleanType"
    BinaryType = "BinaryType"
    TimestampType = "TimestampType"


# Python-friendly aliases resolved to FSBaseType values during compilation.
TYPE_ALIASES: dict[str, str] = {
    "str": FSBaseType.StringType,
    "string": FSBaseType.StringType,
    "int": FSBaseType.LongType,
    "integer": FSBaseType.LongType,
    "long": FSBaseType.LongType,
    "float": FSBaseType.DoubleType,
    "double": FSBaseType.DoubleType,
    "number": FSBaseType.DoubleType,
    "decimal": FSBaseType.DecimalType,
    "bool": FSBaseType.BooleanType,
    "boolean": FSBaseType.BooleanType,
    "binary": FSBaseType.BinaryType,
    "bytes": FSBaseType.BinaryType,
    "datetime": FSBaseType.TimestampType,
    "timestamp": FSBaseType.TimestampType,
}


def normalize_type(raw_type: str) -> str:
    """Resolve a type string to its FSBaseType value.

    Accepts FSBaseType names directly (e.g. ``"StringType"``) or
    Python-friendly aliases (e.g. ``"str"``, ``"float"``). Unknown types
    are returned unchanged.

    Args:
        raw_type: A type string to normalize.

    Returns:
        The canonical FSBaseType value string, or ``raw_type`` if not
        recognized.
    """
    if raw_type in {t.value for t in FSBaseType}:
        return raw_type
    return TYPE_ALIASES.get(raw_type.lower(), raw_type)


ENTITY_TAG_PREFIX: str = "SNOWML_FEATURE_STORE_ENTITY_"
"""Canonical Snowflake tag-name prefix for entity registrations.

The imperative ``FeatureStore`` API persists every registered entity as
a Snowflake tag whose name is ``ENTITY_TAG_PREFIX + entity_name``.  The
declarative client mirrors that convention when parsing ``SHOW TAGS``
rows back into entity objects.

Single source of truth for both the imperative client (``feature_store``
module) and the declarative client (``decl.state``, ``decl.api``,
``decl.imperative_executor``, ``decl.exporter``).  Declarative modules
that need the prefix import this constant directly from ``spec.enums``;
``decl`` is permitted to import from ``spec.enums`` because the latter
is stdlib-only and carries no heavy transitive dependencies.
"""
