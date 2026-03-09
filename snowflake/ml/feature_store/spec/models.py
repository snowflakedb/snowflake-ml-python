"""Pydantic domain models for the unified Feature View spec schema.

These models mirror the Go backend's unified FeatureView struct. They are
spec-internal — constructed only inside the builder, never by external users.

Note: spec.Feature is distinct from feature_store.Feature (user-facing).

Uses Pydantic v1 (``BaseModel.dict()``).
"""

import json
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    SourceType,
    StoreType,
    TableType,
)
from snowflake.snowpark.types import (
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    FloatType,
    StringType,
    StructType,
    TimestampTimeZone,
    TimestampType,
    TimeType,
)

# Supported Snowpark types for FSColumn conversion — aligned with
# stream_source._TYPE_NAME_TO_CLASS.
_SUPPORTED_TYPES: set[type] = {
    StringType,
    FloatType,
    DecimalType,
    BooleanType,
    TimestampType,
    DateType,
    TimeType,
}

# ---------------------------------------------------------------------------
# Snowpark type → FSColumn conversion utilities
# ---------------------------------------------------------------------------


def _make_fs_column(name: str, dt: DataType) -> "FSColumn":
    """Convert a (name, Snowpark DataType) pair to an FSColumn.

    Follows the same pattern as stream_source._schema_to_dict — uses
    type(dt).__name__ directly, no manual mapping needed.

    Args:
        name: Column name.
        dt: Snowpark DataType instance.

    Returns:
        An FSColumn with type metadata extracted from *dt*.

    Raises:
        ValueError: If the DataType is not in the supported set.
    """
    if type(dt) not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported type '{type(dt).__name__}' for column '{name}'. "
            f"Supported types: {sorted(t.__name__ for t in _SUPPORTED_TYPES)}"
        )
    return FSColumn(
        name=name,
        type=type(dt).__name__,
        length=dt.length if isinstance(dt, StringType) and dt.length is not None else None,
        precision=dt.precision if isinstance(dt, DecimalType) else None,
        scale=dt.scale if isinstance(dt, DecimalType) else None,
        timezone=(
            str(dt.tz)
            if isinstance(dt, TimestampType)
            and dt.tz is not None
            and dt.tz not in (TimestampTimeZone.NTZ, TimestampTimeZone.DEFAULT)
            else None
        ),
    )


def _columns_from_struct_type(schema: StructType) -> list["FSColumn"]:
    """Convert a Snowpark StructType to a list of FSColumns.

    Utility for internal callers building OfflineTableConfig or Source from
    an existing schema (e.g., StreamSource.schema).

    Args:
        schema: A Snowpark StructType schema.

    Returns:
        A list of FSColumn instances.
    """
    return [_make_fs_column(f.name, f.datatype) for f in schema.fields]


# ---------------------------------------------------------------------------
# Dollar-quoting sanitization
# ---------------------------------------------------------------------------


def _sanitize_json_for_dollar_quoting(payload: str) -> str:
    """Replace ``$$`` with ``$\\u0024`` in a JSON string.

    Makes the string safe for embedding inside SQL ``$$...$$`` delimiters.
    ``\\u0024`` is a standard JSON unicode escape for ``$``; every conformant
    JSON parser decodes it back to ``$`` automatically.

    Args:
        payload: A JSON-encoded string (output of ``json.dumps`` / Pydantic ``.json()``).

    Returns:
        The same JSON string with every ``$$`` replaced by ``$\\u0024``.
    """
    return payload.replace("$$", "$\\u0024")


# ---------------------------------------------------------------------------
# Domain Models (Pydantic v1)
# ---------------------------------------------------------------------------


class FSColumn(BaseModel):
    """A single column with name and serialized Snowpark type info."""

    name: str
    type: str  # e.g. "StringType", "DecimalType" — from type(dt).__name__
    length: Optional[int] = None  # StringType
    precision: Optional[int] = None  # DecimalType
    scale: Optional[int] = None  # DecimalType
    timezone: Optional[str] = None  # TimestampType


class Source(BaseModel):
    """A data source for a feature view."""

    name: str
    source_type: SourceType
    columns: list[FSColumn]
    source_version: Optional[str] = None
    selected_features: Optional[list[str]] = None


class UDF(BaseModel):
    """A UDF transform definition.

    The ``function_definition`` is stored as plain text.  Safety for SQL
    ``$$`` quoting is handled at serialization time by
    :meth:`FeatureViewSpec.to_json`, which replaces ``$$`` with the JSON
    unicode escape ``$\\u0024`` — decoded back to ``$`` automatically by
    every standard JSON parser.
    """

    name: str
    engine: str
    output_columns: list[FSColumn]
    function_definition: str  # plain-text UDF source code


class Feature(BaseModel):
    """Spec-internal Feature model — not the user-facing Feature class.

    Represents a single feature derivation from a source column to an
    output column, optionally with an aggregation function and window.
    """

    source_column: FSColumn
    output_column: FSColumn
    function: Optional[str] = None
    window_sec: Optional[int] = None
    offset_sec: Optional[int] = None
    function_params: Optional[dict[str, Any]] = None


class OfflineTableConfig(BaseModel):
    """Offline storage configuration for a feature view table."""

    class Config:
        # allow_population_by_field_name: allows construction via Python name
        # (schema_="value") while serializing as the alias ("schema") with
        # by_alias=True.
        allow_population_by_field_name = True

    store_type: StoreType
    table_type: TableType
    database: str
    schema_: str = Field(..., alias="schema")
    table: str
    columns: list[FSColumn]


class Metadata(BaseModel):
    """Feature view metadata — identity and versioning."""

    class Config:
        allow_population_by_field_name = True

    database: str
    schema_: str = Field(..., alias="schema")
    name: str
    version: str
    spec_format_version: str
    internal_data_version: str
    client_version: str


class Spec(BaseModel):
    """The core specification describing sources, features, and transformations."""

    ordered_entity_column_names: list[str]
    sources: list[Source]
    features: list[Feature]
    timestamp_field: Optional[str] = None
    feature_granularity_sec: Optional[int] = None
    feature_aggregation_method: Optional[FeatureAggregationMethod] = None
    udf: Optional[UDF] = None
    target_lag_sec: Optional[int] = None


class FeatureViewSpec(BaseModel):
    """Root model — the complete unified payload for the Go backend."""

    kind: FeatureViewKind
    metadata: Metadata
    offline_configs: list[OfflineTableConfig]
    spec: Spec
    online_store_type: Optional[StoreType] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict with ``omitempty`` and alias resolution.

        Returns:
            A dictionary matching the unified FeatureView JSON schema.
        """
        return self.dict(exclude_none=True, by_alias=True)  # type: ignore[deprecation]

    def to_json(self) -> str:
        """Serialize to a JSON string with ``omitempty`` and alias resolution.

        The output is safe for embedding in SQL ``$$...$$`` delimiters:
        any ``$$`` sequences in the JSON (e.g. from UDF source code) are
        replaced with ``$\\u0024``, which every standard JSON parser
        decodes back to ``$`` automatically.

        Returns:
            A JSON string matching the unified FeatureView JSON schema,
            safe for SQL ``$$`` quoting.
        """
        raw = self.json(exclude_none=True, by_alias=True)  # type: ignore[deprecation]
        return _sanitize_json_for_dollar_quoting(raw)

    def to_yaml(self) -> str:
        """Serialize to a YAML string with ``omitempty`` and alias resolution.

        Returns:
            A YAML string matching the unified FeatureView schema.
        """
        # Round-trip through JSON to coerce enums / custom types to primitives.
        plain = json.loads(self.to_json())
        return yaml.dump(plain, default_flow_style=False, sort_keys=False)
