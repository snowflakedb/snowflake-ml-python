"""Pydantic domain models for the unified Feature View spec schema.

These models mirror the Go backend's unified FeatureView struct. They are
spec-internal — constructed only inside the builder, never by external users.

Note: spec.Feature is distinct from feature_store.Feature (user-facing).

Uses Pydantic v2 (``BaseModel.model_dump()``).
"""

import json
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    SourceType,
    StoreType,
    TableType,
)
from snowflake.snowpark.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructType,
    TimestampTimeZone,
    TimestampType,
)

# Supported Snowpark types for FSColumn conversion — aligned with
# stream_source._TYPE_NAME_TO_CLASS.
_SUPPORTED_TYPES: set[type] = {
    StringType,
    LongType,
    DoubleType,
    DecimalType,
    BooleanType,
    BinaryType,
    TimestampType,
}

# ---------------------------------------------------------------------------
# Snowpark type → FSColumn conversion utilities
# ---------------------------------------------------------------------------


def validate_schema_types(schema: StructType) -> None:
    """Validate that all columns in a schema use supported Snowpark types.

    Reusable validation for any code path that builds a feature view spec
    (batch FV, streaming FV, etc.).

    Args:
        schema: A Snowpark StructType to validate.

    Raises:
        ValueError: If any column uses an unsupported type, listing all
            offending columns and the set of supported types.
    """
    unsupported = []
    for field in schema.fields:
        if type(field.datatype) not in _SUPPORTED_TYPES:
            unsupported.append((field.name, type(field.datatype).__name__))

    if unsupported:
        col_details = ", ".join(f"'{name}' ({typ})" for name, typ in unsupported)
        supported_names = sorted(t.__name__ for t in _SUPPORTED_TYPES)
        raise ValueError(f"Unsupported column types: {col_details}. " f"Supported types: {supported_names}")

    non_ntz = []
    for field in schema.fields:
        if isinstance(field.datatype, TimestampType):
            if field.datatype.tz not in (TimestampTimeZone.NTZ, TimestampTimeZone.DEFAULT):
                non_ntz.append((field.name, str(field.datatype.tz)))

    if non_ntz:
        col_details = ", ".join(f"'{name}' ({tz})" for name, tz in non_ntz)
        raise ValueError(
            f"Timestamp columns must be TIMESTAMP_NTZ: {col_details}. " f"Consider casting to TIMESTAMP_NTZ."
        )


def validate_spec_oft_offline_table_schema(schema: StructType) -> None:
    """Validate columns for a Snowflake table described in OFT ``offline_configs``.

    Same rules as :func:`validate_schema_types`: only types that serialize to
    :class:`FSColumn` in the unified FeatureView spec. Unsupported types (e.g.
    ARRAY, VARIANT) must fail before embedding the schema in
    ``CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION``.

    Use this for served, scalar offline tables (e.g. a non-tiled FV's output
    view). Tiled DTs that carry list-aggregation partial arrays use
    :func:`validate_spec_oft_tiled_offline_table_schema` instead.

    Args:
        schema: Snowpark StructType describing the offline table columns.
    """
    validate_schema_types(schema)


def validate_spec_oft_tiled_offline_table_schema(schema: StructType) -> None:
    """Validate a tiled FV's offline DT schema for OFT ``offline_configs``.

    A tiled DT legitimately carries list-aggregation partial columns stored as
    ``ARRAY`` (e.g. the distinct-N value/timestamp arrays). Those are permitted
    here; every other column is validated with the standard scalar rules via
    :func:`validate_schema_types`.

    Args:
        schema: Snowpark StructType describing the tiled offline DT columns.
    """
    scalar_fields = [field for field in schema.fields if not isinstance(field.datatype, ArrayType)]
    validate_schema_types(StructType(scalar_fields))


def _make_fs_column(name: str, dt: DataType) -> "FSColumn":
    """Convert a (name, Snowpark DataType) pair to an FSColumn.

    Uses ``type(dt).__name__`` directly for the type string, which aligns
    with the Go backend's FSBaseType names (e.g. ``LongType``, ``DoubleType``).

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
    if isinstance(dt, TimestampType) and dt.tz not in (TimestampTimeZone.NTZ, TimestampTimeZone.DEFAULT):
        raise ValueError(
            f"Timestamp column '{name}' must be TIMESTAMP_NTZ, got {dt.tz}. " f"Consider casting to TIMESTAMP_NTZ."
        )
    return FSColumn(
        name=name,
        type=type(dt).__name__,
        length=dt.length if isinstance(dt, StringType) and dt.length is not None else None,
        precision=dt.precision if isinstance(dt, DecimalType) else None,
        scale=dt.scale if isinstance(dt, DecimalType) else None,
    )


def _make_tiled_fs_column(name: str, dt: DataType) -> "FSColumn":
    """Convert a tiled-DT column to an FSColumn, permitting list-aggregation arrays.

    ``ArrayType`` columns are serialized as ``type="ArrayType"`` with
    ``element_type`` taken from the Snowpark ``ArrayType.element_type`` when
    available. Physical tile arrays built with ``ARRAY_AGG`` report no usable
    element type, so callers that need an exact one (e.g. distinct-N partials)
    set it explicitly from the aggregation spec after conversion. Non-array
    columns fall back to the scalar :func:`_make_fs_column` rules.

    Args:
        name: Column name.
        dt: Snowpark DataType instance.

    Returns:
        An FSColumn with type metadata extracted from *dt*.
    """
    if isinstance(dt, ArrayType):
        element = getattr(dt, "element_type", None)
        element_name = type(element).__name__ if element is not None else None
        return FSColumn(name=name, type="ArrayType", element_type=element_name)
    return _make_fs_column(name, dt)


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


def _columns_from_tiled_struct_type(schema: StructType) -> list["FSColumn"]:
    """Convert a tiled-DT StructType to FSColumns, permitting list-aggregation arrays.

    Like :func:`_columns_from_struct_type` but uses :func:`_make_tiled_fs_column`
    so that the list-aggregation partial columns stored as ``ARRAY`` (e.g.
    distinct-N value/timestamp arrays) are carried through instead of rejected.

    Args:
        schema: A Snowpark StructType schema describing a tiled offline DT.

    Returns:
        A list of FSColumn instances.
    """
    return [_make_tiled_fs_column(f.name, f.datatype) for f in schema.fields]


def _format_fs_column_type(col: "FSColumn") -> str:
    if col.type == "DecimalType" and col.precision is not None and col.scale is not None:
        return f"DecimalType({col.precision},{col.scale})"
    if col.type == "StringType" and col.length is not None:
        return f"StringType({col.length})"
    return col.type


def validate_fs_columns_match(
    *,
    expected: list["FSColumn"],
    actual: list["FSColumn"],
    expected_label: str,
    actual_label: str,
    error_prefix: str,
) -> None:
    """Validate that every column in ``expected`` is present in ``actual`` with a compatible type.

    Names match case-insensitively. Type compatibility is governed by
    :func:`_fs_columns_compatible`. Extra columns in ``actual`` are allowed.
    Reports the first offending column.

    Args:
        expected: Canonical column list (e.g. from a declared schema).
        actual: Column list to verify against ``expected``.
        expected_label: Label for the expected schema's owner in error messages
            (e.g. ``"StreamSource 'transaction_events'"``).
        actual_label: Label for the actual schema's owner in error messages
            (e.g. ``"backfill_df"``).
        error_prefix: Domain prefix for the error message (e.g.
            ``"streaming feature view"``).

    Raises:
        ValueError: If any expected column is missing from ``actual`` or has a
            type that is not compatible.
    """
    actual_by_name = {c.name.upper(): c for c in actual}
    for exp in expected:
        act = actual_by_name.get(exp.name.upper())
        if act is None:
            raise ValueError(
                f"{error_prefix}: {actual_label} is missing column '{exp.name}' " f"declared by {expected_label}."
            )
        if not _fs_columns_compatible(expected=exp, actual=act):
            raise ValueError(
                f"{error_prefix}: {actual_label} column '{act.name}' has type "
                f"{_format_fs_column_type(act)} but {expected_label} declares "
                f"column '{exp.name}' with type {_format_fs_column_type(exp)}."
            )


def _fs_columns_compatible(*, expected: "FSColumn", actual: "FSColumn") -> bool:
    """Return True when ``actual`` is type-compatible with ``expected``.

    ``type``, ``timezone``, and ``element_type`` always match exactly.
    ``StringType`` length on ``expected`` is a wildcard when ``None``: Snowpark
    can report a specific length even for unbounded ``VARCHAR`` columns, so a
    schema declared with ``StringType()`` must accept those.
    ``DecimalType`` precision/scale must match exactly — different scales would
    truncate data silently.

    Args:
        expected: Declared FSColumn from the canonical schema.
        actual: FSColumn observed in the schema being validated.

    Returns:
        True if the two columns are compatible per the rules above.
    """
    if expected.type != actual.type:
        return False
    if expected.timezone != actual.timezone:
        return False
    if expected.element_type != actual.element_type:
        return False
    if expected.type == "StringType":
        return expected.length is None or expected.length == actual.length
    if expected.type == "DecimalType":
        return expected.precision == actual.precision and expected.scale == actual.scale
    return True


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
# Domain Models (Pydantic v2)
# ---------------------------------------------------------------------------


class FSColumn(BaseModel):
    """A single column with name and serialized Snowpark type info."""

    name: str
    type: str  # e.g. "StringType", "DecimalType", "ArrayType" — from type(dt).__name__
    length: Optional[int] = None  # StringType
    precision: Optional[int] = None  # DecimalType
    scale: Optional[int] = None  # DecimalType
    timezone: Optional[str] = None  # TimestampType
    element_type: Optional[str] = None


class Source(BaseModel):
    """A data source for a feature view.

    For ``FEATURES`` sources (references to upstream feature views), ``columns``
    contains exactly the subset of the upstream FV's exposed columns that this
    consumer wants, in the consumer's desired order. The full upstream FV
    schema lives in the upstream FV's own spec on the server and is not
    repeated here.
    """

    name: str
    source_type: SourceType
    columns: list[FSColumn]
    source_version: Optional[str] = None


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

    ``source_name`` / ``source_version`` identify the upstream FV when the
    feature comes from a ``FEATURES`` source, and are ``None`` otherwise.
    Required for ``FeatureGroup`` (whose spec omits the parallel ``sources``
    block); ``None`` for all other kinds today.
    """

    source_column: FSColumn
    output_column: FSColumn
    function: Optional[str] = None
    window_sec: Optional[int] = None
    offset_sec: Optional[int] = None
    function_params: Optional[dict[str, Any]] = None
    source_name: Optional[str] = None
    source_version: Optional[str] = None


class OfflineTableConfig(BaseModel):
    """Offline storage configuration for a feature view table."""

    # populate_by_name: allows construction via Python name (schema_="value")
    # while serializing as the alias ("schema") with by_alias=True.
    model_config = ConfigDict(populate_by_name=True)

    store_type: StoreType
    table_type: TableType
    database: str
    schema_: str = Field(..., alias="schema")
    table: str
    columns: list[FSColumn]


class Metadata(BaseModel):
    """Feature view metadata — identity and versioning."""

    model_config = ConfigDict(populate_by_name=True)

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
    ordered_secondary_key_column_names: Optional[list[str]] = None
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
        return self.model_dump(exclude_none=True, by_alias=True)

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
        raw = self.model_dump_json(exclude_none=True, by_alias=True)
        return _sanitize_json_for_dollar_quoting(raw)

    def to_yaml(self) -> str:
        """Serialize to a YAML string with ``omitempty`` and alias resolution.

        Returns:
            A YAML string matching the unified FeatureView schema.
        """
        # Round-trip through JSON to coerce enums / custom types to primitives.
        plain = json.loads(self.to_json())
        return yaml.dump(plain, default_flow_style=False, sort_keys=False)
