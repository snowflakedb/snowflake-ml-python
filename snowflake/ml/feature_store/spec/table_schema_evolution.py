"""Table schema evolution validation (Snowpark column types).

Used for any Snowflake table-like object described by Snowpark schemas (e.g. offline feature
storage). Uses :data:`snowflake.ml.feature_store.spec.models._SUPPORTED_TYPES` as the allowed type
set (same as FSColumn / ``validate_schema_types``).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from snowflake.ml.feature_store.spec.models import _SUPPORTED_TYPES
from snowflake.ml.feature_store.stream_config import _snowpark_type_to_sql
from snowflake.snowpark.types import DataType, StructType


def schema_upper_name_map(schema: StructType) -> dict[str, DataType]:
    """Map uppercased column names to Snowpark types for a ``StructType``."""
    return {f.name.upper(): f.datatype for f in schema.fields}


def _schema_dict_must_use_supported_types(which: str, schema: Mapping[str, DataType]) -> None:
    """Ensure every column type is in :data:`_SUPPORTED_TYPES`."""
    bad: list[tuple[str, str]] = []
    for col_name, dt in schema.items():
        if type(dt) not in _SUPPORTED_TYPES:
            bad.append((col_name, type(dt).__name__))
    if bad:
        details = ", ".join(f"'{n}' ({t})" for n, t in bad)
        allowed = sorted(t.__name__ for t in _SUPPORTED_TYPES)
        raise ValueError(f"{which} uses unsupported Snowpark types: {details}. Allowed: {allowed}")


def get_table_schema_evolution_extend_only_commands(
    old_table: str,
    old_schema: Mapping[str, DataType],
    new_schema: Mapping[str, DataType],
    *,
    required_old_columns: Optional[Iterable[str]] = None,
) -> tuple[list[str], list[str]]:
    """Compute extend-only DDL commands to evolve *old_table* from *old_schema* to *new_schema*.

    Extend-only means:

    - All columns in *old_schema* must appear in *new_schema* in the **same order** (no drops,
      no reorderings) and with **identical** Snowpark types (no type changes).
    - *new_schema* may declare additional columns **at the end**; these become ``ADD COLUMN``
      statements.

    Only types in :data:`_SUPPORTED_TYPES` may appear in either schema.

    Args:
        old_table: Table identifier used verbatim in the generated DDL (e.g. ``"DB.SCHEMA.T"`` or
            an unquoted name). The caller is responsible for any required quoting/qualification.
        old_schema: Existing column types in canonical column order (e.g. from
            :func:`schema_upper_name_map`).
        new_schema: Target column types in canonical column order.
        required_old_columns: Optional column names that must already be present in *old_schema*
            (compared case-insensitively). When supplied, this is checked **before** the
            structural prefix/type comparison so callers see a domain-friendly "missing required
            columns" error rather than the generic position-mismatch error that would otherwise
            fire when a required column is absent. Pass this when *old_schema* describes
            user-supplied data that must contain specific columns to be semantically valid (e.g.
            entity join keys and the timestamp column for an offline backfill table).

    Returns:
        ``(forward_commands, rollback_commands)``. ``forward_commands`` apply the evolution as
        ``ALTER TABLE ... ADD COLUMN`` statements for each newly appended column.
        ``rollback_commands`` reverse each forward command in LIFO order so applying them undoes
        the forward batch. Both lists are empty when the two schemas are identical.

    Raises:
        ValueError: Unsupported type in either schema, *required_old_columns* missing from
            *old_schema*, fewer columns in *new_schema* than *old_schema* (implied drop), an
            existing column appears at a different position in *new_schema*, or an existing
            column's type changed.
    """
    if required_old_columns is not None:
        old_keys_upper = {k.upper() for k in old_schema}
        missing = [c for c in required_old_columns if c.upper() not in old_keys_upper]
        if missing:
            raise ValueError(f"{old_table} is missing required columns: {', '.join(missing)}.")

    _schema_dict_must_use_supported_types("Existing table schema", old_schema)
    _schema_dict_must_use_supported_types("New source schema", new_schema)

    old_keys = list(old_schema.keys())
    new_keys = list(new_schema.keys())

    if len(new_keys) < len(old_keys):
        raise ValueError(
            f"new_schema has fewer columns ({len(new_keys)}) than old_schema ({len(old_keys)}); "
            "extend-only evolution cannot drop columns."
        )
    for idx, old_name in enumerate(old_keys):
        if new_keys[idx] != old_name:
            raise ValueError(
                f"Column at position {idx} mismatch: old='{old_name}', new='{new_keys[idx]}'. "
                "Extend-only evolution requires existing columns to remain in the same order."
            )

    forward: list[str] = []
    rollback: list[str] = []

    for col_name, old_type in old_schema.items():
        new_type = new_schema[col_name]
        if old_type != new_type:
            raise ValueError(
                f"Column '{col_name}' type changed from {old_type} to {new_type}; "
                "extend-only evolution does not support type changes."
            )

    for new_name in new_keys[len(old_keys) :]:
        new_type = new_schema[new_name]
        forward.append(f'ALTER TABLE {old_table} ADD COLUMN "{new_name}" {_snowpark_type_to_sql(new_type)}')
        rollback.append(f'ALTER TABLE {old_table} DROP COLUMN "{new_name}"')

    rollback.reverse()
    return forward, rollback
