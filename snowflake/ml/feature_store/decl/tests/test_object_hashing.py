"""Canonical hash table for each feature-store object kind.

This file is the single source of truth for the question "for kind X, what is
the canonical hash basis, and which edits MUST or MUST NOT change the hash".
Existing hash tests in :mod:`test_invariants`, :mod:`test_invariants_feature_group`,
:mod:`test_export_plan_round_trip`, and :mod:`test_golden_spec_round_trip`
continue to cover validator-level concerns (VERSION_CONFLICT, COLUMN_ADDED,
dependency resolution) and end-to-end round-trip pipelines; this file focuses
narrowly on the hash function contract for each object kind.

Three layers of assertion per kind:

1. **Authoring ↔ compiled parity** — the local-compile hash matches the
   hash of the raw authoring dict (or, for Entity / Source / FeatureGroup
   kinds where there is no separate compile step, matches the Pydantic-
   normalised model dump).
2. **Compiled ↔ applied parity** — the local-compile hash matches the
   hash of the applied-state ``spec_payload`` that
   :func:`fetch_applied_state` would emit for the same logical object.
3. **Edit sensitivity** — every documented MUST-bump / MUST-NOT-bump edit
   is exercised once.

Hash routing (matches the planner's strategy in :mod:`planner` /
:func:`_check_idempotency`):

- ``Entity``                       → :func:`structural_fingerprint_hash`
- ``BatchSource`` / ``StreamingSource`` → :func:`structural_fingerprint_hash`
- ``BatchFeatureView`` /
  ``StreamingFeatureView`` /
  ``RealtimeFeatureView``          → :func:`_full_spec_hash`
                                     (``compute_local_spec_hash`` on the
                                     authoring side, which is just
                                     ``_full_spec_hash(compile_to_spec(...))``)
- ``FeatureGroup``                 → :func:`fg_content_hash`

Important nuance for Source kinds. ``_structural_fingerprint`` only folds
``sources``/``join_keys``/``description`` into the basis for FV kinds
and for ``Entity`` — for ``BatchSource`` / ``StreamingSource`` the
fingerprint reduces to ``{name, version, columns_from_features}``. A
``BatchSource``'s ``table`` and ``columns`` therefore do NOT contribute
to the Source hash; their drift surfaces (a) through the dependent FV's
``_full_spec_hash`` (the FV embeds the binding) and (b) through the
column-evolution / source-compatibility validators. The tests below
document this contract explicitly so the surprise lives in one place.

Reinforced by a planner-side derivation rule.  Since sources are virtual,
the planner cannot rely on the Source hash alone to decide ``NO_CHANGE``:
the deployed runtime never persists a ``Datasource`` object whose name
matches the authored ``BatchSource.name`` (snowml-core's
``DESCRIBE … TYPE = SPECIFICATION`` drops ``spec.sources[]``, so
``state._inject_batch_fv_source_from_dt_text`` reconstructs the applied
``Datasource`` using the underlying table name or a synthetic
``<FV>__SOURCE`` for query-backed sources).  ``planner._sources_with_deployed_fv``
therefore overrides the ``applied is None`` branch for source kinds: when
≥1 referencing local FV has an applied counterpart, the source emits
``NO_CHANGE`` regardless of the Source hash.  The hash contract documented
above is preserved verbatim — the override is an upstream routing layer,
not a hash-function change — and the contract is exercised end-to-end by
``test_planner_virtual_sources.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    _normalise_fv_sources_for_hash,
    compute_local_spec_hash,
    fg_content_hash,
    model_to_dict,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureGroup,
    FeatureView,
    SpecBase,
    StreamingSource,
)

_DB = "DB"
_SCHEMA = "SCH"

_STRUCT_KINDS = frozenset({"Entity", "BatchSource", "StreamingSource"})
_FV_KINDS = frozenset({"StreamingFeatureView", "BatchFeatureView", "RealtimeFeatureView"})
_FG_KIND = "FeatureGroup"

_MODEL_FOR_KIND: dict[str, type[SpecBase]] = {
    "Entity": Entity,
    "BatchSource": BatchSource,
    "StreamingSource": StreamingSource,
    "StreamingFeatureView": FeatureView,
    "BatchFeatureView": FeatureView,
    "RealtimeFeatureView": FeatureView,
    "FeatureGroup": FeatureGroup,
}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def _hash_for_kind(kind: str, payload: dict[str, Any]) -> str:
    """Compute the canonical hash for *payload* using the kind-appropriate function.

    Args:
        kind: One of the seven object kinds (Entity / BatchSource /
            StreamingSource / StreamingFeatureView / BatchFeatureView /
            RealtimeFeatureView / FeatureGroup).
        payload: A spec dict in either the local-compile shape or the
            applied DESCRIBE shape — both must hash identically for a
            clean round-trip.

    Returns:
        The 64-character lowercase hex SHA-256 digest produced by the
        kind's canonical hash function.

    Raises:
        ValueError: When *kind* is not one of the seven supported kinds.
    """
    if kind in _FV_KINDS:
        return _full_spec_hash(payload)
    if kind == _FG_KIND:
        return fg_content_hash(payload)
    if kind in _STRUCT_KINDS:
        return structural_fingerprint_hash(payload)
    raise ValueError(f"unknown kind: {kind!r}")


def _hash_authoring(
    kind: str,
    authoring: dict[str, Any],
    *,
    db: str = _DB,
    schema: str = _SCHEMA,
) -> str:
    """Hash the authoring-side dict the same way the planner does.

    For FV kinds this composes :func:`compile_to_spec` with
    :func:`_full_spec_hash` (i.e. exactly what
    :func:`compute_local_spec_hash` does). For Entity / Source / FG
    kinds there is no compile step — the dict is hashed directly after
    the schema_ → schema canonicalisation.

    Args:
        kind: One of the seven object kinds.
        authoring: The YAML-shape dict the loader would produce.
        db: Connection-context database used during the FV compile step.
            Ignored for non-FV kinds.
        schema: Connection-context schema used during the FV compile step.
            Ignored for non-FV kinds.

    Returns:
        The hex digest of the authoring side.

    Raises:
        ValueError: When *kind* is not one of the seven supported kinds.
    """
    if kind in _FV_KINDS:
        compile_input = _strip_schema_alias(authoring)
        return compute_local_spec_hash(compile_input, db, schema)
    if kind in _STRUCT_KINDS or kind == _FG_KIND:
        return _hash_for_kind(kind, _strip_schema_alias(authoring))
    raise ValueError(f"unknown kind: {kind!r}")


def _compile_for_kind(
    kind: str,
    authoring: dict[str, Any],
    *,
    db: str = _DB,
    schema: str = _SCHEMA,
) -> dict[str, Any]:
    """Produce the compiled-shape dict the planner would feed to hash routing.

    Args:
        kind: One of the seven object kinds.
        authoring: The YAML-shape dict the loader would produce.
        db: Connection-context database used during the FV compile step.
        schema: Connection-context schema used during the FV compile step.

    Returns:
        The compiled spec dict — for FV kinds this is the output of
        :func:`compile_to_spec`; for Entity / Source / FG kinds it is
        the Pydantic-validated dict with ``schema_`` renamed to
        ``schema`` (i.e. what :func:`model_to_dict` produces).
    """
    if kind in _FV_KINDS:
        return compile_to_spec(_strip_schema_alias(authoring), db, schema)
    model_cls = _MODEL_FOR_KIND[kind]
    return model_to_dict(model_cls.model_validate(authoring))


def _strip_schema_alias(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *spec* with ``schema_`` renamed to ``schema``.

    The authoring shape uses ``schema_`` (Pydantic-friendly) while the
    compile/hash side expects ``schema``. Callers that pass either form
    must work, so we normalise up front.

    Args:
        spec: The spec dict to normalise.

    Returns:
        A shallow copy of *spec* with ``schema_`` renamed to ``schema``
        (the original is left untouched).
    """
    out = dict(spec)
    if "schema_" in out and "schema" not in out:
        out["schema"] = out.pop("schema_")
    elif "schema_" in out:
        out.pop("schema_")
    return out


def _edit(authoring: dict[str, Any], edit_fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Deep-copy *authoring* and apply *edit_fn* in place; return the result."""
    new = copy.deepcopy(authoring)
    edit_fn(new)
    return new


# ---------------------------------------------------------------------------
# Per-kind fixtures
# ---------------------------------------------------------------------------
#
# Each fixture returns ``(authoring, applied)``:
#   - authoring is the YAML-shape dict (model_validate compatible).
#   - applied is the dict shape ``fetch_applied_state`` emits for the same
#     logical object — i.e. what :class:`AppliedObject.spec_payload`
#     would carry after a clean apply + DESCRIBE round-trip.
#
# Fixtures are kept small and deliberately representative of the
# corresponding ``declarative_feature_store/example_store/`` YAMLs so a
# reader can map them back to a live example.


def _entity_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a basic Entity (matches ``example_store/sources/entities/USER_ID.yaml``)."""
    authoring: dict[str, Any] = {
        "kind": "Entity",
        "name": "USER_ID",
        "database": _DB,
        "schema_": _SCHEMA,
        "description": "Shared user identifier.",
        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
    }
    applied: dict[str, Any] = {
        "kind": "Entity",
        "name": "USER_ID",
        "database": _DB,
        "schema": _SCHEMA,
        "description": "Shared user identifier.",
        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
    }
    return authoring, applied


def _streaming_source_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a REST StreamingSource (mirrors ``CLICKSTREAM_EVENTS.yaml``)."""
    columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "SESSION_ID", "type": "StringType"},
        {"name": "EVENT_TYPE", "type": "StringType"},
        {"name": "TIMESTAMP", "type": "TimestampType"},
    ]
    authoring: dict[str, Any] = {
        "kind": "StreamingSource",
        "name": "CLICKSTREAM_EVENTS",
        "database": _DB,
        "schema_": _SCHEMA,
        "type": "REST",
        "columns": columns,
    }
    # Applied state surfaces sources under the canonical ``Datasource`` kind
    # (``state._datasource_objects_from_specs``). ``_structural_fingerprint``
    # does not key off ``kind`` for Source kinds, so both shapes hash equal.
    applied: dict[str, Any] = {
        "kind": "Datasource",
        "name": "CLICKSTREAM_EVENTS",
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": "Stream",
        "columns": columns,
    }
    return authoring, applied


def _batch_source_table_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a table-backed BatchSource (mirrors ``EVENTS_BATCH_DECL.yaml``)."""
    columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "EVENT_TS", "type": "TimestampType"},
        {"name": "METRIC_VAL", "type": "FloatType"},
    ]
    authoring: dict[str, Any] = {
        "kind": "BatchSource",
        "name": "EVENTS_BATCH_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "table": "RAW_EVENTS_BATCH_DECL",
        "columns": columns,
    }
    applied: dict[str, Any] = {
        "kind": "Datasource",
        "name": "EVENTS_BATCH_DECL",
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": "Batch",
        "table": "RAW_EVENTS_BATCH_DECL",
        "columns": columns,
    }
    return authoring, applied


def _batch_source_query_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a query-backed BatchSource (post sidecar inlining)."""
    columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "EVENT_TS", "type": "TimestampType"},
        {"name": "METRIC_VAL", "type": "FloatType"},
    ]
    authoring: dict[str, Any] = {
        "kind": "BatchSource",
        "name": "EVENTS_SQL_BATCH_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "query": "SELECT user_id, event_ts, metric_val FROM raw.events",
        "columns": columns,
    }
    applied: dict[str, Any] = {
        "kind": "Datasource",
        "name": "EVENTS_SQL_BATCH_DECL",
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": "Batch",
        "query": "SELECT user_id, event_ts, metric_val FROM raw.events",
        "columns": columns,
    }
    return authoring, applied


def _streaming_fv_with_udf_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a StreamingFeatureView with a pandas UDF.

    Mirrors the BUG_BASH streaming shape (with feature aggregation windows
    and a UDF). The applied side is exactly what
    ``DESCRIBE … TYPE = SPECIFICATION`` returns for this FV after a clean
    apply: ``metadata`` + ``offline_configs`` + ``spec`` + ``online_store_type``.

    Returns:
        ``(authoring, applied)`` dicts that must hash identically via
        :func:`compute_local_spec_hash` and :func:`_full_spec_hash`.
    """
    udf_body = "def transform(df):\n    return df['EVENT'].count()"
    authoring: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "USER_CLICK_STATS",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "entities": ["USER_ID"],
        "timestamp_col": "TIMESTAMP",
        "feature_granularity_sec": 300,
        "feature_aggregation_method": "tiles",
        "sources": [
            {
                "name": "CLICKSTREAM_EVENTS",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT", "type": "StringType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "EVENT", "type": "StringType"},
                "output_column": {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                "function": "count",
                "window_sec": 3600,
            }
        ],
        "udf": {
            "name": "transform",
            "engine": "pandas",
            "function_definition": udf_body,
            "output_columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
            ],
        },
    }
    applied: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_CLICK_STATS",
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "1.38.0",  # volatile — stripped by _full_spec_hash
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "UDFTransformed",
                "database": _DB,
                "schema": _SCHEMA,
                "table": "USER_CLICK_STATS$V1$UDF_TRANSFORMED",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "EVENT", "type": "StringType"},
                    "output_column": {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                    "function": "count",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "tiles",
            "udf": {
                "name": "transform",
                "engine": "pandas",
                "function_definition": udf_body,
                "output_columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            },
            "target_lag_sec": 0,  # runtime-stamped; stripped by _full_spec_hash
        },
        "online_store_type": "postgres",
    }
    return authoring, applied


def _streaming_fv_with_backfill_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a StreamingFV that adds an FV-level ``backfill:`` block.

    The backfill block is operational metadata (it routes through
    ``StreamConfig`` at registration time) and must be stripped from
    :func:`_full_spec_hash` — otherwise a backfill-less DESCRIBE payload
    cannot round-trip against a backfill-authored YAML.

    Returns:
        A ``(authoring, applied)`` tuple where ``authoring`` is the
        Pydantic-shaped dict the loader produces and ``applied`` is the
        DESCRIBE-shaped payload the planner compares against.
    """
    authoring, applied = _streaming_fv_with_udf_fixture()
    authoring = copy.deepcopy(authoring)
    authoring["name"] = "USER_CLICK_BACKFILL_DECL"
    authoring["backfill"] = {
        "table": f"{_DB}.{_SCHEMA}.RAW_CLICK_HISTORY",
        "start_time": "2020-01-01T00:00:00Z",
    }
    applied = copy.deepcopy(applied)
    applied["metadata"]["name"] = "USER_CLICK_BACKFILL_DECL"
    applied["offline_configs"][0]["table"] = "USER_CLICK_BACKFILL_DECL$V1$UDF_TRANSFORMED"
    return authoring, applied


def _batch_fv_table_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for the BUG_BASH §5 BatchFV shape (table-backed source).

    The applied side carries the lossy SPECIFICATION shape:
    ``sources[0]`` has ``table`` recovered via DT-text injection, and
    ``features`` carries auto-derived 1:1 pass-throughs that
    :func:`_full_spec_hash` strips away.

    Returns:
        A ``(authoring, applied)`` tuple where ``authoring`` is the
        Pydantic-shaped dict the loader produces and ``applied`` is the
        DESCRIBE-shaped payload the planner compares against.
    """
    authoring: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "MY_BATCH_FV_BATCH_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "target_lag": "1 minute",
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
        "features": [],
        "refresh_freq": "1 minute",
    }
    applied: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "RAW_EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL",
                },
            ],
            "features": [
                # Snowflake auto-derives a 1:1 entry per source column on
                # CREATE; both halves get normalised away by
                # _full_spec_hash's _is_auto_derived_feature pass.
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 60,
        },
        "online_store_type": "postgres",
    }
    return authoring, applied


def _batch_fv_query_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a query-backed BatchFV (Phase 4 DT-text recovery path)."""
    query = "SELECT user_id, event_ts, metric_val FROM raw.events WHERE metric_val > 0"
    authoring: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "MY_SQL_BATCH_FV_BATCH_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "target_lag": "5 minutes",
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_SQL_BATCH_DECL",
                "source_type": "Batch",
                "query": query,
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
        "features": [],
        "refresh_freq": "5 minutes",
    }
    applied: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_SQL_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    # The Phase 4 DT-text recovery path synthesises this
                    # name when the deployed body is a query — both halves
                    # of the hash normalise to ``{"binding": "QUERY:<body>"}``.
                    "name": "MY_SQL_BATCH_FV_BATCH_DECL__SOURCE",
                    "source_type": "Batch",
                    "query": query,
                },
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 300,
        },
        "online_store_type": "postgres",
    }
    return authoring, applied


def _batch_fv_advanced_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for the all-six-advanced-fields BatchFV (ADVANCED_BVT_BUGBASH §4).

    Covers ``warehouse`` / ``cluster_by`` / ``refresh_mode`` / ``initialize``
    / ``aggregation_secondary_keys`` (and an explicit aggregation feature
    that prevents the auto-derive normalisation from kicking in).
    ``storage_config`` deliberately omitted — see the
    :class:`TestEditSensitivity` row that flips it to iceberg and asserts
    a hash bump.

    Returns:
        A ``(authoring, applied)`` tuple where ``authoring`` is the
        Pydantic-shaped dict the loader produces and ``applied`` is the
        DESCRIBE-shaped payload the planner compares against.
    """
    authoring: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "MY_ADV_BFV_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_ADV_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_ADV_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "SESSION_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "FloatType"},
                ],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "feature_granularity": "1h",
        "feature_aggregation_method": "tiles",
        "refresh_freq": "5 minutes",
        "warehouse": "MY_WAREHOUSE",
        "cluster_by": ["USER_ID"],
        "refresh_mode": "INCREMENTAL",
        "initialize": "ON_CREATE",
        "aggregation_secondary_keys": ["SESSION_ID"],
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window": "1h",
            }
        ],
    }
    applied: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_ADV_BFV_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "RAW_EVENTS_ADV_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_ADV_DECL",
                }
            ],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "target_lag_sec": 300,
            "cluster_by": ["USER_ID"],
            "refresh_mode": "INCREMENTAL",
            "initialize": "ON_CREATE",
            "aggregation_secondary_keys": ["SESSION_ID"],
        },
    }
    return authoring, applied


def _realtime_fv_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a RealtimeFeatureView."""
    authoring: dict[str, Any] = {
        "kind": "RealtimeFeatureView",
        "name": "USER_REQUEST_FV",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "REQUEST_PAYLOAD",
                "source_type": "Request",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "REQ_SCORE", "type": "DoubleType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "REQ_SCORE", "type": "DoubleType"},
                "output_column": {"name": "REQ_SCORE", "type": "DoubleType"},
            }
        ],
    }
    applied: dict[str, Any] = {
        "kind": "RealtimeFeatureView",
        "metadata": {
            "name": "USER_REQUEST_FV",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "REQUEST_PAYLOAD",
                    "source_type": "Request",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "REQ_SCORE", "type": "DoubleType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "REQ_SCORE", "type": "DoubleType"},
                    "output_column": {"name": "REQ_SCORE", "type": "DoubleType"},
                }
            ],
        },
        "online_store_type": "postgres",
    }
    return authoring, applied


def _feature_group_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixture for a Postgres-backed FeatureGroup."""
    authoring: dict[str, Any] = {
        "kind": "FeatureGroup",
        "name": "USER_FRAUD_FG_DECL",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "desc": "User-level fraud-signal feature group.",
        "auto_prefix": True,
        "feature_views": [
            {"name": "USER_CLICKS_FG_DECL", "version": "V1"},
            {"name": "USER_AMOUNTS_FG_DECL", "version": "V1"},
        ],
    }
    applied: dict[str, Any] = {
        "kind": "FeatureGroup",
        "name": "USER_FRAUD_FG_DECL",
        "database": _DB,
        "schema": _SCHEMA,
        "version": "V1",
        "desc": "User-level fraud-signal feature group.",
        "auto_prefix": True,
        # Applied side may emit sources in any order; the FG hash sorts.
        "feature_views": [
            {"name": "USER_AMOUNTS_FG_DECL", "version": "V1"},
            {"name": "USER_CLICKS_FG_DECL", "version": "V1"},
        ],
    }
    return authoring, applied


# Parametrise key for the three-shape parity table. Each entry is the
# ``label`` + a callable returning the ``(authoring, applied)`` fixture
# tuple. Labels are used as the pytest test id so a failure points
# straight at the kind variant.
_KIND_FIXTURES: list[tuple[str, str, Callable[[], tuple[dict[str, Any], dict[str, Any]]]]] = [
    ("Entity", "Entity", _entity_fixture),
    ("StreamingSource", "StreamingSource", _streaming_source_fixture),
    ("BatchSource_table", "BatchSource", _batch_source_table_fixture),
    ("BatchSource_query", "BatchSource", _batch_source_query_fixture),
    ("StreamingFeatureView_udf", "StreamingFeatureView", _streaming_fv_with_udf_fixture),
    ("StreamingFeatureView_backfill", "StreamingFeatureView", _streaming_fv_with_backfill_fixture),
    ("BatchFeatureView_table", "BatchFeatureView", _batch_fv_table_fixture),
    ("BatchFeatureView_query", "BatchFeatureView", _batch_fv_query_fixture),
    ("BatchFeatureView_advanced", "BatchFeatureView", _batch_fv_advanced_fixture),
    ("RealtimeFeatureView", "RealtimeFeatureView", _realtime_fv_fixture),
    ("FeatureGroup", "FeatureGroup", _feature_group_fixture),
]


# ---------------------------------------------------------------------------
# Three-shape parity
# ---------------------------------------------------------------------------


class TestThreeShapeParity:
    """For every kind variant, authoring ↔ compiled ↔ applied all hash equal.

    This pins the canonical hash basis. If a fixture fails here, the
    planner will emit a phantom CREATE / RECREATE op on a clean
    round-trip and the operator will see "all green, then dirty plan"
    until the normalisation gap is closed in
    :mod:`snowflake.ml.feature_store.decl.invariants`.
    """

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn"),
        _KIND_FIXTURES,
        ids=[label for label, _, _ in _KIND_FIXTURES],
    )
    def test_authoring_hash_matches_compiled_hash(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """The authoring-side hash equals the hash of the compiled dict.

        For FV kinds this is the ``compute_local_spec_hash`` ↔
        ``_full_spec_hash(compile_to_spec(...))`` equivalence (true by
        construction, but the test pins it so a refactor that splits
        the helpers cannot drift). For Entity / Source / FG kinds this
        pins the schema_ → schema rename + ``exclude_none=True`` model
        dump that :func:`model_to_dict` performs.

        Args:
            label: Parametrised fixture id used in the failure message.
            kind: Object-kind discriminator routed through
                :func:`_compile_for_kind` / :func:`_hash_for_kind`.
            fixture_fn: Zero-arg callable returning the
                ``(authoring, applied)`` fixture pair for this row.
        """
        authoring, _applied = fixture_fn()
        compiled = _compile_for_kind(kind, authoring)
        assert _hash_authoring(kind, authoring) == _hash_for_kind(kind, compiled), (
            f"{label}: authoring hash != compiled hash. Either the model "
            f"validation or the compile step is mutating a hash-bearing field."
        )

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn"),
        _KIND_FIXTURES,
        ids=[label for label, _, _ in _KIND_FIXTURES],
    )
    def test_compiled_hash_matches_applied_hash(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """The compiled-side hash equals the applied (DESCRIBE-shape) hash.

        This is the load-bearing assertion: it pins that the planner's
        local-compile-vs-applied diff lands as ``NO_CHANGE`` for every
        kind variant after a clean apply + DESCRIBE recovery. A failure
        here means the normalisation in :func:`_full_spec_hash` /
        :func:`_structural_fingerprint` / :func:`fg_content_hash` does
        not symmetrise the kind's authoring shape against its applied
        shape.

        Args:
            label: Parametrised fixture id used in the failure message.
            kind: Object-kind discriminator routed through
                :func:`_compile_for_kind` / :func:`_hash_for_kind`.
            fixture_fn: Zero-arg callable returning the
                ``(authoring, applied)`` fixture pair for this row.
        """
        authoring, applied = fixture_fn()
        compiled = _compile_for_kind(kind, authoring)
        assert _hash_for_kind(kind, compiled) == _hash_for_kind(kind, applied), (
            f"{label}: compiled hash != applied hash. The hash function for "
            f"this kind is not symmetric across the apply + DESCRIBE recovery."
        )

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn"),
        _KIND_FIXTURES,
        ids=[label for label, _, _ in _KIND_FIXTURES],
    )
    def test_authoring_hash_matches_applied_hash(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Transitive closure of the two prior asserts.

        Stated explicitly so a future refactor that bypasses the
        compile step (e.g. a direct authoring → applied diff path)
        still has the canonical equality as a single assertion.

        Args:
            label: Parametrised fixture id used in the failure message.
            kind: Object-kind discriminator routed through
                :func:`_hash_authoring` / :func:`_hash_for_kind`.
            fixture_fn: Zero-arg callable returning the
                ``(authoring, applied)`` fixture pair for this row.
        """
        authoring, applied = fixture_fn()
        assert _hash_authoring(kind, authoring) == _hash_for_kind(
            kind, applied
        ), f"{label}: authoring hash != applied hash on a clean round-trip."


# ---------------------------------------------------------------------------
# Edit sensitivity
# ---------------------------------------------------------------------------
#
# Each row is ``(label, kind, fixture_fn, edit_fn, expect_change, reason)``.
# ``label`` is the pytest id. ``edit_fn`` mutates the authoring dict
# in place. ``expect_change`` is True when the edit MUST bump the hash
# and False when it MUST NOT.
#
# Coverage strategy:
#
# * For each kind variant, exercise the documented MUST-bump and
#   MUST-NOT-bump cases.
# * For Source kinds, document the surprising contract — the hash basis
#   for ``_structural_fingerprint`` collapses to ``{name, version, []}``
#   for non-Entity/non-FV kinds, so ``columns`` / ``table`` / ``query``
#   edits do NOT bump the Source hash. Their drift is caught by
#   (a) the dependent FV's ``_full_spec_hash`` (which carries the
#   binding) and (b) the column-evolution / source-compatibility
#   validators. Test rows below pin this with an explicit
#   ``expect_change=False`` reason.


def _entity_set_description(d: dict[str, Any]) -> None:
    d["description"] = "edited description"


def _entity_change_join_key_name(d: dict[str, Any]) -> None:
    d["join_keys"][0]["name"] = "USER_ID_V2"


def _entity_change_join_key_type(d: dict[str, Any]) -> None:
    d["join_keys"][0]["type"] = "IntegerType"


def _source_change_name(d: dict[str, Any]) -> None:
    d["name"] = d["name"] + "_V2"


def _source_swap_table(d: dict[str, Any]) -> None:
    d["table"] = "RAW_EVENTS_V2"


def _source_reorder_columns(d: dict[str, Any]) -> None:
    d["columns"] = list(reversed(d["columns"]))


def _source_swap_query(d: dict[str, Any]) -> None:
    d["query"] = "SELECT user_id FROM raw.events_v2"


def _source_whitespace_query(d: dict[str, Any]) -> None:
    # Add a redundant tab + double-space + trailing whitespace; the
    # Source fingerprint does not normalise this, but it ALSO doesn't
    # carry ``query`` in its basis, so the hash is unchanged regardless.
    d["query"] = "  SELECT\tuser_id, event_ts, metric_val\n  FROM raw.events  "


def _fv_change_version(d: dict[str, Any]) -> None:
    d["version"] = "V2"


def _fv_change_udf_body(d: dict[str, Any]) -> None:
    d["udf"]["function_definition"] = "def transform(df):\n    return df['EVENT'].nunique()"


def _fv_swap_source_table(d: dict[str, Any]) -> None:
    d["sources"][0]["table"] = "RAW_EVENTS_V2"


def _fv_add_explicit_aggregation_feature(d: dict[str, Any]) -> None:
    d["features"].append(
        {
            "source_column": {"name": "METRIC_VAL", "type": "FloatType"},
            "output_column": {"name": "METRIC_VAL_AVG_1H", "type": "DoubleType"},
            "function": "avg",
            "window_sec": 3600,
        }
    )


def _fv_add_passthrough_feature(d: dict[str, Any]) -> None:
    # 1:1 source_column == output_column — _is_auto_derived_feature strips.
    d["features"].append(
        {
            "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
            "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
        }
    )


def _fv_add_backfill_block(d: dict[str, Any]) -> None:
    d["backfill"] = {
        "table": f"{_DB}.{_SCHEMA}.RAW_HISTORY",
        "start_time": "2020-01-01T00:00:00Z",
    }


def _fv_change_warehouse(d: dict[str, Any]) -> None:
    d["warehouse"] = "ANOTHER_WAREHOUSE"


def _fv_promote_narrow_to_wide_type(d: dict[str, Any]) -> None:
    # The local-compile path keeps narrow types (IntegerType/FloatType);
    # snowml-core widens them to LongType/DoubleType on the deployed side.
    # _full_spec_hash normalises both sides to the widened form.
    for feat in d.get("features", []):
        for col_key in ("source_column", "output_column"):
            col = feat.get(col_key)
            if isinstance(col, dict) and col.get("type") == "FloatType":
                col["type"] = "DoubleType"


def _fv_change_initialize(d: dict[str, Any]) -> None:
    d["initialize"] = "ON_SCHEDULE"


def _fv_change_refresh_mode(d: dict[str, Any]) -> None:
    d["refresh_mode"] = "FULL"


def _fv_change_cluster_by(d: dict[str, Any]) -> None:
    d["cluster_by"] = ["USER_ID", "SESSION_ID"]


def _fv_change_secondary_keys(d: dict[str, Any]) -> None:
    d["aggregation_secondary_keys"] = []


def _fv_change_storage_to_iceberg(d: dict[str, Any]) -> None:
    d["storage_config"] = {
        "format": "iceberg",
        "external_volume": "MY_EXTERNAL_VOL",
        "base_location": "fv/path",
    }


def _fv_strip_feature_aggregation_method(d: dict[str, Any]) -> None:
    d.pop("feature_aggregation_method", None)


def _fg_change_desc(d: dict[str, Any]) -> None:
    d["desc"] = "edited description"


def _fg_flip_auto_prefix(d: dict[str, Any]) -> None:
    d["auto_prefix"] = not d["auto_prefix"]


def _fg_bump_fv_version(d: dict[str, Any]) -> None:
    d["feature_views"][0]["version"] = "V2"


def _fg_set_slice_columns(d: dict[str, Any]) -> None:
    d["feature_views"][0]["slice_columns"] = ["AMOUNT_SUM_1H"]


def _fg_set_alias_empty_vs_none(d: dict[str, Any]) -> None:
    # ``alias = ""`` is preserved as a distinct value from None — the
    # FG hash basis pins this distinction.
    d["feature_views"][0]["alias"] = ""


def _fg_reorder_sources(d: dict[str, Any]) -> None:
    d["feature_views"] = list(reversed(d["feature_views"]))


def _fg_inject_output_columns(d: dict[str, Any]) -> None:
    # Derived field — ``fg_content_hash`` must ignore it.
    d["output_columns"] = ["A", "B", "C"]


_EDIT_ROWS: list[
    tuple[str, str, Callable[[], tuple[dict[str, Any], dict[str, Any]]], Callable[[dict[str, Any]], None], bool, str]
] = [
    # --- Entity ----------------------------------------------------------
    (
        "Entity:description_change",
        "Entity",
        _entity_fixture,
        _entity_set_description,
        True,
        "Entity description folds into the structural fingerprint via the Entity-aware branch.",
    ),
    (
        "Entity:join_key_name_change",
        "Entity",
        _entity_fixture,
        _entity_change_join_key_name,
        True,
        "Join-key name is part of the Entity-aware structural fingerprint.",
    ),
    (
        "Entity:join_key_type_change",
        "Entity",
        _entity_fixture,
        _entity_change_join_key_type,
        True,
        "Join-key type is part of the Entity-aware structural fingerprint.",
    ),
    # --- BatchSource (table-backed) -------------------------------------
    (
        "BatchSource_table:rename",
        "BatchSource",
        _batch_source_table_fixture,
        _source_change_name,
        True,
        "Name is the only stable identity for Source kinds — a rename is a new logical object.",
    ),
    (
        "BatchSource_table:swap_table_invisible",
        "BatchSource",
        _batch_source_table_fixture,
        _source_swap_table,
        False,
        (
            "Source fingerprint reduces to {name, version, []} — the table swap "
            "is caught by the dependent FV's _full_spec_hash (which carries the "
            "binding) and by source-compatibility validators, NOT by the Source's "
            "own hash. Documenting this contract here so the surprise lives in "
            "one place."
        ),
    ),
    (
        "BatchSource_table:reorder_columns_invisible",
        "BatchSource",
        _batch_source_table_fixture,
        _source_reorder_columns,
        False,
        (
            "Same as the table swap — column ordering is not in the Source "
            "fingerprint; column-add/remove flows through _check_source_compatibility."
        ),
    ),
    # --- BatchSource (query-backed) -------------------------------------
    (
        "BatchSource_query:rename",
        "BatchSource",
        _batch_source_query_fixture,
        _source_change_name,
        True,
        "Name is the only stable identity for Source kinds.",
    ),
    (
        "BatchSource_query:swap_query_invisible",
        "BatchSource",
        _batch_source_query_fixture,
        _source_swap_query,
        False,
        (
            "Source fingerprint reduces to {name, version, []} — the query body "
            "is caught by the dependent FV's _full_spec_hash through the "
            "_normalise_fv_sources_for_hash projection (which projects to "
            "{'binding': 'QUERY:<body>'} for query-backed sources)."
        ),
    ),
    (
        "BatchSource_query:whitespace_in_query_invisible",
        "BatchSource",
        _batch_source_query_fixture,
        _source_whitespace_query,
        False,
        "Source fingerprint does not carry the query body at all.",
    ),
    # --- StreamingSource ------------------------------------------------
    (
        "StreamingSource:rename",
        "StreamingSource",
        _streaming_source_fixture,
        _source_change_name,
        True,
        "Name is the only stable identity for Source kinds.",
    ),
    (
        "StreamingSource:reorder_columns_invisible",
        "StreamingSource",
        _streaming_source_fixture,
        _source_reorder_columns,
        False,
        "Source fingerprint ignores column ordering.",
    ),
    # --- StreamingFV (with UDF) -----------------------------------------
    (
        "StreamingFeatureView_udf:version_change",
        "StreamingFeatureView",
        _streaming_fv_with_udf_fixture,
        _fv_change_version,
        True,
        "Version is part of the FV full-spec hash basis (metadata.version).",
    ),
    (
        "StreamingFeatureView_udf:udf_body_change",
        "StreamingFeatureView",
        _streaming_fv_with_udf_fixture,
        _fv_change_udf_body,
        True,
        "UDF function_definition is part of the spec.udf block in the hash.",
    ),
    (
        "StreamingFeatureView_udf:backfill_block_invisible",
        "StreamingFeatureView",
        _streaming_fv_with_udf_fixture,
        _fv_add_backfill_block,
        False,
        (
            "FV-level backfill is operational metadata that routes through "
            "StreamConfig; _OPERATIONAL_FV_KEYS strips it from _full_spec_hash."
        ),
    ),
    # --- StreamingFV (with backfill) ------------------------------------
    (
        "StreamingFeatureView_backfill:backfill_table_invisible",
        "StreamingFeatureView",
        _streaming_fv_with_backfill_fixture,
        lambda d: d["backfill"].update({"table": f"{_DB}.{_SCHEMA}.RAW_HISTORY_V2"}),
        False,
        "Backfill block is stripped from the structural hash entirely.",
    ),
    # --- BatchFV (table-backed) -----------------------------------------
    (
        "BatchFeatureView_table:source_table_swap",
        "BatchFeatureView",
        _batch_fv_table_fixture,
        _fv_swap_source_table,
        True,
        (
            "Source-table swap MUST surface as a structural diff so the planner "
            "emits RECREATE_FV. _normalise_fv_sources_for_hash projects to "
            "{'binding': <TABLE>} and a table change bumps the hash."
        ),
    ),
    (
        "BatchFeatureView_table:explicit_aggregation_feature_added",
        "BatchFeatureView",
        _batch_fv_table_fixture,
        _fv_add_explicit_aggregation_feature,
        True,
        "Explicit aggregations (function/window) survive _is_auto_derived_feature.",
    ),
    (
        "BatchFeatureView_table:passthrough_feature_invisible",
        "BatchFeatureView",
        _batch_fv_table_fixture,
        _fv_add_passthrough_feature,
        False,
        (
            "1:1 source_column == output_column is auto-derived by snowml-core "
            "on CREATE; _is_auto_derived_feature strips both sides so a local "
            "YAML with the explicit pass-through hashes equal to one without."
        ),
    ),
    (
        "BatchFeatureView_table:warehouse_change_invisible",
        "BatchFeatureView",
        _batch_fv_table_fixture,
        _fv_change_warehouse,
        False,
        (
            "Warehouse is operational, not structural. The planner routes "
            "warehouse-only edits to UPDATE_FV through _batch_fv_operational_drift; "
            "_OPERATIONAL_FV_KEYS strips it from the hash."
        ),
    ),
    (
        "BatchFeatureView_table:narrow_to_wide_type_invisible",
        "BatchFeatureView",
        _batch_fv_table_fixture,
        _fv_promote_narrow_to_wide_type,
        False,
        ("_BFV_TYPE_PROMOTION widens IntegerType→LongType and FloatType→DoubleType " "on both halves of the hash."),
    ),
    # --- BatchFV (query-backed) -----------------------------------------
    (
        "BatchFeatureView_query:version_change",
        "BatchFeatureView",
        _batch_fv_query_fixture,
        _fv_change_version,
        True,
        "Version is part of the metadata block.",
    ),
    (
        "BatchFeatureView_query:passthrough_feature_invisible",
        "BatchFeatureView",
        _batch_fv_query_fixture,
        _fv_add_passthrough_feature,
        False,
        "1:1 features are auto-derived and stripped from both sides.",
    ),
    # --- BatchFV (advanced fields) --------------------------------------
    (
        "BatchFeatureView_advanced:cluster_by_change",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_cluster_by,
        True,
        (
            "A non-default cluster_by surfaces as a structural diff. The "
            "default (entity columns) is stripped by _strip_default_cluster_by "
            "on the applied side, but an explicit non-default list is preserved."
        ),
    ),
    (
        "BatchFeatureView_advanced:refresh_mode_change",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_refresh_mode,
        True,
        "refresh_mode != INCREMENTAL bumps the hash; both INCREMENTAL and FULL are "
        "explicit authored strategies preserved in the hash.",
    ),
    (
        "BatchFeatureView_advanced:initialize_change",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_initialize,
        True,
        "initialize != ON_CREATE bumps the hash; ON_CREATE is the default and stripped.",
    ),
    (
        "BatchFeatureView_advanced:secondary_keys_change",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_secondary_keys,
        True,
        "aggregation_secondary_keys is structural; clearing it bumps the hash.",
    ),
    (
        "BatchFeatureView_advanced:storage_iceberg",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_storage_to_iceberg,
        True,
        "Iceberg storage_config is preserved in the hash; only the trivial " "default {format: snowflake} is stripped.",
    ),
    (
        "BatchFeatureView_advanced:warehouse_change_invisible",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_change_warehouse,
        False,
        "Warehouse change routes to UPDATE_FV — not in the hash.",
    ),
    (
        "BatchFeatureView_advanced:strip_feature_aggregation_method_invisible",
        "BatchFeatureView",
        _batch_fv_advanced_fixture,
        _fv_strip_feature_aggregation_method,
        False,
        (
            "feature_aggregation_method is a decl-side authoring marker that "
            "the executor strips before forwarding to snowml-core's BFV "
            "constructor, so the deployed SPECIFICATION never carries it; "
            "_full_spec_hash strips it from BatchFeatureView inner specs."
        ),
    ),
    # --- RealtimeFV ------------------------------------------------------
    (
        "RealtimeFeatureView:version_change",
        "RealtimeFeatureView",
        _realtime_fv_fixture,
        _fv_change_version,
        True,
        "Version is part of the metadata block.",
    ),
    # --- FeatureGroup ----------------------------------------------------
    (
        "FeatureGroup:desc_change",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_change_desc,
        True,
        "desc is in the FG content-hash basis.",
    ),
    (
        "FeatureGroup:auto_prefix_flip",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_flip_auto_prefix,
        True,
        "auto_prefix is in the FG content-hash basis.",
    ),
    (
        "FeatureGroup:fv_version_bump",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_bump_fv_version,
        True,
        "Each source FV version is in the FG content-hash basis.",
    ),
    (
        "FeatureGroup:slice_columns_set",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_set_slice_columns,
        True,
        "slice_columns is in the FG basis (None → list flip is significant).",
    ),
    (
        "FeatureGroup:alias_empty_vs_none",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_set_alias_empty_vs_none,
        True,
        "alias='' is preserved as a distinct value from None — pins the "
        "'no prefix' vs 'auto_prefix' semantics in the basis.",
    ),
    (
        "FeatureGroup:reorder_sources_invisible",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_reorder_sources,
        False,
        "FG basis sorts sources by (fv_name, fv_version) so the YAML order does not matter.",
    ),
    (
        "FeatureGroup:output_columns_derived_invisible",
        "FeatureGroup",
        _feature_group_fixture,
        _fg_inject_output_columns,
        False,
        "output_columns is derived from the source FVs and explicitly excluded " "from the FG basis.",
    ),
]


class TestEditSensitivity:
    """Each MUST-bump / MUST-NOT-bump edit is exercised exactly once.

    The edit-sensitivity table is the canonical documentation for which
    fields contribute to each kind's hash. When an edit row fails:

    * ``expect_change=True`` but hashes equal → the normalisation
      drops or collapses a field that should be structurally
      significant. Look at the corresponding key in
      :data:`_VOLATILE_METADATA_KEYS` / :data:`_OPERATIONAL_FV_KEYS` /
      etc. — the field may have been overcollapsed.
    * ``expect_change=False`` but hashes differ → the normalisation
      is missing a stripping step. Add the field to the relevant
      keys set in :mod:`invariants` and re-run.
    """

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn", "edit_fn", "expect_change", "reason"),
        _EDIT_ROWS,
        ids=[row[0] for row in _EDIT_ROWS],
    )
    def test_edit(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
        edit_fn: Callable[[dict[str, Any]], None],
        expect_change: bool,
        reason: str,
    ) -> None:
        """Pin the documented edit sensitivity for *label*.

        Args:
            label: pytest id for the row.
            kind: Object kind for routing.
            fixture_fn: Returns the baseline (authoring, applied) tuple.
            edit_fn: In-place mutation applied to the authoring fixture.
            expect_change: ``True`` when the edit MUST bump the hash.
            reason: Human-readable explanation of why this row is in the
                table — surfaced in the failure message so the operator
                does not have to grep for the architectural intent.
        """
        baseline_authoring, _baseline_applied = fixture_fn()
        edited_authoring = _edit(baseline_authoring, edit_fn)

        baseline_hash = _hash_authoring(kind, baseline_authoring)
        edited_hash = _hash_authoring(kind, edited_authoring)

        if expect_change:
            assert baseline_hash != edited_hash, (
                f"{label}: hash did NOT change but the edit was expected to "
                f"bump it.\nReason: {reason}\n"
                f"Both sides hashed to {baseline_hash}; consider whether the "
                "normalisation in invariants.py is over-collapsing this field."
            )
        else:
            assert baseline_hash == edited_hash, (
                f"{label}: hash CHANGED but the edit must not bump it.\n"
                f"Reason: {reason}\n"
                f"Baseline hash: {baseline_hash}\n"
                f"Edited hash:   {edited_hash}\n"
                "Consider whether invariants.py needs to strip this field "
                "from the structural hash basis."
            )


# ---------------------------------------------------------------------------
# Volatile-metadata edits exercised on the COMPILED shape
# ---------------------------------------------------------------------------
#
# The authoring shape has no top-level ``metadata`` block — those keys
# (``client_version`` / ``oft_id`` / ``spec_format_version`` /
# ``internal_data_version``) only appear in the SPECIFICATION JSON
# Snowflake stamps onto the deployed FV. To pin that ``_full_spec_hash``
# strips them we mutate the compiled-shape dict directly.


class TestVolatileMetadataStripped:
    """``_VOLATILE_METADATA_KEYS`` are removed before hashing the compiled FV.

    Snowflake stamps a fresh ``client_version`` / ``oft_id`` /
    ``spec_format_version`` / ``internal_data_version`` onto every
    ``DESCRIBE … TYPE = SPECIFICATION`` payload at CREATE time. A
    tool-version bump or a re-create must NOT bump the structural
    hash — otherwise every operator upgrade triggers a recreate storm
    on the next ``snow feature plan``.
    """

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn"),
        [
            ("StreamingFeatureView_udf", "StreamingFeatureView", _streaming_fv_with_udf_fixture),
            ("BatchFeatureView_table", "BatchFeatureView", _batch_fv_table_fixture),
            ("BatchFeatureView_advanced", "BatchFeatureView", _batch_fv_advanced_fixture),
            ("RealtimeFeatureView", "RealtimeFeatureView", _realtime_fv_fixture),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_volatile_metadata_edits_do_not_change_hash(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Bumping volatile metadata on the applied side must not change the hash."""
        _authoring, applied = fixture_fn()
        baseline = _hash_for_kind(kind, applied)

        bumped = copy.deepcopy(applied)
        metadata = bumped.setdefault("metadata", {})
        metadata["client_version"] = "9.99.99-DEV"
        metadata["oft_id"] = "999999999"
        metadata["spec_format_version"] = "99"
        metadata["internal_data_version"] = "99"
        edited = _hash_for_kind(kind, bumped)

        assert baseline == edited, (
            f"{label}: bumping _VOLATILE_METADATA_KEYS changed the hash. "
            f"baseline={baseline} edited={edited}. Check "
            "invariants._VOLATILE_METADATA_KEYS — every key in this set must "
            "be popped from metadata before canonicalisation."
        )


# ---------------------------------------------------------------------------
# Sanity: all fixtures actually round-trip through Pydantic validation
# ---------------------------------------------------------------------------


class TestFixturesValidateThroughPydantic:
    """Every authoring fixture must survive ``model_validate`` cleanly.

    Catches accidental typos / schema drift in the canonical fixtures
    without depending on the hash assertions firing first.
    """

    @pytest.mark.parametrize(
        ("label", "kind", "fixture_fn"),
        _KIND_FIXTURES,
        ids=[label for label, _, _ in _KIND_FIXTURES],
    )
    def test_fixture_validates(
        self,
        label: str,
        kind: str,
        fixture_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Validate the authoring fixture through its Pydantic model."""
        authoring, _applied = fixture_fn()
        model_cls = _MODEL_FOR_KIND[kind]
        try:
            model_cls.model_validate(authoring)
        except Exception as exc:  # noqa: BLE001 — surface every kind of validation error.
            pytest.fail(f"{label}: authoring fixture did not validate through {model_cls.__name__}: {exc}")


# ---------------------------------------------------------------------------
# Streaming / realtime FV hash parity against runtime-stamped target_lag_sec=0
# ---------------------------------------------------------------------------
#
# The Snowflake runtime always stamps ``target_lag_sec: 0`` onto the
# deployed SPECIFICATION for streaming and realtime FVs regardless of
# the authored value (and ``_RUNTIME_STAMPED_SPEC_KEYS`` in
# ``invariants.py`` strips it from the deployed-side hash for that
# reason).  This contract has three load-bearing edges that the
# planner depends on for ``NO_CHANGE`` to fire cleanly on a re-apply:
#
# 1. The Pydantic validator rejects authored ``target_lag`` /
#    ``target_lag_sec`` on streaming / realtime kinds
#    (``spec_models.FeatureView._reject_target_lag_on_stream_or_realtime``).
# 2. The compiler drops the keys from streaming / realtime compiled
#    specs (``spec_compiler.compile_to_spec`` +
#    ``compiler.normalize_durations``).
# 3. The hash function strips ``target_lag_sec`` from the applied
#    side (``invariants._RUNTIME_STAMPED_SPEC_KEYS``).
#
# A regression in any one of those three would surface as a phantom
# RECREATE_FV on a clean re-apply.  The tests below explicitly pin
# the third-edge invariant — the hash function's runtime-stamped
# strip — against streaming and realtime fixtures so a refactor of
# ``_RUNTIME_STAMPED_SPEC_KEYS`` or ``_full_spec_hash`` that drops
# the strip surfaces here as a targeted, named failure.


def _streaming_fv_no_target_lag_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    """Streaming FV fixture with NO ``target_lag_sec`` on the applied side.

    Mirrors :func:`_streaming_fv_with_udf_fixture` but explicitly omits
    ``spec.target_lag_sec`` so the parity assertion below proves the
    hash is the same whether the applied side carries the runtime
    stamp or not.  After the Step 4 compiler strip the authored side
    never carries the key, so the only difference between the two
    fixtures is whether the *applied* side does — and the hash must
    be invariant under that difference.

    Returns:
        ``(authoring, applied)`` matching :func:`_streaming_fv_with_udf_fixture`
        with ``spec.target_lag_sec`` removed from ``applied``.
    """
    authoring, applied = _streaming_fv_with_udf_fixture()
    applied = copy.deepcopy(applied)
    applied["spec"].pop("target_lag_sec", None)
    return authoring, applied


class TestStreamingFvHashParityAgainstRuntimeStampedTargetLagZero:
    """Streaming / realtime FV hashes are invariant under the
    runtime-stamped ``target_lag_sec: 0`` on the applied side.

    Pins the ``_RUNTIME_STAMPED_SPEC_KEYS`` strip in
    :mod:`invariants` end-to-end against streaming / realtime
    fixtures.  A regression in the strip would surface as a phantom
    RECREATE_FV on a clean re-apply of any streaming or realtime FV.
    """

    def test_streaming_fv_hash_invariant_under_runtime_stamped_target_lag_zero(self) -> None:
        """The applied-side ``target_lag_sec: 0`` MUST NOT change the
        streaming FV hash.  Compares the canonical applied payload
        (which carries the stamp) against an applied payload with the
        stamp removed — both must hash equal.
        """
        authoring, applied_with_stamp = _streaming_fv_with_udf_fixture()
        _authoring2, applied_without_stamp = _streaming_fv_no_target_lag_fixture()

        hash_with_stamp = _full_spec_hash(applied_with_stamp)
        hash_without_stamp = _full_spec_hash(applied_without_stamp)

        assert hash_with_stamp == hash_without_stamp, (
            "Streaming FV hash changed when target_lag_sec: 0 was added "
            "to the applied side. The _RUNTIME_STAMPED_SPEC_KEYS strip "
            "in invariants.py must drop target_lag_sec from streaming "
            "applied payloads — otherwise the planner emits a phantom "
            "RECREATE_FV on a clean re-apply.\n"
            f"hash_with_stamp:    {hash_with_stamp}\n"
            f"hash_without_stamp: {hash_without_stamp}"
        )

    def test_streaming_fv_authoring_hash_matches_runtime_stamped_applied(self) -> None:
        """The authoring-side (post-Step 4 compile, no target_lag_sec)
        hash MUST equal the applied side that carries the runtime-
        stamped ``target_lag_sec: 0``.  This is the planner's
        ``NO_CHANGE`` invariant: a clean re-apply of a streaming FV
        that omits ``target_lag`` from the YAML must hash equal to
        the deployed FV (which carries the stamped ``0``).
        """
        authoring, applied = _streaming_fv_with_udf_fixture()

        # Sanity guard — the applied fixture is what DESCRIBE returns,
        # which carries the runtime stamp.
        assert applied["spec"].get("target_lag_sec") == 0

        # Sanity guard — the compiled side, post-Step 4, never carries
        # the key (the Pydantic validator would have rejected an
        # authored value anyway).
        compiled = _compile_for_kind("StreamingFeatureView", authoring)
        assert "target_lag_sec" not in compiled["spec"], (
            "spec_compiler.compile_to_spec must not emit target_lag_sec "
            "on streaming kinds — see Step 4 of the stream-fv-reject-"
            "target-lag plan."
        )

        # Load-bearing assertion.
        assert _hash_authoring("StreamingFeatureView", authoring) == _full_spec_hash(applied), (
            "Streaming FV authoring hash (no target_lag) does not equal "
            "applied hash (with runtime-stamped target_lag_sec=0). "
            "Either the compile-side strip or the runtime-stamped strip "
            "regressed; see _RUNTIME_STAMPED_SPEC_KEYS in invariants.py "
            "and the kind gate in spec_compiler.compile_to_spec."
        )

    def test_realtime_fv_hash_invariant_under_runtime_stamped_target_lag_zero(self) -> None:
        """Same contract for RealtimeFeatureView — the runtime treats
        realtime FVs identically to streaming FVs for target_lag.
        """
        _authoring, applied = _realtime_fv_fixture()

        baseline_hash = _full_spec_hash(applied)

        # Inject the runtime stamp onto the realtime fixture's applied
        # side (the canonical fixture omits it for brevity, but the
        # live runtime stamps it on every realtime FV at CREATE time).
        with_stamp = copy.deepcopy(applied)
        with_stamp["spec"]["target_lag_sec"] = 0
        stamped_hash = _full_spec_hash(with_stamp)

        assert baseline_hash == stamped_hash, (
            "Realtime FV hash changed when target_lag_sec: 0 was added "
            "to the applied side. _RUNTIME_STAMPED_SPEC_KEYS must strip "
            "target_lag_sec from realtime applied payloads.\n"
            f"baseline_hash: {baseline_hash}\n"
            f"stamped_hash:  {stamped_hash}"
        )


# ---------------------------------------------------------------------------
# Unused helper sentinel: explicitly export the routing helpers under
# ``__all__`` so a downstream consumer (e.g. a future cross-test fixture
# refactor) can reuse them without depending on private-by-convention names.
# ---------------------------------------------------------------------------

__all__ = [
    "_hash_for_kind",
    "_hash_authoring",
    "_compile_for_kind",
    "_strip_schema_alias",
    "_edit",
]


# ---------------------------------------------------------------------------
# B5 — Imperative (FV_SOURCE_REFS) vs. legacy DT-text hash equivalence
# ---------------------------------------------------------------------------
#
# Phase A introduced an ``FV_SOURCE_REFS`` metadata row that lets
# ``decl/state.py`` recover a feature view's authored ``spec.sources``
# directly from the metadata (the new "imperative" path).  Pre-upgrade
# deployed FVs do not carry the metadata row and fall back to the legacy
# DT-text recovery path (``_inject_batch_fv_source_from_dt_text``).
#
# AS9 (assumption register) commits to the contract that BOTH paths
# produce a ``spec_payload`` that hashes identically via
# :func:`_full_spec_hash`.  Without that contract, the first re-plan
# after a code upgrade emits a phantom ``RECREATE_FV`` for every FV
# whose source list is now recovered via the new path — a recreate
# storm.  ``B5`` closes the gap with the
# :func:`_normalise_fv_sources_for_hash` projection (sources collapse
# to a sorted ``[{"binding": <table-or-query-or-name>}]`` list, so the
# differences between the two paths — synthetic vs. authored
# ``name``, presence vs. absence of ``columns``, ``source_type``
# preservation — all wash out).  The test class below is the AS9
# regression pin.


def _bfv_table_via_source_refs() -> dict[str, Any]:
    """Authoring-shape applied payload built via the FV_SOURCE_REFS path.

    The new path recovers the operator's authored ``BatchSource``
    fields verbatim: ``name``, ``table``, ``source_type``, and the
    full ``columns`` schema.  This mirrors what
    :func:`state._build_offline_fv_object` produces when the
    ``feature_view_rows[i]['source_refs']`` JSON column is populated.

    Returns:
        AppliedObject ``spec_payload`` dict with operator-authored
        ``BatchSource`` fields recovered verbatim from FV_SOURCE_REFS.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                },
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 60,
        },
        "online_store_type": "postgres",
    }


def _bfv_table_via_legacy_dt_text() -> dict[str, Any]:
    """Same logical FV as :func:`_bfv_table_via_source_refs`, recovered
    via the legacy DT-text path.

    The legacy path stamps ``name = <table>`` (uses the physical table
    identifier as the synthetic source name) and never recovers
    ``columns``.  Without normalisation the resulting payload differs
    from the FV_SOURCE_REFS shape on three keys:

    1. ``sources[0].name``: ``RAW_EVENTS_BATCH_DECL`` (synthetic)
       vs. ``EVENTS_BATCH_DECL`` (authored).
    2. ``sources[0].columns``: missing (legacy) vs. populated (new).
    3. Possibly ``sources[0].source_type`` shape — both paths set it
       to ``"Batch"`` here, but the legacy path could omit it.

    All three differences must wash out under
    :func:`_normalise_fv_sources_for_hash` so the two payloads hash
    identically.

    Returns:
        AppliedObject ``spec_payload`` dict in the legacy DT-text
        recovered shape (physical table as synthetic source name,
        no per-column schema).
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "RAW_EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL",
                },
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 60,
        },
        "online_store_type": "postgres",
    }


def _bfv_tiled_via_source_refs() -> dict[str, Any]:
    """Tiled BatchFV applied payload via the FV_SOURCE_REFS path.

    Mirrors :func:`_batch_fv_advanced_fixture`'s applied side but
    forces the source binding through the authored ``BatchSource``
    shape (the operator's chosen name + the columns schema).

    Returns:
        AppliedObject ``spec_payload`` dict for the tiled BatchFV via the FV_SOURCE_REFS recovery path.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_ADV_BFV_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_ADV_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_ADV_DECL",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "SESSION_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "AMOUNT", "type": "FloatType"},
                    ],
                },
            ],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "target_lag_sec": 300,
            "cluster_by": ["USER_ID"],
            "refresh_mode": "INCREMENTAL",
            "initialize": "ON_CREATE",
            "aggregation_secondary_keys": ["SESSION_ID"],
        },
    }


def _bfv_tiled_via_legacy_dt_text() -> dict[str, Any]:
    """Same tiled FV via the legacy DT-text recovery path.

    Source ``name`` is synthesised from the underlying table; columns
    are absent.  Both differences wash out under
    :func:`_normalise_fv_sources_for_hash`.

    Returns:
        AppliedObject ``spec_payload`` dict for the tiled BatchFV via the legacy DT-text recovery path.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_ADV_BFV_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "RAW_EVENTS_ADV_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_ADV_DECL",
                },
            ],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "target_lag_sec": 300,
            "cluster_by": ["USER_ID"],
            "refresh_mode": "INCREMENTAL",
            "initialize": "ON_CREATE",
            "aggregation_secondary_keys": ["SESSION_ID"],
        },
    }


def _bfv_query_via_source_refs() -> dict[str, Any]:
    """Query-backed BatchFV via the FV_SOURCE_REFS path.

    The new path preserves the operator's authored ``name`` and emits
    the SQL body verbatim under ``query``.

    Returns:
        AppliedObject ``spec_payload`` dict for the query-backed
        BatchFV via the FV_SOURCE_REFS path (preserves operator-authored
        name + verbatim SQL body).
    """
    query = "SELECT user_id, event_ts, metric_val FROM raw.events WHERE metric_val > 0"
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_SQL_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_SQL_BATCH_DECL",
                    "source_type": "Batch",
                    "query": query,
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                },
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 300,
        },
        "online_store_type": "postgres",
    }


def _bfv_query_via_legacy_dt_text() -> dict[str, Any]:
    """Same query-backed FV via the legacy DT-text recovery path.

    The legacy path synthesizes ``<FV>__SOURCE`` for the source name
    (because there is no underlying table identifier to use) and the
    SQL body is whitespace-normalised by the recovery helpers.
    Columns are absent.  All differences wash out under the
    ``[{"binding": "QUERY:<body>"}]`` projection.

    Returns:
        AppliedObject ``spec_payload`` dict for the query-backed
        BatchFV via the legacy DT-text path (synthetic
        ``<FV>__SOURCE`` name, whitespace-normalised SQL body).
    """
    query = "SELECT user_id, event_ts, metric_val FROM raw.events WHERE metric_val > 0"
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_SQL_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    # Synthetic name produced by the DT-text recovery path
                    # for query-backed BatchFVs (no underlying table).
                    "name": "MY_SQL_BATCH_FV_BATCH_DECL__SOURCE",
                    "source_type": "Batch",
                    "query": query,
                },
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 300,
        },
        "online_store_type": "postgres",
    }


def _streaming_fv_via_source_refs() -> dict[str, Any]:
    """Streaming FV applied payload via the FV_SOURCE_REFS path.

    Streaming sources don't carry a ``table`` field so the binding
    reduces to the source ``name``; the new path preserves the
    operator's authored name + columns.

    Returns:
        AppliedObject ``spec_payload`` dict for the streaming FV
        via the FV_SOURCE_REFS path (operator-authored name + columns
        preserved).
    """
    udf_body = "def transform(df):\n    return df['EVENT'].count()"
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_CLICK_STATS",
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "1.38.0",
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "UDFTransformed",
                "database": _DB,
                "schema": _SCHEMA,
                "table": "USER_CLICK_STATS$V1$UDF_TRANSFORMED",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "EVENT", "type": "StringType"},
                    "output_column": {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                    "function": "count",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "tiles",
            "udf": {
                "name": "transform",
                "engine": "pandas",
                "function_definition": udf_body,
                "output_columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            },
            "target_lag_sec": 0,
        },
        "online_store_type": "postgres",
    }


def _streaming_fv_via_legacy_dt_text() -> dict[str, Any]:
    """Same streaming FV via the legacy DESCRIBE-only path.

    For streaming FVs the legacy DESCRIBE TYPE = SPECIFICATION
    payload preserves ``sources[0].name`` and columns, so the two
    paths look almost identical here — but the legacy fixture
    deliberately omits ``columns`` (some legacy specs lack them) to
    pin that the hash collapses both shapes to the same binding.

    Returns:
        AppliedObject ``spec_payload`` dict for the streaming FV via
        the legacy DESCRIBE-only path (name + columns already present,
        so payload is nearly identical to the FV_SOURCE_REFS path).
    """
    udf_body = "def transform(df):\n    return df['EVENT'].count()"
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_CLICK_STATS",
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "1.38.0",
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "UDFTransformed",
                "database": _DB,
                "schema": _SCHEMA,
                "table": "USER_CLICK_STATS$V1$UDF_TRANSFORMED",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    # Legacy fixture omits columns to verify the hash
                    # collapses with-columns vs. without-columns to the
                    # same binding.
                }
            ],
            "features": [
                {
                    "source_column": {"name": "EVENT", "type": "StringType"},
                    "output_column": {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                    "function": "count",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "tiles",
            "udf": {
                "name": "transform",
                "engine": "pandas",
                "function_definition": udf_body,
                "output_columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_COUNT_1H", "type": "IntegerType"},
                ],
            },
            "target_lag_sec": 0,
        },
        "online_store_type": "postgres",
    }


_IMPERATIVE_VS_LEGACY_FIXTURES: list[tuple[str, Callable[[], dict[str, Any]], Callable[[], dict[str, Any]]]] = [
    ("non_tiled_BFV_table", _bfv_table_via_source_refs, _bfv_table_via_legacy_dt_text),
    ("tiled_BFV", _bfv_tiled_via_source_refs, _bfv_tiled_via_legacy_dt_text),
    ("BFV_with_query", _bfv_query_via_source_refs, _bfv_query_via_legacy_dt_text),
    ("streaming_FV", _streaming_fv_via_source_refs, _streaming_fv_via_legacy_dt_text),
]


class TestImperativeVsLegacyHashEquivalence:
    """The new FV_SOURCE_REFS recovery path and the legacy DT-text
    recovery path must produce identical hashes for the same logical
    FV (AS9).

    Why this matters.  When the snowml-core upgrade lands, every
    pre-upgrade deployed FV will continue to be recovered via the
    legacy DT-text path (the metadata row was never written for it).
    Newly-applied FVs after the upgrade will be recovered via
    FV_SOURCE_REFS.  If the two paths hash differently for the same
    FV, the first re-plan after the upgrade — even on a clean,
    idempotent set of YAMLs — will emit a phantom RECREATE_FV for
    every legacy FV.  AS9 commits to the contract that this does not
    happen; this test class is the regression pin.
    """

    @pytest.mark.parametrize(
        ("label", "via_source_refs", "via_legacy"),
        _IMPERATIVE_VS_LEGACY_FIXTURES,
        ids=[label for label, _, _ in _IMPERATIVE_VS_LEGACY_FIXTURES],
    )
    def test_full_spec_hash_equivalence(
        self,
        label: str,
        via_source_refs: Callable[[], dict[str, Any]],
        via_legacy: Callable[[], dict[str, Any]],
    ) -> None:
        """``_full_spec_hash`` is invariant across the two recovery paths.

        Args:
            label: Parametrised fixture id used in the failure message.
            via_source_refs: Zero-arg callable returning the applied
                payload as the FV_SOURCE_REFS path would produce it.
            via_legacy: Zero-arg callable returning the applied
                payload as the legacy DT-text recovery path would
                produce it.
        """
        new_payload = via_source_refs()
        legacy_payload = via_legacy()

        new_hash = _full_spec_hash(new_payload)
        legacy_hash = _full_spec_hash(legacy_payload)

        assert new_hash == legacy_hash, (
            f"{label}: hash differs across the FV_SOURCE_REFS recovery "
            f"path and the legacy DT-text recovery path. AS9 in the "
            f"metadata-roundtrip plan commits to the two paths producing "
            f"identical hashes — without that, the first re-plan after "
            f"the snowml-core upgrade emits a phantom RECREATE_FV for "
            f"every legacy FV.\n"
            f"  via_FV_SOURCE_REFS: {new_hash}\n"
            f"  via_legacy_DT_text: {legacy_hash}\n"
            f"Inspect the source binding under "
            f"_normalise_fv_sources_for_hash and verify both paths "
            f"reduce to the same `[{{'binding': ...}}]` projection."
        )


# ===========================================================================
# Phase 3 RED — refresh_freq propagation through the canonicalization
# pipeline.  These pin the four-mechanism Canonicalization Invariant
# (docs/DEVELOPMENT_STANDARDS.md "Phase E") for the refresh_freq rename:
# Mechanism #1 (_OPERATIONAL_FV_KEYS strip) and Mechanism #3 (applied-side
# decoding via state._inject_*) must both target the renamed field.
# ===========================================================================


class TestOperationalKeysContainsRefreshFreq:
    """``_OPERATIONAL_FV_KEYS`` carries the renamed key ``refresh_freq``.

    The operational-keys frozenset is Mechanism #1 of the Phase E
    Canonicalization Invariant: a field listed here is stripped from
    both the local-compile and applied-state hash bases so that an
    operational-only edit (``UPDATE_FV``) does not bump the structural
    hash and trigger a destructive ``RECREATE_FV``.

    After the ``refresh_freq -> refresh_freq`` rename, ``refresh_freq``
    is the canonical authoring + wire name, so the operational-keys set
    must list it (not the legacy ``refresh_freq``).
    """

    def test_refresh_freq_is_in_operational_fv_keys(self) -> None:
        from snowflake.ml.feature_store.decl import invariants

        assert "refresh_freq" in invariants._OPERATIONAL_FV_KEYS, (
            "_OPERATIONAL_FV_KEYS must list 'refresh_freq' so the "
            "structural-hash strip and the planner's "
            "_batch_fv_operational_drift route a refresh_freq edit to "
            "UPDATE_FV instead of RECREATE_FV."
        )

    def test_batch_schedule_is_not_in_operational_fv_keys(self) -> None:
        from snowflake.ml.feature_store.decl import invariants

        assert "batch_schedule" not in invariants._OPERATIONAL_FV_KEYS, (
            "Legacy 'batch_schedule' must be removed from "
            "_OPERATIONAL_FV_KEYS post-rename — the authoring side now "
            "rejects it as a hard migration error, so any leftover entry "
            "would be dead code that confuses future readers."
        )


class TestRefreshFreqRoundTrip:
    """Symmetric round-trip pin per the Phase E "Canonicalization
    Invariant" checklist.

    A BFV authored with ``refresh_freq: "5 minutes"`` and a recovered
    applied payload that surfaces the same DT cadence must hash
    identically.  Without this, the second ``snow feature plan`` after
    a clean apply emits a spurious ``UPDATE_FV`` (or worse,
    ``RECREATE_FV``) for every BFV with an authored cadence.

    Mechanism walked here:
    * Local side authors ``refresh_freq``.  ``_full_spec_hash`` strips
      it (Mechanism #1) so it does not contribute to the structural
      hash directly.
    * Applied side carries the runtime-stamped wire form
      ``spec.target_lag_sec`` (which the OFT-row injection helper
      mirrors back into ``spec.refresh_freq`` for human-readable
      authoring round-trip).  ``_full_spec_hash`` strips both keys.
    * Result: the two hashes collapse regardless of authored cadence,
      provided every other field round-trips.
    """

    def test_refresh_freq_strip_makes_local_and_applied_hash_match(self) -> None:
        """A local BFV authored with ``refresh_freq`` and an applied
        payload carrying ``target_lag_sec`` for the same logical FV
        must hash identically.
        """
        local: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {
                "name": "BFV_REFRESH_FREQ_PIN",
                "version": "V1",
                "database": _DB,
                "schema": _SCHEMA,
                "client_version": "1.38.0",
                "spec_format_version": "1",
            },
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [
                    {
                        "name": "RAW_EVENTS",
                        "source_type": "Batch",
                        "table": "RAW_EVENTS",
                    }
                ],
                "features": [],
                "refresh_freq": "5 minutes",
            },
        }
        applied = copy.deepcopy(local)
        applied["spec"].pop("refresh_freq")
        applied["spec"]["target_lag_sec"] = 300

        assert _full_spec_hash(local) == _full_spec_hash(applied), (
            "_OPERATIONAL_FV_KEYS must strip 'refresh_freq' from the "
            "local side and 'target_lag_sec' from the applied side so "
            "the two collapse to the same structural hash regardless of "
            "the authored DT cadence."
        )

    def test_refresh_freq_edit_does_not_change_full_spec_hash(self) -> None:
        """Editing only ``refresh_freq`` must NOT bump the structural hash.

        This is the strict Mechanism #1 contract — an operational-only
        edit (cadence change) must route through ``UPDATE_FV``, which
        the planner can only do if the structural hashes match.
        """
        base: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {
                "name": "BFV_OP_DRIFT_PIN",
                "version": "V1",
                "database": _DB,
                "schema": _SCHEMA,
                "client_version": "1.38.0",
                "spec_format_version": "1",
            },
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [
                    {
                        "name": "RAW_EVENTS",
                        "source_type": "Batch",
                        "table": "RAW_EVENTS",
                    }
                ],
                "features": [],
                "refresh_freq": "5 minutes",
            },
        }
        edited = copy.deepcopy(base)
        edited["spec"]["refresh_freq"] = "1 hour"

        assert _full_spec_hash(base) == _full_spec_hash(edited), (
            "An isolated refresh_freq edit must not change the "
            "structural hash; otherwise the planner cannot route the "
            "change through UPDATE_FV (it would emit RECREATE_FV)."
        )


class TestQuerySourceHashCanonicalization:
    """Query-backed FV source bindings are canonicalized before hashing so a
    comment / keyword-case / whitespace-only SQL edit does not bump the
    structural hash (which would spuriously emit RECREATE_FV / RECREATE_SOURCE
    and, for append-only BFVs, wipe the ``$SNAPSHOTS`` history)."""

    def _sources(self, query: str) -> Any:
        return [{"name": "ORDERS", "query": query}]

    def test_comment_only_query_edit_same_binding(self) -> None:
        a = _normalise_fv_sources_for_hash(self._sources("SELECT id FROM orders -- v1"))
        b = _normalise_fv_sources_for_hash(self._sources("SELECT id FROM orders -- v2 changed"))
        assert a == b

    def test_keyword_case_only_query_edit_same_binding(self) -> None:
        a = _normalise_fv_sources_for_hash(self._sources("select id from orders"))
        b = _normalise_fv_sources_for_hash(self._sources("SELECT id FROM orders"))
        assert a == b

    def test_whitespace_only_query_edit_same_binding(self) -> None:
        a = _normalise_fv_sources_for_hash(self._sources("SELECT id\n  FROM orders"))
        b = _normalise_fv_sources_for_hash(self._sources("SELECT id FROM orders"))
        assert a == b

    def test_semantic_query_edit_differs(self) -> None:
        a = _normalise_fv_sources_for_hash(self._sources("SELECT id FROM orders"))
        b = _normalise_fv_sources_for_hash(self._sources("SELECT id, total FROM orders"))
        assert a != b

    def test_structural_fingerprint_ignores_comment_only_query_edit(self) -> None:
        base: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "FV",
            "version": "V1",
            "columns": [{"name": "ID", "type": "LongType"}],
            "sources": [{"name": "ORDERS", "query": "SELECT id FROM orders -- a"}],
        }
        edited = copy.deepcopy(base)
        edited["sources"][0]["query"] = "SELECT id FROM orders -- b (edited)"
        assert structural_fingerprint_hash(base) == structural_fingerprint_hash(edited)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
