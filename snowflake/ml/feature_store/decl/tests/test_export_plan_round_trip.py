"""Round-trip invariant: export → load → plan emits zero non-NO_CHANGE ops.

Pins the contract that ``snow feature export`` writes YAMLs that re-apply
cleanly without spurious ``RECREATE_FV`` / ``VERSION_CONFLICT`` errors.

Two layers of coverage:

1. :class:`TestCompileToSpecHashSymmetry` — unit-level: for each FV kind in
   the live SPECIFICATION-shape fixtures, the hash of the local compiled
   spec equals the hash of the deployed spec payload.  This is the lowest
   layer at which round-trip drift can appear.

2. :class:`TestExportThenPlanEmitsNoChange` — end-to-end: drive the actual
   exporter, the actual loader, and the
   ``resolve_datasource_columns`` → ``validate_specs`` → ``generate_plan``
   pipeline; assert every op kind is ``NO_CHANGE``.  (The legacy
   ``generate_apply_sql`` orchestrator was removed alongside the dry-run
   apply path; the underlying primitives it used to call are exercised
   directly here so the round-trip pin survives that refactor.)

Hypothesis gate H2 (per ``plans/restore_export_plan_invariant.plan.md``):
if any symmetry check fails the failure surfaces the canonical-JSON diff
of the offending field so we can stop and report to the user before
widening :data:`~snowflake.ml.feature_store.decl.invariants._VOLATILE_METADATA_KEYS`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.exporter import export_specs
from snowflake.ml.feature_store.decl.invariants import (
    _VOLATILE_METADATA_KEYS,
    _full_spec_hash,
)
from snowflake.ml.feature_store.decl.loader import load_specs
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import PlanOptions
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Fixtures — SPECIFICATION-shape JSON dicts (one per FV kind)
# ---------------------------------------------------------------------------
#
# These mirror the JSON shape that ``DESCRIBE ... TYPE = SPECIFICATION``
# returns.  Each carries the keys ``compile_to_spec`` will produce locally
# (sans the volatile metadata keys, which are stripped by ``_full_spec_hash``).


_STREAMING_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_clicks",
        "version": "v1",
        "spec_format_version": "1",
        "internal_data_version": "1",
        "client_version": "0.1.0",
    },
    "offline_configs": [
        {
            "store_type": "snowflake",
            "table_type": "UDFTransformed",
            "database": "MYDB",
            "schema": "PUBLIC",
            "table": "USER_CLICKS$V1$UDF_TRANSFORMED",
            "columns": [{"name": "event_count", "type": "IntegerType"}],
        }
    ],
    "spec": {
        "ordered_entity_column_names": ["user_id"],
        "sources": [
            {
                "name": "user_events",
                "source_type": "Stream",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "event", "type": "StringType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "event", "type": "StringType"},
                "output_column": {"name": "event_count", "type": "IntegerType"},
                "function": "count",
                "window_sec": 3600,
                "offset_sec": 0,
            }
        ],
        "udf": {
            "name": "transform",
            "function_definition": "def transform(x):\n    return len(x)",
            "engine": "python",
            "output_columns": [{"name": "event_count", "type": "IntegerType"}],
        },
        "timestamp_field": "event_time",
        "feature_granularity_sec": 60,
        "feature_aggregation_method": "tiles",
    },
    "online_store_type": "postgres",
}


# ---------------------------------------------------------------------------
# BatchFeatureView — modelled on tests/golden_specs/USER_PROFILE_INFO_BATCH.json
# ---------------------------------------------------------------------------
#
# Exercises two paths the streaming fixture doesn't:
# 1. ``spec.target_lag_sec`` carried at the top of ``spec`` (no aggregation
#    windows), which an earlier pass dropped from both the exporter and the
#    compiler.
# 2. Empty ``spec.sources`` — the FV pulls from an offline table tracked
#    only via ``offline_configs`` (which is now stripped from the hash by
#    :data:`_DERIVED_TOP_LEVEL_KEYS`).

_BATCH_SPEC: dict[str, Any] = {
    "kind": "BatchFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_profile_info_batch",
        "version": "v1",
        "spec_format_version": "1",
        "internal_data_version": "1",
        "client_version": "0.1.0",
        "oft_id": "999000",  # volatile — stripped by _full_spec_hash
    },
    "offline_configs": [
        {
            "store_type": "snowflake",
            "table_type": "BatchSource",
            "database": "MYDB",
            "schema": "PUBLIC",
            "table": "USER_PROFILE_INFO_BATCH$v1",
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "UPDATED_AT", "type": "TimestampType"},
                {"name": "CITY", "type": "StringType"},
            ],
        }
    ],
    "spec": {
        "ordered_entity_column_names": ["USER_ID"],
        "sources": [],
        "features": [
            {
                "source_column": {"name": "CITY", "type": "StringType"},
                "output_column": {"name": "CITY", "type": "StringType"},
            }
        ],
        "timestamp_field": "UPDATED_AT",
        "target_lag_sec": 10,
    },
    "online_store_type": "postgres",
}


# ---------------------------------------------------------------------------
# Continuous-flagged StreamingFeatureView — modelled on
# tests/golden_specs/USER_CLICK_STATS_M2_CONTINUOUS.json
# ---------------------------------------------------------------------------
#
# Same ``StreamingFeatureView`` kind as the basic streaming case but with a
# ``feature_aggregation_method`` of ``"continuous"`` and a UDF.  Pins that
# the continuous variant round-trips identically to the tiles variant.

_CONTINUOUS_STREAMING_NO_GRANULARITY_AUTHORING: dict[str, Any] = {
    # Authoring-shape dict (the YAML-loaded ``FeatureView.model_dump``
    # equivalent) for a StreamingFeatureView declared with
    # ``feature_aggregation_method: continuous`` but **without** an
    # explicit ``feature_granularity`` / ``feature_granularity_sec``.
    # The compile-time default in
    # :func:`spec_compiler.compile_to_spec` must stamp
    # ``feature_granularity_sec: 60`` onto the wire spec so the local
    # hash matches the deployed SPECIFICATION JSON (which the runtime
    # post-defaults to ``60`` via snowml-core's
    # ``_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = "1m"``).
    "kind": "StreamingFeatureView",
    "name": "user_clicks_continuous",
    "version": "v1",
    "database": "MYDB",
    "schema": "PUBLIC",
    "entities": ["user_id"],
    "sources": [
        {
            "name": "user_events",
            "source_type": "Stream",
            "columns": [
                {"name": "user_id", "type": "StringType"},
                {"name": "click", "type": "IntegerType"},
            ],
        }
    ],
    "features": [
        {
            "source_column": {"name": "click", "type": "IntegerType"},
            "output_column": {"name": "score", "type": "FloatType"},
            "function": "sum",
            "window_sec": 600,
            "offset_sec": 0,
        }
    ],
    "udf": {
        "name": "score_clicks",
        "engine": "pandas",
        "function_definition": "def score_clicks(df):\n    return df['click'].sum()",
        "output_columns": [{"name": "score", "type": "FloatType"}],
    },
    "timestamp_col": "event_time",
    "feature_aggregation_method": "continuous",
    "refresh_freq": "5 minutes",
}


_CONTINUOUS_STREAMING_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_clicks_continuous",
        "version": "v1",
        "spec_format_version": "1",
        "internal_data_version": "1",
        "client_version": "0.1.0",
    },
    "offline_configs": [
        {
            "store_type": "snowflake",
            "table_type": "UDFTransformed",
            "database": "MYDB",
            "schema": "PUBLIC",
            "table": "USER_CLICKS_CONTINUOUS$V1$UDF_TRANSFORMED",
            "columns": [{"name": "score", "type": "FloatType"}],
        }
    ],
    "spec": {
        "ordered_entity_column_names": ["user_id"],
        "sources": [
            {
                "name": "user_events",
                "source_type": "Stream",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "click", "type": "IntegerType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "click", "type": "IntegerType"},
                "output_column": {"name": "score", "type": "FloatType"},
                "function": "sum",
                "window_sec": 600,
                "offset_sec": 0,
            }
        ],
        "udf": {
            "name": "score_clicks",
            "engine": "pandas",
            "function_definition": "def score_clicks(df):\n    return df['click'].sum()",
            "output_columns": [{"name": "score", "type": "FloatType"}],
        },
        "timestamp_field": "event_time",
        "feature_granularity_sec": 60,
        "feature_aggregation_method": "continuous",
    },
    "online_store_type": "postgres",
}


def _show_row(name: str, version: str) -> dict[str, Any]:
    return {
        "name": f"{name.upper()}${version.upper()}$ONLINE",
        "database_name": "MYDB",
        "schema_name": "PUBLIC",
        "scheduling_state": "ACTIVE",
    }


def _strip_volatile(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *spec* with volatile metadata keys removed.

    Mirrors the cleaning done inside :func:`_full_spec_hash` so we can show
    a meaningful diff when the symmetry assertion fails.

    Args:
        spec: Original spec payload to clean.

    Returns:
        dict: deep copy with volatile metadata keys removed.
    """
    cleaned = copy.deepcopy(spec)
    md = cleaned.get("metadata") if isinstance(cleaned, dict) else None
    if isinstance(md, dict):
        for key in _VOLATILE_METADATA_KEYS:
            md.pop(key, None)
    return cleaned


def _live_applied_state(spec_payload: dict[str, Any]) -> object:
    """Build a live-shape AppliedState via :func:`fetch_applied_state`.

    Mirrors what the CLI's ``_fetch_oft_state`` produces: a single FV
    AppliedObject from ``specification_map``, plus auto-derived Entity
    and Datasource objects from the same spec payload.  This is the
    shape the round-trip plan operates against in production.

    Args:
        spec_payload: SPECIFICATION-shape dict (carries ``metadata`` +
            ``spec.ordered_entity_column_names`` + ``spec.sources``).

    Returns:
        object: an AppliedState covering the FV, every referenced entity,
        and every referenced datasource.
    """
    metadata = spec_payload.get("metadata", {})
    name = metadata.get("name", "")
    version = metadata.get("version", "")
    db = metadata.get("database", "")
    schema = metadata.get("schema", "")
    oft_name = f"{name.upper()}${version.upper()}$ONLINE"

    show_rows = [_show_row(name, version)]
    spec_map = {oft_name: spec_payload}

    entity_rows = []
    for col_name in spec_payload.get("spec", {}).get("ordered_entity_column_names", []):
        entity_rows.append(
            {
                "name": f"SNOWML_FEATURE_STORE_ENTITY_{col_name.upper()}",
                "database_name": db,
                "schema_name": schema,
                "allowed_values": f'["{col_name.upper()}"]',
            }
        )

    return fetch_applied_state(
        show_rows,
        None,
        specification_map=spec_map,
        entity_rows=entity_rows,
        default_database=db,
        default_schema=schema,
    )


# ---------------------------------------------------------------------------
# H2 — Symmetry: compile_to_spec(loaded_yaml) hashes equal spec_payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_payload",
    [
        pytest.param(_STREAMING_SPEC, id="StreamingFeatureView"),
        pytest.param(_BATCH_SPEC, id="BatchFeatureView"),
        pytest.param(_CONTINUOUS_STREAMING_SPEC, id="ContinuousStreamingFeatureView"),
    ],
)
class TestCompileToSpecHashSymmetry:
    """Hypothesis H2: hash(compile_to_spec(loaded YAML)) == hash(spec_payload)."""

    def test_export_then_load_then_compile_hash_equals_payload(self, tmp_path: Path, spec_payload: Any) -> None:
        """End-to-end symmetry: export YAML, load, compile, hash equals payload hash."""
        oft_name = (
            f"{spec_payload['metadata']['name'].upper()}" f"${spec_payload['metadata']['version'].upper()}$ONLINE"
        )
        export_specs(
            show_rows=[_show_row(spec_payload["metadata"]["name"], spec_payload["metadata"]["version"])],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={oft_name: spec_payload},
        )

        export_root = tmp_path / "MYDB.PUBLIC"
        batch = load_specs([f"{export_root}/..."])

        fv_kind = spec_payload["kind"]
        fv_specs = [s for s in batch.specs if getattr(s, "kind", "") == fv_kind]
        assert fv_specs, (
            f"loader returned no spec of kind {fv_kind} from {export_root}; "
            f"got kinds={[getattr(s, 'kind', '?') for s in batch.specs]!r}"
        )
        loaded_dict = fv_specs[0].model_dump(exclude_none=True)
        if "schema_" in loaded_dict:
            loaded_dict["schema"] = loaded_dict.pop("schema_")

        compiled = compile_to_spec(loaded_dict, "MYDB", "PUBLIC")

        if _full_spec_hash(compiled) != _full_spec_hash(spec_payload):
            expected = json.dumps(_strip_volatile(spec_payload), sort_keys=True, indent=2)
            actual = json.dumps(_strip_volatile(compiled), sort_keys=True, indent=2)
            pytest.fail(
                "compile_to_spec(loaded YAML) hash != spec_payload hash.\n"
                "This means a freshly-exported tree will not round-trip cleanly.\n"
                "STOP and report to the user (plan H2 gate).\n\n"
                f"--- expected (spec_payload, volatile keys stripped) ---\n{expected}\n"
                f"--- actual (compiled, volatile keys stripped) ---\n{actual}\n"
            )


# ---------------------------------------------------------------------------
# CONTINUOUS-granularity compile-time default: authoring spec without
# ``feature_granularity_sec`` must hash-equal the runtime-defaulted
# SPECIFICATION JSON (``feature_granularity_sec: 60``).
# ---------------------------------------------------------------------------


class TestContinuousGranularityCompileDefaultHashSymmetry:
    """Pins the round-trip hash symmetry for the
    ``feature_aggregation_method: continuous`` default.

    Mirrors :class:`TestCompileToSpecHashSymmetry` but skips the
    export → reload step because that loop just preserves whatever the
    deployed JSON carries.  The interesting failure mode is the
    asymmetry between:

    * the **deployed** SPECIFICATION JSON (which the runtime always
      stamps with ``feature_granularity_sec: 60`` for a continuous
      streaming FV — see snowml-core's
      ``_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = "1m"``), and
    * the **authoring** YAML on disk (which may legitimately omit
      ``feature_granularity_sec`` for an author who only chose
      ``feature_aggregation_method: continuous``).

    Without the compile-time default this asymmetry surfaces as a
    spurious ``RECREATE_FV`` on every ``snow feature plan`` against an
    unchanged remote state.  With the default in place the local
    compile stamps the same ``60`` and the hash matches.
    """

    def test_authoring_no_granularity_compiles_to_same_hash_as_deployed_60(self) -> None:
        """compile_to_spec(authoring_no_granularity) must hash-equal the
        deployed SPECIFICATION JSON that carries
        ``feature_granularity_sec: 60``."""
        compiled = compile_to_spec(
            _CONTINUOUS_STREAMING_NO_GRANULARITY_AUTHORING,
            "MYDB",
            "PUBLIC",
        )

        if _full_spec_hash(compiled) != _full_spec_hash(_CONTINUOUS_STREAMING_SPEC):
            expected = json.dumps(_strip_volatile(_CONTINUOUS_STREAMING_SPEC), sort_keys=True, indent=2)
            actual = json.dumps(_strip_volatile(compiled), sort_keys=True, indent=2)
            pytest.fail(
                "compile_to_spec(authoring no-granularity) hash != deployed spec hash.\n"
                "An authoring SFV+CONTINUOUS YAML without feature_granularity_sec must "
                "compile to a wire spec carrying feature_granularity_sec=60 (matching "
                "snowml-core's _DEFAULT_CONTINUOUS_FEATURE_GRANULARITY='1m').  Without "
                "this default, snow feature plan will emit a spurious RECREATE_FV "
                "against an unchanged remote state.\n\n"
                f"--- expected (deployed spec_payload, volatile keys stripped) ---\n{expected}\n"
                f"--- actual (compiled, volatile keys stripped) ---\n{actual}\n"
            )


# ---------------------------------------------------------------------------
# End-to-end — export → load → resolve+validate+plan emits NO_CHANGE per op
# ---------------------------------------------------------------------------


def _validate_and_plan(batch: Any, applied_state: Any, options: Any, database: Any, schema: Any) -> tuple[Any, Any]:
    """Run the same pipeline ``manager.plan`` runs and return ``(errors, plan)``.

    Mirrors the validate-then-plan code path in
    ``snowflake.cli._plugins.feature.manager.FeatureManager.plan``:

    1. ``resolve_datasource_columns`` — inject datasource columns into FV refs.
    2. ``validate_specs`` — return ERROR severities for the caller to assert on.
    3. ``generate_plan`` — produce the structured ``Plan`` (only when no errors).

    Args:
        batch: The compiled spec batch from ``load_specs``.
        applied_state: The ``AppliedState`` snapshot to plan against.
        options: ``PlanOptions`` controlling the planner.
        database: Default database for spec-key qualification.
        schema: Default schema for spec-key qualification.

    Returns:
        Tuple ``(errors, plan)``.  ``plan`` is ``None`` when ``errors``
        is non-empty (matching the manager's short-circuit).
    """
    decl_api.resolve_datasource_columns(batch)
    errors = [
        r
        for r in decl_api.validate_specs(
            batch,
            applied_state,
            target_database=database,
            target_schema=schema,
        )
        if r.severity == "ERROR"
    ]
    if errors:
        return errors, None
    plan = decl_api.generate_plan(
        batch,
        applied_state,
        options,
        database=database,
        schema=schema,
    )
    return errors, plan


@pytest.mark.parametrize(
    "spec_payload",
    [
        pytest.param(_STREAMING_SPEC, id="StreamingFeatureView"),
        pytest.param(_BATCH_SPEC, id="BatchFeatureView"),
        pytest.param(_CONTINUOUS_STREAMING_SPEC, id="ContinuousStreamingFeatureView"),
    ],
)
class TestExportThenPlanEmitsNoChange:
    """Highest-level round-trip pin: a freshly-exported tree plans as zero ops."""

    def test_export_then_plan_emits_no_change(self, tmp_path: Path, spec_payload: Any) -> None:
        """Export, load, then validate+plan; every op must be NO_CHANGE."""
        applied_state = _live_applied_state(spec_payload)

        oft_name = (
            f"{spec_payload['metadata']['name'].upper()}" f"${spec_payload['metadata']['version'].upper()}$ONLINE"
        )
        export_specs(
            show_rows=[_show_row(spec_payload["metadata"]["name"], spec_payload["metadata"]["version"])],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={oft_name: spec_payload},
        )

        export_root = tmp_path / "MYDB.PUBLIC"
        batch = load_specs([f"{export_root}/..."])

        errors, plan = _validate_and_plan(
            batch,
            applied_state,
            PlanOptions(),
            database="MYDB",
            schema="PUBLIC",
        )

        assert errors == [], (
            f"validation_failed on round-trip; this means the exporter→loader→validator "
            f"pipeline disagrees on an unchanged spec. errors={errors!r}"
        )

        op_kinds = {op.kind.value for op in plan.ops}
        non_no_change_ops = [op for op in plan.ops if op.kind.value != "NO_CHANGE"]
        assert non_no_change_ops == [], (
            f"Round-trip invariant violated: every op must be NO_CHANGE, got "
            f"{op_kinds!r}; non-NO_CHANGE ops = {non_no_change_ops!r}"
        )


# ---------------------------------------------------------------------------
# Orphan-entity round-trip: export(...entity_rows=...) → plan ./... emits
# zero DROP_ENTITY ops even when the schema has tags not referenced by any FV
# ---------------------------------------------------------------------------
#
# This is the end-to-end pin for the export ↔ plan asymmetry fix.  Without
# entity_rows the planner in full-directory mode will see the orphan tag in
# the applied state, fail to find a local YAML for it, and emit a spurious
# ``DROP_ENTITY`` op.  With entity_rows the exporter writes a YAML stub for
# every tag and the planner reports ``NO_CHANGE``.


def _orphan_applied_state(
    spec_payload: dict[str, Any],
    orphan_entity_names: list[str],
) -> tuple[Any, Any]:
    """Build an AppliedState covering an FV plus orphan entity tags.

    Extends :func:`_live_applied_state` with extra entity_rows for tags
    that no FV references — exactly the configuration that triggers
    ``DROP_ENTITY`` ops in full-directory mode without the export fix.

    Args:
        spec_payload: SPECIFICATION-shape FV dict.
        orphan_entity_names: Entity names registered as tags but not
            attached to any FV's ``ordered_entity_column_names``.

    Returns:
        AppliedState covering the FV, every FV-referenced entity, and
        every supplied orphan entity.
    """
    metadata = spec_payload.get("metadata", {})
    name = metadata.get("name", "")
    version = metadata.get("version", "")
    db = metadata.get("database", "")
    schema = metadata.get("schema", "")
    oft_name = f"{name.upper()}${version.upper()}$ONLINE"

    show_rows = [_show_row(name, version)]
    spec_map = {oft_name: spec_payload}

    entity_rows: list[dict[str, Any]] = []
    for col_name in spec_payload.get("spec", {}).get("ordered_entity_column_names", []):
        entity_rows.append(
            {
                "name": f"SNOWML_FEATURE_STORE_ENTITY_{col_name.upper()}",
                "database_name": db,
                "schema_name": schema,
                "allowed_values": f'["{col_name.upper()}"]',
            }
        )
    for orphan_name in orphan_entity_names:
        entity_rows.append(
            {
                "name": f"SNOWML_FEATURE_STORE_ENTITY_{orphan_name.upper()}",
                "database_name": db,
                "schema_name": schema,
                "allowed_values": f'["{orphan_name.upper()}"]',
                "comment": f"orphan tag {orphan_name}",
            }
        )

    return (
        fetch_applied_state(
            show_rows,
            None,
            specification_map=spec_map,
            entity_rows=entity_rows,
            default_database=db,
            default_schema=schema,
        ),
        entity_rows,
    )


class TestExportThenPlanWithOrphanEntities:
    """Pin: export(entity_rows=...) → plan in full-directory mode emits zero DROP_ENTITY."""

    def test_orphan_entity_round_trips_to_no_change_in_full_directory_mode(self, tmp_path: Path) -> None:
        """An orphan entity tag round-trips cleanly when entity_rows is forwarded."""
        applied_state, entity_rows = _orphan_applied_state(
            _STREAMING_SPEC,
            orphan_entity_names=["SESSION_ID_TEST3", "SESSION_ID_TEST_NEW"],
        )

        oft_name = (
            f"{_STREAMING_SPEC['metadata']['name'].upper()}" f"${_STREAMING_SPEC['metadata']['version'].upper()}$ONLINE"
        )
        decl_api.export_specs(
            show_rows=[
                _show_row(
                    _STREAMING_SPEC["metadata"]["name"],
                    _STREAMING_SPEC["metadata"]["version"],
                )
            ],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={oft_name: _STREAMING_SPEC},
            entity_rows=entity_rows,
        )

        export_root = tmp_path / "MYDB.PUBLIC"
        # Orphan YAMLs must exist on disk after export — that is the
        # entire point of forwarding entity_rows.
        for orphan in ("SESSION_ID_TEST3", "SESSION_ID_TEST_NEW"):
            assert (export_root / "entities" / f"{orphan}.yaml").exists(), (
                f"orphan entity '{orphan}' was not exported; " "entity_rows is not driving emission"
            )

        batch = load_specs([f"{export_root}/..."])
        errors, plan = _validate_and_plan(
            batch,
            applied_state,
            PlanOptions(full_directory_mode=True),
            database="MYDB",
            schema="PUBLIC",
        )

        assert errors == [], f"validation_failed on orphan round-trip; errors={errors!r}"

        drop_ops = [op for op in plan.ops if op.kind.value.startswith("DROP_")]
        assert drop_ops == [], (
            "full-directory plan must emit ZERO DROP ops after an unmodified "
            f"export, but got: {drop_ops!r}.  This indicates the exporter "
            "is missing entities the planner sees in applied_state."
        )

        non_no_change_ops = [op for op in plan.ops if op.kind.value != "NO_CHANGE"]
        assert non_no_change_ops == [], (
            "Every op must be NO_CHANGE on a clean export ↔ plan round-trip "
            f"with full_directory_mode=True; got non-NO_CHANGE ops: {non_no_change_ops!r}"
        )

    def test_entity_only_schema_round_trips_to_no_change(self, tmp_path: Path) -> None:
        """A schema with only entities (no FVs) must export and plan as NO_CHANGE."""
        # No FVs at all — only registered entity tags.  Pre-fix the
        # exporter early-returned for show_rows=[] and emitted nothing,
        # so plan would see DROP_ENTITY for every tag.
        entity_rows = [
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_LONELY_KEY",
                "database_name": "MYDB",
                "schema_name": "PUBLIC",
                "allowed_values": '["LONELY_KEY"]',
                "comment": "registered without any FV referencing it",
            }
        ]
        applied_state = fetch_applied_state(
            [],
            None,
            specification_map={},
            entity_rows=entity_rows,
            default_database="MYDB",
            default_schema="PUBLIC",
        )

        decl_api.export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=entity_rows,
        )

        export_root = tmp_path / "MYDB.PUBLIC"
        assert (export_root / "entities" / "LONELY_KEY.yaml").exists()

        batch = load_specs([f"{export_root}/..."])
        errors, plan = _validate_and_plan(
            batch,
            applied_state,
            PlanOptions(full_directory_mode=True),
            database="MYDB",
            schema="PUBLIC",
        )

        assert errors == [], f"validation_failed on entity-only round-trip; errors={errors!r}"

        non_no_change_ops = [op for op in plan.ops if op.kind.value != "NO_CHANGE"]
        assert (
            non_no_change_ops == []
        ), f"entity-only schema must round-trip to all-NO_CHANGE; got {non_no_change_ops!r}"


# ---------------------------------------------------------------------------
# Phase 7 — BatchSource ``query`` → ``.sql`` sidecar emission
# ---------------------------------------------------------------------------
#
# Mirrors :func:`exporter._extract_udf_to_py_file`: when a recovered
# datasource doc carries a non-empty ``query``, the body is written to a
# sibling ``<source_name>.sql`` file and the YAML is rewritten to point at
# it via ``query_file:``.  Compiler.inline_query_source reverses the move
# on load, so the round-trip is byte-stable for ``query`` modulo whitespace
# normalisation.


class TestExtractQueryToSqlFile:
    """Unit tests for the standalone ``_extract_query_to_sql_file`` helper.

    Mirrors :class:`TestExtractUdfToPyFile` (would-be) — the helper mutates
    the datasource doc in place, swapping the inline ``query`` for a
    ``query_file`` filename pointer.
    """

    def test_extracts_query_to_sidecar_and_rewrites_doc(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import _extract_query_to_sql_file

        ds_doc: dict[str, Any] = {
            "kind": "BatchSource",
            "name": "EVENTS",
            "query": "SELECT user_id, event_ts FROM RAW.EVENTS",
        }
        out_path = _extract_query_to_sql_file(ds_doc, tmp_path, "EVENTS")
        from pathlib import Path

        assert out_path is not None
        assert Path(out_path).is_file()
        assert Path(out_path).name == "EVENTS.sql"
        assert Path(out_path).read_text() == "SELECT user_id, event_ts FROM RAW.EVENTS"
        assert "query" not in ds_doc
        assert ds_doc["query_file"] == "EVENTS.sql"

    def test_no_op_when_query_missing(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import _extract_query_to_sql_file

        ds_doc: dict[str, Any] = {"kind": "BatchSource", "name": "EVENTS", "table": "RAW.EVENTS"}
        out_path = _extract_query_to_sql_file(ds_doc, tmp_path, "EVENTS")
        assert out_path is None
        assert "query_file" not in ds_doc
        assert ds_doc["table"] == "RAW.EVENTS"

    def test_no_op_when_query_empty_string(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import _extract_query_to_sql_file

        ds_doc: dict[str, Any] = {"kind": "BatchSource", "name": "EVENTS", "query": ""}
        out_path = _extract_query_to_sql_file(ds_doc, tmp_path, "EVENTS")
        assert out_path is None
        assert "query_file" not in ds_doc
        assert ds_doc["query"] == ""

    def test_no_op_when_query_non_string(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import _extract_query_to_sql_file

        ds_doc: dict[str, Any] = {"kind": "BatchSource", "name": "EVENTS", "query": ["SELECT 1"]}
        out_path = _extract_query_to_sql_file(ds_doc, tmp_path, "EVENTS")
        assert out_path is None
        assert "query_file" not in ds_doc

    def test_long_query_always_sidecared(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import _extract_query_to_sql_file

        long_query = (
            "SELECT a.user_id, a.event_ts, b.session_id, b.country, "
            "COUNT(*) OVER (PARTITION BY a.user_id ORDER BY a.event_ts ROWS "
            "BETWEEN 99 PRECEDING AND CURRENT ROW) AS rolling_event_count "
            "FROM raw.events a JOIN raw.sessions b USING (session_id) "
            "WHERE a.event_ts > DATEADD('day', -30, CURRENT_TIMESTAMP())"
        )
        ds_doc: dict[str, Any] = {"kind": "BatchSource", "name": "EVENTS_LONG", "query": long_query}
        out_path = _extract_query_to_sql_file(ds_doc, tmp_path, "EVENTS_LONG")
        from pathlib import Path

        assert out_path is not None
        assert Path(out_path).read_text() == long_query
        assert ds_doc.get("query_file") == "EVENTS_LONG.sql"


# ---------------------------------------------------------------------------
# End-to-end exporter behaviour for query-backed BatchSources.  Uses the
# layout="sources" path so the resulting tree drops straight into the
# manifest project structure consumed by ``snow feature plan / apply``.
# ---------------------------------------------------------------------------


def _query_show_row() -> dict[str, Any]:
    return {
        "name": "FV_QUERY$V1",
        "database_name": "DB1",
        "schema_name": "SCH1",
        "scheduling_state": "ACTIVE",
    }


def _query_spec_payload() -> dict[str, Any]:
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "FV_QUERY",
            "version": "V1",
            "database": "DB1",
            "schema": "SCH1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "target_lag_sec": 3600,
            "sources": [
                {
                    "name": "FV_QUERY__SOURCE",
                    "source_type": "Batch",
                    "query": "SELECT user_id, event_ts, amount FROM raw.events WHERE amount > 0",
                    "columns": [
                        {"name": "user_id", "type": "StringType"},
                        {"name": "event_ts", "type": "TimestampType"},
                        {"name": "amount", "type": "FloatType"},
                    ],
                }
            ],
            "features": [],
        },
    }


def _query_entity_row() -> dict[str, Any]:
    return {
        "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
        "database_name": "DB1",
        "schema_name": "SCH1",
        "allowed_values": '["USER_ID"]',
        "comment": "",
    }


class TestExportSpecsQueryBackedDatasource:
    """End-to-end exporter: a query-backed FV emits a ``query_file:`` sidecar."""

    def test_query_backed_fv_exports_query_file_sidecar(self, tmp_path: Path) -> None:
        from pathlib import Path

        import yaml as _yaml

        result = export_specs(
            show_rows=[_query_show_row()],
            describe_rows_by_oft={"FV_QUERY$V1": []},
            output_dir=str(tmp_path),
            database="DB1",
            schema="SCH1",
            specification_map={"FV_QUERY$V1": _query_spec_payload()},
            entity_rows=[_query_entity_row()],
            layout="sources",
        )

        assert result["status"] == "exported"
        sources_dir = Path(result["directory"])
        ds_yaml = sources_dir / "datasources" / "FV_QUERY__SOURCE.yaml"
        ds_sql = sources_dir / "datasources" / "FV_QUERY__SOURCE.sql"
        assert ds_yaml.is_file(), f"datasource YAML not written; created files: {result['files']}"
        assert ds_sql.is_file(), f"sidecar .sql not written; created files: {result['files']}"

        ds_doc = _yaml.safe_load(ds_yaml.read_text())
        assert ds_doc["kind"] == "BatchSource"
        assert ds_doc["name"] == "FV_QUERY__SOURCE"
        assert ds_doc.get("query_file") == "FV_QUERY__SOURCE.sql"
        assert "query" not in ds_doc, "inline query: must not appear once sidecared"

        sql_body = ds_sql.read_text()
        assert sql_body == "SELECT user_id, event_ts, amount FROM raw.events WHERE amount > 0"

        all_paths = {Path(p) for p in result["files"]}
        assert ds_yaml in all_paths
        assert ds_sql in all_paths

    def test_table_backed_fv_does_not_emit_sql_sidecar(self, tmp_path: Path) -> None:
        from pathlib import Path

        import yaml as _yaml

        spec_payload = _query_spec_payload()
        spec_payload["spec"]["sources"] = [
            {
                "name": "RAW_EVENTS",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "event_ts", "type": "TimestampType"},
                ],
            }
        ]

        result = export_specs(
            show_rows=[_query_show_row()],
            describe_rows_by_oft={"FV_QUERY$V1": []},
            output_dir=str(tmp_path),
            database="DB1",
            schema="SCH1",
            specification_map={"FV_QUERY$V1": spec_payload},
            entity_rows=[_query_entity_row()],
            layout="sources",
        )

        sources_dir = Path(result["directory"])
        ds_yaml = sources_dir / "datasources" / "RAW_EVENTS.yaml"
        ds_sql = sources_dir / "datasources" / "RAW_EVENTS.sql"
        assert ds_yaml.is_file()
        assert not ds_sql.exists(), "table-backed source must not produce a .sql sidecar"
        ds_doc = _yaml.safe_load(ds_yaml.read_text())
        assert ds_doc.get("table") == "RAW_EVENTS"
        assert "query" not in ds_doc
        assert "query_file" not in ds_doc


class TestExporterRoundTripThroughLoader:
    """Round-trip integration: exporter → loader → compiler.

    Exports a query-backed FV, re-loads the on-disk tree through
    :func:`loader.load_from_project`, and asserts the resulting
    ``BatchSource`` spec carries the inlined, whitespace-normalised
    ``query`` (i.e. the compiler reversed the sidecar).
    """

    def test_query_sidecar_round_trips_through_loader(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        # Bootstrap a manifest so load_from_project finds the project root.
        (tmp_path / "manifest.yml").write_text("kind: Manifest\nname: roundtrip\ndatabase: DB1\nschema: SCH1\n")

        export_specs(
            show_rows=[_query_show_row()],
            describe_rows_by_oft={"FV_QUERY$V1": []},
            output_dir=str(tmp_path),
            database="DB1",
            schema="SCH1",
            specification_map={"FV_QUERY$V1": _query_spec_payload()},
            entity_rows=[_query_entity_row()],
            layout="sources",
        )

        batch = load_from_project(tmp_path, database="DB1", schema="SCH1")

        ds_specs = [s for s in batch.specs if s.kind == "BatchSource"]
        assert len(ds_specs) == 1, (
            "exporter+loader round-trip should surface exactly one BatchSource spec; "
            f"got {len(ds_specs)} ({[s.name for s in ds_specs]!r})"
        )
        ds: Any = ds_specs[0]
        assert ds.name == "FV_QUERY__SOURCE"
        assert ds.query_file is None, (
            "compiler.inline_query_source must inline the sidecar so the recovered "
            "BatchSource carries query directly"
        )
        assert ds.table is None
        assert ds.query == "SELECT user_id, event_ts, amount FROM raw.events WHERE amount > 0"


# ---------------------------------------------------------------------------
# BFV source-name recovery: prefer the local logical BatchSource.name over
# the recovered physical table when re-exporting from the deployed runtime.
# ---------------------------------------------------------------------------
#
# Repro of the operator-visible bug from
# .cursor/plans/bfv_source-name_recovery_82070393.plan.md:
#
# - Operator authors a BatchSource YAML  ``name: EVENTS_FG_DECL, table:
#   RAW_EVENTS_FG_DECL`` and a BFV YAML whose ``sources[0].name`` points
#   at the logical source ``EVENTS_FG_DECL``.
# - After deploy + export + re-plan, the recovered FV YAML ends up with
#   ``sources[0].name = RAW_EVENTS_FG_DECL`` (the physical table) and
#   the next ``snow feature plan`` trips ``MISSING_SOURCE`` because no
#   BatchSource with that name exists.
#
# Root cause: the wire SPECIFICATION JSON drops the authored source
# name (the imperative spec builder hardcodes ``name="batch"``), so the
# decl-side recovery in :func:`state._inject_batch_fv_source_from_dt_text`
# defaults to ``name = <recovered_table>``.  The fix threads a
# ``datasources_by_table`` lookup (physical → logical name) built from
# the local ``sources/datasources/`` tree into the recovery helper.
# ---------------------------------------------------------------------------


def _bfv_with_logical_source_spec() -> dict[str, Any]:
    """SPECIFICATION-shape BFV payload mirroring the bug-repro example_store FG.

    Matches the on-disk pair:
    - sources/datasources/EVENTS_FG_DECL.yaml: name=EVENTS_FG_DECL, table=RAW_EVENTS_FG_DECL
    - sources/feature_views/USER_AMOUNTS_FG_DECL.yaml: sources[0].name=EVENTS_FG_DECL

    The wire spec carries the post-deploy lossy shape ``sources=[]``
    (the imperative side hardcodes ``name="batch"`` and the snowml
    serializer drops the entire entry); the recovery helper is what
    re-injects ``sources[0]`` from the DT text.

    Returns:
        dict: a SPECIFICATION-shape BFV payload (``metadata`` +
        ``spec``) with empty ``sources`` so the recovery helper has
        something to reinject.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "USER_AMOUNTS_FG_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "0.1.0",
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "BatchTable",
                "database": "JKEW_DB",
                "schema": "JKEW_SCHEMA",
                "table": "USER_AMOUNTS_FG_DECL$V1",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                ],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "target_lag_sec": 300,
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        },
    }


class TestBfvSourceNameRecoveryFromLocalDatasources:
    """The recovered BFV ``sources[0].name`` MUST equal the operator's
    authored ``BatchSource.name`` on a clean round-trip.

    Phase B3 promoted ``FV_SOURCE_REFS`` to the authoritative source for
    the logical name (no more physical-table lookup): the operator's
    authored ``SourceRef`` rides on the metadata row and the recovery
    side reads it verbatim.  The legacy
    ``_build_datasources_by_table`` shim is preserved for FVs that
    pre-date the metadata row, but it no longer drives the typical
    recovery path — the metadata payload does.
    """

    def test_recovered_source_uses_local_logical_name_with_lookup(self) -> None:
        """Phase B3 metadata-driven recovery: the operator-authored
        logical name (``EVENTS_FG_DECL``) survives the export round-trip
        because it rides on ``FV_SOURCE_REFS`` directly, not on a
        physical-table lookup.

        Migrated from the legacy
        ``_inject_batch_fv_source_from_dt_text`` shape (deleted in B1)
        to the new
        :func:`state._inject_batch_fv_source_from_metadata` contract.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_source_from_metadata,
        )

        spec_payload = _bfv_with_logical_source_spec()
        # The metadata row carries the operator-authored ``SourceRef``
        # — logical name + physical table + columns — so the recovery
        # side does not need a physical → logical lookup.
        source_refs = [
            {
                "name": "EVENTS_FG_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_FG_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ]
        injected = _inject_batch_fv_source_from_metadata(spec_payload, source_refs)
        assert injected is True

        sources = spec_payload["spec"]["sources"]
        assert len(sources) == 1
        assert sources[0]["name"] == "EVENTS_FG_DECL", (
            "post-Phase-B3: the operator's authored logical name must "
            "survive the export round-trip via authoritative FV_SOURCE_REFS "
            f"metadata; got {sources[0]['name']!r}"
        )
        assert sources[0]["table"] == "RAW_EVENTS_FG_DECL"
        # Columns ride through verbatim (Bug 1 closure — no symmetry shim).
        col_names = [c["name"] for c in sources[0].get("columns") or []]
        assert col_names == ["USER_ID", "EVENT_TS", "AMOUNT"]

    def test_full_round_trip_against_local_datasources_tree_yields_no_missing_source(self, tmp_path: Path) -> None:
        """End-to-end: a synthetic project tree carrying the operator's
        local ``EVENTS_FG_DECL`` BatchSource + the deployed-runtime
        side's BFV state must plan to zero ``MISSING_SOURCE`` errors
        when ``fetch_applied_state`` is given the same
        ``datasources_by_table`` map.

        This is the operator-visible bug pin: pre-fix the planner
        raised ``MISSING_SOURCE: USER_AMOUNTS_FG_DECL references source
        'RAW_EVENTS_FG_DECL' but no source with that name exists`` on
        every re-plan after an init export.

        Args:
            tmp_path: pytest-provided temporary directory used as the
                synthetic project root for the load/plan cycle.
        """
        from snowflake.ml.feature_store.decl.loader import load_from_project

        # 1. Operator-authored project tree on disk: the logical
        #    BatchSource YAML pointing at the physical table, plus the
        #    BFV YAML that refers to the logical source name.
        (tmp_path / "manifest.yml").write_text("kind: Manifest\nname: rt\ndatabase: JKEW_DB\nschema: JKEW_SCHEMA\n")
        sources_root = tmp_path / "sources"
        for sub in ("entities", "datasources", "feature_views", "feature_groups"):
            (sources_root / sub).mkdir(parents=True, exist_ok=True)

        (sources_root / "entities" / "USER_ID.yaml").write_text(
            "kind: Entity\nname: USER_ID\n" "join_keys:\n" "  - name: USER_ID\n    type: StringType\n"
        )
        (sources_root / "datasources" / "EVENTS_FG_DECL.yaml").write_text(
            "kind: BatchSource\n"
            "name: EVENTS_FG_DECL\n"
            "table: RAW_EVENTS_FG_DECL\n"
            "columns:\n"
            "  - name: USER_ID\n    type: StringType\n"
            "  - name: EVENT_TS\n    type: TimestampType\n"
            "  - name: AMOUNT\n    type: DoubleType\n"
        )
        (sources_root / "feature_views" / "USER_AMOUNTS_FG_DECL.yaml").write_text(
            "kind: BatchFeatureView\n"
            "name: USER_AMOUNTS_FG_DECL\n"
            "version: V1\n"
            "database: JKEW_DB\n"
            "schema: JKEW_SCHEMA\n"
            "online: true\n"
            "entities:\n  - USER_ID\n"
            "timestamp_col: EVENT_TS\n"
            "feature_granularity_sec: 3600\n"
            "feature_aggregation_method: tiles\n"
            "refresh_freq: 300 seconds\n"
            "sources:\n"
            "  - name: EVENTS_FG_DECL\n"
            "    source_type: Batch\n"
            "features:\n"
            "  - output_column:\n      name: AMOUNT_SUM_1H\n      type: DoubleType\n"
            "    window_sec: 3600\n"
            "    function: sum\n"
            "    source_column:\n      name: AMOUNT\n      type: DoubleType\n"
        )

        # 2. Load the local project — this is the same call the CLI
        #    manager makes; the resulting batch is what the helper
        #    walks to build ``datasources_by_table``.
        batch = load_from_project(
            tmp_path,
            database="JKEW_DB",
            schema="JKEW_SCHEMA",
        )
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        lookup = _build_datasources_by_table(batch.specs)
        # Sanity: the locally-declared BatchSource produces the
        # expected physical → logical mapping.
        assert lookup == {"RAW_EVENTS_FG_DECL": "EVENTS_FG_DECL"}

        # 3. Build the deployed-runtime side: the lossy SPECIFICATION
        #    JSON plus the DT body that carries the source binding.
        spec_payload = _bfv_with_logical_source_spec()
        oft_name = "USER_AMOUNTS_FG_DECL$V1$ONLINE"
        dt_name = "USER_AMOUNTS_FG_DECL$V1"
        dt_text = (
            f"CREATE DYNAMIC TABLE JKEW_DB.JKEW_SCHEMA.{dt_name} "
            "lag = '300 seconds' "
            "AS SELECT * FROM JKEW_DB.JKEW_SCHEMA.RAW_EVENTS_FG_DECL"
        )

        # 4. Build the AppliedState with the lookup threaded in — the
        #    decl recovery layer now prefers the local logical name.
        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[
                {
                    "name": oft_name,
                    "database_name": "JKEW_DB",
                    "schema_name": "JKEW_SCHEMA",
                    "scheduling_state": "ACTIVE",
                }
            ],
            specification_map={oft_name: spec_payload},
            dt_text_map={dt_name: dt_text},
            entity_rows=[
                {
                    "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
                    "database_name": "JKEW_DB",
                    "schema_name": "JKEW_SCHEMA",
                    "allowed_values": '["USER_ID"]',
                }
            ],
            datasources_by_table=lookup,
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        # 5. The planner's full validator must NOT raise MISSING_SOURCE
        #    when the operator-authored logical name was preserved
        #    through the round-trip.
        decl_api.resolve_datasource_columns(batch)
        results = decl_api.validate_specs(
            batch,
            applied_state,
            target_database="JKEW_DB",
            target_schema="JKEW_SCHEMA",
        )
        missing_source = [r for r in results if r.code == "MISSING_SOURCE"]
        assert missing_source == [], (
            "Post-fix: zero MISSING_SOURCE errors when the operator's "
            "local BatchSource.name is preserved through the export "
            f"round-trip; got {missing_source!r}"
        )


# ---------------------------------------------------------------------------
# Phase C1 — Five LIMITATION fixtures pinned through the live planner
# ---------------------------------------------------------------------------
#
# One parametric scenario per documented L1–L5 case from
# ``docs/LIMITATIONS.md`` (cross-referenced in
# ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
# §"Phase C → C1").  Each scenario mirrors the live operator
# walkthrough:
#
#   1. Build the "previously-applied" authoring dict for the FV
#      (the YAML the user already deployed) and compile it to the
#      SPECIFICATION-shape ``spec_payload`` the runtime would return.
#   2. Strip ``spec.sources = []`` to model the lossy ``DESCRIBE …
#      TYPE = SPECIFICATION`` round-trip then populate them from a
#      ``FV_SOURCE_REFS`` payload via
#      :func:`state._inject_batch_fv_source_from_metadata`
#      (Phase B3 contract — sources are authoritative from metadata,
#      not from DT-text parsing).
#   3. Hash the resulting spec_payload via :func:`_full_spec_hash`
#      so the planner's ``current_hash == applied.content_hash``
#      branch fires (any operational drift then routes to UPDATE_FV
#      via the kind-agnostic operational helper).
#   4. Layer the operational knob that diverges between local and
#      applied (warehouse / desc / backfill) POST-hash so the
#      structural hash stays stable while the drift detector still
#      observes the divergence.
#   5. Build the local SpecBatch via :meth:`FeatureView.model_validate`
#      and run the live planner (``decl_api.resolve_datasource_columns``
#      → ``decl_api.validate_specs`` → ``decl_api.generate_plan``).
#   6. Assert the FV op kind AND the payload field carrying the
#      operator's intent (so a future refactor that flips the op
#      kind without propagating the field surfaces here).
#
# These fixtures are R6's safety net: Phase B's normalisation surface
# is the production fix, but if a future "quick fix" widens the
# canonicalizer in a way that hides a real edit (or narrows it in a
# way that re-introduces phantom UPDATE_FV / RECREATE_FV), one of
# these parametric scenarios fails.
# ---------------------------------------------------------------------------


def _limitation_applied_state(
    applied_authoring: dict[str, Any],
    source_refs: list[dict[str, Any]],
    *,
    database: str = "DB1",
    schema: str = "SC1",
    post_hash_inner_overrides: dict[str, Any] | None = None,
    post_hash_top_overrides: dict[str, Any] | None = None,
) -> object:
    """Build an ``AppliedState`` for a single FV via the Phase-B contract.

    Mirrors what :func:`state.fetch_applied_state` would produce for
    *applied_authoring* once the runtime side has returned its
    ``DESCRIBE … TYPE = SPECIFICATION`` payload plus the
    ``list_feature_views().source_refs`` metadata column:

    1. ``compile_to_spec(applied_authoring)`` produces the SPECIFICATION
       JSON the planner uses as the applied side ``spec_payload``.
    2. The inner ``spec.sources`` list is cleared and re-populated via
       :func:`state._inject_batch_fv_source_from_metadata` so the
       FV_SOURCE_REFS authoritative path (Phase B3) is exercised end-to-
       end on every fixture.
    3. The ``content_hash`` is computed by :func:`_full_spec_hash`
       BEFORE layering any post-hash overrides — mirrors the
       hash-stability contract enforced by the existing
       ``TestBatchOperationalDrift`` / ``TestStreamingOperationalDrift``
       suites in :mod:`tests.test_planner`.
    4. Post-hash overrides land on the inner ``spec`` (operational
       fields the drift detector reads via ``applied.spec_payload``)
       and / or on the top-level spec_payload (e.g. ``desc`` from
       ``list_feature_views().desc``) so the planner observes the
       operational divergence without bumping the structural hash.

    Args:
        applied_authoring: Authoring-format FV dict that mirrors the
            previously-applied YAML.  Drives the SPECIFICATION-shape
            compile and the structural hash.
        source_refs: ``FV_SOURCE_REFS``-shape list of dicts (each with
            ``name`` / ``source_type`` / optional ``table`` / ``query`` /
            ``columns``).  Threaded through
            :func:`state._inject_batch_fv_source_from_metadata`.
        database: Snowflake database used for the SPECIFICATION metadata
            block and the AppliedObject key qualifier.
        schema: Snowflake schema used for the SPECIFICATION metadata
            block and the AppliedObject key qualifier.
        post_hash_inner_overrides: Optional ``{field: value}`` mapping
            written into ``spec_payload["spec"]`` AFTER the
            ``content_hash`` is computed.  Use this for operational
            knobs the drift detector reads from the inner spec
            (``warehouse``, etc.) — applying them post-hash mirrors the
            runtime-stamping contract (the hash uses the stripped form
            while the drift helper reads the unstripped value).
        post_hash_top_overrides: Same as *post_hash_inner_overrides* but
            written into the top-level ``spec_payload`` (e.g. ``desc``
            from the ``list_feature_views()`` row).

    Returns:
        object: An :class:`AppliedState` carrying exactly one FV
        :class:`AppliedObject` whose ``content_hash`` matches a
        ``compile_to_spec(local)`` hash with the same baseline shape.
    """
    from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_source_from_metadata,
    )
    from snowflake.ml.feature_store.decl.types import AppliedObject, AppliedState

    compile_input = dict(applied_authoring)
    if "schema_" in compile_input:
        compile_input["schema"] = compile_input.pop("schema_")
    compiled = compile_to_spec(compile_input, database, schema)

    # Strip the inner ``sources`` and replay the Phase-B3 metadata
    # path so the recovered ``spec.sources`` carries operator-authored
    # names + columns verbatim (no synthetic ``<FV>__SOURCE``).
    inner = compiled.get("spec")
    if isinstance(inner, dict):
        inner["sources"] = []
    injected = _inject_batch_fv_source_from_metadata(compiled, source_refs)
    assert injected, (
        "fixture precondition: source_refs must yield at least one valid entry "
        "so the Phase-B3 metadata-injection contract is exercised end-to-end"
    )

    content_hash = _full_spec_hash(compiled)

    if post_hash_inner_overrides:
        compiled["spec"].update(post_hash_inner_overrides)
    if post_hash_top_overrides:
        compiled.update(post_hash_top_overrides)

    kind = applied_authoring["kind"]
    name = applied_authoring["name"]
    version = applied_authoring["version"]
    key = f"{kind}:{database.upper()}.{schema.upper()}:{name.upper()}:{version.upper()}"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind=kind,
                name=name,
                version=version,
                content_hash=content_hash,
                spec_payload=compiled,
                columns=[],
                from_specification=True,
            )
        }
    )


def _limitation_spec_batch(local_authoring: dict[str, Any], *, source_kind: str = "BatchSource") -> Any:
    """Build a SpecBatch for a single FV with its declared Entity + Source.

    The Entity / Source dependencies are required so
    :func:`decl_api.validate_specs` resolves the FV's references without
    surfacing ERROR severities.  Each source carries inline columns so
    :func:`decl_api.resolve_datasource_columns` is a no-op on the FV
    spec (the source binding rides on the local YAML rather than being
    threaded through a name-only reference).

    Args:
        local_authoring: Authoring-format FV dict.  ``sources[0]`` drives
            the BatchSource / StreamingSource spec emitted into the
            batch.
        source_kind: ``"BatchSource"`` (default) or ``"StreamingSource"``
            — selects which Pydantic model is used for the dependency
            spec.

    Returns:
        A ``SpecBatch`` carrying the FV plus its dependencies.
    """
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
        FSColumn,
        StreamingSource,
    )

    src0 = local_authoring["sources"][0]
    entity_name = local_authoring["entities"][0]
    entity = Entity(
        kind="Entity",
        name=entity_name,
        join_keys=[FSColumn(name=entity_name, type="StringType")],
    )
    src_kwargs = {
        "kind": source_kind,
        "name": src0["name"],
        "columns": src0.get("columns", []),
    }
    if "table" in src0:
        src_kwargs["table"] = src0["table"]
    if source_kind == "BatchSource":
        # BatchSource requires either ``table`` or ``query``; default
        # to a synthetic table when neither is set so the validator's
        # ``BATCH_FV_SOURCE_NO_TABLE`` rule doesn't fire on fixtures
        # that intentionally exercise the FV's source-name-only
        # reference shape.
        if "table" not in src_kwargs and "query" not in src0:
            src_kwargs["table"] = f"{src0['name']}_TBL"
        source_spec: Any = BatchSource(**src_kwargs)
    else:
        # StreamingSource doesn't carry table/query; keep the source
        # surface minimal (name + columns) so the L5 streaming fixture
        # exercises the Phase-B3 metadata path without dragging in
        # batch-specific knobs.
        src_kwargs.pop("table", None)
        source_spec = StreamingSource(
            kind="StreamingSource",
            name=src0["name"],
            columns=src0.get("columns", []),
        )
    fv = FeatureView.model_validate(local_authoring)
    from snowflake.ml.feature_store.decl.types import SpecBatch

    return SpecBatch(specs=[entity, source_spec, fv])


# Common dimensions shared across all fixtures.  Centralised so a
# refactor lands in one place rather than five.
_C1_DB = "DB1"
_C1_SCH = "SC1"


def _c1_user_id_source_ref(*, name: str, source_type: str = "Batch", table: str | None = None) -> list[dict[str, Any]]:
    """Build the canonical FV_SOURCE_REFS payload for the C1 fixtures.

    Keeps the column shape uniform across fixtures (USER_ID +
    EVENT_TS + AMOUNT) so the Phase-B3 injection contract is
    exercised against a consistent column inventory and a single
    typo wouldn't cascade through five separate fixtures.

    Args:
        name: Operator-authored ``BatchSource`` / ``StreamingSource``
            name — preserved verbatim through the metadata round-trip.
        source_type: ``"Batch"`` (default) or ``"Stream"``.  Matches
            ``SourceRef.source_type`` on the metadata write path.
        table: Optional physical table identifier.  Omitted for
            Stream source_refs since :class:`StreamingSource` carries
            no ``table`` attribute.

    Returns:
        A single-entry list of source-ref dicts.
    """
    entry: dict[str, Any] = {
        "name": name,
        "source_type": source_type,
        "columns": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
            {"name": "AMOUNT", "type": "DoubleType"},
        ],
    }
    if table:
        entry["table"] = table
    return [entry]


# ---------------------------------------------------------------------------
# Fixture 1 — L1: tiled BatchFV warehouse flip → UPDATE_FV
# ---------------------------------------------------------------------------
#
# Pre-Phase-A a tiled BFV with an ``aggregation_secondary_keys`` value
# round-tripped lossily through ``DESCRIBE … TYPE = SPECIFICATION``;
# any warehouse-only edit fell through to ``RECREATE_FV`` because the
# structural-equivalence pass saw a phantom diff.  Post Phase-A2 +
# B-invariants the field is in ``_BATCH_FV_STRUCTURAL_INNER_KEYS`` and
# the operational-drift fast path correctly routes the warehouse edit
# to UPDATE_FV.


def _c1_tiled_bfv_authoring(*, warehouse: str) -> dict[str, Any]:
    """Build the tiled BFV authoring dict for the L1 warehouse-flip case.

    The aggregation + secondary-key surface is held constant across
    the local + applied baselines so the only divergence the planner
    can observe is the warehouse value.  Single-element
    ``aggregation_secondary_keys`` respects the private-preview cap
    enforced by
    :func:`invariants._check_batch_feature_view_constraints`.

    Args:
        warehouse: Warehouse identifier the YAML authors.  ``"NEW_WH"``
            on the local side, ``"OLD_WH"`` on the applied baseline.

    Returns:
        Authoring-format BatchFV dict ready for
        :meth:`FeatureView.model_validate`.
    """
    return {
        "kind": "BatchFeatureView",
        "name": "TILED_BFV_L1",
        "version": "V1",
        "database": _C1_DB,
        "schema_": _C1_SCH,
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
        "refresh_freq": "5 minutes",
        "aggregation_secondary_keys": ["EVENT_TS"],
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                "function": "sum",
                "window_sec": 3600,
            }
        ],
        "warehouse": warehouse,
    }


# ---------------------------------------------------------------------------
# Fixture 2 — L2: BFV with FV-level ``backfill:`` block → NO_CHANGE
# ---------------------------------------------------------------------------
#
# Per ``docs/LIMITATIONS.md`` §"FV-level ``backfill:`` block —
# operational, non-recovered, no ``backfill_end_time``" the
# ``backfill`` block is hash-invariant (stripped via
# ``_OPERATIONAL_FV_KEYS``).  Holding the block constant on both
# halves of the diff therefore reduces to NO_CHANGE.  The streaming-
# shape ``backfill.table`` / ``backfill.start_time`` fields are
# rejected on BatchFV by
# :func:`FeatureView._validate_backfill_against_kind`, so the BFV
# fixture uses ``backfill.initialize`` — the only BFV-valid subfield
# that exercises the round-trip contract end-to-end without tripping
# the kind validator.  The streaming counterpart of the
# ``backfill.table`` / ``backfill.start_time`` round-trip ships in
# the live verify-roundtrip script (Stream C-verify's territory).


def _c1_bfv_with_backfill_authoring() -> dict[str, Any]:
    """Authoring-format BFV carrying a BFV-valid ``backfill:`` block.

    ``backfill.initialize: ON_CREATE`` is the only BatchFV-valid
    backfill subfield in the current scope of the plan
    (``backfill.table`` / ``backfill.start_time`` are streaming-only
    and rejected by the model validator).  It exercises the L2
    contract: the block round-trips cleanly on the structural hash
    (stripped via :data:`_OPERATIONAL_FV_KEYS`) and the planner
    reports ``NO_CHANGE`` when local + applied carry the same block.

    Returns:
        Authoring-format BatchFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    return {
        "kind": "BatchFeatureView",
        "name": "BFV_BACKFILL_L2",
        "version": "V1",
        "database": _C1_DB,
        "schema_": _C1_SCH,
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "refresh_freq": "5 minutes",
        "backfill": {"initialize": "ON_CREATE"},
    }


# ---------------------------------------------------------------------------
# Fixture 3 — L3: BFV description-only edit → UPDATE_FV
# ---------------------------------------------------------------------------
#
# Post Phase-B6 the planner's kind-agnostic operational-drift helper
# folds ``description`` (local) ↔ ``desc`` (applied, from
# ``list_feature_views().desc``) into the UPDATE_FV decision.  The
# structural hash is unchanged on the edit because
# :func:`spec_compiler.compile_to_spec` does not emit ``description``
# into the SPECIFICATION JSON.


def _c1_bfv_with_desc_authoring(*, description: str) -> dict[str, Any]:
    """Authoring-format BFV for the L3 description-edit case.

    Args:
        description: FV-level ``description:`` text the YAML authors.
            ``"v2 docs"`` on the local side, ``"v1 docs"`` on the
            applied baseline.

    Returns:
        Authoring-format BatchFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    return {
        "kind": "BatchFeatureView",
        "name": "BFV_DESC_L3",
        "version": "V1",
        "database": _C1_DB,
        "schema_": _C1_SCH,
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "refresh_freq": "5 minutes",
        "description": description,
    }


# ---------------------------------------------------------------------------
# Fixture 4 — L4: online BFV refresh-frequency edit → UPDATE_FV
# ---------------------------------------------------------------------------
#
# Local authors ``refresh_freq: 30 minutes``; the applied baseline
# was deployed with ``refresh_freq: 1 hour`` (compile_to_spec
# normalises this into ``spec.target_lag_sec: 3600`` on the applied
# payload).  ``target_lag_sec`` lives in
# :data:`_RUNTIME_STAMPED_SPEC_KEYS` and ``refresh_freq`` in
# :data:`_OPERATIONAL_FV_KEYS`, so :func:`_full_spec_hash` strips
# both before hashing — the structural hash matches and the planner
# falls into the operational-drift fast path.  The L4 closure is the
# operational/structural split that routes the edit to UPDATE_FV
# instead of RECREATE_FV.


def _c1_bfv_refresh_freq_authoring(*, refresh_freq: str) -> dict[str, Any]:
    """Authoring-format online BFV for the L4 refresh_freq edit case.

    Args:
        refresh_freq: DT refresh cadence string (e.g.
            ``"30 minutes"`` on the local side, ``"1 hour"`` on the
            applied baseline).

    Returns:
        Authoring-format BatchFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    return {
        "kind": "BatchFeatureView",
        "name": "BFV_REFRESH_L4",
        "version": "V1",
        "database": _C1_DB,
        "schema_": _C1_SCH,
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "refresh_freq": refresh_freq,
    }


# ---------------------------------------------------------------------------
# Fixture 5 — L5: StreamingFV warehouse flip → UPDATE_FV
# ---------------------------------------------------------------------------
#
# Mirrors the L1 BFV case for the streaming kind.  Phase B7 extends
# the kind-agnostic operational surface to StreamingFV
# (``_FV_OPERATIONAL_FIELDS_BY_KIND["StreamingFeatureView"]``
# includes ``warehouse``) so a warehouse-only edit must land as
# UPDATE_FV, not RECREATE_FV.


def _c1_streaming_fv_authoring(*, warehouse: str) -> dict[str, Any]:
    """Authoring-format StreamingFV for the L5 warehouse-flip case.

    Args:
        warehouse: Warehouse identifier the YAML authors.
            ``"NEW_WH"`` on the local side, ``"OLD_WH"`` on the
            applied baseline.

    Returns:
        Authoring-format StreamingFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    return {
        "kind": "StreamingFeatureView",
        "name": "STREAMING_FV_L5",
        "version": "V1",
        "database": _C1_DB,
        "schema_": _C1_SCH,
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_STREAM_DECL",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                "output_column": {"name": "AMOUNT", "type": "DoubleType"},
            }
        ],
        "udf": {
            "name": "identity",
            "engine": "pandas",
            "function_definition": "def identity(df):\n    return df\n",
            "output_columns": [{"name": "AMOUNT", "type": "DoubleType"}],
        },
        "warehouse": warehouse,
    }


class TestLimitationFixturesPlan:
    """C1 — five parametric LIMITATION fixtures pinned through the
    live ``plan(...)`` pipeline.

    One parametric scenario per documented L1–L5 case from
    ``docs/LIMITATIONS.md``.  Each fixture builds the local + applied
    sides via the Phase-B3 metadata-injection contract
    (:func:`state._inject_batch_fv_source_from_metadata`) and asserts
    the FV op kind plus the operator-intent payload field so a
    future refactor that flips the op kind without propagating the
    field surfaces here.
    """

    def _run_plan(self, batch: Any, applied_state: Any) -> tuple[Any, Any]:
        return _validate_and_plan(
            batch,
            applied_state,
            PlanOptions(),
            database=_C1_DB,
            schema=_C1_SCH,
        )

    def test_tiled_bfv_warehouse_flip_emits_update_fv(self) -> None:
        """L1: tiled BFV warehouse-only edit must emit ``UPDATE_FV``.

        Post Phase-A2 ``aggregation_secondary_keys`` lives in
        :data:`_BATCH_FV_STRUCTURAL_INNER_KEYS` so a structural
        comparison treats the secondary-key surface as identical on
        both halves.  The only divergence is the warehouse value,
        and B-invariants strips ``warehouse`` from the hash so the
        operational-drift fast path routes the edit to ``UPDATE_FV``.
        Pre-fix the planner emitted ``RECREATE_FV`` because the
        structural equivalence pass saw a phantom diff on the
        secondary-key projection.
        """
        local_authoring = _c1_tiled_bfv_authoring(warehouse="NEW_WH")
        applied_authoring = _c1_tiled_bfv_authoring(warehouse="OLD_WH")
        source_refs = _c1_user_id_source_ref(name="EVENTS_BATCH_DECL", table="RAW_EVENTS_BATCH_DECL")

        applied_state = _limitation_applied_state(
            applied_authoring,
            source_refs,
            post_hash_inner_overrides={"warehouse": "OLD_WH"},
        )
        batch = _limitation_spec_batch(local_authoring)

        errors, plan = self._run_plan(batch, applied_state)
        assert errors == [], f"validation_failed for tiled-BFV warehouse flip; errors={errors!r}"
        fv_ops = [op for op in plan.ops if op.name == local_authoring["name"]]
        assert len(fv_ops) == 1, f"expected one FV op; got {[(o.kind.value, o.name) for o in plan.ops]!r}"
        assert fv_ops[0].kind.value == "UPDATE_FV", (
            "Tiled BFV warehouse flip must emit UPDATE_FV (not RECREATE_FV) — L1 closure; "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False
        assert fv_ops[0].payload.get("warehouse") == "NEW_WH", (
            "UPDATE_FV payload must carry the operator-intent warehouse value; "
            f"got payload.warehouse={fv_ops[0].payload.get('warehouse')!r}"
        )

    def test_bfv_with_backfill_round_trips_to_no_change(self) -> None:
        """L2: a BFV carrying an unchanged ``backfill:`` block must emit
        ``NO_CHANGE`` on the round-trip.

        ``backfill`` lives in :data:`_OPERATIONAL_FV_KEYS` and is
        stripped from :func:`_full_spec_hash` so both sides hash
        identically.  The planner's destructive ``CREATE_FV`` branch
        (``backfill.overwrite=True``) does not fire because the
        fixture deliberately uses ``backfill.initialize: ON_CREATE`` —
        the only BFV-valid subfield, plus a non-destructive variant.
        """
        local_authoring = _c1_bfv_with_backfill_authoring()
        applied_authoring = _c1_bfv_with_backfill_authoring()
        source_refs = _c1_user_id_source_ref(name="EVENTS_BATCH_DECL", table="RAW_EVENTS_BATCH_DECL")

        applied_state = _limitation_applied_state(applied_authoring, source_refs)
        batch = _limitation_spec_batch(local_authoring)

        errors, plan = self._run_plan(batch, applied_state)
        assert errors == [], f"validation_failed for BFV-with-backfill round-trip; errors={errors!r}"
        fv_ops = [op for op in plan.ops if op.name == local_authoring["name"]]
        assert len(fv_ops) == 1, f"expected one FV op; got {[(o.kind.value, o.name) for o in plan.ops]!r}"
        assert fv_ops[0].kind.value == "NO_CHANGE", (
            "BFV carrying an unchanged backfill block must round-trip to NO_CHANGE (L2 closure); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False

    def test_bfv_description_edit_emits_update_fv(self) -> None:
        """L3: BFV description-only edit must emit ``UPDATE_FV`` with the
        new ``description`` text on the op payload.

        The local YAML key is ``description`` (canonical authoring
        form, per :class:`SpecBase`).  Compile-to-spec does NOT
        propagate ``description`` into the SPECIFICATION JSON, so
        the structural hash stays stable; the planner's
        operational-drift helper resolves ``description`` ↔ ``desc``
        (the applied-side key surfaced by the
        ``list_feature_views()`` row) and routes the diff to
        ``UPDATE_FV``.
        """
        local_authoring = _c1_bfv_with_desc_authoring(description="v2 docs")
        applied_authoring = _c1_bfv_with_desc_authoring(description="v1 docs")
        source_refs = _c1_user_id_source_ref(name="EVENTS_BATCH_DECL", table="RAW_EVENTS_BATCH_DECL")

        applied_state = _limitation_applied_state(
            applied_authoring,
            source_refs,
            post_hash_top_overrides={"desc": "v1 docs"},
        )
        batch = _limitation_spec_batch(local_authoring)

        errors, plan = self._run_plan(batch, applied_state)
        assert errors == [], f"validation_failed for BFV desc edit; errors={errors!r}"
        fv_ops = [op for op in plan.ops if op.name == local_authoring["name"]]
        assert len(fv_ops) == 1, f"expected one FV op; got {[(o.kind.value, o.name) for o in plan.ops]!r}"
        assert fv_ops[0].kind.value == "UPDATE_FV", (
            "BFV description-only edit must emit UPDATE_FV (L3 closure); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False
        # The authoring-side key is ``description``; the planner
        # forwards the authoring dict verbatim, so the new text must
        # land on ``op.payload["description"]``.  Accept ``desc`` as
        # a fallback in case a future refactor renames the wire form,
        # so the contract pin is about the new value flowing through,
        # not the specific key.
        payload_desc = fv_ops[0].payload.get("description") or fv_ops[0].payload.get("desc")
        assert payload_desc == "v2 docs", (
            "UPDATE_FV payload must carry the new description text; "
            f"got payload.description={fv_ops[0].payload.get('description')!r}, "
            f"payload.desc={fv_ops[0].payload.get('desc')!r}"
        )

    def test_bfv_refresh_freq_edit_emits_update_fv(self) -> None:
        """L4: online BFV refresh-cadence edit must emit ``UPDATE_FV``.

        Local sets ``refresh_freq: 30 minutes``; the applied
        baseline was deployed with ``refresh_freq: 1 hour``
        (``compile_to_spec`` normalises into
        ``spec.target_lag_sec: 3600``).  Both are stripped from the
        structural hash (``_RUNTIME_STAMPED_SPEC_KEYS`` +
        ``_OPERATIONAL_FV_KEYS``); operational-drift detection picks
        up the divergence via
        :func:`planner._refresh_freq_drifted` and routes the edit to
        ``UPDATE_FV`` — NOT ``RECREATE_FV``.  The operational
        downstream NO-OP is exercised by C-verify's live trip-wire.
        """
        local_authoring = _c1_bfv_refresh_freq_authoring(refresh_freq="30 minutes")
        applied_authoring = _c1_bfv_refresh_freq_authoring(refresh_freq="1 hour")
        source_refs = _c1_user_id_source_ref(name="EVENTS_BATCH_DECL", table="RAW_EVENTS_BATCH_DECL")

        applied_state = _limitation_applied_state(applied_authoring, source_refs)
        batch = _limitation_spec_batch(local_authoring)

        errors, plan = self._run_plan(batch, applied_state)
        assert errors == [], f"validation_failed for BFV refresh_freq edit; errors={errors!r}"
        fv_ops = [op for op in plan.ops if op.name == local_authoring["name"]]
        assert len(fv_ops) == 1, f"expected one FV op; got {[(o.kind.value, o.name) for o in plan.ops]!r}"
        assert fv_ops[0].kind.value == "UPDATE_FV", (
            "BFV refresh_freq-only edit must emit UPDATE_FV (L4 closure); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False
        # Authoring key is ``refresh_freq`` (the canonical
        # decoupled-cadence source per docs/BATCH_FV_BUG_BASH.md §5).
        # ``refresh_freq`` is the imperative-side rename and only
        # appears on ``list_feature_views()`` rows.
        assert fv_ops[0].payload.get("refresh_freq") == "30 minutes", (
            "UPDATE_FV payload must carry the new refresh_freq cadence; "
            f"got payload.refresh_freq={fv_ops[0].payload.get('refresh_freq')!r}"
        )

    def test_streaming_fv_warehouse_flip_emits_update_fv(self) -> None:
        """L5: StreamingFV warehouse-only edit must emit ``UPDATE_FV``.

        Phase B7 extended the kind-agnostic operational surface to
        StreamingFV (``_FV_OPERATIONAL_FIELDS_BY_KIND
        ["StreamingFeatureView"]`` includes ``warehouse``) so a
        warehouse-only edit routes to ``UPDATE_FV`` via
        :func:`planner._warehouse_drifted` — NOT ``RECREATE_FV``.
        ``_OPERATIONAL_FV_KEYS`` strips ``warehouse`` from the
        structural hash, so the hash-matches branch fires.
        """
        local_authoring = _c1_streaming_fv_authoring(warehouse="NEW_WH")
        applied_authoring = _c1_streaming_fv_authoring(warehouse="OLD_WH")
        source_refs = _c1_user_id_source_ref(name="EVENTS_STREAM_DECL", source_type="Stream")

        applied_state = _limitation_applied_state(
            applied_authoring,
            source_refs,
            post_hash_inner_overrides={"warehouse": "OLD_WH"},
        )
        batch = _limitation_spec_batch(local_authoring, source_kind="StreamingSource")

        errors, plan = self._run_plan(batch, applied_state)
        assert errors == [], f"validation_failed for StreamingFV warehouse flip; errors={errors!r}"
        fv_ops = [op for op in plan.ops if op.name == local_authoring["name"]]
        assert len(fv_ops) == 1, f"expected one FV op; got {[(o.kind.value, o.name) for o in plan.ops]!r}"
        assert fv_ops[0].kind.value == "UPDATE_FV", (
            "StreamingFV warehouse-only edit must emit UPDATE_FV (L5 closure); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False
        assert fv_ops[0].payload.get("warehouse") == "NEW_WH", (
            "UPDATE_FV payload must carry the operator-intent warehouse value; "
            f"got payload.warehouse={fv_ops[0].payload.get('warehouse')!r}"
        )


if __name__ == "__main__":
    pytest_driver.main()
