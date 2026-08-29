"""Planner tests for the virtual-source NO_CHANGE override.

Source specs (``BatchSource`` / ``StreamingSource``) are virtual.  Their
plan ops must follow their dependents — when any local FV that references
a source has an applied counterpart in ``applied_state``, the source
emits ``NO_CHANGE`` regardless of whether the planner's spec-key lookup
finds a matching applied ``Datasource``.  The user-visible bug being
fixed: BatchSources surface as ``CREATE_SOURCE`` on every plan because
the applied-state ``Datasource`` recovered from the deployed FV's DT
text uses the underlying table name (or a synthetic ``<FV>__SOURCE``
for query-backed sources), so the planner's key
``Datasource:DB.SCHEMA:<authored_name>`` never collides with the
recovered key ``Datasource:DB.SCHEMA:<table_or_synthetic>``.

Today (red): every test in this file fails — sources missing an
applied-state match emit ``CREATE_SOURCE`` unconditionally.

After the fix (green): a source whose referencing local FV is deployed
emits ``NO_CHANGE``; a source whose referencing local FV is new (or
unreferenced entirely) still emits ``CREATE_SOURCE``.
"""

from __future__ import annotations

import copy
from typing import Any

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FSColumn,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver

_DB = "DB1"
_SCHEMA = "SC1"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _entity_user() -> Entity:
    return Entity(
        kind="Entity",
        name="USER",
        database=_DB,
        schema_=_SCHEMA,
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )


def _batch_fv_dict(*, name: str, source_name: str) -> dict[str, Any]:
    """Return a minimal BatchFeatureView authoring dict that references ``source_name``."""
    return {
        "kind": "BatchFeatureView",
        "name": name,
        "version": "V1",
        "database": _DB,
        "schema": _SCHEMA,
        "online": True,
        "target_lag_sec": 60,
        "entities": ["USER_ID"],
        "sources": [{"name": source_name, "source_type": "Batch"}],
        "refresh_freq": "1 minute",
    }


def _streaming_fv_dict(*, name: str, source_name: str) -> dict[str, Any]:
    """Return a minimal StreamingFeatureView authoring dict that references ``source_name``."""
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "version": "V1",
        "database": _DB,
        "schema": _SCHEMA,
        "online": True,
        "feature_granularity_sec": 300,
        "feature_aggregation_method": "tiles",
        "entities": ["USER_ID"],
        "sources": [{"name": source_name, "source_type": "Stream"}],
        "features": [
            {
                "source_column": {"name": "VALUE", "type": "FloatType"},
                "output_column": {"name": "VALUE_SUM_5M", "type": "FloatType"},
                "function": "sum",
                "window_sec": 300,
            }
        ],
        "timestamp_col": "EVENT_TS",
    }


def _resolve_and_compile(batch: SpecBatch, fv: FeatureView) -> dict[str, Any]:
    """Resolve datasource columns onto the FV and return its compiled spec."""
    decl_api.resolve_datasource_columns(batch)
    return compile_to_spec(fv.model_dump(exclude_none=True), _DB, _SCHEMA)


def _applied_fv(compiled: dict[str, Any], *, kind: str, name: str) -> AppliedObject:
    """Build an AppliedObject for the FV reflecting a deployed runtime."""
    key = f"{kind}:{_DB}.{_SCHEMA}:{name}:V1"
    return AppliedObject(
        key=key,
        kind=kind,
        name=name,
        version="V1",
        content_hash=_full_spec_hash(compiled),
        spec_payload=copy.deepcopy(compiled),
        columns=[],
        from_specification=True,
    )


def _applied_state(*applied_objects: AppliedObject) -> AppliedState:
    return AppliedState(objects={ao.key: ao for ao in applied_objects})


def _ops_by_name(plan: Any, name: str) -> list[Any]:
    return [op for op in plan.ops if op.name == name]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_batch_source_with_deployed_table_backed_fv_is_no_change() -> None:
    """BatchSource(table=...) whose referencing FV is deployed → NO_CHANGE.

    The applied ``Datasource`` recovered from the deployed BatchFV's DT
    text carries the underlying table name (``RAW_EVENTS_BATCH_DECL``),
    not the authored source name (``EVENTS_BATCH_DECL``), so the
    planner's ``Datasource:DB1.SC1:EVENTS_BATCH_DECL`` lookup misses.
    The new override must still emit ``NO_CHANGE`` because the parent
    FV is deployed.
    """
    src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        database=_DB,
        schema_=_SCHEMA,
        table="RAW_EVENTS_BATCH_DECL",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(_batch_fv_dict(name="MY_BATCH_FV", source_name="EVENTS_BATCH_DECL"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    compiled = _resolve_and_compile(batch, fv)

    # The recovered Datasource AppliedObject uses the *table* name as its
    # canonical name (mirrors state._datasource_objects_from_specs).
    derived_datasource_key = f"Datasource:{_DB}.{_SCHEMA}:RAW_EVENTS_BATCH_DECL"
    applied = _applied_state(
        _applied_fv(compiled, kind="BatchFeatureView", name="MY_BATCH_FV"),
        AppliedObject(
            key=derived_datasource_key,
            kind="Datasource",
            name="RAW_EVENTS_BATCH_DECL",
            spec_payload={
                "kind": "Datasource",
                "name": "RAW_EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_BATCH_DECL",
                "columns": [],
            },
            from_specification=True,
        ),
    )

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "EVENTS_BATCH_DECL")
    assert len(src_ops) == 1, f"expected one source op, got {[(o.kind.value, o.name) for o in plan.ops]}"
    assert src_ops[0].kind == OpKind.NO_CHANGE, (
        f"expected NO_CHANGE for BatchSource referenced by deployed FV; "
        f"got {src_ops[0].kind.value} (reason={src_ops[0].reason!r})"
    )
    assert src_ops[0].destructive is False


def test_batch_source_with_deployed_query_backed_fv_is_no_change() -> None:
    """BatchSource(query=...) whose referencing FV is deployed → NO_CHANGE.

    The applied ``Datasource`` for a query-backed FV is recovered with a
    synthetic ``<FV>__SOURCE`` name (mirrors
    state._inject_batch_fv_source_from_dt_text query-shape branch),
    which cannot collide with the authored source name.  The override
    still routes through to ``NO_CHANGE``.
    """
    src = BatchSource(
        kind="BatchSource",
        name="EVENTS_SQL_BATCH_DECL",
        database=_DB,
        schema_=_SCHEMA,
        query="SELECT USER_ID, EVENT_TS, METRIC_VAL FROM RAW_EVENTS_BATCH_DECL LIMIT 100",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(_batch_fv_dict(name="MY_SQL_BATCH_FV", source_name="EVENTS_SQL_BATCH_DECL"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    compiled = _resolve_and_compile(batch, fv)

    derived_datasource_key = f"Datasource:{_DB}.{_SCHEMA}:MY_SQL_BATCH_FV__SOURCE"
    applied = _applied_state(
        _applied_fv(compiled, kind="BatchFeatureView", name="MY_SQL_BATCH_FV"),
        AppliedObject(
            key=derived_datasource_key,
            kind="Datasource",
            name="MY_SQL_BATCH_FV__SOURCE",
            spec_payload={
                "kind": "Datasource",
                "name": "MY_SQL_BATCH_FV__SOURCE",
                "source_type": "Batch",
                "query": "SELECT USER_ID, EVENT_TS, METRIC_VAL FROM RAW_EVENTS_BATCH_DECL LIMIT 100",
                "columns": [],
            },
            from_specification=True,
        ),
    )

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "EVENTS_SQL_BATCH_DECL")
    assert len(src_ops) == 1
    assert src_ops[0].kind == OpKind.NO_CHANGE, (
        f"expected NO_CHANGE for query-backed BatchSource referenced by deployed FV; "
        f"got {src_ops[0].kind.value} (reason={src_ops[0].reason!r})"
    )


def test_streaming_source_with_deployed_fv_is_no_change() -> None:
    """StreamingSource whose referencing FV is deployed → NO_CHANGE.

    Streaming sources preserve their ``name`` on the deployed FV's
    ``spec.sources[]`` so today's matching path already emits
    ``NO_CHANGE`` via the standard spec-key lookup.  Regression-pin
    that behaviour so the override does not break it.
    """
    src = StreamingSource(
        kind="StreamingSource",
        name="CLICKSTREAM_EVENTS",
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )
    fv = FeatureView.model_validate(_streaming_fv_dict(name="USER_CLICK_STATS", source_name="CLICKSTREAM_EVENTS"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    compiled = _resolve_and_compile(batch, fv)

    # Streaming path: the applied Datasource uses the authored name.
    # Mirror the full shape that ``state._build_stream_source_object`` /
    # ``state._datasource_objects_from_specs`` stamp on the runtime
    # payload (database, schema, description) so the four-way decision
    # in :func:`compute_source_diff_kind` recognises this as a
    # structural round-trip rather than a recreate.
    datasource_key = f"Datasource:{_DB}.{_SCHEMA}:CLICKSTREAM_EVENTS"
    datasource_payload = {
        "kind": "Datasource",
        "name": "CLICKSTREAM_EVENTS",
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": "Stream",
        "columns": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "VALUE", "type": "FloatType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ],
        "description": "",
    }
    applied = _applied_state(
        _applied_fv(compiled, kind="StreamingFeatureView", name="USER_CLICK_STATS"),
        AppliedObject(
            key=datasource_key,
            kind="Datasource",
            name="CLICKSTREAM_EVENTS",
            content_hash=structural_fingerprint_hash(datasource_payload),
            spec_payload=datasource_payload,
            from_specification=True,
        ),
    )

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
    assert len(src_ops) == 1
    assert src_ops[0].kind == OpKind.NO_CHANGE, (
        f"expected NO_CHANGE for StreamingSource referenced by deployed FV; " f"got {src_ops[0].kind.value}"
    )


def test_brand_new_batch_source_emits_create_source() -> None:
    """BatchSource referenced only by a brand-new local FV → CREATE_SOURCE.

    No applied FV exists, so the override has no anchor — the source
    is genuinely new and the planner must emit ``CREATE_SOURCE``
    (which the executor treats as a virtual no-op for batch).
    """
    src = BatchSource(
        kind="BatchSource",
        name="NEW_BATCH_SOURCE",
        database=_DB,
        schema_=_SCHEMA,
        table="RAW_NEW_BATCH",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(_batch_fv_dict(name="NEW_BATCH_FV", source_name="NEW_BATCH_SOURCE"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    decl_api.resolve_datasource_columns(batch)

    applied = AppliedState(objects={})

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "NEW_BATCH_SOURCE")
    assert len(src_ops) == 1
    assert (
        src_ops[0].kind == OpKind.CREATE_SOURCE
    ), f"expected CREATE_SOURCE for brand-new BatchSource; got {src_ops[0].kind.value}"
    assert src_ops[0].destructive is False


def test_brand_new_streaming_source_emits_create_source() -> None:
    """StreamingSource referenced only by a brand-new local FV → CREATE_SOURCE.

    Even though streaming sources are also virtual at the planner
    surface, the executor branch in ``imperative_executor.execute_plan``
    calls ``FeatureStore.register_stream_source`` on CREATE_SOURCE for
    StreamingSource, and downstream CREATE_FV's
    ``get_stream_source`` lookup depends on it.  So a brand-new
    StreamingSource must continue to emit CREATE_SOURCE.
    """
    src = StreamingSource(
        kind="StreamingSource",
        name="NEW_STREAM_SOURCE",
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )
    fv = FeatureView.model_validate(_streaming_fv_dict(name="NEW_STREAM_FV", source_name="NEW_STREAM_SOURCE"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    decl_api.resolve_datasource_columns(batch)

    applied = AppliedState(objects={})

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "NEW_STREAM_SOURCE")
    assert len(src_ops) == 1
    assert (
        src_ops[0].kind == OpKind.CREATE_SOURCE
    ), f"expected CREATE_SOURCE for brand-new StreamingSource; got {src_ops[0].kind.value}"


def test_orphan_source_unreferenced_locally_emits_create_source() -> None:
    """Source with no local FV referencing it → CREATE_SOURCE.

    The override is anchored on "this source is referenced by a deployed
    FV".  When no local FV references the source at all, we cannot prove
    the source is part of any deployed FV's compiled spec, so it must
    surface as ``CREATE_SOURCE`` (CLI operators can see they added a
    source YAML without any consumer).
    """
    src = BatchSource(
        kind="BatchSource",
        name="ORPHAN_SRC",
        database=_DB,
        schema_=_SCHEMA,
        table="RAW_ORPHAN",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    batch = SpecBatch(specs=[_entity_user(), src])

    applied = AppliedState(objects={})

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "ORPHAN_SRC")
    assert len(src_ops) == 1
    assert src_ops[0].kind == OpKind.CREATE_SOURCE, (
        f"expected CREATE_SOURCE for orphan BatchSource (no local FV reference); " f"got {src_ops[0].kind.value}"
    )


def test_source_reason_string_distinguishes_virtual_no_change() -> None:
    """The override reason string makes the virtual-source path identifiable.

    Operators reading the plan output should see the FV-derivation
    explanation rather than the generic "Already deployed: no structural
    changes detected." string that fires when the standard hash lookup
    matches.  Strings tie together UX surfaces (``snow feature plan``
    output, plan-file JSON ``reason`` field, log lines).
    """
    src = BatchSource(
        kind="BatchSource",
        name="EVENTS_FG_DECL",
        database=_DB,
        schema_=_SCHEMA,
        table="RAW_EVENTS_FG_DECL",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(_batch_fv_dict(name="USER_AMOUNTS_FV", source_name="EVENTS_FG_DECL"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    compiled = _resolve_and_compile(batch, fv)

    applied = _applied_state(_applied_fv(compiled, kind="BatchFeatureView", name="USER_AMOUNTS_FV"))

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "EVENTS_FG_DECL")
    assert len(src_ops) == 1
    assert src_ops[0].kind == OpKind.NO_CHANGE
    reason = src_ops[0].reason
    # Reason must mention the FV-derivation rationale; the exact wording is
    # fixed so the plan file output stays operator-friendly across plans.
    assert "virtual" in reason.lower() or "feature view" in reason.lower(), (
        f"NO_CHANGE reason for a virtual source should explain the FV-derivation; " f"got: {reason!r}"
    )
    # The generic structural-hash reason is wrong for the virtual path.
    assert "no structural changes detected" not in reason.lower(), (
        f"virtual-source NO_CHANGE should not reuse the structural-hash reason; " f"got: {reason!r}"
    )


# ---------------------------------------------------------------------------
# Example-store-shape R1 pin
# ---------------------------------------------------------------------------


def test_runtime_row_authority_short_circuits_virtual_override() -> None:
    """Runtime row in applied state beats the virtual-FV override.

    Wave 3A contract from ``plans/stream_source_contract.md`` §7b: when
    a source has BOTH an applied-state entry (built via
    :func:`state._build_stream_source_object` from a runtime row) AND a
    referencing FV that is already deployed, the planner must enter the
    ``applied is not None`` arm and route through
    :func:`compute_source_diff_kind` rather than the
    ``_sources_with_deployed_fv`` virtual override.

    Simplest case — identical local vs. runtime spec → ``NO_CHANGE``
    with the source-level diff reason (NOT the virtual-override
    reason).  This pins the precedence: the diff helper is authoritative
    once a runtime row exists.
    """
    from snowflake.ml.feature_store.decl.state import _build_stream_source_object

    src = StreamingSource(
        kind="StreamingSource",
        name="CLICKSTREAM_EVENTS",
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )
    fv = FeatureView.model_validate(_streaming_fv_dict(name="USER_CLICK_STATS", source_name="CLICKSTREAM_EVENTS"))
    batch = SpecBatch(specs=[_entity_user(), src, fv])
    compiled = _resolve_and_compile(batch, fv)

    # Build the applied source from a runtime row (Wave 1+2 path) so
    # the spec_payload mirrors what state.fetch_applied_state(
    # stream_source_rows=...) now produces.
    row = {
        "name": "CLICKSTREAM_EVENTS",
        "schema": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "VALUE", "type": "FloatType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ],
        "desc": "",
        "owner": "ROLE_X",
    }
    runtime_src = _build_stream_source_object(row, default_db=_DB, default_schema=_SCHEMA)

    applied = _applied_state(
        _applied_fv(compiled, kind="StreamingFeatureView", name="USER_CLICK_STATS"),
        runtime_src,
    )

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
    assert len(src_ops) == 1, f"expected one op for CLICKSTREAM_EVENTS; got {[op.kind.value for op in plan.ops]}"
    assert src_ops[0].kind == OpKind.NO_CHANGE
    # Reason must come from the diff path (mentions "source-level"),
    # not the virtual override (mentions "virtual" / "feature view").
    reason_low = src_ops[0].reason.lower()
    assert "source-level" in reason_low, (
        f"runtime-row authority must use the source-level NO_CHANGE reason, "
        f"not the virtual-override reason; got {src_ops[0].reason!r}"
    )


def test_example_store_r1_shape_emits_zero_non_no_change_ops() -> None:
    """Replay the example-store plan shape — unit-level pin for R1.

    Mirrors the user-visible shape captured in
    ``declarative_feature_store/example_store/out/plan/feature_plan_20260526T145830.json``
    that exposed the bug:

    * 4 BatchSources (1 query-backed, 3 table-backed), 1 StreamingSource
    * 4 FVs that reference them: 2 BFVs, 2 StreamingFVs (one with each
      source category)
    * 1 Entity
    * Every FV is already deployed (matching applied counterpart)

    Today the planner emits 4 ``CREATE_SOURCE`` ops (1 per BatchSource);
    after the fix every op kind must be ``NO_CHANGE``.  This is the
    unit-level analogue of ``verify_roundtrip.sh`` Step 4 (R1) — it
    catches a regression without a Snowflake connection.
    """
    entity = _entity_user()

    # Three table-backed BatchSources + one query-backed BatchSource +
    # one StreamingSource.  Each is referenced by exactly one FV.
    batch_sources = [
        BatchSource(
            kind="BatchSource",
            name="EVENTS_FG_DECL",
            database=_DB,
            schema_=_SCHEMA,
            table="RAW_EVENTS_FG_DECL",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        ),
        BatchSource(
            kind="BatchSource",
            name="EVENTS_BATCH_DECL",
            database=_DB,
            schema_=_SCHEMA,
            table="RAW_EVENTS_BATCH_DECL",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        ),
        BatchSource(
            kind="BatchSource",
            name="EVENTS_ADV_DECL",
            database=_DB,
            schema_=_SCHEMA,
            table="RAW_EVENTS_ADV_DECL",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        ),
        BatchSource(
            kind="BatchSource",
            name="EVENTS_SQL_BATCH_DECL",
            database=_DB,
            schema_=_SCHEMA,
            query="SELECT USER_ID FROM RAW_EVENTS_BATCH_DECL LIMIT 100",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        ),
    ]
    stream_source = StreamingSource(
        kind="StreamingSource",
        name="CLICKSTREAM_EVENTS",
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )

    fvs = [
        FeatureView.model_validate(_batch_fv_dict(name="USER_AMOUNTS_FG_DECL", source_name="EVENTS_FG_DECL")),
        FeatureView.model_validate(_batch_fv_dict(name="MY_BATCH_FV_BATCH_DECL", source_name="EVENTS_BATCH_DECL")),
        FeatureView.model_validate(_batch_fv_dict(name="MY_ADV_BFV_DECL", source_name="EVENTS_ADV_DECL")),
        FeatureView.model_validate(
            _batch_fv_dict(name="MY_SQL_BATCH_FV_BATCH_DECL", source_name="EVENTS_SQL_BATCH_DECL")
        ),
        FeatureView.model_validate(_streaming_fv_dict(name="USER_CLICK_STATS_DECL", source_name="CLICKSTREAM_EVENTS")),
    ]

    batch = SpecBatch(specs=[entity, *batch_sources, stream_source, *fvs])
    decl_api.resolve_datasource_columns(batch)

    # Build a per-FV applied state via compile_to_spec so every FV is
    # NO_CHANGE.
    applied_objs = []
    for fv in fvs:
        compiled = compile_to_spec(fv.model_dump(exclude_none=True), _DB, _SCHEMA)
        applied_objs.append(_applied_fv(compiled, kind=fv.kind, name=fv.name))

    # Also seed an Entity AppliedObject so the Entity op is NO_CHANGE.
    entity_dump = entity.model_dump(exclude_none=True)
    entity_dump["schema"] = entity_dump.pop("schema_", _SCHEMA)
    applied_objs.append(
        AppliedObject(
            key=f"Entity:{_DB}.{_SCHEMA}:USER",
            kind="Entity",
            name="USER",
            content_hash=structural_fingerprint_hash(entity_dump),
            spec_payload=entity_dump,
            from_specification=False,
        )
    )

    # Throw in a derived Datasource for the streaming source — matches
    # how state._datasource_objects_from_specs / _build_stream_source_object
    # stamp the full payload (database, schema, description) so the
    # four-way decision in :func:`compute_source_diff_kind` recognises
    # this as a structural round-trip rather than a recreate.
    streaming_datasource_payload = {
        "kind": "Datasource",
        "name": "CLICKSTREAM_EVENTS",
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": "Stream",
        "columns": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "VALUE", "type": "FloatType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ],
        "description": "",
    }
    applied_objs.append(
        AppliedObject(
            key=f"Datasource:{_DB}.{_SCHEMA}:CLICKSTREAM_EVENTS",
            kind="Datasource",
            name="CLICKSTREAM_EVENTS",
            content_hash=structural_fingerprint_hash(streaming_datasource_payload),
            spec_payload=streaming_datasource_payload,
            from_specification=True,
        )
    )

    applied = AppliedState(objects={ao.key: ao for ao in applied_objs})

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)

    non_no_change = [op for op in plan.ops if op.kind != OpKind.NO_CHANGE]
    assert non_no_change == [], (
        f"R1 invariant: every op must be NO_CHANGE; "
        f"found {[(op.kind.value, op.name, op.reason) for op in non_no_change]}"
    )


if __name__ == "__main__":
    pytest_driver.main()
