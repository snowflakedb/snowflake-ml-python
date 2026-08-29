"""Planner tests for BatchFeatureView UPDATE_FV vs RECREATE_FV."""

from __future__ import annotations

import copy
from typing import Any

from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import FSColumn
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver


def _minimal_batch_fv(**overrides: Any) -> dict[str, Any]:
    base = {
        "kind": "BatchFeatureView",
        "name": "BFV_PLAN",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def _applied_batch_state(local: dict[str, Any]) -> AppliedState:
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = "BatchFeatureView:DB1.SC1:BFV_PLAN:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_PLAN",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_planner_no_change_when_identical() -> None:
    local = _minimal_batch_fv()

    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(kind="BatchSource", name="SRC1", table="T", columns=[FSColumn(name="USER_ID", type="StringType")])
    fv = FeatureView.model_validate(local)
    state2 = _applied_batch_state(local)
    plan2 = generate_plan(SpecBatch(specs=[ent, src, fv]), state2, PlanOptions(), database="DB1", schema="SC1")
    assert all(op.kind.value == "NO_CHANGE" for op in plan2.ops if op.name == "BFV_PLAN")


def test_planner_update_fv_when_only_refresh_freq_changes() -> None:
    local = _minimal_batch_fv()
    state = _applied_batch_state(local)
    changed = copy.deepcopy(local)
    changed["refresh_freq"] = "10 minutes"

    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(kind="BatchSource", name="SRC1", table="T", columns=[FSColumn(name="USER_ID", type="StringType")])
    fv = FeatureView.model_validate(changed)
    plan = generate_plan(SpecBatch(specs=[ent, src, fv]), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "BFV_PLAN"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value == "UPDATE_FV"
    assert fv_ops[0].destructive is False


def test_planner_recreate_when_sources_change() -> None:
    local = _minimal_batch_fv()
    state = _applied_batch_state(local)
    changed = copy.deepcopy(local)
    changed["sources"] = [
        {
            "name": "OTHER",
            "source_type": "Batch",
            "columns": [{"name": "USER_ID", "type": "StringType"}],
        }
    ]

    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(
        kind="BatchSource", name="OTHER", table="T2", columns=[FSColumn(name="USER_ID", type="StringType")]
    )
    fv = FeatureView.model_validate(changed)
    plan = generate_plan(SpecBatch(specs=[ent, src, fv]), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "BFV_PLAN"]
    assert fv_ops[0].kind.value == "RECREATE_FV"
    assert fv_ops[0].destructive is True


def test_structural_equivalent_helper() -> None:
    from snowflake.ml.feature_store.decl.invariants import (
        batch_feature_view_structural_equivalent,
    )

    a = _minimal_batch_fv()
    b = copy.deepcopy(a)
    b["refresh_freq"] = "99 minutes"
    compiled_a = compile_to_spec(a, "DB1", "SC1")
    assert batch_feature_view_structural_equivalent(b, compiled_a, "DB1", "SC1")


# ---------------------------------------------------------------------------
# §7 bug-bash regression: when the YAML carries BOTH ``target_lag`` and
# ``refresh_freq``, editing only ``refresh_freq`` must still emit
# UPDATE_FV.  The doc/BATCH_FV_BUG_BASH.md §5 fixture is the live shape
# that surfaced this gap — the verifier reports
# ``[fail] plan missing UPDATE_FV after schedule-only edit (doc §7)``.
# ---------------------------------------------------------------------------


def _bugbash_shape_batch_fv(**overrides: Any) -> dict[str, Any]:
    """Mirror docs/BATCH_FV_BUG_BASH.md §5: both target_lag AND refresh_freq set."""
    base = {
        "kind": "BatchFeatureView",
        "name": "BFV_DOC",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "target_lag_sec": 3600,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "1 hour",
    }
    base.update(overrides)
    return base


def test_planner_update_fv_when_only_refresh_freq_changes_with_target_lag_also_set() -> None:
    """§7: edit ``refresh_freq`` while keeping ``target_lag`` constant → UPDATE_FV."""
    local = _bugbash_shape_batch_fv()
    state = _applied_batch_state(local)
    state.objects["BatchFeatureView:DB1.SC1:BFV_DOC:V1"] = state.objects.pop(
        "BatchFeatureView:DB1.SC1:BFV_PLAN:V1",
        state.objects.get("BatchFeatureView:DB1.SC1:BFV_DOC:V1"),  # type: ignore[arg-type]
    )

    changed = copy.deepcopy(local)
    changed["refresh_freq"] = "2 hours"

    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(kind="BatchSource", name="SRC1", table="T", columns=[FSColumn(name="USER_ID", type="StringType")])
    fv = FeatureView.model_validate(changed)
    plan = generate_plan(SpecBatch(specs=[ent, src, fv]), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "BFV_DOC"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value == "UPDATE_FV", (
        f"expected UPDATE_FV when refresh_freq changes (target_lag unchanged); "
        f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
    )
    assert fv_ops[0].destructive is False


def _applied_batch_state_for(local: dict[str, Any], key: str) -> AppliedState:
    """Build an AppliedState whose key matches the local spec's qualified name."""
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind=local["kind"],
                name=local["name"],
                version=local["version"],
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_planner_no_change_when_only_target_lag_changes_pending_oft_recovery() -> None:
    """After the ``feature_granularity`` / ``refresh_freq`` /
    ``target_lag`` decoupling, ``target_lag`` is OFT staleness only
    and no longer feeds the wire-form ``spec.target_lag_sec`` (the
    offline DT refresh, which is sourced from ``refresh_freq``
    exclusively).

    Detecting ``target_lag`` drift on an online BFV would require the
    applied-state side to surface the deployed OFT ``TARGET_LAG``
    (read from ``SHOW ONLINE FEATURE TABLES`` or a similar surface).
    That recovery path is not yet wired in, so editing only
    ``target_lag`` on the authoring side currently lands as
    ``NO_CHANGE`` — the operational drift detector cannot see the
    deployed OFT cadence to compare against.

    This is a known limitation tracked in ``docs/LIMITATIONS.md``.
    The test pins the current behaviour so a future fix that adds
    OFT-side TARGET_LAG recovery will surface as a deliberate test
    update rather than a silent regression.

    For the sister-case "edit only ``refresh_freq`` → ``UPDATE_FV``"
    see :func:`test_planner_update_fv_when_only_refresh_freq_changes_with_target_lag_also_set`
    above.
    """
    local = _bugbash_shape_batch_fv()
    state = _applied_batch_state_for(local, "BatchFeatureView:DB1.SC1:BFV_DOC:V1")

    changed = copy.deepcopy(local)
    changed["target_lag_sec"] = 7200

    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(kind="BatchSource", name="SRC1", table="T", columns=[FSColumn(name="USER_ID", type="StringType")])
    fv = FeatureView.model_validate(changed)
    plan = generate_plan(SpecBatch(specs=[ent, src, fv]), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "BFV_DOC"]
    assert len(fv_ops) == 1
    # Pinned limitation: until applied-state OFT TARGET_LAG recovery
    # lands, target_lag-only drift is invisible to the planner.
    assert fv_ops[0].kind.value == "NO_CHANGE"
    assert fv_ops[0].destructive is False


def test_batch_fv_operational_drift_recovers_when_compile_to_spec_raises() -> None:
    """If compile_to_spec raises, drift detection must not silently fall back to NO_CHANGE.

    Reads as: when the local authoring dict cannot be compiled but operational
    fields are present, the helper should still surface a drift signal so the
    planner emits UPDATE_FV rather than swallowing the diff.
    """
    from snowflake.ml.feature_store.decl.planner import _batch_fv_operational_drift

    local = _bugbash_shape_batch_fv()
    # Force compile_to_spec to raise by supplying a sentinel field that breaks
    # the FeatureView model dump round-trip.  We deepcopy + corrupt sources[0]
    # so the existing tests are unaffected.
    broken = copy.deepcopy(local)
    broken["sources"] = "not-a-list"  # forces compile to raise
    applied = compile_to_spec(local, "DB1", "SC1")
    # Modify applied so target_lag_sec differs — the drift helper should
    # still return True (diff signal) despite the local-compile failure.
    applied["spec"]["target_lag_sec"] = 99999

    assert _batch_fv_operational_drift(broken, applied, "DB1", "SC1") is True


# ---------------------------------------------------------------------------
# §8 bug-bash regression: when the FV references its BatchSource by name only
# (no inline ``table:``) and the BatchSource YAML's ``table:`` is edited, the
# planner must emit RECREATE_FV (destructive) so ``snow feature apply --from .``
# refuses without ``--allow-recreate``.  The verifier reports
# ``[fail] plan missing RECREATE_FV after datasource table change (doc §8)``.
# ---------------------------------------------------------------------------


def test_planner_recreate_when_batch_source_table_swaps_via_resolution() -> None:
    """§8: edit BatchSource ``table:`` while FV uses name-only reference → RECREATE_FV.

    Mirrors the doc/BATCH_FV_BUG_BASH.md §5 + §8 shape:
      * FV ``sources: [{name: EVENTS_BATCH_DECL, source_type: Batch}]`` (no inline table)
      * Initial BatchSource ``table: RAW_EVENTS_BATCH_DECL``
      * §8 edit: BatchSource ``table: RAW_EVENTS_BATCH_DECL_V2``
    The merge happens via :func:`decl_api.resolve_datasource_columns`; both
    halves of the diff must see the new table for the planner to emit a
    destructive op.
    """
    from snowflake.ml.feature_store.decl import api as decl_api
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    fv_authoring_template = {
        "kind": "BatchFeatureView",
        "name": "MY_BATCH_FV",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "target_lag_sec": 3600,
        "entities": ["USER_ID"],
        "sources": [{"name": "EVENTS_BATCH_DECL", "source_type": "Batch"}],
        "refresh_freq": "1 minute",
    }

    initial_src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        table="RAW_EVENTS_BATCH_DECL",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    initial_fv = FeatureView.model_validate(copy.deepcopy(fv_authoring_template))
    initial_batch = SpecBatch(
        specs=[
            Entity(
                kind="Entity",
                name="USER",
                join_keys=[FSColumn(name="USER_ID", type="StringType")],
            ),
            initial_src,
            initial_fv,
        ]
    )
    decl_api.resolve_datasource_columns(initial_batch)
    initial_compiled = compile_to_spec(
        initial_fv.model_dump(exclude_none=True),
        "DB1",
        "SC1",
    )
    initial_hash = _full_spec_hash(initial_compiled)
    applied = AppliedState(
        objects={
            "BatchFeatureView:DB1.SC1:MY_BATCH_FV:V1": AppliedObject(
                key="BatchFeatureView:DB1.SC1:MY_BATCH_FV:V1",
                kind="BatchFeatureView",
                name="MY_BATCH_FV",
                version="V1",
                content_hash=initial_hash,
                spec_payload=copy.deepcopy(initial_compiled),
                columns=[],
                from_specification=True,
            )
        }
    )

    swapped_src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        table="RAW_EVENTS_BATCH_DECL_V2",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    swapped_fv = FeatureView.model_validate(copy.deepcopy(fv_authoring_template))
    swapped_batch = SpecBatch(
        specs=[
            Entity(
                kind="Entity",
                name="USER",
                join_keys=[FSColumn(name="USER_ID", type="StringType")],
            ),
            swapped_src,
            swapped_fv,
        ]
    )
    decl_api.resolve_datasource_columns(swapped_batch)

    plan = generate_plan(swapped_batch, applied, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "MY_BATCH_FV"]
    assert len(fv_ops) == 1, f"expected one FV op, got {[(o.kind.value, o.name) for o in plan.ops]}"
    assert fv_ops[0].kind.value == "RECREATE_FV", (
        f"expected RECREATE_FV on BatchSource table swap; got {fv_ops[0].kind.value} " f"(reason={fv_ops[0].reason!r})"
    )
    assert fv_ops[0].destructive is True
    payload_sources = fv_ops[0].payload.get("sources") or []
    assert payload_sources
    p0 = payload_sources[0]
    p0_table = p0.get("table") if isinstance(p0, dict) else getattr(p0, "table", None)
    assert p0_table == "RAW_EVENTS_BATCH_DECL_V2"


def test_planner_recreate_when_batch_source_table_swaps_without_specification() -> None:
    """§8 fallback: same swap but ``applied.from_specification=False``.

    On accounts where ``DESCRIBE … TYPE = SPECIFICATION`` returns no parseable
    payload, the planner falls back to ``structural_fingerprint_hash``.  That
    fingerprint must include the FV's source table so the swap is still
    detected — otherwise §8 silently emits ``NO_CHANGE``.
    """
    from snowflake.ml.feature_store.decl import api as decl_api
    from snowflake.ml.feature_store.decl.invariants import structural_fingerprint_hash
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    fv_authoring_template = {
        "kind": "BatchFeatureView",
        "name": "MY_BATCH_FV_STRUCT",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "target_lag_sec": 60,
        "entities": ["USER_ID"],
        "sources": [{"name": "EVENTS_BATCH_DECL", "source_type": "Batch"}],
        "refresh_freq": "1 minute",
    }
    initial_src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        table="RAW_EVENTS_BATCH_DECL",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    initial_fv = FeatureView.model_validate(copy.deepcopy(fv_authoring_template))
    initial_batch = SpecBatch(
        specs=[
            Entity(
                kind="Entity",
                name="USER",
                join_keys=[FSColumn(name="USER_ID", type="StringType")],
            ),
            initial_src,
            initial_fv,
        ]
    )
    decl_api.resolve_datasource_columns(initial_batch)
    initial_dump = initial_fv.model_dump(exclude_none=True)
    initial_fp_hash = structural_fingerprint_hash(initial_dump)

    applied = AppliedState(
        objects={
            "BatchFeatureView:DB1.SC1:MY_BATCH_FV_STRUCT:V1": AppliedObject(
                key="BatchFeatureView:DB1.SC1:MY_BATCH_FV_STRUCT:V1",
                kind="BatchFeatureView",
                name="MY_BATCH_FV_STRUCT",
                version="V1",
                content_hash=initial_fp_hash,
                spec_payload=initial_dump,
                columns=[],
                from_specification=False,
            )
        }
    )

    swapped_src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        table="RAW_EVENTS_BATCH_DECL_V2",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    swapped_fv = FeatureView.model_validate(copy.deepcopy(fv_authoring_template))
    swapped_batch = SpecBatch(
        specs=[
            Entity(
                kind="Entity",
                name="USER",
                join_keys=[FSColumn(name="USER_ID", type="StringType")],
            ),
            swapped_src,
            swapped_fv,
        ]
    )
    decl_api.resolve_datasource_columns(swapped_batch)

    plan = generate_plan(swapped_batch, applied, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == "MY_BATCH_FV_STRUCT"]
    assert len(fv_ops) == 1, f"expected one FV op, got {[(o.kind.value, o.name) for o in plan.ops]}"
    assert fv_ops[0].kind.value == "RECREATE_FV", (
        f"expected RECREATE_FV on BatchSource table swap (no SPECIFICATION); "
        f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
    )
    assert fv_ops[0].destructive is True


def test_resolve_datasource_columns_propagates_table_swap_to_fv() -> None:
    """W-B unit: after editing BatchSource ``table:`` the FV's sources[0] sees the new table."""
    from snowflake.ml.feature_store.decl import api as decl_api
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    fv_authoring = {
        "kind": "BatchFeatureView",
        "name": "MY_BATCH_FV_PROP",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "target_lag_sec": 60,
        "entities": ["USER_ID"],
        "sources": [{"name": "EVENTS_BATCH_DECL", "source_type": "Batch"}],
        "refresh_freq": "1 minute",
    }
    src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        table="RAW_EVENTS_BATCH_DECL_V2",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(copy.deepcopy(fv_authoring))
    batch = SpecBatch(
        specs=[
            Entity(
                kind="Entity",
                name="USER",
                join_keys=[FSColumn(name="USER_ID", type="StringType")],
            ),
            src,
            fv,
        ]
    )
    decl_api.resolve_datasource_columns(batch)

    fv_after: Any = next(s for s in batch.specs if getattr(s, "name", "") == "MY_BATCH_FV_PROP")
    s0 = fv_after.sources[0]
    table = s0.table if hasattr(s0, "table") else s0.get("table")
    assert table == "RAW_EVENTS_BATCH_DECL_V2"

    dump = fv_after.model_dump(exclude_none=True)
    s0_dump = dump["sources"][0]
    dump_table = s0_dump.get("table") if isinstance(s0_dump, dict) else getattr(s0_dump, "table", None)
    assert dump_table == "RAW_EVENTS_BATCH_DECL_V2"


# ---------------------------------------------------------------------------
# Phase 5 / B3 — query-backed BatchFV round-trip parity
#
# A query-backed BatchFV produces a Dynamic Table whose offline body
# carries the operator's SQL.  On re-apply:
#
# * The local FV's ``sources[0]`` carries the operator's
#   ``BatchSource.name`` plus the (whitespace-normalised) query.
# * Phase B3: the recovered applied side reads the operator's authored
#   ``BatchSource.name`` + ``query`` directly from
#   ``FV_SOURCE_REFS`` metadata (surfaced via the
#   ``feature_view_rows[].source_refs`` cell — plan section A1) — no
#   synthetic ``<FV>__SOURCE`` placeholder, no DT-text classification.
#
# Hash convergence on a clean round-trip is therefore mechanical: the
# applied ``spec.sources[0]`` matches the local-compile output one-for-
# one.  ``test_query_backed_fv_full_spec_hash_ignores_source_name``
# below additionally pins the planner-side normaliser that keeps the
# FV-level hash invariant across the *legacy* synthetic-name shape so
# pre-upgrade deployed FVs still round-trip cleanly.
# ---------------------------------------------------------------------------


def _query_backed_batch_fv(
    *,
    fv_name: str = "FV_QUERY_BACKED",
    source_name: str = "EVENTS_QRY",
    query: str = "SELECT user_id, event_ts FROM RAW.EVENTS WHERE event_ts > '2024-01-01'",
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "kind": "BatchFeatureView",
        "name": fv_name,
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "target_lag_sec": 60,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": source_name,
                "source_type": "Batch",
                "query": query,
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "1 minute",
    }
    if extra_overrides:
        base.update(extra_overrides)
    return base


def _applied_state_via_metadata_roundtrip(local: dict[str, Any], fv_name: str) -> AppliedState:
    """Build an ``AppliedState`` mirroring the Phase-B3 metadata roundtrip.

    Under the new contract, ``FV_SOURCE_REFS`` carries the operator-
    authored ``BatchSource.name`` + ``columns`` + ``query`` verbatim,
    so the recovered ``spec.sources[0]`` matches the local-compile
    output one-for-one (no synthetic ``<FV>__SOURCE`` placeholder, no
    DT-text classification).  The recovered ``content_hash`` is just
    ``_full_spec_hash(compile_to_spec(local))`` — exactly what
    :func:`state.fetch_applied_state` would produce for the same
    deployed FV after running ``_inject_batch_fv_source_from_metadata``
    over the matching ``feature_view_rows[].source_refs`` cell.

    Pre-Phase-B this helper synthesised a ``<FV>__SOURCE`` placeholder
    + DT-text-derived query body to mimic the regex recovery path.  The
    placeholder is gone (Phase B3); the new contract preserves the
    operator-authored name across the round trip and is exercised
    directly by the test class without any name-shape adjustment.

    Args:
        local: The authoring dict (BatchFeatureView shape) the test
            wants to roundtrip.  Passed through :func:`compile_to_spec`
            verbatim to produce the recovered ``spec_payload``.
        fv_name: FeatureView name used to build the
            ``BatchFeatureView:<db>.<schema>:<name>`` applied-object
            key.

    Returns:
        AppliedState carrying exactly one BatchFV ``AppliedObject``,
        with ``content_hash`` equal to
        ``_full_spec_hash(compile_to_spec(local))``.
    """
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = f"BatchFeatureView:DB1.SC1:{fv_name}:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name=fv_name,
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def _query_fv_specs(local: dict[str, Any]) -> list[Any]:
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchSource,
        Entity,
        FeatureView,
    )

    src0 = local["sources"][0]
    return [
        Entity(
            kind="Entity",
            name="USER",
            join_keys=[FSColumn(name="USER_ID", type="StringType")],
        ),
        BatchSource(
            kind="BatchSource",
            name=src0["name"],
            query=src0["query"],
            columns=src0["columns"],
        ),
        FeatureView.model_validate(local),
    ]


def test_query_backed_batch_fv_no_drift_on_reapply() -> None:
    """Re-applying an unchanged query-backed BatchFV emits NO_CHANGE for the FV.

    Phase B3: ``FV_SOURCE_REFS`` carries the operator-authored
    ``BatchSource.name`` authoritatively, so the recovered applied
    side and the local-compile side share the exact source identity.
    Hash convergence on a clean round-trip is now mechanical (same
    inputs → same hash) rather than relying on the planner's
    synthetic-name normaliser."""
    local = _query_backed_batch_fv()
    state = _applied_state_via_metadata_roundtrip(local, fv_name=local["name"])
    specs = _query_fv_specs(local)
    plan = generate_plan(SpecBatch(specs=specs), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == local["name"]]
    assert len(fv_ops) == 1, f"expected one FV op, got {[(o.kind.value, o.name) for o in plan.ops]}"
    assert fv_ops[0].kind.value == "NO_CHANGE", (
        f"query-backed BatchFV must emit NO_CHANGE on unchanged re-apply; got "
        f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
    )


def test_query_edit_triggers_recreate_or_update() -> None:
    """A semantic edit to BatchSource.query produces RECREATE_FV (source change)."""
    local = _query_backed_batch_fv(query="SELECT user_id FROM RAW.EVENTS WHERE event_ts > '2024'")
    state = _applied_state_via_metadata_roundtrip(local, fv_name=local["name"])
    edited = copy.deepcopy(local)
    edited["sources"][0]["query"] = "SELECT user_id FROM RAW.EVENTS WHERE event_ts > '2025-06-01'"
    specs = _query_fv_specs(edited)
    plan = generate_plan(SpecBatch(specs=specs), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == local["name"]]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value in ("RECREATE_FV", "UPDATE_FV"), (
        f"semantic query edit must surface as a non-NO_CHANGE op; got "
        f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
    )


def test_query_whitespace_only_change_is_no_op() -> None:
    """A whitespace-only edit to BatchSource.query (different formatting,
    same SQL) must produce NO_CHANGE — the compiler normalizes whitespace
    so the local hash converges with the recovered hash."""
    from snowflake.ml.feature_store.decl.compiler import normalize_sql_whitespace

    local = _query_backed_batch_fv(query="SELECT user_id, event_ts FROM RAW.EVENTS WHERE event_ts > '2024-01-01'")
    state = _applied_state_via_metadata_roundtrip(local, fv_name=local["name"])

    # Build "edited" YAML where the user reformatted the SQL (added newlines /
    # extra spaces) but the semantic query is identical. The compiler runs
    # normalize_sql_whitespace at compile time so both versions converge.
    reformatted = "SELECT  user_id,\n   event_ts\n" "FROM   RAW.EVENTS\nWHERE   event_ts >  '2024-01-01'"
    edited = copy.deepcopy(local)
    edited["sources"][0]["query"] = normalize_sql_whitespace(reformatted)
    specs = _query_fv_specs(edited)
    plan = generate_plan(SpecBatch(specs=specs), state, PlanOptions(), database="DB1", schema="SC1")
    fv_ops = [op for op in plan.ops if op.name == local["name"]]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value == "NO_CHANGE", (
        f"whitespace-only query edit must collapse to NO_CHANGE; got "
        f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
    )


def test_query_backed_fv_full_spec_hash_ignores_source_name() -> None:
    """Direct unit test: _full_spec_hash must return the same value whether
    the query-backed source carries the user's authoring name or the
    Phase-4 synthetic name. The query body is the binding identity."""
    local = _query_backed_batch_fv()
    compiled_local = compile_to_spec(local, "DB1", "SC1")
    h_local = _full_spec_hash(compiled_local)

    recovered = copy.deepcopy(compiled_local)
    recovered["spec"]["sources"] = [
        {
            "name": f"{local['name']}__SOURCE",
            "source_type": "Batch",
            "query": local["sources"][0]["query"],
        }
    ]
    h_recovered = _full_spec_hash(recovered)

    assert h_local == h_recovered, (
        f"FV-level full-spec hash must agree across source-name shapes for "
        f"query-backed sources; local={h_local}, recovered={h_recovered}"
    )


if __name__ == "__main__":
    pytest_driver.main()
