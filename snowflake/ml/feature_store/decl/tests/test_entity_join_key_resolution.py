"""Planner round-trip coverage for entity-name → join-key resolution.

The declarative compile path historically wrote the wire field
``ordered_entity_column_names`` straight from the authored **entity
names** (``compile_to_spec`` → ``list(spec_dict["entities"])``), while the
**applied** side recovers entity **join-key column names** (core
``_build_batch_feature_view_spec`` iterates ``entity.join_keys``).  When an
entity's name differs from its join-key column (e.g. ``USER_ID_DIAG`` whose
join key is ``USER_ID``), the two halves diverge and every replan emits
``RECREATE_FV`` — the persistent loop reported in
``bug_explicit_cluster_by_secondary_key_fv_recreate_loop.md``.

The compounded ``cluster_by`` half falls out of the same root cause:
``invariants._strip_default_cluster_by`` derives its "default" candidate
set from ``ordered_entity_column_names``, so an authored join-key cluster
list is not recognised as the default (and not stripped) while the
name-based entity list is in play.

These tests pin the desired post-fix behaviour: once the planner resolves
entity references to their join-key columns, a clean round-trip lands as
``NO_CHANGE`` for both the control (no ``cluster_by``) and the explicit
``cluster_by:[join_key]`` + secondary-key shapes.  The name==join-key case
must keep working (regression guard for the overwhelmingly common shape).
"""

from __future__ import annotations

import copy
from typing import Any, Optional
from unittest.mock import MagicMock

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash, validate_specs
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
    ObjectKind,
    Plan,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.feature_store.entity import Entity as CoreEntity
from snowflake.ml.feature_store.feature_view import FeatureView as CoreFeatureView
from snowflake.ml.test_utils import pytest_driver
from snowflake.snowpark.types import StringType, StructField, StructType

_DB = "DB1"
_SCHEMA = "SC1"
_FV_NAME = "BFV_DIAG"


def _tiled_bfv(
    *,
    entity_name: str,
    join_key: str,
    cluster_by: Optional[list[str]] = None,
    secondary_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    """A minimal tiled BatchFV authoring dict referencing *entity_name*.

    The source carries the *join_key* column (the real underlying entity
    column) so the compiled spec is internally consistent regardless of how
    the entity is named.

    Args:
        entity_name: The name the FV references in its ``entities`` list.
        join_key: The real underlying entity column carried by the source.
        cluster_by: Optional explicit ``cluster_by`` list.
        secondary_keys: Optional ``aggregation_secondary_keys`` list.

    Returns:
        The authoring-shape BatchFV dict.
    """
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": _FV_NAME,
        "version": "V1",
        "database": _DB,
        "schema": _SCHEMA,
        "online": False,
        "entities": [entity_name],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [
                    {"name": join_key, "type": "StringType"},
                    {"name": "SESSION_ID", "type": "StringType"},
                    {"name": "AMOUNT", "type": "FloatType"},
                ],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "refresh_freq": "5 minutes",
        "feature_granularity": "1h",
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window": "1h",
            }
        ],
    }
    if cluster_by is not None:
        base["cluster_by"] = cluster_by
    if secondary_keys is not None:
        base["aggregation_secondary_keys"] = secondary_keys
    return base


def _spec_batch(local: dict[str, Any], *, entity_name: str, join_key: str) -> SpecBatch:
    """Build the SpecBatch the planner receives, with the Entity spec included.

    The Entity's ``name`` is the reference used inside the FV's ``entities``
    list; its ``join_keys`` carry the real underlying column.

    Args:
        local: The FV authoring dict to validate into a ``FeatureView``.
        entity_name: The Entity name referenced by the FV.
        join_key: The Entity's join-key column.

    Returns:
        The ``SpecBatch`` of ``[Entity, BatchSource, FeatureView]``.
    """
    ent = Entity(
        kind="Entity",
        name=entity_name,
        join_keys=[FSColumn(name=join_key, type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[
            FSColumn(name=join_key, type="StringType"),
            FSColumn(name="SESSION_ID", type="StringType"),
            FSColumn(name="AMOUNT", type="FloatType"),
        ],
    )
    fv = FeatureView.model_validate(local)
    return SpecBatch(specs=[ent, src, fv])


def _streaming_fv(*, entity_name: str, join_key: str) -> dict[str, Any]:
    """A minimal continuous StreamingFV authoring dict referencing *entity_name*."""
    return {
        "kind": "StreamingFeatureView",
        "name": _FV_NAME,
        "version": "V1",
        "database": _DB,
        "schema": _SCHEMA,
        "entities": [entity_name],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Stream",
                "columns": [
                    {"name": join_key, "type": "StringType"},
                    {"name": "AMOUNT", "type": "FloatType"},
                ],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "feature_aggregation_method": "continuous",
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window": "1h",
            }
        ],
    }


def _fv_applied_object(local: dict[str, Any], *, join_key: str) -> tuple[str, AppliedObject]:
    """Build the deployed FV ``AppliedObject`` whose entity columns are the *join keys*.

    Mirrors the real deployed/recovered shape: core recovers
    ``ordered_entity_column_names`` from ``entity.join_keys`` (the column
    name), not the authored entity name.  We model that by compiling a copy
    of *local* whose ``entities`` list is already the join-key column — i.e.
    what the applied side always carries after a clean apply + DESCRIBE.

    Args:
        local: The FV authoring dict whose deployed counterpart to model.
        join_key: The join-key column the applied side recovers.

    Returns:
        A ``(state_key, AppliedObject)`` pair for the deployed FV.
    """
    resolved = copy.deepcopy(local)
    resolved["entities"] = [join_key]
    compiled = compile_to_spec(resolved, _DB, _SCHEMA)
    kind = str(local.get("kind", "BatchFeatureView"))
    key = f"{kind}:{_DB}.{_SCHEMA}:{_FV_NAME}:V1"
    return key, AppliedObject(
        key=key,
        kind=kind,
        name=_FV_NAME,
        version="V1",
        content_hash=_full_spec_hash(compiled),
        spec_payload=copy.deepcopy(compiled),
        columns=[],
        from_specification=True,
    )


def _entity_applied_object(*, entity_name: str, join_key: str) -> tuple[str, AppliedObject]:
    """Build an ``Entity`` ``AppliedObject`` carrying ``join_keys`` (applied-only path)."""
    key = f"{ObjectKind.ENTITY}:{_DB}.{_SCHEMA}:{entity_name}"
    return key, AppliedObject(
        key=key,
        kind=ObjectKind.ENTITY,
        name=entity_name,
        version=None,
        content_hash="",
        spec_payload={
            "kind": ObjectKind.ENTITY,
            "name": entity_name,
            "database": _DB,
            "schema": _SCHEMA,
            "join_keys": [{"name": join_key, "type": "StringType"}],
        },
        columns=[],
        from_specification=False,
    )


def _applied_state_from_join_keys(local: dict[str, Any], *, join_key: str) -> AppliedState:
    """An ``AppliedState`` carrying only the deployed FV (join-key entity columns)."""
    key, obj = _fv_applied_object(local, join_key=join_key)
    return AppliedState(objects={key: obj})


def _fv_op_kind(plan: Plan) -> OpKind:
    fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
    assert len(fv_ops) == 1, f"expected exactly one FV op, got {fv_ops}"
    return fv_ops[0].kind


def test_control_entity_name_differs_from_join_key_no_change() -> None:
    """Control (no ``cluster_by``): name != join key must round-trip NO_CHANGE."""
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")
    plan = generate_plan(
        _spec_batch(local, entity_name="USER_ID_DIAG", join_key="USER_ID"),
        applied,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


def test_explicit_cluster_by_join_key_with_secondary_key_no_change() -> None:
    """Explicit ``cluster_by:[join_key]`` + secondary key must round-trip NO_CHANGE.

    This is the exact bug shape: a tiled BFV with
    ``aggregation_secondary_keys`` and an explicit ``cluster_by`` that omits
    the secondary key, whose entity name differs from its join-key column.
    """
    local = _tiled_bfv(
        entity_name="USER_ID_DIAG",
        join_key="USER_ID",
        cluster_by=["USER_ID"],
        secondary_keys=["SESSION_ID"],
    )
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")
    plan = generate_plan(
        _spec_batch(local, entity_name="USER_ID_DIAG", join_key="USER_ID"),
        applied,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


def test_regression_entity_name_equals_join_key_still_no_change() -> None:
    """Common shape (name == join key) must remain NO_CHANGE after the fix."""
    local = _tiled_bfv(entity_name="USER_ID", join_key="USER_ID")
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")
    plan = generate_plan(
        _spec_batch(local, entity_name="USER_ID", join_key="USER_ID"),
        applied,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


def test_create_fv_payload_keeps_authoring_entities_and_no_stamp() -> None:
    """CREATE_FV payload must stay authoring-shaped: names preserved, no internal stamp key.

    The executor resolves ``entities`` (names) via ``fs.get_entity``; the
    join-key resolution is a planner/compiler concern only and must not leak
    into ``PlanOp.payload``.
    """
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    plan = generate_plan(
        _spec_batch(local, entity_name="USER_ID_DIAG", join_key="USER_ID"),
        AppliedState(objects={}),
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
    assert len(fv_ops) == 1 and fv_ops[0].kind is OpKind.CREATE_FV
    payload = fv_ops[0].payload or {}
    assert payload.get("entities") == ["USER_ID_DIAG"], (
        "PlanOp.payload must keep the authored entity names for executor get_entity; "
        f"got {payload.get('entities')!r}"
    )
    assert "_ordered_entity_columns" not in payload, (
        "The join-key resolution must not leak into PlanOp.payload; "
        f"found _ordered_entity_columns={payload.get('_ordered_entity_columns')!r}"
    )


def test_applied_only_entity_resolves_join_keys_no_change() -> None:
    """Incremental plan: Entity absent from batch, present in applied state → NO_CHANGE.

    The batch carries only the FV (and its source); the entity-name -> join-key
    map must fall back to the applied ``Entity`` object so the wire field still
    resolves to the join key and the replan is clean.
    """
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    fv_key, fv_obj = _fv_applied_object(local, join_key="USER_ID")
    ent_key, ent_obj = _entity_applied_object(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = AppliedState(objects={fv_key: fv_obj, ent_key: ent_obj})

    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="SESSION_ID", type="StringType"),
            FSColumn(name="AMOUNT", type="FloatType"),
        ],
    )
    batch = SpecBatch(specs=[src, FeatureView.model_validate(local)])

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


def test_streaming_fv_name_differs_from_join_key_no_change() -> None:
    """StreamingFeatureView (name != join key) must also round-trip NO_CHANGE."""
    local = _streaming_fv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")
    ent = Entity(
        kind="Entity",
        name="USER_ID_DIAG",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = StreamingSource(
        kind="StreamingSource",
        name="SRC1",
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="AMOUNT", type="FloatType"),
        ],
    )
    batch = SpecBatch(specs=[ent, src, FeatureView.model_validate(local)])
    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


def test_validate_specs_entity_by_name_no_missing_entity() -> None:
    """validate_specs must accept an FV that references an entity by **name**.

    Today ``_check_dependencies`` only indexes join-key columns, so an
    ``entities: [USER_ID_DIAG]`` reference (name != join key) falsely trips
    ``MISSING_ENTITY`` and the CLI never plans.
    """
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    batch = _spec_batch(local, entity_name="USER_ID_DIAG", join_key="USER_ID")
    results = validate_specs(batch, AppliedState(objects={}), target_database=_DB, target_schema=_SCHEMA)
    missing_entity = [r for r in results if r.code == "MISSING_ENTITY"]
    assert not missing_entity, (
        "An FV that references its entity by name must not trip MISSING_ENTITY when the "
        f"entity is in the batch; got {[(r.code, r.message) for r in missing_entity]}"
    )


def test_api_compile_to_spec_forwards_entity_join_keys() -> None:
    """The public ``decl.api.compile_to_spec`` facade forwards ``entity_join_keys``."""
    from snowflake.ml.feature_store.decl import api as decl_api

    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    compiled = decl_api.compile_to_spec(
        local,
        _DB,
        _SCHEMA,
        entity_join_keys={"USER_ID_DIAG": ["USER_ID"]},
    )
    assert compiled["spec"]["ordered_entity_column_names"] == ["USER_ID"]


def test_compiler_matches_core_ordered_entity_columns() -> None:
    """Parity guard: the compiler's wire field must equal core's own derivation.

    Every other test here builds the "applied" half with ``compile_to_spec``
    too, so both sides of each hash comparison come from the function under
    test -- they prove self-consistency but cannot detect divergence from
    core ``FeatureView.ordered_entity_columns`` (the property the applied side
    really recovers via ``entity.join_keys``).  A core casing/ordering change
    would leave all the round-trips green.

    This test crosses that boundary: a real core ``Entity`` + ``FeatureView``
    (name != join key) produce ``ordered_entity_columns`` independently, and we
    assert the compiler's ``ordered_entity_column_names`` matches it byte-for-byte.
    """
    mock_df = MagicMock()
    mock_df.queries = {"queries": ["SELECT * FROM RAW_EVENTS"]}
    mock_df.columns = ["USER_ID", "SESSION_ID", "AMOUNT"]
    mock_df.schema = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("SESSION_ID", StringType()),
            StructField("AMOUNT", StringType()),
        ]
    )

    core_entity = CoreEntity(name="USER_ID_DIAG", join_keys=["USER_ID"])
    core_fv = CoreFeatureView(name=_FV_NAME, entities=[core_entity], feature_df=mock_df)

    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    compiled = compile_to_spec(
        local,
        _DB,
        _SCHEMA,
        entity_join_keys={"USER_ID_DIAG": ["USER_ID"]},
    )

    assert compiled["spec"]["ordered_entity_column_names"] == core_fv.ordered_entity_columns


# ---------------------------------------------------------------------------
# Applied-first join-key resolution + plan-time immutable ERROR
# ---------------------------------------------------------------------------


def test_build_entity_join_key_map_applied_wins_for_deployed_entity() -> None:
    """A deployed entity resolves from applied join keys, not the edited batch.

    The executor reproduces the applied join keys (``update_entity`` is
    description-only), so the FV hash must be computed from what will
    actually be materialised, not from an unappliable YAML edit.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import build_entity_join_key_map

    batch_entity = {
        "kind": ObjectKind.ENTITY,
        "name": "USER_ID_DIAG",
        "join_keys": [{"name": "ORG_ID", "type": "StringType"}],
    }
    ent_key, ent_obj = _entity_applied_object(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = AppliedState(objects={ent_key: ent_obj})

    mapping = build_entity_join_key_map([batch_entity], applied)
    assert mapping["USER_ID_DIAG"] == ["USER_ID"]


def test_build_entity_join_key_map_batch_wins_for_new_entity() -> None:
    """An entity absent from applied state (new CREATE) resolves from the batch."""
    from snowflake.ml.feature_store.decl.spec_compiler import build_entity_join_key_map

    batch_entity = {
        "kind": ObjectKind.ENTITY,
        "name": "USER_ID_DIAG",
        "join_keys": [{"name": "ORG_ID", "type": "StringType"}],
    }
    mapping = build_entity_join_key_map([batch_entity], AppliedState(objects={}))
    assert mapping["USER_ID_DIAG"] == ["ORG_ID"]


def _renamed_join_key_batch(local: dict[str, Any]) -> SpecBatch:
    """Batch whose Entity join key differs from the deployed one (rename)."""
    ent = Entity(
        kind="Entity",
        name="USER_ID_DIAG",
        join_keys=[FSColumn(name="ORG_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="SESSION_ID", type="StringType"),
            FSColumn(name="AMOUNT", type="FloatType"),
        ],
    )
    return SpecBatch(specs=[ent, src, FeatureView.model_validate(local)])


def _applied_state_with_entity(local: dict[str, Any], *, join_key: str) -> AppliedState:
    """Applied state carrying both the deployed FV and its deployed Entity."""
    fv_key, fv_obj = _fv_applied_object(local, join_key=join_key)
    ent_key, ent_obj = _entity_applied_object(entity_name="USER_ID_DIAG", join_key=join_key)
    return AppliedState(objects={fv_key: fv_obj, ent_key: ent_obj})


def test_validate_specs_rejects_entity_join_key_edit() -> None:
    """A join-key rename on a deployed entity is a plan-time ERROR, not MISSING_ENTITY."""
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = _applied_state_with_entity(local, join_key="USER_ID")

    results = validate_specs(
        _renamed_join_key_batch(local),
        applied,
        target_database=_DB,
        target_schema=_SCHEMA,
    )
    assert any(
        r.code == "ENTITY_JOIN_KEY_IMMUTABLE" for r in results
    ), f"expected ENTITY_JOIN_KEY_IMMUTABLE; got {[(r.code, r.message) for r in results]}"
    assert not [r for r in results if r.code == "MISSING_ENTITY"]


def test_generate_plan_join_key_edit_keeps_fv_no_change() -> None:
    """Editing the entity YAML must not recreate dependent FVs.

    With applied-first resolution the FV compiles against the deployed
    join key, so its op stays ``NO_CHANGE`` even though the batch entity
    carries a different (unappliable) join key.  The entity itself still
    diffs (``UPDATE_ENTITY``); the plan-time ERROR blocks it at the CLI.
    """
    local = _tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = _applied_state_with_entity(local, join_key="USER_ID")

    plan = generate_plan(
        _renamed_join_key_batch(local),
        applied,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    assert _fv_op_kind(plan) is OpKind.NO_CHANGE


# ---------------------------------------------------------------------------
# validate_specs idempotency must thread the entity join-key map
# ---------------------------------------------------------------------------


def _validatable_tiled_bfv(*, entity_name: str, join_key: str) -> dict[str, Any]:
    """A tiled BFV whose feature carries ``window_sec`` (post-load shape).

    ``validate_specs`` runs the authoring-completeness check
    (``_check_feature_aggregations``), which requires an integer
    ``window_sec`` on each aggregation row — the shape a loaded project
    carries.  The bare ``window`` string on ``_tiled_bfv`` alone trips
    ``FEATURE_INCOMPLETE`` independently of the idempotency path under test.

    Args:
        entity_name: The name the FV references in its ``entities`` list.
        join_key: The real underlying entity column carried by the source.

    Returns:
        The authoring-shape BatchFV dict with ``window_sec`` populated.
    """
    local = _tiled_bfv(entity_name=entity_name, join_key=join_key)
    for feat in local["features"]:
        feat["window_sec"] = 3600
    return local


def test_validate_specs_no_change_for_name_differs_from_join_key() -> None:
    """A clean replan of a name != join-key project must short-circuit NO_CHANGE.

    ``_check_idempotency`` must compile the local FV with the entity
    join-key map (like the planner), or its name-based local hash mismatches
    the column-based applied hash and ``validate_specs`` never emits
    ``NO_CHANGE`` for exactly the projects 6c1 fixes.
    """
    local = _validatable_tiled_bfv(entity_name="USER_ID_DIAG", join_key="USER_ID")
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")

    results = validate_specs(
        _spec_batch(local, entity_name="USER_ID_DIAG", join_key="USER_ID"),
        applied,
        target_database=_DB,
        target_schema=_SCHEMA,
    )
    no_change = [r for r in results if r.code == "NO_CHANGE" and r.object_name == _FV_NAME]
    assert no_change, f"expected NO_CHANGE for {_FV_NAME}; got {[(r.code, r.object_name) for r in results]}"
    errors = [r for r in results if r.severity == "ERROR"]
    assert not errors, f"clean replan must not error; got {[(r.code, r.message) for r in errors]}"


def test_validate_specs_no_change_for_name_equals_join_key() -> None:
    """Regression guard: the common name == join-key shape stays NO_CHANGE."""
    local = _validatable_tiled_bfv(entity_name="USER_ID", join_key="USER_ID")
    applied = _applied_state_from_join_keys(local, join_key="USER_ID")

    results = validate_specs(
        _spec_batch(local, entity_name="USER_ID", join_key="USER_ID"),
        applied,
        target_database=_DB,
        target_schema=_SCHEMA,
    )
    no_change = [r for r in results if r.code == "NO_CHANGE" and r.object_name == _FV_NAME]
    assert no_change
    assert not [r for r in results if r.severity == "ERROR"]


if __name__ == "__main__":
    pytest_driver.main()
