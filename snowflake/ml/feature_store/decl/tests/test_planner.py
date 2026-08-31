"""Tests for decl/planner.py — plan generation."""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    model_to_dict,
    spec_key,
    structural_fingerprint_hash,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    Entity,
    FeatureView,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_model(name: str = "user") -> Entity:
    return Entity.model_validate(
        {
            "kind": "Entity",
            "name": name,
            "database": "DB",
            "schema_": "SCH",
            "join_keys": [{"name": "user_id", "type": "StringType"}],
        }
    )


def _source_model(name: str = "clicks") -> StreamingSource:
    return StreamingSource.model_validate(
        {
            "kind": "StreamingSource",
            "name": name,
            "database": "DB",
            "schema_": "SCH",
            "columns": [],
        }
    )


def _fv_model(name: str = "click_fv", version: str = "V1", output_name: str = "event") -> FeatureView:
    return FeatureView.model_validate(
        {
            "kind": "StreamingFeatureView",
            "name": name,
            "database": "DB",
            "schema_": "SCH",
            "version": version,
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": output_name, "type": "StringType"},
                }
            ],
        }
    )


def _batch(*models: Any) -> SpecBatch:
    return SpecBatch(specs=list(models), source_files=[])


def _empty_applied() -> AppliedState:
    return AppliedState(objects={})


def _applied_with_model(model: Any) -> AppliedState:
    normalized = model_to_dict(model)
    key = spec_key(normalized)
    ao = AppliedObject(
        key=key,
        kind=normalized.get("kind", ""),
        name=normalized.get("name", ""),
        version=normalized.get("version"),
        content_hash=structural_fingerprint_hash(normalized),
        spec_payload=normalized,
    )
    return AppliedState(objects={key: ao})


def _opts(**kwargs: Any) -> PlanOptions:
    return PlanOptions(**kwargs)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestGeneratePlan:
    def test_new_entity_produces_create_op(self) -> None:
        batch = _batch(_entity_model())
        plan = generate_plan(batch, _empty_applied(), _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_ENTITY in kinds

    def test_new_fv_produces_create_fv_op(self) -> None:
        fv = _fv_model()
        batch = _batch(fv)
        plan = generate_plan(batch, _empty_applied(), _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_FV in kinds

    def test_new_source_produces_create_source_op(self) -> None:
        batch = _batch(_source_model())
        plan = generate_plan(batch, _empty_applied(), _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_SOURCE in kinds

    def test_unchanged_spec_produces_no_change_op(self) -> None:
        fv = _fv_model()
        applied = _applied_with_model(fv)
        batch = _batch(fv)
        plan = generate_plan(batch, applied, _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds

    def test_updated_spec_produces_recreate_fv_op(self) -> None:
        """A content change at the *same* version produces RECREATE (OFTs
        cannot be updated in place).

        Identity is ``(name, version)``, so the edit must keep ``V1`` on
        both sides — a version bump would instead be a brand-new object.
        """
        old_fv = _fv_model(version="V1")
        new_fv = _fv_model(version="V1", output_name="event_renamed")  # same version, changed content
        applied = _applied_with_model(old_fv)
        batch = _batch(new_fv)
        plan = generate_plan(batch, applied, _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.RECREATE_FV in kinds

    def test_plan_ops_are_topologically_ordered(self) -> None:
        entity = _entity_model()
        source = _source_model()
        fv = _fv_model()
        # Pass in reverse order — plan should still sort correctly
        batch = _batch(fv, source, entity)
        plan = generate_plan(batch, _empty_applied(), _opts())
        names = [op.name for op in plan.ops]
        # Entity and source must come before FV
        if "click_fv" in names and "user" in names:
            assert names.index("user") < names.index("click_fv")

    def test_plan_op_includes_payload(self) -> None:
        fv = _fv_model()
        batch = _batch(fv)
        plan = generate_plan(batch, _empty_applied(), _opts())
        create_ops = [op for op in plan.ops if op.kind == OpKind.CREATE_FV]
        assert len(create_ops) == 1
        assert create_ops[0].payload != {}

    def test_plan_op_name_matches_spec_name(self) -> None:
        fv = _fv_model(name="my_fv")
        batch = _batch(fv)
        plan = generate_plan(batch, _empty_applied(), _opts())
        names = [op.name for op in plan.ops]
        assert "my_fv" in names

    def test_changed_spec_always_recreates(self) -> None:
        """OFTs cannot be updated in place — any change produces RECREATE."""
        old_fv_spec = {
            "kind": "StreamingFeatureView",
            "name": "click_fv",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event_upper", "type": "StringType"},
                    "function": "lower",
                }
            ],
        }
        new_fv_spec = {
            "kind": "StreamingFeatureView",
            "name": "click_fv",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",  # same version, changed content → destructive recreate
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    # Renamed output column → real structural change at same version.
                    "output_column": {"name": "event_renamed", "type": "StringType"},
                    "function": "upper",
                }
            ],
        }
        old_fv = FeatureView.model_validate(old_fv_spec)
        new_fv = FeatureView.model_validate(new_fv_spec)
        applied = _applied_with_model(old_fv)
        batch = _batch(new_fv)
        # Even without allow_recreate, OFTs always recreate
        plan = generate_plan(batch, applied, _opts())
        recreate_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_FV]
        assert len(recreate_ops) == 1

    def test_allow_recreate_produces_recreate_op(self) -> None:
        old_fv_spec = {
            "kind": "StreamingFeatureView",
            "name": "click_fv",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event_upper", "type": "StringType"},
                    "function": "lower",
                }
            ],
        }
        new_fv_spec = {
            "kind": "StreamingFeatureView",
            "name": "click_fv",
            "database": "DB",
            "schema_": "SCH",
            "version": "V1",  # same version, changed content → destructive
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    # Renamed output column → real structural change at same version.
                    "output_column": {"name": "event_renamed", "type": "StringType"},
                }
            ],
        }
        old_fv = FeatureView.model_validate(old_fv_spec)
        new_fv = FeatureView.model_validate(new_fv_spec)
        applied = _applied_with_model(old_fv)
        batch = _batch(new_fv)
        plan = generate_plan(batch, applied, _opts(allow_recreate=True))
        kinds = [op.kind for op in plan.ops]
        assert OpKind.RECREATE_FV in kinds

    def test_empty_batch_produces_empty_plan(self) -> None:
        plan = generate_plan(SpecBatch(), _empty_applied(), _opts())
        assert plan.ops == []

    def test_plan_warnings_for_no_change(self) -> None:
        fv = _fv_model()
        applied = _applied_with_model(fv)
        batch = _batch(fv)
        plan = generate_plan(batch, applied, _opts())
        # NO_CHANGE op is emitted; no warnings needed for this case
        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds

    def test_generate_plan_no_change_when_specs_lack_database_schema(self) -> None:
        """Connection context must qualify keys for unqualified specs.

        Reproduces the ``snow feature plan`` UI/disk parity bug: when a
        spec dict lacks ``database`` / ``schema_`` (e.g. a bare entity
        YAML or a single-file ``apply`` invocation against a path that
        doesn't encode db/schema), the planner used to build a key like
        ``Entity::USER`` while the applied-state row built
        ``Entity:JKEW_DB.JKEW_SCHEMA:USER``.  The keys never collided,
        so an entity that already existed showed up as a phantom
        ``CREATE_ENTITY`` *and* a phantom ``DROP_ENTITY`` (in
        full-directory mode) on a clean round-trip — and the plan-file
        ⇄ UI op streams diverged whenever ``write_plan`` skipped the
        ``apply``-side ``SpecBatch`` mutation that injected the
        connection context into the dicts.

        After Phase 1, ``generate_plan`` accepts ``database`` /
        ``schema`` kwargs and threads them into every internal
        ``spec_key`` callsite (the diff lookup *and* the
        full-directory-mode batch-key set), so the unqualified spec
        collides with the fully-qualified applied row and we get
        ``NO_CHANGE`` (with zero ``DROP_ENTITY`` orphans).
        """
        # Local spec lacks db/schema (simulates a bare YAML).
        unqualified_entity = Entity.model_validate(
            {
                "kind": "Entity",
                "name": "user",
                "join_keys": [{"name": "user_id", "type": "StringType"}],
            }
        )
        # Applied state mirrors the *server* shape: always fully
        # qualified, content_hash matching the structural fingerprint
        # of the same logical entity.
        applied_dict = {
            "kind": "Entity",
            "name": "USER",
            "database": "JKEW_DB",
            "schema_": "JKEW_SCHEMA",
            "join_keys": [{"name": "user_id", "type": "StringType"}],
        }
        applied_key = "Entity:JKEW_DB.JKEW_SCHEMA:USER"
        applied = AppliedState(
            objects={
                applied_key: AppliedObject(
                    key=applied_key,
                    kind="Entity",
                    name="USER",
                    content_hash=structural_fingerprint_hash(model_to_dict(unqualified_entity)),
                    spec_payload=applied_dict,
                )
            }
        )

        # Full-directory mode is the path that emits DROP orphans, so
        # we test it explicitly here — the parity bug surfaces only
        # when the deletion-detection pass runs.
        plan = generate_plan(
            _batch(unqualified_entity),
            applied,
            _opts(full_directory_mode=True),
            database="JKEW_DB",
            schema="JKEW_SCHEMA",
        )

        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds, (
            f"Expected NO_CHANGE for round-trip on unqualified spec; got {kinds}. "
            "This means the planner failed to thread connection context into "
            "spec_key, so the unqualified spec key didn't collide with the "
            "fully-qualified applied-state key."
        )
        assert OpKind.CREATE_ENTITY not in kinds, (
            "Phantom CREATE_ENTITY emitted — the spec didn't match the "
            "applied row because the lookup key was missing the "
            "connection-context qualifier."
        )
        assert OpKind.DROP_ENTITY not in kinds, (
            "Phantom DROP_ENTITY emitted in full-directory mode — the "
            "deletion-detection pass didn't see the spec's batch key as "
            "matching the applied-state key, so it treated the live "
            "entity as an orphan."
        )


class TestDeletionDetection:
    """Tests for full_directory_mode deletion detection pass."""

    def test_orphaned_fv_produces_drop_fv_op_in_full_directory_mode(self) -> None:
        """FV in applied_state but not in batch → DROP_FV when full_directory_mode=True."""
        deployed_fv = _fv_model(name="old_fv")
        applied = _applied_with_model(deployed_fv)
        # Batch does NOT contain old_fv
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_FV in kinds
        drop_op = next(op for op in plan.ops if op.kind == OpKind.DROP_FV)
        assert drop_op.name == "old_fv"

    def test_orphaned_fv_not_dropped_without_full_directory_mode(self) -> None:
        """FV in applied_state but not in batch → no DROP when full_directory_mode=False."""
        deployed_fv = _fv_model(name="old_fv")
        applied = _applied_with_model(deployed_fv)
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=False))
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_FV not in kinds

    def test_drop_op_is_destructive(self) -> None:
        """DROP ops must have destructive=True."""
        deployed_fv = _fv_model(name="stale_fv")
        applied = _applied_with_model(deployed_fv)
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_FV]
        assert len(drop_ops) == 1
        assert drop_ops[0].destructive is True

    def test_drop_op_reason_mentions_local_spec(self) -> None:
        """DROP ops must explain why the drop is happening."""
        deployed_fv = _fv_model(name="stale_fv")
        applied = _applied_with_model(deployed_fv)
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_FV]
        assert "local spec" in drop_ops[0].reason.lower() or "not present" in drop_ops[0].reason.lower()

    def test_orphaned_entity_produces_drop_entity_op(self) -> None:
        """Entity in applied_state but not in batch → DROP_ENTITY in full_directory_mode."""
        deployed_entity = _entity_model(name="old_entity")
        applied = _applied_with_model(deployed_entity)
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_ENTITY in kinds
        drop_op = next(op for op in plan.ops if op.kind == OpKind.DROP_ENTITY)
        assert drop_op.name == "old_entity"
        assert drop_op.destructive is True

    def test_present_fv_not_dropped_in_full_directory_mode(self) -> None:
        """FV in both applied_state and batch → no DROP op even in full_directory_mode."""
        fv = _fv_model(name="keep_fv")
        applied = _applied_with_model(fv)
        batch = _batch(fv)
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        kinds = [op.kind for op in plan.ops]
        assert OpKind.DROP_FV not in kinds

    def test_multiple_orphaned_fvs_all_produce_drops(self) -> None:
        """Multiple orphaned FVs each get their own DROP_FV op."""
        fv1 = _fv_model(name="old_fv1")
        fv2 = _fv_model(name="old_fv2")
        d1 = model_to_dict(fv1)
        d2 = model_to_dict(fv2)
        key1 = spec_key(d1)
        key2 = spec_key(d2)
        from snowflake.ml.feature_store.decl.types import AppliedObject

        ao1 = AppliedObject(
            key=key1,
            kind=d1["kind"],
            name=d1["name"],
            version=d1.get("version"),
            content_hash=structural_fingerprint_hash(d1),
            spec_payload=d1,
        )
        ao2 = AppliedObject(
            key=key2,
            kind=d2["kind"],
            name=d2["name"],
            version=d2.get("version"),
            content_hash=structural_fingerprint_hash(d2),
            spec_payload=d2,
        )
        applied = AppliedState(objects={key1: ao1, key2: ao2})
        batch = _batch()
        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))
        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_FV]
        assert len(drop_ops) == 2
        drop_names = {op.name for op in drop_ops}
        assert "old_fv1" in drop_names
        assert "old_fv2" in drop_names

    def test_entity_rename_orders_create_before_drop_in_ops_list(self) -> None:
        """Pin Bug D contract: in an entity rename under full_directory_mode,
        the CREATE_ENTITY op for the new name must appear at a strictly
        lower index than the DROP_ENTITY op for the old name.

        This ordering is the executor's atomicity guarantee for renames:
        the executor ``raise``s on the first error, so a failed CREATE
        will skip the DROP and leave the old entity preserved (rather
        than half-applying the rename and orphaning data).

        If a future planner refactor flips this ordering (e.g. by
        running deletion detection before the diff loop), this test
        flips red and forces an explicit re-decision about atomicity.
        """
        # Applied state: a single entity named SESSION_ID_OLD.
        deployed_entity = _entity_model(name="SESSION_ID_OLD")
        applied = _applied_with_model(deployed_entity)

        # Local batch: only the renamed entity SESSION_ID_NEW. Same join
        # keys / database / schema, so this *is* a rename, not an
        # unrelated change.
        renamed_entity = _entity_model(name="SESSION_ID_NEW")
        batch = _batch(renamed_entity)

        plan = generate_plan(batch, applied, _opts(full_directory_mode=True))

        kinds = [op.kind for op in plan.ops]
        assert OpKind.CREATE_ENTITY in kinds, f"expected CREATE_ENTITY for new name; got {kinds}"
        assert OpKind.DROP_ENTITY in kinds, f"expected DROP_ENTITY for old name; got {kinds}"

        create_idx = next(
            i for i, op in enumerate(plan.ops) if op.kind == OpKind.CREATE_ENTITY and op.name == "SESSION_ID_NEW"
        )
        drop_idx = next(
            i for i, op in enumerate(plan.ops) if op.kind == OpKind.DROP_ENTITY and op.name == "SESSION_ID_OLD"
        )
        assert create_idx < drop_idx, (
            "Atomic-rename contract broken: CREATE_ENTITY for the new name "
            "MUST precede DROP_ENTITY for the old name in plan.ops so that "
            "a failed CREATE skips the DROP and the old entity is "
            "preserved. "
            f"Got CREATE at index {create_idx}, DROP at index {drop_idx}. "
            "If you intentionally changed this ordering, see the "
            "'Apply Lifecycle Resilience' plan (Bug D pin)."
        )


# ---------------------------------------------------------------------------
# Full-spec diff tests (from_specification=True path)
# ---------------------------------------------------------------------------


def _fv_dict_with_udf(
    *,
    udf_body: str = "def transform(x):\n    return x",
    window: str | None = None,
    name: str = "click_fv",
    version: str = "v1",
) -> dict[str, Any]:
    """Build a FeatureView authoring dict that mirrors the YAML format.

    The dict is suitable for both ``FeatureView.model_validate`` and
    ``compile_to_spec``.  When ``window`` is set the feature carries a
    time window so aggregation properties end up in the compiled spec.

    Args:
        udf_body: Python source for the UDF ``transform`` function.
        window: Optional aggregation window (e.g. ``"1h"``); when set the
            feature is given a ``count`` aggregation over that window.
        name: Feature view name.
        version: Feature view version.

    Returns:
        Authoring-format feature view dict.
    """
    feature: dict[str, Any] = {
        "source_column": {"name": "event", "type": "StringType"},
        "output_column": {"name": "event", "type": "StringType"},
    }
    if window is not None:
        feature["function"] = "count"
        feature["window"] = window
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema_": "SCH",
        "version": version,
        "entities": ["user_id"],
        "sources": [{"name": "clicks", "source_type": "Stream"}],
        "feature_aggregation_method": "tiles",
        "features": [feature],
        "udf": {
            "name": "transform",
            "engine": "pandas",
            "function_definition": udf_body,
            "output_columns": [{"name": "event", "type": "StringType"}],
        },
    }


def _applied_from_compiled(local_dict: dict[str, Any], *, from_specification: bool = True) -> AppliedState:
    """Build an AppliedState whose FV came from DESCRIBE ... TYPE = SPECIFICATION."""
    # model_to_dict converts schema_ -> schema; mirror that for compile_to_spec.
    compile_input = dict(local_dict)
    if "schema_" in compile_input:
        compile_input["schema"] = compile_input.pop("schema_")
    deployed_spec = compile_to_spec(compile_input, "DB", "SCH")
    key = spec_key({**compile_input, "kind": local_dict["kind"]})
    applied = AppliedObject(
        key=key,
        kind=local_dict["kind"],
        name=local_dict["name"],
        version=local_dict["version"],
        content_hash=_full_spec_hash(deployed_spec),
        spec_payload=deployed_spec,
        from_specification=from_specification,
    )
    return AppliedState(objects={key: applied})


class TestPlannerFullSpecDiff:
    """Full-spec diff path: ``applied.from_specification == True``."""

    def test_udf_change_triggers_recreate_when_specification_available(self) -> None:
        """UDF source code change must trigger RECREATE_FV under full-spec diff."""
        deployed_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x")
        applied = _applied_from_compiled(deployed_dict)

        local_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x.upper()")
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        recreate_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_FV]
        assert len(recreate_ops) == 1
        assert "click_fv" == recreate_ops[0].name
        assert "full-spec" in recreate_ops[0].reason.lower()

    def test_aggregation_window_change_triggers_recreate(self) -> None:
        """Aggregation window change must trigger RECREATE_FV under full-spec diff."""
        deployed_dict = _fv_dict_with_udf(window="5m")
        applied = _applied_from_compiled(deployed_dict)

        local_dict = _fv_dict_with_udf(window="10m")
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        recreate_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_FV]
        assert len(recreate_ops) == 1
        assert "full-spec" in recreate_ops[0].reason.lower()

    def test_full_spec_match_triggers_no_change(self) -> None:
        """Identical specs under full-spec diff produce NO_CHANGE."""
        deployed_dict = _fv_dict_with_udf()
        applied = _applied_from_compiled(deployed_dict)

        local_fv = FeatureView.model_validate(deployed_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds
        assert OpKind.RECREATE_FV not in kinds

    def test_falls_back_to_structural_fingerprint_when_from_specification_false(self) -> None:
        """Without a specification, only column-schema changes trigger RECREATE_FV.

        Legacy structural-fingerprint behaviour: a UDF body change that
        keeps the same output columns must continue to produce NO_CHANGE.
        """
        deployed_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x")
        # Mirror schema_ -> schema for the spec_payload that AppliedObject sees.
        applied_payload = dict(deployed_dict)
        applied_payload["schema"] = applied_payload.pop("schema_")
        key = spec_key(applied_payload)
        applied_obj = AppliedObject(
            key=key,
            kind=applied_payload["kind"],
            name=applied_payload["name"],
            version=applied_payload["version"],
            content_hash=structural_fingerprint_hash(applied_payload),
            spec_payload=applied_payload,
            from_specification=False,
        )
        state = AppliedState(objects={key: applied_obj})

        local_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x.lower()")
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), state, _opts())

        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds
        assert OpKind.RECREATE_FV not in kinds

    def test_volatile_metadata_fields_ignored(self) -> None:
        """Differences in metadata.client_version must not flag a deployed spec as changed."""
        deployed_dict = _fv_dict_with_udf()
        applied = _applied_from_compiled(deployed_dict)
        # Mutate the deployed spec's volatile metadata field; the hash on
        # the AppliedObject was computed from the original (clean) spec,
        # so we directly tamper with the in-memory payload to prove the
        # planner strips these fields before comparing.
        applied_obj = next(iter(applied.objects.values()))
        applied_obj.spec_payload["metadata"]["client_version"] = "9.9.9-dev"
        # Recompute the content hash from the *tampered* payload — the
        # helper must strip volatile keys, so the hash should still match
        # what the local compiled spec produces.
        applied_obj.content_hash = _full_spec_hash(applied_obj.spec_payload)

        local_fv = FeatureView.model_validate(deployed_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds
        assert OpKind.RECREATE_FV not in kinds

    def test_entity_diff_unchanged(self) -> None:
        """Entity behaviour stays on the structural fingerprint regardless of from_specification."""
        entity = _entity_model()
        normalized = model_to_dict(entity)
        key = spec_key(normalized)
        applied_obj = AppliedObject(
            key=key,
            kind=normalized["kind"],
            name=normalized["name"],
            version=normalized.get("version"),
            content_hash=structural_fingerprint_hash(normalized),
            spec_payload=normalized,
            from_specification=True,
        )
        state = AppliedState(objects={key: applied_obj})

        plan = generate_plan(_batch(entity), state, _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds
        assert OpKind.RECREATE_FV not in kinds

    def test_entity_hash_change_emits_update_entity(self) -> None:
        """A hash mismatch on an Entity must produce ``UPDATE_ENTITY``
        (not ``RECREATE_FV``) so the plan cleanly reflects the in-place
        ``ALTER TAG`` path that
        :func:`imperative_executor._execute_update_entity` materialises.

        Before this change the ``_RECREATE_OP`` lookup fell through to
        ``RECREATE_FV`` for any non-FV kind whose hash differed, which
        produced a misleading op kind ("RECREATE_FV" for an Entity) in
        plan output.

        We pin the planner's behaviour by force-setting the applied
        ``content_hash`` to a value that cannot match the local
        :func:`structural_fingerprint_hash`.  This isolates the planner
        decision under test from the (orthogonal) question of which
        entity fields contribute to the structural fingerprint.
        """
        local_entity = _entity_model(name="user")
        local_normalized = model_to_dict(local_entity)
        key = spec_key(local_normalized)
        applied_obj = AppliedObject(
            key=key,
            kind=local_normalized["kind"],
            name=local_normalized["name"],
            version=local_normalized.get("version"),
            content_hash="deadbeef" * 8,  # 64-char placeholder ≠ any real digest
            spec_payload=local_normalized,
        )
        state = AppliedState(objects={key: applied_obj})

        plan = generate_plan(_batch(local_entity), state, _opts())
        kinds = [op.kind for op in plan.ops]
        assert OpKind.UPDATE_ENTITY in kinds
        assert OpKind.RECREATE_FV not in kinds
        update_op = next(op for op in plan.ops if op.kind == OpKind.UPDATE_ENTITY)
        assert update_op.destructive is False
        assert update_op.name == "user"

    def test_missing_spec_payload_falls_back_to_generic_reason(self) -> None:
        """When applied.spec_payload is absent the reason must not crash and must still
        contain the 'full-spec' marker so existing callers that check for that string keep
        passing."""
        deployed_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x")
        applied = _applied_from_compiled(deployed_dict)
        # Wipe the payload so the diff path has nothing to compare against.
        applied_obj = next(iter(applied.objects.values()))
        applied_obj.spec_payload = None  # type: ignore[assignment]

        local_dict = _fv_dict_with_udf(udf_body="def transform(x):\n    return x.upper()")
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        recreate_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_FV]
        assert len(recreate_ops) == 1
        reason = recreate_ops[0].reason
        assert "full-spec" in reason.lower(), f"Generic fallback must mention 'full-spec': {reason!r}"


# ---------------------------------------------------------------------------
# BUG_BASH step 7 audit — pin planner payload aggregation preservation
# ---------------------------------------------------------------------------


def _bug_bash_streaming_fv_model() -> FeatureView:
    """Construct the BUG_BASH §5 USER_CLICK_STATS_DECL FeatureView spec.

    Mirrors the post-:func:`compile_spec` shape the loader produces from
    docs/BUG_BASH.md §5: ``window_sec`` integers (not authoring strings),
    ``feature_granularity_sec=300``, ``feature_aggregation_method="tiles"``,
    and two windowed aggregation features keyed off the UDF outputs.

    Returns:
        A validated ``FeatureView`` ready to drop into a ``SpecBatch``.
    """
    return FeatureView.model_validate(
        {
            "kind": "StreamingFeatureView",
            "name": "USER_CLICK_STATS_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema_": "JKEW_SCHEMA",
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
                        {"name": "EVENT_TYPE", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [
                {
                    "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                    "window_sec": 3600,
                    "function": "sum",
                    "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                },
                {
                    "output_column": {"name": "HAS_CONVERSION_24H", "type": "BooleanType"},
                    "window_sec": 86400,
                    "function": "max",
                    "source_column": {"name": "IS_CONVERSION", "type": "BooleanType"},
                },
            ],
            "udf": {
                "name": "compute_engagement_metrics",
                "engine": "pandas",
                "function_definition": "def compute_engagement_metrics(df):\n    return df\n",
                "output_columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                    {"name": "IS_CONVERSION", "type": "BooleanType"},
                    {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                ],
            },
        }
    )


class TestStreamingFvAggregationPayload:
    """Pin the planner payload contract for streaming FV aggregations.

    The executor (``imperative_executor._build_feature_view``) consumes
    ``op.payload`` directly to construct the imperative ``FeatureView``.
    These tests pin that the planner preserves every aggregation field
    in the payload so a regression at the planner / serializer layer
    surfaces here before the executor gets a chance to misbehave on a
    live ``snow feature apply`` round-trip.

    The local model is constructed via ``FeatureView.model_validate``
    directly (matching the post-compile shape the loader produces) so
    the test is hermetic — no tmpdir, no YAML I/O.
    """

    def test_create_fv_payload_carries_two_aggregation_features(self) -> None:
        """The CREATE_FV payload's ``features`` list must round-trip both
        BUG_BASH features with their ``window_sec`` / ``function`` /
        source / output column intact.
        """
        fv = _bug_bash_streaming_fv_model()
        plan = generate_plan(_batch(fv), _empty_applied(), _opts())

        create_op = next(op for op in plan.ops if op.kind == OpKind.CREATE_FV)
        features = create_op.payload.get("features")
        assert isinstance(features, list)
        assert len(features) == 2

        first, second = features
        assert first["window_sec"] == 3600
        assert first["function"] == "sum"
        assert first["source_column"]["name"] == "ENGAGEMENT_SCORE"
        assert first["output_column"]["name"] == "TOTAL_ENGAGEMENT_1H"

        assert second["window_sec"] == 86400
        assert second["function"] == "max"
        assert second["source_column"]["name"] == "IS_CONVERSION"
        assert second["output_column"]["name"] == "HAS_CONVERSION_24H"

    def test_create_fv_payload_carries_feature_aggregation_method(self) -> None:
        """``feature_aggregation_method`` must reach the executor as-is so
        ``FeatureAggregationMethod.TILES`` flows through to ``FeatureView``.
        """
        fv = _bug_bash_streaming_fv_model()
        plan = generate_plan(_batch(fv), _empty_applied(), _opts())

        create_op = next(op for op in plan.ops if op.kind == OpKind.CREATE_FV)
        assert create_op.payload.get("feature_aggregation_method") == "tiles"

    def test_create_fv_payload_carries_feature_granularity_sec(self) -> None:
        """``feature_granularity_sec`` must reach the executor as the
        integer 300 so the executor can convert to ``"300s"`` for the
        imperative ``feature_granularity`` kwarg.
        """
        fv = _bug_bash_streaming_fv_model()
        plan = generate_plan(_batch(fv), _empty_applied(), _opts())

        create_op = next(op for op in plan.ops if op.kind == OpKind.CREATE_FV)
        assert create_op.payload.get("feature_granularity_sec") == 300


# ---------------------------------------------------------------------------
# FV-level backfill block tests
# ---------------------------------------------------------------------------


def _streaming_fv_with_backfill_dict(
    *,
    name: str = "USER_CLICK_BACKFILL_DECL",
    version: str = "V1",
    backfill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a streaming FV authoring dict that mirrors the YAML format,
    optionally carrying a top-level ``backfill:`` block.

    Args:
        name: The feature view name to embed in the payload.
        version: The feature view version to embed in the payload.
        backfill: Optional FV-level backfill block to attach under the
            top-level ``backfill`` key.

    Returns:
        A YAML-shaped streaming FV authoring dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    payload: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema_": "SCH",
        "version": version,
        "entities": ["user_id"],
        "sources": [{"name": "clicks", "source_type": "Stream"}],
        "features": [
            {
                "source_column": {"name": "event", "type": "StringType"},
                "output_column": {"name": "event", "type": "StringType"},
            }
        ],
        "udf": {
            "name": "transform",
            "engine": "pandas",
            "function_definition": "def transform(x):\n    return x\n",
            "output_columns": [{"name": "event", "type": "StringType"}],
        },
        "feature_aggregation_method": "tiles",
    }
    if backfill is not None:
        payload["backfill"] = backfill
    return payload


def _batch_fv_with_backfill_dict(
    *,
    name: str = "ORDERS_TOTAL_DECL",
    version: str = "V1",
    backfill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": name,
        "database": "DB",
        "schema_": "SCH",
        "version": version,
        "online": False,
        "refresh_freq": "1 hour",
        "entities": ["user_id"],
        "sources": [
            {
                "name": "orders",
                "source_type": "Batch",
                "table": "ORDERS_RAW",
            }
        ],
        "features": [],
    }
    if backfill is not None:
        payload["backfill"] = backfill
    return payload


class TestPlannerBackfillBlock:
    """The FV-level ``backfill: {...}`` block is operational, not structural.

    The planner must:

    * Treat backfill-only edits as ``NO_CHANGE`` against an applied state
      whose deployed spec carries no backfill block.
    * Still emit a ``CREATE_FV`` for new FVs that declare backfill (the
      block reaches the executor as part of ``op.payload``).
    * Mark a backfill-driven re-materialisation (``backfill.overwrite=True``
      against an existing FV) as **destructive** so apply requires
      ``--allow-recreate`` even when the structural hash matches.
    """

    def test_streaming_fv_backfill_only_edit_is_no_change(self) -> None:
        """Add ``backfill: {table, start_time}`` to an existing streaming FV
        with no other change → ``NO_CHANGE``."""
        deployed_dict = _streaming_fv_with_backfill_dict()
        applied = _applied_from_compiled(deployed_dict)

        local_dict = _streaming_fv_with_backfill_dict(
            backfill={
                "table": "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL",
                "start_time": "2026-05-18T22:00:00",
            }
        )
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        kinds = [op.kind for op in plan.ops]
        assert OpKind.NO_CHANGE in kinds, (
            "Adding only a backfill block (no structural diff) must be a " f"NO_CHANGE; got plan kinds: {kinds}"
        )
        assert OpKind.RECREATE_FV not in kinds

    def test_batch_fv_backfill_initialize_only_edit_is_recreate_fv(self) -> None:
        """``backfill.initialize`` is now a back-compat alias for the
        first-class top-level ``initialize:`` field (Phase 4 of the
        advanced BFV plan).  It is therefore *structural* — adding it
        after a deploy that did not set it changes what the imperative
        ``FeatureView(initialize=...)`` constructor sees, which
        Snowflake can only honour at Dynamic Table create time.  The
        planner must emit ``RECREATE_FV`` so the change actually takes
        effect."""
        deployed_dict = _batch_fv_with_backfill_dict()
        applied = _applied_from_compiled(deployed_dict)

        local_dict = _batch_fv_with_backfill_dict(backfill={"initialize": "ON_SCHEDULE"})
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        kinds = [op.kind for op in plan.ops]
        assert (
            OpKind.RECREATE_FV in kinds
        ), f"adding backfill.initialize is structural in Phase 4; got plan kinds: {kinds}"
        assert OpKind.NO_CHANGE not in kinds

    def test_create_fv_payload_carries_backfill_block(self) -> None:
        """When a new FV declares ``backfill:``, the generated ``CREATE_FV``
        op's payload must carry the block verbatim so the executor can
        wire it into ``StreamConfig`` / ``register_feature_view``."""
        local_dict = _streaming_fv_with_backfill_dict(
            backfill={
                "table": "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL",
                "start_time": "2026-05-18T22:00:00",
            }
        )
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), _empty_applied(), _opts())

        create_op = next(op for op in plan.ops if op.kind == OpKind.CREATE_FV)
        backfill = create_op.payload.get("backfill") or {}
        assert backfill.get("table") == "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL"
        assert backfill.get("start_time") is not None

    def test_existing_batch_fv_with_backfill_overwrite_emits_destructive_op(self) -> None:
        """Adding ``backfill.overwrite=True`` to an unchanged existing batch FV
        must emit a destructive op so plain ``snow feature apply`` refuses
        the plan and the operator must explicitly pass ``--allow-recreate``.

        The op kind itself can be CREATE_FV (re-register with overwrite=True)
        or RECREATE_FV (drop + register) — both achieve re-materialisation.
        We pin only the destructive flag, which is the gate that matters
        for the apply-time refuse semantics.
        """
        deployed_dict = _batch_fv_with_backfill_dict()
        applied = _applied_from_compiled(deployed_dict)

        local_dict = _batch_fv_with_backfill_dict(backfill={"overwrite": True})
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied, _opts())

        actionable_ops = [op for op in plan.ops if op.kind != OpKind.NO_CHANGE]
        assert actionable_ops, (
            "backfill.overwrite=True against an existing FV must produce an "
            "actionable plan op (not NO_CHANGE) so the operator can run a "
            "destructive re-materialisation; got "
            f"{[op.kind for op in plan.ops]!r}"
        )
        assert any(op.destructive for op in actionable_ops), (
            "backfill.overwrite=True implies re-materialisation; the op MUST "
            "be marked destructive=True so the --allow-recreate gate fires "
            "at apply time."
        )

    def test_new_fv_with_backfill_overwrite_is_not_destructive(self) -> None:
        """A brand-new FV with ``backfill.overwrite=True`` is benign — there
        is nothing to overwrite, so the CREATE_FV op stays non-destructive
        and a plain ``snow feature apply`` (no --allow-recreate) succeeds."""
        local_dict = _batch_fv_with_backfill_dict(backfill={"overwrite": True})
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), _empty_applied(), _opts())

        create_ops = [op for op in plan.ops if op.kind == OpKind.CREATE_FV]
        assert len(create_ops) == 1
        assert create_ops[0].destructive is False, (
            "A first-time CREATE_FV with backfill.overwrite=True must remain "
            "non-destructive; there is nothing to re-materialise yet."
        )


# ---------------------------------------------------------------------------
# B6 — Planner drift extensions: desc (L3) + refresh_freq (L4) on BatchFV
# ---------------------------------------------------------------------------
#
# Closes LIMITATIONS L3 (desc-only edits invisible to the planner) and L4
# (``refresh_freq``-only edits invisible to the planner).  Both fields are
# alterable in-place via ``fs.update_feature_view(desc=..., refresh_freq=...)``,
# so the planner must emit ``UPDATE_FV`` instead of ``NO_CHANGE`` when only
# these knobs differ between the authoring YAML and the deployed spec.


def _bfv_with_desc_authoring_dict(
    *,
    name: str = "BFV_OP_DRIFT",
    version: str = "V1",
    description: str | None = None,
    refresh_freq: str = "5 minutes",
    warehouse: str | None = None,
) -> dict[str, Any]:
    """Build a minimal BatchFV authoring dict for operational-drift testing.

    The compiled spec for this dict deliberately omits ``desc`` /
    ``description`` / ``warehouse`` so the structural hash stays stable
    across edits to those fields — that is the precondition for the
    planner's hash-matches → operational-drift fast path to fire.

    Uses ``DB`` / ``SCH`` (not ``DB1`` / ``SC1``) so the compiled spec
    metadata matches the connection-context defaults the test fixture
    helper :func:`_applied_from_compiled` uses; without that alignment
    the structural hash would differ on the ``metadata.database`` /
    ``metadata.schema`` field and the test would exercise the
    structural-equivalent path instead of the operational-drift fast
    path under test.

    Args:
        name: BatchFV name (uppercase by convention).
        version: BatchFV version string.
        description: Optional FV-level description text.
        refresh_freq: DT refresh cadence string (e.g. ``"5 minutes"``).
        warehouse: Optional refresh warehouse identifier.

    Returns:
        Authoring-format BatchFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": name,
        "version": version,
        "database": "DB",
        "schema_": "SCH",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": refresh_freq,
    }
    if description is not None:
        payload["description"] = description
    if warehouse is not None:
        payload["warehouse"] = warehouse
    return payload


def _applied_bfv_state(local_dict: dict[str, Any], *, applied_desc: str | None = None) -> AppliedState:
    """Build an AppliedState for a BatchFV with optional applied-side ``desc``.

    Mirrors the post-Phase-A enrichment B-state is wiring into
    ``decl/state.py``: the imperative ``list_feature_views()`` row's
    ``desc`` column is surfaced on the applied spec_payload at top
    level so the planner's drift detector can compare against it.

    Args:
        local_dict: Authoring-format BatchFV dict.
        applied_desc: Optional description text the applied side is
            expected to carry (None → applied has no ``desc`` key).

    Returns:
        AppliedState carrying a single BatchFV ``AppliedObject`` whose
        ``content_hash`` matches ``compile_to_spec(local_dict)`` and
        whose ``spec_payload`` optionally carries ``desc``.
    """
    # _applied_from_compiled handles the schema_ -> schema rename and
    # full-spec hashing; we layer the optional ``desc`` injection on
    # top so the structural hash stays stable.
    state = _applied_from_compiled(local_dict)
    if applied_desc is not None:
        applied_obj = next(iter(state.objects.values()))
        applied_obj.spec_payload["desc"] = applied_desc
    return state


class TestBatchOperationalDrift:
    """B6: desc + refresh_freq drift on a clean BatchFV must emit ``UPDATE_FV``.

    Closes LIMITATIONS L3 (desc) and L4 (refresh_freq).  Both fields are
    handled by ``fs.update_feature_view(desc=..., refresh_freq=...)``
    in-place — no destructive RECREATE_FV required.  The planner detects
    drift in these fields when the structural hash matches but the
    operational subset diverges.
    """

    def test_desc_only_change_emits_update_fv_not_no_change(self) -> None:
        """Local edits ``description``; applied carries the old ``desc``.

        The structural hash matches on both sides because ``compile_to_spec``
        does not emit ``desc`` / ``description`` (B-invariants will keep
        them out of the hash so this contract holds).  The planner must
        observe the desc divergence and emit ``UPDATE_FV`` — not collapse
        to ``NO_CHANGE`` (the pre-B6 behaviour that hid L3 from operators).
        """
        local_dict = _bfv_with_desc_authoring_dict(description="new description")
        applied_state = _applied_bfv_state(
            _bfv_with_desc_authoring_dict(description="old description"),
            applied_desc="old description",
        )
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "BFV_OP_DRIFT"]
        assert len(fv_ops) == 1, f"expected exactly one op for BFV_OP_DRIFT; got {fv_ops!r}"
        # Pin the structural-hash-matches precondition so this test does
        # not silently degrade into the structural-equivalent branch
        # (which exercises a different code path than the
        # operational-drift fast path under test).  Use the same
        # connection-context resolution the planner uses internally so
        # the assertion mirrors the in-planner hash comparison
        # ``current_hash == applied.content_hash`` exactly.
        applied_obj = next(iter(applied_state.objects.values()))
        from snowflake.ml.feature_store.decl.invariants import (
            compute_local_spec_hash,
            model_to_dict,
        )

        local_hash = compute_local_spec_hash(model_to_dict(local_fv), "DB", "SCH")
        assert local_hash == applied_obj.content_hash, (
            "Test precondition broken: local and applied structural hashes "
            "must match for the operational-drift fast path to fire.  "
            f"local={local_hash} applied={applied_obj.content_hash}.  Adjust "
            "the test fixture so the only divergence is the desc field."
        )
        assert fv_ops[0].kind == OpKind.UPDATE_FV, (
            "BatchFV ``description``-only edits must emit UPDATE_FV (closes L3); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r}).  The "
            "planner should compare local ``description`` against applied "
            "``desc`` and route the diff through fs.update_feature_view(desc=...)."
        )
        assert fv_ops[0].destructive is False, "desc edits are non-destructive"

    def test_refresh_freq_only_change_emits_update_fv(self) -> None:
        """Local edits ``refresh_freq``; structural hash matches.

        ``refresh_freq`` maps to ``refresh_freq`` on the imperative
        update path.  ``_OPERATIONAL_FV_KEYS`` and ``_RUNTIME_STAMPED_SPEC_KEYS``
        both strip the seconds-form (``target_lag_sec``) from the
        structural hash so the planner's hash-matches branch fires; the
        operational-drift detector then catches the diff and emits
        ``UPDATE_FV`` (closes L4).
        """
        local_dict = _bfv_with_desc_authoring_dict(refresh_freq="10 minutes")
        applied_state = _applied_bfv_state(
            _bfv_with_desc_authoring_dict(refresh_freq="5 minutes"),
        )
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "BFV_OP_DRIFT"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind == OpKind.UPDATE_FV, (
            "BatchFV ``refresh_freq`` / ``refresh_freq`` edits must emit "
            "UPDATE_FV (closes L4); got "
            f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})."
        )
        assert fv_ops[0].destructive is False


# ---------------------------------------------------------------------------
# B7 — Streaming UPDATE_FV (L5): kind-agnostic operational drift
# ---------------------------------------------------------------------------
#
# StreamingFeatureView and RealtimeFeatureView share the operational-drift
# surface with BatchFV but with different operational subsets:
#   * StreamingFV  → {desc, warehouse, refresh_freq, online_config}
#   * RealtimeFV   → {desc, online_config} only (A5 rejects refresh_freq
#                    and warehouse on RTFVs — no DT, no refresh Task).
# ``backfill_table`` and ``backfill_start_time`` are structural for both
# streaming kinds: a change there should fall through to RECREATE_FV via
# the regular hash-mismatch path, not collapse into UPDATE_FV.


def _streaming_fv_authoring_dict(
    *,
    name: str = "SFV_OP_DRIFT",
    version: str = "V1",
    description: str | None = None,
    warehouse: str | None = None,
    refresh_freq: str | None = None,
    backfill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal StreamingFV authoring dict for operational-drift testing.

    Args:
        name: StreamingFV name.
        version: StreamingFV version.
        description: Optional FV-level description text.
        warehouse: Optional refresh warehouse identifier.
        refresh_freq: Optional refresh-frequency string (maps to
            ``refresh_freq`` for streaming kinds at the imperative
            layer).
        backfill: Optional FV-level backfill block (``table`` /
            ``start_time``).  Treated as **structural** by the planner;
            changes here flow through the hash-mismatch RECREATE_FV
            path, not the operational-drift fast path.

    Returns:
        Authoring-format StreamingFV dict suitable for
        :meth:`FeatureView.model_validate`.
    """
    payload: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": name,
        "version": version,
        "database": "DB",
        "schema_": "SCH",
        "online": True,
        "entities": ["USER_ID"],
        "sources": [{"name": "CLICKS", "source_type": "Stream"}],
        "features": [
            {
                "source_column": {"name": "EVENT", "type": "StringType"},
                "output_column": {"name": "EVENT", "type": "StringType"},
            }
        ],
        "udf": {
            "name": "transform",
            "engine": "pandas",
            "function_definition": "def transform(x):\n    return x\n",
            "output_columns": [{"name": "EVENT", "type": "StringType"}],
        },
    }
    if description is not None:
        payload["description"] = description
    if warehouse is not None:
        payload["warehouse"] = warehouse
    if refresh_freq is not None:
        payload["refresh_freq"] = refresh_freq
    if backfill is not None:
        payload["backfill"] = backfill
    return payload


def _applied_sfv_state(local_dict: dict[str, Any], *, applied_desc: str | None = None) -> AppliedState:
    """Build an AppliedState for a StreamingFV with optional applied-side ``desc``."""
    state = _applied_from_compiled(local_dict)
    if applied_desc is not None:
        applied_obj = next(iter(state.objects.values()))
        applied_obj.spec_payload["desc"] = applied_desc
    return state


class TestStreamingOperationalDrift:
    """B7: StreamingFV operational drift routes through UPDATE_FV.

    The kind-agnostic ``_split_operational_vs_structural`` helper covers
    the StreamingFV operational subset (``desc``, ``warehouse``,
    ``refresh_freq``, ``online_config``).  A change in any of these
    fields against an otherwise-unchanged deployed spec must surface
    as ``UPDATE_FV`` (not ``NO_CHANGE``) so the operator's intent
    flows through ``fs.update_feature_view(...)`` (A5).
    """

    def test_streaming_warehouse_only_change_emits_update_fv(self) -> None:
        """Local sets ``warehouse``; applied has none.

        The structural hash is identical because ``warehouse`` is in
        ``_OPERATIONAL_FV_KEYS`` (stripped from the hash).  The planner
        must observe the warehouse drift via the operational-drift
        helper and emit ``UPDATE_FV``.
        """
        local_dict = _streaming_fv_authoring_dict(warehouse="WH_NEW")
        applied_state = _applied_sfv_state(_streaming_fv_authoring_dict())
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "SFV_OP_DRIFT"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind == OpKind.UPDATE_FV, (
            "StreamingFV ``warehouse``-only edits must emit UPDATE_FV (closes L5); "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})."
        )
        assert fv_ops[0].destructive is False

    def test_streaming_desc_only_change_emits_update_fv(self) -> None:
        """Local edits ``description``; applied carries the old ``desc``.

        Mirrors the BatchFV ``test_desc_only_change_emits_update_fv_not_no_change``
        contract for StreamingFV — both kinds share the same
        operational-drift mechanism via the kind-agnostic helper.
        """
        local_dict = _streaming_fv_authoring_dict(description="new desc")
        applied_state = _applied_sfv_state(
            _streaming_fv_authoring_dict(description="old desc"),
            applied_desc="old desc",
        )
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "SFV_OP_DRIFT"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind == OpKind.UPDATE_FV, (
            "StreamingFV ``description``-only edits must emit UPDATE_FV; "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})."
        )
        assert fv_ops[0].destructive is False

    def test_streaming_backfill_change_emits_recreate(self) -> None:
        """Backfill fields are **structural** for streaming FVs.

        ``backfill_table`` / ``backfill_start_time`` are recreate-only —
        the kind-agnostic operational-drift helper must NOT classify them
        as operational drift, so changes flow through the structural
        diff path.  This pins the contract at the helper level: when
        only backfill differs, the operational-drift helper returns
        ``False`` (no operational drift), so the planner does not
        collapse a structural backfill change into ``UPDATE_FV``.
        """
        from snowflake.ml.feature_store.decl.planner import (
            _FV_OPERATIONAL_FIELDS_BY_KIND,
            _split_operational_vs_structural,
        )

        local_with_backfill = _streaming_fv_authoring_dict(
            backfill={
                "table": "DB.SCH.RAW_HISTORY",
                "start_time": "2026-05-18T22:00:00",
            },
        )
        applied_without_backfill = compile_to_spec(
            {k: v for k, v in _streaming_fv_authoring_dict().items() if k != "schema_"} | {"schema": "SCH"},
            "DB",
            "SCH",
        )
        operational_fields = _FV_OPERATIONAL_FIELDS_BY_KIND["StreamingFeatureView"]
        # ``backfill`` is intentionally absent from the operational-fields
        # set, so the helper must return False (no operational drift).
        # The planner's hash-mismatch RECREATE_FV path (driven by
        # B-invariants making backfill_table / backfill_start_time
        # structural) is what catches this drift end-to-end.
        assert "backfill" not in operational_fields
        assert "backfill_table" not in operational_fields
        assert "backfill_start_time" not in operational_fields
        assert (
            _split_operational_vs_structural(
                local_with_backfill,
                applied_without_backfill,
                operational_fields=operational_fields,
                database="DB1",
                schema="SC1",
            )
            is False
        ), (
            "``backfill_table`` / ``backfill_start_time`` are structural for "
            "streaming FVs (recreate-only).  The operational-drift helper "
            "must not classify them as drift so a backfill change cannot "
            "collapse a destructive recreate into a non-destructive UPDATE_FV."
        )


# ---------------------------------------------------------------------------
# B7 — Realtime FV operational drift: kind-restricted subset
# ---------------------------------------------------------------------------
#
# RealtimeFeatureView is OFT-only — no Dynamic Table, no refresh Task.
# A5 made ``fs.update_feature_view`` reject ``refresh_freq`` and
# ``warehouse`` for RTFVs.  The planner must mirror that contract:
# only ``desc`` and ``online_config`` are in the operational subset;
# everything else (including ``warehouse`` and ``refresh_freq``) flows
# through the structural / non-operational path.


def _realtime_fv_authoring_dict(
    *,
    name: str = "RTFV_OP_DRIFT",
    version: str = "V1",
    warehouse: str | None = None,
    refresh_freq: str | None = None,
) -> dict[str, Any]:
    """Build a minimal RealtimeFV authoring dict for operational-drift testing.

    Args:
        name: RealtimeFV name.
        version: RealtimeFV version.
        warehouse: Optional warehouse (ignored by A5 — not applicable to RTFV).
        refresh_freq: Optional refresh_freq (ignored by A5 — RTFV has no DT).

    Returns:
        Authoring-format RealtimeFV dict.
    """
    payload: dict[str, Any] = {
        "kind": "RealtimeFeatureView",
        "name": name,
        "version": version,
        "database": "DB",
        "schema_": "SCH",
        "entities": ["USER_ID"],
    }
    if warehouse is not None:
        payload["warehouse"] = warehouse
    if refresh_freq is not None:
        payload["refresh_freq"] = refresh_freq
    return payload


class TestRealtimeOperationalDrift:
    """B7: RealtimeFV operational subset is restricted to ``{desc, online_config}``.

    A5 made ``fs.update_feature_view`` reject ``refresh_freq`` and
    ``warehouse`` on RealtimeFV.  The planner's kind-agnostic operational
    helper must reflect that: ``warehouse`` and ``refresh_freq`` are
    **not** in the RTFV operational subset, so a drift in either of
    those fields does NOT collapse into UPDATE_FV.
    """

    def test_realtime_warehouse_or_refresh_freq_change_emits_recreate(self) -> None:
        """RTFV operational set is ``{desc, online_config}`` only.

        Verifies at the helper level that ``warehouse`` and
        ``refresh_freq`` are absent from the RTFV operational subset.
        End-to-end, a drift in either field falls through to
        ``RECREATE_FV`` via the structural-hash-mismatch path (which
        B-invariants pins by removing these from the hash strip list
        for RTFV).  The planner cannot route them through UPDATE_FV
        because the imperative ``update_feature_view`` would reject
        them with A5's INVALID_ARGUMENT diagnostic.
        """
        from snowflake.ml.feature_store.decl.planner import (
            _FV_OPERATIONAL_FIELDS_BY_KIND,
            _split_operational_vs_structural,
        )

        operational_fields = _FV_OPERATIONAL_FIELDS_BY_KIND["RealtimeFeatureView"]
        assert (
            "warehouse" not in operational_fields
        ), "RTFV cannot alter warehouse — fs.update_feature_view rejects it (A5)."
        assert (
            "refresh_freq" not in operational_fields
        ), "RTFV cannot alter refresh_freq — no Dynamic Table or refresh Task (A5)."
        # The operational subset must still cover ``desc`` and ``online_config``.
        assert "desc" in operational_fields
        assert "online_config" in operational_fields

        # Helper-level pin: warehouse/refresh_freq drift against the RTFV
        # operational subset returns False (the helper sees no drift in
        # its restricted set), so the planner must NOT emit UPDATE_FV
        # for these fields on an RTFV.
        local = _realtime_fv_authoring_dict(warehouse="WH_NEW", refresh_freq="10 minutes")
        # Applied mirrors the canonical RTFV ``online_store_type`` so the
        # operational-drift check focuses on warehouse / refresh_freq;
        # without this the default-online divergence would mask the
        # field-membership contract under test.
        applied = {
            "kind": "RealtimeFeatureView",
            "online_store_type": "postgres",
            "spec": {},
        }
        assert (
            _split_operational_vs_structural(
                local,
                applied,
                operational_fields=operational_fields,
                database="DB",
                schema="SCH",
            )
            is False
        ), (
            "RTFV warehouse / refresh_freq edits must not register as "
            "operational drift — the imperative update_feature_view would "
            "reject them (A5).  These changes flow through the structural "
            "RECREATE_FV path instead."
        )


# ---------------------------------------------------------------------------
# Phase 3 RED — refresh_freq drift routes through UPDATE_FV (not RECREATE_FV)
# ---------------------------------------------------------------------------


def _bfv_with_refresh_freq_authoring_dict(
    *,
    name: str = "BFV_RF_DRIFT",
    version: str = "V1",
    refresh_freq: str = "5 minutes",
) -> dict[str, Any]:
    """BatchFV authoring dict using the renamed ``refresh_freq`` key."""
    return {
        "kind": "BatchFeatureView",
        "name": name,
        "version": version,
        "database": "DB",
        "schema_": "SCH",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": refresh_freq,
    }


class TestRefreshFreqDriftRoutesAsUpdateFv:
    """A BFV that drifts only on the renamed ``refresh_freq`` field must
    route through ``UPDATE_FV`` (operational, non-destructive) — not
    ``RECREATE_FV``.

    Two preconditions:
    * ``_OPERATIONAL_FV_KEYS`` lists ``refresh_freq`` so the structural
      hash is stable across cadence edits.
    * The planner's ``_refresh_freq_drifted`` helper reads from the
      renamed local-side authoring key.

    Without both halves the planner falls through to ``RECREATE_FV`` (the
    pre-rename L4 regression).  This pin closes Phase 3's planner /
    invariants surface for the rename.
    """

    def test_refresh_freq_only_change_emits_update_fv(self) -> None:
        local_dict = _bfv_with_refresh_freq_authoring_dict(refresh_freq="10 minutes")
        applied_state = _applied_from_compiled(_bfv_with_refresh_freq_authoring_dict(refresh_freq="5 minutes"))
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "BFV_RF_DRIFT"]
        assert len(fv_ops) == 1, f"expected exactly one op for BFV_RF_DRIFT; got {fv_ops!r}"
        assert fv_ops[0].kind == OpKind.UPDATE_FV, (
            "BatchFV refresh_freq edits must route through UPDATE_FV (not "
            "RECREATE_FV) — the field is operational, alterable in-place "
            f"via fs.update_feature_view(refresh_freq=...). Got "
            f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})."
        )
        assert fv_ops[0].destructive is False, (
            "refresh_freq edits are non-destructive; UPDATE_FV must not " "carry destructive=True."
        )

    def test_refresh_freq_unchanged_emits_no_change(self) -> None:
        """Regression pin: identical authoring vs. applied refresh_freq
        must collapse to ``NO_CHANGE`` (the planner must not see phantom
        drift across the field rename).
        """
        local_dict = _bfv_with_refresh_freq_authoring_dict(refresh_freq="5 minutes")
        applied_state = _applied_from_compiled(local_dict)
        local_fv = FeatureView.model_validate(local_dict)
        plan = generate_plan(_batch(local_fv), applied_state, _opts())

        fv_ops = [op for op in plan.ops if op.name == "BFV_RF_DRIFT"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind == OpKind.NO_CHANGE, (
            "Identical refresh_freq on local + applied must emit "
            "NO_CHANGE; got "
            f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})."
        )


class TestRefreshFreqDriftStreamingCarveOut:
    """A tiled streaming FV's ``refresh_freq`` (the offline tile Dynamic
    Table cadence) must be compared against the applied *recovered*
    ``spec.refresh_freq`` — NOT the applied ``spec.target_lag_sec``, which
    for a streaming FV is the Online Feature Table's ingest lag that the
    Snowflake runtime always stamps to ``0``.

    Without this carve-out, a deployed tiled streaming FV authored
    ``refresh_freq="5 minutes"`` would drift ``300 != 0`` against the OFT
    sentinel on every replan and emit a spurious ``UPDATE_FV``.
    """

    def _drifted(self, local_refresh_freq: str, applied_refresh_freq: str | None) -> bool:
        from snowflake.ml.feature_store.decl.planner import _refresh_freq_drifted

        local = _streaming_fv_authoring_dict(refresh_freq=local_refresh_freq)
        # Make the local FV tiled so it is a shape that legitimately
        # carries refresh_freq (aggregation window).
        local["features"] = [
            {
                "function": "sum",
                "window_sec": 3600,
                "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                "output_column": {"name": "AMOUNT_1H", "type": "DoubleType"},
            }
        ]
        applied_spec: dict[str, Any] = {"target_lag_sec": 0}
        if applied_refresh_freq is not None:
            applied_spec["refresh_freq"] = applied_refresh_freq
        applied_payload = {"kind": "StreamingFeatureView", "spec": applied_spec}
        return _refresh_freq_drifted(local, applied_payload, "DB", "SCH")

    def test_identical_streaming_refresh_freq_is_not_drift(self) -> None:
        assert self._drifted("5 minutes", "5 minutes") is False, (
            "A tiled streaming FV whose recovered applied refresh_freq "
            "equals the local cadence must not drift — the OFT "
            "target_lag_sec=0 sentinel must not be read as the DT cadence."
        )

    def test_changed_streaming_refresh_freq_is_drift(self) -> None:
        assert self._drifted("10 minutes", "5 minutes") is True, (
            "A genuine tiled streaming refresh_freq edit must be detected " "as drift so the planner emits UPDATE_FV."
        )

    def test_unrecovered_streaming_cadence_is_not_spurious_drift(self) -> None:
        """When the applied side has not recovered ``refresh_freq`` (only
        the OFT ``target_lag_sec=0`` sentinel), the planner must NOT treat
        the local cadence as drift — that is the spurious-UPDATE_FV bug.
        """
        assert self._drifted("5 minutes", None) is False


# ---------------------------------------------------------------------------
# PR A — cosmetic SQL edits on a query-backed BatchFV route to NO_CHANGE
# ---------------------------------------------------------------------------


def _bfv_query_backed_authoring_dict(
    *,
    name: str = "BFV_QUERY_BACKED",
    version: str = "V1",
    query: str = "SELECT USER_ID FROM ORDERS_RAW",
) -> dict[str, Any]:
    """BatchFV authoring dict whose single source binds on an inline SQL query."""
    return {
        "kind": "BatchFeatureView",
        "name": name,
        "version": version,
        "database": "DB",
        "schema_": "SCH",
        "online": False,
        "refresh_freq": "1 hour",
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "query": query,
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
    }


class TestQueryBackedBatchFvCosmeticSqlEdits:
    """A comment / keyword-case / whitespace-only edit to a query-backed
    ``BatchFeatureView`` must route to ``NO_CHANGE`` — the SQL is
    canonicalized at the hash boundary so the destructive ``RECREATE_FV``
    (which would drop the companion ``$SNAPSHOTS``) is not emitted.  A
    genuine query change still recreates."""

    def _plan_kinds_for(self, deployed_query: str, local_query: str) -> list[OpKind]:
        applied_state = _applied_from_compiled(_bfv_query_backed_authoring_dict(query=deployed_query))
        local_fv = FeatureView.model_validate(_bfv_query_backed_authoring_dict(query=local_query))
        plan = generate_plan(_batch(local_fv), applied_state, _opts())
        return [op.kind for op in plan.ops if op.name == "BFV_QUERY_BACKED"]

    def test_comment_only_query_edit_is_no_change(self) -> None:
        kinds = self._plan_kinds_for(
            "SELECT USER_ID FROM ORDERS_RAW",
            "SELECT USER_ID FROM ORDERS_RAW -- add a comment",
        )
        assert OpKind.NO_CHANGE in kinds, f"comment-only SQL edit must be NO_CHANGE; got {kinds}"
        assert OpKind.RECREATE_FV not in kinds

    def test_keyword_case_only_query_edit_is_no_change(self) -> None:
        kinds = self._plan_kinds_for(
            "SELECT USER_ID FROM ORDERS_RAW",
            "select USER_ID from ORDERS_RAW",
        )
        assert OpKind.NO_CHANGE in kinds, f"keyword-case-only SQL edit must be NO_CHANGE; got {kinds}"
        assert OpKind.RECREATE_FV not in kinds

    def test_whitespace_only_query_edit_is_no_change(self) -> None:
        kinds = self._plan_kinds_for(
            "SELECT USER_ID FROM ORDERS_RAW",
            "SELECT USER_ID\n   FROM   ORDERS_RAW",
        )
        assert OpKind.NO_CHANGE in kinds, f"whitespace-only SQL edit must be NO_CHANGE; got {kinds}"
        assert OpKind.RECREATE_FV not in kinds

    def test_semantic_query_edit_still_recreates(self) -> None:
        kinds = self._plan_kinds_for(
            "SELECT USER_ID FROM ORDERS_RAW",
            "SELECT USER_ID, TOTAL FROM ORDERS_RAW",
        )
        assert OpKind.RECREATE_FV in kinds, f"a genuine query change must still RECREATE_FV; got {kinds}"


if __name__ == "__main__":
    pytest_driver.main()
