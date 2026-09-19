"""Round-trip coverage for an append-only BatchFeatureView CRON ``refresh_freq``.

An append-only BFV is *required* by ``spec_models._validate_append_only`` to
carry a CRON ``refresh_freq`` (e.g. ``"0 0 * * * UTC"``).  A CRON cadence drives
the offline Dynamic Table via ``TARGET_LAG = 'DOWNSTREAM'`` plus a companion
Task — there is no numeric per-refresh cadence, and
``compiler.parse_duration_to_seconds`` raises ``Invalid interval format`` on the
CRON string.

Before the guard, ``spec_compiler.compile_to_spec`` parsed the required CRON
``refresh_freq`` unconditionally as a duration and raised.  The planner's
full-spec diff (``compute_local_spec_hash``) then degraded to a structural
fingerprint that can never equal the applied ``_full_spec_hash``, emitting a
destructive ``RECREATE_FV`` on every plan and silently dropping the accumulated
``$SNAPSHOTS`` point-in-time history.

These tests pin: the compiler skips the duration parse for a CRON cadence, a
plain duration cadence still populates ``target_lag_sec``, and an unedited
append-only BFV replans as ``NO_CHANGE`` — both against a compiled applied
payload and against the production ``state._build_offline_fv_object`` recovery
path.
"""

from __future__ import annotations

import json
from typing import Any

from snowflake.ml.feature_store.decl import state
from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    compute_local_spec_hash,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FSColumn,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver


def _minimal_append_only_authoring(**overrides: Any) -> dict[str, Any]:
    """Build a valid append-only BatchFeatureView authoring dict.

    Args:
        **overrides: Field values that replace the valid-baseline defaults.

    Returns:
        A dict suitable for ``FeatureView.model_validate`` / ``compile_to_spec``.
    """
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_AO",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "timestamp_col": "EVENT_TS",
        "refresh_mode": "FULL",
        "refresh_freq": "0 0 * * * UTC",
        "append_only": True,
    }
    base.update(overrides)
    return base


def _entity_and_source() -> tuple[Entity, BatchSource]:
    """Return the Entity + BatchSource that back ``_minimal_append_only_authoring``."""
    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    return ent, src


def _applied_state_for(local: dict[str, Any]) -> AppliedState:
    """Build an AppliedState whose payload is the compiled local spec."""
    import copy

    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = "BatchFeatureView:DB1.SC1:BFV_AO:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_AO",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_compile_to_spec_does_not_raise_on_cron_refresh_freq() -> None:
    """A CRON ``refresh_freq`` compiles without raising and stamps no ``target_lag_sec``."""
    compiled = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
    assert isinstance(compiled, dict)
    assert "target_lag_sec" not in compiled.get("spec", {})


def test_compute_local_spec_hash_does_not_raise_on_cron() -> None:
    """``compute_local_spec_hash`` returns a digest instead of raising on CRON."""
    data = _minimal_append_only_authoring()
    digest = compute_local_spec_hash(data, "DB1", "SC1")
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_duration_refresh_freq_still_compiles_target_lag_sec() -> None:
    """A plain duration cadence still populates ``target_lag_sec`` (guard is narrow)."""
    plain = _minimal_append_only_authoring(append_only=False, refresh_freq="5 minutes")
    plain.pop("refresh_mode")
    compiled = compile_to_spec(plain, "DB1", "SC1")
    assert compiled["spec"]["target_lag_sec"] == 300


def test_append_only_bfv_replan_is_no_change() -> None:
    """An unedited append-only BFV replans as NO_CHANGE against a compiled payload.

    Regression: compile_to_spec parsed the required CRON refresh_freq as a
    duration and raised, degrading the planner to a structural fingerprint that
    can never match the applied full-spec hash.
    """
    local = _minimal_append_only_authoring()
    applied = _applied_state_for(local)
    ent, src = _entity_and_source()
    fv = FeatureView.model_validate(local)
    plan = generate_plan(
        SpecBatch(specs=[ent, src, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.NO_CHANGE


def test_append_only_bfv_recovered_state_replan_is_no_change() -> None:
    """The production recovery path replans as NO_CHANGE, not destructive RECREATE_FV.

    Builds the applied object via ``state._build_offline_fv_object`` from a
    list-FV row carrying a CRON ``refresh_freq``, then confirms the recovered
    ``content_hash`` matches the local compile hash and the plan is NO_CHANGE.
    """
    local = _minimal_append_only_authoring()
    compiled = compile_to_spec(local, "DB1", "SC1")
    local_hash = compute_local_spec_hash(local, "DB1", "SC1")

    row = {
        "name": "BFV_AO",
        "version": "V1",
        "database_name": "DB1",
        "schema_name": "SC1",
        "kind": "BATCH",
        "entities": ["USER_ID"],
        "physical_dt_name": "BFV_AO$V1",
        "refresh_freq": "0 0 * * * UTC",
        "target_lag": "",
        "refresh_mode": "FULL",
        "source_refs": json.dumps({"source_refs": [{"name": "SRC1", "table": "RAW_EVENTS"}]}),
        "spec_text": compiled,
    }
    applied_obj = state._build_offline_fv_object(row, default_database="DB1", default_schema="SC1")
    assert applied_obj is not None
    assert applied_obj.content_hash == local_hash

    applied = AppliedState(objects={applied_obj.key: applied_obj})
    ent, src = _entity_and_source()
    fv = FeatureView.model_validate(local)
    plan = generate_plan(
        SpecBatch(specs=[ent, src, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.NO_CHANGE


if __name__ == "__main__":
    pytest_driver.main()
