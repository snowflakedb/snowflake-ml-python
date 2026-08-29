"""Tests for refresh_mode asymmetric normalization.

Covers the three surfaces where Snowflake's resolved ``INCREMENTAL``/``FULL``
value on the applied side must not produce a spurious diff when the operator
never authored ``refresh_mode``:

1. ``_normalize_applied_bfv_for_hash`` — strips the key from the applied
   payload before hashing when the local compiled spec omits it.
2. ``batch_feature_view_structural_equivalent`` — asymmetric ``_project()``
   strips ``refresh_mode`` from the applied projection when absent in local.
3. Planner NO_CHANGE — end-to-end: operator omits ``refresh_mode``, applied
   row carries ``INCREMENTAL``; second plan must be ``NO_CHANGE``.

All tests in this module are RED until the implementation ships in
``invariants.py`` and ``planner.py``.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    _normalize_applied_bfv_for_hash,
    batch_feature_view_structural_equivalent,
    model_to_dict,
    spec_key,
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

_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"
_FV_NAME = "MY_BATCH_FV_BATCH_DECL"
_FV_VERSION = "V1"
_SOURCE_TABLE = "RAW_EVENTS_BATCH_DECL"


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


def _minimal_bfv_authoring(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": _FV_NAME,
        "version": _FV_VERSION,
        "database": _DB,
        "schema": _SCH,
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": _SOURCE_TABLE,
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
        "refresh_freq": "1 minute",
    }
    base.update(overrides)
    base.pop("target_lag", None)
    base.pop("target_lag_sec", None)
    return base


def _minimal_applied_payload(*, refresh_mode: str = "INCREMENTAL") -> dict[str, Any]:
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": _FV_NAME,
            "version": _FV_VERSION,
            "database": _DB,
            "schema": _SCH,
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": _SOURCE_TABLE,
                    "source_type": "Batch",
                    "table": _SOURCE_TABLE,
                }
            ],
            "features": [],
            "target_lag_sec": 60,
            "refresh_freq": "1 minute",
            "refresh_mode": refresh_mode,
        },
    }


def _spec_batch(authoring: dict[str, Any]) -> SpecBatch:
    ent = Entity(
        kind="Entity",
        name="USER",
        database=_DB,
        schema_=_SCH,
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="EVENTS_BATCH_DECL",
        database=_DB,
        schema_=_SCH,
        table=_SOURCE_TABLE,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
            FSColumn(name="METRIC_VAL", type="FloatType"),
        ],
    )
    fv = FeatureView.model_validate(authoring)
    return SpecBatch(specs=[ent, src, fv])


def _applied_state_incremental() -> AppliedState:
    """Build the deployed-state snapshot for the BFV as Snowflake resolves it.

    Constructs the ``AppliedState`` directly (rather than via
    ``state.fetch_applied_state``) so this module has no dependency on the
    layer-3 ``decl.state`` package.  The applied spec_payload is produced by
    the same ``compile_to_spec`` pipeline the planner uses for the local side,
    with the Snowflake-resolved ``refresh_mode: INCREMENTAL`` stamped on — so
    the planner's ``compute_local_spec_hash`` / ``_full_spec_hash`` comparison
    (and the asymmetric ``_normalize_applied_bfv_for_hash`` normalisation) is
    exercised end-to-end.

    Returns:
        An ``AppliedState`` with a single BatchFeatureView ``AppliedObject``
        (``from_specification=True``) whose spec_payload carries
        ``refresh_mode=INCREMENTAL``.
    """
    applied_data = model_to_dict(FeatureView.model_validate(_minimal_bfv_authoring(refresh_mode="INCREMENTAL")))
    applied_payload = compile_to_spec(applied_data, _DB, _SCH)
    key = spec_key(applied_data, database=_DB, schema=_SCH)
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name=_FV_NAME,
                version=_FV_VERSION,
                content_hash=_full_spec_hash(applied_payload),
                spec_payload=applied_payload,
                from_specification=True,
            )
        }
    )


# ---------------------------------------------------------------------------
# _normalize_applied_bfv_for_hash
# ---------------------------------------------------------------------------


class TestNormalizeAppliedBfvForHash:
    """Unit tests for the ``_normalize_applied_bfv_for_hash`` helper."""

    def test_strips_refresh_mode_when_absent_locally(self) -> None:
        """When local compiled spec has no ``refresh_mode`` key, the
        function must strip it from the applied payload copy."""
        applied = {"spec": {"refresh_mode": "INCREMENTAL", "other": "val"}}
        local = {"spec": {"other": "val"}}
        result = _normalize_applied_bfv_for_hash(applied, local)
        assert "refresh_mode" not in result["spec"]

    def test_keeps_refresh_mode_when_present_locally(self) -> None:
        """When the local compiled spec has ``refresh_mode``, the applied
        payload must retain it so the values can be compared."""
        applied = {"spec": {"refresh_mode": "INCREMENTAL", "other": "val"}}
        local = {"spec": {"refresh_mode": "INCREMENTAL", "other": "val"}}
        result = _normalize_applied_bfv_for_hash(applied, local)
        assert result["spec"]["refresh_mode"] == "INCREMENTAL"

    def test_keeps_when_local_authors_full_applied_has_incremental(self) -> None:
        """Local FULL vs applied INCREMENTAL — key must be kept so the
        values remain visible for comparison (-> will differ -> RECREATE_FV)."""
        applied = {"spec": {"refresh_mode": "INCREMENTAL"}}
        local = {"spec": {"refresh_mode": "FULL"}}
        result = _normalize_applied_bfv_for_hash(applied, local)
        assert result["spec"]["refresh_mode"] == "INCREMENTAL"

    def test_does_not_mutate_input(self) -> None:
        """The function must return a deep copy — the original applied
        payload must be unchanged."""
        applied = {"spec": {"refresh_mode": "INCREMENTAL"}}
        local: dict[str, Any] = {"spec": {}}
        _normalize_applied_bfv_for_hash(applied, local)
        assert "refresh_mode" in applied["spec"]

    def test_no_op_when_no_spec_key(self) -> None:
        """Applied payload without a ``spec`` dict must pass through unchanged."""
        applied: dict[str, Any] = {"kind": "BatchFeatureView"}
        local: dict[str, Any] = {}
        result = _normalize_applied_bfv_for_hash(applied, local)
        assert result == applied

    def test_other_keys_preserved(self) -> None:
        """Only ``refresh_mode`` is stripped; all other keys pass through."""
        applied = {"spec": {"refresh_mode": "INCREMENTAL", "target_lag_sec": 300}}
        local = {"spec": {"target_lag_sec": 300}}
        result = _normalize_applied_bfv_for_hash(applied, local)
        assert result["spec"]["target_lag_sec"] == 300
        assert "refresh_mode" not in result["spec"]


# ---------------------------------------------------------------------------
# batch_feature_view_structural_equivalent — asymmetric refresh_mode
# ---------------------------------------------------------------------------


class TestBatchFvStructuralEquivalentAsymmetric:
    """Pins the asymmetric ``refresh_mode`` normalisation in
    ``batch_feature_view_structural_equivalent``."""

    def test_equivalent_when_local_omits_and_applied_has_incremental(self) -> None:
        """Local spec omits ``refresh_mode``; applied (SHOW) reports INCREMENTAL.
        Snowflake stamps INCREMENTAL on every DT even when the operator never
        authored the field. The function must return True (structurally
        equivalent) rather than False (spurious structural diff)."""
        local = _minimal_bfv_authoring()  # no refresh_mode
        applied = _minimal_applied_payload(refresh_mode="INCREMENTAL")
        assert batch_feature_view_structural_equivalent(local, applied, _DB, _SCH) is True

    def test_not_equivalent_when_local_authors_full_and_applied_has_incremental(self) -> None:
        """When the operator explicitly pins FULL but applied is INCREMENTAL,
        the function must return False — this is a genuine structural change
        that requires RECREATE_FV."""
        local = _minimal_bfv_authoring(refresh_mode="FULL")
        applied = _minimal_applied_payload(refresh_mode="INCREMENTAL")
        assert batch_feature_view_structural_equivalent(local, applied, _DB, _SCH) is False

    def test_equivalent_when_both_omit_refresh_mode(self) -> None:
        """Neither side carries ``refresh_mode`` — no diff, returns True."""
        local = _minimal_bfv_authoring()
        applied = _minimal_applied_payload()
        del applied["spec"]["refresh_mode"]
        assert batch_feature_view_structural_equivalent(local, applied, _DB, _SCH) is True

    def test_equivalent_when_both_author_incremental(self) -> None:
        """Both sides explicitly carry INCREMENTAL — match, returns True."""
        local = _minimal_bfv_authoring(refresh_mode="INCREMENTAL")
        applied = _minimal_applied_payload(refresh_mode="INCREMENTAL")
        assert batch_feature_view_structural_equivalent(local, applied, _DB, _SCH) is True


# ---------------------------------------------------------------------------
# Planner NO_CHANGE regression — end-to-end
# ---------------------------------------------------------------------------


class TestPlannerRefreshModeNormalization:
    """End-to-end planner tests: refresh_mode round-trip must produce NO_CHANGE."""

    def test_no_change_when_local_omits_and_applied_has_incremental(self) -> None:
        """Scenario: operator never authors ``refresh_mode``; after apply
        Snowflake stamps INCREMENTAL on the DT. The second plan must be
        NO_CHANGE — not RECREATE_FV."""
        local = _minimal_bfv_authoring()  # no refresh_mode
        state = _applied_state_incremental()
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "NO_CHANGE", (
            f"Expected NO_CHANGE but got {fv_ops[0].kind.value} "
            f"(reason: {fv_ops[0].reason}). "
            "Snowflake stamps INCREMENTAL on the DT when the operator omits "
            "refresh_mode; the planner must normalise the applied hash."
        )

    def test_recreate_when_local_authors_full_and_applied_has_incremental(self) -> None:
        """Operator explicitly pins FULL; applied is INCREMENTAL.
        The planner must emit RECREATE_FV — a genuine intent change."""
        local = _minimal_bfv_authoring(refresh_mode="FULL")
        state = _applied_state_incremental()
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "RECREATE_FV", f"Expected RECREATE_FV but got {fv_ops[0].kind.value}."

    def test_no_change_when_both_author_incremental(self) -> None:
        """Operator pins INCREMENTAL; Snowflake also reports INCREMENTAL.
        Must be NO_CHANGE."""
        local = _minimal_bfv_authoring(refresh_mode="INCREMENTAL")
        state = _applied_state_incremental()
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "NO_CHANGE", f"Expected NO_CHANGE but got {fv_ops[0].kind.value}."


if __name__ == "__main__":
    pytest_driver.main()
