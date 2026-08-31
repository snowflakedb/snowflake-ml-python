"""Planner tests for offline-only BatchFeatureView replan idempotency.

Pins hypotheses H2 and H5 from
``plans/offline_bfv_state_fix_b9da0006.plan.md``.

H2 — pre-existing pin: when ``applied is None`` (lookup miss), the
planner correctly emits ``CREATE_FV`` with the documented reason.
This test must keep passing — it is the negative-control that locks
the planner's contract while the fix happens upstream in
``state.fetch_applied_state``.

H5 — RED until Phase 2: with a reconstructed offline-BFV
``AppliedObject`` in the applied state, ``generate_plan`` emits the
correct op kind for each scenario:

- identical authoring spec → ``NO_CHANGE``
- ``refresh_freq`` change → ``UPDATE_FV`` (non-destructive)
- ``sources[].table`` change → ``RECREATE_FV`` (destructive)
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FSColumn,
)
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import PlanOptions, SpecBatch
from snowflake.ml.test_utils import pytest_driver

_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"
_FV_NAME = "MY_BATCH_FV_BATCH_DECL"
_FV_VERSION = "V1"
_OFFLINE_DT_NAME = f"{_FV_NAME}${_FV_VERSION}"
_SOURCE_TABLE = "RAW_EVENTS_BATCH_DECL"
_SOURCE_TABLE_V2 = "RAW_EVENTS_BATCH_DECL_V2"


def _offline_bfv_authoring(**overrides: Any) -> dict[str, Any]:
    """Return the authoring-side BFV dict with overrides applied.

    Args:
        **overrides: Keys to override on the base dict (e.g.
            ``refresh_freq``, ``sources``).

    Returns:
        Authoring-side BFV dict.
    """
    base = {
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
        "target_lag": "1 minute",
        "refresh_freq": "1 minute",
    }
    base.update(overrides)
    # ``target_lag`` is OFT-only after the decoupling; strip it on
    # offline-only fixtures so the validator does not reject the dict.
    if not base.get("online", False):
        base.pop("target_lag", None)
        base.pop("target_lag_sec", None)
    return base


def _imperative_list_fv_row(
    *,
    target_lag: str = "1 minute",
    source_table: str = _SOURCE_TABLE,
) -> dict[str, Any]:
    """Phase 1 contract row shape — see Section 7 of the plan.

    Phase B3: ``source_refs`` carries the authoritative
    :class:`FvSourceRefsMetadata` payload so
    :func:`state._inject_batch_fv_source_from_metadata` can populate
    ``spec.sources[]`` directly — no DT-text parsing.

    Args:
        target_lag: Snowflake target-lag string for both ``target_lag``
            and ``refresh_freq`` row fields.
        source_table: Source-table identifier the deployed BFV's
            ``BatchSource`` points at (mirrors the recovered
            ``spec.sources[0].table``).

    Returns:
        Row dict in the Phase-1 contract shape.
    """
    return {
        "name": _FV_NAME,
        "version": _FV_VERSION,
        "database_name": _DB,
        "schema_name": _SCH,
        "kind": "BATCH",
        "entities": ["USER_ID"],
        "online_enabled": False,
        "target_lag": target_lag,
        "refresh_freq": target_lag,
        "warehouse": "TEST_WH",
        "cluster_by": "",
        "refresh_mode": "",
        "desc": "",
        "physical_dt_name": _OFFLINE_DT_NAME,
        "source_refs": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": source_table,
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
    }


def _offline_dt_text(source_table: str = _SOURCE_TABLE) -> str:
    """Render the SHOW DYNAMIC TABLES ``text`` for the offline DT.

    Args:
        source_table: Source-table identifier the BFV reads from.

    Returns:
        DDL string mirroring the live ``text`` column for the offline DT.
    """
    return (
        f"CREATE DYNAMIC TABLE {_DB}.{_SCH}.{_OFFLINE_DT_NAME}\n"
        "TARGET_LAG = '1 minute'\n"
        "WAREHOUSE = TEST_WH\n"
        f"AS SELECT * FROM {_DB}.{_SCH}.{source_table}"
    )


def _spec_batch(authoring: dict[str, Any]) -> SpecBatch:
    """Build a SpecBatch from an authoring dict — entity, source, FV.

    Args:
        authoring: Authoring-side BFV dict.

    Returns:
        ``SpecBatch`` containing the entity, the batch source, and the
        FV ready for :func:`generate_plan`.
    """
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
        table=authoring["sources"][0].get("table", _SOURCE_TABLE),
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
            FSColumn(name="METRIC_VAL", type="FloatType"),
        ],
    )
    fv = FeatureView.model_validate(authoring)
    return SpecBatch(specs=[ent, src, fv])


def _applied_state_from_imperative_path(
    *,
    target_lag: str = "1 minute",
    source_table: str = _SOURCE_TABLE,
) -> Any:
    """Build AppliedState via the new ``feature_view_rows`` path.

    Args:
        target_lag: Snowflake target-lag string for the deployed DT.
        source_table: Source-table identifier the deployed DT reads from.

    Returns:
        ``AppliedState`` reconstructed via the Phase-2
        ``feature_view_rows`` merge path.
    """
    return fetch_applied_state(
        raw_show_results=[],
        raw_table_results=[],
        specification_map={},
        entity_rows=[],
        dt_text_map={_OFFLINE_DT_NAME: _offline_dt_text(source_table)},
        feature_view_rows=[_imperative_list_fv_row(target_lag=target_lag, source_table=source_table)],
        default_database=_DB,
        default_schema=_SCH,
    )


# ===========================================================================
# H2 — Lookup miss yields CREATE_FV (pre-existing pin)
# ===========================================================================


class TestPlannerCreateFvOnLookupMiss:
    def test_offline_bfv_create_fv_when_applied_state_empty(self) -> None:
        """When ``fetch_applied_state`` returns nothing for the FV
        (the bug repro), the planner emits exactly one
        ``CREATE_FV`` with the documented "not found" reason — this
        is the pre-existing planner contract that the fix MUST NOT
        break.  The fix happens upstream by populating applied
        state, not by changing the planner's miss handling.
        """
        local = _offline_bfv_authoring()
        empty_state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={},
            default_database=_DB,
            default_schema=_SCH,
        )
        plan = generate_plan(
            _spec_batch(local),
            empty_state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "CREATE_FV"
        assert "not found in applied state" in fv_ops[0].reason


# ===========================================================================
# H5 — Replan idempotency for offline-only BFV
# ===========================================================================


class TestPlannerOfflineBfvReplanIdempotency:
    """RED until Phase 2 — with the fix in place,
    ``plan → apply → plan`` over an unchanged offline-only BFV
    produces ``NO_CHANGE`` on the second plan.
    """

    def test_no_change_on_clean_round_trip(self) -> None:
        local = _offline_bfv_authoring()
        state = _applied_state_from_imperative_path()
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
            f"Expected NO_CHANGE for offline-only BFV on clean round-trip, "
            f"got {fv_ops[0].kind.value} (reason: {fv_ops[0].reason})"
        )

    def test_update_fv_when_only_refresh_freq_changes(self) -> None:
        """Editing only ``refresh_freq`` (operational knob) must
        produce a non-destructive ``UPDATE_FV``."""
        local = _offline_bfv_authoring(refresh_freq="2 minutes")
        # Applied side reflects the deployed cadence (1 minute).
        state = _applied_state_from_imperative_path(target_lag="1 minute")
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "UPDATE_FV"
        assert fv_ops[0].destructive is False

    def test_no_change_preserves_destructive_flag_false(self) -> None:
        """A NO_CHANGE op must never carry ``destructive=True`` —
        replan idempotency would surface as a destructive operation
        in the rendered plan otherwise.
        """
        local = _offline_bfv_authoring()
        state = _applied_state_from_imperative_path()
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert all(op.destructive is False for op in fv_ops)

    def test_recreate_fv_when_source_table_changes(self) -> None:
        """Editing ``sources[0].table`` (structural change) must
        produce a destructive ``RECREATE_FV``."""
        local = _offline_bfv_authoring(
            sources=[
                {
                    "name": "EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": _SOURCE_TABLE_V2,
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                }
            ]
        )
        # Applied side still points at the old table.
        state = _applied_state_from_imperative_path(source_table=_SOURCE_TABLE)
        plan = generate_plan(
            _spec_batch(local),
            state,
            PlanOptions(),
            database=_DB,
            schema=_SCH,
        )
        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "RECREATE_FV"
        assert fv_ops[0].destructive is True


if __name__ == "__main__":
    pytest_driver.main()
