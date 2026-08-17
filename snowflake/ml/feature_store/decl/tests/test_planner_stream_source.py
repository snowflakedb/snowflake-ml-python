"""Planner tests for the four-way source decision + DROP_SOURCE orphan pass.

Pins Wave 3A of ``plans/stream_source_contract.md`` §7 — the planner
must distinguish four outcomes when reconciling a local source spec
against the new runtime-derived applied-state ``Datasource`` entry that
:func:`state.fetch_applied_state` now produces from the
``stream_source_rows=`` kwarg added in commit ``a31cc6588``:

* ``applied is None`` + source NOT referenced by any deployed FV →
  ``CREATE_SOURCE`` (non-destructive; new object).
* ``applied is None`` + source referenced by a deployed FV → virtual
  override → ``NO_CHANGE`` (today's :func:`_sources_with_deployed_fv`
  behaviour; regression-pinned).
* ``applied is not None`` + :func:`compute_source_diff_kind` returns
  ``"no_change"`` → ``NO_CHANGE``.
* ``applied is not None`` + ``"update_desc_only"`` → ``UPDATE_SOURCE``
  (non-destructive).
* ``applied is not None`` + ``"recreate"`` → ``RECREATE_SOURCE``
  (destructive; ``--allow-recreate``).

The orphan-drop pass adds a sixth contract: any applied ``Datasource``
key that is not present in the local batch must emit ``DROP_SOURCE``
in full-sync mode (and stay silent in incremental mode), carrying a
``payload.kind`` of ``"StreamingSource"`` or ``"BatchSource"`` derived
from ``applied.spec_payload["source_type"]`` so the imperative
executor can route to the right delete branch.

The user-reported regression: ``CLICKSTREAM_EVENTS`` showed up as
``CREATE_SOURCE`` on every plan because before Wave 1+2 the planner's
``Datasource:DB.SCHEMA:CLICKSTREAM_EVENTS`` lookup against applied
state always missed (no runtime row was ever fetched). With Wave 1+2
landed, the applied entry now exists; this wave teaches the planner
to compare the two payloads via :func:`compute_source_diff_kind` and
emit ``NO_CHANGE`` for a clean round-trip.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import structural_fingerprint_hash
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FSColumn,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.state import _build_stream_source_object
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    ObjectKind,
    PlanOptions,
    SpecBatch,
)

_DB = "DB1"
_SCHEMA = "SC1"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _entity_user() -> Entity:
    """Minimal Entity referenced by every FV in this file."""
    return Entity(
        kind="Entity",
        name="USER",
        database=_DB,
        schema_=_SCHEMA,
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )


def _streaming_source(*, name: str = "CLICKSTREAM_EVENTS") -> StreamingSource:
    """A minimal StreamingSource that round-trips cleanly through the diff helper."""
    return StreamingSource(
        kind="StreamingSource",
        name=name,
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )


def _batch_source(*, name: str = "EVENTS_BATCH", table: str = "RAW_EVENTS") -> BatchSource:
    return BatchSource(
        kind="BatchSource",
        name=name,
        database=_DB,
        schema_=_SCHEMA,
        table=table,
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )


def _applied_datasource_payload(
    *,
    name: str,
    source_type: str = "Stream",
    columns: list[dict[str, Any]] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Mirror of :func:`state._build_stream_source_object` spec_payload."""
    if columns is None:
        columns = [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "VALUE", "type": "FloatType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ]
    return {
        "kind": ObjectKind.DATASOURCE,
        "name": name.upper(),
        "database": _DB,
        "schema": _SCHEMA,
        "source_type": source_type,
        "columns": columns,
        "description": description,
    }


def _applied_datasource(
    *,
    name: str,
    source_type: str = "Stream",
    columns: list[dict[str, Any]] | None = None,
    description: str = "",
) -> AppliedObject:
    payload = _applied_datasource_payload(
        name=name,
        source_type=source_type,
        columns=columns,
        description=description,
    )
    key = f"{ObjectKind.DATASOURCE}:{_DB}.{_SCHEMA}:{name.upper()}"
    return AppliedObject(
        key=key,
        kind=ObjectKind.DATASOURCE,
        name=name.upper(),
        version=None,
        content_hash=structural_fingerprint_hash(payload),
        spec_payload=payload,
        columns=[],
        from_specification=True,
        details={"source_type": source_type, "column_count": len(payload["columns"])},
    )


def _ops_by_name(plan: Any, name: str) -> list[Any]:
    return [op for op in plan.ops if op.name == name.upper() or op.name == name]


# ---------------------------------------------------------------------------
# Four-way decision
# ---------------------------------------------------------------------------


class TestPlanSourceFourWayDecision:
    """Pins the four-way source decision contract from §7b."""

    def test_no_applied_no_deployed_fv_emits_create_source(self) -> None:
        """``applied is None`` + no deployed FV references the source → CREATE_SOURCE."""
        src = _streaming_source(name="NEW_STREAM_SRC")
        batch = SpecBatch(specs=[_entity_user(), src])
        applied = AppliedState(objects={})

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)

        src_ops = _ops_by_name(plan, "NEW_STREAM_SRC")
        assert len(src_ops) == 1, f"expected one op for NEW_STREAM_SRC; got {[op.kind.value for op in plan.ops]}"
        assert src_ops[0].kind == OpKind.CREATE_SOURCE
        assert src_ops[0].destructive is False

    def test_no_applied_but_deployed_fv_is_virtual_no_change(self) -> None:
        """``applied is None`` + a deployed FV references the source → virtual NO_CHANGE.

        Regression-pin: the existing virtual-source override must keep
        firing for sources that have no runtime row yet but whose
        referencing FV is already deployed (BatchSource path).
        """
        src = BatchSource(
            kind="BatchSource",
            name="EVENTS_BATCH_DECL",
            database=_DB,
            schema_=_SCHEMA,
            table="RAW_EVENTS_BATCH_DECL",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        )
        # Build a FV that references the source so _sources_with_deployed_fv hits.
        fv_dict = {
            "kind": "BatchFeatureView",
            "name": "MY_BATCH_FV",
            "version": "V1",
            "database": _DB,
            "schema": _SCHEMA,
            "online": True,
            "target_lag_sec": 60,
            "entities": ["USER_ID"],
            "sources": [{"name": "EVENTS_BATCH_DECL", "source_type": "Batch"}],
            "refresh_freq": "1 minute",
        }
        from snowflake.ml.feature_store.decl.spec_models import FeatureView

        fv = FeatureView.model_validate(fv_dict)
        batch = SpecBatch(specs=[_entity_user(), src, fv])

        # Mark the FV as deployed so _sources_with_deployed_fv picks it up,
        # but leave the source's applied entry absent so we exercise the
        # ``applied is None`` arm of the four-way decision.
        fv_payload = {"kind": "BatchFeatureView", "name": "MY_BATCH_FV", "version": "V1"}
        applied = AppliedState(
            objects={
                f"BatchFeatureView:{_DB}.{_SCHEMA}:MY_BATCH_FV": AppliedObject(
                    key=f"BatchFeatureView:{_DB}.{_SCHEMA}:MY_BATCH_FV",
                    kind="BatchFeatureView",
                    name="MY_BATCH_FV",
                    version="V1",
                    content_hash="deadbeef",
                    spec_payload=fv_payload,
                    from_specification=False,
                )
            }
        )

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "EVENTS_BATCH_DECL")
        assert len(src_ops) == 1
        assert src_ops[0].kind == OpKind.NO_CHANGE
        # The virtual-override reason must remain identifiable.
        reason_low = src_ops[0].reason.lower()
        assert "virtual" in reason_low or "feature view" in reason_low, src_ops[0].reason

    def test_applied_identical_payload_returns_no_change(self) -> None:
        """``applied is not None`` + identical spec_payload → NO_CHANGE.

        The four-way decision routes through
        :func:`compute_source_diff_kind`, which returns ``"no_change"``
        for an identical structural+description round-trip.  The reason
        string must clearly distinguish this path from the legacy
        structural-hash NO_CHANGE so plan readers can tell which arm
        fired.
        """
        src = _streaming_source(name="CLICKSTREAM_EVENTS")
        batch = SpecBatch(specs=[_entity_user(), src])
        applied = AppliedState(
            objects={_applied_datasource(name="CLICKSTREAM_EVENTS").key: _applied_datasource(name="CLICKSTREAM_EVENTS")}
        )

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
        assert len(src_ops) == 1
        assert src_ops[0].kind == OpKind.NO_CHANGE
        assert src_ops[0].destructive is False
        assert "source-level" in src_ops[0].reason.lower(), (
            f"NO_CHANGE for an applied-state source must use the source-level reason; " f"got {src_ops[0].reason!r}"
        )

    def test_applied_desc_only_change_emits_update_source(self) -> None:
        """``applied is not None`` + description differs only → UPDATE_SOURCE (non-destructive)."""
        src = _streaming_source(name="CLICKSTREAM_EVENTS")
        batch = SpecBatch(specs=[_entity_user(), src])
        applied = AppliedState(
            objects={
                _applied_datasource(name="CLICKSTREAM_EVENTS", description="updated description").key: (
                    _applied_datasource(name="CLICKSTREAM_EVENTS", description="updated description")
                )
            }
        )

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
        assert len(src_ops) == 1
        assert src_ops[0].kind == OpKind.UPDATE_SOURCE, f"expected UPDATE_SOURCE; got {src_ops[0].kind.value}"
        assert src_ops[0].destructive is False

    def test_applied_columns_differ_emits_recreate_source(self) -> None:
        """``applied is not None`` + columns differ → RECREATE_SOURCE (destructive)."""
        src = _streaming_source(name="CLICKSTREAM_EVENTS")
        batch = SpecBatch(specs=[_entity_user(), src])
        # Drop one column from the applied side so the structural fingerprint
        # mismatches and ``compute_source_diff_kind`` returns ``"recreate"``.
        applied_obj = _applied_datasource(
            name="CLICKSTREAM_EVENTS",
            columns=[
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
            ],
        )
        applied = AppliedState(objects={applied_obj.key: applied_obj})

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
        assert len(src_ops) == 1
        assert src_ops[0].kind == OpKind.RECREATE_SOURCE, f"expected RECREATE_SOURCE; got {src_ops[0].kind.value}"
        assert src_ops[0].destructive is True

    def test_applied_source_type_differs_emits_recreate_source(self) -> None:
        """``applied is not None`` + ``source_type`` differs → RECREATE_SOURCE."""
        src = _streaming_source(name="CLICKSTREAM_EVENTS")  # local kind StreamingSource → source_type Stream
        batch = SpecBatch(specs=[_entity_user(), src])
        applied_obj = _applied_datasource(name="CLICKSTREAM_EVENTS", source_type="Batch")
        applied = AppliedState(objects={applied_obj.key: applied_obj})

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
        assert len(src_ops) == 1
        assert src_ops[0].kind == OpKind.RECREATE_SOURCE
        assert src_ops[0].destructive is True


# ---------------------------------------------------------------------------
# Orphan-drop pass
# ---------------------------------------------------------------------------


class TestPlanSourceOrphanDrop:
    """Pins §7c — DROP_SOURCE for ``Datasource`` orphans in full-sync mode."""

    def test_full_sync_streaming_orphan_emits_drop_source(self) -> None:
        """Full-sync mode + applied Datasource (Stream) not in batch → DROP_SOURCE."""
        # Batch carries only an entity — no source spec.
        batch = SpecBatch(specs=[_entity_user()])
        orphan = _applied_datasource(name="CLICKSTREAM_EVENTS", source_type="Stream")
        applied = AppliedState(objects={orphan.key: orphan})

        plan = generate_plan(
            batch,
            applied,
            PlanOptions(full_directory_mode=True),
            database=_DB,
            schema=_SCHEMA,
        )

        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_SOURCE]
        assert len(drop_ops) == 1, f"expected one DROP_SOURCE op; got {[op.kind.value for op in plan.ops]}"
        op = drop_ops[0]
        assert op.name == "CLICKSTREAM_EVENTS"
        assert op.destructive is True
        assert op.payload.get("kind") == "StreamingSource", (
            f"DROP_SOURCE payload.kind for source_type=Stream must be 'StreamingSource'; "
            f"got {op.payload.get('kind')!r}"
        )

    def test_non_full_sync_does_not_drop_orphan_source(self) -> None:
        """Incremental mode → orphan Datasource silently skipped (no DROP_SOURCE)."""
        batch = SpecBatch(specs=[_entity_user()])
        orphan = _applied_datasource(name="CLICKSTREAM_EVENTS", source_type="Stream")
        applied = AppliedState(objects={orphan.key: orphan})

        plan = generate_plan(
            batch,
            applied,
            PlanOptions(full_directory_mode=False),
            database=_DB,
            schema=_SCHEMA,
        )

        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_SOURCE]
        assert drop_ops == [], (
            f"non-full-sync mode must not emit DROP_SOURCE for orphans; " f"got {[op.kind.value for op in plan.ops]}"
        )

    def test_full_sync_batch_orphan_emits_drop_source_with_batch_kind(self) -> None:
        """Full-sync mode + applied Datasource (Batch) orphan → DROP_SOURCE w/ BatchSource."""
        batch = SpecBatch(specs=[_entity_user()])
        orphan = _applied_datasource(name="EVENTS_BATCH", source_type="Batch", columns=[])
        applied = AppliedState(objects={orphan.key: orphan})

        plan = generate_plan(
            batch,
            applied,
            PlanOptions(full_directory_mode=True),
            database=_DB,
            schema=_SCHEMA,
        )

        drop_ops = [op for op in plan.ops if op.kind == OpKind.DROP_SOURCE]
        assert len(drop_ops) == 1, f"expected one DROP_SOURCE op; got {[op.kind.value for op in plan.ops]}"
        op = drop_ops[0]
        assert op.name == "EVENTS_BATCH"
        assert op.destructive is True
        assert op.payload.get("kind") == "BatchSource", (
            f"DROP_SOURCE payload.kind for source_type=Batch must be 'BatchSource'; " f"got {op.payload.get('kind')!r}"
        )


# ---------------------------------------------------------------------------
# CLICKSTREAM_EVENTS regression — the user-reported bug
# ---------------------------------------------------------------------------


class TestClickstreamRegression:
    """User-visible regression: ``CLICKSTREAM_EVENTS`` plans as CREATE_SOURCE on every run.

    With Wave 1+2 wired in, the applied state now carries a runtime row
    for the source.  The planner must compare local vs. runtime via
    :func:`compute_source_diff_kind` and emit ``NO_CHANGE`` for a clean
    round-trip — *not* fall through to the ``applied is None``
    ``CREATE_SOURCE`` arm.
    """

    def test_local_yaml_matches_runtime_row_emits_no_change(self) -> None:
        """Local YAML + ``_build_stream_source_object``-shaped applied entry → NO_CHANGE."""
        src = _streaming_source(name="CLICKSTREAM_EVENTS")
        batch = SpecBatch(specs=[_entity_user(), src])

        # Build the applied entry the same way Wave 1+2 produces it at
        # runtime: feed a canonical row through ``_build_stream_source_object``
        # so the diff helper sees the exact shape ``fetch_applied_state``
        # threads through ``stream_source_rows=`` in production.
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
        runtime_obj = _build_stream_source_object(row, default_db=_DB, default_schema=_SCHEMA)
        applied = AppliedState(objects={runtime_obj.key: runtime_obj})

        plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
        src_ops = _ops_by_name(plan, "CLICKSTREAM_EVENTS")
        assert len(src_ops) == 1, f"expected one op for CLICKSTREAM_EVENTS; got {[op.kind.value for op in plan.ops]}"
        assert src_ops[0].kind == OpKind.NO_CHANGE, (
            f"CLICKSTREAM_EVENTS must plan as NO_CHANGE when local matches runtime row; "
            f"got {src_ops[0].kind.value} (reason={src_ops[0].reason!r}) — "
            f"this is the user-reported regression."
        )
        # The reason must come from the source-diff path, not the virtual
        # override or the legacy structural-hash NO_CHANGE.
        assert "source-level" in src_ops[0].reason.lower(), (
            f"regression NO_CHANGE must reference the source-level diff path; " f"got {src_ops[0].reason!r}"
        )


# ---------------------------------------------------------------------------
# Misc: payload preservation
# ---------------------------------------------------------------------------


def test_recreate_source_payload_carries_local_spec_dict() -> None:
    """The RECREATE_SOURCE op payload must be the local authoring dict.

    Mirrors the CREATE/UPDATE branches so the imperative executor's
    ``_execute_recreate_stream_source`` can re-register using the
    authored columns.  Catches a regression where the planner accidentally
    forwards ``applied.spec_payload`` instead of ``data``.
    """
    src = _streaming_source(name="CLICKSTREAM_EVENTS")
    batch = SpecBatch(specs=[_entity_user(), src])
    applied_obj = _applied_datasource(
        name="CLICKSTREAM_EVENTS",
        columns=[{"name": "USER_ID", "type": "StringType"}],
    )
    applied = AppliedState(objects={applied_obj.key: applied_obj})

    plan = generate_plan(batch, applied, PlanOptions(), database=_DB, schema=_SCHEMA)
    src_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_SOURCE and op.name == "CLICKSTREAM_EVENTS"]
    assert len(src_ops) == 1
    payload = src_ops[0].payload
    assert payload.get("kind") == "StreamingSource", payload
    assert payload.get("name") == "CLICKSTREAM_EVENTS"
    # Local-side columns must be carried through unchanged so the
    # executor's register_stream_source call sees the authored schema.
    local_payload = copy.deepcopy(src.model_dump(exclude_none=True))
    assert payload.get("columns") == local_payload.get("columns")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
