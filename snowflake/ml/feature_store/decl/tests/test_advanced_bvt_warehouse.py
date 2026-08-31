"""Phase 1 — declarative-side coverage for the BFV ``warehouse`` field.

The ``warehouse:`` authoring key is the only *operational* field in the
advanced-BFV set: editing it on a deployed feature view must route through
``FeatureStore.update_feature_view(warehouse=...)`` (``UPDATE_FV`` —
non-destructive) rather than through ``RECREATE_FV``.  This file is the
per-field deep coverage that complements the matrix row in
``test_advanced_bvt_fields.py``.

Coverage map:

* ``test_warehouse_round_trips_through_model_validate`` — the spec model
  must accept and preserve ``warehouse``.
* ``test_hash_ignores_warehouse_edit`` — ``_full_spec_hash`` must strip
  ``warehouse`` so a pure warehouse edit leaves the structural hash
  unchanged (planner pre-condition for the ``UPDATE_FV`` branch).
* ``test_planner_emits_update_fv_when_only_warehouse_changes`` — given an
  applied state for the unedited spec, an edited local spec that only
  changes ``warehouse`` lands as ``UPDATE_FV(destructive=False)``.
* ``test_executor_create_passes_warehouse_kwarg`` — on ``CREATE_FV``, the
  payload's ``warehouse`` reaches the ``FeatureView(warehouse=...)``
  constructor verbatim (overriding the connection default).
* ``test_executor_update_passes_warehouse_kwarg`` — on ``UPDATE_FV``,
  the payload's ``warehouse`` lands on
  ``fs.update_feature_view(..., warehouse=...)``.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.imperative_executor import (
    _build_feature_view,
    _execute_op,
)
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
    PlanOp,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver


def _minimal_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_WH",
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
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def _applied_state_for(local: dict[str, Any]) -> AppliedState:
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = "BatchFeatureView:DB1.SC1:BFV_WH:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_WH",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_warehouse_round_trips_through_model_validate() -> None:
    """Authoring ``warehouse: WH_OVERRIDE`` survives spec-model validation."""
    payload = _minimal_batch_fv_authoring(warehouse="WH_OVERRIDE")
    fv = FeatureView.model_validate(payload)
    assert fv.warehouse == "WH_OVERRIDE"
    assert fv.model_dump(exclude_none=True)["warehouse"] == "WH_OVERRIDE"


def test_hash_ignores_warehouse_edit() -> None:
    """Editing only ``warehouse`` must not change the structural hash."""
    base = _minimal_batch_fv_authoring()
    edited = _minimal_batch_fv_authoring(warehouse="WH_OVERRIDE")
    assert compute_local_spec_hash(base, "DB1", "SC1") == compute_local_spec_hash(edited, "DB1", "SC1")


def test_planner_emits_update_fv_when_only_warehouse_changes() -> None:
    """A pure ``warehouse`` edit lands as ``UPDATE_FV(destructive=False)``."""
    local = _minimal_batch_fv_authoring()
    applied = _applied_state_for(local)
    edited = _minimal_batch_fv_authoring(warehouse="WH_NEW")

    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(edited)
    plan = generate_plan(
        SpecBatch(specs=[ent, src, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_WH"]
    assert len(fv_ops) == 1, [op for op in plan.ops]
    assert (
        fv_ops[0].kind is OpKind.UPDATE_FV
    ), f"expected UPDATE_FV when only warehouse changes; got {fv_ops[0].kind} (reason={fv_ops[0].reason!r})"
    assert fv_ops[0].destructive is False
    assert fv_ops[0].payload.get("warehouse") == "WH_NEW"


def test_executor_create_passes_warehouse_kwarg() -> None:
    """``_build_feature_view`` forwards the payload's ``warehouse`` to ``FeatureView``."""
    payload = _minimal_batch_fv_authoring(warehouse="WH_OVERRIDE")
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    captured: dict[str, Any] = {}

    class _FakeFV:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", _FakeFV,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        MagicMock(),
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_CONNECTION_DEFAULT", fs=fs)

    assert (
        captured.get("warehouse") == "WH_OVERRIDE"
    ), f"warehouse from payload must beat connection default; got kwargs={sorted(captured)}"


def test_executor_update_passes_warehouse_kwarg() -> None:
    """``UPDATE_FV`` execution forwards ``warehouse`` into ``update_feature_view``."""
    fs = MagicMock()
    op = PlanOp(
        kind=OpKind.UPDATE_FV,
        name="BFV_WH",
        depends_on=[],
        destructive=False,
        reason="test",
        payload={
            "kind": "BatchFeatureView",
            "name": "BFV_WH",
            "version": "V1",
            "warehouse": "WH_FROM_PAYLOAD",
        },
    )
    session = MagicMock()
    opts = PlanOptions()
    _execute_op(fs, session, op, "DB1", "SC1", "WH_CONNECTION_DEFAULT", opts)
    fs.update_feature_view.assert_called_once()
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs.get("warehouse") == "WH_FROM_PAYLOAD"


if __name__ == "__main__":
    pytest_driver.main()
