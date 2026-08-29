"""Phase 3 — declarative-side coverage for the BFV ``refresh_mode`` field.

``refresh_mode`` is *structural* (Snowflake cannot change the refresh
strategy on an existing Dynamic Table), so an edit must route through
``RECREATE_FV(destructive=True)``.  The accepted explicit values are
``FULL`` and ``INCREMENTAL``; omit the field to let Snowflake choose
automatically.  Anything else (including ``AUTO``) is rejected at
``model_validate`` time.

Coverage map:

* ``test_refresh_mode_round_trips_through_model_validate`` — each
  valid enum value survives spec-model validation.
* ``test_refresh_mode_validator_rejects_unknown_value`` — typoed
  values raise ``ValidationError`` at load time, not later.
* ``test_compile_to_spec_threads_refresh_mode`` — compile injects
  ``refresh_mode`` into the inner ``spec``.
* ``test_hash_changes_when_refresh_mode_changes`` — a pure
  ``refresh_mode`` edit bumps the structural hash.
* ``test_planner_emits_recreate_fv_when_refresh_mode_changes`` —
  the diff resolves to ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_refresh_mode_kwarg`` — executor
  forwards the value as a kwarg to the imperative
  ``FeatureView(refresh_mode=...)`` constructor.
* ``test_exporter_recovers_refresh_mode_from_inner_spec`` — the
  exporter's ``_build_full_fidelity_fv`` round-trips the value into
  the authoring-shape YAML dict.
* ``test_state_inject_refresh_mode_from_list_row`` — the metadata-driven
  injector populates ``refresh_mode`` from the
  ``list_feature_views()`` row's dedicated ``refresh_mode`` column
  (Phase B1: replaces the DT-text regex parser).
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.imperative_executor import _build_feature_view
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
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


def _minimal_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_RM",
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
    key = "BatchFeatureView:DB1.SC1:BFV_RM:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_RM",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


@pytest.mark.parametrize("value", ["FULL", "INCREMENTAL"])
def test_refresh_mode_round_trips_through_model_validate(value: str) -> None:
    """Each canonical ``refresh_mode`` value survives spec-model validation."""
    payload = _minimal_batch_fv_authoring(refresh_mode=value)
    fv = FeatureView.model_validate(payload)
    assert fv.refresh_mode == value
    assert fv.model_dump(exclude_none=True)["refresh_mode"] == value


def test_refresh_mode_validator_rejects_unknown_value() -> None:
    """An unknown ``refresh_mode`` value must fail at ``model_validate`` time."""
    payload = _minimal_batch_fv_authoring(refresh_mode="LATER")
    with pytest.raises(ValidationError):
        FeatureView.model_validate(payload)


def test_compile_to_spec_threads_refresh_mode() -> None:
    """``compile_to_spec`` must surface ``refresh_mode`` into the inner ``spec``."""
    local = _minimal_batch_fv_authoring(refresh_mode="FULL")
    compiled = compile_to_spec(local, "DB1", "SC1")
    inner = compiled.get("spec", {})
    assert inner.get("refresh_mode") == "FULL"


def test_hash_changes_when_refresh_mode_changes() -> None:
    """A pure ``refresh_mode`` edit must bump the structural hash."""
    a = compile_to_spec(_minimal_batch_fv_authoring(refresh_mode="INCREMENTAL"), "DB1", "SC1")
    b = compile_to_spec(_minimal_batch_fv_authoring(refresh_mode="FULL"), "DB1", "SC1")
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_planner_emits_recreate_fv_when_refresh_mode_changes() -> None:
    """A pure ``refresh_mode`` edit lands as ``RECREATE_FV(destructive=True)``."""
    local = _minimal_batch_fv_authoring(refresh_mode="INCREMENTAL")
    applied = _applied_state_for(local)
    edited = _minimal_batch_fv_authoring(refresh_mode="FULL")

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
    fv_ops = [op for op in plan.ops if op.name == "BFV_RM"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.RECREATE_FV
    assert fv_ops[0].destructive is True
    assert fv_ops[0].payload.get("refresh_mode") == "FULL"


def test_executor_create_passes_refresh_mode_kwarg() -> None:
    """``_build_feature_view`` forwards the payload's ``refresh_mode`` to ``FeatureView``."""
    payload = _minimal_batch_fv_authoring(refresh_mode="INCREMENTAL")
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
        _build_feature_view(payload, session, "DB1", "SC1", "WH_DEFAULT", fs=fs)

    assert captured.get("refresh_mode") == "INCREMENTAL"


def test_exporter_recovers_refresh_mode_from_inner_spec() -> None:
    """``_build_full_fidelity_fv`` copies recovered ``refresh_mode`` into the YAML doc."""
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_RM", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "refresh_mode": "FULL",
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_RM",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("refresh_mode") == "FULL"


def test_state_inject_refresh_mode_from_list_row() -> None:
    """The metadata-driven injector populates ``refresh_mode`` from the
    ``list_feature_views()`` row column.

    Phase B1: replaces the legacy DT-text regex parser
    (``_parse_refresh_mode_from_dt_text``).  ``refresh_mode`` now rides
    on the dedicated ``refresh_mode`` column of the list-FV row.  The
    injector uppercases the cell so casing differences between the
    Snowflake-side string and the canonical authoring enum collapse.
    """
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_fields_from_list_row,
    )

    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row = {"cluster_by": "", "refresh_mode": "FULL"}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert spec_payload["spec"]["refresh_mode"] == "FULL"

    # Lowercase cell -> uppercase canonical form.
    spec_payload2: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row2 = {"cluster_by": "", "refresh_mode": "incremental"}
    _inject_batch_fv_fields_from_list_row(spec_payload2, row2, fv_obj=None)
    assert spec_payload2["spec"]["refresh_mode"] == "INCREMENTAL"

    # Empty string / missing cell -> key is not injected.
    spec_payload3: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    _inject_batch_fv_fields_from_list_row(spec_payload3, {"cluster_by": "", "refresh_mode": ""}, fv_obj=None)
    assert "refresh_mode" not in spec_payload3["spec"]


if __name__ == "__main__":
    pytest_driver.main()
