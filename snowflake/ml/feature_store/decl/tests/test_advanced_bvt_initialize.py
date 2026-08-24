"""Phase 4 — promote BFV ``initialize`` from ``backfill.initialize`` to top-level.

Today the authoring shape carries ``initialize`` nested under the
``backfill`` block (`backfill.initialize: ON_CREATE | ON_SCHEDULE`).
The imperative ``FeatureView`` constructor takes ``initialize`` as a
first-class kwarg, and the imperative semantics are *structural* — the
value affects the Dynamic Table at create time only, so an edit must
route through ``RECREATE_FV(destructive=True)``.

This phase promotes ``initialize`` to a top-level authoring key while
keeping ``backfill.initialize`` as a back-compat alias.  When both are
authored the top-level value wins (single source of truth for the spec
hash).

Coverage map:

* ``test_initialize_round_trips_through_model_validate`` — top-level
  ``initialize:`` survives spec-model validation.
* ``test_initialize_validator_rejects_unknown_value`` — typoed values
  raise ``ValidationError``.
* ``test_compile_to_spec_threads_initialize_top_level`` — compile puts
  top-level ``initialize`` into the inner ``spec``.
* ``test_compile_to_spec_threads_initialize_legacy_backfill`` — back-
  compat: when only ``backfill.initialize`` is authored the compiled
  spec carries the same ``inner["initialize"]``.
* ``test_compile_to_spec_top_level_wins_over_backfill`` — both forms
  authored: top-level value is the compiled-spec source of truth.
* ``test_hash_changes_when_initialize_changes`` — pure ``initialize``
  edit bumps the structural hash.
* ``test_planner_emits_recreate_fv_when_initialize_changes`` — the
  diff lands as ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_initialize_kwarg_top_level`` —
  executor forwards the top-level value to
  ``FeatureView(initialize=...)``.
* ``test_executor_create_passes_initialize_kwarg_legacy`` — executor
  still threads ``backfill.initialize`` (no regression).
* ``test_exporter_emits_initialize_top_level_not_under_backfill`` —
  exporter emits top-level ``initialize:`` (the canonical form), never
  nested under a synthetic ``backfill`` block.
* ``test_state_inject_initialize_from_imperative_api`` — the
  metadata-driven injector populates ``initialize`` from the
  rehydrated ``FeatureView`` object returned by
  ``imperative_executor.fetch_feature_view_object`` (Phase B1:
  replaces the DT-text regex parser).
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


def _minimal_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_INIT",
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
    key = "BatchFeatureView:DB1.SC1:BFV_INIT"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_INIT",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


@pytest.mark.parametrize("value", ["ON_CREATE", "ON_SCHEDULE"])
def test_initialize_round_trips_through_model_validate(value: str) -> None:
    """Top-level ``initialize:`` survives spec-model validation."""
    payload = _minimal_batch_fv_authoring(initialize=value)
    fv = FeatureView.model_validate(payload)
    assert fv.initialize == value
    assert fv.model_dump(exclude_none=True)["initialize"] == value


def test_initialize_validator_rejects_unknown_value() -> None:
    """Top-level ``initialize:`` rejects typoed values at load time."""
    payload = _minimal_batch_fv_authoring(initialize="LATER")
    with pytest.raises(ValidationError):
        FeatureView.model_validate(payload)


def test_compile_to_spec_threads_initialize_top_level() -> None:
    local = _minimal_batch_fv_authoring(initialize="ON_SCHEDULE")
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert compiled.get("spec", {}).get("initialize") == "ON_SCHEDULE"


def test_compile_to_spec_threads_initialize_legacy_backfill() -> None:
    """Back-compat: ``backfill.initialize`` still surfaces into the compiled spec."""
    local = _minimal_batch_fv_authoring(backfill={"initialize": "ON_SCHEDULE"})
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert compiled.get("spec", {}).get("initialize") == "ON_SCHEDULE"


def test_compile_to_spec_top_level_wins_over_backfill() -> None:
    """When both forms are authored the top-level value is the source of truth."""
    local = _minimal_batch_fv_authoring(
        initialize="ON_CREATE",
        backfill={"initialize": "ON_SCHEDULE"},
    )
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert compiled.get("spec", {}).get("initialize") == "ON_CREATE"


def test_hash_changes_when_initialize_changes() -> None:
    a = compile_to_spec(_minimal_batch_fv_authoring(initialize="ON_CREATE"), "DB1", "SC1")
    b = compile_to_spec(_minimal_batch_fv_authoring(initialize="ON_SCHEDULE"), "DB1", "SC1")
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_planner_emits_recreate_fv_when_initialize_changes() -> None:
    local = _minimal_batch_fv_authoring(initialize="ON_CREATE")
    applied = _applied_state_for(local)
    edited = _minimal_batch_fv_authoring(initialize="ON_SCHEDULE")

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
    fv_ops = [op for op in plan.ops if op.name == "BFV_INIT"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.RECREATE_FV
    assert fv_ops[0].destructive is True


def _patched_fv() -> tuple[type, dict[str, Any]]:
    captured: dict[str, Any] = {}

    class _FakeFV:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    return _FakeFV, captured


def test_executor_create_passes_initialize_kwarg_top_level() -> None:
    """Top-level ``initialize`` reaches ``FeatureView(initialize=...)``."""
    payload = _minimal_batch_fv_authoring(initialize="ON_SCHEDULE")
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    fake, captured = _patched_fv()

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", fake), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig", MagicMock()
    ), patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock()):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_DEFAULT", fs=fs)

    assert captured.get("initialize") == "ON_SCHEDULE"


def test_executor_create_passes_initialize_kwarg_legacy() -> None:
    """Back-compat: ``backfill.initialize`` still threads into ``FeatureView``."""
    payload = _minimal_batch_fv_authoring(backfill={"initialize": "ON_SCHEDULE"})
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    fake, captured = _patched_fv()

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", fake), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig", MagicMock()
    ), patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock()):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_DEFAULT", fs=fs)

    assert captured.get("initialize") == "ON_SCHEDULE"


def test_exporter_emits_initialize_top_level_not_under_backfill() -> None:
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_INIT", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "initialize": "ON_SCHEDULE",
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_INIT",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("initialize") == "ON_SCHEDULE"
    assert "backfill" not in doc


def test_state_inject_initialize_from_imperative_api() -> None:
    """The metadata-driven injector populates ``initialize`` from the
    rehydrated ``FeatureView`` object.

    Phase B1: replaces the legacy DT-text regex parser
    (``_parse_initialize_from_dt_text``).  ``initialize`` is not
    surfaced on the ``list_feature_views()`` row, so the recovery side
    pulls it from the :class:`FeatureView` object returned by
    :func:`imperative_executor.fetch_feature_view_object` and threaded
    in through the ``fv_obj`` argument of
    :func:`_inject_batch_fv_fields_from_list_row`.
    """
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_fields_from_list_row,
    )

    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row = {"cluster_by": "", "refresh_mode": ""}
    fv_obj = MagicMock()
    fv_obj.initialize = "ON_SCHEDULE"
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=fv_obj)
    assert spec_payload["spec"]["initialize"] == "ON_SCHEDULE"

    # Bare-string variant survives the same uppercase normalisation.
    spec_payload2: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    fv_obj2 = MagicMock()
    fv_obj2.initialize = "on_create"
    _inject_batch_fv_fields_from_list_row(spec_payload2, row, fv_obj=fv_obj2)
    assert spec_payload2["spec"]["initialize"] == "ON_CREATE"

    # No fv_obj / no initialize attribute -> key is not injected.
    spec_payload3: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    _inject_batch_fv_fields_from_list_row(spec_payload3, row, fv_obj=None)
    assert "initialize" not in spec_payload3["spec"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
