"""Phase 6 — declarative-side coverage for ``aggregation_secondary_keys``.

``aggregation_secondary_keys`` is a private-preview, tiled-only knob on
``BatchFeatureView`` that adds secondary group-by columns to a tiled
aggregation.  Snowflake caps the list at length 1 in this preview, and
the field only makes sense alongside tiled ``features:`` — non-tiled
BFVs (which have no aggregation windows) cannot honour secondary keys
at all.

Both constraints are enforced at authoring time so the operator gets a
clear error before plan time:

* ``BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES`` — list authored on a
  non-tiled BFV.
* ``BATCH_FV_SECONDARY_KEYS_MAX_LENGTH`` — list with >1 entry.

Coverage map:

* ``test_secondary_keys_round_trips_through_model_validate`` — the
  list survives ``model_validate``.
* ``test_secondary_keys_validator_requires_tiled_features`` — list
  on a non-tiled BFV raises the tiled-only error.
* ``test_secondary_keys_validator_rejects_more_than_one`` — list
  with two entries raises the length-cap error.
* ``test_compile_to_spec_threads_secondary_keys`` — compile injects
  the list into the inner ``spec``.
* ``test_hash_changes_when_secondary_keys_change`` — pure secondary-
  keys edit bumps the structural hash.
* ``test_planner_emits_recreate_fv_when_secondary_keys_change`` —
  planner emits ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_secondary_keys_kwarg`` — executor
  forwards the list as ``aggregation_secondary_keys=[...]`` to
  ``FeatureView``.
* ``test_exporter_recovers_secondary_keys_from_inner_spec`` —
  exporter copies the list back into the YAML doc.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.imperative_executor import _build_feature_view
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash, validate_specs
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


def _tiled_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    """A minimal *tiled* BatchFV that satisfies the existing tiling invariants."""
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_SK",
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
    base.update(overrides)
    return base


def _non_tiled_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    """A minimal *non-tiled* BatchFV (no aggregation windows)."""
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_SK_NONTILED",
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


def _applied_state_for(local: dict[str, Any], key_name: str = "BFV_SK") -> AppliedState:
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = f"BatchFeatureView:DB1.SC1:{key_name}"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name=key_name,
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def _spec_batch_for_tiled(local: dict[str, Any]) -> SpecBatch:
    ent = Entity(
        kind="Entity",
        name="USER",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(local)
    return SpecBatch(specs=[ent, src, fv])


def test_secondary_keys_round_trips_through_model_validate() -> None:
    payload = _tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    fv = FeatureView.model_validate(payload)
    assert fv.aggregation_secondary_keys == ["SESSION_ID"]
    assert fv.model_dump(exclude_none=True)["aggregation_secondary_keys"] == ["SESSION_ID"]


def test_secondary_keys_validator_requires_tiled_features() -> None:
    payload = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    fv = FeatureView.model_validate(payload)

    ent = Entity(
        kind="Entity",
        name="USER",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    results = validate_specs(
        SpecBatch(specs=[ent, src, fv]),
        AppliedState(objects={}),
        target_database="DB1",
        target_schema="SC1",
    )
    assert "BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES" in [r.code for r in results]


def test_secondary_keys_validator_rejects_more_than_one() -> None:
    payload = _tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID", "DEVICE_ID"])
    fv = FeatureView.model_validate(payload)
    ent = Entity(
        kind="Entity",
        name="USER",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    results = validate_specs(
        SpecBatch(specs=[ent, src, fv]),
        AppliedState(objects={}),
        target_database="DB1",
        target_schema="SC1",
    )
    assert "BATCH_FV_SECONDARY_KEYS_MAX_LENGTH" in [r.code for r in results]


def test_compile_to_spec_threads_secondary_keys() -> None:
    local = _tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert compiled.get("spec", {}).get("aggregation_secondary_keys") == ["SESSION_ID"]


def test_hash_changes_when_secondary_keys_change() -> None:
    a = compile_to_spec(_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"]), "DB1", "SC1")
    b = compile_to_spec(_tiled_batch_fv_authoring(aggregation_secondary_keys=["DEVICE_ID"]), "DB1", "SC1")
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_planner_emits_recreate_fv_when_secondary_keys_change() -> None:
    local = _tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    applied = _applied_state_for(local)
    edited = _tiled_batch_fv_authoring(aggregation_secondary_keys=["DEVICE_ID"])
    plan = generate_plan(
        _spec_batch_for_tiled(edited),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_SK"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.RECREATE_FV
    assert fv_ops[0].destructive is True


def test_executor_create_passes_secondary_keys_kwarg() -> None:
    payload = _tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
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
    ), patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock(),), patch(
        "snowflake.ml.feature_store.feature_view.FeatureAggregationMethod",
        lambda v: v,
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_DEFAULT", fs=fs)

    assert captured.get("aggregation_secondary_keys") == ["SESSION_ID"]


def test_exporter_recovers_secondary_keys_from_inner_spec() -> None:
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_SK", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "aggregation_secondary_keys": ["SESSION_ID"],
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_SK",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("aggregation_secondary_keys") == ["SESSION_ID"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
