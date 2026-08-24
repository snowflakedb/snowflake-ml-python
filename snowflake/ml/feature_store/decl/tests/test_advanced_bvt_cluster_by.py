"""Phase 2 — declarative-side coverage for the BFV ``cluster_by`` field.

``cluster_by`` is *structural*: editing the value alters the Dynamic Table's
``CLUSTER BY (...)`` clause, which Snowflake cannot apply in-place.  The
planner must therefore emit ``RECREATE_FV`` (``destructive=True``,
``--allow-recreate`` gated) instead of ``UPDATE_FV``.

Coverage map:

* ``test_cluster_by_round_trips_through_model_validate`` — spec model
  accepts and preserves the list.
* ``test_compile_to_spec_threads_cluster_by`` — ``compile_to_spec``
  injects ``cluster_by`` into the inner ``spec`` dict so it contributes
  to ``_full_spec_hash``.
* ``test_hash_changes_when_cluster_by_changes`` — a pure cluster_by edit
  bumps the structural hash (planner pre-condition).
* ``test_planner_emits_recreate_fv_when_cluster_by_changes`` — the diff
  resolves to ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_cluster_by_kwarg`` — the executor's
  ``_build_feature_view`` forwards the payload's ``cluster_by`` as a
  list kwarg into the imperative ``FeatureView`` constructor.
* ``test_exporter_recovers_cluster_by_from_inner_spec`` — when state
  recovery has injected ``cluster_by`` into the applied payload, the
  exporter's ``_build_full_fidelity_fv`` round-trips it back into the
  authoring-shape YAML dict.
* ``test_state_inject_cluster_by_from_list_row`` — the metadata-driven
  injector populates ``cluster_by`` from the ``list_feature_views()``
  row's ``cluster_by`` column (Phase B1: replaces the DT-text regex
  parser).
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
        "name": "BFV_CB",
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
    key = "BatchFeatureView:DB1.SC1:BFV_CB"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_CB",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_cluster_by_round_trips_through_model_validate() -> None:
    """Authoring ``cluster_by: [USER_ID]`` survives spec-model validation."""
    payload = _minimal_batch_fv_authoring(cluster_by=["USER_ID", "SESSION_ID"])
    fv = FeatureView.model_validate(payload)
    assert fv.cluster_by == ["USER_ID", "SESSION_ID"]
    assert fv.model_dump(exclude_none=True)["cluster_by"] == ["USER_ID", "SESSION_ID"]


def test_compile_to_spec_threads_cluster_by() -> None:
    """``compile_to_spec`` must surface ``cluster_by`` into the inner ``spec``."""
    local = _minimal_batch_fv_authoring(cluster_by=["USER_ID"])
    compiled = compile_to_spec(local, "DB1", "SC1")
    inner = compiled.get("spec", {})
    assert inner.get("cluster_by") == ["USER_ID"], f"compile_to_spec dropped cluster_by; got inner keys={sorted(inner)}"


def test_hash_changes_when_cluster_by_changes() -> None:
    """A pure ``cluster_by`` edit must bump the structural hash."""
    a = compile_to_spec(_minimal_batch_fv_authoring(cluster_by=["USER_ID"]), "DB1", "SC1")
    b = compile_to_spec(_minimal_batch_fv_authoring(cluster_by=["SESSION_ID"]), "DB1", "SC1")
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_planner_emits_recreate_fv_when_cluster_by_changes() -> None:
    """A pure ``cluster_by`` edit lands as ``RECREATE_FV(destructive=True)``."""
    local = _minimal_batch_fv_authoring(cluster_by=["USER_ID"])
    applied = _applied_state_for(local)
    edited = _minimal_batch_fv_authoring(cluster_by=["SESSION_ID"])

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
    fv_ops = [op for op in plan.ops if op.name == "BFV_CB"]
    assert len(fv_ops) == 1
    assert (
        fv_ops[0].kind is OpKind.RECREATE_FV
    ), f"expected RECREATE_FV when cluster_by changes; got {fv_ops[0].kind} (reason={fv_ops[0].reason!r})"
    assert fv_ops[0].destructive is True
    assert fv_ops[0].payload.get("cluster_by") == ["SESSION_ID"]


def test_executor_create_passes_cluster_by_kwarg() -> None:
    """``_build_feature_view`` forwards the payload's ``cluster_by`` to ``FeatureView``."""
    payload = _minimal_batch_fv_authoring(cluster_by=["USER_ID", "SESSION_ID"])
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

    assert captured.get("cluster_by") == [
        "USER_ID",
        "SESSION_ID",
    ], f"cluster_by missing from FV kwargs; got {sorted(captured)}"


def test_exporter_recovers_cluster_by_from_inner_spec() -> None:
    """``_build_full_fidelity_fv`` copies recovered ``cluster_by`` into the YAML dict."""
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_CB", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "cluster_by": ["USER_ID"],
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_CB",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("cluster_by") == ["USER_ID"], f"exporter dropped cluster_by; got keys={sorted(doc)}"


def test_state_inject_cluster_by_from_list_row() -> None:
    """The metadata-driven injector populates ``cluster_by`` from a
    ``list_feature_views()`` row column.

    Phase B1: replaces the legacy DT-text regex parser
    (``_parse_cluster_by_from_dt_text`` / ``_inject_advanced_bfv_fields_from_dt_text``).
    ``cluster_by`` now rides on the dedicated ``cluster_by`` column of
    the list-FV row — accepted as a Python list (Snowpark surfaces
    ARRAY-of-string columns as lists) or as a comma-separated string
    (legacy cursor adapters).
    """
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_fields_from_list_row,
        _parse_cluster_by_list,
    )

    # The injector merges into ``spec_payload['spec']`` in place.
    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row = {"cluster_by": ["USER_ID", "SESSION_ID"], "refresh_mode": ""}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert spec_payload["spec"]["cluster_by"] == ["USER_ID", "SESSION_ID"]

    # Legacy cursor shape: comma-separated string.
    spec_payload2: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row2 = {"cluster_by": '"USER_ID","SESSION_ID"', "refresh_mode": ""}
    _inject_batch_fv_fields_from_list_row(spec_payload2, row2, fv_obj=None)
    assert spec_payload2["spec"]["cluster_by"] == ["USER_ID", "SESSION_ID"]

    # Direct parse helper coverage: empty / None inputs return None;
    # populated inputs return the canonical bare-identifier list.
    assert _parse_cluster_by_list(["USER_ID", "SESSION_ID"]) == ["USER_ID", "SESSION_ID"]
    assert _parse_cluster_by_list('"USER_ID","SESSION_ID"') == ["USER_ID", "SESSION_ID"]
    assert _parse_cluster_by_list(None) is None
    assert _parse_cluster_by_list("") is None

    # Phase E gap 1 regression — Snowpark surfaces the deployed FV's
    # ``cluster_by`` column as a single-element list whose sole entry
    # is a JSON-encoded array literal: ``['["USER_ID"]']``.  The
    # parser must decode the inner JSON so the structural hash matches
    # the local-compile shape ``['USER_ID']``.  Without this the live
    # ``snow feature plan`` after a clean apply emitted
    # ``RECREATE_FV`` for offline tiled BFVs like ``MY_ADV_BFV_DECL``.
    assert _parse_cluster_by_list(['["USER_ID"]']) == ["USER_ID"]
    assert _parse_cluster_by_list(['["USER_ID", "SESSION_ID"]']) == [
        "USER_ID",
        "SESSION_ID",
    ]
    # Non-JSON bracketed strings degrade gracefully — the token is
    # preserved verbatim rather than dropped.
    assert _parse_cluster_by_list(["[malformed"]) == ["[malformed"]
    # Direct JSON-string input (not wrapped in a list) also decodes.
    assert _parse_cluster_by_list('["USER_ID"]') == ["USER_ID"]

    # Phase E gap 1 (sub-bug) — multi-key JSON-string input must NOT be
    # comma-split before JSON-decoding.  Splitting first produced
    # garbage tokens (``['["USER_ID', 'TILE_START"]']``), which is what
    # the live tiled-BFV warehouse-flip case (L1) exposed: the default
    # tiled-BFV cluster shape is ``[entities..., TILE_START]``, so the
    # applied cell is a two-element JSON-array literal and the parser
    # must decode it intact to match the local-compile default-stripped
    # shape.  Without this fix L1 plan v2 emitted RECREATE_FV
    # (cluster_by drift) instead of UPDATE_FV (warehouse-only drift).
    assert _parse_cluster_by_list('["USER_ID","TILE_START"]') == [
        "USER_ID",
        "TILE_START",
    ]
    assert _parse_cluster_by_list(['["USER_ID","TILE_START"]']) == [
        "USER_ID",
        "TILE_START",
    ]
    # Whitespace inside the JSON literal is tolerated.
    assert _parse_cluster_by_list('  [ "USER_ID" , "TILE_START" ]  ') == [
        "USER_ID",
        "TILE_START",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
