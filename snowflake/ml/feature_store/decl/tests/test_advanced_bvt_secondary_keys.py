"""Phase 6 — declarative-side coverage for ``aggregation_secondary_keys``.

``aggregation_secondary_keys`` is a private-preview knob on
``BatchFeatureView``.  On a **tiled** BFV it adds a secondary group-by
column to every aggregation; on a **non-tiled** (passthrough) BFV it is
still a first-class identity column — the imperative register path
persists it, ``_build_batch_feature_view_spec`` folds it into
``entity_columns`` / ``secondary_key_columns``, and the POSTGRES OFT
widens its primary key with it — so the declarative surface must accept
it there too (otherwise ``snow feature init`` re-exports a spec the
loader would reject).  Snowflake caps the list at length 1 in this
preview.

The one remaining authoring constraint is enforced at plan time so the
operator gets a clear error before apply:

* ``BATCH_FV_SECONDARY_KEYS_MAX_LENGTH`` — list with >1 entry.

The former ``BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES`` tiled-only gate was
removed: it rejected a state the imperative side persists and the
exporter re-emits.

Coverage map:

* ``test_secondary_keys_round_trips_through_model_validate`` — the
  list survives ``model_validate``.
* ``test_secondary_keys_allowed_on_non_tiled_batch_fv`` — list on a
  non-tiled BFV validates (no tiled-only error).
* ``test_secondary_keys_init_roundtrip_non_tiled_validates`` — an
  exporter-recovered non-tiled SK doc re-validates clean (the init
  hole the tiled-only gate used to open).
* ``test_secondary_keys_validator_rejects_more_than_one`` — list
  with two entries raises the length-cap error (tiled).
* ``test_secondary_keys_validator_rejects_more_than_one_non_tiled`` —
  same cap fires on a non-tiled BFV.
* ``test_compile_to_spec_threads_secondary_keys`` — compile injects
  the list into the inner ``spec``.
* ``test_compile_to_spec_threads_secondary_keys_non_tiled`` — same,
  non-tiled.
* ``test_hash_changes_when_secondary_keys_change`` — pure secondary-
  keys edit bumps the structural hash.
* ``test_hash_changes_when_secondary_keys_change_non_tiled`` — same,
  non-tiled.
* ``test_planner_emits_recreate_fv_when_secondary_keys_change`` —
  planner emits ``RECREATE_FV(destructive=True)``.
* ``test_planner_emits_recreate_fv_when_secondary_keys_change_non_tiled``
  — same, non-tiled.
* ``test_executor_create_passes_secondary_keys_kwarg`` — executor
  forwards the list as ``aggregation_secondary_keys=[...]`` to
  ``FeatureView``.
* ``test_executor_create_passes_secondary_keys_kwarg_non_tiled`` —
  same, non-tiled.
* ``test_exporter_recovers_secondary_keys_from_inner_spec`` —
  exporter copies the list back into the YAML doc.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

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
from snowflake.ml.test_utils import pytest_driver


def _tiled_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    """A minimal *tiled* BatchFV that satisfies the existing tiling invariants.

    Durations are the post-``compile_spec`` ``*_sec`` integers the
    executor's ``_build_features`` / ``_build_feature_view`` consume
    (plan-op shape after ``normalize_durations``), not authoring
    strings like ``window: "1h"``.

    Args:
        **overrides: Keys to override on the base dict (e.g.
            ``aggregation_secondary_keys``).

    Returns:
        Tiled BatchFV authoring dict.
    """
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
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "AMOUNT", "type": "FloatType"},
                "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window_sec": 3600,
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
    key = f"BatchFeatureView:DB1.SC1:{key_name}:V1"
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


def test_secondary_keys_allowed_on_non_tiled_batch_fv() -> None:
    """A non-tiled (passthrough) BFV may author ``aggregation_secondary_keys``.

    The imperative register path persists the key, folds it into the
    spec's ``entity_columns`` / ``secondary_key_columns``, and widens the
    POSTGRES OFT primary key with it — so the field is a real identity
    column on non-tiled BFVs, not a no-op.  The declarative surface must
    not reject it (the removed ``BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES``
    gate did).
    """
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
    codes = [r.code for r in results]
    assert "BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES" not in codes
    # The length-1 cap still applies, but a single key must not trip it.
    assert "BATCH_FV_SECONDARY_KEYS_MAX_LENGTH" not in codes


def test_secondary_keys_init_roundtrip_non_tiled_validates() -> None:
    """The ``snow feature init`` hole the tiled-only gate opened is closed.

    A non-tiled BFV that carries ``aggregation_secondary_keys`` in its
    applied SPECIFICATION is re-exported by ``_build_full_fidelity_fv``
    with the key intact; re-validating that YAML must not reject it.
    """
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_SK_NT", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER"],
            "aggregation_secondary_keys": ["SESSION_ID"],
            "sources": [
                {
                    "name": "SRC1",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_SK_NT",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("aggregation_secondary_keys") == ["SESSION_ID"]

    fv = FeatureView.model_validate(doc)
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
    assert "BATCH_FV_SECONDARY_KEYS_REQUIRE_TILES" not in [r.code for r in results]


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


def test_secondary_keys_validator_rejects_more_than_one_non_tiled() -> None:
    """The length-1 cap fires on a non-tiled BFV too (same code path)."""
    payload = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID", "DEVICE_ID"])
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


def test_compile_to_spec_threads_secondary_keys_non_tiled() -> None:
    local = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert compiled.get("spec", {}).get("aggregation_secondary_keys") == ["SESSION_ID"]


def test_hash_changes_when_secondary_keys_change() -> None:
    a = compile_to_spec(_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"]), "DB1", "SC1")
    b = compile_to_spec(_tiled_batch_fv_authoring(aggregation_secondary_keys=["DEVICE_ID"]), "DB1", "SC1")
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_hash_changes_when_secondary_keys_change_non_tiled() -> None:
    a = compile_to_spec(_non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"]), "DB1", "SC1")
    b = compile_to_spec(_non_tiled_batch_fv_authoring(aggregation_secondary_keys=["DEVICE_ID"]), "DB1", "SC1")
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


def test_planner_emits_recreate_fv_when_secondary_keys_change_non_tiled() -> None:
    local = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
    applied = _applied_state_for(local, key_name="BFV_SK_NONTILED")
    edited = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["DEVICE_ID"])
    plan = generate_plan(
        _spec_batch_for_tiled(edited),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_SK_NONTILED"]
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


def test_executor_create_passes_secondary_keys_kwarg_non_tiled() -> None:
    payload = _non_tiled_batch_fv_authoring(aggregation_secondary_keys=["SESSION_ID"])
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
    pytest_driver.main()
