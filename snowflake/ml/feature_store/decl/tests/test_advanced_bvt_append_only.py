"""Declarative-side coverage for the BFV ``append_only`` field.

``append_only`` opts a ``BatchFeatureView`` into the imperative
``snowflake.ml.feature_store.FeatureView(append_only=True)`` snapshot-
accumulation path (companion ``$SNAPSHOTS`` table + CRON refresh Task for
point-in-time training).  It is *structural* — toggling it changes whether the
snapshot table + Task exist, so an edit must route through
``RECREATE_FV(destructive=True)``.  The default (``False`` / absent) is stripped
symmetrically on the local and applied sides so a plain BFV never sees a
spurious diff.

Coverage map mirrors ``test_advanced_bvt_refresh_mode.py``:

* ``test_compile_to_spec_threads_append_only`` — compile injects
  ``append_only`` into the inner ``spec`` only when truthy.
* ``test_hash_changes_when_append_only_toggles`` — an ``append_only`` edit
  bumps the structural hash.
* ``test_hash_stable_when_default_false_stripped`` — a local BFV without
  ``append_only`` hashes equal to an applied payload carrying
  ``append_only: false``.
* ``test_planner_emits_recreate_fv_when_append_only_toggles`` — the diff
  resolves to ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_append_only_kwarg`` — executor forwards the
  value to the imperative ``FeatureView(append_only=...)`` constructor.
* ``test_exporter_recovers_append_only_from_inner_spec`` — the exporter's
  ``_build_full_fidelity_fv`` round-trips the value into the authoring-shape
  YAML dict.
* ``test_state_inject_append_only_from_list_row`` — the metadata-driven
  injector populates ``append_only`` from the ``list_feature_views()`` row.
* ``TestPlannerAppendOnlyEvolveRouting`` — the planner routes a
  schema-preserving append-only edit (source-only / trailing-feature append)
  to the non-destructive ``EVOLVE_FV`` op and keeps the destructive
  ``RECREATE_FV`` for a genuinely breaking edit (feature retype).  Breaking
  edits must be validator-legal (``append_only=true`` still requires
  ``refresh_mode: FULL``); a ``refresh_mode`` flip is rejected at load time and
  never reaches the planner.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.imperative_executor import (
    _build_feature_view,
    _execute_op,
    _serialize_batch_fv_spec,
)
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    append_only_schema_evolution_kind,
    compute_local_spec_and_hash,
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


def _applied_state_for(local: dict[str, Any]) -> AppliedState:
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


def test_compile_to_spec_threads_append_only() -> None:
    """``compile_to_spec`` surfaces ``append_only`` into the inner ``spec``."""
    compiled = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
    assert compiled.get("spec", {}).get("append_only") is True


def test_compile_to_spec_omits_append_only_when_false() -> None:
    """A plain BFV never carries the default ``append_only`` into the spec."""
    local = _minimal_append_only_authoring(append_only=False)
    # Drop the append-only-only companions so the plain BFV is valid.
    local.pop("refresh_mode")
    compiled = compile_to_spec(local, "DB1", "SC1")
    assert "append_only" not in compiled.get("spec", {})


def test_hash_changes_when_append_only_toggles() -> None:
    """Toggling ``append_only`` bumps the structural hash."""
    on = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
    off = compile_to_spec(_minimal_append_only_authoring(append_only=False), "DB1", "SC1")
    assert _full_spec_hash(on) != _full_spec_hash(off)


def test_hash_stable_when_default_false_stripped() -> None:
    """Local (no ``append_only``) hashes equal to applied ``append_only: false``."""
    local = _minimal_append_only_authoring(append_only=False)
    compiled_local = compile_to_spec(local, "DB1", "SC1")
    # Simulate an applied payload where the runtime stamped the default False.
    applied = copy.deepcopy(compiled_local)
    applied["spec"]["append_only"] = False
    assert _full_spec_hash(compiled_local) == _full_spec_hash(applied)


def test_planner_emits_recreate_fv_when_append_only_toggles() -> None:
    """A pure ``append_only`` edit lands as ``RECREATE_FV(destructive=True)``."""
    baseline = _minimal_append_only_authoring(append_only=False)
    baseline.pop("refresh_mode")
    applied = _applied_state_for(baseline)
    edited = _minimal_append_only_authoring()

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
    fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.RECREATE_FV
    assert fv_ops[0].destructive is True
    assert fv_ops[0].payload.get("append_only") is True


def test_planner_no_change_for_unmodified_append_only() -> None:
    """An unchanged append-only spec plans ``NO_CHANGE``."""
    local = _minimal_append_only_authoring()
    applied = _applied_state_for(local)
    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
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


def test_executor_create_passes_append_only_kwarg() -> None:
    """``_build_feature_view`` forwards ``append_only`` to ``FeatureView``."""
    payload = _minimal_append_only_authoring()
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

    assert captured.get("append_only") is True


def test_executor_omits_append_only_kwarg_when_absent() -> None:
    """A plain BFV payload does not pass ``append_only`` to ``FeatureView``."""
    payload = _minimal_append_only_authoring(append_only=False)
    payload.pop("refresh_mode")
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

    assert "append_only" not in captured


class _AppendOnlyFakeFV:
    """FeatureView stand-in that reflects the ``append_only`` kwarg back on the
    instance so the executor's ``getattr(fv, 'append_only', ...)`` overwrite
    guard sees a truthy flag."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.append_only = bool(kwargs.get("append_only", False))


class _Op:
    def __init__(self, kind: OpKind, payload: dict[str, Any]) -> None:
        self.kind = kind
        self.payload = payload


def test_executor_create_forces_overwrite_false_for_append_only() -> None:
    """Core rejects ``register_feature_view(append_only_fv, overwrite=True)``
    unconditionally, so the CREATE_FV path must force ``overwrite=False`` for
    append-only targets even when ``PlanOptions.overwrite`` is ``True``."""
    payload = _minimal_append_only_authoring()
    op = _Op(OpKind.CREATE_FV, payload)
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    options = PlanOptions(overwrite=True)

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", _AppendOnlyFakeFV,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        MagicMock(),
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", options)

    assert fs.register_feature_view.call_count == 1
    _, kwargs = fs.register_feature_view.call_args
    assert kwargs.get("overwrite") is False


def test_executor_recreate_forces_overwrite_false_for_append_only() -> None:
    """The RECREATE_FV path deletes first, then re-registers; for append-only
    targets the re-register must use ``overwrite=False`` (core rejects the
    append_only + ``overwrite=True`` combination outright)."""
    payload = _minimal_append_only_authoring()
    op = _Op(OpKind.RECREATE_FV, payload)
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    options = PlanOptions(allow_recreate=True)

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", _AppendOnlyFakeFV,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        MagicMock(),
    ), patch(
        "snowflake.ml.feature_store.feature_view.OnlineStoreType",
        MagicMock(),
    ):
        _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", options)

    assert fs.register_feature_view.call_count == 1
    _, kwargs = fs.register_feature_view.call_args
    assert kwargs.get("overwrite") is False


def test_serialize_batch_fv_spec_does_not_pass_cron_as_target_lag() -> None:
    """Append-only BFVs carry a CRON ``refresh_freq`` (e.g. ``*/2 * * * * UTC``).
    ``_serialize_batch_fv_spec`` must NOT forward it as ``target_lag`` to
    ``_build_batch_feature_view_spec`` — that helper parses ``target_lag`` as a
    ``<number> <unit>`` duration and raises "Invalid interval format" on a CRON
    string, which would drop applied-state recovery to the minimal spec and loop
    the planner on RECREATE_FV instead of converging to NO_CHANGE."""
    fv = MagicMock(name="fv")
    fv.is_tiled = False
    fs = MagicMock(name="fs")
    fs.get_feature_view.return_value = fv
    built_spec = MagicMock(name="spec")
    built_spec.to_dict.return_value = {"kind": "BatchFeatureView", "spec": {}}
    fs._build_batch_feature_view_spec.return_value = built_spec
    session = MagicMock(name="session")

    row_dict = {
        "name": "MY_APPEND_BFV_DECL",
        "version": "V1",
        "database_name": "DB1",
        "schema_name": "SC1",
        "physical_dt_name": "MY_APPEND_BFV_DECL$V1",
        "refresh_freq": "*/2 * * * * UTC",
        "target_lag": "",
        "online_enabled": False,
        "append_only": True,
    }

    result = _serialize_batch_fv_spec(fs, session, row_dict)

    assert result is not None, "recovery must not fall back to None for a CRON append-only BFV"
    _, call_kwargs = fs._build_batch_feature_view_spec.call_args
    passed_target_lag = call_kwargs.get("target_lag")
    assert passed_target_lag != "*/2 * * * * UTC", "CRON refresh_freq must not be passed as target_lag"
    # A benign duration is used instead (parseable by the interval helper).
    assert passed_target_lag == "0 seconds"


def test_exporter_recovers_append_only_from_inner_spec() -> None:
    """``_build_full_fidelity_fv`` copies ``append_only`` into the YAML doc.

    The append-only export gate (``_append_only_export_blocker``) only emits the
    flag when the full loader companion contract is met, so the recovered inner
    spec carries a CRON ``refresh_freq`` and a ``timestamp_field`` alongside
    ``refresh_mode: FULL`` on a non-tiled BatchFV.
    """
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_AO", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "append_only": True,
            "refresh_mode": "FULL",
            "refresh_freq": "0 0 * * * UTC",
            "timestamp_field": "TS",
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_AO",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("append_only") is True


def test_exporter_omits_append_only_when_false() -> None:
    """The exporter does not emit ``append_only`` for a plain BFV."""
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_AO", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "append_only": False,
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_AO",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert "append_only" not in doc


def test_state_inject_append_only_from_list_row() -> None:
    """The injector populates ``append_only`` from the list-FV row cell."""
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_fields_from_list_row,
    )

    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    row = {"cluster_by": "", "refresh_mode": "FULL", "append_only": True}
    _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
    assert spec_payload["spec"]["append_only"] is True

    # False / missing cell -> key is not injected (default-stripped symmetry).
    spec_payload2: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    _inject_batch_fv_fields_from_list_row(spec_payload2, {"cluster_by": "", "append_only": False}, fv_obj=None)
    assert "append_only" not in spec_payload2["spec"]


def test_state_inject_append_only_from_fv_obj() -> None:
    """When the row lacks the cell, the rehydrated FV object supplies it."""
    from snowflake.ml.feature_store.decl.state import (
        _inject_batch_fv_fields_from_list_row,
    )

    spec_payload: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
    }
    fv_obj = MagicMock()
    fv_obj.append_only = True
    fv_obj.initialize = None
    _inject_batch_fv_fields_from_list_row(spec_payload, {"cluster_by": ""}, fv_obj=fv_obj)
    assert spec_payload["spec"]["append_only"] is True


# ---------------------------------------------------------------------------
# PR B — schema-evolution classifier + snapshot-preserving EVOLVE_FV routing
# ---------------------------------------------------------------------------


def _feature(src_name: str, out_name: str, type_: str = "LongType") -> dict[str, Any]:
    """A non-auto-derived feature entry (source name != output name)."""
    return {
        "source_column": {"name": src_name, "type": type_},
        "output_column": {"name": out_name, "type": type_},
    }


def _query_authoring(query: str, **overrides: Any) -> dict[str, Any]:
    """Build a valid append-only, query-backed BatchFeatureView authoring dict.

    Args:
        query: The ``BatchSource.query`` SQL body.
        **overrides: Field values that replace the valid-baseline defaults.

    Returns:
        A dict suitable for ``compile_to_spec`` / ``FeatureView.model_validate``.
    """
    return _minimal_append_only_authoring(
        sources=[{"name": "SRC1", "source_type": "Batch", "query": query}],
        **overrides,
    )


def test_local_half_entry_points_agree_for_mixed_case_spec() -> None:
    """``compute_local_spec_hash`` and ``compute_local_spec_and_hash`` must agree.

    Both route the authored (local) half through the single ``_hash_local_half``
    entry point.  This canary uses mixed-case identifiers so it stays green now
    (the halves normalise identically on this slice) but turns red at ``6e-int``
    if jkew/decl-fs/6e1.1's authored-identifier resolution is threaded into only
    one of the two local-hash entry points.
    """
    mixed_case = _minimal_append_only_authoring(
        entities=["userId"],
        sources=[
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "rawEvents",
                "columns": [{"name": "userId", "type": "StringType"}],
            }
        ],
        timestamp_col="eventTs",
    )
    hash_only = compute_local_spec_hash(mixed_case, "DB1", "SC1")
    _, compiled_hash = compute_local_spec_and_hash(mixed_case, "DB1", "SC1")
    assert hash_only == compiled_hash


class TestAppendOnlySchemaEvolutionClassifier:
    """``append_only_schema_evolution_kind`` labels a hash-differing append-only
    BFV edit as ``preserving`` (transformation-only), ``extend_only`` (features
    appended), or ``breaking`` (anything else)."""

    def test_source_only_change_is_preserving(self) -> None:
        applied = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        edited = _minimal_append_only_authoring(
            sources=[
                {
                    "name": "SRC1",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_V2",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ]
        )
        local = compile_to_spec(edited, "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "preserving"

    def test_appended_feature_is_breaking(self) -> None:
        # An authored ``spec.features`` append on an append-only BFV cannot be
        # realised by EVOLVE_FV: ``imperative_executor._build_features`` drops a
        # rename-only entry (no ``function`` / ``window_sec``) and append-only
        # forbids tiling, so every surviving feature entry is a rename the
        # executor never materialises.  Promising ``extend_only`` would re-emit
        # EVOLVE_FV on every replan and never converge, so any surviving
        # ``spec.features`` delta must be ``breaking``.
        applied = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        edited = _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT")])
        local = compile_to_spec(edited, "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_entity_change_is_breaking(self) -> None:
        applied = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        edited = _minimal_append_only_authoring(entities=["USER_ID", "ACCOUNT_ID"])
        local = compile_to_spec(edited, "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_feature_retype_is_breaking(self) -> None:
        applied = compile_to_spec(
            _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT", "LongType")]),
            "DB1",
            "SC1",
        )
        edited = _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT", "StringType")])
        local = compile_to_spec(edited, "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    # --- query-backed SELECT-list analysis (6e1.3) ---

    def test_query_where_only_rewrite_is_preserving(self) -> None:
        applied = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        local = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW WHERE AMOUNT > 0"), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "preserving"

    def test_query_select_reorder_is_breaking(self) -> None:
        applied = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        local = compile_to_spec(_query_authoring("SELECT USER_ID, AMOUNT, EVENT_TS FROM RAW"), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_query_select_drop_is_breaking(self) -> None:
        applied = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        local = compile_to_spec(_query_authoring("SELECT USER_ID FROM RAW"), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_query_select_trailing_append_is_extend_only(self) -> None:
        applied = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        local = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS, AMOUNT FROM RAW"), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "extend_only"

    def test_cte_where_only_rewrite_is_preserving(self) -> None:
        applied = compile_to_spec(
            _query_authoring("WITH c AS (SELECT USER_ID, EVENT_TS FROM RAW) SELECT USER_ID, EVENT_TS FROM c"),
            "DB1",
            "SC1",
        )
        local = compile_to_spec(
            _query_authoring(
                "WITH c AS (SELECT USER_ID, EVENT_TS FROM RAW WHERE AMOUNT > 0) SELECT USER_ID, EVENT_TS FROM c"
            ),
            "DB1",
            "SC1",
        )
        assert append_only_schema_evolution_kind(local, applied) == "preserving"

    def test_cte_outer_select_reorder_is_breaking(self) -> None:
        applied = compile_to_spec(
            _query_authoring("WITH c AS (SELECT USER_ID, EVENT_TS FROM RAW) SELECT USER_ID, EVENT_TS FROM c"),
            "DB1",
            "SC1",
        )
        local = compile_to_spec(
            _query_authoring("WITH c AS (SELECT USER_ID, EVENT_TS FROM RAW) SELECT EVENT_TS, USER_ID FROM c"),
            "DB1",
            "SC1",
        )
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_set_op_where_only_rewrite_is_preserving(self) -> None:
        applied = compile_to_spec(
            _query_authoring("SELECT USER_ID FROM RAW UNION ALL SELECT USER_ID FROM RAW2"), "DB1", "SC1"
        )
        local = compile_to_spec(
            _query_authoring("SELECT USER_ID FROM RAW WHERE AMOUNT > 0 UNION ALL SELECT USER_ID FROM RAW2"),
            "DB1",
            "SC1",
        )
        assert append_only_schema_evolution_kind(local, applied) == "preserving"

    def test_set_op_branch_change_is_breaking(self) -> None:
        applied = compile_to_spec(
            _query_authoring("SELECT USER_ID FROM RAW UNION ALL SELECT USER_ID FROM RAW2"), "DB1", "SC1"
        )
        local = compile_to_spec(
            _query_authoring("SELECT USER_ID FROM RAW UNION ALL SELECT ACCOUNT_ID FROM RAW2"), "DB1", "SC1"
        )
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_table_retarget_stays_preserving(self) -> None:
        applied = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        edited = _minimal_append_only_authoring(
            sources=[
                {
                    "name": "SRC1",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_V2",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ]
        )
        local = compile_to_spec(edited, "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "preserving"

    def test_table_to_query_is_breaking(self) -> None:
        # Swapping a table binding for a query exposes a visible output schema,
        # so it must not be waved through like a pure table retarget.
        applied = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        local = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"

    def test_query_to_table_is_breaking(self) -> None:
        applied = compile_to_spec(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"), "DB1", "SC1")
        local = compile_to_spec(_minimal_append_only_authoring(), "DB1", "SC1")
        assert append_only_schema_evolution_kind(local, applied) == "breaking"


def _plan_for(edited: dict[str, Any], applied: AppliedState, *, src_table: str = "RAW_EVENTS") -> Any:
    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table=src_table,
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(edited)
    return generate_plan(
        SpecBatch(specs=[ent, src, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )


def _plan_for_query(edited: dict[str, Any], applied: AppliedState) -> Any:
    """Plan a query-backed append-only FV edit (inline ``query`` source)."""
    ent = Entity(kind="Entity", name="USER", join_keys=[FSColumn(name="USER_ID", type="StringType")])
    fv = FeatureView.model_validate(edited)
    return generate_plan(
        SpecBatch(specs=[ent, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )


class TestPlannerAppendOnlyEvolveRouting:
    """The planner routes a schema-preserving append-only BFV edit to the
    non-destructive ``EVOLVE_FV`` op (which preserves ``$SNAPSHOTS``) and keeps
    ``RECREATE_FV`` for schema-breaking edits."""

    def test_preserving_transform_emits_evolve_fv(self) -> None:
        applied = _applied_state_for(_minimal_append_only_authoring())
        edited = _minimal_append_only_authoring(
            sources=[
                {
                    "name": "SRC1",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_V2",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ]
        )
        plan = _plan_for(edited, applied, src_table="RAW_EVENTS_V2")
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.EVOLVE_FV, f"expected EVOLVE_FV; got {fv_ops[0].kind.value}"
        assert fv_ops[0].destructive is False

    def test_breaking_change_emits_recreate_fv(self) -> None:
        # A genuinely schema-breaking edit must route to the destructive
        # RECREATE_FV path, NOT EVOLVE_FV.  Author the breaking change as a
        # feature retype (LongType -> StringType), mirroring
        # ``TestAppendOnlySchemaEvolutionClassifier.test_feature_retype_is_breaking``.
        # Do NOT express it as ``refresh_mode="INCREMENTAL"``: append_only=true
        # requires refresh_mode: FULL, so ``FeatureView._validate_append_only``
        # rejects that spec at load time and the planner is never reached.
        applied = _applied_state_for(
            _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT", "LongType")])
        )
        edited = _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT", "StringType")])
        plan = _plan_for(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV, f"expected RECREATE_FV; got {fv_ops[0].kind.value}"
        assert fv_ops[0].destructive is True

    def test_features_append_emits_recreate_fv(self) -> None:
        # An authored ``spec.features`` append is a schema change EVOLVE_FV
        # cannot materialise: ``_build_features`` drops rename-only entries and
        # append-only forbids tiling, so a promised EVOLVE_FV would never
        # converge.  It must route to the destructive RECREATE_FV path instead.
        applied = _applied_state_for(_minimal_append_only_authoring())
        edited = _minimal_append_only_authoring(features=[_feature("AMOUNT", "AMOUNT_OUT")])
        plan = _plan_for(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV, f"expected RECREATE_FV; got {fv_ops[0].kind.value}"
        assert fv_ops[0].destructive is True

    def test_non_append_only_transform_still_recreates(self) -> None:
        """A plain (non-append-only) BFV keeps the destructive RECREATE_FV path."""
        base = _minimal_append_only_authoring(append_only=False)
        base.pop("refresh_mode")
        applied = _applied_state_for(base)
        edited = dict(base)
        edited["sources"] = [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS_V2",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ]
        plan = _plan_for(edited, applied, src_table="RAW_EVENTS_V2")
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV
        assert fv_ops[0].destructive is True

    def test_query_where_only_rewrite_emits_evolve_fv(self) -> None:
        applied = _applied_state_for(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"))
        edited = _query_authoring("SELECT USER_ID, EVENT_TS FROM RAW WHERE AMOUNT > 0")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.EVOLVE_FV, f"expected EVOLVE_FV; got {fv_ops[0].kind.value}"
        assert fv_ops[0].destructive is False

    def test_query_select_reorder_emits_recreate_fv(self) -> None:
        applied = _applied_state_for(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"))
        edited = _query_authoring("SELECT USER_ID, AMOUNT, EVENT_TS FROM RAW")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV, f"expected RECREATE_FV; got {fv_ops[0].kind.value}"
        assert fv_ops[0].destructive is True

    def test_query_select_drop_emits_recreate_fv(self) -> None:
        applied = _applied_state_for(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"))
        edited = _query_authoring("SELECT USER_ID FROM RAW")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV
        assert fv_ops[0].destructive is True

    def test_table_to_query_emits_recreate_fv(self) -> None:
        # A table->query source flip is breaking: destructive RECREATE_FV, not
        # a snapshot-preserving EVOLVE_FV.
        applied = _applied_state_for(_minimal_append_only_authoring())
        edited = _query_authoring("SELECT USER_ID, EVENT_TS FROM RAW")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.RECREATE_FV
        assert fv_ops[0].destructive is True

    def test_query_select_trailing_append_emits_evolve_fv(self) -> None:
        applied = _applied_state_for(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"))
        edited = _query_authoring("SELECT USER_ID, EVENT_TS, AMOUNT FROM RAW")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.EVOLVE_FV
        assert fv_ops[0].destructive is False

    def test_query_select_reorder_recreate_reason_names_schema_change(self) -> None:
        applied = _applied_state_for(_query_authoring("SELECT USER_ID, EVENT_TS FROM RAW"))
        edited = _query_authoring("SELECT USER_ID, AMOUNT, EVENT_TS FROM RAW")
        plan = _plan_for_query(edited, applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        reason = fv_ops[0].reason.lower()
        assert "output schema" in reason
        assert "$snapshots" in reason


def test_executor_evolve_fv_preserves_snapshots() -> None:
    """``EVOLVE_FV`` dispatches to ``fs.update_feature_view(updated_feature_df=...)``
    and never calls ``delete_feature_view`` (which would drop ``$SNAPSHOTS``)."""
    payload = _minimal_append_only_authoring()
    op = _Op(OpKind.EVOLVE_FV, payload)
    session = MagicMock()
    session.table.return_value = MagicMock(name="df")
    fs = MagicMock()
    options = PlanOptions()

    _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", options)

    assert fs.delete_feature_view.call_count == 0, "EVOLVE_FV must not drop the FV (snapshots would be lost)"
    assert fs.update_feature_view.call_count == 1
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs.get("updated_feature_df") is not None


def test_executor_evolve_fv_builds_query_df() -> None:
    """A query-backed append-only BFV evolves via ``session.sql`` on the query."""
    payload = _minimal_append_only_authoring(
        sources=[{"name": "SRC1", "source_type": "Batch", "query": "SELECT USER_ID, EVENT_TS FROM RAW_EVENTS"}]
    )
    op = _Op(OpKind.EVOLVE_FV, payload)
    session = MagicMock()
    session.sql.return_value = MagicMock(name="df")
    fs = MagicMock()

    _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", PlanOptions())

    session.sql.assert_called_once_with("SELECT USER_ID, EVENT_TS FROM RAW_EVENTS")
    assert fs.update_feature_view.call_count == 1


def test_executor_evolve_fv_forwards_source_refs() -> None:
    """``EVOLVE_FV`` forwards the authored ``sources`` as ``_source_refs`` so core
    re-stamps ``FV_SOURCE_REFS`` on the recreate.  Without it the next plan
    recovers the stale query and re-emits ``EVOLVE_FV`` forever (silent drift)."""
    sources = [
        {
            "name": "SRC1",
            "source_type": "Batch",
            "query": "SELECT USER_ID, EVENT_TS FROM RAW_EVENTS WHERE AMOUNT > 0",
        }
    ]
    payload = _minimal_append_only_authoring(sources=sources)
    op = _Op(OpKind.EVOLVE_FV, payload)
    session = MagicMock()
    session.sql.return_value = MagicMock(name="df")
    fs = MagicMock()

    _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", PlanOptions())

    assert fs.update_feature_view.call_count == 1
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs.get("_source_refs") == sources, (
        "EVOLVE_FV must forward payload['sources'] as _source_refs so FV_SOURCE_REFS "
        f"is re-stamped on the recreate; got kwargs={kwargs!r}"
    )


class TestEvolveRestampConvergence:
    """Re-stamping ``FV_SOURCE_REFS`` on evolve means the recovered applied
    baseline tracks the *deployed* query.  With that contract the EVOLVE loop
    converges (re-planning the evolved YAML is ``NO_CHANGE``) and a YAML revert
    surfaces drift instead of a silent ``NO_CHANGE``.  Without the re-stamp the
    applied baseline keeps the stale query: the evolved YAML never converges and
    a revert hashes equal (hidden drift)."""

    _ORIGINAL = "SELECT USER_ID, EVENT_TS FROM RAW"
    _EVOLVED = "SELECT USER_ID, EVENT_TS FROM RAW WHERE AMOUNT > 0"

    def test_evolved_query_restamp_replans_no_change(self) -> None:
        """Applied baseline re-stamped with the evolved query -> same YAML is NO_CHANGE."""
        applied = _applied_state_for(_query_authoring(self._EVOLVED))
        plan = _plan_for_query(_query_authoring(self._EVOLVED), applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is OpKind.NO_CHANGE, (
            "A re-stamped EVOLVE_FV must converge: the evolved YAML re-plans to NO_CHANGE; "
            f"got {fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )

    def test_reverted_yaml_against_evolved_applied_surfaces_drift(self) -> None:
        """Reverting the YAML while the baseline holds the evolved query is NOT hidden."""
        applied = _applied_state_for(_query_authoring(self._EVOLVED))
        plan = _plan_for_query(_query_authoring(self._ORIGINAL), applied)
        fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind is not OpKind.NO_CHANGE, (
            "Reverting the query while the deployed baseline holds the evolved query must "
            f"surface drift, not a silent NO_CHANGE; got {fv_ops[0].kind.value}"
        )


class TestEvolveFvForwardsOperationalKwargs:
    """``EVOLVE_FV`` must realise the *whole* authored spec in one apply:
    operational drift (warehouse / refresh cadence / online routing) that
    rides alongside a snapshot-preserving transformation edit is forwarded
    on the same ``update_feature_view(updated_feature_df=...)`` call.

    Without this, a single ``snow feature apply`` that retargets the source
    *and* changes the warehouse evolves the query but leaves the refresh
    running on the stale warehouse until the next plan emits ``UPDATE_FV``.
    """

    @staticmethod
    def _run_evolve(payload: dict[str, Any], default_warehouse: str = "WH_DEFAULT") -> dict[str, Any]:
        """Dispatch an ``EVOLVE_FV`` op and return the ``update_feature_view`` kwargs."""
        op = _Op(OpKind.EVOLVE_FV, payload)
        session = MagicMock()
        session.table.return_value = MagicMock(name="df")
        session.sql.return_value = MagicMock(name="df")
        fs = MagicMock()

        _execute_op(fs, session, op, "DB1", "SC1", default_warehouse, PlanOptions())

        assert fs.delete_feature_view.call_count == 0, "EVOLVE_FV must never drop the FV ($SNAPSHOTS loss)"
        assert fs.update_feature_view.call_count == 1, "EVOLVE_FV must realise the spec in a single update call"
        _, kwargs = fs.update_feature_view.call_args
        assert kwargs.get("updated_feature_df") is not None, "EVOLVE_FV must still forward updated_feature_df"
        return dict(kwargs)

    def test_authored_warehouse_forwarded_with_feature_df(self) -> None:
        payload = _minimal_append_only_authoring(warehouse="WH_AUTHORED")
        kwargs = self._run_evolve(payload)
        assert kwargs.get("warehouse") == "WH_AUTHORED", (
            "An append-only BFV that authors a warehouse must forward it on the EVOLVE_FV "
            f"update_feature_view call; got kwargs={kwargs!r}"
        )

    def test_warehouse_falls_back_to_connection_default(self) -> None:
        payload = _minimal_append_only_authoring()
        payload.pop("warehouse", None)
        kwargs = self._run_evolve(payload, default_warehouse="WH_CONNECTION")
        assert kwargs.get("warehouse") == "WH_CONNECTION", (
            "An append-only BFV with no authored warehouse must fall back to the connection "
            f"default on the EVOLVE_FV update_feature_view call; got kwargs={kwargs!r}"
        )

    def test_refresh_freq_forwarded(self) -> None:
        payload = _minimal_append_only_authoring(refresh_freq="0 0 1 * * UTC")
        kwargs = self._run_evolve(payload)
        assert kwargs.get("refresh_freq") == "0 0 1 * * UTC", (
            "An append-only BFV refresh cadence must be forwarded on the EVOLVE_FV "
            f"update_feature_view call; got kwargs={kwargs!r}"
        )

    def test_online_enable_forwarded_as_online_config(self) -> None:
        payload = _minimal_append_only_authoring(online=True)
        kwargs = self._run_evolve(payload)
        online_config = kwargs.get("online_config")
        assert online_config is not None, (
            "An online append-only BFV must forward online_config on the EVOLVE_FV "
            f"update_feature_view call; got kwargs={kwargs!r}"
        )
        assert online_config.enable is True

    def test_online_disable_forwarded_as_online_config(self) -> None:
        payload = _minimal_append_only_authoring(online=False)
        kwargs = self._run_evolve(payload)
        online_config = kwargs.get("online_config")
        assert online_config is not None, (
            "An offline append-only BFV must still forward online_config (enable=False) so the "
            f"OFT is dropped in the same apply; got kwargs={kwargs!r}"
        )
        assert online_config.enable is False


def test_planner_source_and_warehouse_change_evolves_with_warehouse() -> None:
    """A single apply that retargets the source *and* changes the warehouse of an
    append-only BFV plans as one ``EVOLVE_FV`` whose payload carries the new
    warehouse, and the executor forwards both ``updated_feature_df`` and the new
    ``warehouse`` on the same ``update_feature_view`` call."""
    applied = _applied_state_for(_minimal_append_only_authoring(warehouse="WH_OLD"))
    edited = _minimal_append_only_authoring(
        warehouse="WH_NEW",
        sources=[
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS_V2",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
    )
    plan = _plan_for(edited, applied, src_table="RAW_EVENTS_V2")
    fv_ops = [op for op in plan.ops if op.name == "BFV_AO"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.EVOLVE_FV, f"expected EVOLVE_FV; got {fv_ops[0].kind.value}"
    assert fv_ops[0].payload.get("warehouse") == "WH_NEW"

    op = _Op(OpKind.EVOLVE_FV, fv_ops[0].payload)
    session = MagicMock()
    session.table.return_value = MagicMock(name="df")
    fs = MagicMock()
    _execute_op(fs, session, op, "DB1", "SC1", "WH_DEFAULT", PlanOptions())

    assert fs.delete_feature_view.call_count == 0
    assert fs.update_feature_view.call_count == 1
    _, kwargs = fs.update_feature_view.call_args
    assert kwargs.get("updated_feature_df") is not None
    assert kwargs.get("warehouse") == "WH_NEW", (
        "The source+warehouse edit must evolve the query AND move the refresh warehouse in one "
        f"apply; got kwargs={kwargs!r}"
    )


if __name__ == "__main__":
    pytest_driver.main()
