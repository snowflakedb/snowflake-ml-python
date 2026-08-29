"""Orchestrator-level regression pins for the three drift bugs the
``metadata-roundtrip-limitations`` plan closes.

Phase C / C1 deliverable 2: three explicit Bug 1 / 2 / 3 regression
tests that simulate the EXACT user-reported repro at the orchestrator
boundary — author YAML → compile + ``fetch_applied_state`` → plan →
assert NO_CHANGE (or no DROP_SOURCE) on the resulting ops.  These
tests are the unit-level safety net for R6 (per the plan's risk
register): if a future "quick fix" to ``decl/state.py`` /
``decl/invariants.py`` / ``decl/planner.py`` re-introduces one of the
three drifts that closed the user's regression, one of these three
tests fails.

The three pins:

1. :class:`TestBug1Regression` — table-backed BatchFV column round-trip.
   The local ``BatchSource`` carries the operator-authored
   ``columns`` list verbatim; the applied side recovers the same
   list via the Phase-B3 ``FV_SOURCE_REFS`` metadata column on
   ``list_feature_views()``.  Plan must NOT emit any
   ``CREATE_SOURCE`` / ``RECREATE_SOURCE`` / ``UPDATE_SOURCE`` /
   ``DROP_SOURCE`` for the source.
2. :class:`TestBug2Regression` — tiled online BatchFV operational
   canonicalization.  Same warehouse, same refresh_freq, same
   online_store_type on both sides → plan must emit
   ``NO_CHANGE`` (not ``UPDATE_FV``).  Pins
   :func:`invariants._canonicalize_operational_for_drift` (Phase
   B5) — the canonicalizer must collapse the mixed-case enum
   values + duration aliases the runtime returns into the local-
   compile form.
3. :class:`TestBug3Regression` — tiled BFV with FG-shape source +
   query-shape BFV with synthetic ``<FV>__SOURCE`` name.  After a
   fresh apply, full-directory-mode plan must emit ZERO
   ``DROP_SOURCE`` ops.  Pins B3's elimination of the synthetic
   placeholder + the metadata-authoritative name preservation
   across both source-binding shapes.

Each test simulates the exact orchestrator flow:

    local YAML
      → spec_compiler.compile_to_spec(local)
      → "register via FeatureStore" (mocked: feed the compiled
         SPECIFICATION JSON + the ``FV_SOURCE_REFS`` payload through
         :func:`decl_api.fetch_applied_state`)
      → :func:`decl_api.generate_plan`(applied, local)
      → assert all ops match the per-bug contract.

References:
* ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
  §"Phase C → C1" (test descriptions) and §"Risk register" R6.
* ``docs/LIMITATIONS.md`` §"BatchFeatureView — tiled BFV ``DESCRIBE
  TYPE = SPECIFICATION`` round-trip is lossy" (L1) and
  §"FV-level ``backfill:`` block — operational, non-recovered" (L2).
* ``snowml/snowflake/ml/feature_store/decl/state.py`` —
  ``_inject_batch_fv_source_from_metadata`` (the Phase-B3
  authoritative recovery path).
"""

from __future__ import annotations

import copy
from typing import Any

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FSColumn,
)
from snowflake.ml.feature_store.decl.types import PlanOptions, SpecBatch
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Shared environment — matches the user-reported repro from
# .cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md §"Risk
# register" R6 (JKEW_DB.JKEW_SCHEMA, EVENTS_BATCH_DECL physical table).
# ---------------------------------------------------------------------------

_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"


def _entity_row() -> dict[str, Any]:
    """Build a ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'`` row.

    Returns:
        dict: Matches the row shape :func:`state._build_entity_object`
        consumes; carries the entity's allowed_values join-key.
    """
    return {
        "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
        "database_name": _DB,
        "schema_name": _SCH,
        "allowed_values": '["USER_ID"]',
        "comment": "",
    }


def _show_oft_row(name: str, version: str) -> dict[str, Any]:
    """Build a ``SHOW ONLINE FEATURE TABLES`` row for an online FV.

    Args:
        name: FV name (the OFT base name).
        version: FV version string.

    Returns:
        Dict matching the row shape ``state.fetch_applied_state``
        consumes for OFT-backed FVs.
    """
    return {
        "name": f"{name.upper()}${version.upper()}$ONLINE",
        "database_name": _DB,
        "schema_name": _SCH,
        "scheduling_state": "ACTIVE",
    }


def _list_fv_row(
    *,
    name: str,
    version: str,
    kind: str = "BATCH",
    online_enabled: bool = False,
    target_lag: str = "",
    refresh_freq: str = "5 minutes",
    warehouse: str = "TEST_WH",
    desc: str = "",
    source_refs: list[dict[str, Any]] | None = None,
    spec_text: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``list_feature_views()`` row matching the Phase-1 contract.

    Mirrors :func:`imperative_executor.fetch_feature_view_rows`'s
    canonical row shape (also used by the offline-BFV idempotency
    suite in the CLI repo).  The ``source_refs`` cell is the
    Phase-A1 / B3 authoritative source-binding payload — every
    re-plan after a successful apply MUST surface it here so the
    decl recovery path reads operator-authored names instead of
    falling back to the legacy shim.

    Args:
        name: FV name.
        version: FV version.
        kind: ``"BATCH"`` / ``"STREAMING"`` / ``"REALTIME"`` —
            mapped to the declarative kind via
            :data:`state._FV_KIND_MAP`.
        online_enabled: ``True`` when the FV has an OFT (online).
        target_lag: Optional ``TARGET_LAG`` string (online OFT
            staleness).
        refresh_freq: DT refresh cadence string the runtime returns
            for the offline Dynamic Table.
        warehouse: Warehouse identifier the runtime returns.
        desc: FV-level description text from the row's ``desc``
            column.
        source_refs: ``FV_SOURCE_REFS`` payload (list of
            source-ref dicts).
        spec_text: Optional pre-enriched SPECIFICATION dict (e.g.
            from :func:`imperative_executor._serialize_batch_fv_spec`).

    Returns:
        Dict matching the canonical row shape consumed by
        :func:`state._build_offline_fv_object` /
        :func:`state.fetch_applied_state`.
    """
    return {
        "name": name,
        "version": version,
        "database_name": _DB,
        "schema_name": _SCH,
        "kind": kind,
        "entities": ["USER_ID"],
        "online_enabled": online_enabled,
        "target_lag": target_lag,
        "refresh_freq": refresh_freq,
        "warehouse": warehouse,
        "desc": desc,
        "physical_dt_name": f"{name}${version}",
        "spec_text": spec_text,
        "source_refs": source_refs or [],
    }


def _user_id_columns() -> list[dict[str, Any]]:
    """Canonical USER_ID column schema shared across all C1 fixtures."""
    return [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "EVENT_TS", "type": "TimestampType"},
        {"name": "AMOUNT", "type": "DoubleType"},
    ]


def _entity_user_id() -> Entity:
    return Entity(
        kind="Entity",
        name="USER_ID",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )


def _run_plan(
    batch: SpecBatch,
    applied_state: Any,
    *,
    full_directory_mode: bool = False,
) -> Any:
    """Drive the exact CLI orchestrator pipeline and return (errors, plan).

    Mirrors :func:`tests.test_export_plan_round_trip._validate_and_plan`
    so a regression in either harness yields parallel signal.  Driven
    via :mod:`decl.api` (not the internal modules) so a future
    refactor of the api surface immediately surfaces here.

    Args:
        batch: The local :class:`SpecBatch` (Entity + Source + FV).
        applied_state: The :class:`AppliedState` returned by
            :func:`decl_api.fetch_applied_state` against a simulated
            post-apply runtime snapshot.
        full_directory_mode: When ``True``, the planner emits
            ``DROP_*`` ops for applied objects not present in the
            batch.  Bug 3's pin specifically requires this so the
            orphan-detection pass is exercised.

    Returns:
        Tuple of ``(errors, plan)`` where ``errors`` is a list of
        validator results with severity ``"ERROR"``.
    """
    decl_api.resolve_datasource_columns(batch)
    errors = [
        r
        for r in decl_api.validate_specs(
            batch,
            applied_state,
            target_database=_DB,
            target_schema=_SCH,
        )
        if r.severity == "ERROR"
    ]
    if errors:
        return errors, None
    plan = decl_api.generate_plan(
        batch,
        applied_state,
        PlanOptions(full_directory_mode=full_directory_mode),
        database=_DB,
        schema=_SCH,
    )
    return errors, plan


# ===========================================================================
# Bug 1 — Source columns round-trip on a table-backed BatchSource
# ===========================================================================


class TestBug1Regression:
    """Bug 1: a freshly applied table-backed BatchFV must NOT trigger
    a spurious source-level ``RECREATE_SOURCE`` on the next plan.

    User-reported repro (per
    ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
    §"Risk register" R6): the operator authors a BatchSource with
    a full ``columns:`` schema, applies the FV, and the very next
    ``snow feature plan`` emits ``RECREATE_SOURCE`` /
    ``DROP_SOURCE`` for the source because the recovered applied
    side had ``columns: []`` (the lossy DESCRIBE round-trip).

    Phase B3 closes this by reading the operator-authored
    ``columns`` list directly from the ``FV_SOURCE_REFS`` metadata
    column on the ``list_feature_views()`` row, so the recovered
    Datasource ``spec_payload`` carries the same columns the local
    YAML authored.  See ``docs/LIMITATIONS.md`` §"BatchFeatureView
    — tiled BFV ``DESCRIBE TYPE = SPECIFICATION`` round-trip is
    lossy" for the upstream context.
    """

    _FV_NAME = "EVENTS_FV_DECL"
    _FV_VERSION = "V1"
    _SOURCE_NAME = "EVENTS_BATCH_DECL"
    _SOURCE_TABLE = f"{_DB}.{_SCH}.EVENTS_BATCH"

    def _local_authoring(self) -> dict[str, Any]:
        """Operator-authored BFV (offline, table-backed)."""
        return {
            "kind": "BatchFeatureView",
            "name": self._FV_NAME,
            "version": self._FV_VERSION,
            "database": _DB,
            "schema_": _SCH,
            "online": False,
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": self._SOURCE_NAME,
                    "source_type": "Batch",
                    "table": self._SOURCE_TABLE,
                    "columns": _user_id_columns(),
                }
            ],
            "refresh_freq": "5 minutes",
        }

    def _source_refs(self) -> list[dict[str, Any]]:
        """``FV_SOURCE_REFS`` payload mirroring what the runtime would
        write after a successful apply.

        Carries the operator-authored ``name`` + ``columns`` verbatim
        so the Phase-B3 recovery is bit-for-bit symmetric with the
        local YAML.

        Returns:
            list[dict]: A single-entry list with the operator-authored
            BatchSource ref shape consumed by
            :func:`state._inject_batch_fv_source_from_metadata`.
        """
        return [
            {
                "name": self._SOURCE_NAME,
                "source_type": "Batch",
                "table": self._SOURCE_TABLE,
                "columns": _user_id_columns(),
            }
        ]

    def test_clean_apply_no_recreate_source_on_table_backed_datasource(self) -> None:
        """Re-plan against a freshly applied table-backed BFV must emit
        zero non-NO_CHANGE ops naming the BatchSource.

        Pinned by ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
        §"Phase C → C1" first bullet.  Bug 1 closure surface is the
        Phase-B3 ``_inject_batch_fv_source_from_metadata`` helper
        plus state.py's ``_datasource_objects_from_specs`` keying on
        the operator-authored ``BatchSource.name``.
        """
        local_authoring = self._local_authoring()
        local_fv = FeatureView.model_validate(local_authoring)
        local_src = BatchSource(
            kind="BatchSource",
            name=self._SOURCE_NAME,
            table=self._SOURCE_TABLE,
            columns=[FSColumn(**c) for c in _user_id_columns()],
        )
        batch = SpecBatch(specs=[_entity_user_id(), local_src, local_fv])

        # Simulated post-apply runtime snapshot:
        # offline BFV — no OFT row.  feature_view_rows carries the
        # full spec_text + FV_SOURCE_REFS metadata payload.
        compiled = compile_to_spec(local_authoring, _DB, _SCH)
        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[],
            entity_rows=[_entity_row()],
            feature_view_rows=[
                _list_fv_row(
                    name=self._FV_NAME,
                    version=self._FV_VERSION,
                    kind="BATCH",
                    online_enabled=False,
                    refresh_freq="5 minutes",
                    source_refs=self._source_refs(),
                    spec_text=compiled,
                )
            ],
            default_database=_DB,
            default_schema=_SCH,
        )

        errors, plan = _run_plan(batch, applied_state)
        assert errors == [], f"validation_failed on Bug-1 repro; errors={errors!r}"

        spurious = [op for op in plan.ops if op.name == self._SOURCE_NAME and op.kind.value != "NO_CHANGE"]
        assert spurious == [], (
            "Bug 1 regression: a clean re-plan against a table-backed BatchFV "
            "must NOT emit any non-NO_CHANGE op for the BatchSource — every "
            "source op should resolve to NO_CHANGE because FV_SOURCE_REFS "
            "preserves columns + name through the round-trip.  Got: "
            f"{[(o.kind.value, o.reason) for o in spurious]!r}"
        )

        # Belt-and-suspenders: the FV itself must also round-trip
        # cleanly (the column drift the source recovery would
        # otherwise introduce would bubble into the FV hash via
        # _resolve_datasource_columns).
        fv_ops = [op for op in plan.ops if op.name == self._FV_NAME]
        assert len(fv_ops) == 1, f"expected one FV op; got {fv_ops!r}"
        assert fv_ops[0].kind.value == "NO_CHANGE", (
            "Bug 1 closure: the table-backed BFV must also round-trip to "
            "NO_CHANGE on a clean apply (otherwise the source-column drift "
            "leaked into the FV-level hash).  Got "
            f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )


# ===========================================================================
# Bug 2 — Tiled online BFV operational canonicalization
# ===========================================================================


class TestBug2Regression:
    """Bug 2: a freshly applied tiled online BatchFV with identical
    operational knobs on both sides must emit ``NO_CHANGE`` (not
    ``UPDATE_FV``).

    User-reported repro: the operator authors a tiled online BFV
    (``aggregation_specs`` + ``aggregation_secondary_keys`` +
    ``online_config``); after applying, the very next ``snow
    feature plan`` emits a phantom ``UPDATE_FV`` because the
    operational-drift detector sees mixed-case ``online_store_type``
    strings (``"POSTGRES"`` deployed vs. ``"postgres"`` compiled)
    or differently-formatted ``target_lag_sec`` (``"5 minutes"``
    vs. ``"300 SECONDS"``).

    Phase B5 closes this by routing both halves of the operational
    diff through :func:`invariants._canonicalize_operational_for_drift`
    (a wrapper over the snowml-core
    ``_canonicalize_operational_fields`` helper introduced in
    Phase A4).  See ``docs/LIMITATIONS.md`` §"Batch vs streaming —
    declarative ``UPDATE_FV`` asymmetry" for the upstream context.
    """

    _FV_NAME = "TILED_BFV_DECL"
    _FV_VERSION = "V1"
    _SOURCE_NAME = "EVENTS_BATCH_DECL"
    _SOURCE_TABLE = f"{_DB}.{_SCH}.EVENTS_BATCH"

    def _local_authoring(self) -> dict[str, Any]:
        """Operator-authored tiled online BFV."""
        return {
            "kind": "BatchFeatureView",
            "name": self._FV_NAME,
            "version": self._FV_VERSION,
            "database": _DB,
            "schema_": _SCH,
            "online": True,
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": self._SOURCE_NAME,
                    "source_type": "Batch",
                    "table": self._SOURCE_TABLE,
                    "columns": _user_id_columns(),
                }
            ],
            "timestamp_col": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "refresh_freq": "5 minutes",
            "aggregation_secondary_keys": ["EVENT_TS"],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        }

    def _source_refs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self._SOURCE_NAME,
                "source_type": "Batch",
                "table": self._SOURCE_TABLE,
                "columns": _user_id_columns(),
            }
        ]

    def test_clean_apply_no_update_fv_on_tiled_online_bfv(self) -> None:
        """Identical tiled online BFV on both sides → NO_CHANGE.

        Pinned by ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
        §"Phase C → C1" second bullet.  Closes the canonicalization
        surface that prevents phantom UPDATE_FV ops on a clean
        replan.  This is R6's explicit safety net — if Phase B5's
        Bug-2 root cause was actually a different field (not
        ``online_store_type`` / ``target_lag_sec``), the resulting
        UPDATE_FV would surface here.
        """
        local_authoring = self._local_authoring()
        local_fv = FeatureView.model_validate(local_authoring)
        local_src = BatchSource(
            kind="BatchSource",
            name=self._SOURCE_NAME,
            table=self._SOURCE_TABLE,
            columns=[FSColumn(**c) for c in _user_id_columns()],
        )
        batch = SpecBatch(specs=[_entity_user_id(), local_src, local_fv])

        # Simulated post-apply runtime snapshot: online BFV has an OFT
        # row plus a SPECIFICATION JSON.  The SPECIFICATION JSON
        # mirrors what compile_to_spec(local) produces because the
        # operator made no edits between apply and replan.  The
        # feature_view_rows entry carries FV_SOURCE_REFS so
        # _inject_batch_fv_source_from_metadata is exercised end-to-end.
        compiled = compile_to_spec(local_authoring, _DB, _SCH)
        oft_name = f"{self._FV_NAME}${self._FV_VERSION}$ONLINE"
        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[_show_oft_row(self._FV_NAME, self._FV_VERSION)],
            specification_map={oft_name: copy.deepcopy(compiled)},
            entity_rows=[_entity_row()],
            feature_view_rows=[
                _list_fv_row(
                    name=self._FV_NAME,
                    version=self._FV_VERSION,
                    kind="BATCH",
                    online_enabled=True,
                    refresh_freq="5 minutes",
                    source_refs=self._source_refs(),
                    spec_text=copy.deepcopy(compiled),
                )
            ],
            default_database=_DB,
            default_schema=_SCH,
        )

        errors, plan = _run_plan(batch, applied_state)
        assert errors == [], f"validation_failed on Bug-2 repro; errors={errors!r}"

        fv_ops = [op for op in plan.ops if op.name == self._FV_NAME]
        assert len(fv_ops) == 1, f"expected one FV op; got {fv_ops!r}"
        assert fv_ops[0].kind.value == "NO_CHANGE", (
            "Bug 2 regression: a clean re-plan against an unchanged tiled "
            "online BFV must emit NO_CHANGE (not UPDATE_FV).  This is the "
            "R6 safety net — any residual canonicalization gap (the "
            "_canonicalize_operational_for_drift surface) surfaces as a "
            "phantom UPDATE_FV here.  Got "
            f"{fv_ops[0].kind.value} (reason={fv_ops[0].reason!r})"
        )
        assert fv_ops[0].destructive is False


# ===========================================================================
# Bug 3 — Zero DROP_SOURCE for tiled-BFV / synthetic-query-source after apply
# ===========================================================================


class TestBug3Regression:
    """Bug 3: a freshly applied tree carrying BOTH a tiled BFV (with a
    Datasource-shaped source) AND a query-shape BFV (with a
    synthetic ``<FV>__SOURCE`` pre-Phase-B name shape) must emit
    ZERO ``DROP_SOURCE`` ops on the next full-directory plan.

    User-reported repro: pre Phase-B3 the recovery path leaked the
    inner-materialised DT table name into ``sources[0].name``
    (tiled BFV case) and the synthetic ``<FV>__SOURCE`` placeholder
    into ``sources[0].name`` (query-shape BFV case).  The applied
    side therefore registered ``Datasource:DB.SCHEMA:RAW_EVENTS_TBL``
    / ``Datasource:DB.SCHEMA:MYFV__SOURCE`` while the local side
    declared ``Datasource:DB.SCHEMA:EVENTS_FG_DECL`` /
    ``Datasource:DB.SCHEMA:EVENTS_QUERY_DECL`` — a key mismatch the
    orphan-detection pass converted to ``DROP_SOURCE`` ops in
    full-directory mode.

    Phase B3 closes this by (1) eliminating the synthetic
    ``<FV>__SOURCE`` placeholder and (2) reading the
    operator-authored name straight from ``FV_SOURCE_REFS``.
    Pre-fix the regression surfaced live on every
    ``snow feature plan --from .`` after an init+apply cycle.
    """

    _TILED_FV_NAME = "TILED_FG_BFV_DECL"
    _TILED_SOURCE_NAME = "EVENTS_FG_DECL"
    _TILED_SOURCE_TABLE = f"{_DB}.{_SCH}.RAW_EVENTS_FG_TBL"

    _QUERY_FV_NAME = "QUERY_BFV_DECL"
    _QUERY_SOURCE_NAME = "EVENTS_QUERY_DECL"
    _QUERY_SQL = "SELECT user_id, event_ts, amount FROM " f"{_DB}.{_SCH}.RAW_EVENTS WHERE event_ts > '2024-01-01'"

    def _tiled_local_authoring(self) -> dict[str, Any]:
        """Operator-authored tiled BFV (FG-shape source)."""
        return {
            "kind": "BatchFeatureView",
            "name": self._TILED_FV_NAME,
            "version": "V1",
            "database": _DB,
            "schema_": _SCH,
            "online": True,
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": self._TILED_SOURCE_NAME,
                    "source_type": "Batch",
                    "table": self._TILED_SOURCE_TABLE,
                    "columns": _user_id_columns(),
                }
            ],
            "timestamp_col": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "refresh_freq": "5 minutes",
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        }

    def _query_local_authoring(self) -> dict[str, Any]:
        """Operator-authored query-shape BFV (synthetic ``__SOURCE``
        pre-Phase-B; now uses authored name).

        Returns:
            dict: Authoring-format BFV spec wired against the
            query-shape ``EVENTS_QUERY_DECL`` BatchSource so
            ``compile_to_spec`` produces a wire-form FV payload bound
            to the synthetic query source.
        """
        return {
            "kind": "BatchFeatureView",
            "name": self._QUERY_FV_NAME,
            "version": "V1",
            "database": _DB,
            "schema_": _SCH,
            "online": False,
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": self._QUERY_SOURCE_NAME,
                    "source_type": "Batch",
                    "query": self._QUERY_SQL,
                    "columns": _user_id_columns(),
                }
            ],
            "refresh_freq": "10 minutes",
        }

    def test_clean_apply_no_drop_source_on_tiled_bfv_or_synthetic_query_source(self) -> None:
        """Re-plan in full-directory mode against TWO freshly applied
        BFVs (tiled FG-shape + query-shape) must emit ZERO
        ``DROP_SOURCE`` ops.

        Pinned by ``.cursor/plans/metadata-roundtrip-limitations_fe945c22.plan.md``
        §"Phase C → C1" third bullet.  Both source-binding shapes
        (table-via-FG and query-shape) must preserve the
        operator-authored name end-to-end so the orphan-detection
        pass does not synthesise a phantom DROP_SOURCE on the
        round-trip.
        """
        tiled_local = self._tiled_local_authoring()
        query_local = self._query_local_authoring()

        tiled_fv = FeatureView.model_validate(tiled_local)
        tiled_src = BatchSource(
            kind="BatchSource",
            name=self._TILED_SOURCE_NAME,
            table=self._TILED_SOURCE_TABLE,
            columns=[FSColumn(**c) for c in _user_id_columns()],
        )
        query_fv = FeatureView.model_validate(query_local)
        query_src = BatchSource(
            kind="BatchSource",
            name=self._QUERY_SOURCE_NAME,
            query=self._QUERY_SQL,
            columns=[FSColumn(**c) for c in _user_id_columns()],
        )
        batch = SpecBatch(
            specs=[
                _entity_user_id(),
                tiled_src,
                query_src,
                tiled_fv,
                query_fv,
            ]
        )

        # Build the simulated runtime snapshot: tiled BFV → OFT +
        # SPECIFICATION; query BFV → feature_view_rows only (online:
        # false).  Both FVs surface FV_SOURCE_REFS payloads carrying
        # the operator-authored names so the recovered applied side's
        # Datasource entries collide on key with the local
        # BatchSource specs (no orphan → no DROP_SOURCE).
        tiled_compiled = compile_to_spec(tiled_local, _DB, _SCH)
        query_compiled = compile_to_spec(query_local, _DB, _SCH)
        tiled_oft = f"{self._TILED_FV_NAME}$V1$ONLINE"

        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[_show_oft_row(self._TILED_FV_NAME, "V1")],
            specification_map={tiled_oft: copy.deepcopy(tiled_compiled)},
            entity_rows=[_entity_row()],
            feature_view_rows=[
                _list_fv_row(
                    name=self._TILED_FV_NAME,
                    version="V1",
                    kind="BATCH",
                    online_enabled=True,
                    refresh_freq="5 minutes",
                    source_refs=[
                        {
                            "name": self._TILED_SOURCE_NAME,
                            "source_type": "Batch",
                            "table": self._TILED_SOURCE_TABLE,
                            "columns": _user_id_columns(),
                        }
                    ],
                    spec_text=copy.deepcopy(tiled_compiled),
                ),
                _list_fv_row(
                    name=self._QUERY_FV_NAME,
                    version="V1",
                    kind="BATCH",
                    online_enabled=False,
                    refresh_freq="10 minutes",
                    source_refs=[
                        {
                            "name": self._QUERY_SOURCE_NAME,
                            "source_type": "Batch",
                            "query": self._QUERY_SQL,
                            "columns": _user_id_columns(),
                        }
                    ],
                    spec_text=copy.deepcopy(query_compiled),
                ),
            ],
            default_database=_DB,
            default_schema=_SCH,
        )

        # Sanity: the recovered Datasource AppliedObjects carry the
        # operator-authored names (not the inner-materialised table
        # ident, not the synthetic ``<FV>__SOURCE``).  Pre-Phase-B3
        # this assertion failed — the synthetic name leaked into
        # the applied state and produced the orphan key mismatch.
        applied_datasource_names = {obj.name for obj in applied_state.objects.values() if obj.kind == "Datasource"}
        assert self._TILED_SOURCE_NAME in applied_datasource_names, (
            "Bug 3 precondition: the tiled BFV's recovered Datasource "
            "must carry the operator-authored name (not the inner "
            "materialised table ident).  Recovered datasource names: "
            f"{sorted(applied_datasource_names)!r}"
        )
        assert self._QUERY_SOURCE_NAME in applied_datasource_names, (
            "Bug 3 precondition: the query-shape BFV's recovered "
            "Datasource must carry the operator-authored name (not the "
            "synthetic ``<FV>__SOURCE`` placeholder).  Recovered "
            f"datasource names: {sorted(applied_datasource_names)!r}"
        )

        # full_directory_mode=True is required so the orphan-detection
        # pass exercises the DROP_SOURCE branch.  Incremental mode
        # would skip the pass entirely and hide a residual regression.
        errors, plan = _run_plan(batch, applied_state, full_directory_mode=True)
        assert errors == [], f"validation_failed on Bug-3 repro; errors={errors!r}"

        drop_source_ops = [op for op in plan.ops if op.kind.value == "DROP_SOURCE"]
        assert drop_source_ops == [], (
            "Bug 3 regression: a clean re-plan against a tiled-BFV / "
            "query-shape-BFV pair must emit ZERO DROP_SOURCE ops in "
            "full-directory mode.  Pre Phase-B3 the synthetic "
            "``<FV>__SOURCE`` placeholder and the inner-materialised "
            "table-name leak created phantom orphan keys here.  Got: "
            f"{[(o.name, o.reason) for o in drop_source_ops]!r}"
        )

        # Defensive belt-and-suspenders: both FVs themselves should
        # also round-trip cleanly to NO_CHANGE.  A failure here
        # would indicate the source-name leak resurrected through a
        # different surface (e.g. FV hash divergence).
        fv_ops = {op.name: op for op in plan.ops if op.name in (self._TILED_FV_NAME, self._QUERY_FV_NAME)}
        for fv_name in (self._TILED_FV_NAME, self._QUERY_FV_NAME):
            assert fv_name in fv_ops, f"missing FV op for {fv_name} in plan {[op.name for op in plan.ops]!r}"
            assert fv_ops[fv_name].kind.value == "NO_CHANGE", (
                f"Bug 3 belt-and-suspenders: {fv_name} must round-trip "
                "to NO_CHANGE on a clean apply; got "
                f"{fv_ops[fv_name].kind.value} (reason={fv_ops[fv_name].reason!r})"
            )


# ---------------------------------------------------------------------------
# Hash-symmetry sanity pin: prove the compiled local hash equals the
# recovered applied hash for each Bug regression FV.  This is the
# precondition the planner's hash-matches branch relies on; if a
# future refactor breaks the symmetry one of the per-bug NO_CHANGE
# assertions above masks the real failure (it would surface as
# RECREATE_FV / UPDATE_FV instead of the hash-equality failure mode).
# Surfacing the precondition explicitly turns that masking into a
# direct error message.
# ---------------------------------------------------------------------------


class TestRegressionHashSymmetryPreconditions:
    """Pin the hash-symmetry precondition each Bug regression relies on."""

    def test_bug1_local_compile_hash_equals_recovered_applied_hash(self) -> None:
        bug1 = TestBug1Regression()
        local_authoring = bug1._local_authoring()
        compiled_local = compile_to_spec(local_authoring, _DB, _SCH)
        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[],
            entity_rows=[_entity_row()],
            feature_view_rows=[
                _list_fv_row(
                    name=bug1._FV_NAME,
                    version=bug1._FV_VERSION,
                    kind="BATCH",
                    online_enabled=False,
                    refresh_freq="5 minutes",
                    source_refs=bug1._source_refs(),
                    spec_text=copy.deepcopy(compiled_local),
                )
            ],
            default_database=_DB,
            default_schema=_SCH,
        )
        applied_obj = next(
            o for o in applied_state.objects.values() if o.kind == "BatchFeatureView" and o.name == bug1._FV_NAME
        )
        assert _full_spec_hash(compiled_local) == applied_obj.content_hash, (
            "Bug 1 precondition: local-compile hash must equal "
            "applied content_hash for the planner's NO_CHANGE branch "
            "to fire.  A drift here is a hash-symmetry regression in "
            "compile_to_spec / _full_spec_hash / "
            "_inject_batch_fv_source_from_metadata."
        )

    def test_bug2_local_compile_hash_equals_recovered_applied_hash(self) -> None:
        bug2 = TestBug2Regression()
        local_authoring = bug2._local_authoring()
        compiled_local = compile_to_spec(local_authoring, _DB, _SCH)
        oft_name = f"{bug2._FV_NAME}${bug2._FV_VERSION}$ONLINE"
        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[_show_oft_row(bug2._FV_NAME, bug2._FV_VERSION)],
            specification_map={oft_name: copy.deepcopy(compiled_local)},
            entity_rows=[_entity_row()],
            feature_view_rows=[
                _list_fv_row(
                    name=bug2._FV_NAME,
                    version=bug2._FV_VERSION,
                    kind="BATCH",
                    online_enabled=True,
                    refresh_freq="5 minutes",
                    source_refs=bug2._source_refs(),
                    spec_text=copy.deepcopy(compiled_local),
                )
            ],
            default_database=_DB,
            default_schema=_SCH,
        )
        applied_obj = next(
            o for o in applied_state.objects.values() if o.kind == "BatchFeatureView" and o.name == bug2._FV_NAME
        )
        assert _full_spec_hash(compiled_local) == applied_obj.content_hash, (
            "Bug 2 precondition: local-compile hash must equal "
            "applied content_hash for the planner's NO_CHANGE branch "
            "to fire on the tiled online BFV.  Any drift here is a "
            "Phase-B normalisation regression."
        )


if __name__ == "__main__":
    pytest_driver.main()
