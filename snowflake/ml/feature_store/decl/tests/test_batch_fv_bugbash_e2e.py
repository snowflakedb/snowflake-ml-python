"""High-fidelity end-to-end reproduction of ``scripts/verify_batch_fv_bug_bash.sh``.

Each test in this module mirrors one of the open TODOs injected into
``declarative_feature_store/BATCH_FV_BUG_BASH.md`` by the live verifier:

- §6  CREATE_FV / RECREATE_FV for both the table-backed and SQL-backed BFVs.
- §7  UPDATE_FV after a ``refresh_freq``-only edit on the table FV.
- §7b Offline-only BFV ``plan → apply → plan`` idempotency
       (NO_CHANGE / UPDATE_FV / RECREATE_FV).
- §8  RECREATE_FV after a ``BatchSource.table`` swap on the table FV.

The tests author the *exact* YAML strings the verify script writes into a
temporary project directory, then drive the **real** loader, datasource
resolution, ``fetch_applied_state`` reconstruction, and ``generate_plan``
— the layer the unit-level planner tests bypass with synthesized
``AppliedObject``s.  Failures here indict whichever layer a synthetic
fixture would mask.

See ``plans/fix_batch-bugbash_todos_tdd_*.plan.md`` Phase 0.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.compiler import normalize_sql_whitespace
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
from snowflake.ml.feature_store.decl.manifest import FSTarget
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.feature_store.decl.types import AppliedState, PlanOptions, SpecBatch

# ---------------------------------------------------------------------------
# Doc constants — keep verbatim with declarative_feature_store/BATCH_FV_BUG_BASH.md §5
# ---------------------------------------------------------------------------

_DB = "BUGBASH_DB"
_SCH = "BUGBASH_SC"

_TABLE_FV_NAME = "MY_BATCH_FV_BATCH_DECL"
_SQL_FV_NAME = "MY_SQL_BATCH_FV_BATCH_DECL"
_FV_VERSION = "V1"

_TABLE_BFV_OFFLINE_DT = f"{_TABLE_FV_NAME}${_FV_VERSION}"
_SQL_BFV_OFFLINE_DT = f"{_SQL_FV_NAME}${_FV_VERSION}"

_TABLE_BFV_OFT = f"{_TABLE_FV_NAME}${_FV_VERSION}$ONLINE"
_SQL_BFV_OFT = f"{_SQL_FV_NAME}${_FV_VERSION}$ONLINE"

_RAW_TABLE = "RAW_EVENTS_BATCH_DECL"
_RAW_TABLE_V2 = "RAW_EVENTS_BATCH_DECL_V2"

# Sidecar SQL body the verify script writes to ``EVENTS_SQL_BATCH_DECL.sql``.
# Includes a trailing ``;`` because the verify-script's
# ``EVENTS_SQL_BATCH_DECL.sql`` ends with one (SQL convention), and that
# is exactly the round-trip the planner has to keep stable.  Both
# ``normalize_sql_whitespace`` and ``state._classify_dt_body`` strip
# trailing ``;`` so the local-compile output matches the deployed DT
# body stored by Snowflake.
_SQL_BODY_RAW = "SELECT USER_ID, EVENT_TS, METRIC_VAL\nFROM RAW_EVENTS_BATCH_DECL;\n"
_SQL_BODY_NORMALIZED = normalize_sql_whitespace(_SQL_BODY_RAW)


# ---------------------------------------------------------------------------
# Project authoring helpers — mirror scripts/verify_batch_fv_bug_bash.sh §5
# ---------------------------------------------------------------------------


def _entity_yaml() -> str:
    return textwrap.dedent(
        """\
        kind: Entity
        name: USER_BATCH_DECL
        join_keys:
          - name: USER_ID
            type: StringType
        """
    )


def _table_batch_source_yaml(*, table: str = _RAW_TABLE) -> str:
    return textwrap.dedent(
        f"""\
        kind: BatchSource
        name: EVENTS_BATCH_DECL
        table: {table}
        columns:
          - name: USER_ID
            type: StringType
          - name: EVENT_TS
            type: TimestampType
          - name: METRIC_VAL
            type: FloatType
        """
    )


def _sql_batch_source_yaml() -> str:
    return textwrap.dedent(
        """\
        kind: BatchSource
        name: EVENTS_SQL_BATCH_DECL
        query_file: EVENTS_SQL_BATCH_DECL.sql
        columns:
          - name: USER_ID
            type: StringType
          - name: EVENT_TS
            type: TimestampType
          - name: METRIC_VAL
            type: FloatType
        """
    )


def _table_fv_yaml(*, refresh_freq: str = "1 minute") -> str:
    return textwrap.dedent(
        f"""\
        kind: BatchFeatureView
        name: {_TABLE_FV_NAME}
        version: {_FV_VERSION}
        online: true
        target_lag: 1 minute
        entities:
          - USER_ID
        sources:
          - name: EVENTS_BATCH_DECL
            source_type: Batch
        refresh_freq: {refresh_freq}
        """
    )


def _sql_fv_yaml(*, refresh_freq: str = "1 minute") -> str:
    return textwrap.dedent(
        f"""\
        kind: BatchFeatureView
        name: {_SQL_FV_NAME}
        version: {_FV_VERSION}
        online: true
        target_lag: 1 minute
        entities:
          - USER_ID
        sources:
          - name: EVENTS_SQL_BATCH_DECL
            source_type: Batch
        refresh_freq: {refresh_freq}
        """
    )


def _offline_table_fv_yaml(*, refresh_freq: str = "1 minute") -> str:
    """Offline-only sibling of the table-backed BFV (§7b walkthrough)."""
    return textwrap.dedent(
        f"""\
        kind: BatchFeatureView
        name: MY_OFFLINE_BFV_BATCH_DECL
        version: {_FV_VERSION}
        online: false
        entities:
          - USER_ID
        sources:
          - name: EVENTS_BATCH_DECL
            source_type: Batch
        refresh_freq: {refresh_freq}
        """
    )


def _manifest_yaml() -> str:
    return textwrap.dedent(
        f"""\
        manifest_version: 1
        type: feature_store
        default_target: DEV
        targets:
          DEV:
            account_identifier: ORG-ACCOUNT
            database: {_DB}
            schema: {_SCH}
            role: TESTROLE
        """
    )


def _write_doc_project(
    project_root: Path,
    *,
    table_refresh_freq: str = "1 minute",
    sql_refresh_freq: str = "1 minute",
    table_source_table: str = _RAW_TABLE,
    include_offline_fv: bool = False,
    offline_fv_schedule: str = "1 minute",
) -> None:
    """Write the verify-script project layout into ``project_root``.

    Mirrors ``scripts/verify_batch_fv_bug_bash.sh`` step 5 verbatim (entity,
    both BatchSources, sidecar ``.sql``, both BatchFeatureViews) plus an
    optional offline-only BFV for §7b coverage.

    Args:
        project_root: Directory that will receive ``manifest.yml`` and
            ``sources/`` subtree.  Created if missing.
        table_refresh_freq: ``refresh_freq`` for the table-backed BFV.
        sql_refresh_freq: ``refresh_freq`` for the SQL-backed BFV.
        table_source_table: ``BatchSource.table`` for ``EVENTS_BATCH_DECL``
            (used for the §8 swap to ``RAW_EVENTS_BATCH_DECL_V2``).
        include_offline_fv: When True, also writes
            ``sources/feature_views/MY_OFFLINE_BFV_BATCH_DECL.yaml``.
        offline_fv_schedule: ``refresh_freq`` for the offline-only BFV.
    """
    sources = project_root / "sources"
    (sources / "entities").mkdir(parents=True, exist_ok=True)
    (sources / "datasources").mkdir(parents=True, exist_ok=True)
    (sources / "feature_views").mkdir(parents=True, exist_ok=True)

    (project_root / "manifest.yml").write_text(_manifest_yaml())

    (sources / "entities" / "USER_BATCH_DECL.yaml").write_text(_entity_yaml())

    (sources / "datasources" / "EVENTS_BATCH_DECL.yaml").write_text(_table_batch_source_yaml(table=table_source_table))

    (sources / "datasources" / "EVENTS_SQL_BATCH_DECL.yaml").write_text(_sql_batch_source_yaml())
    (sources / "datasources" / "EVENTS_SQL_BATCH_DECL.sql").write_text(_SQL_BODY_RAW)

    (sources / "feature_views" / f"{_TABLE_FV_NAME}.yaml").write_text(_table_fv_yaml(refresh_freq=table_refresh_freq))
    (sources / "feature_views" / f"{_SQL_FV_NAME}.yaml").write_text(_sql_fv_yaml(refresh_freq=sql_refresh_freq))

    if include_offline_fv:
        (sources / "feature_views" / "MY_OFFLINE_BFV_BATCH_DECL.yaml").write_text(
            _offline_table_fv_yaml(refresh_freq=offline_fv_schedule)
        )


def _bugbash_target() -> FSTarget:
    return FSTarget(
        name="DEV",
        account_identifier="ORG-ACCOUNT",
        database=_DB,
        schema=_SCH,
        role="TESTROLE",
    )


def _load_resolved_batch(project_root: Path) -> SpecBatch:
    """Load the project, resolve datasource columns, return SpecBatch."""
    target = _bugbash_target()
    batch = decl_api.load_project(project_root, target=target)
    decl_api.resolve_datasource_columns(batch)
    return batch


def _generate_plan(batch: SpecBatch, applied_state: AppliedState) -> Any:
    return generate_plan(batch, applied_state, PlanOptions(), database=_DB, schema=_SCH)


# ---------------------------------------------------------------------------
# Live applied-state shapes — synthesize what ``manager._fetch_*`` deliver.
# ---------------------------------------------------------------------------


def _table_bfv_dt_text(*, source_table: str = _RAW_TABLE) -> str:
    """``CREATE DYNAMIC TABLE … AS SELECT * FROM <source_table>`` body the
    imperative serializer produces for a table-backed BFV.

    Args:
        source_table: Unqualified source table name embedded in the FROM
            clause; matched by ``state._classify_dt_body`` to recover
            ``sources[0].table``.

    Returns:
        DDL string mirroring the live ``SHOW DYNAMIC TABLES.text`` column.
    """
    return (
        f"CREATE DYNAMIC TABLE {_DB}.{_SCH}.{_TABLE_BFV_OFFLINE_DT}\n"
        "TARGET_LAG = '1 minute'\n"
        "WAREHOUSE = TEST_WH\n"
        f"AS SELECT * FROM {_DB}.{_SCH}.{source_table}"
    )


def _sql_bfv_dt_text() -> str:
    """``CREATE DYNAMIC TABLE … AS <user query>`` body for the SQL BFV.

    The user-authored ``EVENTS_SQL_BATCH_DECL.sql`` is a non-flat SELECT
    (projected columns), so ``state._classify_dt_body`` classifies the
    body as ``query`` shape and emits a synthetic ``<FV>__SOURCE`` source.

    Returns:
        The exact DT text the live ``GET_DDL`` would emit for the SQL BFV.
    """
    return (
        f"CREATE DYNAMIC TABLE {_DB}.{_SCH}.{_SQL_BFV_OFFLINE_DT}\n"
        "TARGET_LAG = '1 minute'\n"
        "WAREHOUSE = TEST_WH\n"
        f"AS {_SQL_BODY_NORMALIZED}"
    )


def _online_bfv_specification_payload(
    *,
    fv_name: str,
    offline_dt: str,
    target_lag_sec: int = 60,
) -> dict[str, Any]:
    """Synthesize the ``DESCRIBE … TYPE = SPECIFICATION`` JSON for an online BFV.

    Mirrors the live snowml-core serializer output: ``spec.sources`` is
    empty (the source binding is encoded into the offline DT body) and
    ``offline_configs[0].table`` carries the offline DT name so
    ``state._inject_batch_fv_source_from_dt_text`` can recover the
    binding via ``dt_text_map``.

    Args:
        fv_name: Feature-view name (the ``BatchFeatureView`` metadata).
        offline_dt: Offline DT name (``<NAME>$<VERSION>``) keyed in
            ``dt_text_map``.
        target_lag_sec: Deployed cadence in seconds.

    Returns:
        Dict shaped like ``DESCRIBE … TYPE = SPECIFICATION``.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCH,
            "name": fv_name,
            "version": _FV_VERSION,
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "0.1.0",
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "BatchTable",
                "database": _DB,
                "schema": _SCH,
                "table": offline_dt,
                "columns": [],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
            "target_lag_sec": target_lag_sec,
        },
        "online_store_type": "postgres",
    }


def _online_oft_row(*, oft_name: str) -> dict[str, Any]:
    return {
        "name": oft_name,
        "database_name": _DB,
        "schema_name": _SCH,
        "created_on": "2024-01-01 00:00:00",
    }


def _offline_list_fv_row(
    *,
    fv_name: str,
    offline_dt: str,
    target_lag: str = "1 minute",
    source_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Phase-1 row shape for ``imperative_executor.fetch_feature_view_rows``.

    Phase B3: ``source_refs`` is the authoritative ``FvSourceRefsMetadata``
    payload — without it, ``state._inject_batch_fv_source_from_metadata``
    falls back to the legacy shim and emits a once-per-FV warning, leaving
    ``spec.sources`` empty (which then breaks downstream hash equality
    against the local-compile spec).

    Args:
        fv_name: FeatureView name (``MY_BATCH_FV`` / ``MY_SQL_BFV``).
        offline_dt: Physical offline Dynamic Table identifier the
            FV is backed by.
        target_lag: ``TARGET_LAG`` / ``REFRESH_FREQ`` cadence string
            (the row uses the same value for both since the test
            fixtures pre-date the L4 split).
        source_refs: Authored ``FV_SOURCE_REFS`` payload to thread
            through to the recovered applied state.  ``None`` /
            empty list intentionally exercises the legacy-shim
            fallback path.

    Returns:
        A dict matching the
        :func:`imperative_executor.fetch_feature_view_rows` row
        shape, ready to be passed to
        :func:`fetch_applied_state(feature_view_rows=...)`.
    """
    return {
        "name": fv_name,
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
        "physical_dt_name": offline_dt,
        "source_refs": source_refs or [],
    }


def _table_bfv_source_refs(*, table: str = _RAW_TABLE) -> list[dict[str, Any]]:
    """Authored ``FV_SOURCE_REFS`` payload for the table-backed BFV.

    Mirrors the shape :meth:`SourceRef.model_dump` produces for a
    ``BatchSource(table=...)`` declaration — exactly what
    :func:`compile_to_spec` writes into ``spec.sources[0]`` for the
    same FV.  Threaded into the list-FV row so the recovered applied
    state hashes identically to the local-compile spec on a clean
    round-trip (Phase B5 hash convergence).

    Args:
        table: Physical raw table identifier the
            ``BatchSource.table`` field should report.  Defaults to
            the production fixture's ``_RAW_TABLE``; tests that
            simulate the §8 table-swap repro pass an alternate
            identifier.

    Returns:
        A single-entry ``list[dict]`` matching the
        :class:`FvSourceRefsMetadata` payload shape.
    """
    return [
        {
            "name": "EVENTS_BATCH_DECL",
            "source_type": "Batch",
            "table": table,
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
                {"name": "METRIC_VAL", "type": "FloatType"},
            ],
        }
    ]


def _sql_bfv_source_refs() -> list[dict[str, Any]]:
    """Authored ``FV_SOURCE_REFS`` payload for the SQL-backed BFV.

    Carries the inlined, whitespace-normalised query body (mirrors
    :func:`compile_to_spec`'s output for a query-shape source) so the
    recovered ``spec.sources[0]`` matches the local-compile spec
    verbatim.

    Returns:
        A single-entry ``list[dict]`` carrying the query-shape
        ``BatchSource`` payload (``name``, ``source_type``,
        ``query``, ``columns``).
    """
    return [
        {
            "name": "EVENTS_SQL_BATCH_DECL",
            "source_type": "Batch",
            "query": _SQL_BODY_NORMALIZED,
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
                {"name": "METRIC_VAL", "type": "FloatType"},
            ],
        }
    ]


def _online_applied_state(
    *,
    table_bfv_target_lag_sec: int = 60,
    sql_bfv_target_lag_sec: int = 60,
    table_bfv_source: str = _RAW_TABLE,
) -> AppliedState:
    """AppliedState as if both online BFVs were already deployed.

    Builds the four streams ``manager.py:862-871`` threads into
    ``fetch_applied_state``: OFT rows, SPECIFICATION JSON map, DT-text
    map, and feature-view list rows.

    Args:
        table_bfv_target_lag_sec: Deployed cadence (seconds) for the
            table-backed BFV.
        sql_bfv_target_lag_sec: Deployed cadence (seconds) for the
            SQL-backed BFV.
        table_bfv_source: Source-table identifier currently in the
            offline DT's FROM clause for the table-backed BFV.

    Returns:
        Fully reconstructed ``AppliedState``.
    """
    raw_show_results = [
        _online_oft_row(oft_name=_TABLE_BFV_OFT),
        _online_oft_row(oft_name=_SQL_BFV_OFT),
    ]
    specification_map = {
        _TABLE_BFV_OFT: _online_bfv_specification_payload(
            fv_name=_TABLE_FV_NAME,
            offline_dt=_TABLE_BFV_OFFLINE_DT,
            target_lag_sec=table_bfv_target_lag_sec,
        ),
        _SQL_BFV_OFT: _online_bfv_specification_payload(
            fv_name=_SQL_FV_NAME,
            offline_dt=_SQL_BFV_OFFLINE_DT,
            target_lag_sec=sql_bfv_target_lag_sec,
        ),
    }
    dt_text_map = {
        _TABLE_BFV_OFFLINE_DT: _table_bfv_dt_text(source_table=table_bfv_source),
        _SQL_BFV_OFFLINE_DT: _sql_bfv_dt_text(),
    }
    # Phase B3: thread the authoritative ``FV_SOURCE_REFS`` metadata
    # via the matching ``feature_view_rows`` entry.  The OFT-backed
    # recovery path in ``fetch_applied_state`` reads ``source_refs``
    # from the row keyed on (name, version) and injects it into the
    # SPECIFICATION-shape ``spec_payload`` BEFORE hashing — closing
    # the round-trip-hash invariant.
    feature_view_rows = [
        _offline_list_fv_row(
            fv_name=_TABLE_FV_NAME,
            offline_dt=_TABLE_BFV_OFFLINE_DT,
            target_lag=f"{table_bfv_target_lag_sec} seconds",
            source_refs=_table_bfv_source_refs(table=table_bfv_source),
        ),
        _offline_list_fv_row(
            fv_name=_SQL_FV_NAME,
            offline_dt=_SQL_BFV_OFFLINE_DT,
            target_lag=f"{sql_bfv_target_lag_sec} seconds",
            source_refs=_sql_bfv_source_refs(),
        ),
    ]
    return fetch_applied_state(
        raw_show_results=raw_show_results,
        raw_table_results=[],
        specification_map=specification_map,
        entity_rows=[],
        dt_text_map=dt_text_map,
        feature_view_rows=feature_view_rows,
        default_database=_DB,
        default_schema=_SCH,
    )


def _ops_for(plan: Any, name: str) -> list[Any]:
    return [op for op in plan.ops if op.name == name]


# ===========================================================================
# §6 — first plan + apply (CREATE_FV expected on a clean schema)
# ===========================================================================


class TestStep6FirstCreate:
    """Mirrors ``scripts/verify_batch_fv_bug_bash.sh`` §6 plan-time gate.

    The verify script writes the project YAMLs, then asserts the latest
    ``out/plan/feature_plan_*.json`` carries CREATE_FV (or RECREATE_FV)
    ops for both ``MY_BATCH_FV_BATCH_DECL`` and
    ``MY_SQL_BATCH_FV_BATCH_DECL`` with merged ``sources[0].table`` /
    non-empty ``sources[0].query``.  These tests pin the same contract.
    """

    def test_create_fv_on_pristine_schema_table_bfv(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        plan = _generate_plan(batch, AppliedState(objects={}))

        ops = _ops_for(plan, _TABLE_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "CREATE_FV"
        assert ops[0].destructive is False
        payload = ops[0].payload
        sources = payload.get("sources") or []
        assert sources, payload
        assert sources[0].get("table") == _RAW_TABLE
        assert sources[0].get("source_type") == "Batch"

    def test_create_fv_on_pristine_schema_sql_bfv(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        plan = _generate_plan(batch, AppliedState(objects={}))

        ops = _ops_for(plan, _SQL_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "CREATE_FV"
        assert ops[0].destructive is False
        payload = ops[0].payload
        sources = payload.get("sources") or []
        assert sources, payload
        query = sources[0].get("query")
        assert isinstance(query, str) and query.strip(), payload
        assert "RAW_EVENTS_BATCH_DECL" in query.upper()
        assert sources[0].get("source_type") == "Batch"

    def test_no_change_on_re_apply_with_identical_schema(self, tmp_path: Path) -> None:
        """Both BFVs already deployed with hash-equal specs → NO_CHANGE.

        This is what the live verify script trips over today: when a
        prior run left the schema with the same FVs, ``snow feature
        plan`` correctly emits NO_CHANGE, but the script's grep wants
        ``CREATE_FV|RECREATE_FV``.  Phase 1's verify-script tightening
        will make the script drop pre-existing FVs at the start of §6
        so this scenario never arises in CI; the contract this test
        pins is that the *planner* is not the one emitting wrong ops.

        Args:
            tmp_path: pytest tmp_path fixture providing the project root.
        """
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state()

        plan = _generate_plan(batch, applied)

        for fv_name in (_TABLE_FV_NAME, _SQL_FV_NAME):
            ops = _ops_for(plan, fv_name)
            assert len(ops) == 1, (fv_name, [(o.kind.value, o.name) for o in plan.ops])
            assert ops[0].kind.value == "NO_CHANGE", (
                f"{fv_name}: identical re-apply must collapse to NO_CHANGE; "
                f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
            )
            assert ops[0].destructive is False

    def test_recreate_fv_when_prior_run_drifted_target_lag(self, tmp_path: Path) -> None:
        """Prior deploy used a different cadence than the current YAML.

        ``target_lag_sec`` is part of the full-spec hash, so a 30-second
        deploy vs a 60-second YAML must surface as an op (RECREATE or
        UPDATE).  The verify-script's grep accepts either, and the
        planner picks RECREATE for full-spec drift on online BFVs.

        Args:
            tmp_path: pytest tmp_path fixture providing the project root.
        """
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state(
            table_bfv_target_lag_sec=30,
            sql_bfv_target_lag_sec=30,
        )

        plan = _generate_plan(batch, applied)

        for fv_name in (_TABLE_FV_NAME, _SQL_FV_NAME):
            ops = _ops_for(plan, fv_name)
            assert len(ops) == 1, (fv_name, [(o.kind.value, o.name) for o in plan.ops])
            assert ops[0].kind.value in ("RECREATE_FV", "UPDATE_FV"), (
                f"{fv_name}: cadence drift must surface as RECREATE_FV or UPDATE_FV; "
                f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
            )


# ===========================================================================
# §7 — operational-only edit on the table-backed FV → UPDATE_FV
# ===========================================================================


class TestStep7UpdateFv:
    """Mirrors ``scripts/verify_batch_fv_bug_bash.sh`` §7 (table FV only).

    The verify script edits *only* ``refresh_freq`` on
    ``MY_BATCH_FV_BATCH_DECL`` (1 minute → 2 minutes) and asserts the
    plan has one ``UPDATE_FV`` for the table-backed FV; the SQL-backed
    FV must stay ``NO_CHANGE``.
    """

    def test_update_fv_after_schedule_only_edit_table_bfv(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path, table_refresh_freq="2 minutes")
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state()  # both deployed at 60s cadence

        plan = _generate_plan(batch, applied)

        ops = _ops_for(plan, _TABLE_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "UPDATE_FV", (
            f"schedule-only edit must produce UPDATE_FV; " f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
        )
        assert ops[0].destructive is False

    def test_sql_bfv_unchanged_during_step7_update(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path, table_refresh_freq="2 minutes")
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state()

        plan = _generate_plan(batch, applied)

        ops = _ops_for(plan, _SQL_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "NO_CHANGE", (
            f"SQL FV is untouched in §7; got {ops[0].kind.value} " f"(reason={ops[0].reason!r})"
        )


# ===========================================================================
# §7b — offline-only BFV ``plan → apply → plan`` idempotency
# ===========================================================================


class TestStep7bOfflineRoundTrip:
    """Mirrors ``declarative_feature_store/BATCH_FV_BUG_BASH.md §7b``.

    The doc adds an offline-only BFV (``online: false``) and walks
    NO_CHANGE / UPDATE_FV / RECREATE_FV.  The CLI's
    ``manager._fetch_feature_view_rows`` surfaces the offline BFV via
    the imperative-list path; the planner must converge to the right
    op kind for each transition.
    """

    def _offline_applied_state(
        self,
        *,
        target_lag: str = "1 minute",
        source_table: str = _RAW_TABLE,
    ) -> AppliedState:
        return fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={
                "MY_OFFLINE_BFV_BATCH_DECL$V1": (
                    f"CREATE DYNAMIC TABLE {_DB}.{_SCH}.MY_OFFLINE_BFV_BATCH_DECL$V1\n"
                    f"TARGET_LAG = '{target_lag}'\n"
                    "WAREHOUSE = TEST_WH\n"
                    f"AS SELECT * FROM {_DB}.{_SCH}.{source_table}"
                ),
            },
            feature_view_rows=[
                _offline_list_fv_row(
                    fv_name="MY_OFFLINE_BFV_BATCH_DECL",
                    offline_dt="MY_OFFLINE_BFV_BATCH_DECL$V1",
                    target_lag=target_lag,
                    source_refs=_table_bfv_source_refs(table=source_table),
                )
            ],
            default_database=_DB,
            default_schema=_SCH,
        )

    def test_offline_bfv_replan_no_change(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path, include_offline_fv=True)
        batch = _load_resolved_batch(tmp_path)
        # Online BFVs already deployed too (they share the project tree),
        # but the focus of §7b is the offline FV's behaviour.
        online = _online_applied_state()
        offline = self._offline_applied_state()
        merged = AppliedState(
            objects={**online.objects, **offline.objects},
        )

        plan = _generate_plan(batch, merged)

        ops = _ops_for(plan, "MY_OFFLINE_BFV_BATCH_DECL")
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "NO_CHANGE", (
            f"unchanged offline BFV must collapse to NO_CHANGE; " f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
        )

    def test_offline_bfv_update_fv_after_schedule_edit(self, tmp_path: Path) -> None:
        _write_doc_project(
            tmp_path,
            include_offline_fv=True,
            offline_fv_schedule="2 minutes",
        )
        batch = _load_resolved_batch(tmp_path)
        online = _online_applied_state()
        offline = self._offline_applied_state(target_lag="1 minute")
        merged = AppliedState(objects={**online.objects, **offline.objects})

        plan = _generate_plan(batch, merged)

        ops = _ops_for(plan, "MY_OFFLINE_BFV_BATCH_DECL")
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "UPDATE_FV", (
            f"schedule-only edit on offline BFV must produce UPDATE_FV; "
            f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
        )
        assert ops[0].destructive is False

    def test_offline_bfv_recreate_after_table_swap(self, tmp_path: Path) -> None:
        _write_doc_project(
            tmp_path,
            include_offline_fv=True,
            table_source_table=_RAW_TABLE_V2,
        )
        batch = _load_resolved_batch(tmp_path)
        # Live deployed offline BFV still points at the original table.
        online = _online_applied_state(table_bfv_source=_RAW_TABLE)
        offline = self._offline_applied_state(source_table=_RAW_TABLE)
        merged = AppliedState(objects={**online.objects, **offline.objects})

        plan = _generate_plan(batch, merged)

        ops = _ops_for(plan, "MY_OFFLINE_BFV_BATCH_DECL")
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "RECREATE_FV", (
            f"BatchSource.table swap must produce RECREATE_FV on the offline BFV; "
            f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
        )
        assert ops[0].destructive is True


# ===========================================================================
# §8 — structural ``BatchSource.table`` swap → RECREATE_FV (table FV only)
# ===========================================================================


class TestStep8RecreateFv:
    """Mirrors ``scripts/verify_batch_fv_bug_bash.sh`` §8.

    The verify script edits ``EVENTS_BATCH_DECL.yaml`` to swap
    ``table:`` from ``RAW_EVENTS_BATCH_DECL`` to
    ``RAW_EVENTS_BATCH_DECL_V2``, then asserts the plan emits
    ``RECREATE_FV`` for ``MY_BATCH_FV_BATCH_DECL`` and that plain
    apply refuses (must be ``--allow-recreate``-gated).  The SQL FV
    is untouched in §8.
    """

    def test_recreate_fv_after_datasource_table_swap_table_bfv(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path, table_source_table=_RAW_TABLE_V2)
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state(table_bfv_source=_RAW_TABLE)

        plan = _generate_plan(batch, applied)

        ops = _ops_for(plan, _TABLE_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "RECREATE_FV", (
            f"BatchSource.table swap must produce RECREATE_FV on the table BFV; "
            f"got {ops[0].kind.value} (reason={ops[0].reason!r})"
        )
        assert ops[0].destructive is True

    def test_sql_bfv_unchanged_during_step8_table_swap(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path, table_source_table=_RAW_TABLE_V2)
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state(table_bfv_source=_RAW_TABLE)

        plan = _generate_plan(batch, applied)

        ops = _ops_for(plan, _SQL_FV_NAME)
        assert len(ops) == 1, [(o.kind.value, o.name) for o in plan.ops]
        assert ops[0].kind.value == "NO_CHANGE", (
            f"SQL FV is untouched in §8; got {ops[0].kind.value} " f"(reason={ops[0].reason!r})"
        )

    def test_step8_destructive_flag_present(self, tmp_path: Path) -> None:
        """Apply-time gate: the destructive flag is what makes the CLI
        emit ``Status: refused`` on plain apply and ``Status: applied``
        on ``--allow-recreate``.  The bug-bash §8 cascade
        (``Status: applied`` then ``Status: no_plan``) is the symptom of
        this flag being absent because the planner emitted NO_CHANGE.

        Args:
            tmp_path: pytest tmp_path fixture providing the project root.
        """
        _write_doc_project(tmp_path, table_source_table=_RAW_TABLE_V2)
        batch = _load_resolved_batch(tmp_path)
        applied = _online_applied_state(table_bfv_source=_RAW_TABLE)

        plan = _generate_plan(batch, applied)

        destructive_table_ops = [op for op in plan.ops if op.name == _TABLE_FV_NAME and op.destructive]
        assert len(destructive_table_ops) == 1
        assert destructive_table_ops[0].kind.value == "RECREATE_FV"


# ===========================================================================
# Sequential walk: §6 → §7 → §8 modeled as a single test (live ordering)
# ===========================================================================


class TestSequentialBugbashWalk:
    """End-to-end replay: §6 CREATE → §7 UPDATE → §8 RECREATE.

    Captures the live verify-script flow as a single test sequence so a
    regression in any transition surfaces as a failure here even when
    the per-step suites above pass in isolation.
    """

    def _applied_state_after_create(self, project_root: Path) -> AppliedState:
        """Compute applied state from the local-compile output of §6.

        After a successful first apply, the deployed spec equals the
        compiled local spec (modulo lossy ``sources`` round-trip the
        DT-text recovery handles).  We synthesize the deployed
        ``DESCRIBE … TYPE = SPECIFICATION`` JSON by compiling the §6
        local spec and zeroing ``sources`` (the deployed
        round-trip behaviour).

        Args:
            project_root: project root used by §6 (parameter is reserved
                for future fidelity work that compiles directly off the
                authored YAML; today the synthetic state covers the full
                §6 contract).

        Returns:
            ``AppliedState`` containing both online BFVs at 60s cadence
            with the original ``RAW_EVENTS_BATCH_DECL`` source.
        """
        del project_root
        return _online_applied_state()  # both at 60s cadence, original tables

    def test_create_then_update_then_recreate(self, tmp_path: Path) -> None:
        # §6 — fresh schema, expect CREATE_FV for both BFVs.
        _write_doc_project(tmp_path)
        batch_v6 = _load_resolved_batch(tmp_path)
        plan_v6 = _generate_plan(batch_v6, AppliedState(objects={}))
        assert {op.kind.value for op in _ops_for(plan_v6, _TABLE_FV_NAME)} == {"CREATE_FV"}
        assert {op.kind.value for op in _ops_for(plan_v6, _SQL_FV_NAME)} == {"CREATE_FV"}

        # §7 — schedule edit on table FV; SQL FV stays NO_CHANGE.
        _write_doc_project(tmp_path, table_refresh_freq="2 minutes")
        batch_v7 = _load_resolved_batch(tmp_path)
        plan_v7 = _generate_plan(batch_v7, self._applied_state_after_create(tmp_path))
        assert {op.kind.value for op in _ops_for(plan_v7, _TABLE_FV_NAME)} == {"UPDATE_FV"}
        assert {op.kind.value for op in _ops_for(plan_v7, _SQL_FV_NAME)} == {"NO_CHANGE"}

        # §8 — table swap on table FV; SQL FV stays NO_CHANGE.
        _write_doc_project(
            tmp_path,
            table_refresh_freq="2 minutes",
            table_source_table=_RAW_TABLE_V2,
        )
        batch_v8 = _load_resolved_batch(tmp_path)
        # Applied side now reflects post-§7 deploy (cadence still 60s — the
        # update bumped DT TARGET_LAG; for the table-swap diff we just
        # need the applied table to differ from the local YAML).
        applied_post7 = _online_applied_state(table_bfv_target_lag_sec=120)
        plan_v8 = _generate_plan(batch_v8, applied_post7)
        ops_table = _ops_for(plan_v8, _TABLE_FV_NAME)
        assert ops_table and ops_table[0].kind.value == "RECREATE_FV"
        assert ops_table[0].destructive is True
        assert {op.kind.value for op in _ops_for(plan_v8, _SQL_FV_NAME)} == {"NO_CHANGE"}


# ===========================================================================
# Round-trip hash invariant — primary indictment surface
# ===========================================================================


class TestAppliedStateHashConvergence:
    """The contract that backs every test above: the reconstructed
    applied ``content_hash`` must equal the local-compile hash on a
    clean round trip.  When this invariant breaks, every NO_CHANGE
    test in this module degrades to a phantom op — which is exactly
    the live verify-script failure mode.
    """

    def test_table_bfv_hash_round_trip(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        fv = next(s for s in batch.specs if getattr(s, "name", "") == _TABLE_FV_NAME)
        local_dict = fv.model_dump(exclude_none=True)
        local_compiled = compile_to_spec(local_dict, _DB, _SCH)
        local_hash = _full_spec_hash(local_compiled)

        applied = _online_applied_state()
        key = f"BatchFeatureView:{_DB}.{_SCH}:{_TABLE_FV_NAME}"
        assert key in applied.objects, sorted(applied.objects)
        applied_obj = applied.objects[key]
        assert applied_obj.content_hash == local_hash, (
            f"Round-trip hash mismatch for table BFV: " f"local={local_hash} applied={applied_obj.content_hash}"
        )

    def test_sql_bfv_hash_round_trip(self, tmp_path: Path) -> None:
        _write_doc_project(tmp_path)
        batch = _load_resolved_batch(tmp_path)
        fv = next(s for s in batch.specs if getattr(s, "name", "") == _SQL_FV_NAME)
        local_dict = fv.model_dump(exclude_none=True)
        local_compiled = compile_to_spec(local_dict, _DB, _SCH)
        local_hash = _full_spec_hash(local_compiled)

        applied = _online_applied_state()
        key = f"BatchFeatureView:{_DB}.{_SCH}:{_SQL_FV_NAME}"
        assert key in applied.objects, sorted(applied.objects)
        applied_obj = applied.objects[key]
        assert applied_obj.content_hash == local_hash, (
            f"Round-trip hash mismatch for SQL BFV: " f"local={local_hash} applied={applied_obj.content_hash}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
