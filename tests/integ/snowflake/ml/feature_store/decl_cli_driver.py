"""In-process equivalent of ``snow feature plan`` / ``apply`` for integ tests.

The real orchestration lives in the ``snowflake-cli`` ``FeatureStoreManager``
(``_compute_plan`` / ``_fetch_applied_state_bundle`` / ``_apply_from_plan_file``),
which is NOT importable from the snowml test environment. This module replicates
that loop over the public ``decl.api`` facade against a live Snowpark session so
the online-service bundle can exercise the genuine declarative plan/apply path
(``load_project`` -> ``fetch_applied_state`` -> ``validate_specs`` ->
``generate_plan`` -> ``execute_plan``) without the CLI plugin.

The planner is always driven with ``full_directory_mode=False`` (``no_delete``)
so objects other bundle classes created in the shared schema are never flagged
for deletion -- see the plan's "Interaction with imperative objects" note.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.manifest import FSTarget
from snowflake.ml.feature_store.decl.types import Plan, PlanOptions

logger = logging.getLogger(__name__)


class DeclDriverError(RuntimeError):
    """Raised when plan validation or planner-side checks surface an ERROR."""


def _rows(session: Any, sql: str) -> list[dict[str, Any]]:
    """Run ``sql`` and project each Snowpark ``Row`` to a plain dict.

    Mirrors the CLI manager's ``_rows_to_dicts`` (which uses a ``DictCursor``);
    ``decl.state`` tolerates either identifier case in the resulting keys.

    Args:
        session: Live Snowpark session used to run the query.
        sql: SQL text to execute.

    Returns:
        One plain ``dict`` per result row.
    """
    return [row.as_dict() for row in session.sql(sql).collect()]


def fetch_applied_state(
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
    *,
    datasources_by_table: dict[str, Any] | None = None,
) -> Any:
    """Reconstruct the applied-state snapshot for ``database.schema``.

    Faithful port of ``FeatureStoreManager._fetch_applied_state_bundle``: it
    runs the state SQL (``SHOW ONLINE FEATURE TABLES`` / ``SHOW TABLES`` /
    ``SHOW DYNAMIC TABLES`` + per-OFT ``DESCRIBE ... TYPE = SPECIFICATION``)
    plus the imperative inventory facades, then folds everything through
    ``decl_api.fetch_applied_state``.

    Args:
        session: A live ``snowflake.snowpark.Session``.
        database: Target database name.
        schema: Target schema name.
        warehouse: Warehouse forwarded to the imperative inventory facades.
        datasources_by_table: Optional physical-table -> logical-source-name
            lookup (``decl_api.build_datasources_by_table``) threaded through
            for BatchFV source-binding recovery.

    Returns:
        The ``AppliedState`` snapshot.
    """
    sqls = decl_api.state_queries(database, schema)

    raw_show = _rows(session, sqls["show_ofts"])
    raw_tables = _rows(session, sqls["show_tables"])

    dt_text_map: dict[str, str] = {}
    show_dynamic_tables = sqls.get("show_dynamic_tables")
    if show_dynamic_tables:
        for row in _rows(session, show_dynamic_tables):
            name = row.get("name") or row.get("NAME") or ""
            text = row.get("text") or row.get("TEXT") or ""
            if isinstance(name, str) and isinstance(text, str) and name and text:
                dt_text_map[name] = text

    entity_rows = decl_api.fetch_entity_rows(session, database, schema, warehouse)
    feature_group_rows = decl_api.fetch_feature_group_rows(session, database, schema, warehouse)
    feature_view_rows = decl_api.fetch_feature_view_rows(session, database, schema, warehouse)
    stream_source_rows = decl_api.fetch_stream_source_rows(session, database, schema, warehouse)

    specification_map: dict[str, dict[str, Any]] = {}
    for row in raw_show:
        name = row.get("name") or row.get("NAME") or ""
        if not name:
            continue
        desc_rows = _rows(session, decl_api.describe_specification_query(database, schema, name))
        spec = decl_api.parse_specification_rows(desc_rows)
        if spec is not None:
            specification_map[name] = spec

    return decl_api.fetch_applied_state(
        raw_show,
        raw_tables,
        specification_map=specification_map,
        dt_text_map=dt_text_map,
        entity_rows=entity_rows,
        feature_view_rows=feature_view_rows,
        feature_group_rows=feature_group_rows,
        stream_source_rows=stream_source_rows,
        datasources_by_table=datasources_by_table,
        default_database=database,
        default_schema=schema,
    )


def compute_plan(
    session: Any,
    project_root: Path | str,
    target: FSTarget,
    warehouse: str,
    *,
    runtime_vars: dict[str, Any] | None = None,
    no_delete: bool = True,
) -> Plan:
    """Load a project, diff it against live applied state, and return a Plan.

    Mirrors ``FeatureStoreManager._compute_plan``: ``load_project`` ->
    ``build_datasources_by_table`` -> ``fetch_applied_state`` ->
    ``resolve_datasource_columns`` -> ``validate_specs`` -> ``generate_plan``.

    Args:
        session: A live ``snowflake.snowpark.Session``.
        project_root: Directory containing ``manifest.yml``.
        target: Resolved manifest target.
        warehouse: Warehouse for the imperative inventory facades.
        runtime_vars: Optional ``--variable`` style templating overrides.
        no_delete: When ``True`` (default) the planner runs with
            ``full_directory_mode=False`` so orphan objects are never dropped.

    Returns:
        The generated ``Plan``.

    Raises:
        DeclDriverError: If ``validate_specs`` or the planner surface any ERROR.
    """
    batch = decl_api.load_project(Path(project_root), target=target, runtime_vars=runtime_vars or None)
    datasources_by_table = decl_api.build_datasources_by_table(batch.specs)

    applied_state = fetch_applied_state(
        session,
        target.database,
        target.schema,
        warehouse,
        datasources_by_table=datasources_by_table,
    )

    decl_api.resolve_datasource_columns(batch)

    validation = decl_api.validate_specs(
        batch,
        applied_state,
        target_database=target.database,
        target_schema=target.schema,
    )
    errors = [r for r in validation if r.severity == "ERROR"]
    if errors:
        raise DeclDriverError("validate_specs ERROR: " + "; ".join(r.message for r in errors))

    options = PlanOptions(allow_recreate=False, full_directory_mode=not no_delete)
    plan = decl_api.generate_plan(
        batch,
        applied_state,
        options,
        database=target.database,
        schema=target.schema,
    )

    plan_errors = getattr(plan, "errors", None) or []
    if plan_errors:
        raise DeclDriverError("planner ERROR: " + "; ".join(str(e) for e in plan_errors))
    return plan


def apply_plan(session: Any, plan: Plan, target: FSTarget, warehouse: str) -> Any:
    """Execute ``plan`` through ``decl_api.execute_plan`` against the session.

    Args:
        session: A live ``snowflake.snowpark.Session``.
        plan: Plan returned by :func:`compute_plan`.
        target: Resolved manifest target (supplies database / schema).
        warehouse: Warehouse used to materialise Dynamic Tables / OFTs.

    Returns:
        The ``ApplyResult`` from ``execute_plan``.
    """
    options = PlanOptions()
    return decl_api.execute_plan(
        plan,
        session,
        target.database,
        target.schema,
        warehouse,
        options,
    )


def op_summary(plan: Plan) -> list[tuple[str, str, bool]]:
    """Flatten a Plan into ``(op_kind, object_name, destructive)`` triples."""
    summary: list[tuple[str, str, bool]] = []
    for op in plan.ops:
        kind = op.kind.value if hasattr(op.kind, "value") else str(op.kind)
        summary.append((kind, op.name, bool(getattr(op, "destructive", False))))
    return summary


def op_details(plan: Plan) -> list[tuple[str, str, bool, str]]:
    """Flatten a Plan into ``(op_kind, object_name, destructive, reason)`` tuples.

    The companion to :func:`op_summary` for diagnostics: it preserves the
    planner's ``reason`` string (which :func:`op_summary` drops) so a surprising
    op kind -- e.g. a destructive ``RECREATE_FV`` on a clean re-plan -- surfaces
    the exact spec section the planner thinks changed.

    Args:
        plan: Plan returned by :func:`compute_plan`.

    Returns:
        One ``(kind, name, destructive, reason)`` tuple per op.
    """
    details: list[tuple[str, str, bool, str]] = []
    for op in plan.ops:
        kind = op.kind.value if hasattr(op.kind, "value") else str(op.kind)
        reason = str(getattr(op, "reason", "") or "")
        details.append((kind, op.name, bool(getattr(op, "destructive", False)), reason))
    return details


def format_plan(plan: Plan) -> str:
    """Render every op of a Plan as a human-readable multi-line string.

    Intended for INFO logging at each plan step so a CI failure log shows the
    full plan (kind / name / destructive / reason) rather than only the op kind
    the test asserted on.

    Args:
        plan: Plan returned by :func:`compute_plan`.

    Returns:
        A multi-line string, one op per line, plus any planner warnings.
    """
    lines = [f"plan: {len(plan.ops)} op(s)"]
    for kind, name, destructive, reason in op_details(plan):
        destr = " DESTRUCTIVE" if destructive else ""
        lines.append(f"  - {kind}{destr} {name}: {reason}")
    warnings = getattr(plan, "warnings", None) or []
    for warning in warnings:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def wait_fv_specification_recoverable(
    session: Any,
    database: str,
    schema: str,
    fv_name: str,
    version: str,
    *,
    timeout_s: float = 180.0,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Poll until a feature view's OFT is visible with a non-empty recovered spec.

    Closes the OFT-visibility race (hypothesis 2): immediately after ``apply``
    the new Online Feature Table may not yet appear in
    ``SHOW ONLINE FEATURE TABLES``.  When it is missing, applied-state recovery
    falls back to ``state._build_offline_fv_object``, which reconstructs an
    empty-``features`` spec -> a guaranteed full-spec hash mismatch ->
    ``RECREATE_FV`` on the next re-plan.  Gating the re-plan on the OFT being
    both listed and returning a parseable ``DESCRIBE ... TYPE = SPECIFICATION``
    payload with a non-empty ``spec.features`` keeps the healthy
    SPECIFICATION recovery path authoritative.

    Also absorbs most of the Postgres async-provisioning lag (hypothesis 1):
    the DESCRIBE payload is not parseable / is incomplete until the backing
    tables settle.

    Args:
        session: A live ``snowflake.snowpark.Session``.
        database: Target database name.
        schema: Target schema name.
        fv_name: Feature-view name (resolved, suffix-expanded).
        version: Feature-view version (e.g. ``"V1"``).
        timeout_s: Max seconds to poll before raising.
        poll_s: Sleep between polls.

    Returns:
        The recovered specification dict (parsed DESCRIBE payload).

    Raises:
        DeclDriverError: When the OFT is not recoverable with a non-empty
            ``spec.features`` within ``timeout_s`` (message includes the last
            observed ``SHOW`` names and parse state for triage).
    """
    deadline = time.time() + timeout_s
    last_oft_names: list[str] = []
    last_error: str | None = None
    while time.time() < deadline:
        raw_show = _rows(session, decl_api.list_query(database, schema))
        last_oft_names = [str(r.get("name") or r.get("NAME") or "") for r in raw_show]
        oft_name, resolve_err = decl_api.resolve_oft_name(raw_show, fv_name, version)
        if oft_name is None:
            last_error = resolve_err or "OFT not yet listed in SHOW ONLINE FEATURE TABLES"
        else:
            desc_rows = _rows(session, decl_api.describe_specification_query(database, schema, oft_name))
            spec = decl_api.parse_specification_rows(desc_rows)
            if spec is None:
                last_error = f"DESCRIBE SPECIFICATION for {oft_name} not yet parseable"
            else:
                inner = spec.get("spec") if isinstance(spec.get("spec"), dict) else {}
                features = inner.get("features") if isinstance(inner, dict) else None
                if features:
                    logger.info(
                        "wait_fv_specification_recoverable: %s/%s recoverable (oft=%s, %d feature(s)).",
                        fv_name,
                        version,
                        oft_name,
                        len(features),
                    )
                    return spec
                last_error = f"{oft_name} recovered with empty spec.features"
        time.sleep(poll_s)
    raise DeclDriverError(
        f"Feature view {fv_name}/{version} not recoverable from its OFT within {timeout_s:.0f}s "
        f"(last_error={last_error!r}; SHOW ONLINE FEATURE TABLES names={last_oft_names!r})."
    )


def diagnose_fv_drift(
    session: Any,
    project_root: Path | str,
    target: FSTarget,
    warehouse: str,
    fv_name: str,
    *,
    runtime_vars: dict[str, Any] | None = None,
) -> str:
    """Build a human-readable local-vs-applied drift report for one feature view.

    The on-failure inspector for a surprising ``RECREATE_FV``: it recompiles the
    local spec and recovers the applied spec exactly as :func:`compute_plan`
    does, then reports the drifted spec sections (via the production
    ``invariants.full_spec_changed_properties``) plus the two normalised
    payloads and the raw live state (``SHOW ONLINE FEATURE TABLES``, per-OFT
    ``DESCRIBE ... TYPE = SPECIFICATION``, ``SHOW DYNAMIC TABLES``,
    ``list_feature_views`` row).

    Every step is guarded so the diagnostic itself never raises -- it is meant
    to enrich an existing assertion failure, not replace it.

    Args:
        session: A live ``snowflake.snowpark.Session``.
        project_root: Directory containing ``manifest.yml``.
        target: Resolved manifest target.
        warehouse: Warehouse for the imperative inventory facades.
        fv_name: Feature-view name (resolved, suffix-expanded).
        runtime_vars: Optional ``--variable`` style templating overrides.

    Returns:
        A multi-line diagnostic string (never raises).
    """
    lines: list[str] = [f"=== drift diagnostic for {fv_name} ==="]
    try:
        from snowflake.ml.feature_store.decl.invariants import (  # noqa: PLC0415
            _canonical_spec_for_hash,
            compute_local_spec_and_hash,
            full_spec_changed_properties,
            model_to_dict,
        )
        from snowflake.ml.feature_store.decl.spec_compiler import (  # noqa: PLC0415
            build_entity_join_key_map,
        )

        database = target.database
        schema = target.schema

        batch = decl_api.load_project(Path(project_root), target=target, runtime_vars=runtime_vars or None)
        datasources_by_table = decl_api.build_datasources_by_table(batch.specs)
        decl_api.resolve_datasource_columns(batch)
        applied_state = fetch_applied_state(
            session, database, schema, warehouse, datasources_by_table=datasources_by_table
        )

        spec_dicts = [model_to_dict(s) for s in batch.specs]
        entity_join_keys = build_entity_join_key_map(spec_dicts, applied_state)
        local_data = next(
            (d for d in spec_dicts if str(d.get("name", "")).upper() == fv_name.upper()),
            None,
        )
        applied = next(
            (o for o in applied_state.objects.values() if str(getattr(o, "name", "")).upper() == fv_name.upper()),
            None,
        )

        if local_data is None:
            lines.append("local: no matching spec found in project.")
        if applied is None:
            lines.append("applied: no matching object found in applied state.")

        if local_data is not None and applied is not None and isinstance(getattr(applied, "spec_payload", None), dict):
            local_compiled, local_hash = compute_local_spec_and_hash(
                local_data, database, schema, entity_join_keys=entity_join_keys
            )
            applied_payload = applied.spec_payload
            applied_hash = str(getattr(applied, "content_hash", ""))
            lines.append(f"local_hash={local_hash}")
            lines.append(f"applied_hash={applied_hash}")
            lines.append(f"from_specification={getattr(applied, 'from_specification', None)}")

            local_norm = _canonical_spec_for_hash(local_compiled, authored=True)
            applied_norm = _canonical_spec_for_hash(applied_payload)
            changed = full_spec_changed_properties(local_norm, applied_norm)
            lines.append(f"changed_sections={changed}")
            lines.append("--- normalized local spec ---")
            lines.append(json.dumps(local_norm, indent=2, sort_keys=True, default=str))
            lines.append("--- normalized applied spec ---")
            lines.append(json.dumps(applied_norm, indent=2, sort_keys=True, default=str))
    except Exception as exc:  # noqa: BLE001 — diagnostics must never mask the real failure
        lines.append(f"(diagnostic spec-diff step failed: {type(exc).__name__}: {exc})")

    # Raw live-state dumps (each guarded independently).
    for label, sql in (
        ("SHOW ONLINE FEATURE TABLES", decl_api.list_query(target.database, target.schema)),
        ("SHOW DYNAMIC TABLES", decl_api.dynamic_tables_query(target.database, target.schema)),
    ):
        try:
            rows = _rows(session, sql)
            names = [str(r.get("name") or r.get("NAME") or "") for r in rows]
            lines.append(f"--- {label} ({len(rows)} row(s)) ---")
            lines.append(json.dumps(names, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001 — best-effort dump
            lines.append(f"(failed to dump {label}: {type(exc).__name__}: {exc})")

    try:
        raw_show = _rows(session, decl_api.list_query(target.database, target.schema))
        for row in raw_show:
            oft_name = str(row.get("name") or row.get("NAME") or "")
            if not oft_name or fv_name.upper() not in oft_name.upper():
                continue
            desc_rows = _rows(session, decl_api.describe_specification_query(target.database, target.schema, oft_name))
            spec = decl_api.parse_specification_rows(desc_rows)
            lines.append(f"--- DESCRIBE SPECIFICATION {oft_name} ---")
            lines.append(json.dumps(spec, indent=2, sort_keys=True, default=str))
    except Exception as exc:  # noqa: BLE001 — best-effort dump
        lines.append(f"(failed to dump DESCRIBE SPECIFICATION: {type(exc).__name__}: {exc})")

    lines.append("=== end drift diagnostic ===")
    return "\n".join(lines)
