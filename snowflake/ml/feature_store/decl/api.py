"""Top-level facade functions for the declarative feature store library.

This module is the **only** entry point for the CLI plugin. It must never
be bypassed in favour of importing internal modules (``invariants``,
``planner``, ``state``, etc.) directly. This keeps the substitution surface
small when individual responsibilities are migrated to Global Services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from snowflake.ml.feature_store.decl.errors import FeatureStoreNotInitializedError
from snowflake.ml.feature_store.decl.imperative_executor import (
    assert_feature_store_initialized,
)
from snowflake.ml.feature_store.decl.manifest import (
    FSManifest,
    FSTarget,
    FSTemplating,
    InvalidManifestError,
    ManifestConfigurationError,
    ManifestNotFoundError,
    TargetContext,
)
from snowflake.ml.feature_store.decl.project_paths import FSProjectPaths
from snowflake.ml.feature_store.decl.types import (
    AppliedState,
    Plan,
    PlanFile,
    PlanOptions,
    SpecBatch,
    ValidationResult,
)
from snowflake.ml.feature_store.spec.enums import ENTITY_TAG_PREFIX

__all__ = [
    "FSManifest",
    "FSProjectPaths",
    "FSTarget",
    "FSTemplating",
    "FeatureStoreNotInitializedError",
    "InvalidManifestError",
    "ManifestConfigurationError",
    "ManifestNotFoundError",
    "TargetContext",
    "assert_feature_store_initialized",
    "build_datasources_by_table",
    "discover_project",
    "load_manifest",
    "load_project",
    "resolve_datasource_columns",
    "resolve_target",
]

# ---------------------------------------------------------------------------
# Service / HTTP facades
# ---------------------------------------------------------------------------


def parse_service_status(raw_json: str) -> dict[str, Any]:
    """Parse the JSON string from SYSTEM$GET_FEATURE_STORE_RUNTIME_STATUS.

    Delegates to :func:`decl.service.parse_service_status`.

    Args:
        raw_json: JSON string returned by the system function.

    Returns:
        Parsed status dict.
    """
    from snowflake.ml.feature_store.decl.service import parse_service_status as _parse

    return _parse(raw_json)


def get_service_endpoint(status: dict[str, Any], name: str) -> Optional[str]:
    """Extract an endpoint URL by name from a parsed status dict.

    Args:
        status: Dict returned by :func:`parse_service_status`.
        name: Endpoint name (e.g. ``"ingest"`` or ``"query"``).

    Returns:
        URL string, or ``None`` if the endpoint is not found.
    """
    from snowflake.ml.feature_store.decl.service import get_service_endpoint as _get

    return _get(status, name)


def format_status_display(
    status: dict[str, Any],
    user: str = "",
    database: str = "",
    schema: str = "",
    *,
    verbose: bool = False,
) -> str:
    """Format the parsed online-service status into a rich multi-line display.

    Args:
        status: Dict returned by :func:`parse_service_status`.
        user: Current Snowflake user.
        database: Current database.
        schema: Current schema.
        verbose: When ``True``, render every per-component detail row
            and the Network Rules block.  When ``False`` (the default),
            render only the heading summaries plus endpoints.  The
            Postgres password is never collected or displayed.

    Returns:
        Formatted multi-line string.
    """
    from snowflake.ml.feature_store.decl.service import format_status_display as _fmt

    return _fmt(status, user, database, schema, verbose=verbose)


def format_describe_display(
    fv_name: str,
    version: str,
    database: str,
    schema: str,
    oft_name: str,
    entities: list[str],
    describe_rows: list[dict[str, Any]],
    show_row: Optional[dict[str, Any]] = None,
    spec: Optional[dict[str, Any]] = None,
    examples: Optional[list[str]] = None,
) -> str:
    """Format a rich describe display for a feature view."""
    from snowflake.ml.feature_store.decl.service import format_describe_display as _fmt

    return _fmt(
        fv_name,
        version,
        database,
        schema,
        oft_name,
        entities,
        describe_rows,
        show_row,
        spec,
        examples,
    )


def service_sql(
    database: str,
    schema: str,
    producer_role: str = "ACCOUNTADMIN",
    consumer_role: str = "PUBLIC",
) -> dict[str, str]:
    """Return SQL strings for service management, keyed by operation.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.
        producer_role: Snowflake role for producing features.
        consumer_role: Snowflake role for consuming features.

    Returns:
        Dict with keys: get_status, create, drop, show_ofts,
        drop_oft_template.
    """
    from snowflake.ml.feature_store.decl.service import service_sql as _sql

    return _sql(database, schema, producer_role, consumer_role)


def build_describe_examples(
    fv_name: str,
    version: str,
    source_name: str,
    describe_rows: list[dict[str, Any]],
    ingest_url: Optional[str],
    query_url: Optional[str],
    spec: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Build example curl commands for ingest and query.

    When a *spec* dict (parsed YAML) is provided, examples use the UDF
    output columns for ingest and the entity columns for query.

    Args:
        fv_name: Feature view name (user-facing, lowercase).
        version: Feature view version (e.g. ``"v1"``).
        source_name: Source name for ingest (e.g. ``"user_events"``).
        describe_rows: Rows from DESCRIBE ONLINE FEATURE TABLE (fallback).
        ingest_url: Base ingest endpoint URL, or ``None``.
        query_url: Base query endpoint URL, or ``None``.
        spec: Optional parsed YAML spec dict for the feature view.

    Returns:
        List of example strings (may be empty if no endpoints available).
    """
    from snowflake.ml.feature_store.decl.service import (
        build_describe_examples as _build,
    )

    return _build(fv_name, version, source_name, describe_rows, ingest_url, query_url, spec)


def load_specs(files: list[str], config: Optional[dict[str, Any]] = None) -> SpecBatch:
    """Load and parse spec files into a ``SpecBatch``.

    Accepts ``.py``, ``.yaml``, ``.yml``, and ``.json`` files. Applies Jinja2
    templating if ``config`` is provided and the file contains template syntax.
    Glob patterns and ``./...`` recursive expansion are resolved automatically.

    Args:
        files: Paths to spec files. Glob patterns and ``./...`` recursive
            expansion are resolved before loading.
        config: Jinja2 template variables. May be ``None`` if no templating
            is needed.

    Returns:
        SpecBatch containing all loaded and compiled spec objects.
    """
    from snowflake.ml.feature_store.decl import loader

    return loader.load_specs(files, config=config)


def validate_specs(
    batch: SpecBatch,
    applied_state: AppliedState,
    target_database: str = "",
    target_schema: str = "",
) -> list[ValidationResult]:
    """Validate a spec batch against invariant rules and applied state.

    Runs all invariant checks defined in ``DevExAndSchemaAndCICD.md``,
    including version management, column evolution, dependency resolution,
    and state synchronization.

    Args:
        batch: The parsed spec batch to validate.
        applied_state: The current applied state snapshot from Snowflake.
        target_database: Connection target database; forwarded to the
            idempotency check so it can compile local specs against the
            same database used by the planner / state-fetcher.
        target_schema: Connection target schema; forwarded to the
            idempotency check (same purpose as ``target_database``).

    Returns:
        Combined list of ``ValidationResult`` objects (ERRORRs and WARNINGGs).
    """
    from snowflake.ml.feature_store.decl.invariants import (
        validate_specs as _validate_specs,
    )

    return _validate_specs(
        batch,
        applied_state,
        target_database=target_database,
        target_schema=target_schema,
    )


def generate_plan(
    batch: SpecBatch,
    applied_state: AppliedState,
    options: PlanOptions,
    database: str = "",
    schema: str = "",
) -> Plan:
    """Generate a dependency-ordered execution plan.

    Diffs the incoming ``SpecBatch`` against ``AppliedState`` to classify
    each spec as CREATE, UPDATE, RECREATE, or NO_CHANGE. Returns a
    topologically-sorted ``Plan`` with one ``PlanOp`` per operation.

    For FeatureView kinds whose ``AppliedObject`` carries a full spec
    (``from_specification=True``) the diff is computed against the
    compiled full spec.  This catches changes the structural fingerprint
    misses — UDF code, source-column schema, aggregation windows, and
    target lag.  ``database`` and ``schema`` parameterise the local
    compile step; when empty the planner falls back to each spec's own
    ``database``/``schema`` fields.

    Args:
        batch: The validated spec batch.
        applied_state: The current applied state snapshot from Snowflake.
        options: Flags that control plan generation (overwrite, allow_recreate, etc.).
        database: Snowflake database name for compiling local FV specs.
        schema: Snowflake schema name for compiling local FV specs.

    Returns:
        Topologically-sorted execution plan.
    """
    from snowflake.ml.feature_store.decl.planner import generate_plan as _generate_plan

    return _generate_plan(batch, applied_state, options, database=database, schema=schema)


def export_specs(
    show_rows: list[dict[str, Any]],
    describe_rows_by_oft: dict[str, list[dict[str, Any]]],
    output_dir: str,
    database: str,
    schema: str,
    *,
    specification_map: Optional[dict[str, dict[str, Any]]] = None,
    entity_rows: Optional[list[dict[str, Any]]] = None,
    feature_group_rows: Optional[list[dict[str, Any]]] = None,
    applied_state: Optional[Any] = None,
    layout: str = "db_schema",
) -> dict[str, Any]:
    """Reconstruct YAML specs from raw SHOW/DESCRIBE results and write to disk.

    Args:
        show_rows: Rows from ``SHOW ONLINE FEATURE TABLES`` (list of dicts).
        describe_rows_by_oft: Map of OFT name → ``DESCRIBE`` rows (list of
            dicts).
        output_dir: Base output directory.
        database: Connection database name.
        schema: Connection schema name.
        specification_map: Optional map of OFT name → parsed spec JSON
            returned by ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE =
            SPECIFICATION``.  When an entry is present for an OFT, the
            corresponding YAML is written in full-fidelity mode (UDF source,
            source columns, feature aggregations preserved).  Missing
            entries fall back to the legacy partial-export path.
        entity_rows: Optional rows from
            ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'`` (or the
            equivalent imperative ``FeatureStore.list_entities()``
            output, see :func:`fetch_entity_rows`).  When non-empty,
            the exporter writes a YAML stub for every tag — including
            orphan tags not referenced by any FV — and enforces the
            ``FV.ordered_entity_column_names`` ⊆ ``entity_rows.names``
            invariant.  ``None`` and ``[]`` are treated identically:
            no entity YAMLs are written and the subset check is
            skipped (callers wanting authoritative emission must
            forward the :func:`fetch_entity_rows` output).
        feature_group_rows: Optional rows produced by
            :func:`fetch_feature_group_rows` (the imperative
            ``FeatureStore.list_feature_groups()`` output, normalised
            into the executor row shape).  When non-empty, every row
            materialises a YAML in
            ``<base>/feature_groups/<NAME>.yaml``.  ``None`` and ``[]``
            are treated identically — no FG YAMLs are written.
        applied_state: Optional reconstructed
            :class:`~snowflake.ml.feature_store.decl.state.AppliedState`
            snapshot to drive full-fidelity exporter overlays (BatchFV
            source binding recovery from ``FV_SOURCE_REFS`` metadata,
            secondary keys, advanced authoring knobs, offline-only
            BFVs).  Forwarded verbatim to
            :func:`decl.exporter.export_specs` so the exporter can
            prefer the recovered ``spec_payload`` over the lossy raw
            ``DESCRIBE … TYPE = SPECIFICATION`` JSON.  FVs missing from
            *applied_state* (legacy / fresh-init case) soft-fall-back
            to ``specification_map``.  ``None`` runs the legacy
            ``specification_map``-only path unchanged.
        layout: ``"db_schema"`` (default, back-compat) writes to
            ``<output_dir>/<database>.<schema>/{...}/``.  ``"sources"``
            writes directly to ``<output_dir>/sources/{...}/`` so the
            result drops into the manifest project tree consumed by
            ``snow feature plan`` / ``apply``.

    Returns:
        Dict with keys ``status``, ``directory``, and ``files``.
    """
    from snowflake.ml.feature_store.decl.exporter import export_specs as _export_specs

    return _export_specs(
        show_rows,
        describe_rows_by_oft,
        output_dir,
        database,
        schema,
        specification_map=specification_map,
        entity_rows=entity_rows,
        feature_group_rows=feature_group_rows,
        applied_state=applied_state,
        layout=layout,
    )


def fetch_applied_state(
    raw_show_results: list[dict[str, Any]],
    raw_table_results: Optional[list[dict[str, Any]]] = None,
    describe_map: Optional[dict[str, list[dict[str, Any]]]] = None,
    *,
    specification_map: Optional[dict[str, dict[str, Any]]] = None,
    entity_rows: Optional[list[dict[str, Any]]] = None,
    dt_text_map: Optional[dict[str, str]] = None,
    feature_view_rows: Optional[list[dict[str, Any]]] = None,
    feature_group_rows: Optional[list[dict[str, Any]]] = None,
    stream_source_rows: Optional[list[dict[str, Any]]] = None,
    datasources_by_table: Optional[dict[str, Any]] = None,
    default_database: str = "",
    default_schema: str = "",
) -> AppliedState:
    """Parse raw Snowflake query results into an ``AppliedState`` snapshot.

    The CLI is responsible for executing the SQL queries and passing the
    raw results to this function. This design allows GS to provide the data
    via a REST API in the future without changing this function's signature.

    Args:
        raw_show_results: Rows from ``SHOW ONLINE FEATURE TABLES``.
        raw_table_results: Rows from ``SHOW TABLES`` (offline backing tables).
        describe_map: Mapping of OFT name → ``DESCRIBE ONLINE FEATURE TABLE``
            rows.  Used as the structural-fingerprint fallback when a
            specification is unavailable.
        specification_map: Mapping of OFT name → parsed spec JSON returned by
            ``DESCRIBE ONLINE FEATURE TABLE <name> TYPE = SPECIFICATION``.
            When provided, the FV's ``spec_payload`` is populated from this
            authoritative source and ``from_specification=True`` is set.
        entity_rows: Rows from
            ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'``.  Each row
            becomes an Entity ``AppliedObject``.
        dt_text_map: Mapping of Dynamic Table name (matching the OFT name
            on the BatchFV side) → DT DDL text returned by ``SHOW DYNAMIC
            TABLES``.  Used to recover the BatchFV's offline source-table
            binding that is dropped by ``DESCRIBE ONLINE FEATURE TABLE …
            TYPE = SPECIFICATION``; without it BatchFV planning falls back
            to ``RECREATE_FV`` on every operational edit because local-vs-
            applied source signatures cannot match.  See
            docs/BATCH_FV_BUG_BASH.md §6–§8 and the W-G(A) plan.
        feature_view_rows: Rows from
            :func:`fetch_feature_view_rows` (delegates to the
            imperative ``FeatureStore.list_feature_views()``).
            Surfaces offline-only ``BatchFeatureView``s that
            ``SHOW ONLINE FEATURE TABLES`` cannot enumerate.  When a
            row already appears via the OFT path, the OFT-derived
            ``spec_payload`` wins.  See
            ``plans/offline_bfv_state_fix_b9da0006.plan.md``.
        feature_group_rows: Rows from :func:`fetch_feature_group_rows`
            (delegates to the imperative
            ``FeatureStore.list_feature_groups()``).  ``FeatureGroup``
            is a fully imperative-side construct — there is no SHOW
            equivalent — so this is the only source of truth for FG
            applied state.  Each row becomes a
            ``AppliedObject(kind="FeatureGroup")`` whose
            ``spec_payload`` reconstructs the declarative shape so the
            planner's ``_fg_content_hash`` matches the local hash on
            an unchanged round-trip.
        stream_source_rows: Optional rows produced by
            :func:`fetch_stream_source_rows` (delegates to the
            imperative ``FeatureStore.list_stream_sources()``).  When
            provided, each row becomes a runtime-authoritative
            ``AppliedObject(kind="Datasource")`` whose ``spec_payload``
            matches the shape produced by the FV-derived datasource
            path so the planner's source-diff helper sees identical
            dedup keys; runtime entries win on collision with the
            FV-derived path.  ``None`` (the default) falls back to
            today's FV-derived-only behaviour for back-compat with
            CLI callers that have not yet wired the new read path.
            See ``plans/stream_source_contract.md`` §§5–6.
        datasources_by_table: Optional ``{physical_table_ident →
            logical BatchSource.name}`` lookup constructed by
            :func:`build_datasources_by_table` from the locally
            loaded specs.  When provided, BatchFV source-binding
            recovery prefers the operator-authored logical name
            from the local ``sources/datasources/`` tree over the
            physical table identifier recovered from the Dynamic
            Table DDL.  ``None`` (the default) preserves the legacy
            table-as-name behaviour (cold-start contract for fresh
            ``snow feature init`` against a never-seen schema).
        default_database: Database name used when a row does not include one.
        default_schema: Schema name used when a row does not include one.

    Returns:
        AppliedState snapshot parsed from the raw query results, including
        FeatureView objects, Entity objects (from ``entity_rows``), and
        derived Datasource objects (unioned across all FV
        ``spec.sources[]`` lists).
    """
    from snowflake.ml.feature_store.decl.state import (
        fetch_applied_state as _fetch_applied_state,
    )

    return _fetch_applied_state(
        raw_show_results,
        raw_table_results,
        describe_map=describe_map,
        specification_map=specification_map,
        entity_rows=entity_rows,
        dt_text_map=dt_text_map,
        feature_view_rows=feature_view_rows,
        feature_group_rows=feature_group_rows,
        stream_source_rows=stream_source_rows,
        datasources_by_table=datasources_by_table,
        default_database=default_database,
        default_schema=default_schema,
    )


def build_datasources_by_table(specs: Sequence[Any]) -> dict[str, Any]:
    """Build a physical-table → logical-source-name lookup from local specs.

    Public facade over :func:`decl.state._build_datasources_by_table`.
    The CLI plugin uses this lookup to thread operator-authored
    ``BatchSource.name`` values into :func:`fetch_applied_state` so the
    BatchFV source-binding recovery prefers the local logical name
    over the recovered physical table identifier.

    Args:
        specs: Iterable of loaded specs (the ``SpecBatch.specs`` list
            from :func:`load_project` is the typical input).
            Non-``BatchSource`` entries are skipped.

    Returns:
        Dict keyed by the uppercased unqualified table ident.  Values
        are either ``str`` (unique match — the operator's authored
        logical name) or ``list[str]`` (multi-match collision marker).
    """
    from snowflake.ml.feature_store.decl.state import (
        _build_datasources_by_table as _build,
    )

    return _build(specs)


def parse_specification_rows(
    rows: Optional[list[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """Parse rows from ``DESCRIBE ... TYPE = SPECIFICATION`` into spec JSON.

    Delegates to :func:`decl.state.parse_specification_rows`.

    Args:
        rows: Raw rows (list of dicts) returned by the SQL execution.

    Returns:
        Parsed spec JSON dict, or ``None`` if no row contained a parseable
        spec.
    """
    from snowflake.ml.feature_store.decl.state import parse_specification_rows as _parse

    return _parse(rows)


def compile_to_spec(
    spec_dict: dict[str, Any],
    database: str,
    schema: str,
    *,
    entity_join_keys: Optional[Mapping[str, list[str]]] = None,
) -> dict[str, Any]:
    """Compile a YAML authoring-format spec into imperative FeatureViewSpec format.

    Args:
        spec_dict: The loaded/compiled spec dict from the authoring YAML.
        database: Snowflake database name (from connection context).
        schema: Snowflake schema name (from connection context).
        entity_join_keys: Optional entity-name → ordered join-key-columns map.
            When provided, the FV's authored entity names are resolved to their
            join-key columns for the wire field ``ordered_entity_column_names``.
            When omitted, the authored ``entities`` list is copied verbatim
            (historical behaviour; keeps existing callers and hashes stable).

    Returns:
        Dict matching FeatureViewSpec.to_dict() output, ready for
        JSON serialization in FROM SPECIFICATION $$ ... $$.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import (
        compile_to_spec as _compile,
    )

    return _compile(spec_dict, database, schema, entity_join_keys=entity_join_keys)


# ---------------------------------------------------------------------------
# Query string factories (CLI should use these instead of hardcoded SQL)
# ---------------------------------------------------------------------------


def state_queries(database: str, schema: str) -> dict[str, str]:
    """Return SQL strings for fetching applied state."""
    from snowflake.ml.feature_store.decl.queries import state_queries as _sq

    return _sq(database, schema)


def list_query(database: str, schema: str) -> str:
    """Return the SHOW ONLINE FEATURE TABLES SQL string."""
    from snowflake.ml.feature_store.decl.queries import list_query as _lq

    return _lq(database, schema)


def describe_query(name: str, database: str, schema: str) -> str:
    """Return the SHOW ... LIKE SQL string for a named object."""
    from snowflake.ml.feature_store.decl.queries import describe_query as _dq

    return _dq(name, database, schema)


def describe_columns_query(name: str, database: str, schema: str) -> str:
    """Return the DESCRIBE ONLINE FEATURE TABLE SQL string."""
    from snowflake.ml.feature_store.decl.queries import describe_columns_query as _dcq

    return _dcq(name, database, schema)


def dynamic_tables_query(database: str, schema: str) -> str:
    """Return the ``SHOW DYNAMIC TABLES IN SCHEMA <db>.<schema>`` SQL string.

    Thin re-export over :func:`queries.dynamic_tables_query`.  The CLI uses
    this to issue one schema-wide ``SHOW DYNAMIC TABLES`` per state fetch
    so :func:`state.fetch_applied_state` can recover the BatchFV
    ``sources[0].table`` binding the deployed SPECIFICATION JSON loses on
    the FROM SPECIFICATION round-trip (deployed ``spec.sources`` is always
    ``[]`` for BatchFVs — the binding is encoded in the offline DT's
    ``SELECT ... FROM <table>`` body).

    Args:
        database: Snowflake database holding the target schema.
        schema: Snowflake schema to enumerate Dynamic Tables in.

    Returns:
        A ``SHOW DYNAMIC TABLES IN SCHEMA <db>.<schema>`` SQL string ready
        for ``execute_query``.
    """
    from snowflake.ml.feature_store.decl.queries import dynamic_tables_query as _dtq

    return _dtq(database, schema)


def drop_queries(names: list[str], database: str, schema: str) -> list[str]:
    """Return DROP SQL strings for the named objects."""
    from snowflake.ml.feature_store.decl.queries import drop_queries as _dqs

    return _dqs(names, database, schema)


def describe_specification_query(database: str, schema: str, name: str) -> str:
    """Return the ``DESCRIBE ... TYPE = SPECIFICATION`` SQL string."""
    from snowflake.ml.feature_store.decl.queries import (
        describe_specification_query as _q,
    )

    return _q(database, schema, name)


def fetch_entity_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch entity tag rows via the imperative ``FeatureStore.list_entities()``.

    Thin facade over :func:`imperative_executor.fetch_entity_rows`.  This
    is the only place outside of ``imperative_executor.py`` that may
    transitively reach into
    ``snowflake.ml.feature_store.feature_store``; the lazy import lives
    inside the executor function.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name to scope the listing to.
        schema: Snowflake schema name to scope the listing to.
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of row dicts in the legacy ``SHOW TAGS`` shape (keys
        ``name``, ``database_name``, ``schema_name``, ``allowed_values``,
        ``comment``, ``owner``).  Returns an empty list when no
        entities are registered.
    """
    from snowflake.ml.feature_store.decl.imperative_executor import (
        fetch_entity_rows as _fetch,
    )

    return _fetch(session, database, schema, warehouse)


def fetch_feature_group_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch FG rows via the imperative ``FeatureStore.list_feature_groups()``.

    Thin facade over :func:`imperative_executor.fetch_feature_group_rows`.
    Mirrors :func:`fetch_entity_rows` / :func:`fetch_feature_view_rows` and
    is the only sanctioned entry point for the CLI manager.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name to scope the listing to.
        schema: Snowflake schema name to scope the listing to.
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of FG row dicts (see
        :func:`imperative_executor.fetch_feature_group_rows`).  Returns an
        empty list when no FeatureGroup is registered.
    """
    from snowflake.ml.feature_store.decl.imperative_executor import (
        fetch_feature_group_rows as _fetch,
    )

    return _fetch(session, database, schema, warehouse)


def fetch_feature_view_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch FV rows via the imperative ``FeatureStore.list_feature_views()``.

    Thin facade over :func:`imperative_executor.fetch_feature_view_rows`.
    This is the parallel of :func:`fetch_entity_rows` for
    ``FeatureView`` enumeration — its existence allows
    :func:`fetch_applied_state` to surface offline-only
    ``BatchFeatureView``s that ``SHOW ONLINE FEATURE TABLES`` cannot
    enumerate.  Without it, an offline-only BFV is missing from
    ``AppliedState`` after apply and the next ``snow feature plan``
    re-emits a spurious ``CREATE_FV``.  See
    ``plans/offline_bfv_state_fix_b9da0006.plan.md``.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name to scope the listing to.
        schema: Snowflake schema name to scope the listing to.
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of FV row dicts in the Phase-1 contract shape (Section 7
        of the plan).  Returns an empty list when no FVs are
        registered.
    """
    from snowflake.ml.feature_store.decl.imperative_executor import (
        fetch_feature_view_rows as _fetch,
    )

    return _fetch(session, database, schema, warehouse)


def fetch_stream_source_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch stream-source rows via the imperative ``FeatureStore.list_stream_sources()``.

    Thin facade over :func:`imperative_executor.fetch_stream_source_rows`.
    Mirrors :func:`fetch_entity_rows` / :func:`fetch_feature_view_rows` /
    :func:`fetch_feature_group_rows` and is the only sanctioned entry
    point for the CLI manager — the lazy import of
    ``snowflake.ml.feature_store`` lives inside the executor module so
    the planning-time wheel stays free of the imperative dependency.

    The returned rows feed :func:`fetch_applied_state` via the
    ``stream_source_rows`` kwarg, where they become runtime-authoritative
    ``AppliedObject(kind="Datasource")`` entries that beat the
    FV-derived datasource path on dedup-key collision.  See
    ``plans/stream_source_contract.md`` §3 for the row-shape contract
    and §6 for the facade contract.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name to scope the listing to.
        schema: Snowflake schema name to scope the listing to.
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of stream-source row dicts in the contract §3 shape
        (``name`` / ``schema`` / ``desc`` / ``owner``).  Returns an
        empty list when no stream source is registered.
    """
    from snowflake.ml.feature_store.decl.imperative_executor import (
        fetch_stream_source_rows as _fetch,
    )

    return _fetch(session, database, schema, warehouse)


def list_state_queries(database: str, schema: str) -> dict[str, str]:
    """Return the SQL set used by ``snow feature list``.

    Note: entity rows are no longer fetched via SQL; the CLI calls
    :func:`fetch_entity_rows` (which delegates to the imperative
    ``FeatureStore.list_entities()``) to enumerate entities.  Only OFT
    SHOW + DESCRIBE SQL is returned here.

    Args:
        database: Snowflake database to scope the queries to.
        schema: Snowflake schema to scope the queries to.

    Returns:
        Dict containing ``show_ofts`` and the
        ``describe_specification_template`` (a ``{name}``-format string
        for per-OFT spec retrieval).
    """
    from snowflake.ml.feature_store.decl.queries import list_state_queries as _q

    return _q(database, schema)


def enrich_list_results(
    show_rows: list[dict[str, Any]],
    describe_map: Optional[dict[str, list[dict[str, Any]]]] = None,
    *,
    entity_rows: Optional[list[dict[str, Any]]] = None,
    specification_map: Optional[dict[str, dict[str, Any]]] = None,
    feature_group_rows: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Enrich raw SHOW/DESCRIBE/TAG rows into a multi-kind list output.

    Produces a single, ordered list of display dicts that the CLI can
    render in a table with a ``type`` column.  The output covers three
    kinds: ``FeatureView``, ``Entity``, and ``Datasource``.

    Ordering: FeatureView rows in input order, then Entity rows in input
    order, then Datasource rows sorted by name for deterministic output.

    Args:
        show_rows: Rows from ``SHOW ONLINE FEATURE TABLES``.
        describe_map: Optional mapping of OFT name → ``DESCRIBE`` column
            rows.  Used as the fallback for the ``entities`` column when
            ``specification_map`` does not contain an entry for that OFT.
        entity_rows: Optional rows from
            ``SHOW TAGS LIKE 'SNOWML_FEATURE_STORE_ENTITY_%'``.  Each row
            becomes one Entity output row.
        specification_map: Optional mapping of OFT name → parsed spec
            JSON returned by ``DESCRIBE ... TYPE = SPECIFICATION``.  When
            present, the FV's ``entities`` column comes from
            ``ordered_entity_column_names`` and Datasource rows are
            derived from the union of ``spec.sources[]``.
        feature_group_rows: Optional rows produced by
            :func:`fetch_feature_group_rows`.  Reserved for the
            FeatureGroup-aware enrichment shipping in Phase 5b.  v1 of
            FG support accepts the kwarg so the CLI can already wire
            it through, but does not yet emit FG rows into the list
            output (Phase 5b extends this function to surface them).

    Returns:
        Ordered list of enriched row dicts.  Each dict carries a
        ``type`` column plus the columns used by the CLI's table
        projection (``feature_view``, ``version``, ``entities``,
        ``database_name``, ``schema_name``) and a ``details`` dict for
        kind-specific extras.
    """
    import json as _json

    from snowflake.ml.feature_store.decl.state import _parse_oft_name
    from snowflake.ml.feature_store.decl.types import ObjectKind

    _ENTITY_TAG_PREFIX = ENTITY_TAG_PREFIX

    describe_lookup = describe_map or {}
    enriched: list[dict[str, Any]] = []

    # 1. FeatureView rows (preserve original SHOW columns).
    _FV_SUBKINDS = {
        "StreamingFeatureView",
        "RealtimeFeatureView",
        "BatchFeatureView",
    }
    for row in show_rows:
        r = dict(row)
        oft_name = r.get("name", "") or r.get("NAME", "")
        base_name, version = _parse_oft_name(oft_name)

        # Resolve the spec dict once, in priority order:
        #   1. ``specification_map[oft_name]`` — the parsed spec from
        #      ``DESCRIBE … TYPE = SPECIFICATION`` (most authoritative).
        #   2. ``json.loads(row['specification'])`` — the legacy
        #      embedded JSON column on ``SHOW ONLINE FEATURE TABLES``.
        # The same ``spec`` is then used to derive both the FV subkind
        # (for the ``type`` column) and the entities list, keeping the
        # two derivations consistent.
        spec: Optional[dict[str, Any]] = specification_map.get(oft_name) if specification_map else None
        if spec is None:
            raw_embedded = r.get("specification") or r.get("SPECIFICATION") or ""
            if raw_embedded:
                try:
                    parsed = _json.loads(raw_embedded)
                    if isinstance(parsed, dict):
                        spec = parsed
                except (ValueError, TypeError):
                    pass

        # Surface the FV subkind in ``type`` when the spec carries one
        # of the canonical values; otherwise fall back to the generic
        # ``"FeatureView"`` so callers downstream (CLI table, planner)
        # only ever see one of the four well-known strings.
        fv_type = ObjectKind.FEATURE_VIEW
        if isinstance(spec, dict):
            candidate = spec.get("kind")
            if isinstance(candidate, str) and candidate in _FV_SUBKINDS:
                fv_type = candidate
        r["type"] = fv_type

        # Surface the user-facing name as ``name`` for consistent table
        # display across all kinds; the raw OFT name is preserved in
        # ``oft_name`` for any downstream consumers.  Names are kept in
        # the canonical Snowflake form that ``_parse_oft_name`` returned
        # from ``SHOW ONLINE FEATURE TABLES`` (upper-case for unquoted
        # identifiers, original case for quoted) — no client-side case
        # folding.  Folding to lower-case here made ``snow feature
        # list`` disagree with every other surface (planner ops, OFT
        # names, ``feature describe``) that already renders identifiers
        # in canonical form.
        r["oft_name"] = oft_name
        r["name"] = base_name
        r["feature_view"] = base_name
        r["version"] = version

        entities_str = ""
        if spec is not None:
            inner = spec.get("spec") if isinstance(spec.get("spec"), dict) else spec
            cols = inner.get("ordered_entity_column_names") if isinstance(inner, dict) else None
            if isinstance(cols, list) and cols:
                entities_str = ", ".join(str(c) for c in cols)

        if not entities_str:
            desc_rows = describe_lookup.get(oft_name, [])
            pk_cols: list[str] = []
            for col in desc_rows:
                is_pk = False
                for pk_key in ("primary key", "PRIMARY KEY", "primary_key"):
                    val = col.get(pk_key, "")
                    if val and str(val).upper() in ("Y", "YES", "TRUE", "1"):
                        is_pk = True
                        break
                if is_pk:
                    pk_cols.append(col.get("name", col.get("NAME", "")))
            entities_str = ", ".join(pk_cols) if pk_cols else ""

        r["entities"] = entities_str
        if "database_name" not in r:
            r["database_name"] = r.get("DATABASE_NAME", "") or ""
        if "schema_name" not in r:
            r["schema_name"] = r.get("SCHEMA_NAME", "") or ""
        r["details"] = {"scheduling_state": r.get("scheduling_state", "")}
        enriched.append(r)

    # 2. Entity rows derived from ``SHOW TAGS``.  These are authoritative;
    # the previous "inferred from PK columns" fallback has been removed,
    # so every Entity row in the output corresponds to a registered tag.
    if entity_rows:
        for ent_row in entity_rows:
            raw_name = ent_row.get("name") or ent_row.get("NAME") or ""
            if not raw_name or not raw_name.startswith(_ENTITY_TAG_PREFIX):
                continue
            # Tag names come back from ``SHOW TAGS`` in Snowflake's
            # canonical form (upper-case for unquoted, original case
            # for quoted).  Strip the prefix and preserve that casing —
            # the previous ``.lower()`` made the list output disagree
            # with the canonical entity name used everywhere else.
            entity_name = raw_name[len(_ENTITY_TAG_PREFIX) :]
            db = ent_row.get("database_name") or ent_row.get("DATABASE_NAME") or ""
            schema_val = ent_row.get("schema_name") or ent_row.get("SCHEMA_NAME") or ""

            join_keys: list[str] = []
            raw_allowed = (
                ent_row.get("allowed_values") or ent_row.get("ALLOWED_VALUES") or ent_row.get("allowed_values_list")
            )
            if raw_allowed:
                if isinstance(raw_allowed, list):
                    join_keys = [str(v) for v in raw_allowed]
                elif isinstance(raw_allowed, str):
                    try:
                        parsed = _json.loads(raw_allowed)
                        if isinstance(parsed, list):
                            join_keys = [str(v) for v in parsed]
                    except (ValueError, TypeError):
                        join_keys = [v.strip() for v in raw_allowed.strip("[]").split(",") if v.strip()]

            details: dict[str, Any] = {}
            comment_val = ent_row.get("comment") or ent_row.get("COMMENT")
            if comment_val:
                details["comment"] = comment_val

            enriched.append(
                {
                    "type": ObjectKind.ENTITY,
                    "name": entity_name,
                    "feature_view": "",
                    "version": "",
                    "entities": ", ".join(join_keys),
                    "database_name": db,
                    "schema_name": schema_val,
                    "details": details,
                }
            )

    # 3. FeatureGroup rows from feature_group_rows.  One output row
    # per input row, with the canonical CLI-table columns (name,
    # version, database_name, schema_name) lifted directly and an
    # FG-specific ``details`` block summarising the FV-source composition
    # (count + ordered ``<fv_name>:<fv_version>`` list).  FGs sit between
    # Entity and Datasource rows so the table groups deployed objects
    # (FV / Entity / FG) above derived Datasource rows.
    if feature_group_rows:
        for fg_row in feature_group_rows:
            fg_name = str(fg_row.get("name", "") or "")
            if not fg_name:
                continue
            fg_version = str(fg_row.get("version", "") or "")
            fg_db = str(fg_row.get("database_name", "") or "")
            fg_schema = str(fg_row.get("schema_name", "") or "")
            sources = fg_row.get("sources") or []
            source_summary: list[str] = []
            for src in sources:
                if not isinstance(src, dict):
                    continue
                fv_name = str(src.get("fv_name", "") or "")
                fv_version = str(src.get("fv_version", "") or "")
                if not fv_name:
                    continue
                source_summary.append(f"{fv_name}:{fv_version}" if fv_version else fv_name)
            details = {
                "source_count": len(source_summary),
                "sources": source_summary,
            }
            desc_val = fg_row.get("desc")
            if isinstance(desc_val, str) and desc_val:
                details["desc"] = desc_val
            enriched.append(
                {
                    "type": ObjectKind.FEATURE_GROUP,
                    "name": fg_name,
                    "feature_view": "",
                    "version": fg_version,
                    "entities": "",
                    "database_name": fg_db,
                    "schema_name": fg_schema,
                    "details": details,
                }
            )

    # 4. Datasource rows: union spec.sources[] across all FVs, dedupe
    # case-insensitively (Snowflake unquoted identifiers are
    # case-insensitive), then sort for deterministic output.  The
    # displayed ``name`` preserves the case as stored in the recovered
    # spec JSON; only the dedup key is case-folded.
    ds_by_key: dict[str, dict[str, Any]] = {}
    if specification_map:
        for spec in specification_map.values():
            if not isinstance(spec, dict):
                continue
            inner = spec.get("spec") if isinstance(spec.get("spec"), dict) else spec
            sources = inner.get("sources", []) if isinstance(inner, dict) else []
            if not isinstance(sources, list):
                continue
            _md = spec.get("metadata")
            metadata = _md if isinstance(_md, dict) else {}
            db = metadata.get("database", "") or ""
            schema_val = metadata.get("schema", "") or ""

            for src in sources:
                if not isinstance(src, dict):
                    continue
                src_name = src.get("name", "")
                if not src_name:
                    continue
                dedup_key = src_name.upper()
                if dedup_key in ds_by_key:
                    continue
                src_type = src.get("source_type") or src.get("type") or ""
                cols = src.get("columns", [])
                col_count = len(cols) if isinstance(cols, list) else 0
                ds_by_key[dedup_key] = {
                    "type": ObjectKind.DATASOURCE,
                    "name": src_name,
                    "feature_view": "",
                    "version": "",
                    "entities": "",
                    "database_name": db,
                    "schema_name": schema_val,
                    "details": {
                        "source_type": src_type,
                        "column_count": col_count,
                    },
                }

    # Finalize referenced_by lists deterministically and emit sorted by
    # case-folded key (matches the dedup key so output is stable across
    # case-different spellings of the same source).
    for ds_key in sorted(ds_by_key):
        ds_entry = ds_by_key[ds_key]
        ref_by = ds_entry.get("details", {}).get("referenced_by")
        if isinstance(ref_by, list):
            ds_entry["details"]["referenced_by"] = sorted(ref_by)
        enriched.append(ds_entry)

    return enriched


# ---------------------------------------------------------------------------
# Plan file serialization
# ---------------------------------------------------------------------------


def serialize_plan(
    plan: Plan,
    database: str,
    schema: str,
    source_files: list[str],
    target_name: str = "",
) -> str:
    """Serialize a Plan to a JSON string for writing to a plan file.

    Wraps the plan in a :class:`PlanFile` envelope, computes an op-kind
    summary, stamps a UTC ISO-8601 timestamp, and returns the result as a
    pretty-printed JSON string.

    Args:
        plan: The execution plan to serialize.
        database: Target Snowflake database name.
        schema: Target Snowflake schema name.
        source_files: Input spec file paths that produced the plan.
        target_name: Optional manifest target name (e.g. ``"DEV"``)
            the plan was generated against (D4-ext / Phase 3+4).
            Defaults to ``""`` for plans generated outside the
            manifest-driven CLI path (callers that omit the kwarg
            keep the pre-D4-ext envelope shape; ``apply --plan
            <file>`` then accepts the plan regardless of
            ``--target``).

    Returns:
        JSON string representing the plan file.
    """
    import json
    from datetime import datetime, timezone

    # Build summary: count ops by kind
    summary: dict[str, int] = {}
    for op in getattr(plan, "ops", []):
        kind_val = op.kind.value if hasattr(op.kind, "value") else str(op.kind)
        summary[kind_val] = summary.get(kind_val, 0) + 1

    pf = PlanFile(
        created_at=datetime.now(timezone.utc).isoformat(),
        target_database=database,
        target_schema=schema,
        target_name=target_name,
        source_files=list(source_files),
        plan=plan,
        summary=summary,
    )
    return json.dumps(pf.model_dump(mode="json"), indent=2)


def deserialize_plan(json_str: str) -> PlanFile:
    """Deserialize a JSON plan file string into a :class:`PlanFile`.

    Args:
        json_str: JSON string previously produced by :func:`serialize_plan`.

    Returns:
        PlanFile instance deserialized from the JSON string.

    Raises:
        ValueError: If *json_str* is not valid JSON or cannot be parsed into a
            :class:`PlanFile`.
    """
    import json

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid plan file JSON: {exc}") from exc
    return PlanFile.model_validate(data)


# ---------------------------------------------------------------------------
# Combined apply pipeline
# ---------------------------------------------------------------------------


def resolve_datasource_columns(batch: Any) -> None:
    """Resolve datasource columns and batch **table** / **query** into FV sources.

    Scans the batch for ``StreamingSource`` / ``BatchSource`` specs, then for
    each feature view's ``sources[]`` entries that match by name (case
    insensitive):

    * Injects ``columns`` when the reference omits them (streaming ingest
      contract and batch validation).
    * For ``BatchSource`` (and any datasource carrying a ``table`` / ``query``),
      injects ``table``, ``query``, ``source_database``, and ``source_schema``
      when the FV reference does not already declare a resolvable location — this
      is required for :func:`decl.imperative_executor._build_feature_df` because
      apply replays plan payloads without re-reading datasource YAML files.

    Mutates specs in place (including replacing ``SourceRef`` models with
    ``model_copy`` updates when new fields are merged).

    Both ``manager.plan`` and ``manager.write_plan`` call this helper before
    ``generate_plan`` — see ARCHITECTURE.md "Step 4 — Datasource Resolution".

    Args:
        batch: A compiled batch containing specs to resolve.
    """
    from snowflake.ml.feature_store.decl.spec_models import FSColumn, SourceRef

    ds_columns: dict[str, list[Any]] = {}
    ds_location: dict[str, dict[str, Any]] = {}

    for spec in getattr(batch, "specs", []):
        kind = getattr(spec, "kind", "")
        if kind not in ("StreamingSource", "BatchSource"):
            continue
        name = getattr(spec, "name", "")
        if not name:
            continue
        lk = name.lower()
        columns = getattr(spec, "columns", [])
        if columns:
            col_dicts = []
            for c in columns:
                if hasattr(c, "model_dump"):
                    d = c.model_dump()
                elif hasattr(c, "dict"):
                    d = c.dict()
                elif isinstance(c, dict):
                    d = dict(c)
                else:
                    continue
                col_dicts.append({k: v for k, v in d.items() if v is not None})
            if col_dicts:
                ds_columns[lk] = col_dicts

        if kind == "BatchSource":
            loc: dict[str, Any] = {}
            for attr in ("table", "query", "source_database", "source_schema"):
                v = getattr(spec, attr, None)
                if v is not None and str(v).strip() != "":
                    loc[attr] = v
            if loc:
                ds_location[lk] = loc

    if not ds_columns and not ds_location:
        return

    def _fscolumn_list(col_dicts: list[Any]) -> list[FSColumn]:
        out: list[FSColumn] = []
        for raw in col_dicts:
            if isinstance(raw, FSColumn):
                out.append(raw)
                continue
            if isinstance(raw, dict):
                out.append(FSColumn.model_validate(raw))
        return out

    for spec in getattr(batch, "specs", []):
        kind = getattr(spec, "kind", "")
        if kind not in ("StreamingFeatureView", "RealtimeFeatureView", "BatchFeatureView"):
            continue
        sources = getattr(spec, "sources", [])
        for i, src in enumerate(list(sources)):
            if isinstance(src, dict):
                src_name = str(src.get("name", "") or "")
                src_cols = src.get("columns")
                has_table = bool(src.get("table"))
                has_query = bool(src.get("query"))
            elif isinstance(src, SourceRef):
                src_name = str(src.name or "")
                src_cols = src.columns or None
                has_table = bool(src.table)
                has_query = bool(src.query)
            else:
                src_name = str(getattr(src, "name", "") or "")
                src_cols = getattr(src, "columns", None)
                has_table = bool(getattr(src, "table", None))
                has_query = bool(getattr(src, "query", None))

            if not src_name:
                continue
            lk = src_name.lower()
            resolved_cols = ds_columns.get(lk)
            loc = ds_location.get(lk) or {}

            need_cols = resolved_cols and not src_cols
            need_loc = loc and not has_table and not has_query
            if not need_cols and not need_loc:
                continue

            if isinstance(src, SourceRef):
                updates: dict[str, Any] = {}
                if need_cols:
                    updates["columns"] = _fscolumn_list(resolved_cols or [])
                if need_loc:
                    for key, val in loc.items():
                        if val is not None and str(val).strip() != "":
                            updates[key] = val
                sources[i] = src.model_copy(update=updates)
            elif isinstance(src, dict):
                if need_cols:
                    src["columns"] = resolved_cols
                if need_loc:
                    for key, val in loc.items():
                        if src.get(key) in (None, "") and val not in (None, ""):
                            src[key] = val
            else:
                try:
                    if need_cols and hasattr(src, "columns"):
                        src.columns = _fscolumn_list(resolved_cols or [])
                    if need_loc:
                        for key, val in loc.items():
                            if getattr(src, key, None) in (None, "") and val not in (None, ""):
                                setattr(src, key, val)
                except (AttributeError, TypeError, ValueError):
                    pass


# ---------------------------------------------------------------------------
# Imperative execution
# ---------------------------------------------------------------------------


def execute_plan(
    plan: Any,
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
    options: Any,
) -> Any:
    """Execute a Plan by delegating each PlanOp to FeatureStore methods.

    This is the primary execution path for ``snow feature apply``.  The
    executor lazy-imports ``FeatureStore``, ``FeatureView``, and ``Entity``
    from ``snowflake.ml.feature_store`` — after upstream lazy-import fixes,
    these imports do NOT pull in numpy/pandas/pyarrow.

    Args:
        plan: Plan from ``generate_plan()``.
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name.
        warehouse: Snowflake warehouse name.
        options: PlanOptions controlling execution (overwrite, etc.).

    Returns:
        ApplyResult with per-operation status.
    """
    from snowflake.ml.feature_store.decl.imperative_executor import (
        execute_plan as _execute,
    )

    return _execute(plan, session, database, schema, warehouse, options)


# ---------------------------------------------------------------------------
# Manifest / project facades (Phase 2)
# ---------------------------------------------------------------------------
#
# These wrap ``decl.manifest``, ``decl.project_paths``, and ``decl.loader``
# so CLI plugins import them through ``decl_api`` only.  Per the
# ARCHITECTURE boundary rule, plugin code must not reach into the
# internal modules directly.  Locked decisions enforced here:
#
# * D2 (with-role): ``resolve_target`` returns a target whose
#   ``database`` / ``schema`` are non-empty; warehouse is never part of
#   the resolved target — it is sourced from the active connection.
# * D5 (drop-config): template-variable precedence is
#   ``defaults < configurations[target.templating_config] < runtime_vars``.
#   Dictionary-typed defaults cannot be overridden at runtime; a
#   ``ValueError`` is raised naming the offending key.
# * D7 (defer): macros logic is intentionally absent here.


def load_manifest(project_root: Path) -> FSManifest:
    """Load and validate the ``manifest.yml`` at ``<project_root>/manifest.yml``.

    Errors from :meth:`FSManifest.load` propagate unchanged
    (:class:`ManifestNotFoundError` when the file is absent,
    :class:`InvalidManifestError` for malformed or unsupported
    manifests, :class:`ManifestConfigurationError` for semantic issues
    such as missing required target fields or a forbidden
    ``warehouse``).

    Args:
        project_root: Resolved project root (the directory containing
            ``manifest.yml``).

    Returns:
        FSManifest parsed and validated from the on-disk file.
    """
    return FSManifest.load(project_root)


def discover_project(start: Optional[Path] = None) -> FSProjectPaths:
    """Walk up from ``start`` (default cwd) until ``manifest.yml`` is found.

    Thin facade over :meth:`FSProjectPaths.discover`. The underlying
    :class:`ManifestNotFoundError` propagates unchanged when no
    ``manifest.yml`` is found in ``start`` or any ancestor.

    Args:
        start: Directory (or file) to start the walk from.  Defaults to
            :func:`pathlib.Path.cwd`.

    Returns:
        FSProjectPaths rooted at the first ancestor containing
        ``manifest.yml``.
    """
    return FSProjectPaths.discover(start)


def resolve_target(
    manifest: FSManifest,
    target_name: Optional[str] = None,
) -> FSTarget:
    """Return the effective :class:`FSTarget` by name or default.

    When ``target_name`` is ``None`` the manifest's ``default_target``
    is used (auto-derived to the sole target for single-target
    manifests). :class:`ManifestConfigurationError` from
    :meth:`FSManifest.get_effective_target` propagates unchanged when
    the name is unknown or no default is configured.

    Args:
        manifest: Parsed manifest returned by :func:`load_manifest`.
        target_name: Optional explicit target name (case-insensitive).

    Returns:
        FSTarget whose ``database`` / ``schema`` are non-empty (per
        D2). ``warehouse`` is never part of the resolved target.
    """
    return manifest.get_effective_target(target_name)


def _merge_template_vars(
    manifest: FSManifest,
    target: FSTarget,
    runtime_vars: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the effective template-variable dict for a target.

    Implements the D5 precedence rule:

        ``defaults < configurations[target.templating_config] < runtime_vars``

    Dictionary-typed defaults cannot be overridden at runtime; a
    :class:`ValueError` is raised naming the offending key.  This mirrors
    the DCM rule that runtime variable overrides are scalar-only.

    Phase 5A may extract this helper into ``decl/templating.py``; for
    now it lives here so the manifest / loader modules can stay
    dependency-free.

    Args:
        manifest: Parsed manifest carrying ``templating.defaults`` and
            ``templating.configurations``.
        target: Resolved target; its ``templating_config`` selects which
            configuration is layered on top of the defaults.
        runtime_vars: ``--variable key=value`` overrides supplied at the
            CLI.  ``None`` and ``{}`` are treated identically.

    Returns:
        New dict containing the merged template variables.  The input
        manifest / runtime_vars are never mutated.

    Raises:
        ValueError: If ``runtime_vars`` attempts to override a
            dictionary-typed default; the message names the key.
    """
    defaults = dict(manifest.templating.defaults)

    merged: dict[str, Any] = dict(defaults)
    if target.templating_config:
        configuration = manifest.templating.configurations.get(target.templating_config) or {}
        merged.update(configuration)

    if not runtime_vars:
        return merged

    for key, value in runtime_vars.items():
        default_value = defaults.get(key)
        if isinstance(default_value, dict):
            raise ValueError(
                f"Cannot override templating default '{key}' from runtime variables: "
                "dictionary-typed defaults are not overridable (per D5)."
            )
        merged[key] = value

    return merged


def load_project(
    project_root: Path,
    *,
    target: FSTarget,
    runtime_vars: Optional[dict[str, Any]] = None,
) -> SpecBatch:
    """Load every spec under ``<project_root>/sources/`` qualified by ``target``.

    Threads ``database`` / ``schema`` from ``target`` into every spec
    (per D2 — warehouse is intentionally never injected). Template
    variables are merged via :func:`_merge_template_vars` so that
    ``manifest.templating.defaults <
    manifest.templating.configurations[target.templating_config] <
    runtime_vars`` (per D5). When ``runtime_vars`` tries to override a
    dictionary-typed default the :class:`ValueError` raised by
    :func:`_merge_template_vars` propagates unchanged; spec-file load
    failures surface as :class:`SpecLoadError` from
    :func:`loader.load_from_project`.

    Args:
        project_root: Resolved project root (the directory containing
            ``manifest.yml``).  ``<project_root>/sources/`` MUST exist.
        target: Resolved target from :func:`resolve_target`.  Provides
            the active ``database`` / ``schema`` and selects the
            templating configuration.
        runtime_vars: Optional ``--variable key=value`` overrides.
            Dictionary-typed defaults cannot be overridden.

    Returns:
        SpecBatch with every loaded spec in deterministic order
        (entities, then datasources, then feature_views; lexicographic
        within each sub-directory).
    """
    from snowflake.ml.feature_store.decl import loader

    manifest = load_manifest(project_root)
    template_vars = _merge_template_vars(manifest, target, runtime_vars)
    return loader.load_from_project(
        project_root,
        database=target.database,
        schema=target.schema,
        template_vars=template_vars or None,
    )
