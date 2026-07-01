"""Streaming feature view registration helpers.

Extracted from ``FeatureStore`` to isolate the streaming-specific logic:

- Preamble: probe schema inference, empty udf_transformed table creation.
- Postamble: backfill task graph creation, metadata save, ref_count increment.
- Spec builder: ``_build_streaming_feature_view_spec`` for the OFT spec.
- Cleanup: drop backfill resources and decrement stream source ref count.

DT and OFT creation reuse the existing ``FeatureStore`` code paths; after the
preamble runs, ``feature_view.query`` / ``output_schema`` already reflect the
udf_transformed table.
"""

from __future__ import annotations

import datetime
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Optional

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewVersion,
    _FeatureViewMetadata,
)
from snowflake.ml.feature_store.metadata_manager import (
    BACKFILL_STATE_RUNNING,
    FeatureStoreMetadataManager,
    StreamingMetadata,
)
from snowflake.ml.feature_store.spec.builder import (
    FeatureViewSpecBuilder,
    SnowflakeTableInfo,
)
from snowflake.ml.feature_store.spec.enums import FeatureViewKind, TableType
from snowflake.ml.feature_store.spec.models import (
    FeatureViewSpec,
    _columns_from_struct_type,
    validate_fs_columns_match,
)
from snowflake.ml.feature_store.stream_config import (
    _infer_structtype_from_pandas,
    _snowpark_type_to_sql,
)
from snowflake.ml.feature_store.stream_source import (
    StreamSource,
    validate_schema_field_types,
)
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preamble — runs before the existing DT/OFT creation paths
# ---------------------------------------------------------------------------


_BACKFILL_TABLE_SUFFIX = "$BACKFILL"

# Backfill task graph naming. The shared ``$BACKFILL_`` prefix lets
# ``get_refresh_history`` and integ polling find all members with a single
# ``NAME LIKE '<fv>$<ver>$BACKFILL\\_%' ESCAPE '\\'``. Phase 2 will add
# ``$BACKFILL_W<NNN>`` child tasks; the suffix below is reserved for that.
_BACKFILL_ROOT_SUFFIX = "$BACKFILL_ROOT"
_BACKFILL_FINALIZE_SUFFIX = "$BACKFILL_FINALIZE"
_BACKFILL_WINDOW_PREFIX = "$BACKFILL_W"
_BACKFILL_PROC_SUFFIX = "$BACKFILL_PROC"
_BACKFILL_UDTF_SUFFIX = "$BACKFILL_UDTF"

# Snowflake only supports the un-tagged ``$$`` dollar-quote delimiter
# (unlike Postgres ``$tag$``). Proc body and UDTF body are emitted as
# separate top-level DDL, never nested, so one delimiter is enough.
_DOLLAR_QUOTE = "$$"


def _get_backfill_table_name(udf_table_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill table name from a udf_transformed table name."""
    return SqlIdentifier(f"{udf_table_name.resolved()}{_BACKFILL_TABLE_SUFFIX}", case_sensitive=True)


def _get_backfill_root_task_name(physical_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill root task name from the physical FV name (``<fv>$<ver>``)."""
    return SqlIdentifier(
        f"{physical_name.resolved()}{_BACKFILL_ROOT_SUFFIX}",
        case_sensitive=True,
    )


def _get_backfill_finalize_task_name(physical_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill finalizer task name from the physical FV name."""
    return SqlIdentifier(
        f"{physical_name.resolved()}{_BACKFILL_FINALIZE_SUFFIX}",
        case_sensitive=True,
    )


def _get_backfill_window_task_name(physical_name: SqlIdentifier, window_idx: int) -> SqlIdentifier:
    """Derive the backfill window task name (Phase 2; unused in Phase 1).

    The %03d suffix keeps lexical = chronological order; capped at 999 windows.

    Args:
        physical_name: Physical FV name (``<fv>$<ver>``).
        window_idx: Zero-based window index (0–999).

    Returns:
        ``<physical_name>$BACKFILL_W<NNN>`` identifier.
    """
    return SqlIdentifier(
        f"{physical_name.resolved()}{_BACKFILL_WINDOW_PREFIX}{window_idx:03d}",
        case_sensitive=True,
    )


def _get_backfill_proc_name(physical_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill stored procedure name from the physical FV name."""
    return SqlIdentifier(
        f"{physical_name.resolved()}{_BACKFILL_PROC_SUFFIX}",
        case_sensitive=True,
    )


# Single source of truth for the backfill proc parameter list — keeps the
# ``CREATE PROCEDURE`` (named) and ``DROP PROCEDURE`` (type-only) renderers
# from drifting and silently mismatching the registered overload.
_BACKFILL_PROC_PARAMS: tuple[tuple[str, str], ...] = (
    ("WINDOW_START", "TIMESTAMP_NTZ"),
    ("WINDOW_END", "TIMESTAMP_NTZ"),
)


def _get_backfill_proc_param_list() -> str:
    """Named parameter list for ``CREATE PROCEDURE`` — e.g. ``(A T1, B T2)``."""
    return "(" + ", ".join(f"{n} {t}" for n, t in _BACKFILL_PROC_PARAMS) + ")"


def _get_backfill_proc_signature() -> str:
    """Type-only signature for ``DROP PROCEDURE`` — e.g. ``(T1, T2)``."""
    return "(" + ", ".join(t for _, t in _BACKFILL_PROC_PARAMS) + ")"


def _render_udtf_signature(input_col_types: list[str]) -> str:
    """Argument-type signature for the backfill UDTF, used in ``DROP FUNCTION``.

    Snowflake disambiguates overloads by argument types, so this must match
    the original ``CREATE FUNCTION`` exactly. Persisted in metadata so cleanup
    paths can drop the UDTF without re-deriving the schema.

    Args:
        input_col_types: SQL type strings (e.g. ``["VARCHAR", "FLOAT"]``).

    Returns:
        Parenthesized comma-separated signature, e.g. ``"(VARCHAR, FLOAT)"``.
    """
    return f"({', '.join(input_col_types)})"


def _get_backfill_udtf_name(physical_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill UDTF name from the physical FV name.

    Created as a permanent function because Snowflake disallows
    ``CREATE TEMPORARY FUNCTION`` from inside a stored procedure (even via
    ``EXECUTE IMMEDIATE``). The finalizer task drops it alongside the proc
    and tasks, so it does not outlive the backfill graph.

    Args:
        physical_name: Physical FV name (``<fv>$<ver>``).

    Returns:
        ``<physical_name>$BACKFILL_UDTF`` identifier.
    """
    return SqlIdentifier(
        f"{physical_name.resolved()}{_BACKFILL_UDTF_SUFFIX}",
        case_sensitive=True,
    )


def _get_backfill_task_name_pattern(physical_name: SqlIdentifier) -> str:
    """``NAME LIKE`` pattern matching every task in this FV's backfill graph.

    Pairs with ``ESCAPE '\\\\'``; the doubled backslashes produce the
    single-backslash escape Snowflake's parser expects, so the underscore
    matches literally rather than as the single-char wildcard. Matches root,
    finalizer, and any future ``BACKFILL_W<NNN>`` child tasks. Used by
    cleanup (which must find *every* backfill object including the
    finalizer) and the verbose ``get_refresh_history`` hint.

    Args:
        physical_name: Physical FV name (``<fv>$<ver>``).

    Returns:
        ``LIKE``-compatible pattern (with escaped underscore).
    """
    return f"{physical_name.resolved()}$BACKFILL\\\\_%"


def _get_user_visible_backfill_task_name_patterns(
    physical_name: SqlIdentifier,
) -> tuple[str, str]:
    """``NAME LIKE`` patterns for backfill tasks the user cares about.

    The finalizer is intentionally excluded — it is internal cleanup
    plumbing that has no user-meaningful refresh semantics.

    Args:
        physical_name: Physical FV name (``<fv>$<ver>``).

    Returns:
        ``(root_pattern, window_pattern)`` where ``root_pattern`` is an
        exact match for the Phase-1 root task and ``window_pattern`` is
        a prefix-match for Phase-2 ``$BACKFILL_W<NNN>`` child tasks
        (no rows yet in Phase 1; harmless to ``OR`` against).
    """
    base = physical_name.resolved()
    return (
        f"{base}{_BACKFILL_ROOT_SUFFIX}",
        f"{base}{_BACKFILL_WINDOW_PREFIX}%",
    )


def _drop_stale_backfill_graph_on_overwrite(
    *,
    session: Session,
    old_meta: StreamingMetadata,
    telemetry_stmp: dict[str, Any],
) -> None:
    """Best-effort teardown of a leftover backfill graph during overwrite.

    The root is suspended before any DROP because Snowflake disallows
    modifying a running task graph; suspending the root cascades to the
    finalize-bound child. Drops then proceed in dependency order
    (finalizer → root → proc → UDTF) so each child is gone before its
    parent. Every step swallows its exception: the typical case is that
    the previous finalizer already self-dropped everything and these are
    no-op ``IF EXISTS`` calls; if a real failure occurs we still want the
    rest of the registration flow to proceed.

    Args:
        session: Snowpark session used to issue the DDL.
        old_meta: Metadata of the FV being overwritten;
            ``None``-valued fields are skipped.
        telemetry_stmp: Statement parameters threaded onto every ``collect``.
    """

    def _safe(stmt: str, what: str) -> None:
        try:
            session.sql(stmt).collect(statement_params=telemetry_stmp)
        except Exception as e:
            logger.warning(f"Overwrite: failed to {what}: {e}")

    if old_meta.backfill_root_task_name:
        _safe(
            f"ALTER TASK IF EXISTS {old_meta.backfill_root_task_name} SUSPEND",
            f"suspend old backfill root task {old_meta.backfill_root_task_name}",
        )
    for task_name in (
        old_meta.backfill_finalize_task_name,
        old_meta.backfill_root_task_name,
    ):
        if task_name:
            _safe(f"DROP TASK IF EXISTS {task_name}", f"drop old backfill task {task_name}")
    if old_meta.backfill_proc_name:
        _safe(
            f"DROP PROCEDURE IF EXISTS {old_meta.backfill_proc_name}{_get_backfill_proc_signature()}",
            f"drop old backfill procedure {old_meta.backfill_proc_name}",
        )
    if old_meta.backfill_udtf_name and old_meta.backfill_udtf_signature:
        _safe(
            f"DROP FUNCTION IF EXISTS {old_meta.backfill_udtf_name}{old_meta.backfill_udtf_signature}",
            f"drop old backfill UDTF {old_meta.backfill_udtf_name}",
        )


@dataclass
class StreamingPreambleResult:
    """Outputs of the streaming preamble consumed by the rest of the registration flow."""

    fq_udf_table: str
    """Fully qualified udf_transformed table name (tracked for rollback)."""

    fq_backfill_table: str
    """Fully qualified backfill table name (OFT reads from this)."""

    resolved_source_name: str
    """Resolved stream source name (for ref_count + metadata)."""


def run_streaming_preamble(
    *,
    session: Session,
    feature_view: FeatureView,
    version: FeatureViewVersion,
    feature_view_name: SqlIdentifier,
    overwrite: bool,
    metadata_manager: FeatureStoreMetadataManager,
    telemetry_stmp: dict[str, Any],
    get_stream_source_fn: Callable[..., StreamSource],
    get_fully_qualified_name_fn: Callable[..., str],
) -> StreamingPreambleResult:
    """Run streaming-specific setup before the existing DT/OFT paths.

    Steps:
      1. On overwrite, decrement old stream source ref_count.
      2. Validate stream source exists.
      3. Probe — infer UDF output schema (10 rows, fast).
      4. Apply backfill_start_time filter if provided.
      5. Create empty udf_transformed and backfill tables (fast).

    After this returns, the caller should call
    ``feature_view._initialize_from_feature_df(session.table(fq_udf_table))``
    to complete the FeatureView initialization with the transformed schema.

    Args:
        session: Snowpark session.
        feature_view: The streaming FeatureView (must have ``stream_config``).
        version: Feature view version.
        feature_view_name: Physical name as SqlIdentifier.
        overwrite: Whether to overwrite existing objects.
        metadata_manager: Metadata manager for streaming metadata.
        telemetry_stmp: Telemetry statement parameters.
        get_stream_source_fn: Bound ``FeatureStore.get_stream_source``.
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.

    Returns:
        A ``StreamingPreambleResult`` with the udf_transformed table info.

    Raises:
        ValueError: If the backfill probe returns zero rows.
    """
    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    # 1. On overwrite, drop the old stream source ref and tear down any
    #    leftover backfill graph. Snowflake disallows modifying a running
    #    graph, so SUSPEND the root before DROPping. Usually a no-op because
    #    the previous finalizer self-drops on success.
    if overwrite:
        old_meta = metadata_manager.get_streaming_metadata(str(feature_view.name), str(version))
        if old_meta is not None:
            if old_meta.stream_source_name:
                metadata_manager.decrement_stream_source_ref_count(old_meta.stream_source_name)
            _drop_stale_backfill_graph_on_overwrite(
                session=session,
                old_meta=old_meta,
                telemetry_stmp=telemetry_stmp,
            )

    # 2. Resolve and validate stream source.
    raw_source_name = stream_config.get_stream_source_name()
    stream_source = get_stream_source_fn(raw_source_name)
    resolved_source_name = stream_source.name.resolved()

    # 3. Probe (10-row sample) to infer the output schema. Apply the
    #    backfill_start_time filter so the probe sees what the backfill will.
    backfill_df = stream_config.backfill_df
    if stream_config.backfill_start_time is not None and feature_view.timestamp_col is not None:
        from snowflake.snowpark import functions as F

        backfill_df = backfill_df.filter(
            F.col(feature_view.timestamp_col.resolved()) >= F.lit(stream_config.backfill_start_time)
        )
    # Reject TZ/LTZ and unsupported types now so the failure surfaces at
    # registration time, not as an "Invalid argument types" task failure.
    validate_schema_field_types(backfill_df.schema, context="backfill_df schema")
    validate_fs_columns_match(
        expected=_columns_from_struct_type(stream_source.schema),
        actual=_columns_from_struct_type(backfill_df.schema),
        expected_label=f"StreamSource '{stream_source.name}'",
        actual_label="backfill_df",
        error_prefix="streaming feature view",
    )
    sample_pdf = backfill_df.limit(10).to_pandas()
    if sample_pdf.empty:
        raise ValueError(
            "Backfill probe returned zero rows. Check that backfill_df has data "
            "and that backfill_start_time (if set) is not filtering out all rows."
        )
    probe_result = stream_config.transformation_fn(sample_pdf)
    udf_output_schema = _infer_structtype_from_pandas(probe_result)

    # 4. Create empty udf_transformed and backfill tables.
    udf_table_name = FeatureView._get_udf_transformed_table_name(feature_view_name)
    backfill_table_name = _get_backfill_table_name(udf_table_name)
    fq_udf_table = get_fully_qualified_name_fn(udf_table_name)
    fq_backfill_table = get_fully_qualified_name_fn(backfill_table_name)

    _create_empty_table(
        session=session,
        fq_table_name=fq_udf_table,
        schema=udf_output_schema,
        overwrite=overwrite,
        telemetry_stmp=telemetry_stmp,
    )
    _create_empty_table(
        session=session,
        fq_table_name=fq_backfill_table,
        schema=udf_output_schema,
        overwrite=overwrite,
        telemetry_stmp=telemetry_stmp,
    )

    return StreamingPreambleResult(
        fq_udf_table=fq_udf_table,
        fq_backfill_table=fq_backfill_table,
        resolved_source_name=resolved_source_name,
    )


# ---------------------------------------------------------------------------
# Postamble — runs after the existing DT/OFT creation paths
# ---------------------------------------------------------------------------


@dataclass
class StreamingPostambleResult:
    """Names of the backfill resources created by the postamble (for rollback tracking)."""

    fq_backfill_root_task: str
    """Fully-qualified backfill root task name."""

    fq_backfill_finalize_task: str
    """Fully-qualified backfill finalizer task name."""

    fq_backfill_proc: str
    """Fully-qualified backfill stored procedure name (no signature)."""

    fq_backfill_udtf: str
    """Fully-qualified backfill UDTF name (no signature)."""

    fq_backfill_udtf_signature: str
    """Argument-type signature of the backfill UDTF (e.g. ``"(VARCHAR, FLOAT)"``).

    Required for ``DROP FUNCTION`` since Snowflake disambiguates overloads by
    argument types. Persisted in ``StreamingMetadata`` for cleanup.
    """


def run_streaming_postamble(
    *,
    session: Session,
    feature_view: FeatureView,
    version: FeatureViewVersion,
    feature_view_name: SqlIdentifier,
    preamble: StreamingPreambleResult,
    metadata_manager: FeatureStoreMetadataManager,
    default_warehouse: Optional[SqlIdentifier],
    get_fully_qualified_name_fn: Callable[..., str],
    telemetry_stmp: dict[str, Any],
    on_resource_created: Optional[Callable[[str, str], None]] = None,
) -> StreamingPostambleResult:
    """Save streaming metadata, increment ref count, and kick off server-side backfill.

    Called after the DT and OFT are created. The backfill runs as a
    Snowflake Task graph (root + finalizer) so it survives client
    disconnects. The root fires every 10s, runs the INSERT ALL once
    (Snowflake's overlap-prevention blocks re-fires while the previous
    instance is still running), and the finalizer suspends and drops the
    entire graph the moment the root terminates. Failure detail stays in
    ``INFORMATION_SCHEMA.TASK_HISTORY`` for 7 days after the drop.

    Args:
        session: Snowpark session.
        feature_view: The streaming FeatureView.
        version: Feature view version.
        feature_view_name: Physical FV name (``<NAME>$<VERSION>``).
        preamble: Result from ``run_streaming_preamble``.
        metadata_manager: Metadata manager.
        default_warehouse: Feature Store default warehouse; used only when neither
            ``feature_view.initialization_warehouse`` nor ``feature_view.warehouse``
            is set (one of the three must be set).
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.
        telemetry_stmp: Telemetry statement parameters.
        on_resource_created: Optional callback invoked immediately after each
            backfill resource is created, with ``(kind, fq_name)``:

              - ``"BACKFILL_UDTF"`` — ``fq_name`` is ``"<fq_udtf><signature>"``
                so rollback can ``DROP FUNCTION`` without re-deriving types.
              - ``"BACKFILL_PROC"`` — fully-qualified proc name.
              - ``"BACKFILL_ROOT_TASK"`` / ``"BACKFILL_FINALIZE_TASK"`` —
                fully-qualified task names.

            Lets the caller track resources for rollback *as they are
            created*, so failures later in the postamble (metadata save,
            ref-count increment) still leave a complete cleanup trail. If
            ``None``, no tracking is
            performed and the caller is responsible for cleanup based
            on the returned ``StreamingPostambleResult``. Exceptions
            raised by the callback are propagated.

    Returns:
        A ``StreamingPostambleResult`` with the fully-qualified task names
        for rollback tracking.

    Raises:
        ValueError: If the feature view does not have a stream_config, or if
            no warehouse is available for the backfill task graph.
    """
    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    # The backfill is the streaming FV's one-time, full-scan initialization, so it
    # runs on the initialization warehouse when set, mirroring the dynamic table's
    # initial/reinit refresh. Falls back to the FV warehouse, then the FS default.
    task_warehouse = (
        feature_view.initialization_warehouse
        if feature_view.initialization_warehouse is not None
        else feature_view.warehouse
        if feature_view.warehouse is not None
        else default_warehouse
    )
    if task_warehouse is None:
        raise ValueError(
            "No warehouse available for streaming backfill task graph. Either set "
            "FeatureView.warehouse or configure a default warehouse on the FeatureStore."
        )

    # Resolve schemas for the per-FV UDTF + INSERT. We re-read the
    # udf_transformed table's schema (one DESCRIBE round-trip) rather than
    # reusing the pandas-probe schema: Snowflake canonicalizes types on
    # storage (e.g. ``VARCHAR`` -> ``VARCHAR(16777216)``) and the UDTF
    # ``RETURNS TABLE(...)`` and the ``INSERT ALL ... SELECT t."<col>"``
    # projection both need the canonicalized form to match. Probe-inferred
    # types have been observed to produce subtle mismatches that populate
    # the UDF table with the wrong column shape.
    backfill_df = stream_config.backfill_df
    udf_output_schema = session.table(preamble.fq_udf_table).schema
    input_schema = backfill_df.schema
    input_col_names = [SqlIdentifier(f.name).resolved() for f in input_schema.fields]
    input_col_types = [_snowpark_type_to_sql(f.datatype) for f in input_schema.fields]
    output_col_names = [SqlIdentifier(f.name).resolved() for f in udf_output_schema.fields]
    output_col_types = [_snowpark_type_to_sql(f.datatype) for f in udf_output_schema.fields]

    # Create the per-FV permanent UDTF. Permanent (not TEMPORARY) because
    # Snowflake disallows CREATE TEMPORARY FUNCTION inside a stored
    # procedure; the finalizer task drops it as part of cleanup so it does
    # not outlive the backfill graph.
    fq_udtf = get_fully_qualified_name_fn(_get_backfill_udtf_name(feature_view_name))
    udtf_sql = _render_backfill_udtf_sql(
        fq_udtf=fq_udtf,
        input_col_names=input_col_names,
        input_col_types=input_col_types,
        output_col_names=output_col_names,
        output_col_types=output_col_types,
        user_fn_name=stream_config.get_function_name(),
        user_fn_source=stream_config.get_function_source(),
    )
    session.sql(udtf_sql).collect(statement_params=telemetry_stmp)
    udtf_signature = _render_udtf_signature(input_col_types)
    if on_resource_created is not None:
        on_resource_created("BACKFILL_UDTF", f"{fq_udtf}{udtf_signature}")

    insert_all_sql = _render_backfill_insert_all_sql(
        fq_udf_table=preamble.fq_udf_table,
        fq_backfill_table=preamble.fq_backfill_table,
        fq_udtf=fq_udtf,
        backfill_source_select=backfill_df.queries["queries"][-1],
        input_col_names=input_col_names,
        input_col_types=input_col_types,
        output_col_names=output_col_names,
        timestamp_col=(feature_view.timestamp_col.resolved() if feature_view.timestamp_col is not None else None),
    )

    # Create the backfill stored procedure.
    fq_proc = _create_backfill_proc(
        session=session,
        feature_view_name=feature_view_name,
        insert_all_sql=insert_all_sql,
        get_fully_qualified_name_fn=get_fully_qualified_name_fn,
        telemetry_stmp=telemetry_stmp,
    )
    if on_resource_created is not None:
        on_resource_created("BACKFILL_PROC", fq_proc)

    # Create the backfill task graph; the root task CALLs the proc.
    # The graph is created in suspended state — we persist metadata first
    # (including ``backfill_state='RUNNING'``) so the finalizer can never
    # race ahead of the metadata row it needs to update.
    fv_name_str = str(feature_view.name)
    version_str = str(version)
    fq_root_task, fq_finalize_task = _create_backfill_task_graph(
        session=session,
        feature_view_name=feature_view_name,
        warehouse=task_warehouse,
        fq_proc=fq_proc,
        fq_udtf=fq_udtf,
        udtf_signature=udtf_signature,
        backfill_start_time=stream_config.backfill_start_time,
        get_fully_qualified_name_fn=get_fully_qualified_name_fn,
        telemetry_stmp=telemetry_stmp,
        metadata_table_path=metadata_manager.table_path,
        fv_metadata_name=fv_name_str,
        fv_metadata_version=version_str,
        on_resource_created=on_resource_created,
    )
    logger.info(
        f"Backfill task graph created for streaming FV {feature_view.name}/{version} "
        f"(udtf={fq_udtf}, proc={fq_proc}, root={fq_root_task}, finalize={fq_finalize_task})."
    )

    # Persist metadata (includes backfill task + proc + udtf names + state).
    backfill_start_str = (
        stream_config.backfill_start_time.isoformat() if stream_config.backfill_start_time is not None else None
    )

    metadata_manager.save_streaming_metadata(
        fv_name=fv_name_str,
        version=version_str,
        metadata=StreamingMetadata(
            stream_source_name=preamble.resolved_source_name,
            transformation_fn_name=stream_config.get_function_name(),
            transformation_fn_source=stream_config.get_function_source(),
            backfill_start_time=backfill_start_str,
            backfill_root_task_name=fq_root_task,
            backfill_finalize_task_name=fq_finalize_task,
            backfill_proc_name=fq_proc,
            backfill_udtf_name=fq_udtf,
            backfill_udtf_signature=udtf_signature,
            backfill_state=BACKFILL_STATE_RUNNING,
        ),
    )

    metadata_manager.increment_stream_source_ref_count(preamble.resolved_source_name)

    # Now safe to resume — the finalizer's terminal-state UPDATE will
    # find the row written above.
    _resume_backfill_task_graph(
        session=session,
        fq_root_task=fq_root_task,
        fq_finalize_task=fq_finalize_task,
        telemetry_stmp=telemetry_stmp,
    )

    return StreamingPostambleResult(
        fq_backfill_root_task=fq_root_task,
        fq_backfill_finalize_task=fq_finalize_task,
        fq_backfill_proc=fq_proc,
        fq_backfill_udtf=fq_udtf,
        fq_backfill_udtf_signature=udtf_signature,
    )


def _render_backfill_udtf_sql(
    *,
    fq_udtf: str,
    input_col_names: list[str],
    input_col_types: list[str],
    output_col_names: list[str],
    output_col_types: list[str],
    user_fn_name: str,
    user_fn_source: str,
) -> str:
    """Render a ``CREATE OR REPLACE FUNCTION`` for a permanent vectorized Python UDTF.

    The handler runs the user's ``transformation_fn`` per vectorized batch
    via the ``process`` method. Snowflake decides batch sizing and
    parallelizes execution across UDF servers; ``transformation_fn`` must
    be row-wise (no cross-row state, rolling windows, ranks, or lags).

    Permanent (not TEMPORARY) because Snowflake disallows
    ``CREATE TEMPORARY FUNCTION`` from inside a stored procedure (even via
    ``EXECUTE IMMEDIATE``); the finalizer task drops it during cleanup.

    Args:
        fq_udtf: Fully-qualified UDTF name (``<db>.<sch>.<name>``).
        input_col_names: Resolved input column identifiers.
        input_col_types: Snowflake SQL types matching ``input_col_names``.
        output_col_names: Resolved output column identifiers (match the
            ``$UDF_TRANSFORMED`` table).
        output_col_types: Snowflake SQL types matching ``output_col_names``.
        user_fn_name: ``__name__`` of the user's ``transformation_fn``.
        user_fn_source: Plain-text source of the function.

    Returns:
        SQL DDL string ending with ``;``.

    Raises:
        ValueError: On input/output length mismatch, or if ``user_fn_source``
            contains the reserved ``$$`` sentinel.
    """
    if len(input_col_names) != len(input_col_types):
        raise ValueError("input_col_names and input_col_types must have the same length.")
    if len(output_col_names) != len(output_col_types):
        raise ValueError("output_col_names and output_col_types must have the same length.")
    if _DOLLAR_QUOTE in user_fn_source:
        raise ValueError(
            f"transformation_fn source contains the reserved sentinel {_DOLLAR_QUOTE!r}; "
            "rename or restructure the function to remove it."
        )

    input_signature = ", ".join(f'"{n}" {t}' for n, t in zip(input_col_names, input_col_types))
    output_signature = ", ".join(f'"{n}" {t}' for n, t in zip(output_col_names, output_col_types))
    output_col_list_repr = ", ".join(repr(n) for n in output_col_names)

    # Dedent the user's source and emit it at the handler module's top
    # level. Substituting it into an indented f-string placeholder would
    # only indent the first line (Python's common-prefix dedent then sees
    # zero shared whitespace), producing an ``IndentationError`` when the
    # UDTF compiles.
    user_source_dedented = textwrap.dedent(user_fn_source).rstrip()

    handler_body = (
        "import pandas as pd\n"
        "\n"
        f"{user_source_dedented}\n"
        "\n"
        "class _Handler:\n"
        "    def process(self, df):\n"
        f"        result = {user_fn_name}(df)\n"
        f"        result = result[[{output_col_list_repr}]]\n"
        "        return result\n"
        "\n"
        "_Handler.process._sf_vectorized_input = pd.DataFrame\n"
    )

    return (
        f"CREATE OR REPLACE FUNCTION {fq_udtf}({input_signature})\n"
        f"  RETURNS TABLE({output_signature})\n"
        f"  LANGUAGE PYTHON\n"
        f"  RUNTIME_VERSION = '3.11'\n"
        f"  PACKAGES = ('pandas','numpy')\n"
        f"  HANDLER = '_Handler'\n"
        f"AS {_DOLLAR_QUOTE}\n"
        f"{handler_body}"
        f"{_DOLLAR_QUOTE};"
    )


def _render_backfill_insert_all_sql(
    *,
    fq_udf_table: str,
    fq_backfill_table: str,
    fq_udtf: str,
    backfill_source_select: str,
    input_col_names: list[str],
    input_col_types: list[str],
    output_col_names: list[str],
    timestamp_col: Optional[str],
) -> str:
    """Render the ``INSERT ALL ... SELECT`` body that runs inside the proc.

    Shape::

        INSERT ALL INTO <udf_t> INTO <bf_t>
        SELECT t."<o1>", t."<o2>", ...
        FROM (
                 SELECT * FROM (<source_select>)
                 [WHERE (:WINDOW_START IS NULL OR "<ts>" >= :WINDOW_START)
                    AND (:WINDOW_END   IS NULL OR "<ts>" <  :WINDOW_END)]
             ) AS s,
             TABLE(<fq_udtf>(s."<i1>"::<t1>, s."<i2>"::<t2>, ...)) AS t

    Each UDTF argument is cast to its declared parameter type. Snowflake's
    function resolver does not implicitly coerce precision-bearing types
    (e.g. ``TIMESTAMP_NTZ(0)`` -> ``TIMESTAMP_NTZ(9)``) for vectorized UDTF
    arguments, so without the cast a non-canonical-precision source column
    fails at task run time with ``Invalid argument types``.

    The window predicate is filtered *inside* the source subquery so
    fewer rows pass through the UDTF; an outer WHERE would still be
    correct here, but it would waste compute on rows the writer
    discards.

    The WHERE clause is omitted when ``timestamp_col`` is None.
    ``:WINDOW_START`` / ``:WINDOW_END`` are bind refs to the proc's
    parameters; Snowflake Scripting requires the ``:NAME`` form for proc
    parameters used inside SELECT/INSERT. ``LATERAL`` is omitted because
    the comma-join shape ``s, TABLE(udtf(s.col, ...))`` is the
    documented call form for vectorized ``process`` UDTFs and provides
    implicit lateral correlation.

    Args:
        fq_udf_table: Fully-qualified ``$UDF_TRANSFORMED`` table.
        fq_backfill_table: Fully-qualified ``$UDF_TRANSFORMED$BACKFILL`` table.
        fq_udtf: Fully-qualified per-FV UDTF.
        backfill_source_select: SELECT text for ``stream_config.backfill_df``,
            wrapped in ``SELECT * FROM (...)`` so the inner WHERE can be
            applied without parsing the user's query.
        input_col_names: Resolved input column identifiers.
        input_col_types: Snowflake SQL types matching ``input_col_names``;
            used to cast each UDTF arg to its declared parameter type.
        output_col_names: Resolved output column identifiers.
        timestamp_col: Resolved FV timestamp column, or ``None``.

    Returns:
        SQL string (no trailing semicolon).

    Raises:
        ValueError: If ``input_col_names`` and ``input_col_types`` differ in length.
    """
    if len(input_col_names) != len(input_col_types):
        raise ValueError("input_col_names and input_col_types must have the same length.")
    udtf_args = ", ".join(f's."{n}"::{t}' for n, t in zip(input_col_names, input_col_types))
    select_projection = ", ".join(f't."{n}"' for n in output_col_names)
    if timestamp_col is not None:
        # Alias the inner SELECT so the timestamp reference stays
        # unambiguous if ``backfill_source_select`` is ever changed to
        # emit a join.
        filtered_source = (
            f"SELECT * FROM ({backfill_source_select}) AS _bf"
            f' WHERE (:WINDOW_START IS NULL OR _bf."{timestamp_col}" >= :WINDOW_START)'
            f' AND (:WINDOW_END IS NULL OR _bf."{timestamp_col}" < :WINDOW_END)'
        )
    else:
        filtered_source = backfill_source_select
    return (
        f"INSERT ALL INTO {fq_udf_table} INTO {fq_backfill_table}"
        f" SELECT {select_projection}"
        f" FROM ({filtered_source}) AS s,"
        f" TABLE({fq_udtf}({udtf_args})) AS t"
    )


def _create_backfill_proc(
    *,
    session: Session,
    feature_view_name: SqlIdentifier,
    insert_all_sql: str,
    get_fully_qualified_name_fn: Callable[..., str],
    telemetry_stmp: dict[str, Any],
) -> str:
    """Create the backfill stored procedure that the root task will ``CALL``.

    The proc body wraps ``insert_all_sql`` in a SQL Scripting ``BEGIN ...
    EXCEPTION WHEN OTHER THEN RAISE; END;`` block so failures propagate to
    the task and surface in ``INFORMATION_SCHEMA.TASK_HISTORY``. The UDTF
    referenced by the SQL is created separately (Snowflake disallows
    ``CREATE TEMPORARY FUNCTION`` inside a stored procedure body).

    Args:
        session: Snowpark session.
        feature_view_name: Physical FV SqlIdentifier.
        insert_all_sql: ``INSERT ALL ... SELECT ...`` (no trailing ``;``)
            referencing the permanent UDTF by fully-qualified name.
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.
        telemetry_stmp: Telemetry statement parameters.

    Returns:
        Fully-qualified procedure name (without signature).

    Raises:
        ValueError: If ``insert_all_sql`` contains the reserved ``$$`` sentinel.
    """
    if _DOLLAR_QUOTE in insert_all_sql:
        raise ValueError(
            f"Backfill SQL contains the reserved proc-body sentinel {_DOLLAR_QUOTE!r}; "
            "this should not happen with the current renderers."
        )

    proc_name = _get_backfill_proc_name(feature_view_name)
    fq_proc = get_fully_qualified_name_fn(proc_name)

    proc_body = (
        f"CREATE OR REPLACE PROCEDURE {fq_proc}{_get_backfill_proc_param_list()}\n"
        f"  RETURNS STRING\n"
        f"  LANGUAGE SQL\n"
        f"  COMMENT = 'SnowML SFV backfill procedure'\n"
        f"  EXECUTE AS OWNER\n"
        f"AS {_DOLLAR_QUOTE}\n"
        f"BEGIN\n"
        f"  {insert_all_sql};\n"
        f"  RETURN 'OK';\n"
        f"EXCEPTION\n"
        f"  WHEN OTHER THEN\n"
        f"    RAISE;\n"
        f"END;\n"
        f"{_DOLLAR_QUOTE};"
    )
    session.sql(proc_body).collect(statement_params=telemetry_stmp)
    return fq_proc


def _render_backfill_state_update_sql(
    *,
    metadata_table_path: str,
    fv_name: str,
    version: str,
    new_state_sql_expr: str,
) -> str:
    """Render an ``UPDATE`` that flips ``backfill_state`` in the FV's STREAM_CONFIG row.

    Uses ``OBJECT_INSERT(... TRUE)`` to set-or-replace the single key without
    rewriting the rest of the JSON blob — atomic from the finalizer's point
    of view, no read-modify-write race against the client's ``RUNNING`` write.

    Args:
        metadata_table_path: Fully-qualified ``_FEATURE_STORE_METADATA`` path.
        fv_name: FV name as stored in ``OBJECT_NAME``.
        version: FV version as stored in ``VERSION``.
        new_state_sql_expr: Raw SQL expression for the new state. Pass a
            string literal (``"'COMPLETED'"``) for unconditional writes
            or a ``CASE`` expression for finalizer-style derivation from
            per-task ``TASK_HISTORY`` outcomes.

    Returns:
        Full ``UPDATE`` statement (no trailing semicolon).
    """
    safe_name = fv_name.replace("'", "''")
    safe_version = version.replace("'", "''")
    return (
        f"UPDATE {metadata_table_path}\n"
        f"  SET METADATA = OBJECT_INSERT(METADATA, 'backfill_state', {new_state_sql_expr}, TRUE),\n"
        f"      UPDATED_AT = CURRENT_TIMESTAMP()\n"
        f"  WHERE OBJECT_TYPE = 'FEATURE_VIEW'\n"
        f"    AND OBJECT_NAME = '{safe_name}'\n"
        f"    AND VERSION = '{safe_version}'\n"
        f"    AND METADATA_TYPE = 'STREAM_CONFIG'"
    )


def _create_backfill_task_graph(
    *,
    session: Session,
    feature_view_name: SqlIdentifier,
    warehouse: SqlIdentifier,
    fq_proc: str,
    fq_udtf: str,
    udtf_signature: str,
    backfill_start_time: Optional[datetime.datetime],
    get_fully_qualified_name_fn: Callable[..., str],
    telemetry_stmp: dict[str, Any],
    metadata_table_path: str,
    fv_metadata_name: str,
    fv_metadata_version: str,
    on_resource_created: Optional[Callable[[str, str], None]] = None,
) -> tuple[str, str]:
    """Create the root + finalizer task graph for a streaming-FV backfill.

    - root: schedules every 10s, body is a single ``CALL <fq_proc>(...)``;
      Snowflake's overlap prevention blocks re-runs while in flight.
    - finalizer: ``FINALIZE = <root>``, runs once the graph terminates
      (success or failure). Writes the terminal ``backfill_state`` to
      streaming metadata (``COMPLETED`` if every expected task in the
      graph succeeded, ``FAILED`` otherwise), then suspends the root and
      drops the proc, UDTF, and both tasks. Run history is retained in
      ``INFORMATION_SCHEMA.TASK_HISTORY`` for 7 days after the drop.

    The graph is **not resumed** here. The caller is expected to persist
    streaming metadata (with ``backfill_state='RUNNING'``) and only then
    resume the graph; otherwise the finalizer can race ahead of the
    client's metadata write and find no row to update.

    Phase 2 (max_backfill_interval) will reuse the proc + UDTF and add
    per-window child tasks that ``CALL`` the proc with different windows.
    The finalizer's ``COMPLETED`` check pattern-matches the FV's expected
    task name set in ``TASK_HISTORY``, so it extends to N windows without
    a SQL change.

    Args:
        session: Snowpark session.
        feature_view_name: Physical FV SqlIdentifier (``<fv>$<ver>``).
        warehouse: Warehouse used by both tasks.
        fq_proc: Fully-qualified backfill proc the root task ``CALL``s.
        fq_udtf: Fully-qualified per-FV UDTF the proc invokes; dropped by
            the finalizer.
        udtf_signature: Argument-type signature for ``DROP FUNCTION``
            (e.g. ``"(VARCHAR, FLOAT)"``).
        backfill_start_time: Lower-bound timestamp filter, or ``None`` for
            no lower bound (proc receives ``NULL``).
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.
        telemetry_stmp: Telemetry statement parameters.
        metadata_table_path: Fully-qualified path to ``_FEATURE_STORE_METADATA``,
            embedded in the finalizer's terminal-state ``UPDATE``.
        fv_metadata_name: FV name as stored in ``OBJECT_NAME``.
        fv_metadata_version: FV version as stored in ``VERSION``.
        on_resource_created: Optional callback for rollback tracking; see
            :func:`run_streaming_postamble` for the contract.

    Returns:
        ``(fq_root_task, fq_finalize_task)`` — fully-qualified names.
    """
    root_task_name = _get_backfill_root_task_name(feature_view_name)
    finalize_task_name = _get_backfill_finalize_task_name(feature_view_name)
    fq_root_task = get_fully_qualified_name_fn(root_task_name)
    fq_finalize_task = get_fully_qualified_name_fn(finalize_task_name)

    if backfill_start_time is not None:
        start_arg = f"TO_TIMESTAMP_NTZ('{backfill_start_time.isoformat()}')"
    else:
        start_arg = "NULL"

    # ROOT: schedule + single CALL of the proc.
    # ``SUSPEND_TASK_AFTER_NUM_FAILURES=1`` is a belt-and-braces safeguard
    # in case the finalizer never runs. The EXCEPTION re-raise keeps the
    # task in FAILED state so the finalizer still observes it and cleans up.
    # ``SYSDATE()`` returns UTC ``TIMESTAMP_NTZ`` directly; preferred over
    # ``CURRENT_TIMESTAMP()::TIMESTAMP_NTZ`` (which is TIMESTAMP_LTZ cast to
    # the session local wall clock and so depends on session/account TZ).
    create_root_sql = (
        f"CREATE OR REPLACE TASK {fq_root_task}\n"
        f"  WAREHOUSE = {warehouse}\n"
        f"  SCHEDULE = '10 SECONDS'\n"
        f"  USER_TASK_MINIMUM_TRIGGER_INTERVAL_IN_SECONDS = 10\n"
        f"  SUSPEND_TASK_AFTER_NUM_FAILURES = 1\n"
        f"  COMMENT = 'SnowML SFV backfill root (one-shot graph)'\n"
        f"AS\n"
        f"BEGIN\n"
        f"  CALL {fq_proc}({start_arg}, SYSDATE());\n"
        f"EXCEPTION\n"
        f"  WHEN OTHER THEN\n"
        f"    RAISE;\n"
        f"END;"
    )
    session.sql(create_root_sql).collect(statement_params=telemetry_stmp)
    if on_resource_created is not None:
        on_resource_created("BACKFILL_ROOT_TASK", fq_root_task)

    # FINALIZER: runs after the graph terminates. Each cleanup step gets
    # its own ``BEGIN..EXCEPTION WHEN OTHER THEN NULL; END;`` block so a
    # failure in one (e.g. the self-drop, or a transient permissions
    # hiccup) does not skip the rest. Order: write terminal state to
    # metadata, suspend root, drop proc, drop UDTF, drop root, drop
    # finalize (self) last so the self-drop cannot strand earlier work.
    #
    # Terminal-state derivation: ``COMPLETED`` if every user-visible
    # backfill task (``$BACKFILL_ROOT`` plus, in Phase 2, ``$BACKFILL_W%``)
    # has at least one ``SUCCEEDED`` row in ``TASK_HISTORY``;
    # ``FAILED`` otherwise. We require *every expected name* to have a
    # success row rather than counting successes, so a partially-completed
    # graph never claims success.
    proc_sig = _get_backfill_proc_signature()
    root_pattern, window_pattern = _get_user_visible_backfill_task_name_patterns(feature_view_name)

    def _safe(stmt: str) -> str:
        return f"  BEGIN {stmt} EXCEPTION WHEN OTHER THEN NULL; END;\n"

    # Counts of expected tasks vs successful tasks, both restricted to
    # the same FV's user-visible backfill graph. Phase 1: 1 expected
    # (the root). Phase 2: 1 + N (root + window children).
    terminal_state_case_sql = (
        "(\n"
        "    SELECT CASE WHEN expected = succeeded AND expected > 0 THEN 'COMPLETED' ELSE 'FAILED' END\n"
        "    FROM (\n"
        "      SELECT\n"
        "        COUNT(DISTINCT NAME) AS expected,\n"
        "        COUNT(DISTINCT IFF(STATE = 'SUCCEEDED', NAME, NULL)) AS succeeded\n"
        "      FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(RESULT_LIMIT => 1000))\n"
        f"      WHERE (NAME = '{root_pattern}' OR NAME LIKE '{window_pattern}')\n"
        "    )\n"
        "  )"
    )
    update_state_sql = _render_backfill_state_update_sql(
        metadata_table_path=metadata_table_path,
        fv_name=fv_metadata_name,
        version=fv_metadata_version,
        new_state_sql_expr=terminal_state_case_sql,
    )

    create_finalize_sql = (
        f"CREATE OR REPLACE TASK {fq_finalize_task}\n"
        f"  WAREHOUSE = {warehouse}\n"
        f"  FINALIZE = {fq_root_task}\n"
        f"  COMMENT = 'SnowML SFV backfill finalizer'\n"
        f"AS\n"
        f"BEGIN\n"
        f"{_safe(update_state_sql + ';')}"
        f"{_safe(f'ALTER TASK {fq_root_task} SUSPEND;')}"
        f"{_safe(f'DROP PROCEDURE IF EXISTS {fq_proc}{proc_sig};')}"
        f"{_safe(f'DROP FUNCTION IF EXISTS {fq_udtf}{udtf_signature};')}"
        f"{_safe(f'DROP TASK IF EXISTS {fq_root_task};')}"
        f"{_safe(f'DROP TASK IF EXISTS {fq_finalize_task};')}"
        f"END;"
    )
    session.sql(create_finalize_sql).collect(statement_params=telemetry_stmp)
    if on_resource_created is not None:
        on_resource_created("BACKFILL_FINALIZE_TASK", fq_finalize_task)

    return fq_root_task, fq_finalize_task


def _resume_backfill_task_graph(
    *,
    session: Session,
    fq_root_task: str,
    fq_finalize_task: str,
    telemetry_stmp: dict[str, Any],
) -> None:
    """Resume the backfill graph created by ``_create_backfill_task_graph``.

    Children must be resumed before the root (otherwise the root may fire
    and skip un-resumed members). Resume the finalizer explicitly, then
    let ``SYSTEM$TASK_DEPENDENTS_ENABLE`` walk the graph and resume the
    root. Caller must persist streaming metadata (``backfill_state =
    'RUNNING'``) before invoking this — once the root fires, the
    finalizer can race ahead of any later metadata writes.

    Args:
        session: Snowpark session.
        fq_root_task: Fully-qualified root task name.
        fq_finalize_task: Fully-qualified finalizer task name.
        telemetry_stmp: Telemetry statement parameters.
    """
    session.sql(f"ALTER TASK {fq_finalize_task} RESUME").collect(statement_params=telemetry_stmp)
    session.sql(f"SELECT SYSTEM$TASK_DEPENDENTS_ENABLE('{fq_root_task}')").collect(statement_params=telemetry_stmp)


# ---------------------------------------------------------------------------
# Cleanup (called from delete_feature_view)
# ---------------------------------------------------------------------------


def cleanup_streaming_feature_view(
    *,
    session: Session,
    feature_view_name: SqlIdentifier,
    version: str,
    fv_name: str,
    fv_metadata: _FeatureViewMetadata,
    metadata_manager: FeatureStoreMetadataManager,
    get_fully_qualified_name_fn: Callable[..., str],
    telemetry_stmp: dict[str, Any],
) -> None:
    """Clean up streaming FV resources during deletion.

    Drops the ``$UDF_TRANSFORMED`` and ``$UDF_TRANSFORMED$BACKFILL`` tables,
    decrements the stream source ref count, and best-effort drops any
    leftover backfill graph members. Normally a no-op (the finalizer has
    already dropped tasks and proc/UDTF) thanks to ``IF EXISTS``.

    Args:
        session: Snowpark session.
        feature_view_name: Physical FV SqlIdentifier.
        version: Feature view version string.
        fv_name: Feature view name (for metadata lookup).
        fv_metadata: Feature view metadata (for ``is_streaming`` check).
        metadata_manager: Metadata manager.
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.
        telemetry_stmp: Telemetry statement parameters.
    """
    streaming_meta = metadata_manager.get_streaming_metadata(fv_name, version)

    # Safety-net cleanup of the backfill graph. Only matters if the FV is
    # deleted before the backfill kicks off, or if the finalizer errored.
    # Suspend the root first (Snowflake requires a non-running graph
    # before ALTER/DROP), then drop finalizer before root.
    if streaming_meta is not None:
        if streaming_meta.backfill_root_task_name:
            try:
                session.sql(f"ALTER TASK IF EXISTS {streaming_meta.backfill_root_task_name} SUSPEND").collect(
                    statement_params=telemetry_stmp
                )
            except Exception as e:
                logger.warning(f"Failed to suspend backfill root task {streaming_meta.backfill_root_task_name}: {e}")
        for task_name in (
            streaming_meta.backfill_finalize_task_name,
            streaming_meta.backfill_root_task_name,
        ):
            if not task_name:
                continue
            try:
                session.sql(f"DROP TASK IF EXISTS {task_name}").collect(statement_params=telemetry_stmp)
                logger.info(f"Dropped backfill task {task_name}.")
            except Exception as e:
                logger.warning(f"Failed to drop backfill task {task_name}: {e}")
        if streaming_meta.backfill_proc_name:
            try:
                session.sql(
                    f"DROP PROCEDURE IF EXISTS {streaming_meta.backfill_proc_name}{_get_backfill_proc_signature()}"
                ).collect(statement_params=telemetry_stmp)
                logger.info(f"Dropped backfill procedure {streaming_meta.backfill_proc_name}.")
            except Exception as e:
                logger.warning(f"Failed to drop backfill procedure {streaming_meta.backfill_proc_name}: {e}")
        if streaming_meta.backfill_udtf_name and streaming_meta.backfill_udtf_signature:
            try:
                session.sql(
                    f"DROP FUNCTION IF EXISTS "
                    f"{streaming_meta.backfill_udtf_name}{streaming_meta.backfill_udtf_signature}"
                ).collect(statement_params=telemetry_stmp)
                logger.info(f"Dropped backfill UDTF {streaming_meta.backfill_udtf_name}.")
            except Exception as e:
                logger.warning(f"Failed to drop backfill UDTF {streaming_meta.backfill_udtf_name}: {e}")

    udf_table = FeatureView._get_udf_transformed_table_name(feature_view_name)
    backfill_table = _get_backfill_table_name(udf_table)
    fq_udf_table = get_fully_qualified_name_fn(udf_table)
    fq_backfill_table = get_fully_qualified_name_fn(backfill_table)
    for table_name in (fq_udf_table, fq_backfill_table):
        try:
            session.sql(f"DROP TABLE IF EXISTS {table_name}").collect(statement_params=telemetry_stmp)
            logger.info(f"Dropped table {table_name}.")
        except Exception as e:
            logger.warning(f"Failed to drop table {table_name}: {e}")

    if streaming_meta and streaming_meta.stream_source_name:
        try:
            metadata_manager.decrement_stream_source_ref_count(streaming_meta.stream_source_name)
            logger.info(f"Decremented ref_count for stream source {streaming_meta.stream_source_name}.")
        except Exception as e:
            logger.warning(f"Failed to decrement ref_count for stream source {streaming_meta.stream_source_name}: {e}")


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------


def _build_streaming_feature_view_spec(
    *,
    feature_view: FeatureView,
    feature_view_name: SqlIdentifier,
    version: str,
    target_lag: str,
    stream_source: StreamSource,
    udf_transformed_schema: StructType,
    database: str,
    schema: str,
    tiled_materialized_schema: Optional[StructType] = None,
) -> FeatureViewSpec:
    """Build a ``StreamingFeatureView`` spec for OFT creation.

    Called from ``FeatureStore._create_online_feature_table`` when
    ``feature_view.is_streaming`` is True.

    Handles both non-tiled streaming FVs (1 offline config) and tiled
    streaming FVs (2 offline configs: UDF_TRANSFORMED + TILED).

    Args:
        feature_view: The streaming FeatureView.
        feature_view_name: Physical name SqlIdentifier.
        version: Feature view version string.
        target_lag: Target lag for the OFT (e.g., ``"0 seconds"``).
        stream_source: Resolved ``StreamSource`` object.
        udf_transformed_schema: Schema of the udf_transformed table (raw
            columns after transformation, before any aggregation).
        database: Database name.
        schema: Schema name.
        tiled_materialized_schema: For tiled FVs, schema of the materialized tile
            dynamic table from ``Session.table(fq_dt).schema``. Required when
            ``feature_view.is_tiled``; must be ``None`` when not tiled.

    Returns:
        A validated ``FeatureViewSpec`` ready for serialization.

    Raises:
        ValueError: If the feature view does not have a stream_config.
    """
    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    entity_columns = list(feature_view.ordered_entity_columns)
    # GS dedupes on ``ordered_entity_column_names`` during ``$BACKFILL`` and
    # ignores ``ordered_secondary_key_column_names``; Quake strips SKs back
    # out before storing.
    if feature_view.aggregation_secondary_keys:
        for sk in feature_view.aggregation_secondary_keys:
            if sk not in entity_columns:
                entity_columns.append(sk)

    # UDF_TRANSFORMED offline config (always present)
    udf_table_name = FeatureView._get_udf_transformed_table_name(feature_view_name)
    udf_offline_config = SnowflakeTableInfo(
        table_type=TableType.UDF_TRANSFORMED,
        database=database,
        schema=schema,
        table=udf_table_name.resolved(),
        columns=udf_transformed_schema,
    )

    offline_configs: list[SnowflakeTableInfo] = [udf_offline_config]

    # For tiled streaming FVs, add a TILED offline config for the DT
    if feature_view.is_tiled:
        if tiled_materialized_schema is None:
            raise ValueError(
                "Tiled streaming feature view spec requires tiled_materialized_schema "
                "(Snowflake tile DT schema from Session.table(...).schema)."
            )
        tiled_offline_config = SnowflakeTableInfo(
            table_type=TableType.TILED,
            database=database,
            schema=schema,
            table=feature_view_name.resolved(),
            columns=tiled_materialized_schema,
        )
        offline_configs.append(tiled_offline_config)
    elif tiled_materialized_schema is not None:
        raise ValueError("Non-tiled streaming feature view must not set tiled_materialized_schema.")

    fn_source = stream_config.get_function_source()
    udf_output_cols = [(f.name, f.datatype) for f in udf_transformed_schema.fields]

    builder = (
        FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database=database,
            schema=schema,
            name=feature_view.name.resolved(),
            version=version,
        )
        .set_offline_configs(offline_configs)
        .set_sources([stream_source])
        .set_udf(
            name=stream_config.get_function_name(),
            engine="pandas",
            output_columns=udf_output_cols,
            function_definition=fn_source,
        )
        .set_properties(
            entity_columns=entity_columns,
            secondary_key_columns=feature_view.aggregation_secondary_keys,
            timestamp_field=feature_view.timestamp_col.resolved(),  # type: ignore[union-attr]
            granularity=feature_view.feature_granularity if feature_view.is_tiled else None,
            agg_method=feature_view.feature_aggregation_method if feature_view.is_tiled else None,
            target_lag=target_lag,
        )
    )

    if feature_view.is_tiled and feature_view.aggregation_specs:
        builder.set_features(feature_view.aggregation_specs)

    return builder.build()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_empty_table(
    *,
    session: Session,
    fq_table_name: str,
    schema: StructType,
    overwrite: bool,
    telemetry_stmp: dict[str, Any],
) -> None:
    """Create an empty table with the given schema."""
    overwrite_clause = "OR REPLACE " if overwrite else ""

    col_defs = []
    for field in schema.fields:
        col_defs.append(f'"{field.name}" {_snowpark_type_to_sql(field.datatype)}')
    col_defs_str = ", ".join(col_defs)

    query = f"CREATE {overwrite_clause}TABLE {fq_table_name} ({col_defs_str})"
    session.sql(query).collect(statement_params=telemetry_stmp)
