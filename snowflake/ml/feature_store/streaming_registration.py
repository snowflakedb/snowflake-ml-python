"""Streaming feature view registration helpers.

Extracted from ``FeatureStore`` to keep ``feature_store.py`` manageable.
Contains only the **streaming-specific** logic:

- **Preamble**: probe schema inference, empty udf_transformed table creation.
- **Postamble**: save metadata, increment ref_count, async backfill.
- **Spec builder**: ``_build_streaming_feature_view_spec`` for the OFT spec.
- **Cleanup**: drop udf_transformed table and decrement stream source ref count.

DT creation and OFT creation reuse the existing ``FeatureStore`` code paths.
After the preamble completes and ``_initialize_from_feature_df()`` is called,
``feature_view.query`` and ``feature_view.output_schema`` reflect the
udf_transformed table — so no overrides are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewVersion,
    _FeatureViewMetadata,
)
from snowflake.ml.feature_store.metadata_manager import (
    FeatureStoreMetadataManager,
    StreamingMetadata,
)
from snowflake.ml.feature_store.spec.builder import (
    FeatureViewSpecBuilder,
    SnowflakeTableInfo,
)
from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    TableType,
)
from snowflake.ml.feature_store.spec.models import FeatureViewSpec
from snowflake.ml.feature_store.stream_config import (
    _infer_structtype_from_pandas,
    _snowpark_type_to_sql,
)
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preamble — runs before the existing DT/OFT creation paths
# ---------------------------------------------------------------------------


_BACKFILL_TABLE_SUFFIX = "$BACKFILL"


def _get_backfill_table_name(udf_table_name: SqlIdentifier) -> SqlIdentifier:
    """Derive the backfill table name from a udf_transformed table name."""
    return SqlIdentifier(f"{udf_table_name.resolved()}{_BACKFILL_TABLE_SUFFIX}", case_sensitive=True)


@dataclass
class StreamingPreambleResult:
    """Result of the streaming preamble step.

    Provides the data that existing ``FeatureStore`` code paths need to
    create the DT and OFT for a streaming feature view.
    """

    fq_udf_table: str
    """Fully qualified name of the udf_transformed table (for rollback tracking)."""

    fq_backfill_table: str
    """Fully qualified name of the backfill table (OFT reads from this)."""

    resolved_source_name: str
    """Resolved (uppercased) stream source name (for ref_count and metadata)."""


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
    logging.warning("'StreamConfig' is in private preview since 1.8.5. Do not use it in production.")

    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    # 1. On overwrite, decrement the old stream source ref_count first
    if overwrite:
        old_meta = metadata_manager.get_streaming_metadata(str(feature_view.name), str(version))
        if old_meta and old_meta.stream_source_name:
            metadata_manager.decrement_stream_source_ref_count(old_meta.stream_source_name)

    # 2. Resolve and validate stream source
    raw_source_name = stream_config.get_stream_source_name()
    stream_source = get_stream_source_fn(raw_source_name)
    resolved_source_name = stream_source.name.resolved()

    # 3. Probe — infer output schema (fast, 10 rows)
    #    Apply backfill_start_time filter so the probe sees the same data as the backfill.
    backfill_df = stream_config.backfill_df
    if stream_config.backfill_start_time is not None and feature_view.timestamp_col is not None:
        from snowflake.snowpark import functions as F

        backfill_df = backfill_df.filter(
            F.col(feature_view.timestamp_col.resolved()) >= F.lit(stream_config.backfill_start_time)
        )
    sample_pdf = backfill_df.limit(10).to_pandas()
    if sample_pdf.empty:
        raise ValueError(
            "Backfill probe returned zero rows. Check that backfill_df has data "
            "and that backfill_start_time (if set) is not filtering out all rows."
        )
    probe_result = stream_config.transformation_fn(sample_pdf)
    udf_output_schema = _infer_structtype_from_pandas(probe_result)

    # 4. Create empty udf_transformed table and backfill table
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


def run_streaming_postamble(
    *,
    session: Session,
    feature_view: FeatureView,
    version: FeatureViewVersion,
    preamble: StreamingPreambleResult,
    metadata_manager: FeatureStoreMetadataManager,
) -> None:
    """Save streaming metadata, increment ref count, and kick off async backfill.

    Called after the DT and OFT have been successfully created.

    Args:
        session: Snowpark session.
        feature_view: The streaming FeatureView.
        version: Feature view version.
        preamble: Result from ``run_streaming_preamble``.
        metadata_manager: Metadata manager.

    Raises:
        ValueError: If the feature view does not have a stream_config.
    """
    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    # --- Kick off async backfill first (to get query_id for metadata) ---
    from snowflake.snowpark.dataframe import map_in_pandas

    # Apply backfill_start_time filter if provided
    backfill_df = stream_config.backfill_df
    if stream_config.backfill_start_time is not None and feature_view.timestamp_col is not None:
        from snowflake.snowpark import functions as F

        backfill_df = backfill_df.filter(
            F.col(feature_view.timestamp_col.resolved()) >= F.lit(stream_config.backfill_start_time)
        )

    udf_output_schema = session.table(preamble.fq_udf_table).schema

    # map_in_pandas expects Iterator[pd.DataFrame] -> Iterator[pd.DataFrame].
    # Closure captures user_fn so Snowpark can serialize it as a UDTF.
    user_fn = stream_config.transformation_fn

    def _batched_fn(iterator):  # type: ignore[no-untyped-def]
        for pdf in iterator:
            yield user_fn(pdf)

    transformed_df = map_in_pandas(
        backfill_df,
        _batched_fn,
        udf_output_schema,
        packages=["pandas", "numpy"],
    )

    # Use INSERT ALL to write UDF results to both udf_transformed and backfill
    # tables in a single computation. The UDF runs once; both tables are populated
    # from the same SELECT in one async job.
    select_sql = transformed_df.queries["queries"][-1]
    insert_all_sql = f"INSERT ALL INTO {preamble.fq_udf_table} INTO {preamble.fq_backfill_table} {select_sql}"
    # collect(block=False) returns AsyncJob but Snowpark stubs type it as list[Row]
    async_job: Any = session.sql(insert_all_sql).collect(block=False)
    logger.info(
        f"Async backfill started for streaming FV {feature_view.name}/{version} (query_id={async_job.query_id})."
    )

    # --- Save streaming metadata (includes backfill query_id) ---
    backfill_start_str = (
        stream_config.backfill_start_time.isoformat() if stream_config.backfill_start_time is not None else None
    )

    metadata_manager.save_streaming_metadata(
        fv_name=str(feature_view.name),
        version=str(version),
        metadata=StreamingMetadata(
            stream_source_name=preamble.resolved_source_name,
            transformation_fn_name=stream_config.get_function_name(),
            transformation_fn_source=stream_config.get_function_source(),
            backfill_start_time=backfill_start_str,
            backfill_query_id=async_job.query_id,
        ),
    )

    # --- Increment stream source ref count ---
    metadata_manager.increment_stream_source_ref_count(preamble.resolved_source_name)


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

    Drops the ``$UDF_TRANSFORMED`` and ``$UDF_TRANSFORMED$BACKFILL`` tables
    and decrements the stream source ref count.
    Called from ``FeatureStore.delete_feature_view()``.

    Args:
        session: Snowpark session.
        feature_view_name: Physical name SqlIdentifier of the feature view.
        version: Feature view version string.
        fv_name: Feature view name (for metadata lookup).
        fv_metadata: Feature view metadata (for ``is_streaming`` check).
        metadata_manager: Metadata manager.
        get_fully_qualified_name_fn: Bound ``FeatureStore._get_fully_qualified_name``.
        telemetry_stmp: Telemetry statement parameters.
    """
    # Drop udf_transformed and backfill tables
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

    # Decrement stream source ref count
    streaming_meta = metadata_manager.get_streaming_metadata(fv_name, version)
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

    Returns:
        A validated ``FeatureViewSpec`` ready for serialization.

    Raises:
        ValueError: If the feature view does not have a stream_config.
    """
    stream_config = feature_view.stream_config
    if stream_config is None:
        raise ValueError(f"FeatureView '{feature_view.name}' does not have a stream_config.")

    entity_columns = feature_view.ordered_entity_columns

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
        tiled_offline_config = SnowflakeTableInfo(
            table_type=TableType.TILED,
            database=database,
            schema=schema,
            table=feature_view_name.resolved(),
            columns=feature_view.output_schema,
        )
        offline_configs.append(tiled_offline_config)

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
            timestamp_field=feature_view.timestamp_col.resolved(),  # type: ignore[union-attr]
            granularity=feature_view.feature_granularity if feature_view.is_tiled else None,
            agg_method=FeatureAggregationMethod.TILES if feature_view.is_tiled else None,
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
