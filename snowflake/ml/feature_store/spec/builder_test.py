"""Comprehensive unit tests for spec.builder (FeatureViewSpecBuilder).

Covers:
- Streaming feature view (tiles, continuous, no aggregation)
- Batch feature view (tiles, no aggregation)
- Realtime feature view
- Source conversion (StreamSource, RequestSource, FeatureView, FeatureViewSlice, raw Source)
- Feature resolution
- UDF plain-text storage and $$ safety via to_json()
- Validation rules (all positive and negative paths)
- omitempty serialization
- Method chaining
"""

from __future__ import annotations

import json
from typing import Optional, TypedDict
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.spec.builder import (
    BatchSource,
    FeatureViewSpecBuilder,
    SnowflakeTableInfo,
)
from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    SourceType,
    StoreType,
    TableType,
)
from snowflake.ml.feature_store.spec.models import FSColumn, Source
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import (
    BooleanType,
    DataType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _txn_schema() -> StructType:
    """Reusable transaction stream schema."""
    return StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("AMOUNT", DoubleType()),
            StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
        ]
    )


def _make_stream_source() -> StreamSource:
    return StreamSource("txn_events", _txn_schema())


def _udf_transformed_config(schema: str = "SCH") -> SnowflakeTableInfo:
    return SnowflakeTableInfo(
        table_type=TableType.UDF_TRANSFORMED,
        database="DB",
        schema=schema,
        table="UDF_TBL",
        columns=StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
            ]
        ),
    )


def _tiled_config(schema: str = "SCH") -> SnowflakeTableInfo:
    return SnowflakeTableInfo(
        table_type=TableType.TILED,
        database="DB",
        schema=schema,
        table="TILED_TBL",
        columns=StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT_SUM_24H", DoubleType()),
            ]
        ),
    )


def _batch_source_config() -> SnowflakeTableInfo:
    return SnowflakeTableInfo(
        table_type=TableType.BATCH_SOURCE,
        database="DB",
        schema="SCH",
        table="BATCH_TBL",
        columns=StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
            ]
        ),
    )


def _tiled_dt_schema_for_supported_pg_agg(agg_type: AggregationType) -> StructType:
    """Minimal tiled DT schema matching ``TilingSqlGenerator`` partial columns for *agg_type* on AMOUNT."""
    keys = [
        StructField("USER_ID", StringType()),
        StructField("TILE_START", TimestampType()),
    ]
    if agg_type == AggregationType.SUM:
        extra = [StructField("_PARTIAL_SUM_AMOUNT", DoubleType())]
    elif agg_type == AggregationType.COUNT:
        extra = [StructField("_PARTIAL_COUNT_AMOUNT", LongType())]
    elif agg_type == AggregationType.MIN:
        extra = [StructField("_PARTIAL_MIN_AMOUNT", DoubleType())]
    elif agg_type == AggregationType.MAX:
        extra = [StructField("_PARTIAL_MAX_AMOUNT", DoubleType())]
    elif agg_type == AggregationType.AVG:
        extra = [
            StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
            StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
        ]
    elif agg_type in (AggregationType.STD, AggregationType.VAR):
        extra = [
            StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
            StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
            StructField("_PARTIAL_SUM_SQ_AMOUNT", DoubleType()),
        ]
    else:
        raise ValueError(f"unsupported agg_type for PG tiled schema helper: {agg_type}")
    return StructType(keys + extra)


def _tiled_batch_offline_config_amt_sum() -> SnowflakeTableInfo:
    """Batch FV with tiles: materialized DT schema; table_type is still BatchSource."""
    return SnowflakeTableInfo(
        table_type=TableType.BATCH_SOURCE,
        database="DB",
        schema="SCH",
        table="TILED_TBL",
        columns=_tiled_dt_schema_for_supported_pg_agg(AggregationType.SUM),
    )


class _UdfArgs(TypedDict):
    name: str
    engine: str
    output_columns: list[tuple[str, DataType]]
    function_definition: str


def _simple_udf_args() -> _UdfArgs:
    return {
        "name": "transform_fn",
        "engine": "pandas",
        "output_columns": [("AMOUNT", DoubleType())],
        "function_definition": "def transform_fn(request): return request",
    }


def _request_source() -> RequestSource:
    return RequestSource(
        schema=StructType(
            [
                StructField("TXN_AMOUNT", DoubleType()),
                StructField("MERCHANT_ID", StringType()),
            ]
        )
    )


def _mock_feature_view(
    name: str = "upstream_fv",
    version: str = "v1",
    feature_names: Optional[list[str]] = None,
    *,
    is_streaming: bool = True,
    entity_columns: Optional[list[str]] = None,
) -> mock.MagicMock:
    """Create a mock FeatureView for source conversion tests.

    Uses spec= to ensure isinstance checks pass in set_sources.

    ``is_streaming`` is set explicitly (rather than left to MagicMock's default
    truthy-mock behavior) so the builder's ``_kind_of_fv`` dispatcher returns
    a deterministic ``FeatureViewKind`` and tests intending a Batch upstream
    can opt in via ``is_streaming=False``.

    ``entity_columns`` is exposed via ``ordered_entity_columns`` so the
    builder can capture it in ``set_sources`` for RTFV/FG derivation of
    ``ordered_entity_column_names``.
    """
    from snowflake.ml.feature_store.feature_view import FeatureView

    fv = mock.MagicMock(spec=FeatureView)
    fv.name = name
    fv.version = version
    fv.is_streaming = is_streaming
    names = feature_names or ["SCORE", "RISK"]
    fv.feature_names = names
    fv.output_schema = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("SCORE", DoubleType()),
            StructField("RISK", DoubleType()),
        ]
    )
    fv.ordered_entity_columns = list(entity_columns) if entity_columns is not None else ["USER_ID"]
    return fv


def _mock_feature_view_slice(
    selected_features: Optional[list[str]] = None,
    *,
    entity_columns: Optional[list[str]] = None,
) -> mock.MagicMock:
    """Create a mock FeatureViewSlice for source conversion tests.

    Uses spec= to ensure isinstance checks pass in set_sources.
    """
    from snowflake.ml.feature_store.feature_view import FeatureViewSlice

    fvs = mock.MagicMock(spec=FeatureViewSlice)
    fv = _mock_feature_view(entity_columns=entity_columns)
    fvs.feature_view_ref = fv
    fvs.names = selected_features or ["SCORE"]
    return fvs


def _seed_upstream_kind(
    builder: FeatureViewSpecBuilder,
    *,
    name: str,
    version: Optional[str],
    kind: FeatureViewKind,
) -> None:
    """Stamp an upstream FeatureViewKind on *builder* for validation tests.

    Centralizes access to ``builder._source_kinds`` so individual tests don't
    couple themselves to that internal field name. Used to simulate RTFV / FG
    upstreams while user-facing FeatureView classes for those kinds don't
    exist yet.
    """
    builder._source_kinds[(name, version)] = kind


def _seed_derived_entity_columns(
    builder: FeatureViewSpecBuilder,
    *,
    entity_columns: list[str],
) -> None:
    """Stamp the precomputed entity-columns union on *builder* for tests.

    Used when a test injects pre-built ``Source`` objects directly (bypassing
    :meth:`FeatureViewSpecBuilder.set_sources`) and therefore can't rely on
    ``_append_derived_entity_columns`` to populate the list. Mirrors what
    ``set_sources`` would compute from the upstream FV inputs.
    """
    builder._derived_entity_columns = list(entity_columns)


# ============================================================================
# Streaming Feature View Tests
# ============================================================================


class StreamingBuilderTilesTest(absltest.TestCase):
    """Streaming FV with tiles aggregation — 2 offline configs."""

    def test_build_tiles(self) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="txn_fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
                target_lag="30s",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.SUM,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_SUM_24H",
                    )
                ]
            )
            .build()
        )

        # Root-level checks
        self.assertEqual(result.kind, FeatureViewKind.StreamingFeatureView)
        self.assertEqual(result.online_store_type, StoreType.POSTGRES)

        # Metadata
        meta = result.metadata
        self.assertEqual(meta.database, "DB")
        self.assertEqual(meta.schema_, "SCH")
        self.assertEqual(meta.name, "txn_fv")
        self.assertEqual(meta.version, "v1")
        self.assertEqual(meta.spec_format_version, "1")
        self.assertEqual(meta.internal_data_version, "1")
        self.assertIsNotNone(meta.client_version)

        # Offline configs
        self.assertEqual(len(result.offline_configs), 2)
        types = {c.table_type for c in result.offline_configs}
        self.assertEqual(types, {TableType.UDF_TRANSFORMED, TableType.TILED})

        # Spec
        spec = result.spec
        self.assertEqual(spec.ordered_entity_column_names, ["USER_ID"])
        self.assertEqual(spec.timestamp_field, "EVENT_TIME")
        self.assertEqual(spec.feature_granularity_sec, 3600)
        self.assertEqual(spec.feature_aggregation_method, FeatureAggregationMethod.TILES)
        self.assertEqual(spec.target_lag_sec, 30)

        # Sources
        self.assertEqual(len(spec.sources), 1)
        src = spec.sources[0]
        self.assertEqual(src.source_type, SourceType.STREAM)
        self.assertEqual(len(src.columns), 3)

        # UDF — plain text (no base64)
        udf = spec.udf
        self.assertIsNotNone(udf)
        assert udf is not None
        self.assertEqual(udf.name, "transform_fn")
        self.assertEqual(udf.function_definition, "def transform_fn(request): return request")

        # Features
        self.assertEqual(len(spec.features), 1)
        feat = spec.features[0]
        self.assertEqual(feat.source_column.name, "AMOUNT")
        self.assertEqual(feat.function, "sum")
        self.assertEqual(feat.window_sec, 86400)

    def test_continuous_mode(self) -> None:
        """Continuous aggregation also requires 2 offline configs."""
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="cont_fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.CONTINUOUS,
                target_lag="30s",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .build()
        )
        self.assertEqual(result.spec.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)
        self.assertEqual(len(result.offline_configs), 2)


class StreamingBuilderNoAggTest(absltest.TestCase):
    """Streaming FV with no aggregation — 1 offline config (UDFTransformed only)."""

    def test_build_no_agg(self) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="raw_stream",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .build()
        )
        self.assertIsNone(result.spec.feature_granularity_sec)
        self.assertIsNone(result.spec.feature_aggregation_method)
        self.assertEqual(len(result.offline_configs), 1)

    def test_passthrough_features_from_udf_transformed(self) -> None:
        """Non-time-aggregated streaming FV generates passthrough features from UDF_TRANSFORMED columns."""
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="raw_stream",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .build()
        )
        # Passthrough features auto-generated from UDF_TRANSFORMED columns
        # USER_ID is entity (excluded), AMOUNT is the passthrough feature
        self.assertEqual(len(result.spec.features), 1)
        feat = result.spec.features[0]
        self.assertEqual(feat.source_column.name, "AMOUNT")
        self.assertEqual(feat.output_column.name, "AMOUNT")
        self.assertIsNone(feat.function)
        self.assertIsNone(feat.window_sec)

    def test_passthrough_excludes_entity_and_timestamp(self) -> None:
        """Passthrough features exclude entity columns and timestamp field."""
        # UDF_TRANSFORMED has USER_ID and AMOUNT; entity=USER_ID, timestamp=EVENT_TIME
        # Only AMOUNT should be a passthrough feature (EVENT_TIME not in the config columns)
        udf_config = SnowflakeTableInfo(
            table_type=TableType.UDF_TRANSFORMED,
            database="DB",
            schema="SCH",
            table="UDF_TBL",
            columns=StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("EVENT_TIME", TimestampType()),
                    StructField("AMOUNT", DoubleType()),
                    StructField("IS_LARGE", BooleanType()),
                ]
            ),
        )
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="stream_fv",
                version="v1",
            )
            .set_offline_configs([udf_config])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .build()
        )
        feature_names = [f.source_column.name for f in result.spec.features]
        self.assertNotIn("USER_ID", feature_names)  # entity excluded
        self.assertNotIn("EVENT_TIME", feature_names)  # timestamp excluded
        self.assertIn("AMOUNT", feature_names)
        self.assertIn("IS_LARGE", feature_names)
        self.assertEqual(len(result.spec.features), 2)


# ============================================================================
# Batch Feature View Tests
# ============================================================================


class BatchBuilderTest(absltest.TestCase):
    """Batch feature view tests."""

    def test_build_batch_no_agg(self) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="batch_fv",
                version="v1",
            )
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .build()
        )
        self.assertEqual(result.kind, FeatureViewKind.BatchFeatureView)
        self.assertEqual(len(result.offline_configs), 1)
        self.assertEqual(result.offline_configs[0].table_type, TableType.BATCH_SOURCE)
        self.assertIsNone(result.spec.feature_aggregation_method)
        # Non-tiled batch: no sources, no UDF
        self.assertEqual(result.spec.sources, [])
        self.assertIsNone(result.spec.udf)
        # Passthrough features auto-generated from BATCH_SOURCE columns (entity excluded)
        self.assertEqual(len(result.spec.features), 1)
        feat = result.spec.features[0]
        self.assertEqual(feat.source_column.name, "AMOUNT")
        self.assertEqual(feat.output_column.name, "AMOUNT")
        self.assertIsNone(feat.function)
        self.assertIsNone(feat.window_sec)

    def test_build_batch_no_agg_excludes_entity_and_timestamp(self) -> None:
        """Non-tiled batch: entity and timestamp columns are excluded from passthrough features."""
        config = SnowflakeTableInfo(
            table_type=TableType.BATCH_SOURCE,
            database="DB",
            schema="SCH",
            table="BATCH_TBL",
            columns=StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("EVENT_TIME", TimestampType()),
                    StructField("AMOUNT", DoubleType()),
                    StructField("SCORE", DoubleType()),
                ]
            ),
        )
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="batch_fv_ts",
                version="v1",
            )
            .set_offline_configs([config])
            .set_properties(entity_columns=["USER_ID"], timestamp_field="EVENT_TIME")
            .build()
        )
        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["AMOUNT", "SCORE"])
        for feat in result.spec.features:
            self.assertEqual(feat.source_column, feat.output_column)
            self.assertIsNone(feat.function)
            self.assertIsNone(feat.window_sec)

    def test_build_batch_with_tiles(self) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="batch_tiled",
                version="v1",
            )
            .set_offline_configs([_tiled_batch_offline_config_amt_sum()])
            .set_properties(
                entity_columns=["USER_ID"],
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources(
                [
                    BatchSource(
                        schema=StructType(
                            [
                                StructField("USER_ID", StringType()),
                                StructField("AMOUNT", DoubleType()),
                            ]
                        )
                    )
                ]
            )
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.SUM,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_SUM_24H",
                    )
                ]
            )
            .build()
        )
        self.assertEqual(result.spec.feature_aggregation_method, FeatureAggregationMethod.TILES)
        self.assertEqual(result.spec.feature_granularity_sec, 3600)
        self.assertEqual(len(result.spec.features), 1)
        self.assertEqual(result.offline_configs[0].table_type, TableType.BATCH_SOURCE)
        # BATCH source is builder-internal — must not appear in output
        self.assertEqual(result.spec.sources, [])


# ============================================================================
# Realtime Feature View Tests
# ============================================================================


class RealtimeBuilderTest(absltest.TestCase):
    """Realtime feature view tests."""

    @mock.patch(
        "snowflake.ml.feature_store.spec.builder.FeatureViewSpecBuilder._convert_feature_view",
        side_effect=lambda fv: Source(
            name=str(fv.name),
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version=str(fv.version),
        ),
    )
    def test_build_realtime(self, _mock_convert: mock.MagicMock) -> None:
        mock_fv = _mock_feature_view()
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.RealtimeFeatureView,
                database="DB",
                schema="SCH",
                name="realtime_fv",
                version="v1",
            )
            .set_sources([_request_source(), mock_fv])
            .set_udf(
                name="score_fn",
                engine="pandas",
                output_columns=[("RISK_SCORE", DoubleType())],
                function_definition="def score_fn(txn, features): return 0.5",
            )
            .build()
        )
        self.assertEqual(result.kind, FeatureViewKind.RealtimeFeatureView)
        self.assertEqual(len(result.offline_configs), 0)

        source_types = [s.source_type for s in result.spec.sources]
        self.assertIn(SourceType.REQUEST, source_types)
        self.assertIn(SourceType.FEATURES, source_types)
        self.assertIsNotNone(result.spec.udf)
        self.assertIsNone(result.spec.feature_aggregation_method)

        # Passthrough features are derived from the UDF's output_columns,
        # NOT from the FV source (which exposes SCORE, not RISK_SCORE).
        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["RISK_SCORE"])
        self.assertNotIn("SCORE", feature_names)
        feat = result.spec.features[0]
        self.assertEqual(feat.source_column.name, "RISK_SCORE")
        self.assertEqual(feat.output_column.name, "RISK_SCORE")
        self.assertEqual(feat.source_column, feat.output_column)
        self.assertEqual(feat.source_column.type, "DoubleType")
        self.assertIsNone(feat.function)
        self.assertIsNone(feat.window_sec)


# ============================================================================
# Source Conversion Tests
# ============================================================================


class SourceConversionTest(absltest.TestCase):
    """Tests for set_sources with different source types."""

    def test_stream_source_conversion(self) -> None:
        ss = _make_stream_source()
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_sources([ss])
        self.assertEqual(len(builder._sources), 1)
        src = builder._sources[0]
        self.assertEqual(src.source_type, SourceType.STREAM)
        self.assertEqual(src.name, "TXN_EVENTS")
        self.assertEqual(len(src.columns), 3)

    def test_request_source_conversion(self) -> None:
        rs = _request_source()
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_sources([rs])
        src = builder._sources[0]
        self.assertEqual(src.source_type, SourceType.REQUEST)
        self.assertEqual(src.name, "request")
        self.assertEqual(len(src.columns), 2)

    @mock.patch(
        "snowflake.ml.feature_store.spec.builder.FeatureViewSpecBuilder._convert_feature_view",
    )
    def test_feature_view_conversion(self, mock_convert: mock.MagicMock) -> None:
        mock_convert.return_value = Source(
            name="upstream_fv",
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version="v1",
        )
        fv = _mock_feature_view()
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_sources([fv])
        src = builder._sources[0]
        self.assertEqual(src.source_type, SourceType.FEATURES)
        self.assertEqual(src.name, "upstream_fv")
        mock_convert.assert_called_once_with(fv)

    @mock.patch(
        "snowflake.ml.feature_store.spec.builder.FeatureViewSpecBuilder._convert_feature_view_slice",
    )
    def test_feature_view_slice_conversion(self, mock_convert: mock.MagicMock) -> None:
        mock_convert.return_value = Source(
            name="upstream_fv",
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version="v1",
        )
        fvs = _mock_feature_view_slice(["SCORE"])
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_sources([fvs])
        src = builder._sources[0]
        self.assertEqual([c.name for c in src.columns], ["SCORE"])

    def test_unsupported_source_type(self) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported source type"):
            builder.set_sources(["not_a_source"])  # type: ignore[list-item]

    def test_batch_source_creates_batch_source_object(self) -> None:
        """BatchSource is converted to a Source with source_type=BATCH."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
            ]
        )
        builder.set_sources([BatchSource(schema=schema)])
        self.assertEqual(len(builder._sources), 1)
        src = builder._sources[0]
        self.assertEqual(src.source_type, SourceType.BATCH)
        self.assertEqual(src.name, "batch")
        self.assertEqual(len(src.columns), 2)
        self.assertEqual(src.columns[0].name, "USER_ID")
        self.assertEqual(src.columns[1].name, "AMOUNT")


# ============================================================================
# FeatureView / FeatureViewSlice Conversion Tests (real static methods)
# ============================================================================


class FeatureViewConversionTest(absltest.TestCase):
    """Tests for the real _convert_feature_view, _convert_feature_view_slice,
    and _columns_from_feature_view static methods — using properly-typed
    SqlIdentifier attributes instead of mocking the conversion away.
    """

    @staticmethod
    def _make_typed_fv(
        name: str = "upstream_fv",
        version: str = "v1",
        feature_names: Optional[list[str]] = None,
        output_schema: Optional[StructType] = None,
    ) -> mock.MagicMock:
        """Create a mock FeatureView with SqlIdentifier names and real StructType."""
        from snowflake.ml.feature_store.feature_view import FeatureView

        fv = mock.MagicMock(spec=FeatureView)
        fv.name = SqlIdentifier(name)
        fv.version = version
        fv.feature_names = [SqlIdentifier(n) for n in (feature_names or ["SCORE", "RISK"])]
        fv.output_schema = output_schema or StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("SCORE", DoubleType()),
                StructField("RISK", DecimalType(10, 2)),
            ]
        )
        return fv

    # -- _columns_from_feature_view ----------------------------------------

    def test_columns_from_feature_view_filters_to_features(self) -> None:
        """Only columns whose name appears in feature_names are returned."""
        fv = self._make_typed_fv()
        cols = FeatureViewSpecBuilder._columns_from_feature_view(fv)
        col_names = [c.name for c in cols]
        # USER_ID is not a feature → excluded
        self.assertEqual(col_names, ["SCORE", "RISK"])

    def test_columns_from_feature_view_preserves_types(self) -> None:
        """FSColumn type metadata (precision, scale) is preserved."""
        fv = self._make_typed_fv()
        cols = FeatureViewSpecBuilder._columns_from_feature_view(fv)
        self.assertEqual(cols[0].type, "DoubleType")
        self.assertEqual(cols[1].type, "DecimalType")
        self.assertEqual(cols[1].precision, 10)
        self.assertEqual(cols[1].scale, 2)

    # -- _convert_feature_view ---------------------------------------------

    def test_convert_feature_view_basic(self) -> None:
        """Full Source produced from an unquoted FeatureView name."""
        fv = self._make_typed_fv()
        source = FeatureViewSpecBuilder._convert_feature_view(fv)

        self.assertEqual(source.source_type, SourceType.FEATURES)
        # Unquoted 'upstream_fv' → resolved as 'UPSTREAM_FV'
        self.assertEqual(source.name, "UPSTREAM_FV")
        self.assertEqual(source.source_version, "v1")
        self.assertEqual(len(source.columns), 2)
        self.assertEqual(source.columns[0].name, "SCORE")
        self.assertEqual(source.columns[1].name, "RISK")

    def test_convert_feature_view_no_version(self) -> None:
        """source_version is None when fv.version is falsy."""
        fv = self._make_typed_fv()
        fv.version = None
        source = FeatureViewSpecBuilder._convert_feature_view(fv)
        self.assertIsNone(source.source_version)

    # -- _convert_feature_view_slice ---------------------------------------

    def test_convert_feature_view_slice(self) -> None:
        """Source from a FeatureViewSlice contains only the selected columns."""
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        fv = self._make_typed_fv()
        fvs = mock.MagicMock(spec=FeatureViewSlice)
        fvs.feature_view_ref = fv
        fvs.names = [SqlIdentifier("SCORE")]

        source = FeatureViewSpecBuilder._convert_feature_view_slice(fvs)
        self.assertEqual(source.source_type, SourceType.FEATURES)
        self.assertEqual(source.name, "UPSTREAM_FV")
        self.assertEqual(source.source_version, "v1")
        # columns are filtered to the slice's selection (in slice order).
        self.assertEqual([c.name for c in source.columns], ["SCORE"])

    def test_convert_feature_view_slice_multiple_selected(self) -> None:
        """Slice with multiple selected features preserves slice order."""
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        fv = self._make_typed_fv()
        fvs = mock.MagicMock(spec=FeatureViewSlice)
        fvs.feature_view_ref = fv
        fvs.names = [SqlIdentifier("RISK"), SqlIdentifier("SCORE")]

        source = FeatureViewSpecBuilder._convert_feature_view_slice(fvs)
        self.assertEqual([c.name for c in source.columns], ["RISK", "SCORE"])


# ============================================================================
# UDF Plain-Text Storage and $$ Safety Tests
# ============================================================================


class UDFEncodingTest(absltest.TestCase):
    """Tests for UDF function_definition storage and $$ safety via to_json()."""

    def _make_builder_with_udf(self, code: str) -> FeatureViewSpecBuilder:
        """Helper: create a minimal streaming builder with a UDF."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_udf(
            name="fn",
            engine="pandas",
            output_columns=[("OUT", DoubleType())],
            function_definition=code,
        )
        return builder

    def test_plain_text_storage(self) -> None:
        """function_definition is stored as plain text (no base64)."""
        code = "def transform(x):\n    return x * 2\n"
        builder = self._make_builder_with_udf(code)
        assert builder._udf is not None
        self.assertEqual(builder._udf.function_definition, code)

    def test_dollar_sign_safe_in_to_json(self) -> None:
        """$$ in function definition does not appear in to_json() output,
        and json.loads() recovers the original $$ string."""
        code = "def fn(stream):\n    return '$$dangerous$$'\n"

        # Build a full spec to exercise to_json()
        full_builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(
                name="fn",
                engine="pandas",
                output_columns=[("OUT", StringType())],
                function_definition=code,
            )
        )
        spec_obj = full_builder.build()
        json_output = spec_obj.to_json()

        # Must not contain $$ (would break SQL delimiter)
        self.assertNotIn("$$", json_output)

        # Round-trip: json.loads must recover the original code with $$
        parsed = json.loads(json_output)
        self.assertEqual(parsed["spec"]["udf"]["function_definition"], code)

    def test_triple_dollar_safe_in_to_json(self) -> None:
        """$$$ in function definition is also handled; round-trip recovers original."""
        code = "def fn(stream):\n    return '$$$'\n"
        full_builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(
                name="fn",
                engine="pandas",
                output_columns=[("OUT", DoubleType())],
                function_definition=code,
            )
        )
        spec_obj = full_builder.build()
        json_output = spec_obj.to_json()

        self.assertNotIn("$$", json_output)

        parsed = json.loads(json_output)
        self.assertEqual(parsed["spec"]["udf"]["function_definition"], code)

    def test_unicode_safe(self) -> None:
        """Unicode characters in function definition are stored as-is."""
        code = "def f():\n    return '日本語テスト'\n"
        builder = self._make_builder_with_udf(code)
        assert builder._udf is not None
        self.assertEqual(builder._udf.function_definition, code)

    def test_no_dollar_dollar_passes_through(self) -> None:
        """Code without $$ goes through to_json() unmodified (no spurious escaping)."""
        code = "def fn(x):\n    return x + 1\n"
        full_builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(
                name="fn",
                engine="pandas",
                output_columns=[("OUT", DoubleType())],
                function_definition=code,
            )
        )
        spec_obj = full_builder.build()
        json_output = spec_obj.to_json()

        parsed = json.loads(json_output)
        self.assertEqual(parsed["spec"]["udf"]["function_definition"], code)


# ============================================================================
# Feature Resolution Tests
# ============================================================================


class FeatureResolutionTest(absltest.TestCase):
    """Tests for pipeline-aware _resolve_features."""

    # -- Streaming: resolves from UDF_TRANSFORMED offline config -----------

    def test_streaming_resolves_from_udf_transformed(self) -> None:
        """Streaming FV resolves features from UDF_TRANSFORMED offline config."""
        builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.SUM,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_SUM_24H",
                    )
                ]
            )
        )
        features = builder._resolve_features()
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].source_column.name, "AMOUNT")
        self.assertEqual(features[0].source_column.type, "DoubleType")
        self.assertEqual(features[0].output_column.type, "DoubleType")
        self.assertEqual(features[0].function, "sum")
        self.assertEqual(features[0].window_sec, 86400)

    def test_streaming_enriched_column_from_udf_transformed(self) -> None:
        """Streaming FV resolves from UDF_TRANSFORMED even for UDF-created columns."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="UDF_TBL",
                    columns=StructType(
                        [
                            StructField("USER_ID", StringType()),
                            StructField("ENRICHED_AMOUNT", DoubleType()),
                        ]
                    ),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="ENRICHED_AMOUNT",
                window="1h",
                output_column="ENRICHED_AMOUNT_SUM_1H",
            ),
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.name, "ENRICHED_AMOUNT")
        self.assertEqual(features[0].source_column.type, "DoubleType")

    def test_streaming_missing_column_raises(self) -> None:
        """Column not in UDF_TRANSFORMED raises ValueError."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="UDF_TBL",
                    columns=StructType([StructField("USER_ID", StringType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="NONEXISTENT",
                window="1h",
                output_column="NONEXISTENT_SUM_1H",
            )
        ]
        with self.assertRaisesRegex(ValueError, "not found"):
            builder._resolve_features()

    def test_streaming_no_udf_transformed_raises(self) -> None:
        """Streaming FV with features but no UDF_TRANSFORMED config raises."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="X",
                window="1h",
                output_column="X_SUM_1H",
            )
        ]
        with self.assertRaisesRegex(ValueError, "UDF_TRANSFORMED"):
            builder._resolve_features()

    # -- Batch tiled: resolves from BatchSource schema --------------------

    def test_batch_tiled_resolves_from_batch_source_schema(self) -> None:
        """Batch tiled FV resolves features from BatchSource schema."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.BATCH_SOURCE,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType(
                        [
                            StructField("USER_ID", StringType()),
                            StructField("TILE_START", TimestampType()),
                            StructField("_PARTIAL_SUM_REVENUE", DoubleType()),
                        ]
                    ),
                )
            ]
        )
        builder.set_sources(
            [
                BatchSource(
                    schema=StructType(
                        [
                            StructField("USER_ID", StringType()),
                            StructField("REVENUE", DoubleType()),
                        ]
                    )
                )
            ]
        )
        builder._agg_method = FeatureAggregationMethod.TILES
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="REVENUE",
                window="24h",
                output_column="REVENUE_SUM_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.name, "REVENUE")
        self.assertEqual(features[0].source_column.type, "DoubleType")

    def test_batch_tiled_missing_batch_source_raises(self) -> None:
        """Batch tiled FV without BatchSource raises."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder._agg_method = FeatureAggregationMethod.TILES
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="X",
                window="1h",
                output_column="X_SUM_1H",
            )
        ]
        with self.assertRaisesRegex(ValueError, "BatchSource"):
            builder._resolve_features()

    # -- Batch non-tiled: resolves from BATCH_SOURCE offline config -------

    def test_batch_non_tiled_resolves_from_batch_source_config(self) -> None:
        """Batch non-tiled FV resolves features from BATCH_SOURCE offline config."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs([_batch_source_config()])
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.name, "AMOUNT")

    # -- Realtime: does not support features ------------------------------

    def test_realtime_with_features_raises(self) -> None:
        """Realtime FV does not support aggregation features."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="X",
                window="1h",
                output_column="X_SUM_1H",
            )
        ]
        with self.assertRaisesRegex(ValueError, "does not support"):
            builder._resolve_features()

    # -- Empty features ---------------------------------------------------

    def test_no_features_returns_empty(self) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        self.assertEqual(builder._resolve_features(), [])

    # -- Feature properties (offset, params) ------------------------------

    def test_feature_with_offset(self) -> None:
        """Feature with offset gets offset_sec populated."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("AMOUNT", DoubleType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="7d",
                output_column="AMOUNT_SUM_7D",
                offset="1d",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].offset_sec, 86400)

    def test_feature_with_params(self) -> None:
        """Feature with function_params (e.g., last_n with n)."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("PAGE_ID", StringType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="PAGE_ID_LAST_N_1H",
                params={"n": 10},
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].function_params, {"n": 10})

    # -- Output type determination ----------------------------------------

    def test_count_output_type_is_integer(self) -> None:
        """COUNT always produces DecimalType(18, 0) regardless of source type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("AMOUNT", DoubleType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.COUNT,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_COUNT_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "DoubleType")
        self.assertEqual(features[0].output_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.precision, 18)
        self.assertEqual(features[0].output_column.scale, 0)

    def test_avg_output_type_is_float(self) -> None:
        """AVG always produces DoubleType regardless of source type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("PRICE", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.AVG,
                source_column="PRICE",
                window="1h",
                output_column="PRICE_AVG_1H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.type, "DoubleType")
        self.assertIsNone(features[0].output_column.precision)

    def test_sum_preserves_source_type(self) -> None:
        """SUM preserves the source column type including precision/scale."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("AMOUNT", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="7d",
                output_column="AMOUNT_SUM_7D",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].output_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.precision, 10)
        self.assertEqual(features[0].output_column.scale, 2)

    def test_last_n_preserves_source_type(self) -> None:
        """LAST_N preserves source type (ArrayType not yet supported)."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("PAGE_ID", StringType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.LAST_N,
                source_column="PAGE_ID",
                window="1h",
                output_column="PAGE_ID_LAST_N_1H",
                params={"n": 5},
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].output_column.type, "StringType")

    def test_min_preserves_source_type(self) -> None:
        """MIN preserves the source column type including precision/scale."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("PRICE", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.MIN,
                source_column="PRICE",
                window="24h",
                output_column="PRICE_MIN_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].output_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.precision, 10)
        self.assertEqual(features[0].output_column.scale, 2)

    def test_max_preserves_source_type(self) -> None:
        """MAX preserves the source column type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("SCORE", DoubleType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.MAX,
                source_column="SCORE",
                window="7d",
                output_column="SCORE_MAX_7D",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].output_column.type, "DoubleType")

    def test_stddev_output_type_is_float(self) -> None:
        """STD always produces DoubleType regardless of source type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("VALUE", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.STD,
                source_column="VALUE",
                window="24h",
                output_column="VALUE_STD_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.type, "DoubleType")
        self.assertIsNone(features[0].output_column.precision)

    def test_var_output_type_is_float(self) -> None:
        """VAR always produces DoubleType regardless of source type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("AMOUNT", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.VAR,
                source_column="AMOUNT",
                window="7d",
                output_column="AMOUNT_VAR_7D",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.type, "DoubleType")
        self.assertIsNone(features[0].output_column.precision)

    def test_approx_count_distinct_output_type_is_integer(self) -> None:
        """APPROX_COUNT_DISTINCT always produces DecimalType(18, 0)."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("USER_ID", StringType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.APPROX_COUNT_DISTINCT,
                source_column="USER_ID",
                window="24h",
                output_column="USER_ID_APPROX_COUNT_DISTINCT_24H",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "StringType")
        self.assertEqual(features[0].output_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.precision, 18)
        self.assertEqual(features[0].output_column.scale, 0)

    def test_approx_percentile_output_type_is_float(self) -> None:
        """APPROX_PERCENTILE always produces DoubleType."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("LATENCY", DecimalType(10, 2))]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.APPROX_PERCENTILE,
                source_column="LATENCY",
                window="1h",
                output_column="LATENCY_APPROX_PERCENTILE_1H",
                params={"percentile": 0.95},
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].source_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.type, "DoubleType")
        self.assertIsNone(features[0].output_column.precision)

    def test_first_n_preserves_source_type(self) -> None:
        """FIRST_N preserves the source column type."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("EVENT_ID", StringType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.FIRST_N,
                source_column="EVENT_ID",
                window="1h",
                output_column="EVENT_ID_FIRST_N_1H",
                params={"n": 3},
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(features[0].output_column.type, "StringType")

    # -- Lifetime aggregations: window_sec sentinel -1 → None ---------------

    def test_lifetime_window_omitted(self) -> None:
        """Lifetime aggregation (window='lifetime') produces window_sec=None."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_offline_configs(
            [
                SnowflakeTableInfo(
                    table_type=TableType.UDF_TRANSFORMED,
                    database="DB",
                    schema="SCH",
                    table="T",
                    columns=StructType([StructField("AMOUNT", DoubleType())]),
                )
            ]
        )
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="lifetime",
                output_column="AMOUNT_SUM_LIFETIME",
            )
        ]
        features = builder._resolve_features()
        self.assertEqual(len(features), 1)
        self.assertIsNone(features[0].window_sec)
        self.assertEqual(features[0].function, "sum")

    # -- Error: batch non-tiled missing BATCH_SOURCE config ----------------

    def test_batch_non_tiled_missing_config_raises(self) -> None:
        """Batch non-tiled FV without BATCH_SOURCE config raises ValueError."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        # No offline configs set, but features present
        builder._agg_specs = [
            AggregationSpec(
                function=AggregationType.SUM,
                source_column="AMOUNT",
                window="24h",
                output_column="AMOUNT_SUM_24H",
            )
        ]
        with self.assertRaisesRegex(ValueError, "BATCH_SOURCE"):
            builder._resolve_features()

    def test_duplicate_output_column_rejected_on_streaming(self) -> None:
        """The generic duplicate-output check also fires for non-FG kinds."""
        builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.SUM,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="COLLIDING",
                    ),
                    AggregationSpec(
                        function=AggregationType.MAX,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="COLLIDING",
                    ),
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, r"Duplicate output column name.*COLLIDING"):
            builder.build()


# ============================================================================
# Validation Tests — Streaming
# ============================================================================


class StreamingValidationTest(absltest.TestCase):
    """Validation rules for StreamingFeatureView."""

    def _base_builder(self) -> FeatureViewSpecBuilder:
        return FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )

    def test_missing_udf_transformed(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "UDFTransformed"):
            builder.build()

    def test_tiles_requires_2_configs(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "2 offline configs"):
            builder.build()

    def test_tiles_requires_tiled_config(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config(), _udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "Tiled offline config"):
            builder.build()

    def test_tiles_requires_granularity(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "granularity_sec is required"):
            builder.build()

    def test_no_agg_requires_1_config(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config(), _tiled_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "1 offline config"):
            builder.build()

    def test_no_agg_rejects_granularity(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
            )
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "granularity_sec must be None"):
            builder.build()

    def test_exactly_1_stream_source(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "1 Stream source"):
            builder.build()

    def test_multiple_stream_sources_rejected(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source(), _make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "exactly 1 Stream source"):
            builder.build()

    def test_udf_required(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
        )
        with self.assertRaisesRegex(ValueError, "requires a UDF"):
            builder.build()

    def test_timestamp_required(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(entity_columns=["USER_ID"])
            .set_sources([_make_stream_source()])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "timestamp_field"):
            builder.build()

    def test_unsupported_udf_engine_rejected(self) -> None:
        builder = self._base_builder()
        with self.assertRaisesRegex(ValueError, "Unsupported UDF engine 'python'"):
            builder.set_udf(
                name="fn",
                engine="python",
                output_columns=[("OUT", DoubleType())],
                function_definition="def f(x): return x",
            )


# ============================================================================
# Validation Tests — Batch
# ============================================================================


class BatchValidationTest(absltest.TestCase):
    """Validation rules for BatchFeatureView."""

    def _base_builder(self) -> FeatureViewSpecBuilder:
        return FeatureViewSpecBuilder(
            FeatureViewKind.BatchFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )

    def test_requires_batch_source_config(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(entity_columns=["USER_ID"])
        )
        with self.assertRaisesRegex(ValueError, "BatchSource"):
            builder.build()

    def test_tiled_requires_exactly_1_batch_source(self) -> None:
        """Tiled batch FV must have exactly 1 Batch source, no others."""
        builder = (
            self._base_builder()
            .set_offline_configs([_tiled_batch_offline_config_amt_sum()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
                granularity="1h",
            )
            .set_sources([_make_stream_source()])
        )
        with self.assertRaisesRegex(ValueError, "exactly 1 Batch source"):
            builder.build()

    def test_tiled_no_sources_rejected(self) -> None:
        """Tiled batch FV with no sources → rejected."""
        builder = (
            self._base_builder()
            .set_offline_configs([_tiled_batch_offline_config_amt_sum()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
                granularity="1h",
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly 1 Batch source"):
            builder.build()

    def test_non_tiled_rejects_sources(self) -> None:
        """Non-tiled batch FV must not have sources."""
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .set_sources([BatchSource(schema=StructType([StructField("X", DoubleType())]))])
        )
        with self.assertRaisesRegex(ValueError, "must not have sources"):
            builder.build()

    def test_no_udf_allowed(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .set_udf(**_simple_udf_args())
        )
        with self.assertRaisesRegex(ValueError, "must not have a UDF"):
            builder.build()

    def test_continuous_not_allowed(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.CONTINUOUS,
            )
        )
        with self.assertRaisesRegex(ValueError, "tiles.*or None"):
            builder.build()

    def test_tiles_requires_granularity(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_tiled_batch_offline_config_amt_sum()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([BatchSource(schema=StructType([StructField("X", DoubleType())]))])
        )
        with self.assertRaisesRegex(ValueError, "granularity_sec is required"):
            builder.build()

    def test_tiles_without_batch_source_rejected(self) -> None:
        """Batch tiled FV without a BatchSource in set_sources should fail validation."""
        builder = (
            self._base_builder()
            .set_offline_configs([_tiled_batch_offline_config_amt_sum()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
                granularity="1h",
            )
        )
        with self.assertRaisesRegex(ValueError, "Tiled.*exactly 1 Batch source"):
            builder.build()

    def test_no_agg_rejects_granularity(self) -> None:
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                granularity="1h",
            )
        )
        with self.assertRaisesRegex(ValueError, "granularity_sec must be None"):
            builder.build()

    def test_supported_aggregations_accepted(self) -> None:
        """All PG-supported aggregation types build without error."""
        for agg_type in (
            AggregationType.SUM,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.COUNT,
            AggregationType.AVG,
            AggregationType.STD,
            AggregationType.VAR,
        ):
            batch_schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("EVENT_TIME", TimestampType()),
                    StructField("AMOUNT", DoubleType()),
                ]
            )
            builder = (
                self._base_builder()
                .set_offline_configs(
                    [
                        SnowflakeTableInfo(
                            table_type=TableType.BATCH_SOURCE,
                            database="DB",
                            schema="SCH",
                            table="TILED_TBL",
                            columns=_tiled_dt_schema_for_supported_pg_agg(agg_type),
                        )
                    ]
                )
                .set_properties(
                    entity_columns=["USER_ID"],
                    timestamp_field="EVENT_TIME",
                    granularity="1h",
                    agg_method=FeatureAggregationMethod.TILES,
                    target_lag="30s",
                )
                .set_sources([BatchSource(schema=batch_schema)])
                .set_features(
                    [
                        AggregationSpec(
                            function=agg_type,
                            source_column="AMOUNT",
                            window="24h",
                            output_column=f"AMOUNT_{agg_type.value.upper()}_24H",
                        )
                    ]
                )
            )
            builder.build()

    def test_unsupported_aggregation_rejected(self) -> None:
        """APPROX_PERCENTILE is not supported for PG online store."""
        batch_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType()),
                StructField("AMOUNT", DoubleType()),
            ]
        )
        builder = (
            self._base_builder()
            .set_offline_configs(
                [
                    SnowflakeTableInfo(
                        table_type=TableType.BATCH_SOURCE,
                        database="DB",
                        schema="SCH",
                        table="TILED_TBL",
                        columns=_tiled_dt_schema_for_supported_pg_agg(AggregationType.SUM),
                    )
                ]
            )
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
                target_lag="30s",
            )
            .set_sources([BatchSource(schema=batch_schema)])
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.APPROX_PERCENTILE,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_AP_24H",
                        params={"percentile": 0.5},
                    )
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported aggregation.*Postgres"):
            builder.build()

    def test_mixed_supported_and_unsupported_rejected(self) -> None:
        """A mix of supported and unsupported aggs is rejected."""
        batch_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType()),
                StructField("AMOUNT", DoubleType()),
            ]
        )
        builder = (
            self._base_builder()
            .set_offline_configs(
                [
                    SnowflakeTableInfo(
                        table_type=TableType.BATCH_SOURCE,
                        database="DB",
                        schema="SCH",
                        table="TILED_TBL",
                        columns=_tiled_dt_schema_for_supported_pg_agg(AggregationType.SUM),
                    )
                ]
            )
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
                granularity="1h",
                agg_method=FeatureAggregationMethod.TILES,
                target_lag="30s",
            )
            .set_sources([BatchSource(schema=batch_schema)])
            .set_features(
                [
                    AggregationSpec(
                        function=AggregationType.SUM,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_SUM_24H",
                    ),
                    AggregationSpec(
                        function=AggregationType.LAST_N,
                        source_column="AMOUNT",
                        window="24h",
                        output_column="AMOUNT_LAST_5_24H",
                        params={"n": 5},
                    ),
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported aggregation.*Postgres"):
            builder.build()


# ============================================================================
# Validation Tests — Realtime
# ============================================================================


class RealtimeValidationTest(absltest.TestCase):
    """Validation rules for RealtimeFeatureView."""

    def _base_builder(self) -> FeatureViewSpecBuilder:
        return FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )

    def test_no_offline_configs(self) -> None:
        builder = self._base_builder().set_offline_configs([_batch_source_config()]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have offline configs"):
            builder.build()

    def test_request_only_without_features_rejected(self) -> None:
        """Realtime FV with only a Request source (no Features) is rejected."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
        ]
        with self.assertRaisesRegex(ValueError, "at least 1 Features source"):
            builder.build()

    def test_requires_request_source(self) -> None:
        """Only features, no request → rejected."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exactly 1 Request source"):
            builder.build()

    def test_multiple_request_sources_rejected(self) -> None:
        """More than 1 request source → rejected."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="r1",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="r2",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Z", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exactly 1 Request source.*got 2"):
            builder.build()

    def test_stream_source_rejected(self) -> None:
        """Stream sources not allowed in realtime FVs."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="stream",
                source_type=SourceType.STREAM,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Z", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have Stream sources"):
            builder.build()

    def test_batch_source_rejected(self) -> None:
        """Batch sources not allowed in realtime FVs."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="batch",
                source_type=SourceType.BATCH,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="DoubleType")],
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have Batch sources"):
            builder.build()

    def test_requires_udf(self) -> None:
        builder = self._base_builder()
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "requires a UDF"):
            builder.build()

    def test_no_agg_method(self) -> None:
        builder = (
            self._base_builder().set_properties(agg_method=FeatureAggregationMethod.TILES).set_udf(**_simple_udf_args())
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have an agg_method"):
            builder.build()

    def test_no_granularity(self) -> None:
        builder = self._base_builder().set_properties(granularity="1h").set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have granularity_sec"):
            builder.build()

    def test_no_agg_features(self) -> None:
        """RealtimeFV must not have aggregation features via set_features()."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        builder.set_features(
            [
                AggregationSpec(
                    function=AggregationType.SUM,
                    source_column="X",
                    window="24h",
                    output_column="X_SUM_24H",
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "RealtimeFeatureView must not have aggregation features"):
            builder.build()

    def test_no_timestamp_field(self) -> None:
        builder = self._base_builder().set_properties(timestamp_field="EVENT_TIME").set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "RealtimeFeatureView must not have a timestamp_field"):
            builder.build()

    def _rtfv_with_features_upstream(
        self,
        *,
        upstream_name: str,
        upstream_version: str,
    ) -> FeatureViewSpecBuilder:
        """Build a Realtime FV with one Request + one Features source."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name=upstream_name,
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version=upstream_version,
            ),
        ]
        return builder

    def test_rejects_rtfv_upstream_in_rtfv(self) -> None:
        """RTFV chaining (RTFV-on-RTFV) is not supported."""
        builder = self._rtfv_with_features_upstream(upstream_name="UPSTREAM_RT", upstream_version="v1")
        _seed_upstream_kind(builder, name="UPSTREAM_RT", version="v1", kind=FeatureViewKind.RealtimeFeatureView)

        with self.assertRaises(ValueError) as cm:
            builder.build()
        message = str(cm.exception)
        self.assertIn("RealtimeFeatureView upstream 'UPSTREAM_RT@v1' has kind 'RealtimeFeatureView'", message)
        self.assertIn("RealtimeFeatureView does not support chaining or FeatureGroup upstreams.", message)

    def test_rejects_fg_upstream_in_rtfv(self) -> None:
        """RTFV cannot have a FeatureGroup upstream."""
        builder = self._rtfv_with_features_upstream(upstream_name="UPSTREAM_FG", upstream_version="v1")
        _seed_upstream_kind(builder, name="UPSTREAM_FG", version="v1", kind=FeatureViewKind.FeatureGroup)

        with self.assertRaises(ValueError) as cm:
            builder.build()
        message = str(cm.exception)
        self.assertIn("RealtimeFeatureView upstream 'UPSTREAM_FG@v1' has kind 'FeatureGroup'", message)
        self.assertIn("RealtimeFeatureView does not support chaining or FeatureGroup upstreams.", message)

    def test_accepts_stream_and_batch_upstreams_in_rtfv(self) -> None:
        builder = self._base_builder().set_udf(
            name="transform_fn",
            engine="pandas",
            output_columns=[("AMOUNT", DoubleType())],
            function_definition="def transform_fn(req, stream_fv, batch_fv): return req",
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="UPSTREAM_STREAM",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
            Source(
                name="UPSTREAM_BATCH",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Z", type="DoubleType")],
                source_version="v2",
            ),
        ]
        _seed_upstream_kind(builder, name="UPSTREAM_STREAM", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        _seed_upstream_kind(builder, name="UPSTREAM_BATCH", version="v2", kind=FeatureViewKind.BatchFeatureView)

        # Should build without raising.
        result = builder.build()
        self.assertEqual(result.kind, FeatureViewKind.RealtimeFeatureView)

    def test_set_properties_entity_columns_rejected_for_rtfv(self) -> None:
        """set_properties(entity_columns=...) is not allowed on a RealtimeFeatureView."""
        builder = self._base_builder()
        with self.assertRaisesRegex(ValueError, "RealtimeFeatureView.set_properties.*entity columns are derived"):
            builder.set_properties(entity_columns=["USER_ID"])

    def test_derived_ordered_entity_columns_union_preserves_first_seen_order(self) -> None:
        """RTFV's ``ordered_entity_column_names`` is the first-seen union across
        FEATURES sources; REQUEST contributes nothing."""
        builder = self._base_builder().set_udf(
            name="transform_fn",
            engine="pandas",
            output_columns=[("AMOUNT", DoubleType())],
            function_definition="def transform_fn(req, fv_a, fv_b): return req",
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="UPSTREAM_A",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="A_FEAT", type="DoubleType")],
                source_version="v1",
            ),
            Source(
                name="UPSTREAM_B",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="B_FEAT", type="DoubleType")],
                source_version="v1",
            ),
        ]
        _seed_upstream_kind(builder, name="UPSTREAM_A", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        _seed_upstream_kind(builder, name="UPSTREAM_B", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        # UPSTREAM_A → [USER_ID, MERCHANT_ID]; UPSTREAM_B shares USER_ID and
        # adds DEVICE_ID. Expected union in first-seen order:
        # [USER_ID, MERCHANT_ID, DEVICE_ID].
        _seed_derived_entity_columns(
            builder,
            entity_columns=["USER_ID", "MERCHANT_ID", "DEVICE_ID"],
        )

        result = builder.build()
        self.assertEqual(
            result.spec.ordered_entity_column_names,
            ["USER_ID", "MERCHANT_ID", "DEVICE_ID"],
        )

    def test_request_source_contributes_no_entity_columns(self) -> None:
        """REQUEST source does not contribute to derived entity columns."""
        builder = self._base_builder().set_udf(**_simple_udf_args())
        # Only one upstream FEATURES source contributes USER_ID.
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                # Even a column named "USER_ID" on the REQUEST source must
                # NOT contribute; request data is request-scoped, not an
                # entity join key.
                columns=[FSColumn(name="USER_ID", type="StringType")],
            ),
            Source(
                name="UPSTREAM_FV",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="A", type="DoubleType")],
                source_version="v1",
            ),
        ]
        _seed_upstream_kind(builder, name="UPSTREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        _seed_derived_entity_columns(builder, entity_columns=["MERCHANT_ID"])
        # arity 1 UDF vs 2 sources → update UDF
        builder.set_udf(
            name="transform_fn",
            engine="pandas",
            output_columns=[("AMOUNT", DoubleType())],
            function_definition="def transform_fn(req, fv): return req",
        )

        result = builder.build()
        # Only the FEATURES source contributes entity columns.
        self.assertEqual(result.spec.ordered_entity_column_names, ["MERCHANT_ID"])


# ============================================================================
# UDF Signature Validation Tests — Realtime
# ============================================================================


class RealtimeUdfSignatureValidationTest(absltest.TestCase):
    """Tests for `_validate_udf_signature` and the REQUEST-first rule on RTFV."""

    def _builder(
        self,
        *,
        sources: Optional[list[Source]] = None,
    ) -> FeatureViewSpecBuilder:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        if sources is None:
            sources = [
                Source(
                    name="request",
                    source_type=SourceType.REQUEST,
                    columns=[FSColumn(name="X", type="DoubleType")],
                ),
                Source(
                    name="fv",
                    source_type=SourceType.FEATURES,
                    columns=[FSColumn(name="Y", type="DoubleType")],
                    source_version="v1",
                ),
            ]
        builder._sources = list(sources)
        return builder

    def _set_udf(self, builder: FeatureViewSpecBuilder, function_definition: str, name: str = "score") -> None:
        builder.set_udf(
            name=name,
            engine="pandas",
            output_columns=[("OUT", DoubleType())],
            function_definition=function_definition,
        )

    def test_udf_unparsable_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def f(x: : x")
        with self.assertRaisesRegex(ValueError, "is not valid Python"):
            builder.build()

    def test_udf_missing_named_def_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def other(req): return req")
        with self.assertRaisesRegex(ValueError, "must contain a top-level 'def score"):
            builder.build()

    def test_udf_async_def_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "async def score(req): return req")
        with self.assertRaisesRegex(ValueError, "must not be async"):
            builder.build()

    def test_udf_generator_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def score(req):\n    yield req")
        with self.assertRaisesRegex(ValueError, "must not be a generator"):
            builder.build()

    def test_udf_varargs_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def score(*args): return args")
        with self.assertRaisesRegex(ValueError, r"must not use \*args"):
            builder.build()

    def test_udf_kwargs_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def score(req, **kw): return req")
        with self.assertRaisesRegex(ValueError, r"must not use \*\*kwargs"):
            builder.build()

    def test_udf_kwonly_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def score(*, req): return req")
        with self.assertRaisesRegex(ValueError, "must not use keyword-only args"):
            builder.build()

    def test_udf_arity_too_few_rejected(self) -> None:
        sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="UPSTREAM_FV",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        builder = self._builder(sources=sources)
        _seed_upstream_kind(builder, name="UPSTREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        self._set_udf(builder, "def score(req): return req")
        with self.assertRaisesRegex(ValueError, r"has 1 positional argument\(s\) but the feature view has 2"):
            builder.build()

    def test_udf_arity_too_many_rejected(self) -> None:
        builder = self._builder()
        self._set_udf(builder, "def score(req, fv, extra): return req")
        with self.assertRaisesRegex(ValueError, r"has 3 positional argument\(s\) but the feature view has 2"):
            builder.build()

    def test_udf_arity_matches_ok(self) -> None:
        sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="UPSTREAM_FV",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        builder = self._builder(sources=sources)
        _seed_upstream_kind(builder, name="UPSTREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        self._set_udf(builder, "def score(req, fv): return req")
        result = builder.build()
        self.assertEqual(result.spec.sources[0].source_type, SourceType.REQUEST)
        self.assertEqual(len(result.spec.sources), 2)

    def test_request_source_must_be_first_rejected(self) -> None:
        sources = [
            Source(
                name="UPSTREAM_FV",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
        ]
        builder = self._builder(sources=sources)
        _seed_upstream_kind(builder, name="UPSTREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        self._set_udf(builder, "def score(req, fv): return req")
        with self.assertRaisesRegex(ValueError, "Request source to be first"):
            builder.build()

    def test_udf_with_default_value_ok(self) -> None:
        sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="UPSTREAM_FV",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        builder = self._builder(sources=sources)
        _seed_upstream_kind(builder, name="UPSTREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        self._set_udf(builder, "def score(req, fv=None): return req")
        result = builder.build()
        self.assertEqual(len(result.spec.sources), 2)


# ============================================================================
# UDF Signature Validation Tests — Streaming
# ============================================================================


class StreamingUdfSignatureValidationTest(absltest.TestCase):
    """Tests for `_validate_udf_signature` invocation from `_validate_streaming`."""

    def _builder(self, function_definition: str, name: str = "fn") -> FeatureViewSpecBuilder:
        return (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_sources([_make_stream_source()])
            .set_udf(
                name=name,
                engine="pandas",
                output_columns=[("OUT", DoubleType())],
                function_definition=function_definition,
            )
        )

    def test_streaming_udf_arity_must_be_one(self) -> None:
        builder = self._builder("def fn(stream, extra): return stream")
        with self.assertRaisesRegex(ValueError, r"has 2 positional argument\(s\) but the feature view has 1"):
            builder.build()

    def test_streaming_udf_missing_named_def_rejected(self) -> None:
        builder = self._builder("def other(stream): return stream")
        with self.assertRaisesRegex(ValueError, "must contain a top-level 'def fn"):
            builder.build()

    def test_streaming_udf_arity_one_ok(self) -> None:
        builder = self._builder("def fn(stream): return stream")
        result = builder.build()
        self.assertEqual(result.kind, FeatureViewKind.StreamingFeatureView)


# ============================================================================
# Source Field Validation Tests
# ============================================================================


class SourceFieldValidationTest(absltest.TestCase):
    """Tests for _validate_source_fields across all source types."""

    def _streaming_builder(self) -> FeatureViewSpecBuilder:
        return (
            FeatureViewSpecBuilder(
                FeatureViewKind.StreamingFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_udf_transformed_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                timestamp_field="EVENT_TIME",
            )
            .set_udf(**_simple_udf_args())
        )

    def test_stream_source_no_columns_rejected(self) -> None:
        builder = self._streaming_builder()
        builder._sources = [Source(name="s", source_type=SourceType.STREAM, columns=[])]
        with self.assertRaisesRegex(ValueError, "must have columns"):
            builder.build()

    def test_stream_source_with_version_rejected(self) -> None:
        builder = self._streaming_builder()
        builder._sources = [
            Source(
                name="s",
                source_type=SourceType.STREAM,
                columns=[FSColumn(name="X", type="DoubleType")],
                source_version="v1",
            )
        ]
        with self.assertRaisesRegex(ValueError, "must not have source_version"):
            builder.build()

    def test_request_source_with_version_rejected(self) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
                source_version="v1",
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
                source_version="v1",
            ),
        ]
        builder.set_udf(
            name="fn",
            engine="pandas",
            output_columns=[("OUT", DoubleType())],
            function_definition="def fn(req, features): return None",
        )
        with self.assertRaisesRegex(ValueError, "must not have source_version"):
            builder._validate()

    def test_features_source_without_version_rejected(self) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.RealtimeFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="DoubleType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="DoubleType")],
            ),
        ]
        builder.set_udf(
            name="fn",
            engine="pandas",
            output_columns=[("OUT", DoubleType())],
            function_definition="def fn(req, features): return None",
        )
        with self.assertRaisesRegex(ValueError, "requires source_version"):
            builder._validate()


# ============================================================================
# Omitempty / Serialization Tests
# ============================================================================


class OmitemptySerializationTest(absltest.TestCase):
    """Tests that None fields are stripped from the output (omitempty)."""

    def test_no_optional_fields_in_output(self) -> None:
        """A minimal batch FV should have no optional fields in the spec."""
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="batch_fv",
                version="v1",
            )
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .build()
            .to_dict()
        )
        spec = result["spec"]
        # These should all be absent (None → omitted)
        self.assertNotIn("timestamp_field", spec)
        self.assertNotIn("feature_granularity_sec", spec)
        self.assertNotIn("feature_aggregation_method", spec)
        self.assertNotIn("udf", spec)
        self.assertNotIn("target_lag_sec", spec)
        self.assertEqual(result["online_store_type"], "postgres")
        # Stream / Batch features must not emit source_name / source_version —
        # those JSON keys exist only on FeatureGroup features.
        for feat in spec["features"]:
            self.assertNotIn("source_name", feat)
            self.assertNotIn("source_version", feat)


# ============================================================================
# Client Version Auto-population Test
# ============================================================================


class ClientVersionTest(absltest.TestCase):
    """Tests that client_version is auto-populated from snowflake.ml.version."""

    def test_auto_populated(self) -> None:
        from snowflake.ml.version import VERSION

        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .build()
            .to_dict()
        )
        self.assertEqual(result["metadata"]["client_version"], VERSION)


# ============================================================================
# Online Store Type Test
# ============================================================================


class OnlineStoreTypeTest(absltest.TestCase):
    """online_store_type is auto-populated as 'postgres' — caller never passes it."""

    def test_always_postgres(self) -> None:
        """online_store_type is always 'postgres' in the output."""
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="fv",
                version="v1",
            )
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .build()
            .to_dict()
        )
        self.assertEqual(result["online_store_type"], "postgres")


# ============================================================================
# Interval Conversion Tests
# ============================================================================


class IntervalConversionTest(parameterized.TestCase):
    """Tests for interval string → seconds conversion in set_properties."""

    @parameterized.parameters(  # type: ignore[misc]
        ("1h", 3600),
        ("24h", 86400),
        ("30s", 30),
        ("1d", 86400),
        ("5m", 300),
    )
    def test_granularity_conversion(self, interval: str, expected: int) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_properties(
            entity_columns=["X"],
            granularity=interval,
            agg_method=FeatureAggregationMethod.TILES,
        )
        self.assertEqual(builder._granularity_sec, expected)

    @parameterized.parameters(  # type: ignore[misc]
        ("30s", 30),
        ("1m", 60),
    )
    def test_target_lag_conversion(self, interval: str, expected: int) -> None:
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB",
            schema="SCH",
            name="fv",
            version="v1",
        )
        builder.set_properties(
            entity_columns=["X"],
            target_lag=interval,
        )
        self.assertEqual(builder._target_lag_sec, expected)


# ============================================================================
# Realtime Feature View Passthrough Feature Tests
# ============================================================================


def _make_rt_builder(
    *,
    entity_columns: Optional[list[str]] = None,
    timestamp_field: Optional[str] = None,
) -> FeatureViewSpecBuilder:
    """Helper to construct a Realtime builder with sources + properties set.

    ``entity_columns`` are stamped directly onto ``_derived_entity_columns``
    — the precomputed first-seen union that :meth:`set_sources` would
    otherwise build from upstream FV inputs.

    This helper bypasses :meth:`set_sources` (which expects user-facing types)
    by injecting pre-built ``Source`` instances directly — mirroring the
    pattern used in :func:`_make_fg_builder`.
    """
    builder = FeatureViewSpecBuilder(
        FeatureViewKind.RealtimeFeatureView,
        database="DB",
        schema="SCH",
        name="rt_fv",
        version="v1",
    )
    builder.set_properties(timestamp_field=timestamp_field)
    resolved_entity_columns = entity_columns if entity_columns is not None else ["USER_ID"]
    builder._sources = [
        Source(
            name="request",
            source_type=SourceType.REQUEST,
            columns=[FSColumn(name="REQ_X", type="DoubleType")],
        ),
        Source(
            name="UPSTREAM_FV",
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="UPSTREAM_FEAT", type="DoubleType")],
            source_version="v1",
        ),
    ]
    _seed_derived_entity_columns(builder, entity_columns=resolved_entity_columns)
    return builder


class RealtimePassthroughFeaturesTest(absltest.TestCase):
    """Tests for _resolve_passthrough_features on RealtimeFeatureView."""

    def test_features_derived_from_udf_output_columns(self) -> None:
        builder = _make_rt_builder(entity_columns=["USER_ID"])
        builder.set_udf(
            name="score_fn",
            engine="pandas",
            output_columns=[
                ("USER_ID", StringType()),
                ("RISK_SCORE", DoubleType()),
                ("RISK_BUCKET", StringType()),
            ],
            function_definition="def score_fn(df): return df",
        )

        features = builder._resolve_passthrough_features()

        feature_names = [f.output_column.name for f in features]
        # USER_ID is an entity column → excluded.
        self.assertEqual(feature_names, ["RISK_SCORE", "RISK_BUCKET"])
        for f in features:
            # Passthrough: source_column is the same FSColumn instance as output_column.
            self.assertEqual(f.source_column, f.output_column)
            self.assertIsNone(f.function)
            self.assertIsNone(f.window_sec)
            self.assertIsNone(f.offset_sec)

    def test_entity_columns_excluded(self) -> None:
        builder = _make_rt_builder(entity_columns=["USER_ID", "MERCHANT_ID"])
        builder.set_udf(
            name="score_fn",
            engine="pandas",
            output_columns=[
                ("USER_ID", StringType()),
                ("MERCHANT_ID", StringType()),
                ("RISK_SCORE", DoubleType()),
            ],
            function_definition="def score_fn(df): return df",
        )

        features = builder._resolve_passthrough_features()

        self.assertEqual([f.output_column.name for f in features], ["RISK_SCORE"])

    def test_no_udf_returns_empty(self) -> None:
        builder = _make_rt_builder(entity_columns=["USER_ID"])

        features = builder._resolve_passthrough_features()

        self.assertEqual(features, [])

    def test_udf_with_only_entity_columns_returns_empty(self) -> None:
        builder = _make_rt_builder(entity_columns=["USER_ID"])
        builder.set_udf(
            name="score_fn",
            engine="pandas",
            output_columns=[("USER_ID", StringType())],
            function_definition="def score_fn(df): return df",
        )

        features = builder._resolve_passthrough_features()

        self.assertEqual(features, [])

    def test_column_type_metadata_preserved(self) -> None:
        builder = _make_rt_builder(entity_columns=["USER_ID"])
        builder.set_udf(
            name="score_fn",
            engine="pandas",
            output_columns=[
                ("USER_ID", StringType()),
                ("AMOUNT_CENTS", DecimalType(18, 4)),
                ("RISK_LABEL", StringType(64)),
                ("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
            ],
            function_definition="def score_fn(df): return df",
        )

        features = builder._resolve_passthrough_features()
        by_name = {f.output_column.name: f.output_column for f in features}

        amount = by_name["AMOUNT_CENTS"]
        self.assertEqual(amount.type, "DecimalType")
        self.assertEqual(amount.precision, 18)
        self.assertEqual(amount.scale, 4)

        label = by_name["RISK_LABEL"]
        self.assertEqual(label.type, "StringType")
        self.assertEqual(label.length, 64)

        ts = by_name["EVENT_TIME"]
        self.assertEqual(ts.type, "TimestampType")


# ============================================================================
# FeatureGroup Helpers / Fixtures
# ============================================================================


def _fg_features_source(
    *,
    name: str = "USER_FV",
    columns: Optional[list[FSColumn]] = None,
    source_version: Optional[str] = "v1",
) -> Source:
    """Build a Source with source_type=FEATURES directly (bypassing FV mocks).

    For FEATURES sources, ``columns`` is the caller-selected subset (in the
    desired order) — the same shape ``_convert_feature_view_slice`` produces.
    """
    if columns is None:
        columns = [
            FSColumn(name="TOTAL_SPEND_30D", type="DoubleType"),
            FSColumn(name="AVG_TXN_AMOUNT", type="DoubleType"),
        ]
    return Source(
        name=name,
        source_type=SourceType.FEATURES,
        columns=columns,
        source_version=source_version,
    )


def _make_fg_builder(
    *,
    sources: list[Source],
    entity_columns: Optional[list[str]] = None,
    prefix_map: Optional[dict[tuple[str, Optional[str]], str]] = None,
    name: str = "fg",
    version: str = "v1",
) -> FeatureViewSpecBuilder:
    """Helper to construct a FeatureGroup builder with pre-built Source objects.

    Bypasses set_sources (which expects user-facing types) by injecting Source
    instances directly into the builder's internal state; avoids mocking
    FeatureView/FeatureViewSlice for spec-builder unit tests.

    ``entity_columns`` is stamped directly onto
    ``_derived_entity_columns`` — the precomputed first-seen union that
    :meth:`set_sources` would otherwise build from upstream FV inputs.
    """
    builder = FeatureViewSpecBuilder(
        FeatureViewKind.FeatureGroup,
        database="DB",
        schema="SCH",
        name=name,
        version=version,
    )
    builder._sources = list(sources)
    resolved_entity_columns = entity_columns or ["USER_ID"]
    _seed_derived_entity_columns(builder, entity_columns=resolved_entity_columns)
    if prefix_map is not None:
        builder.set_source_prefixes(prefix_map)
    return builder


# ============================================================================
# FeatureGroup Builder Tests
# ============================================================================


class FeatureGroupBuilderTest(absltest.TestCase):
    """Build a FeatureGroup spec end-to-end."""

    def test_single_features_source(self) -> None:
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
        ).build()

        self.assertEqual(result.kind, FeatureViewKind.FeatureGroup)
        self.assertEqual(len(result.offline_configs), 0)
        # FG emits its FEATURES sources, mirroring RTFV's shape.
        self.assertEqual(len(result.spec.sources), 1)
        src = result.spec.sources[0]
        self.assertEqual(src.source_type, SourceType.FEATURES)
        self.assertEqual(src.name, "USER_FV")
        self.assertEqual(src.source_version, "v1")
        self.assertEqual([c.name for c in src.columns], ["TOTAL_SPEND_30D", "AVG_TXN_AMOUNT"])
        self.assertIsNone(result.spec.udf)
        self.assertIsNone(result.spec.feature_aggregation_method)
        self.assertIsNone(result.spec.feature_granularity_sec)

        # Passthrough: 2 columns from the source, no entity exclusion needed.
        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["TOTAL_SPEND_30D", "AVG_TXN_AMOUNT"])
        for f in result.spec.features:
            # No prefix → output_column == source_column.
            self.assertEqual(f.source_column, f.output_column)
            # Each FG feature carries the upstream FV identity inline.
            self.assertEqual(f.source_name, "USER_FV")
            self.assertEqual(f.source_version, "v1")

    def test_multiple_features_sources(self) -> None:
        user_src = _fg_features_source(
            name="USER_FV",
            columns=[
                FSColumn(name="TOTAL_SPEND_30D", type="DoubleType"),
            ],
        )
        txn_src = _fg_features_source(
            name="TXN_FV",
            columns=[
                FSColumn(name="TXN_COUNT_24H", type="DecimalType", precision=38, scale=0),
                FSColumn(name="LAST_AMOUNT", type="DoubleType"),
            ],
            source_version="v2",
        )

        result = _make_fg_builder(sources=[user_src, txn_src]).build()

        # FG emits its FEATURES sources in the order they were registered.
        self.assertEqual(len(result.spec.sources), 2)
        self.assertEqual(
            [(s.name, s.source_version) for s in result.spec.sources],
            [("USER_FV", "v1"), ("TXN_FV", "v2")],
        )
        self.assertEqual([c.name for c in result.spec.sources[0].columns], ["TOTAL_SPEND_30D"])
        self.assertEqual(
            [c.name for c in result.spec.sources[1].columns],
            ["TXN_COUNT_24H", "LAST_AMOUNT"],
        )

        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["TOTAL_SPEND_30D", "TXN_COUNT_24H", "LAST_AMOUNT"])

        # Each feature points at the right upstream FV via (source_name, source_version).
        identities = [(f.source_name, f.source_version) for f in result.spec.features]
        self.assertEqual(
            identities,
            [
                ("USER_FV", "v1"),
                ("TXN_FV", "v2"),
                ("TXN_FV", "v2"),
            ],
        )

    def test_feature_view_slice_only_selected_features(self) -> None:
        # A slice is encoded as columns[] containing only the selected
        # columns; no separate selected_features field exists.
        src = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="TOTAL_SPEND_30D", type="DoubleType")],
        )

        result = _make_fg_builder(sources=[src]).build()

        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["TOTAL_SPEND_30D"])

    def test_entity_columns_excluded_from_features(self) -> None:
        src = _fg_features_source(
            name="USER_FV",
            columns=[
                FSColumn(name="USER_ID", type="StringType"),
                FSColumn(name="TOTAL_SPEND_30D", type="DoubleType"),
            ],
        )

        result = _make_fg_builder(
            sources=[src],
            entity_columns=["USER_ID"],
        ).build()

        feature_names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(feature_names, ["TOTAL_SPEND_30D"])

    def test_source_with_only_entity_columns_contributes_zero_features(self) -> None:
        src = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        )

        result = _make_fg_builder(
            sources=[src],
            entity_columns=["USER_ID"],
        ).build()

        self.assertEqual(result.spec.features, [])
        # The FV reference still appears in sources[] even though all of its
        # columns are entity columns and contribute no features.
        self.assertEqual(len(result.spec.sources), 1)
        self.assertEqual(result.spec.sources[0].name, "USER_FV")
        self.assertEqual([c.name for c in result.spec.sources[0].columns], ["USER_ID"])

    @mock.patch(
        "snowflake.ml.feature_store.spec.builder.FeatureViewSpecBuilder._convert_feature_view",
        side_effect=lambda fv: Source(
            name=str(fv.name),
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version=str(fv.version),
        ),
    )
    def test_end_to_end_via_set_sources(self, _mock_convert: mock.MagicMock) -> None:
        """Build an FG from real FeatureView objects (through set_sources)."""
        fv = _mock_feature_view(name="USER_FV", version="v1")
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.FeatureGroup,
                database="DB",
                schema="SCH",
                name="fg",
                version="v1",
            )
            .set_sources([fv])
            .set_source_prefixes({("USER_FV", "v1"): "USER_FV_v1_"})
            .build()
        )

        self.assertEqual(result.kind, FeatureViewKind.FeatureGroup)
        # FG emits its FEATURES sources mirroring RTFV's shape.
        self.assertEqual(len(result.spec.sources), 1)
        src = result.spec.sources[0]
        self.assertEqual(src.source_type, SourceType.FEATURES)
        self.assertEqual(src.name, "USER_FV")
        self.assertEqual(src.source_version, "v1")
        self.assertEqual([c.name for c in src.columns], ["SCORE"])
        self.assertEqual([f.output_column.name for f in result.spec.features], ["USER_FV_v1_SCORE"])
        self.assertEqual(result.spec.features[0].source_column.name, "SCORE")
        self.assertEqual(result.spec.features[0].source_name, "USER_FV")
        self.assertEqual(result.spec.features[0].source_version, "v1")


# ============================================================================
# FeatureGroup Validation Tests
# ============================================================================


class FeatureGroupValidationTest(absltest.TestCase):
    """Validation rules specific to FeatureGroup."""

    def test_rejects_offline_configs(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder.set_offline_configs([_udf_transformed_config()])

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have offline configs"):
            builder.build()

    def test_rejects_udf(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder.set_udf(
            name="bad_udf",
            engine="pandas",
            output_columns=[("X", DoubleType())],
            function_definition="def bad_udf(df): return df",
        )

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have a UDF"):
            builder.build()

    def test_rejects_request_source_mixed_with_features(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        # Inject a REQUEST source alongside FEATURES.
        builder._sources.append(
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="TXN_AMOUNT", type="DoubleType")],
            )
        )

        with self.assertRaisesRegex(ValueError, "FeatureGroup must only have Features sources"):
            builder.build()

    def test_rejects_stream_source_mixed_with_features(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._sources.append(
            Source(
                name="TXN_EVENTS",
                source_type=SourceType.STREAM,
                columns=[FSColumn(name="AMOUNT", type="DoubleType")],
            )
        )

        with self.assertRaisesRegex(ValueError, "FeatureGroup must only have Features sources"):
            builder.build()

    def test_rejects_batch_source_mixed_with_features(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._sources.append(
            Source(
                name="batch",
                source_type=SourceType.BATCH,
                columns=[FSColumn(name="AMOUNT", type="DoubleType")],
            )
        )

        with self.assertRaisesRegex(ValueError, "FeatureGroup must only have Features sources"):
            builder.build()

    def test_rejects_empty_sources(self) -> None:
        builder = _make_fg_builder(sources=[])

        with self.assertRaisesRegex(ValueError, "FeatureGroup requires at least 1 Features source"):
            builder.build()

    def test_rejects_agg_method(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._agg_method = FeatureAggregationMethod.TILES

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have an agg_method"):
            builder.build()

    def test_rejects_granularity(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._granularity_sec = 3600

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have granularity_sec"):
            builder.build()

    def test_rejects_agg_features(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder.set_features(
            [
                AggregationSpec(
                    function=AggregationType.SUM,
                    source_column="TOTAL_SPEND_30D",
                    window="24h",
                    output_column="TOTAL_SPEND_24H",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have aggregation features"):
            builder.build()

    def test_rejects_timestamp_field(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._timestamp_field = "EVENT_TIME"

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have a timestamp_field"):
            builder.build()

    def test_rejects_target_lag_sec(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        builder._target_lag_sec = 3600

        with self.assertRaisesRegex(ValueError, "FeatureGroup must not have target_lag_sec"):
            builder.build()

    def test_rejects_duplicate_output_column_names(self) -> None:
        # Two FEATURES sources expose the same column name → collision
        # without prefixing should surface at build() time.
        src_a = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="TOTAL_SPEND_30D", type="DoubleType")],
        )
        src_b = _fg_features_source(
            name="TXN_FV",
            columns=[FSColumn(name="TOTAL_SPEND_30D", type="DoubleType")],
        )
        builder = _make_fg_builder(sources=[src_a, src_b])

        with self.assertRaisesRegex(ValueError, "Duplicate output column name"):
            builder.build()

    def test_rejects_duplicate_after_prefix_collision(self) -> None:
        # A prefix that collides with an existing (unprefixed) column
        # should also be rejected.
        src_a = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="p_AMT", type="DoubleType")],
        )
        src_b = _fg_features_source(
            name="TXN_FV",
            columns=[FSColumn(name="AMT", type="DoubleType")],
        )
        builder = _make_fg_builder(
            sources=[src_a, src_b],
            prefix_map={("TXN_FV", "v1"): "p_"},
        )

        with self.assertRaisesRegex(ValueError, "Duplicate output column name.*p_AMT"):
            builder.build()

    def test_rejects_fg_upstream_in_fg(self) -> None:
        # FG-on-FG chaining is not supported. (User-facing FG FeatureView class
        # doesn't exist yet, so we simulate the upstream kind via the helper.)
        src = _fg_features_source(name="USER_FG", source_version="v1")
        builder = _make_fg_builder(sources=[src])
        _seed_upstream_kind(builder, name="USER_FG", version="v1", kind=FeatureViewKind.FeatureGroup)

        with self.assertRaises(ValueError) as cm:
            builder.build()
        message = str(cm.exception)
        self.assertIn("FeatureGroup upstream 'USER_FG@v1' has kind 'FeatureGroup'", message)
        self.assertIn("FeatureGroup chaining is not supported.", message)

    def test_accepts_rtfv_upstream_in_fg(self) -> None:
        # FG explicitly allows RTFV upstreams (deviation from hand-off doc;
        # reflects updated platform contract).
        src = _fg_features_source(name="USER_RT_FV", source_version="v1")
        builder = _make_fg_builder(sources=[src])
        _seed_upstream_kind(builder, name="USER_RT_FV", version="v1", kind=FeatureViewKind.RealtimeFeatureView)

        result = builder.build()
        self.assertEqual(result.kind, FeatureViewKind.FeatureGroup)
        self.assertEqual(result.spec.features[0].source_name, "USER_RT_FV")
        self.assertEqual(result.spec.features[0].source_version, "v1")

    def test_accepts_stream_and_batch_upstreams_in_fg(self) -> None:
        stream_src = _fg_features_source(
            name="USER_STREAM_FV",
            columns=[FSColumn(name="STREAM_FEAT", type="DoubleType")],
            source_version="v1",
        )
        batch_src = _fg_features_source(
            name="USER_BATCH_FV",
            columns=[FSColumn(name="BATCH_FEAT", type="DoubleType")],
            source_version="v2",
        )
        builder = _make_fg_builder(sources=[stream_src, batch_src])
        _seed_upstream_kind(builder, name="USER_STREAM_FV", version="v1", kind=FeatureViewKind.StreamingFeatureView)
        _seed_upstream_kind(builder, name="USER_BATCH_FV", version="v2", kind=FeatureViewKind.BatchFeatureView)

        # Should build without raising.
        result = builder.build()
        identities = [(f.source_name, f.source_version) for f in result.spec.features]
        self.assertEqual(
            identities,
            [
                ("USER_STREAM_FV", "v1"),
                ("USER_BATCH_FV", "v2"),
            ],
        )

    def test_set_properties_entity_columns_rejected_for_fg(self) -> None:
        """set_properties(entity_columns=...) is not allowed on a FeatureGroup."""
        builder = FeatureViewSpecBuilder(
            FeatureViewKind.FeatureGroup,
            database="DB",
            schema="SCH",
            name="fg",
            version="v1",
        )
        with self.assertRaisesRegex(ValueError, "FeatureGroup.set_properties.*entity columns are derived"):
            builder.set_properties(entity_columns=["USER_ID"])

    def test_derived_ordered_entity_columns_union(self) -> None:
        """FG's ``ordered_entity_column_names`` is the first-seen union across
        FEATURES sources."""
        src_a = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="A", type="DoubleType")],
            source_version="v1",
        )
        src_b = _fg_features_source(
            name="TXN_FV",
            columns=[FSColumn(name="B", type="DoubleType")],
            source_version="v1",
        )
        # USER_FV → [USER_ID, MERCHANT_ID]; TXN_FV → [USER_ID, TXN_ID].
        # Expected: [USER_ID, MERCHANT_ID, TXN_ID].
        builder = _make_fg_builder(
            sources=[src_a, src_b],
            entity_columns=["USER_ID", "MERCHANT_ID", "TXN_ID"],
        )

        result = builder.build()
        self.assertEqual(
            result.spec.ordered_entity_column_names,
            ["USER_ID", "MERCHANT_ID", "TXN_ID"],
        )


# ============================================================================
# FeatureGroup Prefix Tests
# ============================================================================


class FeatureGroupPrefixTest(absltest.TestCase):
    """source_column / output_column prefix resolution for FeatureGroup."""

    def test_no_prefix_source_equals_output(self) -> None:
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
        ).build()

        for f in result.spec.features:
            self.assertEqual(f.source_column, f.output_column)
            self.assertEqual(f.source_column.name, f.output_column.name)

    def test_auto_prefix_applied(self) -> None:
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
            prefix_map={("USER_FV", "v1"): "USER_FV_v1_"},
        ).build()

        names = [(f.source_column.name, f.output_column.name) for f in result.spec.features]
        self.assertEqual(
            names,
            [
                ("TOTAL_SPEND_30D", "USER_FV_v1_TOTAL_SPEND_30D"),
                ("AVG_TXN_AMOUNT", "USER_FV_v1_AVG_TXN_AMOUNT"),
            ],
        )

    def test_with_name_prefix_overrides_default(self) -> None:
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
            prefix_map={("USER_FV", "v1"): "user_"},
        ).build()

        for f in result.spec.features:
            self.assertTrue(
                f.output_column.name.startswith("user_"),
                f"expected 'user_' prefix on {f.output_column.name}",
            )
            # Source column retains the original (unprefixed) name.
            self.assertFalse(f.source_column.name.startswith("user_"))

    def test_empty_string_prefix_means_no_prefix(self) -> None:
        # An explicit empty-string prefix maps to a no-op.
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
            prefix_map={("USER_FV", "v1"): ""},
        ).build()

        for f in result.spec.features:
            self.assertEqual(f.source_column, f.output_column)

    def test_mixed_with_name_and_auto_prefix(self) -> None:
        user_src = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="TOTAL_SPEND_30D", type="DoubleType")],
        )
        txn_src = _fg_features_source(
            name="TXN_FV",
            columns=[FSColumn(name="TXN_COUNT", type="DoubleType")],
            source_version="v2",
        )

        result = _make_fg_builder(
            sources=[user_src, txn_src],
            prefix_map={
                ("USER_FV", "v1"): "user_",
                ("TXN_FV", "v2"): "TXN_FV_v2_",
            },
        ).build()

        by_output = {f.output_column.name: f for f in result.spec.features}
        self.assertIn("user_TOTAL_SPEND_30D", by_output)
        self.assertIn("TXN_FV_v2_TXN_COUNT", by_output)
        # Source columns keep original names regardless of prefix.
        self.assertEqual(by_output["user_TOTAL_SPEND_30D"].source_column.name, "TOTAL_SPEND_30D")
        self.assertEqual(by_output["TXN_FV_v2_TXN_COUNT"].source_column.name, "TXN_COUNT")

    def test_unmapped_source_falls_back_to_no_prefix(self) -> None:
        # When prefix_map omits a source, it should NOT raise — just no prefix.
        user_src = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="TOTAL_SPEND_30D", type="DoubleType")],
        )
        txn_src = _fg_features_source(
            name="TXN_FV",
            columns=[FSColumn(name="TXN_COUNT", type="DoubleType")],
        )

        result = _make_fg_builder(
            sources=[user_src, txn_src],
            prefix_map={("USER_FV", "v1"): "user_"},  # TXN_FV omitted
        ).build()

        names = [(f.source_column.name, f.output_column.name) for f in result.spec.features]
        self.assertEqual(
            names,
            [
                ("TOTAL_SPEND_30D", "user_TOTAL_SPEND_30D"),
                ("TXN_COUNT", "TXN_COUNT"),  # no prefix for unmapped source
            ],
        )

    def test_prefix_preserves_column_type_metadata(self) -> None:
        src = _fg_features_source(
            name="USER_FV",
            columns=[
                FSColumn(name="AMOUNT_CENTS", type="DecimalType", precision=18, scale=4),
                FSColumn(name="LABEL", type="StringType", length=64),
            ],
        )

        result = _make_fg_builder(
            sources=[src],
            prefix_map={("USER_FV", "v1"): "p_"},
        ).build()

        by_output = {f.output_column.name: f.output_column for f in result.spec.features}
        amount = by_output["p_AMOUNT_CENTS"]
        self.assertEqual(amount.type, "DecimalType")
        self.assertEqual(amount.precision, 18)
        self.assertEqual(amount.scale, 4)

        label = by_output["p_LABEL"]
        self.assertEqual(label.type, "StringType")
        self.assertEqual(label.length, 64)

    def test_set_source_prefixes_returns_self(self) -> None:
        builder = _make_fg_builder(sources=[_fg_features_source()])
        result = builder.set_source_prefixes({("USER_FV", "v1"): "p_"})
        self.assertIs(result, builder)

    def test_same_name_different_versions_disambiguated_by_version(self) -> None:
        # Two sources with the same name but different versions (e.g., USER_FV@v1
        # and USER_FV@v2) must be prefixable independently via the
        # (name, version) key.
        src_v1 = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version="v1",
        )
        src_v2 = _fg_features_source(
            name="USER_FV",
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version="v2",
        )

        result = _make_fg_builder(
            sources=[src_v1, src_v2],
            prefix_map={
                ("USER_FV", "v1"): "USER_FV_v1_",
                ("USER_FV", "v2"): "USER_FV_v2_",
            },
        ).build()

        names = [f.output_column.name for f in result.spec.features]
        self.assertEqual(names, ["USER_FV_v1_SCORE", "USER_FV_v2_SCORE"])

    def test_feature_view_slice_order_preserved(self) -> None:
        # FeatureViewSlice preserves caller-requested feature order, encoded
        # as the order of columns[] on the FEATURES source.
        src = _fg_features_source(
            name="USER_FV",
            columns=[
                FSColumn(name="C", type="DoubleType"),
                FSColumn(name="A", type="DoubleType"),
            ],
        )

        result = _make_fg_builder(sources=[src]).build()

        self.assertEqual([f.output_column.name for f in result.spec.features], ["C", "A"])


# ============================================================================
# FeatureGroup / Realtime Serialization Smoke Tests
# ============================================================================


class NewKindsSerializationTest(absltest.TestCase):
    """Ensure to_dict()/to_json() emit the expected shape for new kinds."""

    def test_feature_group_serialization_shape(self) -> None:
        result = _make_fg_builder(
            sources=[_fg_features_source(name="USER_FV")],
            prefix_map={("USER_FV", "v1"): "u_"},
        ).build()

        d = result.to_dict()
        self.assertEqual(d["kind"], "FeatureGroup")
        self.assertNotIn("udf", d["spec"])
        self.assertNotIn("feature_aggregation_method", d["spec"])
        self.assertNotIn("feature_granularity_sec", d["spec"])
        # FG emits its FEATURES sources, mirroring RTFV's wire shape.
        self.assertEqual(len(d["spec"]["sources"]), 1)
        self.assertEqual(d["spec"]["sources"][0]["name"], "USER_FV")
        self.assertEqual(d["spec"]["sources"][0]["source_type"], SourceType.FEATURES)
        self.assertEqual(d["spec"]["sources"][0]["source_version"], "v1")
        self.assertEqual(
            [c["name"] for c in d["spec"]["sources"][0]["columns"]],
            ["TOTAL_SPEND_30D", "AVG_TXN_AMOUNT"],
        )
        self.assertEqual(
            [f["output_column"]["name"] for f in d["spec"]["features"]], ["u_TOTAL_SPEND_30D", "u_AVG_TXN_AMOUNT"]
        )
        for feat in d["spec"]["features"]:
            self.assertEqual(feat["source_name"], "USER_FV")
            self.assertEqual(feat["source_version"], "v1")

        # to_json() should be parseable round-trip.
        import json

        reloaded = json.loads(result.to_json())
        self.assertEqual(reloaded["kind"], "FeatureGroup")

    @mock.patch(
        "snowflake.ml.feature_store.spec.builder.FeatureViewSpecBuilder._convert_feature_view",
        side_effect=lambda fv: Source(
            name=str(fv.name),
            source_type=SourceType.FEATURES,
            columns=[FSColumn(name="SCORE", type="DoubleType")],
            source_version=str(fv.version),
        ),
    )
    def test_realtime_serialization_shape(self, _mock_convert: mock.MagicMock) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.RealtimeFeatureView,
                database="DB",
                schema="SCH",
                name="realtime_fv",
                version="v1",
            )
            .set_sources([_request_source(), _mock_feature_view()])
            .set_udf(
                name="score_fn",
                engine="pandas",
                output_columns=[("RISK_SCORE", DoubleType())],
                function_definition="def score_fn(txn, features): return 0.5",
            )
            .build()
        )

        d = result.to_dict()
        self.assertEqual(d["kind"], "RealtimeFeatureView")
        self.assertIn("udf", d["spec"])
        self.assertEqual(d["spec"]["udf"]["name"], "score_fn")
        self.assertEqual([f["output_column"]["name"] for f in d["spec"]["features"]], ["RISK_SCORE"])
        # RTFV features do NOT carry source_name / source_version
        # (the UDF is the producer of every output column).
        for feat in d["spec"]["features"]:
            self.assertNotIn("source_name", feat)
            self.assertNotIn("source_version", feat)


if __name__ == "__main__":
    absltest.main()
