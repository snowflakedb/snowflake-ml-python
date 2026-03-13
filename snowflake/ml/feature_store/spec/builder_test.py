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
from typing import Optional
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
    DecimalType,
    FloatType,
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
            StructField("AMOUNT", FloatType()),
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
                StructField("AMOUNT", FloatType()),
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
                StructField("AMOUNT_SUM_24H", FloatType()),
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
                StructField("AMOUNT", FloatType()),
            ]
        ),
    )


def _simple_udf_args() -> dict:
    return {
        "name": "transform_fn",
        "engine": "python",
        "output_columns": [("AMOUNT", FloatType())],
        "function_definition": "def f(x): return x * 2",
    }


def _request_source() -> RequestSource:
    return RequestSource(
        schema=StructType(
            [
                StructField("TXN_AMOUNT", FloatType()),
                StructField("MERCHANT_ID", StringType()),
            ]
        )
    )


def _mock_feature_view(
    name: str = "upstream_fv",
    version: str = "v1",
    feature_names: Optional[list[str]] = None,
) -> mock.MagicMock:
    """Create a mock FeatureView for source conversion tests.

    Uses spec= to ensure isinstance checks pass in set_sources.
    """
    from snowflake.ml.feature_store.feature_view import FeatureView

    fv = mock.MagicMock(spec=FeatureView)
    fv.name = name
    fv.version = version
    names = feature_names or ["SCORE", "RISK"]
    fv.feature_names = names
    fv.output_schema = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("SCORE", FloatType()),
            StructField("RISK", FloatType()),
        ]
    )
    return fv


def _mock_feature_view_slice(
    selected_features: Optional[list[str]] = None,
) -> mock.MagicMock:
    """Create a mock FeatureViewSlice for source conversion tests.

    Uses spec= to ensure isinstance checks pass in set_sources.
    """
    from snowflake.ml.feature_store.feature_view import FeatureViewSlice

    fvs = mock.MagicMock(spec=FeatureViewSlice)
    fv = _mock_feature_view()
    fvs.feature_view_ref = fv
    fvs.names = selected_features or ["SCORE"]
    return fvs


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
        self.assertEqual(udf.name, "transform_fn")
        self.assertEqual(udf.function_definition, "def f(x): return x * 2")

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

    def test_build_batch_with_tiles(self) -> None:
        result = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database="DB",
                schema="SCH",
                name="batch_tiled",
                version="v1",
            )
            .set_offline_configs([_batch_source_config()])
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
                                StructField("AMOUNT", FloatType()),
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
            columns=[FSColumn(name="SCORE", type="FloatType")],
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
            .set_properties(entity_columns=["USER_ID"])
            .set_sources([_request_source(), mock_fv])
            .set_udf(
                name="score_fn",
                engine="python",
                output_columns=[("RISK_SCORE", FloatType())],
                function_definition="def score(txn, features): return 0.5",
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
            columns=[FSColumn(name="SCORE", type="FloatType")],
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
            columns=[FSColumn(name="SCORE", type="FloatType")],
            source_version="v1",
            selected_features=["SCORE"],
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
        self.assertEqual(src.selected_features, ["SCORE"])

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
                StructField("AMOUNT", FloatType()),
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
                StructField("SCORE", FloatType()),
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
        self.assertEqual(cols[0].type, "FloatType")
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
        self.assertIsNone(source.selected_features)

    def test_convert_feature_view_no_version(self) -> None:
        """source_version is None when fv.version is falsy."""
        fv = self._make_typed_fv()
        fv.version = None
        source = FeatureViewSpecBuilder._convert_feature_view(fv)
        self.assertIsNone(source.source_version)

    # -- _convert_feature_view_slice ---------------------------------------

    def test_convert_feature_view_slice(self) -> None:
        """Full Source with selected_features from a FeatureViewSlice."""
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        fv = self._make_typed_fv()
        fvs = mock.MagicMock(spec=FeatureViewSlice)
        fvs.feature_view_ref = fv
        fvs.names = [SqlIdentifier("SCORE")]

        source = FeatureViewSpecBuilder._convert_feature_view_slice(fvs)
        self.assertEqual(source.source_type, SourceType.FEATURES)
        self.assertEqual(source.name, "UPSTREAM_FV")
        self.assertEqual(source.source_version, "v1")
        self.assertEqual(source.selected_features, ["SCORE"])
        # columns come from all features (not filtered by slice selection)
        self.assertEqual(len(source.columns), 2)

    def test_convert_feature_view_slice_multiple_selected(self) -> None:
        """Slice with multiple selected features."""
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        fv = self._make_typed_fv()
        fvs = mock.MagicMock(spec=FeatureViewSlice)
        fvs.feature_view_ref = fv
        fvs.names = [SqlIdentifier("SCORE"), SqlIdentifier("RISK")]

        source = FeatureViewSpecBuilder._convert_feature_view_slice(fvs)
        self.assertEqual(source.selected_features, ["SCORE", "RISK"])
        self.assertEqual(len(source.columns), 2)


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
            engine="python",
            output_columns=[("OUT", FloatType())],
            function_definition=code,
        )
        return builder

    def test_plain_text_storage(self) -> None:
        """function_definition is stored as plain text (no base64)."""
        code = "def transform(x):\n    return x * 2\n"
        builder = self._make_builder_with_udf(code)
        self.assertEqual(builder._udf.function_definition, code)

    def test_dollar_sign_safe_in_to_json(self) -> None:
        """$$ in function definition does not appear in to_json() output,
        and json.loads() recovers the original $$ string."""
        code = "def f():\n    return '$$dangerous$$'\n"

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
                engine="python",
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
        code = "x = '$$$'\n"
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
                engine="python",
                output_columns=[("OUT", FloatType())],
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
        self.assertEqual(builder._udf.function_definition, code)

    def test_no_dollar_dollar_passes_through(self) -> None:
        """Code without $$ goes through to_json() unmodified (no spurious escaping)."""
        code = "def compute(x):\n    return x + 1\n"
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
                engine="python",
                output_columns=[("OUT", FloatType())],
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
        self.assertEqual(features[0].source_column.type, "FloatType")
        self.assertEqual(features[0].output_column.type, "FloatType")
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
                            StructField("ENRICHED_AMOUNT", FloatType()),
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
        self.assertEqual(features[0].source_column.type, "FloatType")

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
        builder.set_offline_configs([_batch_source_config()])
        builder.set_sources(
            [
                BatchSource(
                    schema=StructType(
                        [
                            StructField("USER_ID", StringType()),
                            StructField("REVENUE", FloatType()),
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
        self.assertEqual(features[0].source_column.type, "FloatType")

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
                    columns=StructType([StructField("AMOUNT", FloatType())]),
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
                    columns=StructType([StructField("AMOUNT", FloatType())]),
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
        self.assertEqual(features[0].source_column.type, "FloatType")
        self.assertEqual(features[0].output_column.type, "DecimalType")
        self.assertEqual(features[0].output_column.precision, 18)
        self.assertEqual(features[0].output_column.scale, 0)

    def test_avg_output_type_is_float(self) -> None:
        """AVG always produces FloatType regardless of source type."""
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
        self.assertEqual(features[0].output_column.type, "FloatType")
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
                    columns=StructType([StructField("SCORE", FloatType())]),
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
        self.assertEqual(features[0].output_column.type, "FloatType")

    def test_stddev_output_type_is_float(self) -> None:
        """STD always produces FloatType regardless of source type."""
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
        self.assertEqual(features[0].output_column.type, "FloatType")
        self.assertIsNone(features[0].output_column.precision)

    def test_var_output_type_is_float(self) -> None:
        """VAR always produces FloatType regardless of source type."""
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
        self.assertEqual(features[0].output_column.type, "FloatType")
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
        """APPROX_PERCENTILE always produces FloatType."""
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
        self.assertEqual(features[0].output_column.type, "FloatType")
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
                    columns=StructType([StructField("AMOUNT", FloatType())]),
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
            .set_offline_configs([_batch_source_config()])
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
            .set_offline_configs([_batch_source_config()])
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
            .set_sources([BatchSource(schema=StructType([StructField("X", FloatType())]))])
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
            .set_offline_configs([_batch_source_config()])
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_sources([BatchSource(schema=StructType([StructField("X", FloatType())]))])
        )
        with self.assertRaisesRegex(ValueError, "granularity_sec is required"):
            builder.build()

    def test_tiles_without_batch_source_rejected(self) -> None:
        """Batch tiled FV without a BatchSource in set_sources should fail validation."""
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
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
        builder = (
            self._base_builder()
            .set_offline_configs([_batch_source_config()])
            .set_properties(entity_columns=["USER_ID"])
            .set_udf(**_simple_udf_args())
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have offline configs"):
            builder.build()

    def test_request_only_without_features_ok(self) -> None:
        """Realtime FV with only a Request source (no Features) is valid."""
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
        ]
        # Should NOT raise — features source is optional
        result = builder.build()
        source_types = [s.source_type for s in result.spec.sources]
        self.assertEqual(source_types, [SourceType.REQUEST])

    def test_requires_request_source(self) -> None:
        """Only features, no request → rejected."""
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exactly 1 Request source"):
            builder.build()

    def test_multiple_request_sources_rejected(self) -> None:
        """More than 1 request source → rejected."""
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="r1",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="r2",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Z", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exactly 1 Request source.*got 2"):
            builder.build()

    def test_stream_source_rejected(self) -> None:
        """Stream sources not allowed in realtime FVs."""
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="stream",
                source_type=SourceType.STREAM,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Z", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have Stream sources"):
            builder.build()

    def test_batch_source_rejected(self) -> None:
        """Batch sources not allowed in realtime FVs."""
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"]).set_udf(**_simple_udf_args())
        builder._sources = [
            Source(
                name="batch",
                source_type=SourceType.BATCH,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="Y", type="FloatType")],
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have Batch sources"):
            builder.build()

    def test_requires_udf(self) -> None:
        builder = self._base_builder().set_properties(entity_columns=["USER_ID"])
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "requires a UDF"):
            builder.build()

    def test_no_agg_method(self) -> None:
        builder = (
            self._base_builder()
            .set_properties(
                entity_columns=["USER_ID"],
                agg_method=FeatureAggregationMethod.TILES,
            )
            .set_udf(**_simple_udf_args())
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have an agg_method"):
            builder.build()

    def test_no_granularity(self) -> None:
        builder = (
            self._base_builder()
            .set_properties(
                entity_columns=["USER_ID"],
                granularity="1h",
            )
            .set_udf(**_simple_udf_args())
        )
        builder._sources = [
            Source(
                name="request",
                source_type=SourceType.REQUEST,
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "must not have granularity_sec"):
            builder.build()


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
                columns=[FSColumn(name="X", type="FloatType")],
                source_version="v1",
            )
        ]
        with self.assertRaisesRegex(ValueError, "must not have source_version"):
            builder.build()

    def test_stream_source_with_selected_features_rejected(self) -> None:
        builder = self._streaming_builder()
        builder._sources = [
            Source(
                name="s",
                source_type=SourceType.STREAM,
                columns=[FSColumn(name="X", type="FloatType")],
                selected_features=["X"],
            )
        ]
        with self.assertRaisesRegex(ValueError, "must not have selected_features"):
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
                columns=[FSColumn(name="X", type="FloatType")],
                source_version="v1",
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
                source_version="v1",
            ),
        ]
        builder._udf = mock.MagicMock()
        builder.set_properties(entity_columns=["USER_ID"])
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
                columns=[FSColumn(name="X", type="FloatType")],
            ),
            Source(
                name="fv",
                source_type=SourceType.FEATURES,
                columns=[FSColumn(name="Y", type="FloatType")],
            ),
        ]
        builder._udf = mock.MagicMock()
        builder.set_properties(entity_columns=["USER_ID"])
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

    @parameterized.parameters(
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

    @parameterized.parameters(
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


if __name__ == "__main__":
    absltest.main()
