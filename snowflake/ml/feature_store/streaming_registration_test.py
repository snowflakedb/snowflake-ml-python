"""Unit tests for streaming_registration module."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
    _FeatureViewMetadata,
)
from snowflake.ml.feature_store.metadata_manager import StreamingMetadata
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.ml.feature_store.streaming_registration import (
    _build_streaming_feature_view_spec,
    _create_empty_table,
    cleanup_streaming_feature_view,
    run_streaming_postamble,
    run_streaming_preamble,
)
from snowflake.snowpark.types import (
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ============================================================================
# Helpers
# ============================================================================


def _sample_transform(df: pd.DataFrame) -> pd.DataFrame:
    """A sample transformation function for tests."""
    df["AMOUNT_CENTS"] = (df["AMOUNT"] * 100).astype(int)
    df["IS_LARGE"] = df["AMOUNT"] > 1000
    return df


def _make_stream_source() -> StreamSource:
    return StreamSource(
        name="txn_events",
        schema=StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        ),
        desc="Test stream source",
    )


def _make_entity() -> Entity:
    return Entity(name="user_entity", join_keys=["USER_ID"])


def _make_mock_backfill_df() -> MagicMock:
    mock_df = MagicMock()
    mock_df.queries = {"queries": ["SELECT * FROM src"]}
    schema = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("AMOUNT", DoubleType()),
            StructField("EVENT_TIME", TimestampType()),
        ]
    )
    mock_df.schema = schema
    mock_df.columns = [f.name for f in schema.fields]
    # postamble calls backfill_df.to_df([...]) to prefix input columns before map_in_pandas.
    renamed_df = MagicMock()
    renamed_df.queries = {"queries": ["SELECT * FROM renamed_src"]}
    mock_df.to_df.return_value = renamed_df
    return mock_df


# ============================================================================
# cleanup_streaming_feature_view tests
# ============================================================================


class CleanupStreamingTest(absltest.TestCase):
    """Tests for cleanup_streaming_feature_view."""

    def test_drops_udf_and_backfill_tables_and_decrements_ref(self) -> None:
        """Test that cleanup drops both udf_transformed and backfill tables and decrements ref count."""
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="my_fn",
        )

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1"))
        fv_metadata = _FeatureViewMetadata(
            entities=["USER_ENTITY"],
            timestamp_col="EVENT_TIME",
            is_streaming=True,
        )

        cleanup_streaming_feature_view(
            session=session,
            feature_view_name=feature_view_name,
            version="v1",
            fv_name="TEST_FV",
            fv_metadata=fv_metadata,
            metadata_manager=metadata_manager,
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
            telemetry_stmp={},
        )

        # Verify DROP TABLE was called for both udf_transformed and backfill tables
        drop_calls = [c for c in session.sql.call_args_list if "DROP TABLE IF EXISTS" in str(c)]
        self.assertEqual(len(drop_calls), 2)
        drop_sqls = [str(c) for c in drop_calls]
        self.assertTrue(any("$UDF_TRANSFORMED'" in s or '$UDF_TRANSFORMED"' in s for s in drop_sqls))
        self.assertTrue(any("$BACKFILL" in s for s in drop_sqls))

        # Verify ref count was decremented
        metadata_manager.decrement_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")

    def test_cleanup_no_streaming_metadata(self) -> None:
        """Test cleanup when streaming metadata is not found."""
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = None

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1"))
        fv_metadata = _FeatureViewMetadata(
            entities=["USER_ENTITY"],
            timestamp_col="EVENT_TIME",
            is_streaming=True,
        )

        # Should not raise
        cleanup_streaming_feature_view(
            session=session,
            feature_view_name=feature_view_name,
            version="v1",
            fv_name="TEST_FV",
            fv_metadata=fv_metadata,
            metadata_manager=metadata_manager,
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
            telemetry_stmp={},
        )

        # ref count should NOT be decremented
        metadata_manager.decrement_stream_source_ref_count.assert_not_called()


# ============================================================================
# FeatureView streaming validation tests
# ============================================================================


class FeatureViewStreamingValidationTest(absltest.TestCase):
    """Tests for FeatureView validation with stream_config."""

    def test_streaming_requires_timestamp_col(self) -> None:
        """Test that streaming FV without timestamp_col raises error."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        with self.assertRaisesRegex(ValueError, "timestamp_col"):
            FeatureView(
                name="test_fv",
                entities=[entity],
                stream_config=stream_config,
            )

    def test_streaming_rejects_feature_df(self) -> None:
        """Test that streaming FV with feature_df raises error."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        with self.assertRaisesRegex(ValueError, "feature_df and stream_config"):
            FeatureView(
                name="test_fv",
                entities=[entity],
                feature_df=_make_mock_backfill_df(),
                stream_config=stream_config,
                timestamp_col="EVENT_TIME",
            )

    def test_streaming_rejects_online_disable(self) -> None:
        """Test that streaming FV with online_config.enable=False raises error."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        with self.assertRaisesRegex(ValueError, "online to be enabled"):
            FeatureView(
                name="test_fv",
                entities=[entity],
                stream_config=stream_config,
                timestamp_col="EVENT_TIME",
                online_config=OnlineConfig(enable=False),
            )

    def test_streaming_auto_sets_online_config(self) -> None:
        """Test that streaming FV auto-sets online config to POSTGRES."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        self.assertTrue(fv.online)
        assert fv.online_config is not None
        self.assertEqual(fv.online_config.store_type, OnlineStoreType.POSTGRES)

    def test_is_streaming_property(self) -> None:
        """Test is_streaming property."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        self.assertTrue(fv.is_streaming)

    def test_non_streaming_not_streaming(self) -> None:
        """Test that a regular FV has is_streaming=False."""
        mock_df = _make_mock_backfill_df()
        entity = _make_entity()
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )
        self.assertFalse(fv.is_streaming)

    def test_streaming_defers_feature_df(self) -> None:
        """Test that streaming FV has feature_df=None at construction."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        self.assertIsNone(fv.feature_df)

    def test_metadata_includes_is_streaming(self) -> None:
        """Test that _metadata() includes is_streaming flag."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        metadata = fv._metadata()
        self.assertTrue(metadata.is_streaming)

    def test_ordered_entity_columns(self) -> None:
        """Test ordered_entity_columns property."""
        e1 = Entity(name="e1", join_keys=["A", "B"])
        e2 = Entity(name="e2", join_keys=["B", "C"])  # B is duplicated
        mock_df = _make_mock_backfill_df()
        mock_df.columns = ["A", "B", "C", "VAL"]
        mock_df.schema = StructType(
            [
                StructField("A", StringType()),
                StructField("B", StringType()),
                StructField("C", StringType()),
                StructField("VAL", DoubleType()),
            ]
        )
        fv = FeatureView(
            name="test_fv",
            entities=[e1, e2],
            feature_df=mock_df,
        )
        # B should appear only once, in order
        self.assertEqual(fv.ordered_entity_columns, ["A", "B", "C"])


# ============================================================================
# _FeatureViewMetadata backward compatibility tests
# ============================================================================


class FeatureViewMetadataStreamingTest(absltest.TestCase):
    """Tests for _FeatureViewMetadata with is_streaming field."""

    def test_from_json_without_is_streaming(self) -> None:
        """Test backward compatibility: old metadata without is_streaming."""
        old_json = json.dumps(
            {
                "entities": ["ENT1"],
                "timestamp_col": "TS",
                "is_tiled": False,
                "is_iceberg": False,
            }
        )
        metadata = _FeatureViewMetadata.from_json(old_json)
        self.assertFalse(metadata.is_streaming)

    def test_from_json_with_is_streaming(self) -> None:
        """Test parsing metadata with is_streaming=True."""
        new_json = json.dumps(
            {
                "entities": ["ENT1"],
                "timestamp_col": "TS",
                "is_tiled": False,
                "is_iceberg": False,
                "is_streaming": True,
            }
        )
        metadata = _FeatureViewMetadata.from_json(new_json)
        self.assertTrue(metadata.is_streaming)

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        metadata = _FeatureViewMetadata(
            entities=["ENT1", "ENT2"],
            timestamp_col="EVENT_TIME",
            is_streaming=True,
        )
        json_str = metadata.to_json()
        restored = _FeatureViewMetadata.from_json(json_str)
        self.assertTrue(restored.is_streaming)
        self.assertEqual(restored.entities, ["ENT1", "ENT2"])


# ============================================================================
# StreamingMetadata tests
# ============================================================================


class StreamingMetadataTest(absltest.TestCase):
    """Tests for StreamingMetadata dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Test conversion to dictionary without optional fields."""
        meta = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="normalize_txn",
        )
        d = meta.to_dict()
        self.assertEqual(d["stream_source_name"], "TXN_EVENTS")
        self.assertEqual(d["transformation_fn_name"], "normalize_txn")
        self.assertNotIn("backfill_start_time", d)
        self.assertNotIn("backfill_query_id", d)

    def test_to_dict_full(self) -> None:
        """Test conversion with all fields."""
        meta = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="normalize_txn",
            backfill_start_time="2024-06-01T00:00:00",
            backfill_query_id="abc-123",
        )
        d = meta.to_dict()
        self.assertEqual(d["backfill_start_time"], "2024-06-01T00:00:00")
        self.assertEqual(d["backfill_query_id"], "abc-123")

    def test_from_dict_minimal(self) -> None:
        """Test construction from dictionary without optional fields."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertEqual(meta.stream_source_name, "TXN_EVENTS")
        self.assertIsNone(meta.backfill_start_time)
        self.assertIsNone(meta.backfill_query_id)

    def test_from_dict_full(self) -> None:
        """Test construction with all fields."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
            "backfill_start_time": "2024-06-01T00:00:00",
            "backfill_query_id": "abc-123",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertEqual(meta.backfill_start_time, "2024-06-01T00:00:00")
        self.assertEqual(meta.backfill_query_id, "abc-123")

    def test_roundtrip(self) -> None:
        """Test dict roundtrip."""
        meta = StreamingMetadata(
            stream_source_name="SRC",
            transformation_fn_name="fn",
            backfill_start_time="2024-01-01T00:00:00",
            backfill_query_id="q-id",
        )
        restored = StreamingMetadata.from_dict(meta.to_dict())
        self.assertEqual(meta.stream_source_name, restored.stream_source_name)
        self.assertEqual(meta.backfill_query_id, restored.backfill_query_id)


# ============================================================================
# UDF transformed table naming tests
# ============================================================================


class UdfTransformedTableNameTest(absltest.TestCase):
    """Tests for FeatureView._get_udf_transformed_table_name."""

    def test_name_with_version(self) -> None:
        """Test naming convention: FV_NAME$VERSION$UDF_TRANSFORMED."""
        name = FeatureView._get_udf_transformed_table_name(SqlIdentifier("my_fv"), FeatureViewVersion("v1"))
        resolved = name.resolved()
        self.assertIn("$UDF_TRANSFORMED", resolved)
        self.assertIn("MY_FV", resolved)
        self.assertIn("v1", resolved)

    def test_name_from_physical_name(self) -> None:
        """Test naming from an already-computed physical name."""
        physical = FeatureView._get_physical_name(SqlIdentifier("my_fv"), FeatureViewVersion("v1"))
        udf_name = FeatureView._get_udf_transformed_table_name(physical)
        self.assertTrue(udf_name.resolved().endswith("$UDF_TRANSFORMED"))

    def test_mirrors_online_table_pattern(self) -> None:
        """Test that udf_transformed naming mirrors online table naming."""
        physical = FeatureView._get_physical_name(SqlIdentifier("fv"), FeatureViewVersion("v1"))
        online = FeatureView._get_online_table_name(physical)
        udf = FeatureView._get_udf_transformed_table_name(physical)

        online_prefix = online.resolved().replace("$ONLINE", "")
        udf_prefix = udf.resolved().replace("$UDF_TRANSFORMED", "")
        self.assertEqual(online_prefix, udf_prefix)


# ============================================================================
# _initialize_from_feature_df tests
# ============================================================================


class InitializeFromFeatureDfTest(absltest.TestCase):
    """Tests for FeatureView._initialize_from_feature_df."""

    def test_sets_all_derived_fields(self) -> None:
        """Test that _initialize_from_feature_df sets feature_df, query, feature_desc, cluster_by."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        # Before: everything is deferred
        self.assertIsNone(fv.feature_df)
        self.assertEqual(fv._query, "")
        self.assertIsNone(fv._feature_desc)

        # Simulate what registration does: create a mock "udf_transformed" df
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM DB.SCH.UDF_TABLE"]}
        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT_CENTS", DoubleType()),
                StructField("IS_LARGE", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df.schema = schema
        mock_udf_df.columns = [f.name for f in schema.fields]

        fv._initialize_from_feature_df(mock_udf_df)

        # After: all fields derived from the transformed schema
        self.assertIs(fv.feature_df, mock_udf_df)
        self.assertIn("SELECT * FROM DB.SCH.UDF_TABLE", fv.query)
        self.assertIsNotNone(fv._feature_desc)
        assert fv._feature_desc is not None
        feature_names = list(fv._feature_desc.keys())
        self.assertIn(SqlIdentifier("AMOUNT_CENTS"), feature_names)
        self.assertIn(SqlIdentifier("IS_LARGE"), feature_names)


# ============================================================================
# Streaming + Features (tiled) tests
# ============================================================================


class StreamingWithFeaturesTest(absltest.TestCase):
    """Tests for streaming FVs with tiled aggregation features."""

    def test_streaming_with_features_sets_aggregation_specs(self) -> None:
        """Test that features param is captured for streaming FVs."""
        from snowflake.ml.feature_store.feature import Feature

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[
                Feature.sum("AMOUNT_CENTS", "24h").alias("total_amount_24h"),
                Feature.count("AMOUNT_CENTS", "1h").alias("txn_count_1h"),
            ],
        )
        self.assertTrue(fv.is_streaming)
        self.assertTrue(fv.is_tiled)
        self.assertEqual(fv.feature_granularity, "1h")
        self.assertIsNotNone(fv.aggregation_specs)
        assert fv.aggregation_specs is not None
        self.assertEqual(len(fv.aggregation_specs), 2)

    def test_streaming_without_features_not_tiled(self) -> None:
        """Test that streaming FV without features is not tiled."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )
        self.assertTrue(fv.is_streaming)
        self.assertFalse(fv.is_tiled)
        self.assertIsNone(fv.feature_granularity)
        self.assertIsNone(fv.aggregation_specs)

    def test_streaming_with_granularity_requires_features(self) -> None:
        """Test that feature_granularity without features raises error after init_from_df.

        Note: This validation runs in _validate() which is called after
        _initialize_from_feature_df during registration. At construction time,
        streaming FVs skip _validate().
        """
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        # Construction succeeds (validation deferred)
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            # features=None — missing!
        )
        # After _initialize_from_feature_df + _validate(), it would fail
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df.columns = ["USER_ID", "EVENT_TIME"]

        fv._initialize_from_feature_df(mock_udf_df)

        with self.assertRaisesRegex(ValueError, "feature_granularity requires features"):
            fv._validate()


# ============================================================================
# run_streaming_preamble tests
# ============================================================================


class RunStreamingPreambleTest(absltest.TestCase):
    """Tests for run_streaming_preamble."""

    def _make_streaming_fv(
        self, backfill_df: MagicMock, backfill_start_time: datetime.datetime | None = None
    ) -> FeatureView:
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=backfill_df,
            backfill_start_time=backfill_start_time,
        )
        return FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

    def _make_probe_pdf(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "USER_ID": ["u1", "u2"],
                "AMOUNT": [100.0, 200.0],
                "EVENT_TIME": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "AMOUNT_CENTS": [10000, 20000],
                "IS_LARGE": [False, False],
            }
        )

    def test_basic_preamble(self) -> None:
        """Preamble probes schema, creates table, returns result."""
        backfill_df = _make_mock_backfill_df()
        probe_pdf = self._make_probe_pdf()
        backfill_df.limit.return_value.to_pandas.return_value = probe_pdf

        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()
        stream_source = _make_stream_source()

        result = run_streaming_preamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            overwrite=False,
            metadata_manager=metadata_manager,
            telemetry_stmp={},
            get_stream_source_fn=lambda name: stream_source,
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
        )

        self.assertIn("$UDF_TRANSFORMED", result.fq_udf_table)
        self.assertIn("$BACKFILL", result.fq_backfill_table)
        self.assertIn("$UDF_TRANSFORMED$BACKFILL", result.fq_backfill_table)
        self.assertEqual(result.resolved_source_name, "TXN_EVENTS")
        # Verify CREATE TABLE was called for both tables
        create_calls = [c for c in session.sql.call_args_list if "CREATE" in str(c)]
        self.assertEqual(len(create_calls), 2)

    def test_preamble_with_overwrite_decrements_old_ref(self) -> None:
        """On overwrite, preamble decrements old stream source ref count."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.limit.return_value.to_pandas.return_value = self._make_probe_pdf()

        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="OLD_SOURCE",
            transformation_fn_name="old_fn",
        )

        run_streaming_preamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            overwrite=True,
            metadata_manager=metadata_manager,
            telemetry_stmp={},
            get_stream_source_fn=lambda name: _make_stream_source(),
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
        )

        metadata_manager.decrement_stream_source_ref_count.assert_called_once_with("OLD_SOURCE")

    def test_preamble_no_overwrite_skips_decrement(self) -> None:
        """Without overwrite, preamble does not touch ref count."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.limit.return_value.to_pandas.return_value = self._make_probe_pdf()

        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        run_streaming_preamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            overwrite=False,
            metadata_manager=metadata_manager,
            telemetry_stmp={},
            get_stream_source_fn=lambda name: _make_stream_source(),
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
        )

        metadata_manager.decrement_stream_source_ref_count.assert_not_called()

    def test_preamble_backfill_start_time_applies_filter(self) -> None:
        """backfill_start_time causes F.col filter on backfill_df."""
        backfill_df = _make_mock_backfill_df()
        filtered_df = MagicMock()
        backfill_df.filter.return_value = filtered_df
        filtered_df.limit.return_value.to_pandas.return_value = self._make_probe_pdf()

        start_time = datetime.datetime(2024, 6, 1)
        fv = self._make_streaming_fv(backfill_df, backfill_start_time=start_time)
        session = MagicMock()
        metadata_manager = MagicMock()

        run_streaming_preamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            overwrite=False,
            metadata_manager=metadata_manager,
            telemetry_stmp={},
            get_stream_source_fn=lambda name: _make_stream_source(),
            get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
        )

        backfill_df.filter.assert_called_once()

    def test_preamble_empty_probe_raises(self) -> None:
        """Empty probe DataFrame raises ValueError."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.limit.return_value.to_pandas.return_value = pd.DataFrame()

        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        with self.assertRaisesRegex(ValueError, "zero rows"):
            run_streaming_preamble(
                session=session,
                feature_view=fv,
                version=FeatureViewVersion("v1"),
                feature_view_name=SqlIdentifier("TEST_FV$v1"),
                overwrite=False,
                metadata_manager=metadata_manager,
                telemetry_stmp={},
                get_stream_source_fn=lambda name: _make_stream_source(),
                get_fully_qualified_name_fn=lambda name: f"DB.SCH.{name}",
            )


# ============================================================================
# run_streaming_postamble tests
# ============================================================================


class RunStreamingPostambleTest(absltest.TestCase):
    """Tests for run_streaming_postamble."""

    @patch("snowflake.snowpark.dataframe.map_in_pandas")
    def test_postamble_saves_metadata_and_increments_ref(self, mock_map: MagicMock) -> None:
        """Postamble saves metadata, increments ref count, kicks off INSERT ALL backfill."""
        backfill_df = _make_mock_backfill_df()
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=backfill_df,
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

        session = MagicMock()
        session.table.return_value.schema = StructType(
            [StructField("USER_ID", StringType()), StructField("AMOUNT_CENTS", DoubleType())]
        )
        metadata_manager = MagicMock()

        # mock map_in_pandas return — queries provides the SELECT SQL
        transformed_df = MagicMock()
        transformed_df.queries = {"queries": ["SELECT udtf(...) FROM source"]}
        mock_map.return_value = transformed_df
        async_job = MagicMock()
        async_job.query_id = "test-query-id"
        session.sql.return_value.collect.return_value = async_job

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        run_streaming_postamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            preamble=preamble,
            metadata_manager=metadata_manager,
        )

        # Verify INSERT ALL SQL was submitted
        insert_all_call = session.sql.call_args
        insert_sql = insert_all_call[0][0]
        self.assertIn("INSERT ALL", insert_sql)
        self.assertIn("DB.SCH.UDF_TABLE", insert_sql)
        self.assertIn("DB.SCH.UDF_TABLE$BACKFILL", insert_sql)

        metadata_manager.save_streaming_metadata.assert_called_once()
        saved_meta = metadata_manager.save_streaming_metadata.call_args
        self.assertEqual(saved_meta.kwargs["metadata"].stream_source_name, "TXN_EVENTS")
        self.assertEqual(saved_meta.kwargs["metadata"].transformation_fn_name, "_sample_transform")
        self.assertEqual(saved_meta.kwargs["metadata"].backfill_query_id, "test-query-id")

        metadata_manager.increment_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")

    @patch("snowflake.snowpark.dataframe.map_in_pandas")
    def test_postamble_renames_input_columns_with_prefix(self, mock_map: MagicMock) -> None:
        """Postamble prefixes backfill_df columns for map_in_pandas and unprefixes in the wrapper."""
        backfill_df = _make_mock_backfill_df()
        renamed_backfill_df = MagicMock()
        renamed_backfill_df.queries = {"queries": ["SELECT * FROM renamed_src"]}
        backfill_df.to_df.return_value = renamed_backfill_df

        captured = {}

        def user_fn(df: pd.DataFrame) -> pd.DataFrame:
            captured["user_fn_columns"] = list(df.columns)
            df["AMOUNT_CENTS"] = (df["AMOUNT"] * 100).astype(int)
            df["IS_LARGE"] = df["AMOUNT"] > 1000
            return df

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=user_fn,
            backfill_df=backfill_df,
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

        session = MagicMock()
        session.table.return_value.schema = StructType(
            [StructField("USER_ID", StringType()), StructField("AMOUNT_CENTS", DoubleType())]
        )
        metadata_manager = MagicMock()

        transformed_df = MagicMock()
        transformed_df.queries = {"queries": ["SELECT udtf(...) FROM source"]}
        mock_map.return_value = transformed_df
        async_job = MagicMock()
        async_job.query_id = "q"
        session.sql.return_value.collect.return_value = async_job

        from snowflake.ml.feature_store.streaming_registration import (
            _BACKFILL_INPUT_PREFIX,
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        run_streaming_postamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            preamble=preamble,
            metadata_manager=metadata_manager,
        )

        backfill_df.to_df.assert_called_once()
        new_names = backfill_df.to_df.call_args[0][0]
        expected_names = [f"{_BACKFILL_INPUT_PREFIX}{c}" for c in backfill_df.columns]
        self.assertEqual(new_names, expected_names)

        called_df = mock_map.call_args[0][0]
        self.assertIs(called_df, renamed_backfill_df)

        wrapper_fn = mock_map.call_args[0][1]
        prefixed_pdf = pd.DataFrame(
            {
                f"{_BACKFILL_INPUT_PREFIX}USER_ID": ["u1", "u2"],
                f"{_BACKFILL_INPUT_PREFIX}AMOUNT": [100.0, 2000.0],
                f"{_BACKFILL_INPUT_PREFIX}EVENT_TIME": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        out = list(wrapper_fn(iter([prefixed_pdf])))
        self.assertEqual(len(out), 1)
        self.assertEqual(captured["user_fn_columns"], ["USER_ID", "AMOUNT", "EVENT_TIME"])
        self.assertIn("AMOUNT_CENTS", out[0].columns)
        self.assertIn("IS_LARGE", out[0].columns)

    @patch("snowflake.snowpark.dataframe.map_in_pandas")
    def test_postamble_with_backfill_start_time(self, mock_map: MagicMock) -> None:
        """Postamble filters backfill_df and saves start_time in metadata."""
        backfill_df = _make_mock_backfill_df()
        filtered_df = MagicMock()
        backfill_df.filter.return_value = filtered_df

        start_time = datetime.datetime(2024, 6, 1)
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=backfill_df,
            backfill_start_time=start_time,
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

        session = MagicMock()
        session.table.return_value.schema = StructType([StructField("C", StringType())])
        metadata_manager = MagicMock()

        transformed_df = MagicMock()
        transformed_df.queries = {"queries": ["SELECT udtf(...) FROM src"]}
        mock_map.return_value = transformed_df
        async_job = MagicMock()
        async_job.query_id = "q-id"
        session.sql.return_value.collect.return_value = async_job

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.T",
            fq_backfill_table="DB.SCH.T$BACKFILL",
            resolved_source_name="SRC",
        )

        run_streaming_postamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            preamble=preamble,
            metadata_manager=metadata_manager,
        )

        backfill_df.filter.assert_called_once()
        saved_meta = metadata_manager.save_streaming_metadata.call_args.kwargs["metadata"]
        self.assertEqual(saved_meta.backfill_start_time, "2024-06-01T00:00:00")


# ============================================================================
# _create_empty_table tests
# ============================================================================


class CreateEmptyTableTest(absltest.TestCase):
    """Tests for _create_empty_table SQL generation."""

    def test_creates_table_with_correct_ddl(self) -> None:
        """Verify generated SQL has correct column definitions."""
        session = MagicMock()
        schema = StructType(
            [
                StructField("USER_ID", StringType(16777216)),
                StructField("AMOUNT", DoubleType()),
                StructField("TS", TimestampType()),
            ]
        )

        _create_empty_table(
            session=session,
            fq_table_name="DB.SCH.MY_TABLE",
            schema=schema,
            overwrite=False,
            telemetry_stmp={"key": "val"},
        )

        sql_arg = session.sql.call_args[0][0]
        self.assertIn("CREATE TABLE DB.SCH.MY_TABLE", sql_arg)
        self.assertIn('"USER_ID" VARCHAR(16777216)', sql_arg)
        self.assertIn('"AMOUNT" FLOAT', sql_arg)
        self.assertIn('"TS" TIMESTAMP_NTZ', sql_arg)
        self.assertNotIn("OR REPLACE", sql_arg)

    def test_overwrite_adds_or_replace(self) -> None:
        """overwrite=True adds OR REPLACE clause."""
        session = MagicMock()
        schema = StructType([StructField("COL", StringType())])

        _create_empty_table(
            session=session,
            fq_table_name="DB.SCH.T",
            schema=schema,
            overwrite=True,
            telemetry_stmp={},
        )

        sql_arg = session.sql.call_args[0][0]
        self.assertIn("CREATE OR REPLACE TABLE", sql_arg)


# ============================================================================
# cleanup exception handling tests
# ============================================================================


class CleanupExceptionHandlingTest(absltest.TestCase):
    """Tests for exception handling in cleanup_streaming_feature_view."""

    def test_drop_table_failure_logged_not_raised(self) -> None:
        """DROP TABLE failure is caught and logged, not raised."""
        session = MagicMock()
        session.sql.return_value.collect.side_effect = Exception("DROP failed")
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = None

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("fv"), FeatureViewVersion("v1"))
        fv_metadata = _FeatureViewMetadata(entities=["E"], timestamp_col="TS", is_streaming=True)

        # Should not raise
        cleanup_streaming_feature_view(
            session=session,
            feature_view_name=feature_view_name,
            version="v1",
            fv_name="FV",
            fv_metadata=fv_metadata,
            metadata_manager=metadata_manager,
            get_fully_qualified_name_fn=lambda n: f"DB.SCH.{n}",
            telemetry_stmp={},
        )

    def test_decrement_failure_logged_not_raised(self) -> None:
        """Decrement failure is caught and logged, not raised."""
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="SRC",
            transformation_fn_name="fn",
        )
        metadata_manager.decrement_stream_source_ref_count.side_effect = Exception("decrement failed")

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("fv"), FeatureViewVersion("v1"))
        fv_metadata = _FeatureViewMetadata(entities=["E"], timestamp_col="TS", is_streaming=True)

        # Should not raise
        cleanup_streaming_feature_view(
            session=session,
            feature_view_name=feature_view_name,
            version="v1",
            fv_name="FV",
            fv_metadata=fv_metadata,
            metadata_manager=metadata_manager,
            get_fully_qualified_name_fn=lambda n: f"DB.SCH.{n}",
            telemetry_stmp={},
        )


# ============================================================================
# _build_streaming_feature_view_spec tests
# ============================================================================


class BuildStreamingSpecTest(absltest.TestCase):
    """Tests for _build_streaming_feature_view_spec."""

    def _make_non_tiled_streaming_fv(self) -> FeatureView:
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        return FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
        )

    def _make_tiled_streaming_fv(self) -> FeatureView:
        from snowflake.ml.feature_store.feature import Feature

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        return FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[
                Feature.sum("AMOUNT", "24h").alias("total_amount_24h"),
                Feature.count("AMOUNT", "1h").alias("txn_count_1h"),
            ],
        )

    def test_non_tiled_builds_valid_spec(self) -> None:
        """Non-tiled streaming FV spec has 1 offline config (UDF_TRANSFORMED)."""
        fv = self._make_non_tiled_streaming_fv()
        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT_CENTS", DecimalType(38, 0)),
                StructField("IS_LARGE", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        # Initialize with schema so output_schema is populated
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        spec = _build_streaming_feature_view_spec(
            feature_view=fv,
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            version="v1",
            target_lag="0 seconds",
            stream_source=stream_source,
            udf_transformed_schema=udf_schema,
            database="DB",
            schema="SCH",
        )
        self.assertIsNotNone(spec)
        spec_dict = spec.to_dict()
        offline_configs = spec_dict["offline_configs"]
        self.assertEqual(len(offline_configs), 1)
        self.assertEqual(offline_configs[0]["table_type"], "UDFTransformed")

    def test_tiled_builds_spec_with_two_offline_configs(self) -> None:
        """Tiled streaming FV spec has 2 offline configs (UDF_TRANSFORMED + TILED)."""
        fv = self._make_tiled_streaming_fv()
        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
                StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
            ]
        )
        spec = _build_streaming_feature_view_spec(
            feature_view=fv,
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            version="v1",
            target_lag="0 seconds",
            stream_source=stream_source,
            udf_transformed_schema=udf_schema,
            tiled_materialized_schema=tiled_dt_schema,
            database="DB",
            schema="SCH",
        )
        self.assertIsNotNone(spec)
        spec_dict = spec.to_dict()
        offline_configs = spec_dict["offline_configs"]
        self.assertEqual(len(offline_configs), 2)
        table_types = {c["table_type"] for c in offline_configs}
        self.assertEqual(table_types, {"UDFTransformed", "Tiled"})

    def test_tiled_spec_has_features(self) -> None:
        """Tiled streaming FV spec includes aggregation features."""
        fv = self._make_tiled_streaming_fv()
        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
                StructField("_PARTIAL_COUNT_AMOUNT", LongType()),
            ]
        )
        spec = _build_streaming_feature_view_spec(
            feature_view=fv,
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            version="v1",
            target_lag="0 seconds",
            stream_source=stream_source,
            udf_transformed_schema=udf_schema,
            tiled_materialized_schema=tiled_dt_schema,
            database="DB",
            schema="SCH",
        )
        spec_dict = spec.to_dict()
        features = spec_dict["spec"]["features"]
        self.assertEqual(len(features), 2)
        output_col_names = {f["output_column"]["name"] for f in features}
        self.assertIn("TOTAL_AMOUNT_24H", output_col_names)
        self.assertIn("TXN_COUNT_1H", output_col_names)

    def test_tiled_requires_materialized_dt_schema(self) -> None:
        fv = self._make_tiled_streaming_fv()
        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        with self.assertRaisesRegex(ValueError, "tiled_materialized_schema"):
            _build_streaming_feature_view_spec(
                feature_view=fv,
                feature_view_name=SqlIdentifier("TEST_FV$v1"),
                version="v1",
                target_lag="0 seconds",
                stream_source=stream_source,
                udf_transformed_schema=udf_schema,
                database="DB",
                schema="SCH",
            )

    def test_non_tiled_rejects_tiled_materialized_schema(self) -> None:
        fv = self._make_non_tiled_streaming_fv()
        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)
        fake_tiled = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
            ]
        )
        with self.assertRaisesRegex(ValueError, "must not set tiled_materialized_schema"):
            _build_streaming_feature_view_spec(
                feature_view=fv,
                feature_view_name=SqlIdentifier("TEST_FV$v1"),
                version="v1",
                target_lag="0 seconds",
                stream_source=stream_source,
                udf_transformed_schema=udf_schema,
                tiled_materialized_schema=fake_tiled,
                database="DB",
                schema="SCH",
            )

    def test_tiled_spec_uses_fv_aggregation_method(self) -> None:
        """Tiled streaming FV spec uses the FV's feature_aggregation_method, not hardcoded TILES."""
        from snowflake.ml.feature_store.feature import Feature
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[
                Feature.sum("AMOUNT", "24h").alias("total_amount_24h"),
            ],
            feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        )
        self.assertEqual(fv.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
            ]
        )
        spec = _build_streaming_feature_view_spec(
            feature_view=fv,
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            version="v1",
            target_lag="0 seconds",
            stream_source=stream_source,
            udf_transformed_schema=udf_schema,
            tiled_materialized_schema=tiled_dt_schema,
            database="DB",
            schema="SCH",
        )
        spec_dict = spec.to_dict()
        self.assertEqual(spec_dict["spec"]["feature_aggregation_method"], "continuous")

    def test_tiled_spec_default_uses_tiles(self) -> None:
        """Tiled streaming FV without explicit method defaults to TILES in the spec."""
        from snowflake.ml.feature_store.feature import Feature
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[
                Feature.sum("AMOUNT", "24h").alias("total_amount_24h"),
            ],
        )
        self.assertEqual(fv.feature_aggregation_method, FeatureAggregationMethod.TILES)

        stream_source = _make_stream_source()
        udf_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        mock_udf_df = MagicMock()
        mock_udf_df.queries = {"queries": ["SELECT * FROM TBL"]}
        mock_udf_df.schema = udf_schema
        mock_udf_df.columns = [f.name for f in udf_schema.fields]
        fv._initialize_from_feature_df(mock_udf_df)

        tiled_dt_schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("TILE_START", TimestampType()),
                StructField("_PARTIAL_SUM_AMOUNT", DoubleType()),
            ]
        )
        spec = _build_streaming_feature_view_spec(
            feature_view=fv,
            feature_view_name=SqlIdentifier("TEST_FV$v1"),
            version="v1",
            target_lag="0 seconds",
            stream_source=stream_source,
            udf_transformed_schema=udf_schema,
            tiled_materialized_schema=tiled_dt_schema,
            database="DB",
            schema="SCH",
        )
        spec_dict = spec.to_dict()
        self.assertEqual(spec_dict["spec"]["feature_aggregation_method"], "tiles")


if __name__ == "__main__":
    absltest.main()
