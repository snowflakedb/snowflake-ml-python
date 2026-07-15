"""Unit tests for streaming_registration module."""

import datetime
import json
from unittest.mock import MagicMock

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
    return mock_df


# ============================================================================
# cleanup_streaming_feature_view tests
# ============================================================================


class CleanupStreamingTest(absltest.TestCase):
    """Tests for cleanup_streaming_feature_view."""

    def test_drops_udf_and_backfill_tables_and_decrements_ref(self) -> None:
        """Cleanup drops both tables, decrements ref count, and skips task DDL when no task names are recorded."""
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

        # No backfill tasks recorded -> no ALTER TASK / DROP TASK calls.
        sql_strs = [str(c) for c in session.sql.call_args_list]
        self.assertFalse(any("ALTER TASK" in s for s in sql_strs))
        self.assertFalse(any("DROP TASK" in s for s in sql_strs))

        # Verify ref count was decremented
        metadata_manager.decrement_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")

    def test_safety_net_drops_leftover_backfill_tasks_and_proc(self) -> None:
        """When task + proc names are present in metadata, cleanup suspends the root,
        drops finalizer + root in order, and drops the backing procedure (safety net)."""
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="my_fn",
            backfill_root_task_name="DB.SCH.TEST_FV$v1$BACKFILL_ROOT",
            backfill_finalize_task_name="DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE",
            backfill_proc_name="DB.SCH.TEST_FV$v1$BACKFILL_PROC",
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

        sql_strs = [str(c.args[0]) for c in session.sql.call_args_list]

        suspend_idx = next(i for i, s in enumerate(sql_strs) if "ALTER TASK IF EXISTS" in s and "SUSPEND" in s)
        finalize_drop_idx = next(
            i for i, s in enumerate(sql_strs) if "DROP TASK IF EXISTS" in s and "$BACKFILL_FINALIZE" in s
        )
        root_drop_idx = next(i for i, s in enumerate(sql_strs) if "DROP TASK IF EXISTS" in s and "$BACKFILL_ROOT" in s)
        proc_drop_idx = next(
            i for i, s in enumerate(sql_strs) if "DROP PROCEDURE IF EXISTS" in s and "$BACKFILL_PROC" in s
        )

        # SUSPEND root, then drop finalizer, then drop root, then drop proc.
        self.assertLess(suspend_idx, finalize_drop_idx)
        self.assertLess(finalize_drop_idx, root_drop_idx)
        self.assertLess(root_drop_idx, proc_drop_idx)
        self.assertIn("DB.SCH.TEST_FV$v1$BACKFILL_ROOT", sql_strs[suspend_idx])
        # Proc drop carries the full signature so Snowflake disambiguates correctly.
        self.assertIn("(TIMESTAMP_NTZ, TIMESTAMP_NTZ)", sql_strs[proc_drop_idx])

        # Tables are still dropped after task cleanup.
        table_drops = [s for s in sql_strs if "DROP TABLE IF EXISTS" in s]
        self.assertEqual(len(table_drops), 2)
        # Ref count is still decremented.
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
        self.assertNotIn("backfill_root_task_name", d)
        self.assertNotIn("backfill_finalize_task_name", d)
        self.assertNotIn("backfill_proc_name", d)
        self.assertNotIn("backfill_udtf_name", d)
        self.assertNotIn("backfill_udtf_signature", d)
        self.assertNotIn("backfill_state", d)
        self.assertNotIn("backfill_query_id", d)

    def test_to_dict_full(self) -> None:
        """Test conversion with all fields."""
        meta = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="normalize_txn",
            backfill_start_time="2024-06-01T00:00:00",
            backfill_root_task_name="DB.SCH.TEST_FV$v1$BACKFILL_ROOT",
            backfill_finalize_task_name="DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE",
            backfill_proc_name="DB.SCH.TEST_FV$v1$BACKFILL_PROC",
            backfill_udtf_name="DB.SCH.TEST_FV$v1$BACKFILL_UDTF",
            backfill_udtf_signature="(VARCHAR, TIMESTAMP_NTZ, FLOAT)",
            backfill_state="RUNNING",
        )
        d = meta.to_dict()
        self.assertEqual(d["backfill_start_time"], "2024-06-01T00:00:00")
        self.assertEqual(d["backfill_root_task_name"], "DB.SCH.TEST_FV$v1$BACKFILL_ROOT")
        self.assertEqual(d["backfill_finalize_task_name"], "DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE")
        self.assertEqual(d["backfill_proc_name"], "DB.SCH.TEST_FV$v1$BACKFILL_PROC")
        # UDTF name + signature must round-trip: rollback issues
        # ``DROP FUNCTION <name><signature>`` and Snowflake disambiguates
        # overloads by argument types.
        self.assertEqual(d["backfill_udtf_name"], "DB.SCH.TEST_FV$v1$BACKFILL_UDTF")
        self.assertEqual(d["backfill_udtf_signature"], "(VARCHAR, TIMESTAMP_NTZ, FLOAT)")
        self.assertEqual(d["backfill_state"], "RUNNING")
        self.assertNotIn("backfill_query_id", d)

    def test_from_dict_minimal(self) -> None:
        """Test construction from dictionary without optional fields."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertEqual(meta.stream_source_name, "TXN_EVENTS")
        self.assertIsNone(meta.backfill_start_time)
        self.assertIsNone(meta.backfill_root_task_name)
        self.assertIsNone(meta.backfill_finalize_task_name)
        self.assertIsNone(meta.backfill_proc_name)
        self.assertIsNone(meta.backfill_udtf_name)
        self.assertIsNone(meta.backfill_udtf_signature)
        self.assertIsNone(meta.backfill_state)

    def test_from_dict_full(self) -> None:
        """Test construction with all fields."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
            "backfill_start_time": "2024-06-01T00:00:00",
            "backfill_root_task_name": "DB.SCH.TEST_FV$v1$BACKFILL_ROOT",
            "backfill_finalize_task_name": "DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE",
            "backfill_proc_name": "DB.SCH.TEST_FV$v1$BACKFILL_PROC",
            "backfill_udtf_name": "DB.SCH.TEST_FV$v1$BACKFILL_UDTF",
            "backfill_udtf_signature": "(VARCHAR, TIMESTAMP_NTZ, FLOAT)",
            "backfill_state": "COMPLETED",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertEqual(meta.backfill_start_time, "2024-06-01T00:00:00")
        self.assertEqual(meta.backfill_root_task_name, "DB.SCH.TEST_FV$v1$BACKFILL_ROOT")
        self.assertEqual(meta.backfill_finalize_task_name, "DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE")
        self.assertEqual(meta.backfill_proc_name, "DB.SCH.TEST_FV$v1$BACKFILL_PROC")
        self.assertEqual(meta.backfill_udtf_name, "DB.SCH.TEST_FV$v1$BACKFILL_UDTF")
        self.assertEqual(meta.backfill_udtf_signature, "(VARCHAR, TIMESTAMP_NTZ, FLOAT)")
        self.assertEqual(meta.backfill_state, "COMPLETED")

    def test_from_dict_drops_unknown_backfill_state(self) -> None:
        """Forward-compat: an unrecognized ``backfill_state`` value is dropped to None."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
            "backfill_state": "BOGUS_FUTURE_STATE",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertIsNone(meta.backfill_state)

    def test_from_dict_ignores_legacy_backfill_query_id(self) -> None:
        """Legacy rows that still carry a `backfill_query_id` key load cleanly without it."""
        d = {
            "stream_source_name": "TXN_EVENTS",
            "transformation_fn_name": "normalize_txn",
            # Old metadata schema; should be silently ignored.
            "backfill_query_id": "legacy-q-id",
        }
        meta = StreamingMetadata.from_dict(d)
        self.assertEqual(meta.stream_source_name, "TXN_EVENTS")
        self.assertFalse(hasattr(meta, "backfill_query_id"))
        self.assertIsNone(meta.backfill_root_task_name)
        self.assertIsNone(meta.backfill_proc_name)

    def test_roundtrip(self) -> None:
        """Test dict roundtrip."""
        meta = StreamingMetadata(
            stream_source_name="SRC",
            transformation_fn_name="fn",
            backfill_start_time="2024-01-01T00:00:00",
            backfill_root_task_name="DB.SCH.FV$v1$BACKFILL_ROOT",
            backfill_finalize_task_name="DB.SCH.FV$v1$BACKFILL_FINALIZE",
            backfill_proc_name="DB.SCH.FV$v1$BACKFILL_PROC",
            backfill_udtf_name="DB.SCH.FV$v1$BACKFILL_UDTF",
            backfill_udtf_signature="(VARCHAR, FLOAT)",
            backfill_state="FAILED",
        )
        restored = StreamingMetadata.from_dict(meta.to_dict())
        self.assertEqual(meta.stream_source_name, restored.stream_source_name)
        self.assertEqual(meta.backfill_root_task_name, restored.backfill_root_task_name)
        self.assertEqual(meta.backfill_finalize_task_name, restored.backfill_finalize_task_name)
        self.assertEqual(meta.backfill_proc_name, restored.backfill_proc_name)
        # Both UDTF fields must round-trip: cleanup paths combine the name
        # with the signature to form the ``DROP FUNCTION <name><sig>`` SQL,
        # so a partial round-trip would leak the UDTF on FV deletion.
        self.assertEqual(meta.backfill_udtf_name, restored.backfill_udtf_name)
        self.assertEqual(meta.backfill_udtf_signature, restored.backfill_udtf_signature)
        self.assertEqual(meta.backfill_state, restored.backfill_state)


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
        """Streaming FV with feature_granularity but no features raises ValueError at construction."""
        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="src",
            transformation_fn=_sample_transform,
            backfill_df=_make_mock_backfill_df(),
        )
        with self.assertRaisesRegex(
            ValueError, "feature_granularity and feature_aggregation_method require features to be set."
        ):
            FeatureView(
                name="test_fv",
                entities=[entity],
                stream_config=stream_config,
                timestamp_col="EVENT_TIME",
                feature_granularity="1h",
                # features=None — missing!
            )


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

    def test_preamble_overwrite_suspends_and_drops_old_backfill_objects(self) -> None:
        """On overwrite, preamble suspends the old root task and drops finalizer + root + proc."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.limit.return_value.to_pandas.return_value = self._make_probe_pdf()

        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()
        metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="OLD_SOURCE",
            transformation_fn_name="old_fn",
            backfill_root_task_name="DB.SCH.TEST_FV$v1$BACKFILL_ROOT",
            backfill_finalize_task_name="DB.SCH.TEST_FV$v1$BACKFILL_FINALIZE",
            backfill_proc_name="DB.SCH.TEST_FV$v1$BACKFILL_PROC",
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

        sql_strs = [str(c.args[0]) for c in session.sql.call_args_list]
        suspend_idx = next(i for i, s in enumerate(sql_strs) if "ALTER TASK IF EXISTS" in s and "SUSPEND" in s)
        finalize_drop_idx = next(
            i for i, s in enumerate(sql_strs) if "DROP TASK IF EXISTS" in s and "$BACKFILL_FINALIZE" in s
        )
        root_drop_idx = next(i for i, s in enumerate(sql_strs) if "DROP TASK IF EXISTS" in s and "$BACKFILL_ROOT" in s)
        proc_drop_idx = next(
            i for i, s in enumerate(sql_strs) if "DROP PROCEDURE IF EXISTS" in s and "$BACKFILL_PROC" in s
        )
        # SUSPEND root, then drop finalizer, then drop root, then drop proc.
        self.assertLess(suspend_idx, finalize_drop_idx)
        self.assertLess(finalize_drop_idx, root_drop_idx)
        self.assertLess(root_drop_idx, proc_drop_idx)
        self.assertIn("(TIMESTAMP_NTZ, TIMESTAMP_NTZ)", sql_strs[proc_drop_idx])
        # Old ref count still decremented.
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
        # Real Snowpark .filter() preserves schema; mirror that for the mock.
        filtered_df.schema = backfill_df.schema
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

    def test_preamble_rejects_tz_timestamp_in_backfill_df(self) -> None:
        """Backfill source with TIMESTAMP_TZ is rejected at registration time."""
        from snowflake.snowpark.types import TimestampTimeZone

        backfill_df = _make_mock_backfill_df()
        backfill_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType(TimestampTimeZone.TZ)),
            ]
        )
        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        with self.assertRaisesRegex(ValueError, "Only TIMESTAMP_NTZ is supported"):
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
        # Validation must fire before any DDL.
        session.sql.assert_not_called()

    def test_preamble_rejects_backfill_missing_stream_source_column(self) -> None:
        """Backfill missing a column declared by the stream source is rejected before any DDL/probe."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("EVENT_TIME", TimestampType()),
                # AMOUNT (declared by stream source) is missing.
            ]
        )
        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        with self.assertRaisesRegex(
            ValueError,
            r"streaming feature view: backfill_df is missing column 'AMOUNT' "
            r"declared by StreamSource 'TXN_EVENTS'\.",
        ):
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
        # Validation must fire before any DDL or probe.
        session.sql.assert_not_called()
        backfill_df.limit.assert_not_called()

    def test_preamble_rejects_backfill_type_mismatch(self) -> None:
        """Backfill column whose type disagrees with stream source is rejected before any DDL/probe."""
        backfill_df = _make_mock_backfill_df()
        # Stream source declares AMOUNT as DoubleType; backfill has it as StringType.
        backfill_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", StringType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        with self.assertRaisesRegex(
            ValueError,
            r"backfill_df column 'AMOUNT' has type StringType but "
            r"StreamSource 'TXN_EVENTS' declares column 'AMOUNT' with type DoubleType\.",
        ):
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
        session.sql.assert_not_called()
        backfill_df.limit.assert_not_called()

    def test_preamble_allows_backfill_with_extra_columns(self) -> None:
        """Backfill with extra columns beyond the stream source schema is accepted."""
        backfill_df = _make_mock_backfill_df()
        backfill_df.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
                StructField("EXTRA_DERIVATION_COL", DoubleType()),  # not in stream source
            ]
        )
        backfill_df.limit.return_value.to_pandas.return_value = self._make_probe_pdf()
        fv = self._make_streaming_fv(backfill_df)
        session = MagicMock()
        metadata_manager = MagicMock()

        # Should not raise.
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
    """Tests for run_streaming_postamble.

    The postamble flow:
      1. Build an inline ``CREATE TEMPORARY FUNCTION`` with the user
         transformation source baked in.
      2. Build the ``INSERT ALL ... TABLE(<udtf>(...))`` SELECT (vectorized
         ``process`` UDTF — no ``OVER`` clause).
      3. Wrap (1) + (2) in a ``CREATE OR REPLACE PROCEDURE`` (SQL Scripting,
         EXECUTE AS OWNER).
      4. Create a root + finalizer task graph; the root task body is a
         ``CALL <fq_proc>(<start>, SYSDATE())`` and the finalizer drops
         the proc + UDTF + both tasks.
      5. Resume the finalizer, then ``SYSTEM$TASK_DEPENDENTS_ENABLE`` on
         the root to start the graph.
    """

    _METADATA_TABLE_PATH = "DB.SCH._FEATURE_STORE_METADATA"

    def _fq(self, name: object) -> str:
        return f"DB.SCH.{name}"

    def _make_session_with_udf_schema(self) -> MagicMock:
        session = MagicMock()
        session.table.return_value.schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT_CENTS", DoubleType()),
                StructField("IS_LARGE", DoubleType()),
            ]
        )
        return session

    def _make_metadata_manager(self) -> MagicMock:
        """Stub metadata manager whose ``table_path`` is a real string.

        Without this, the finalizer's ``UPDATE <table>`` interpolates a
        ``<MagicMock id=...>`` representation that produces invalid SQL.
        """
        metadata_manager = MagicMock()
        metadata_manager.table_path = self._METADATA_TABLE_PATH
        return metadata_manager

    def test_postamble_creates_proc_then_task_graph_and_saves_metadata(self) -> None:
        """Postamble emits proc DDL, then the root + finalizer, then activates the graph,
        and persists all three names (proc + 2 tasks) in StreamingMetadata."""
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
            warehouse="my_wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1"))

        result = run_streaming_postamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=feature_view_name,
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=None,
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
        )

        sql_calls = [str(c.args[0]) for c in session.sql.call_args_list]

        # 1a. The per-FV permanent UDTF DDL is emitted first. Snowflake
        #     disallows CREATE TEMPORARY FUNCTION inside a stored procedure
        #     (even via EXECUTE IMMEDIATE), so the UDTF is created as a
        #     permanent function up-front and dropped by the finalizer.
        udtf_idx = next(i for i, s in enumerate(sql_calls) if s.startswith("CREATE OR REPLACE FUNCTION"))
        udtf_sql = sql_calls[udtf_idx]
        self.assertIn("$BACKFILL_UDTF", udtf_sql)
        self.assertIn("LANGUAGE PYTHON", udtf_sql)
        self.assertIn("_sf_vectorized_input", udtf_sql)
        self.assertIn("def _sample_transform", udtf_sql)

        # 1b. The procedure DDL is emitted next; its body just does the
        #     INSERT ALL referencing the UDTF (no embedded Python source).
        proc_idx = next(i for i, s in enumerate(sql_calls) if s.startswith("CREATE OR REPLACE PROCEDURE"))
        proc_sql = sql_calls[proc_idx]
        self.assertIn("$BACKFILL_PROC", proc_sql)
        self.assertIn("(WINDOW_START TIMESTAMP_NTZ, WINDOW_END TIMESTAMP_NTZ)", proc_sql)
        self.assertIn("LANGUAGE SQL", proc_sql)
        self.assertIn("EXECUTE AS OWNER", proc_sql)
        self.assertIn("INSERT ALL INTO DB.SCH.UDF_TABLE INTO DB.SCH.UDF_TABLE$BACKFILL", proc_sql)
        self.assertNotIn("OVER (PARTITION BY", proc_sql)
        # Proc body references the UDTF by fully-qualified name.
        self.assertIn("$BACKFILL_UDTF", proc_sql)
        # No embedded Python source in the proc body itself.
        self.assertNotIn("def _sample_transform", proc_sql)
        self.assertNotIn("LANGUAGE PYTHON", proc_sql)
        self.assertNotIn("EXECUTE IMMEDIATE", proc_sql)
        # The window-guard WHERE clause references the proc's parameters
        # using the ``:NAME`` bind syntax (bare identifiers would resolve
        # as column references in a SELECT and trigger
        # ``invalid identifier`` at compile time).
        self.assertIn(":WINDOW_START IS NULL OR", proc_sql)
        self.assertIn(":WINDOW_END IS NULL OR", proc_sql)
        # UDTF must be created before the proc references it.
        self.assertLess(udtf_idx, proc_idx)

        def _create_task_for_suffix(suffix: str) -> tuple[int, str]:
            matches = []
            for i, sql in enumerate(sql_calls):
                if not sql.startswith("CREATE OR REPLACE TASK"):
                    continue
                first_line = sql.split("\n", 1)[0]
                if suffix in first_line:
                    matches.append((i, sql))
            self.assertEqual(len(matches), 1, f"expected exactly one CREATE TASK for {suffix}, got {len(matches)}")
            return matches[0]

        # 2. Root task: schedule, single CALL of the proc with NULL/SYSDATE args.
        root_idx, root_sql = _create_task_for_suffix("$BACKFILL_ROOT")
        self.assertIn("SCHEDULE = '10 SECONDS'", root_sql)
        self.assertIn("USER_TASK_MINIMUM_TRIGGER_INTERVAL_IN_SECONDS = 10", root_sql)
        self.assertIn("SUSPEND_TASK_AFTER_NUM_FAILURES = 1", root_sql)
        # No backfill_start_time -> first arg is NULL; the second is the
        # window upper bound. Use SYSDATE() (UTC TIMESTAMP_NTZ) rather
        # than CURRENT_TIMESTAMP()::TIMESTAMP_NTZ (TIMESTAMP_LTZ cast,
        # depends on session/account TZ).
        self.assertIn("CALL ", root_sql)
        self.assertIn("$BACKFILL_PROC", root_sql)
        self.assertIn("(NULL, SYSDATE())", root_sql)
        self.assertNotIn("CURRENT_TIMESTAMP", root_sql)
        self.assertIn("WHEN OTHER THEN", root_sql)
        self.assertIn("RAISE;", root_sql)
        # FV warehouse is preferred over the default.
        self.assertIn("WAREHOUSE = MY_WH", root_sql)

        # 3. Finalizer: FINALIZE = <root>; each cleanup step is wrapped
        # in its own BEGIN..EXCEPTION block so a partial failure does
        # not skip later steps. Order: backfill_state UPDATE first
        # (terminal-state write), then suspend root, drop proc, drop UDTF,
        # drop root, drop finalizer (self).
        finalize_idx, finalize_sql = _create_task_for_suffix("$BACKFILL_FINALIZE")
        self.assertIn("FINALIZE =", finalize_sql)
        self.assertIn("$BACKFILL_ROOT", finalize_sql)
        self.assertIn("ALTER TASK", finalize_sql)
        self.assertIn(" SUSPEND", finalize_sql)
        self.assertIn("DROP PROCEDURE IF EXISTS", finalize_sql)
        self.assertIn("(TIMESTAMP_NTZ, TIMESTAMP_NTZ)", finalize_sql)
        self.assertIn("DROP FUNCTION IF EXISTS", finalize_sql)
        self.assertIn("$BACKFILL_UDTF", finalize_sql)
        # Terminal-state UPDATE: writes COMPLETED/FAILED into the FV's
        # streaming-metadata row by setting the ``backfill_state`` key
        # in the JSON blob.
        self.assertIn(f"UPDATE {self._METADATA_TABLE_PATH}", finalize_sql)
        self.assertIn("OBJECT_INSERT(METADATA, 'backfill_state'", finalize_sql)
        # The COMPLETED/FAILED selection comes from a TASK_HISTORY count
        # comparison (every expected user-visible task succeeded => COMPLETED).
        self.assertIn("INFORMATION_SCHEMA.TASK_HISTORY", finalize_sql)
        self.assertIn("'COMPLETED'", finalize_sql)
        self.assertIn("'FAILED'", finalize_sql)
        # The match set is the FV's user-visible backfill tasks (root +
        # any future windows), not the finalizer.
        self.assertIn("$BACKFILL_ROOT", finalize_sql)
        self.assertIn("$BACKFILL_W%", finalize_sql)
        self.assertNotIn("$BACKFILL_FINALIZE'", finalize_sql)  # not in match set
        # Each cleanup statement is in its own EXCEPTION-swallowing block;
        # 6 steps (state update + suspend + drop proc + drop UDTF + drop
        # root + drop finalizer).
        self.assertGreaterEqual(finalize_sql.count("EXCEPTION WHEN OTHER THEN NULL"), 6)
        # Drop order: state-update -> suspend -> proc -> udtf -> root ->
        # finalizer (self-drop last so a self-drop failure cannot strand
        # earlier cleanup work).
        state_update_pos = finalize_sql.index("OBJECT_INSERT")
        suspend_pos = finalize_sql.index(" SUSPEND")
        proc_drop_pos = finalize_sql.index("DROP PROCEDURE IF EXISTS")
        udtf_drop_pos = finalize_sql.index("DROP FUNCTION IF EXISTS")
        first_task_drop_pos = finalize_sql.index("DROP TASK IF EXISTS")
        second_task_drop_pos = finalize_sql.index("DROP TASK IF EXISTS", first_task_drop_pos + 1)
        self.assertLess(state_update_pos, suspend_pos)
        self.assertLess(suspend_pos, proc_drop_pos)
        self.assertLess(proc_drop_pos, udtf_drop_pos)
        self.assertLess(udtf_drop_pos, first_task_drop_pos)
        self.assertIn("$BACKFILL_ROOT", finalize_sql[first_task_drop_pos:second_task_drop_pos])
        self.assertIn("$BACKFILL_FINALIZE", finalize_sql[second_task_drop_pos:])

        # Order of DDL emission: udtf before proc before root before finalizer.
        self.assertLess(udtf_idx, proc_idx)
        self.assertLess(proc_idx, root_idx)
        self.assertLess(root_idx, finalize_idx)

        # 4. Activation: RESUME finalizer, then SYSTEM$TASK_DEPENDENTS_ENABLE on root.
        resume_idx = next(
            i for i, s in enumerate(sql_calls) if "ALTER TASK" in s and "$BACKFILL_FINALIZE" in s and "RESUME" in s
        )
        enable_idx = next(i for i, s in enumerate(sql_calls) if "SYSTEM$TASK_DEPENDENTS_ENABLE" in s)
        self.assertLess(finalize_idx, resume_idx)
        self.assertLess(resume_idx, enable_idx)

        # 5. Returned names are populated and fully qualified.
        self.assertIn("$BACKFILL_ROOT", result.fq_backfill_root_task)
        self.assertIn("$BACKFILL_FINALIZE", result.fq_backfill_finalize_task)
        self.assertIn("$BACKFILL_PROC", result.fq_backfill_proc)
        self.assertIn("$BACKFILL_UDTF", result.fq_backfill_udtf)
        self.assertTrue(result.fq_backfill_root_task.startswith("DB.SCH."))
        self.assertTrue(result.fq_backfill_finalize_task.startswith("DB.SCH."))
        self.assertTrue(result.fq_backfill_proc.startswith("DB.SCH."))
        self.assertTrue(result.fq_backfill_udtf.startswith("DB.SCH."))
        # Signature is well-formed and parenthesized for direct interpolation.
        self.assertTrue(result.fq_backfill_udtf_signature.startswith("("))
        self.assertTrue(result.fq_backfill_udtf_signature.endswith(")"))

        # 6a. Activation order: the resume SQLs must run AFTER
        # ``save_streaming_metadata``. If the order were reversed, the
        # finalizer could fire and try to ``UPDATE`` a metadata row that
        # does not yet exist (writing nothing — leaving the FV stuck
        # showing ``backfill_state='RUNNING'`` forever once the client
        # finally writes its row, since the finalizer has already run).
        # ``session.sql`` and ``metadata_manager`` are independent mocks,
        # so the order is asserted indirectly: the postamble's last two
        # ``session.sql`` statements must be the RESUME + ENABLE pair, and
        # ``save_streaming_metadata`` must have been called by the time
        # the postamble returns. The implementation guarantees no SQL is
        # emitted between ``save_streaming_metadata`` and the
        # resume-enable pair.
        finalize_resume_idx = sql_calls.index(
            next(s for s in sql_calls if "$BACKFILL_FINALIZE" in s and " RESUME" in s)
        )
        self.assertEqual(
            finalize_resume_idx,
            len(sql_calls) - 2,
            "finalizer RESUME must be the second-to-last SQL, after metadata save",
        )
        self.assertTrue(
            sql_calls[-1].startswith("SELECT SYSTEM$TASK_DEPENDENTS_ENABLE"),
            f"final SQL must be SYSTEM$TASK_DEPENDENTS_ENABLE; got {sql_calls[-1]!r}",
        )
        # ``save_streaming_metadata`` was definitely invoked (raises
        # StopIteration otherwise).
        next(c for c in metadata_manager.mock_calls if c[0] == "save_streaming_metadata")

        # 6b. Metadata: stream source + task names + proc name + UDTF name + state persisted.
        metadata_manager.save_streaming_metadata.assert_called_once()
        saved_meta = metadata_manager.save_streaming_metadata.call_args.kwargs["metadata"]
        self.assertEqual(saved_meta.stream_source_name, "TXN_EVENTS")
        self.assertEqual(saved_meta.transformation_fn_name, "_sample_transform")
        self.assertEqual(saved_meta.backfill_root_task_name, result.fq_backfill_root_task)
        self.assertEqual(saved_meta.backfill_finalize_task_name, result.fq_backfill_finalize_task)
        self.assertEqual(saved_meta.backfill_proc_name, result.fq_backfill_proc)
        self.assertEqual(saved_meta.backfill_udtf_name, result.fq_backfill_udtf)
        self.assertEqual(saved_meta.backfill_udtf_signature, result.fq_backfill_udtf_signature)
        # Initial state is RUNNING; the finalizer flips it to COMPLETED/FAILED later.
        self.assertEqual(saved_meta.backfill_state, "RUNNING")

        metadata_manager.increment_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")

    def test_postamble_invokes_resource_callback_in_creation_order(self) -> None:
        """``on_resource_created`` fires once per backfill resource, in creation order.

        The contract guards against rollback leaks: each resource must be
        registered with the caller **before** the next DDL runs, so that
        a failure mid-postamble (UDTF created but proc/task DDL fails, or
        proc + tasks created but ``save_streaming_metadata`` throws) still
        leaves a complete cleanup trail.
        """
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
            warehouse="my_wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1"))

        events: list[tuple[str, str]] = []

        result = run_streaming_postamble(
            session=session,
            feature_view=fv,
            version=FeatureViewVersion("v1"),
            feature_view_name=feature_view_name,
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=None,
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
            on_resource_created=lambda kind, fq: events.append((kind, fq)),
        )

        # All four backfill resources are reported, exactly once each, in
        # the dependency order (UDTF first, then proc, then root task,
        # then finalizer task).
        self.assertEqual(
            [k for k, _ in events],
            ["BACKFILL_UDTF", "BACKFILL_PROC", "BACKFILL_ROOT_TASK", "BACKFILL_FINALIZE_TASK"],
        )
        # The UDTF event includes the argument-type signature appended to
        # the FQN so the rollback path can ``DROP FUNCTION`` directly
        # without re-deriving input column types from the FV schema.
        udtf_event = events[0][1]
        self.assertTrue(udtf_event.startswith(result.fq_backfill_udtf))
        self.assertTrue(udtf_event.endswith(result.fq_backfill_udtf_signature))
        # Proc and task events are bare fully-qualified names (no signature).
        self.assertEqual(events[1][1], result.fq_backfill_proc)
        self.assertEqual(events[2][1], result.fq_backfill_root_task)
        self.assertEqual(events[3][1], result.fq_backfill_finalize_task)

    def test_postamble_callback_fires_before_metadata_save(self) -> None:
        """If ``save_streaming_metadata`` raises, all created resources have already been reported.

        This is the rollback-leak regression guard: previously, the caller
        only learned about the UDTF/proc/tasks via the function's return
        value, which is never delivered when an exception propagates from
        ``save_streaming_metadata`` or ``increment_stream_source_ref_count``.
        With ``on_resource_created`` the caller registers each resource
        the moment its DDL succeeds, so the outer ``except`` rollback can
        drop them.
        """
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
            warehouse="my_wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()
        # Force the post-DDL metadata save to fail; the rollback contract
        # demands that all four backfill resources still show up in
        # ``events`` so the caller can drop them.
        metadata_manager.save_streaming_metadata.side_effect = RuntimeError("boom")

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        feature_view_name = FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1"))

        events: list[tuple[str, str]] = []

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_streaming_postamble(
                session=session,
                feature_view=fv,
                version=FeatureViewVersion("v1"),
                feature_view_name=feature_view_name,
                preamble=preamble,
                metadata_manager=metadata_manager,
                default_warehouse=None,
                get_fully_qualified_name_fn=self._fq,
                telemetry_stmp={},
                on_resource_created=lambda kind, fq: events.append((kind, fq)),
            )

        # Even though the postamble raised, every DDL-created resource has
        # already been reported, so the caller's ``created_resources``
        # tracker is complete.
        self.assertEqual(
            [k for k, _ in events],
            ["BACKFILL_UDTF", "BACKFILL_PROC", "BACKFILL_ROOT_TASK", "BACKFILL_FINALIZE_TASK"],
        )
        # The ref-count increment must NOT have run, since metadata save
        # failed first.
        metadata_manager.increment_stream_source_ref_count.assert_not_called()
        # The graph must NOT have been resumed: resume happens only after
        # the metadata row (with backfill_state='RUNNING') has been written,
        # so the finalizer can never race ahead of the row it must update.
        sql_calls = [str(c.args[0]) for c in session.sql.call_args_list]
        for sql in sql_calls:
            self.assertFalse(
                "$BACKFILL_FINALIZE" in sql and " RESUME" in sql,
                f"finalizer was resumed despite metadata save failure: {sql!r}",
            )
            self.assertFalse(
                "SYSTEM$TASK_DEPENDENTS_ENABLE" in sql,
                f"task graph was enabled despite metadata save failure: {sql!r}",
            )

    def test_postamble_uses_default_warehouse_when_fv_has_none(self) -> None:
        """Without an explicit FV warehouse, the FS default warehouse is used for both tasks."""
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

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

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
            feature_view_name=FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1")),
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=SqlIdentifier("default_wh"),
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
        )

        sql_calls = [str(c.args[0]) for c in session.sql.call_args_list]
        for sql in sql_calls:
            if sql.startswith("CREATE OR REPLACE TASK"):
                self.assertIn("WAREHOUSE = DEFAULT_WH", sql)

    def test_postamble_prefers_initialization_warehouse_for_backfill(self) -> None:
        """The backfill task graph runs on the initialization warehouse when set,
        since the backfill is the streaming FV's one-time, full-scan initialization."""
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
            warehouse="small_wh",
            initialization_warehouse="large_wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

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
            feature_view_name=FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1")),
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=SqlIdentifier("default_wh"),
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
        )

        sql_calls = [str(c.args[0]) for c in session.sql.call_args_list]
        for sql in sql_calls:
            if sql.startswith("CREATE OR REPLACE TASK"):
                self.assertIn("WAREHOUSE = LARGE_WH", sql)
                self.assertNotIn("WAREHOUSE = SMALL_WH", sql)

    def test_postamble_no_warehouse_raises(self) -> None:
        """Postamble raises if neither the FV nor the FS provides a warehouse."""
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

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

        from snowflake.ml.feature_store.streaming_registration import (
            StreamingPreambleResult,
        )

        preamble = StreamingPreambleResult(
            fq_udf_table="DB.SCH.UDF_TABLE",
            fq_backfill_table="DB.SCH.UDF_TABLE$BACKFILL",
            resolved_source_name="TXN_EVENTS",
        )

        with self.assertRaisesRegex(ValueError, "No warehouse available"):
            run_streaming_postamble(
                session=session,
                feature_view=fv,
                version=FeatureViewVersion("v1"),
                feature_view_name=FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1")),
                preamble=preamble,
                metadata_manager=metadata_manager,
                default_warehouse=None,
                get_fully_qualified_name_fn=self._fq,
                telemetry_stmp={},
            )

    def test_postamble_with_backfill_start_time(self) -> None:
        """``backfill_start_time`` is rendered as a TO_TIMESTAMP_NTZ literal in the root CALL,
        and the ISO timestamp is preserved in metadata. The dataframe is not pre-filtered;
        the proc's WHERE clause does the filtering server-side."""
        backfill_df = _make_mock_backfill_df()

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
            warehouse="wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

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
            feature_view_name=FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1")),
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=None,
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
        )

        sql_calls = [str(c.args[0]) for c in session.sql.call_args_list]
        # The dataframe-level filter is NOT applied client-side anymore.
        backfill_df.filter.assert_not_called()

        root_sql = next(s for s in sql_calls if s.startswith("CREATE OR REPLACE TASK") and "$BACKFILL_ROOT" in s)
        # The literal must round-trip into the CALL exactly.
        self.assertIn("TO_TIMESTAMP_NTZ('2024-06-01T00:00:00')", root_sql)
        self.assertIn("SYSDATE()", root_sql)

        saved_meta = metadata_manager.save_streaming_metadata.call_args.kwargs["metadata"]
        self.assertEqual(saved_meta.backfill_start_time, "2024-06-01T00:00:00")

    def test_postamble_renders_user_fn_inside_udtf(self) -> None:
        """The per-FV UDTF body embeds the user's transformation source verbatim.

        Snowflake disallows ``CREATE TEMPORARY FUNCTION`` from inside a
        stored procedure, so the UDTF is created as a permanent function
        (named ``<fv>$<ver>$BACKFILL_UDTF``) at registration time. The
        finalizer task drops it as part of cleanup. This test verifies the
        UDTF DDL contains the user's source and the vectorized handler
        class — the proc body itself only contains the ``INSERT ALL`` that
        references the UDTF.
        """
        backfill_df = _make_mock_backfill_df()

        def my_custom_transform(df: pd.DataFrame) -> pd.DataFrame:
            df["AMOUNT_CENTS"] = (df["AMOUNT"] * 100).astype(int)
            df["IS_LARGE"] = df["AMOUNT"] > 1000
            return df

        entity = _make_entity()
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=my_custom_transform,
            backfill_df=backfill_df,
        )
        fv = FeatureView(
            name="test_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            warehouse="wh",
        )

        session = self._make_session_with_udf_schema()
        metadata_manager = self._make_metadata_manager()

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
            feature_view_name=FeatureView._get_physical_name(SqlIdentifier("test_fv"), FeatureViewVersion("v1")),
            preamble=preamble,
            metadata_manager=metadata_manager,
            default_warehouse=None,
            get_fully_qualified_name_fn=self._fq,
            telemetry_stmp={},
        )

        udtf_sql = next(
            str(c.args[0])
            for c in session.sql.call_args_list
            if str(c.args[0]).startswith("CREATE OR REPLACE FUNCTION")
        )
        self.assertIn("$BACKFILL_UDTF", udtf_sql)
        self.assertIn("LANGUAGE PYTHON", udtf_sql)
        self.assertIn("def my_custom_transform", udtf_sql)
        # Vectorized handler that calls the user fn by name.
        self.assertIn("my_custom_transform(df)", udtf_sql)
        self.assertIn("class _Handler", udtf_sql)
        self.assertIn("_sf_vectorized_input", udtf_sql)
        # The proc body itself only references the UDTF, no Python source.
        proc_sql = next(
            str(c.args[0])
            for c in session.sql.call_args_list
            if str(c.args[0]).startswith("CREATE OR REPLACE PROCEDURE")
        )
        self.assertNotIn("def my_custom_transform", proc_sql)
        self.assertNotIn("class _Handler", proc_sql)
        self.assertIn("$BACKFILL_UDTF", proc_sql)

    def test_postamble_without_timestamp_col_omits_window_filter(self) -> None:
        """When ``feature_view.timestamp_col`` is None, the proc SELECT has no WHERE.

        Streaming FVs require a timestamp_col today, but the SELECT-builder
        already supports the no-timestamp case for forward compatibility.
        We exercise it here by calling the renderer directly.
        """
        from snowflake.ml.feature_store.streaming_registration import (
            _render_backfill_insert_all_sql,
        )

        sql_with_ts = _render_backfill_insert_all_sql(
            fq_udf_table="DB.SCH.T",
            fq_backfill_table="DB.SCH.T$BACKFILL",
            fq_udtf="DB.SCH.UDTF",
            backfill_source_select="SELECT * FROM SRC",
            input_col_names=["USER_ID", "AMOUNT", "EVENT_TIME"],
            input_col_types=["VARCHAR", "FLOAT", "TIMESTAMP_NTZ"],
            output_col_names=["USER_ID", "AMOUNT_CENTS"],
            timestamp_col="EVENT_TIME",
        )
        # The window predicate must be applied INSIDE the source subquery
        # (not as an outer WHERE after the implicit lateral join with the
        # UDTF). An outer WHERE would still be correct, but it would
        # waste compute by feeding out-of-window rows through
        # ``transformation_fn``.
        self.assertIn("WHERE", sql_with_ts)
        # Predicate is qualified by the inner alias ``_bf`` so it stays
        # unambiguous if the source query is ever altered to emit a join.
        self.assertIn('_bf."EVENT_TIME" >= :WINDOW_START', sql_with_ts)
        self.assertIn('_bf."EVENT_TIME" < :WINDOW_END', sql_with_ts)
        # The predicate must NOT be qualified by the outer source alias
        # ``s.``: that would mean the WHERE landed after the UDTF
        # cross-join, the bug we're guarding against. ``s."EVENT_TIME"``
        # is allowed elsewhere (it's a UDTF input arg).
        self.assertNotIn('s."EVENT_TIME" >=', sql_with_ts)
        self.assertNotIn('s."EVENT_TIME" <', sql_with_ts)
        # The UDTF call site comes AFTER the filtered source subquery.
        self.assertLess(sql_with_ts.index("WHERE"), sql_with_ts.index("TABLE("))

        sql_no_ts = _render_backfill_insert_all_sql(
            fq_udf_table="DB.SCH.T",
            fq_backfill_table="DB.SCH.T$BACKFILL",
            fq_udtf="DB.SCH.UDTF",
            backfill_source_select="SELECT * FROM SRC",
            input_col_names=["USER_ID", "AMOUNT"],
            input_col_types=["VARCHAR", "FLOAT"],
            output_col_names=["USER_ID", "AMOUNT_CENTS"],
            timestamp_col=None,
        )
        self.assertNotIn(" WHERE ", sql_no_ts)
        self.assertNotIn("WINDOW_START", sql_no_ts)
        self.assertNotIn("WINDOW_END", sql_no_ts)
        # Both still produce the comma-join ``s, TABLE(udtf(...))`` shape
        # with no ``OVER`` clause — the documented call form for a
        # vectorized ``process`` UDTF.
        for sql in (sql_with_ts, sql_no_ts):
            self.assertIn("INSERT ALL INTO DB.SCH.T INTO DB.SCH.T$BACKFILL", sql)
            self.assertIn("TABLE(", sql)
            self.assertNotIn("LATERAL TABLE(", sql)
            self.assertNotIn("OVER (PARTITION BY", sql)

    def test_render_backfill_insert_all_casts_udtf_args_to_declared_types(self) -> None:
        """UDTF args are cast to their declared param type to bridge precision mismatches."""
        from snowflake.ml.feature_store.streaming_registration import (
            _render_backfill_insert_all_sql,
        )

        sql = _render_backfill_insert_all_sql(
            fq_udf_table="DB.SCH.T",
            fq_backfill_table="DB.SCH.T$BACKFILL",
            fq_udtf="DB.SCH.UDTF",
            backfill_source_select="SELECT * FROM SRC",
            input_col_names=["USER_ID", "EVENT_TIME", "AMOUNT"],
            input_col_types=["VARCHAR", "TIMESTAMP_NTZ", "FLOAT"],
            output_col_names=["USER_ID"],
            timestamp_col=None,
        )
        self.assertIn('s."USER_ID"::VARCHAR', sql)
        self.assertIn('s."EVENT_TIME"::TIMESTAMP_NTZ', sql)
        self.assertIn('s."AMOUNT"::FLOAT', sql)
        self.assertNotIn('UDTF(s."USER_ID",', sql)
        self.assertNotIn(', s."EVENT_TIME",', sql)

    def test_render_backfill_insert_all_rejects_mismatched_input_lengths(self) -> None:
        """``input_col_names`` and ``input_col_types`` must be the same length."""
        from snowflake.ml.feature_store.streaming_registration import (
            _render_backfill_insert_all_sql,
        )

        with self.assertRaisesRegex(ValueError, "same length"):
            _render_backfill_insert_all_sql(
                fq_udf_table="DB.SCH.T",
                fq_backfill_table="DB.SCH.T$BACKFILL",
                fq_udtf="DB.SCH.UDTF",
                backfill_source_select="SELECT * FROM SRC",
                input_col_names=["A", "B"],
                input_col_types=["VARCHAR"],
                output_col_names=["A"],
                timestamp_col=None,
            )

    def test_render_backfill_udtf_rejects_dollar_quote_collision(self) -> None:
        """The renderer refuses to embed user source containing ``$$``.

        The UDTF Python body is wrapped in ``$$ ... $$`` (Snowflake's only
        dollar-quote variant), so any ``$$`` in the user code would
        terminate the literal early and corrupt the DDL.
        """
        from snowflake.ml.feature_store.streaming_registration import (
            _render_backfill_udtf_sql,
        )

        bad_source = "def f(df):\n    return df  # contains $$\n"
        with self.assertRaisesRegex(ValueError, "reserved sentinel"):
            _render_backfill_udtf_sql(
                fq_udtf="DB.SCH.UDTF",
                input_col_names=["A"],
                input_col_types=["NUMBER(38,0)"],
                output_col_names=["A"],
                output_col_types=["NUMBER(38,0)"],
                user_fn_name="f",
                user_fn_source=bad_source,
            )


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


# ============================================================================
# delete_feature_view orphan-cleanup tests
# ============================================================================


def _create_feature_store_with_mocks() -> MagicMock:
    """Create a FeatureStore instance with mocked internals (bypasses __init__)."""
    from snowflake.ml.feature_store.feature_store import (
        FeatureStore,
        _FeatureStoreConfig,
    )

    fs = object.__new__(FeatureStore)
    fs._session = MagicMock()
    fs._session.get_current_role.return_value = "ROLE_1"
    fs._session.get_current_warehouse.return_value = "WH_1"
    # Default every SQL query (e.g. SHOW TASKS) to an empty result so probes resolve to
    # "nothing found" unless a test opts in by overriding collect's return value.
    fs._session.sql.return_value.collect.return_value = []
    fs._metadata_manager = MagicMock()
    fs._config = _FeatureStoreConfig(
        database=SqlIdentifier("TEST_DB"),
        schema=SqlIdentifier("TEST_SCHEMA"),
    )
    fs._default_warehouse = SqlIdentifier("WH_1")
    fs._telemetry_stmp = {}
    fs._default_iceberg_external_volume = None
    fs._asof_join_enabled = None
    # Needed by _find_object (used to probe for orphaned online feature tables).
    fs._obj_search_spaces = {
        "DYNAMIC TABLES": (fs._config.full_schema_path, "TABLE"),
        "VIEWS": (fs._config.full_schema_path, "TABLE"),
        "ONLINE FEATURE TABLES": (fs._config.full_schema_path, "TABLE"),
        "TASKS": (fs._config.full_schema_path, "TASK"),
    }
    return fs  # type: ignore[return-value]


class DeleteOrphanedStreamingResourcesTest(absltest.TestCase):
    """Tests for delete_feature_view when the backing DT is already gone."""

    def test_cleanup_orphaned_returns_true_when_streaming_meta_exists(self) -> None:
        """_cleanup_orphaned_feature_view_resources returns True and runs full cleanup."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="my_fn",
        )

        result = fs._cleanup_orphaned_feature_view_resources("MY_FV", "V1")

        self.assertTrue(result)
        # get_streaming_metadata is consulted both by the orphan helper's own guard and by
        # cleanup_streaming_feature_view, so assert it was called with the right args (not once).
        fs._metadata_manager.get_streaming_metadata.assert_any_call("MY_FV", "V1")
        fs._metadata_manager.decrement_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")
        fs._metadata_manager.delete_feature_view_metadata.assert_called_once_with("MY_FV", "V1")

    def test_cleanup_orphaned_returns_false_when_nothing_exists(self) -> None:
        """_cleanup_orphaned_feature_view_resources returns False when there is no streaming
        metadata, no companion refresh task, no snapshot table, and no online feature table
        to clean up."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_streaming_metadata.return_value = None

        with patch.object(fs, "_find_object", return_value=[]):
            result = fs._cleanup_orphaned_feature_view_resources("MY_FV", "V1")

        self.assertFalse(result)
        fs._metadata_manager.decrement_stream_source_ref_count.assert_not_called()
        fs._metadata_manager.delete_feature_view_metadata.assert_not_called()

    def test_cleanup_orphaned_drops_snapshot_table_when_task_already_gone(self) -> None:
        """_cleanup_orphaned_feature_view_resources drops an orphaned snapshot table even when
        the companion refresh task was already removed by an earlier partial deletion."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_streaming_metadata.return_value = None

        # SHOW TASKS finds nothing (task already dropped), but SHOW TABLES finds the snapshot
        # table, which must still be cleaned up.
        def _sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            result.collect.return_value = [MagicMock()] if "SHOW TABLES LIKE" in query else []
            return result

        fs._session.sql.side_effect = _sql_side_effect

        with patch.object(fs, "_find_object", return_value=[]):
            result = fs._cleanup_orphaned_feature_view_resources("MY_FV", "V1")

        self.assertTrue(result)
        fs._metadata_manager.decrement_stream_source_ref_count.assert_not_called()
        fs._metadata_manager.delete_feature_view_metadata.assert_called_once_with("MY_FV", "V1")
        issued_sql = [call.args[0] for call in fs._session.sql.call_args_list if call.args]
        self.assertTrue(
            any("DROP TABLE IF EXISTS" in sql for sql in issued_sql),
            f"expected a snapshot DROP TABLE statement, got: {issued_sql}",
        )
        self.assertFalse(
            any("DROP TASK IF EXISTS" in sql for sql in issued_sql),
            "should not drop a refresh task when none exists",
        )

    def test_cleanup_orphaned_drops_task_for_append_only_fv(self) -> None:
        """_cleanup_orphaned_feature_view_resources drops the companion refresh task and
        snapshot table for an orphaned append-only FV (no streaming metadata, no OFT)."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_streaming_metadata.return_value = None
        # SHOW TASKS returns a row -> the FV had a companion refresh task.
        fs._session.sql.return_value.collect.return_value = [MagicMock()]

        # No orphaned online feature table for this FV.
        with patch.object(fs, "_find_object", return_value=[]):
            result = fs._cleanup_orphaned_feature_view_resources("MY_FV", "V1")

        self.assertTrue(result)
        # No streaming cleanup happened, but the task drop and metadata deletion did.
        fs._metadata_manager.decrement_stream_source_ref_count.assert_not_called()
        fs._metadata_manager.delete_feature_view_metadata.assert_called_once_with("MY_FV", "V1")
        issued_sql = [call.args[0] for call in fs._session.sql.call_args_list if call.args]
        self.assertTrue(
            any("DROP TASK IF EXISTS" in sql for sql in issued_sql),
            f"expected a DROP TASK statement, got: {issued_sql}",
        )

    def test_cleanup_orphaned_drops_online_table_for_plain_online_fv(self) -> None:
        """_cleanup_orphaned_feature_view_resources drops the orphaned $ONLINE feature table
        for a plain online-enabled FV (no streaming metadata, no refresh task)."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_streaming_metadata.return_value = None
        # Default collect -> [] so SHOW TASKS finds no refresh task. _find_object returns a row
        # so the FV is detected as having an orphaned online feature table.
        with patch.object(fs, "_find_object", return_value=[MagicMock()]):
            result = fs._cleanup_orphaned_feature_view_resources("MY_FV", "V1")

        self.assertTrue(result)
        fs._metadata_manager.decrement_stream_source_ref_count.assert_not_called()
        fs._metadata_manager.delete_feature_view_metadata.assert_called_once_with("MY_FV", "V1")
        issued_sql = [call.args[0] for call in fs._session.sql.call_args_list if call.args]
        self.assertTrue(
            any("DROP ONLINE FEATURE TABLE IF EXISTS" in sql for sql in issued_sql),
            f"expected a DROP ONLINE FEATURE TABLE statement, got: {issued_sql}",
        )
        self.assertFalse(
            any("DROP TASK IF EXISTS" in sql for sql in issued_sql),
            "should not drop a refresh task when none exists",
        )

    def test_delete_feature_view_cleans_up_and_returns_when_backing_gone(self) -> None:
        """delete_feature_view(name, version) returns gracefully when backing DT is gone
        but streaming metadata still exists."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_realtime_config.return_value = None
        fs._metadata_manager.get_streaming_metadata.return_value = StreamingMetadata(
            stream_source_name="TXN_EVENTS",
            transformation_fn_name="my_fn",
        )

        # Backing object is gone: the lookup returns no rows, so _get_feature_view_impl
        # resolves to None and delete_feature_view falls into the orphan-cleanup path.
        with patch.object(fs, "_get_fv_backend_representations", return_value=[]):
            # Should not raise — orphaned resources are cleaned up and the call returns.
            fs.delete_feature_view("MY_FV", "V1")

        fs._metadata_manager.decrement_stream_source_ref_count.assert_called_once_with("TXN_EVENTS")
        fs._metadata_manager.delete_feature_view_metadata.assert_called_once_with("MY_FV", "V1")

    def test_delete_feature_view_reraises_when_nothing_to_clean_up(self) -> None:
        """delete_feature_view(name, version) re-raises NOT_FOUND when neither the
        backing object nor any streaming metadata exists."""
        from unittest.mock import patch

        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_realtime_config.return_value = None
        fs._metadata_manager.get_streaming_metadata.return_value = None

        # Backing object is gone and there is no streaming metadata to clean up, so the
        # downstream validation lookup raises NOT_FOUND. The telemetry decorator unwraps the
        # SnowflakeMLException to its original ValueError when no live connection is present
        # (as in this unit test), so we assert on that.
        with patch.object(fs, "_get_fv_backend_representations", return_value=[]):
            with self.assertRaises(ValueError) as cm:
                fs.delete_feature_view("MY_FV", "V1")
            self.assertIn("Failed to find FeatureView", str(cm.exception))

        fs._metadata_manager.decrement_stream_source_ref_count.assert_not_called()

    def test_delete_feature_view_propagates_non_not_found_errors(self) -> None:
        """delete_feature_view propagates errors other than NOT_FOUND, without attempting
        orphan cleanup."""
        from unittest.mock import patch

        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )

        fs = _create_feature_store_with_mocks()

        internal_error = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("something went wrong"),
        )
        # An error other than NOT_FOUND while probing the backend must surface, and orphan
        # cleanup must not be attempted. The telemetry decorator unwraps the
        # SnowflakeMLException to its original RuntimeError when no live connection is present.
        with patch.object(fs, "_get_fv_backend_representations", side_effect=internal_error):
            with self.assertRaises(RuntimeError) as cm:
                fs.delete_feature_view("MY_FV", "V1")
            self.assertIn("something went wrong", str(cm.exception))

        fs._metadata_manager.get_streaming_metadata.assert_not_called()


if __name__ == "__main__":
    absltest.main()
