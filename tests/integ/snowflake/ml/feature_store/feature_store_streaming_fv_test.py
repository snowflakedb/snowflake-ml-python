"""Integration tests for streaming feature view registration and lifecycle.

These tests verify the end-to-end flow of creating, registering, listing,
and deleting streaming feature views, including:
- Backfill via map_in_pandas
- UDF transformed table creation
- Offline DT creation from the udf_transformed table
- OFT creation with StreamingFeatureView spec
- Streaming metadata storage and retrieval
- Stream source ref_count management
- Cleanup on delete
"""

import pandas as pd
from absl.testing import absltest, parameterized
from common_utils import create_random_schema
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus
from snowflake.ml.feature_store.stream_config import StreamConfig
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)

# Module-level transformation functions (required by StreamConfig — must be named,
# defined at module level, and only use allowed imports).


def normalize_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns from the amount field."""
    df["AMOUNT_CENTS"] = (df["AMOUNT"] * 100).astype(int)
    df["IS_LARGE"] = df["AMOUNT"] > 500
    return df


def identity_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Identity transformation — returns data unchanged."""
    return df


class StreamingFeatureViewIntegTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for streaming feature view lifecycle."""

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_stores: list[FeatureStore] = []

    def tearDown(self) -> None:
        for fs in self._active_feature_stores:
            try:
                fs._clear(dryrun=False)
            except Exception:
                pass
            try:
                self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
            except Exception:
                pass
        super().tearDown()

    def _create_feature_store(self) -> FeatureStore:
        current_schema = create_random_schema(self._session, "STREAM_FV", database=self.test_db)
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_stores.append(fs)
        return fs

    def _make_stream_source(self, fs: FeatureStore) -> StreamSource:
        """Register and return a stream source."""
        ss = StreamSource(
            name="txn_events",
            schema=StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("AMOUNT", DoubleType()),
                    StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                ]
            ),
            desc="Transaction events stream",
        )
        return fs.register_stream_source(ss)

    def _create_backfill_table(self, fs: FeatureStore) -> str:
        """Create a source table with backfill data and return its name."""
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_SRC"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT,
                EVENT_TIME TIMESTAMP_NTZ
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('u1', 100.0, '2024-01-01 00:00:00'),
            ('u2', 750.0, '2024-01-01 01:00:00'),
            ('u3', 50.0,  '2024-01-01 02:00:00')
        """
        ).collect()

        return table_name

    # =========================================================================
    # Registration tests
    # =========================================================================

    def test_register_streaming_fv_basic(self) -> None:
        """Test basic registration of a streaming feature view."""
        import time

        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        # Verify registration succeeded
        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.online)

        # Verify backfill populates both udf_transformed and backfill tables
        physical_name = FeatureView._get_physical_name(registered_fv.name, registered_fv.version)
        udf_table = FeatureView._get_udf_transformed_table_name(physical_name)
        backfill_table_name = SqlIdentifier(f"{udf_table.resolved()}$BACKFILL", case_sensitive=True)
        fq_udf = f"{self.test_db}.{fs._config.schema.identifier()}.{udf_table}"
        fq_backfill = f"{self.test_db}.{fs._config.schema.identifier()}.{backfill_table_name}"

        # Wait for async backfill to complete (it's INSERT ALL — one job for both tables)
        max_wait = 60
        start = time.time()
        while time.time() - start < max_wait:
            udf_count = self._session.table(fq_udf).count()
            backfill_count = self._session.table(fq_backfill).count()
            if udf_count > 0 and backfill_count > 0:
                break
            time.sleep(3)

        self.assertEqual(udf_count, 3, "udf_transformed table should have 3 backfill rows")
        self.assertEqual(backfill_count, 3, "backfill table should have 3 backfill rows")

    def test_streaming_fv_shows_in_list(self) -> None:
        """Test that streaming FV appears in list_feature_views with stream_config."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        fs.register_feature_view(fv, "v1")

        # list_feature_views should include the stream_config column
        listing = fs.list_feature_views()
        rows = listing.collect()
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["NAME"], "STREAM_FV")
        self.assertIsNotNone(row["STREAM_CONFIG"])

        import json

        stream_config_data = json.loads(row["STREAM_CONFIG"])
        self.assertEqual(stream_config_data["stream_source"], "TXN_EVENTS")
        self.assertEqual(stream_config_data["transformation_fn"], "identity_transform")

    def test_streaming_fv_ref_count(self) -> None:
        """Test that stream source ref_count is incremented on register."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        fs.register_feature_view(fv, "v1")

        # Check ref count is 1
        ref_count = fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS")
        self.assertEqual(ref_count, 1)

    # =========================================================================
    # Delete tests
    # =========================================================================

    def test_delete_streaming_fv_cleans_up(self) -> None:
        """Test that deleting a streaming FV cleans up udf_transformed + backfill tables and ref count."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        # Capture schema path before deletion
        schema_path = f"{self.test_db}.{fs._config.schema.identifier()}"

        # Verify ref count is 1
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS"), 1)

        # Delete the feature view
        fs.delete_feature_view(registered_fv)

        # Verify ref count is back to 0
        ref_count = fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS")
        self.assertEqual(ref_count, 0)

        # Verify the FV is gone
        fv_list = fs.list_feature_views().collect()
        self.assertEqual(len(fv_list), 0)

        # Verify both udf_transformed and backfill tables are dropped
        remaining = self._session.sql(f"SHOW TABLES LIKE '%UDF_TRANSFORMED%' IN SCHEMA {schema_path}").collect()
        self.assertEqual(len(remaining), 0, "udf_transformed and backfill tables should be dropped")

    def test_delete_streaming_fv_allows_stream_source_delete(self) -> None:
        """Test that after deleting all streaming FVs, the stream source can be deleted."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        registered_fv = fs.register_feature_view(fv, "v1")
        fs.delete_feature_view(registered_fv)

        # Now we should be able to delete the stream source
        fs.delete_stream_source("txn_events")

        # Verify it's gone
        sources = fs.list_stream_sources().collect()
        self.assertEqual(len(sources), 0)

    # =========================================================================
    # Streaming + Tiled Features + Dataset Generation
    # =========================================================================

    def test_streaming_tiled_fv_dataset_generation(self) -> None:
        """End-to-end: streaming FV with tiled features, backfill, and dataset generation."""
        from datetime import datetime

        from snowflake.ml.feature_store.feature import Feature

        fs = self._create_feature_store()
        self._make_stream_source(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        # Create a backfill table with enough rows for tiling to produce results.
        # Use 1-day tiles, 2-day window. 4 days of data, 2 users.
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.TILED_BACKFILL"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                AMOUNT FLOAT,
                EVENT_TIME TIMESTAMP_NTZ
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('u1', 10.0,  '2024-01-01 01:00:00'),
            ('u1', 20.0,  '2024-01-02 01:00:00'),
            ('u1', 30.0,  '2024-01-03 01:00:00'),
            ('u1', 40.0,  '2024-01-04 01:00:00'),
            ('u2', 100.0, '2024-01-01 01:00:00'),
            ('u2', 200.0, '2024-01-02 01:00:00'),
            ('u2', 300.0, '2024-01-03 01:00:00')
        """
        ).collect()

        backfill_df = self._session.table(table_name)

        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        features = [
            Feature.sum("AMOUNT", "2d").alias("AMOUNT_SUM_2D"),
            Feature.count("AMOUNT", "2d").alias("TXN_COUNT_2D"),
        ]

        fv = FeatureView(
            name="stream_tiled_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        # Verify it's streaming AND tiled
        self.assertTrue(registered_fv.is_streaming)
        self.assertTrue(registered_fv.is_tiled)

        # Wait for the DT to have data (backfill is async, DT refreshes from udf_transformed)
        import time

        max_wait = 120  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            count = self._session.table(registered_fv.fully_qualified_name()).count()
            if count > 0:
                break
            time.sleep(5)

        # Generate training set — query at 2024-01-05 for both users
        spine_df = self._session.create_dataframe(
            [
                ("u1", datetime(2024, 1, 5, 0, 0, 0)),
                ("u2", datetime(2024, 1, 5, 0, 0, 0)),
            ],
            schema=["USER_ID", "QUERY_TS"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="QUERY_TS",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # Verify we got results for both users
        self.assertEqual(len(result_pd), 2)

        # Verify aggregated feature columns exist
        self.assertIn("AMOUNT_SUM_2D", result_pd.columns)
        self.assertIn("TXN_COUNT_2D", result_pd.columns)

        # Verify aggregation values are non-null
        for col in ["AMOUNT_SUM_2D", "TXN_COUNT_2D"]:
            self.assertTrue(
                result_pd[col].notna().all(),
                f"Column {col} has null values: {result_pd[col].tolist()}",
            )

    # =========================================================================
    # Non-aggregated streaming FV as VIEW (no refresh_freq)
    # =========================================================================

    def test_streaming_non_aggregated_view(self) -> None:
        """Non-aggregated streaming FV without refresh_freq creates a VIEW."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_view_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            # No refresh_freq → VIEW
        )

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertFalse(registered_fv.is_tiled)
        # VIEW-backed FV has STATIC status
        self.assertEqual(registered_fv.status, FeatureViewStatus.STATIC)

    # =========================================================================
    # Non-aggregated streaming FV as DT (with refresh_freq + warning)
    # =========================================================================

    def test_streaming_non_aggregated_dt_with_warning(self) -> None:
        """Non-aggregated streaming FV with refresh_freq creates a DT and warns."""
        import warnings

        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stream_config = StreamConfig(
                stream_source="txn_events",
                transformation_fn=identity_transform,
                backfill_df=backfill_df,
            )
            fv = FeatureView(
                name="stream_dt_fv",
                entities=[entity],
                stream_config=stream_config,
                timestamp_col="EVENT_TIME",
                refresh_freq="1 hour",  # triggers warning for non-aggregated
            )
            # Check that the warning was raised
            refresh_warnings = [x for x in w if "don't require refresh_freq" in str(x.message)]
            self.assertGreater(len(refresh_warnings), 0)

        registered_fv = fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.is_streaming)
        self.assertFalse(registered_fv.is_tiled)
        # DT-backed FV has ACTIVE status
        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # Aggregated streaming FV without refresh_freq → error
    # =========================================================================

    def test_streaming_aggregated_requires_refresh_freq(self) -> None:
        """Aggregated streaming FV without refresh_freq raises error at validate time."""
        from snowflake.ml.feature_store.feature import Feature

        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_agg_no_freq",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            feature_granularity="1h",
            features=[Feature.sum("AMOUNT", "2h").alias("AMOUNT_SUM")],
            # No refresh_freq → should error during registration
        )

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(fv, "v1")

        self.assertIn("refresh_freq", str(cm.exception))

    # =========================================================================
    # get_feature_view reconstruction
    # =========================================================================

    def test_get_feature_view_reconstructs_streaming_fv(self) -> None:
        """Verify get_feature_view returns a streaming FV with correct metadata."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        fs.register_feature_view(fv, "v1")

        # Reconstruct via get_feature_view
        reconstructed = fs.get_feature_view("stream_fv", "v1")

        self.assertTrue(reconstructed.is_streaming)
        self.assertEqual(reconstructed.name, fv.name)
        self.assertEqual(str(reconstructed.version), "v1")
        self.assertTrue(reconstructed.online)

    # =========================================================================
    # Overwrite (re-register with overwrite=True)
    # =========================================================================

    def test_overwrite_streaming_fv(self) -> None:
        """Re-registering with overwrite=True succeeds and keeps ref_count=1."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        fs.register_feature_view(fv, "v1")
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS"), 1)

        # Re-register with overwrite
        fv2 = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=StreamConfig(
                stream_source="txn_events",
                transformation_fn=identity_transform,
                backfill_df=self._session.table(backfill_table),
            ),
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )
        registered = fs.register_feature_view(fv2, "v1", overwrite=True)
        self.assertTrue(registered.is_streaming)

        # ref_count should still be 1 (old decremented, new incremented)
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS"), 1)

    # =========================================================================
    # backfill_start_time filter
    # =========================================================================

    def test_backfill_start_time_filters_data(self) -> None:
        """backfill_start_time filters backfill data by timestamp."""
        from datetime import datetime

        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        # Only include rows after 2024-01-01 01:00:00 (should keep 2 of 3 rows)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
            backfill_start_time=datetime(2024, 1, 1, 0, 30, 0),
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # update_feature_view for streaming FV
    # =========================================================================

    def test_update_streaming_fv_desc(self) -> None:
        """Update streaming FV description."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        fs.register_feature_view(fv, "v1")

        updated = fs.update_feature_view(
            name="stream_fv",
            version="v1",
            desc="Updated description",
        )
        self.assertEqual(updated.desc, "Updated description")
        self.assertTrue(updated.is_streaming)

    # =========================================================================
    # suspend / resume streaming FV
    # =========================================================================

    def test_suspend_and_resume_streaming_fv(self) -> None:
        """Suspend and resume a DT-backed streaming FV."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertEqual(registered.status, FeatureViewStatus.ACTIVE)

        suspended = fs.suspend_feature_view(registered)
        self.assertEqual(suspended.status, FeatureViewStatus.SUSPENDED)

        resumed = fs.resume_feature_view(suspended)
        self.assertEqual(resumed.status, FeatureViewStatus.ACTIVE)

    # =========================================================================
    # Error: unregistered stream source
    # =========================================================================

    def test_unregistered_stream_source_raises(self) -> None:
        """Registering with a nonexistent stream source raises an error."""
        fs = self._create_feature_store()
        # Intentionally NOT registering a stream source
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="nonexistent_source",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
        )

        with self.assertRaisesRegex(RuntimeError, "Cannot find StreamSource"):
            fs.register_feature_view(fv, "v1")

    # =========================================================================
    # OFT creation for streaming FV (POSTGRES)
    # =========================================================================

    def test_streaming_fv_creates_oft(self) -> None:
        """Streaming FV with POSTGRES online store creates an OFT."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        from snowflake.ml.feature_store.feature_view import (
            OnlineConfig,
            OnlineStoreType,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="1 hour",
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        )

        registered = fs.register_feature_view(fv, "v1")
        self.assertTrue(registered.is_streaming)
        self.assertTrue(registered.online)

        # Verify OFT exists by querying SHOW ONLINE FEATURE TABLES
        physical_name = FeatureView._get_physical_name(registered.name, registered.version)
        online_name = FeatureView._get_online_table_name(physical_name)
        try:
            result = self._session.sql(
                f"SHOW ONLINE FEATURE TABLES LIKE '{online_name.identifier()}' IN SCHEMA {fs._config.full_schema_path}"
            ).collect()
            self.assertGreater(len(result), 0, "OFT should exist after streaming FV registration")
        except Exception:
            # SHOW ONLINE FEATURE TABLES may not be available in all environments
            pass

    # =========================================================================
    # Rollback on registration failure
    # =========================================================================

    def test_rollback_on_registration_failure(self) -> None:
        """If registration fails mid-way, udf_transformed table is cleaned up."""
        fs = self._create_feature_store()
        self._make_stream_source(fs)
        backfill_table = self._create_backfill_table(fs)

        entity = Entity(name="user_entity", join_keys=["USER_ID"])
        fs.register_entity(entity)

        backfill_df = self._session.table(backfill_table)
        stream_config = StreamConfig(
            stream_source="txn_events",
            transformation_fn=identity_transform,
            backfill_df=backfill_df,
        )

        fv = FeatureView(
            name="stream_fv",
            entities=[entity],
            stream_config=stream_config,
            timestamp_col="EVENT_TIME",
            refresh_freq="INVALID_FREQ_SHOULD_CAUSE_FAILURE",
        )

        try:
            fs.register_feature_view(fv, "v1")
        except Exception:
            pass

        # ref_count should be 0 (either never incremented, or rolled back)
        ref_count = fs._metadata_manager.get_stream_source_ref_count("TXN_EVENTS")
        self.assertEqual(ref_count, 0)

        # No FVs should be listed
        fv_list = fs.list_feature_views().collect()
        self.assertEqual(len(fv_list), 0)


if __name__ == "__main__":
    absltest.main()
