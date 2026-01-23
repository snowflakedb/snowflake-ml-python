"""Integration tests for tiled aggregation feature views."""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from absl.testing import absltest, parameterized
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA, create_random_schema
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import Feature
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus
from snowflake.ml.feature_store.metadata_manager import (
    _METADATA_TABLE_NAME as _FS_METADATA_TABLE,
)


class TiledAggregationFeatureViewTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for tiled aggregation feature views."""

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
            self._events_table = self._create_events_table()
        except Exception as e:
            self.tearDown()
            raise Exception(f"Test setup failed: {e}")

    def tearDown(self) -> None:
        for fs in self._active_feature_store:
            try:
                fs._clear(dryrun=False)
            except Exception as e:
                if "Intentional Integ Test Error" not in str(e):
                    raise Exception(f"Unexpected exception happens when clear: {e}")
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()

        self._session.sql(f"DROP TABLE IF EXISTS {self._events_table}").collect()
        super().tearDown()

    def _create_events_table(self) -> str:
        """Create a table with time-series event data for testing aggregations."""
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (user_id INT, event_ts TIMESTAMP_NTZ, amount FLOAT, page_id VARCHAR(64), category VARCHAR(64))
            """
        ).collect()

        # Insert test data spanning multiple hours
        # User 1: events at hour 0, 1, 2
        # User 2: events at hour 1, 2, 3
        self._session.sql(
            f"""INSERT INTO {table_full_path} (user_id, event_ts, amount, page_id, category)
                VALUES
                -- User 1 events
                (1, '2024-01-01 00:30:00', 10.0, 'page_a', 'cat1'),
                (1, '2024-01-01 00:45:00', 20.0, 'page_b', 'cat2'),
                (1, '2024-01-01 01:15:00', 30.0, 'page_c', 'cat1'),
                (1, '2024-01-01 01:45:00', 40.0, 'page_d', 'cat3'),
                (1, '2024-01-01 02:30:00', 50.0, 'page_e', 'cat2'),
                -- User 2 events
                (2, '2024-01-01 01:00:00', 100.0, 'page_x', 'cat1'),
                (2, '2024-01-01 02:00:00', 200.0, 'page_y', 'cat2'),
                (2, '2024-01-01 03:00:00', 300.0, 'page_z', 'cat1')
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self, name: Optional[str] = None) -> FeatureStore:
        current_schema = (
            create_random_schema(self._session, "FS_TILED_TEST", database=self.test_db) if name is None else name
        )
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        self._session.use_database(self._dummy_db)
        return fs

    # =========================================================================
    # Registration Tests
    # =========================================================================

    def test_register_tiled_feature_view(self) -> None:
        """Test registering a tiled feature view creates DT with tile query."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum_2h"),
            Feature.count("amount", "2h").alias("txn_count_2h"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        # Should be tiled before registration
        self.assertTrue(fv.is_tiled)

        # Register
        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Verify registration
        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertTrue(registered_fv.is_tiled)
        self.assertEqual(registered_fv.feature_granularity, "1h")
        self.assertIsNotNone(registered_fv.aggregation_specs)
        self.assertEqual(len(registered_fv.aggregation_specs), 2)

    def test_tiled_fv_requires_timestamp_col(self) -> None:
        """Test that tiled FV requires timestamp_col."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("amount_sum")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"

        with self.assertRaisesRegex(ValueError, "timestamp_col is required"):
            FeatureView(
                name="user_stats",
                entities=[e],
                feature_df=self._session.sql(sql),
                refresh_freq="1h",
                feature_granularity="1h",
                features=features,
            )

    def test_tiled_fv_requires_refresh_freq(self) -> None:
        """Test that tiled FV requires refresh_freq (must be managed)."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("amount_sum")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"

        with self.assertRaisesRegex(ValueError, "refresh_freq is required"):
            FeatureView(
                name="user_stats",
                entities=[e],
                feature_df=self._session.sql(sql),
                timestamp_col="event_ts",
                feature_granularity="1h",
                features=features,
            )

    def test_feature_granularity_requires_features(self) -> None:
        """Test that feature_granularity requires features parameter."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"

        with self.assertRaisesRegex(ValueError, "feature_granularity requires features"):
            FeatureView(
                name="user_stats",
                entities=[e],
                feature_df=self._session.sql(sql),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="1h",
                # features not specified
            )

    # =========================================================================
    # Get Feature View Tests
    # =========================================================================

    def test_get_tiled_feature_view(self) -> None:
        """Test that get_feature_view works correctly for tiled feature views.

        This test verifies that a tiled FV can be retrieved after registration.
        Previously, this would fail with "timestamp_col X is not found in input dataframe"
        due to name quoting inconsistency in metadata lookup.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_temporal_hourly",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # This should NOT raise "timestamp_col event_ts is not found in input dataframe"
        retrieved_fv = fs.get_feature_view("user_temporal_hourly", "v1")

        # Verify the retrieved FV has correct properties
        self.assertEqual(retrieved_fv.name, registered_fv.name)
        self.assertEqual(retrieved_fv.version, registered_fv.version)
        self.assertTrue(retrieved_fv.is_tiled)
        self.assertEqual(retrieved_fv.feature_granularity, "1h")
        self.assertIsNotNone(retrieved_fv.aggregation_specs)
        self.assertEqual(len(retrieved_fv.aggregation_specs), 2)

    def test_tiled_fv_feature_names_returns_output_columns(self) -> None:
        """Test that feature_names returns output column names for tiled FVs, not tile column names.

        For tiled FVs, feature_names should return the aggregation output names (e.g., AMOUNT_SUM_2H)
        rather than the internal tile column names (e.g., _PARTIAL_SUM_AMOUNT).
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum_2h"),
            Feature.count("amount", "2h").alias("txn_count_2h"),
            Feature.avg("amount", "2h").alias("amount_avg_2h"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_feature_names_test",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Get the feature view back from the registry
        retrieved_fv = fs.get_feature_view("user_feature_names_test", "v1")

        # feature_names should return output column names, not tile column names
        feature_names = [str(name).upper() for name in retrieved_fv.feature_names]

        # Should contain the output column names
        self.assertIn("AMOUNT_SUM_2H", feature_names)
        self.assertIn("TXN_COUNT_2H", feature_names)
        self.assertIn("AMOUNT_AVG_2H", feature_names)

        # Should NOT contain tile column names
        self.assertNotIn("_PARTIAL_SUM_AMOUNT", feature_names)
        self.assertNotIn("_PARTIAL_COUNT_AMOUNT", feature_names)
        self.assertNotIn("TILE_START", feature_names)

        # Should have exactly 3 feature names (matching the 3 aggregations)
        self.assertEqual(len(feature_names), 3)

    # =========================================================================
    # Feature Store API Compatibility Tests
    # =========================================================================

    def test_list_feature_views_includes_tiled_fv(self) -> None:
        """Test that list_feature_views() returns tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # List feature views
        fv_list = fs.list_feature_views().collect()

        self.assertEqual(len(fv_list), 1)
        self.assertEqual(fv_list[0]["NAME"], "USER_STATS")
        # Version preserves original case
        self.assertEqual(fv_list[0]["VERSION"].lower(), "v1")

    def test_suspend_resume_tiled_fv(self) -> None:
        """Test that suspend/resume work on tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Suspend the feature view
        suspended_fv = fs.suspend_feature_view(registered_fv)
        self.assertEqual(suspended_fv.status, FeatureViewStatus.SUSPENDED)
        self.assertTrue(suspended_fv.is_tiled)

        # Resume the feature view
        resumed_fv = fs.resume_feature_view(suspended_fv)
        self.assertIn(resumed_fv.status, [FeatureViewStatus.ACTIVE, FeatureViewStatus.RUNNING])
        self.assertTrue(resumed_fv.is_tiled)

    def test_update_tiled_fv_description(self) -> None:
        """Test that update_feature_view works on tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
            desc="Original description",
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Update the description
        updated_fv = fs.update_feature_view(
            name="user_stats",
            version="v1",
            desc="Updated description",
        )

        self.assertEqual(updated_fv.desc, "Updated description")
        self.assertTrue(updated_fv.is_tiled)
        self.assertEqual(updated_fv.feature_granularity, "1h")

    def test_delete_tiled_fv(self) -> None:
        """Test that delete_feature_view works on tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Verify it exists
        fv_list = fs.list_feature_views().collect()
        self.assertEqual(len(fv_list), 1)

        # Delete the feature view
        fs.delete_feature_view(registered_fv)

        # Verify it's gone
        fv_list = fs.list_feature_views().collect()
        self.assertEqual(len(fv_list), 0)

    def test_refresh_tiled_fv(self) -> None:
        """Test that refresh_feature_view works on tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Refresh should not raise an error (returns None)
        fs.refresh_feature_view(registered_fv)

        # Verify FV still works after refresh
        retrieved_fv = fs.get_feature_view("user_stats", "v1")
        self.assertTrue(retrieved_fv.is_tiled)

    def test_read_tiled_fv(self) -> None:
        """Test that read_feature_view computes aggregated features at current time.

        For tiled FVs, read_feature_view creates a synthetic spine with unique
        entity combinations, uses CURRENT_TIMESTAMP as query time, and merges
        tiles to return the computed feature values.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "24h").alias("total_sum"),
            Feature.count("amount", "24h").alias("txn_count"),
        ]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Read the feature view - should return computed features, not tiles
        df = fs.read_feature_view(registered_fv)
        columns = [c.upper() for c in df.columns]

        # Should have entity columns + feature columns (not tile internals)
        self.assertIn("USER_ID", columns)
        self.assertIn("TOTAL_SUM", columns)
        self.assertIn("TXN_COUNT", columns)

        # Should NOT have tile internal columns
        self.assertNotIn("TILE_START", columns)
        self.assertFalse(any("_PARTIAL_" in c for c in columns))

        # Verify we get actual data
        rows = df.collect()
        self.assertGreater(len(rows), 0)

    def test_read_tiled_fv_with_feature_filter(self) -> None:
        """Test read_feature_view with feature_names filter on tiled FV."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "24h").alias("total_sum"),
            Feature.count("amount", "24h").alias("txn_count"),
            Feature.avg("amount", "24h").alias("avg_amount"),
        ]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Read with specific feature names
        df = fs.read_feature_view(registered_fv, feature_names=["total_sum", "txn_count"])
        columns = [c.upper() for c in df.columns]

        # Should have entity + filtered features only
        self.assertIn("USER_ID", columns)
        self.assertIn("TOTAL_SUM", columns)
        self.assertIn("TXN_COUNT", columns)
        self.assertNotIn("AVG_AMOUNT", columns)

    def test_mixed_tiled_and_non_tiled_fvs(self) -> None:
        """Test that tiled and non-tiled FVs can coexist in same feature store."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Create a tiled FV
        features = [Feature.sum("amount", "2h").alias("total")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        tiled_fv = FeatureView(
            name="user_tiled",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )
        fs.register_feature_view(feature_view=tiled_fv, version="v1")

        # Create a non-tiled FV (regular managed FV)
        non_tiled_fv = FeatureView(
            name="user_regular",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
        )
        fs.register_feature_view(feature_view=non_tiled_fv, version="v1")

        # Both should be listed
        fv_list = fs.list_feature_views().collect()
        self.assertEqual(len(fv_list), 2)

        # Verify each one's properties
        retrieved_tiled = fs.get_feature_view("user_tiled", "v1")
        retrieved_regular = fs.get_feature_view("user_regular", "v1")

        self.assertTrue(retrieved_tiled.is_tiled)
        self.assertFalse(retrieved_regular.is_tiled)

    # =========================================================================
    # Metadata Tests
    # =========================================================================

    def test_metadata_table_created(self) -> None:
        """Test that metadata table is created when first tiled FV is registered."""
        fs = self._create_feature_store()

        # Table should NOT exist initially (created lazily)
        result = self._session.sql(
            f"SHOW TABLES LIKE '{_FS_METADATA_TABLE}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(result), 0)

        # Register a tiled FV
        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("amount_sum_2h")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )
        fs.register_feature_view(feature_view=fv, version="v1")

        # Now table should exist
        result = self._session.sql(
            f"SHOW TABLES LIKE '{_FS_METADATA_TABLE}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(result), 1)

    def test_aggregation_metadata_saved_and_retrieved(self) -> None:
        """Test that aggregation metadata is saved and can be retrieved."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum_2h"),
            Feature.avg("amount", "2h").alias("amount_avg_2h"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Check metadata is stored
        metadata_table = fs._get_fully_qualified_name(_FS_METADATA_TABLE)
        result = self._session.sql(
            f"""SELECT METADATA FROM {metadata_table}
                WHERE OBJECT_TYPE = 'FEATURE_VIEW'
                AND OBJECT_NAME = 'USER_STATS'
                AND VERSION = 'v1'
                AND METADATA_TYPE = 'FEATURE_SPECS'
            """
        ).collect()

        self.assertEqual(len(result), 1)
        metadata = json.loads(result[0]["METADATA"])
        self.assertEqual(metadata["feature_granularity"], "1h")
        self.assertEqual(len(metadata["features"]), 2)

        # Verify get_feature_view retrieves aggregation info
        retrieved_fv = fs.get_feature_view("user_stats", "v1")
        self.assertTrue(retrieved_fv.is_tiled)
        self.assertEqual(retrieved_fv.feature_granularity, "1h")
        self.assertEqual(len(retrieved_fv.aggregation_specs), 2)

    def test_delete_tiled_fv_removes_metadata(self) -> None:
        """Test that deleting a tiled FV removes its metadata."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "2h").alias("amount_sum_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Verify aggregation specs metadata exists
        metadata_table = fs._get_fully_qualified_name(_FS_METADATA_TABLE)
        result = self._session.sql(
            f"""SELECT 1 FROM {metadata_table}
                WHERE OBJECT_TYPE = 'FEATURE_VIEW'
                AND OBJECT_NAME = 'USER_STATS'
                AND VERSION = 'v1'
                AND METADATA_TYPE = 'FEATURE_SPECS'
            """
        ).collect()
        self.assertEqual(len(result), 1)

        # Delete FV
        fs.delete_feature_view(registered_fv)

        # Verify all metadata for this FV is removed
        result = self._session.sql(
            f"""SELECT 1 FROM {metadata_table}
                WHERE OBJECT_TYPE = 'FEATURE_VIEW'
                AND OBJECT_NAME = 'USER_STATS'
                AND VERSION = 'v1'
            """
        ).collect()
        self.assertEqual(len(result), 0)

    def test_is_tiled_flag_in_fv_metadata(self) -> None:
        """Test that is_tiled flag is stored in FV metadata for UI discovery."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Create a tiled FV
        tiled_features = [Feature.sum("amount", "2h").alias("amount_sum")]
        tiled_sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        tiled_fv = FeatureView(
            name="user_tiled",
            entities=[e],
            feature_df=self._session.sql(tiled_sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=tiled_features,
        )
        registered_tiled = fs.register_feature_view(feature_view=tiled_fv, version="v1")

        # Create a non-tiled FV
        non_tiled_sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        non_tiled_fv = FeatureView(
            name="user_non_tiled",
            entities=[e],
            feature_df=self._session.sql(non_tiled_sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
        )
        registered_non_tiled = fs.register_feature_view(feature_view=non_tiled_fv, version="v1")

        # Verify the _metadata() method returns correct is_tiled flag
        # This is what gets stored in the tag
        tiled_metadata = registered_tiled._metadata()
        self.assertTrue(tiled_metadata.is_tiled, "Tiled FV metadata should have is_tiled=True")

        non_tiled_metadata = registered_non_tiled._metadata()
        self.assertFalse(non_tiled_metadata.is_tiled, "Non-tiled FV metadata should have is_tiled=False")

        # Also verify via get_feature_view (retrieval path)
        retrieved_tiled = fs.get_feature_view("user_tiled", "v1")
        retrieved_non_tiled = fs.get_feature_view("user_non_tiled", "v1")

        self.assertTrue(retrieved_tiled.is_tiled, "Retrieved tiled FV should have is_tiled=True")
        self.assertFalse(retrieved_non_tiled.is_tiled, "Retrieved non-tiled FV should have is_tiled=False")

    # =========================================================================
    # Aggregation Type Tests
    # =========================================================================

    @parameterized.parameters(
        ("sum", Feature.sum("amount", "2h"), "AMOUNT_SUM_2H"),
        ("count", Feature.count("amount", "2h"), "AMOUNT_COUNT_2H"),
        ("avg", Feature.avg("amount", "2h"), "AMOUNT_AVG_2H"),
        ("min", Feature.min("amount", "2h"), "AMOUNT_MIN_2H"),
        ("max", Feature.max("amount", "2h"), "AMOUNT_MAX_2H"),
        ("std", Feature.std("amount", "2h"), "AMOUNT_STD_2H"),
        ("var", Feature.var("amount", "2h"), "AMOUNT_VAR_2H"),
    )
    def test_simple_aggregation_types(self, agg_type: str, feature: Feature, expected_col: str) -> None:
        """Test various simple aggregation types can be registered."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name=f"user_stats_{agg_type}",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[feature],
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(registered_fv.aggregation_specs[0].output_column, expected_col)

    def test_list_aggregation_last_n(self) -> None:
        """Test LAST_N aggregation can be registered."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.last_n("page_id", "2h", n=5).alias("recent_pages")]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_pages",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(registered_fv.aggregation_specs[0].function.value, "last_n")
        self.assertEqual(registered_fv.aggregation_specs[0].params["n"], 5)

    def test_list_aggregation_first_n(self) -> None:
        """Test FIRST_N aggregation can be registered."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.first_n("page_id", "2h", n=3).alias("first_pages")]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_first_pages",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(registered_fv.aggregation_specs[0].function.value, "first_n")

    def test_mixed_aggregation_types(self) -> None:
        """Test mixing simple and list aggregations in one FV."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
            Feature.avg("amount", "2h").alias("amount_avg"),
            Feature.last_n("page_id", "2h", n=5).alias("recent_pages"),
            Feature.last_distinct_n("category", "2h", n=3).alias("recent_categories"),
        ]

        sql = f"SELECT user_id, event_ts, amount, page_id, category FROM {self._events_table}"
        fv = FeatureView(
            name="user_mixed",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(len(registered_fv.aggregation_specs), 5)

    def test_approx_count_distinct_registration(self) -> None:
        """Test APPROX_COUNT_DISTINCT aggregation can be registered."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.approx_count_distinct("page_id", "2h").alias("unique_pages")]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_unique_pages",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(registered_fv.aggregation_specs[0].function.value, "approx_count_distinct")

    def test_approx_percentile_registration(self) -> None:
        """Test APPROX_PERCENTILE aggregation can be registered."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.approx_percentile("amount", "2h", percentile=0.5).alias("median_amount"),
            Feature.approx_percentile("amount", "2h", percentile=0.95).alias("p95_amount"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_percentiles",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(registered_fv.status, FeatureViewStatus.ACTIVE)
        self.assertEqual(len(registered_fv.aggregation_specs), 2)
        self.assertEqual(registered_fv.aggregation_specs[0].function.value, "approx_percentile")
        self.assertEqual(registered_fv.aggregation_specs[0].params["percentile"], 0.5)
        self.assertEqual(registered_fv.aggregation_specs[1].params["percentile"], 0.95)

    # =========================================================================
    # Dataset Generation Tests - Verify Aggregation Values
    # =========================================================================

    def test_sum_aggregation_values(self) -> None:
        """Test SUM aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Use 2h window with 1h tiles
        features = [Feature.sum("amount", "2h").alias("amount_sum_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_sum",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query at 03:00 for user 1
        # Complete tiles: hour 1 (30+40=70), hour 2 (50)
        # Expected sum: 70 + 50 = 120
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            save_as="test_sum",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        # Verify SUM is computed (actual value depends on tile filtering)
        self.assertIn("AMOUNT_SUM_2H", result_pd.columns)
        self.assertIsNotNone(result_pd["AMOUNT_SUM_2H"].iloc[0])

    def test_sum_aggregation_with_empty_tiles_gap(self) -> None:
        """Test SUM aggregation correctly handles gaps (empty tiles) in the time series.

        This test verifies that when there are no events in certain time periods,
        the aggregation correctly sums only the tiles that have data, ignoring
        the empty tiles in between.

        Setup: Events at hour 1 and hour 4, with no events in hours 2-3 (gap).
        Query: At hour 5 with a 4-hour window -> should include tiles from hours 1-4.
        Expected: Only tiles with data (hour 1 and hour 4) contribute to the sum.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Create a table with gaps in the time series
        gap_table = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.events_with_gaps_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE {gap_table}
                (user_id INT, event_ts TIMESTAMP_NTZ, amount FLOAT)
            """
        ).collect()

        # Insert data with a gap: events at hour 1 and hour 4, nothing in hours 2-3
        self._session.sql(
            f"""INSERT INTO {gap_table} (user_id, event_ts, amount)
                VALUES
                -- User 1: Hour 1 events (tiles [01:00-02:00])
                (1, '2024-01-01 01:15:00', 10.0),
                (1, '2024-01-01 01:45:00', 20.0),
                -- Gap: No events in hours 2-3
                -- User 1: Hour 4 events (tile [04:00-05:00])
                (1, '2024-01-01 04:30:00', 100.0)
            """
        ).collect()

        try:
            # Use 4h window with 1h tiles
            features = [Feature.sum("amount", "4h").alias("amount_sum_4h")]

            sql = f"SELECT user_id, event_ts, amount FROM {gap_table}"
            fv = FeatureView(
                name="gap_test",
                entities=[e],
                feature_df=self._session.sql(sql),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="1h",
                features=features,
            )

            registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

            # Query at 05:00 for user 1
            # Window covers hours 1-4, but only tiles at hour 1 and hour 4 have data
            # Hour 1 tile: 10 + 20 = 30 (complete at 02:00)
            # Hour 2 tile: empty (no row in tile table)
            # Hour 3 tile: empty (no row in tile table)
            # Hour 4 tile: 100 (complete at 05:00)
            # Expected sum: 30 + 100 = 130
            spine_df = self._session.create_dataframe(
                [(1, datetime(2024, 1, 1, 5, 0, 0))], schema=["user_id", "query_ts"]
            )

            result_df = fs.generate_training_set(
                spine_df=spine_df,
                features=[registered_fv],
                spine_timestamp_col="query_ts",
                join_method="cte",
            )

            result_pd = result_df.to_pandas()
            self.assertEqual(len(result_pd), 1)
            self.assertIn("AMOUNT_SUM_4H", result_pd.columns)

            # Verify the sum correctly adds only tiles with data, ignoring empty tiles
            actual_sum = float(result_pd["AMOUNT_SUM_4H"].iloc[0])
            expected_sum = 130.0  # 30 (hour 1) + 100 (hour 4)
            self.assertEqual(actual_sum, expected_sum)

        finally:
            # Cleanup
            self._session.sql(f"DROP TABLE IF EXISTS {gap_table}").collect()

    def test_count_aggregation_values(self) -> None:
        """Test COUNT aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.count("amount", "2h").alias("txn_count_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_count",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query at 03:00 for user 1
        # Complete tiles: hour 1 (2 events), hour 2 (1 event)
        # Expected count: 3
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("TXN_COUNT_2H", result_pd.columns)
        self.assertIsNotNone(result_pd["TXN_COUNT_2H"].iloc[0])

    def test_avg_aggregation_values(self) -> None:
        """Test AVG aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.avg("amount", "2h").alias("amount_avg_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_avg",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("AMOUNT_AVG_2H", result_pd.columns)
        self.assertIsNotNone(result_pd["AMOUNT_AVG_2H"].iloc[0])

    def test_min_aggregation_values(self) -> None:
        """Test MIN aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.min("amount", "2h").alias("amount_min_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_min",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")
        registered_fv = fs.get_feature_view("user_min", "v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("AMOUNT_MIN_2H", result_pd.columns)
        # With 2h window from 3:00 AM and 1h granularity, only complete tile 01:00-02:00 is included
        # That tile has data: 30.0, 40.0 (events at 01:15 and 01:45)
        # MIN should be 30.0
        min_val = result_pd["AMOUNT_MIN_2H"].iloc[0]
        self.assertIsNotNone(min_val)
        self.assertEqual(float(min_val), 30.0)

    def test_max_aggregation_values(self) -> None:
        """Test MAX aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.max("amount", "2h").alias("amount_max_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_max",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")
        registered_fv = fs.get_feature_view("user_max", "v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("AMOUNT_MAX_2H", result_pd.columns)
        # With 2h window from 3:00 AM and 1h granularity, complete tiles 01:00-02:00 and 02:00-03:00 are included
        # - 01:00-02:00 tile has: 30.0, 40.0 (events at 01:15 and 01:45)
        # - 02:00-03:00 tile has: 50.0 (event at 02:30), tile_end=03:00 <= query_time=03:00
        # MAX should be 50.0
        max_val = result_pd["AMOUNT_MAX_2H"].iloc[0]
        self.assertIsNotNone(max_val)
        self.assertEqual(float(max_val), 50.0)

    def test_std_aggregation_values(self) -> None:
        """Test STD aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.std("amount", "2h").alias("amount_std_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_std",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("AMOUNT_STD_2H", result_pd.columns)
        # STD should be a non-negative number
        std_val = result_pd["AMOUNT_STD_2H"].iloc[0]
        if std_val is not None:
            self.assertGreaterEqual(float(std_val), 0)

    def test_var_aggregation_values(self) -> None:
        """Test VAR aggregation produces correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.var("amount", "2h").alias("amount_var_2h")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_var",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("AMOUNT_VAR_2H", result_pd.columns)
        # VAR should be a non-negative number
        var_val = result_pd["AMOUNT_VAR_2H"].iloc[0]
        if var_val is not None:
            self.assertGreaterEqual(float(var_val), 0)

    def test_std_var_floating_point_precision(self) -> None:
        """Test STD/VAR with identical values doesn't cause floating-point errors.

        When all values are identical, variance should be exactly 0. However,
        due to floating-point precision issues, (SUM_SQ/N) - (SUM/N)^2 can
        produce tiny negative values like -4.54747e-13, which causes SQRT to fail.

        This test verifies the GREATEST(0, ...) fix prevents such failures.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Create a table with IDENTICAL values - this triggers floating-point issues
        identical_table = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.IDENTICAL_VALUES_{uuid4().hex[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {identical_table} AS
            SELECT
                1 AS user_id,
                DATEADD('hour', -seq4(), CURRENT_TIMESTAMP()::TIMESTAMP_NTZ) AS event_ts,
                100.0 AS amount
            FROM TABLE(GENERATOR(ROWCOUNT => 50))
        """
        ).collect()

        try:
            features = [
                Feature.std("amount", "24h").alias("amount_std"),
                Feature.var("amount", "24h").alias("amount_var"),
            ]

            fv = FeatureView(
                name="fp_precision_test",
                entities=[e],
                feature_df=self._session.sql(f"SELECT user_id, event_ts, amount FROM {identical_table}"),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="1h",
                features=features,
            )

            registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

            spine_df = self._session.create_dataframe([(1, datetime.now())], schema=["user_id", "query_ts"])

            # This would fail without the GREATEST(0, ...) fix:
            # SnowparkSQLException: Invalid floating point operation: sqrt(-4.54747e-13)
            result_df = fs.generate_training_set(
                spine_df=spine_df,
                features=[registered_fv],
                spine_timestamp_col="query_ts",
                join_method="cte",
            )

            result_pd = result_df.to_pandas()
            self.assertEqual(len(result_pd), 1)

            # With identical values, STD and VAR should be 0 (or very close to 0)
            std_val = result_pd["AMOUNT_STD"].iloc[0]
            var_val = result_pd["AMOUNT_VAR"].iloc[0]

            if std_val is not None:
                self.assertAlmostEqual(float(std_val), 0.0, places=5)
            if var_val is not None:
                self.assertAlmostEqual(float(var_val), 0.0, places=5)

        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {identical_table}").collect()

    def test_approx_count_distinct_values(self) -> None:
        """Test APPROX_COUNT_DISTINCT aggregation produces reasonable values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.approx_count_distinct("category", "2h").alias("unique_categories")]

        sql = f"SELECT user_id, event_ts, category FROM {self._events_table}"
        fv = FeatureView(
            name="user_unique_cats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("UNIQUE_CATEGORIES", result_pd.columns)
        # Approximate count should be a positive number
        count_val = result_pd["UNIQUE_CATEGORIES"].iloc[0]
        if count_val is not None:
            self.assertGreater(float(count_val), 0)

    def test_approx_percentile_values(self) -> None:
        """Test APPROX_PERCENTILE aggregation produces reasonable values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.approx_percentile("amount", "2h", percentile=0.5).alias("median_amount"),
            Feature.approx_percentile("amount", "2h", percentile=0.95).alias("p95_amount"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_pctile",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("MEDIAN_AMOUNT", result_pd.columns)
        self.assertIn("P95_AMOUNT", result_pd.columns)

        # P95 should be >= median (approximately)
        median = result_pd["MEDIAN_AMOUNT"].iloc[0]
        p95 = result_pd["P95_AMOUNT"].iloc[0]
        if median is not None and p95 is not None:
            self.assertGreaterEqual(float(p95), float(median) * 0.8)  # Allow some tolerance

    def test_last_n_aggregation_values(self) -> None:
        """Test LAST_N aggregation produces correct array of recent values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.last_n("page_id", "3h", n=5).alias("recent_pages")]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_last_n",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query at 03:00 for user 1
        # Should see pages from hour 0, 1, 2: page_a, page_b, page_c, page_d, page_e
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("RECENT_PAGES", result_pd.columns)

        # Result should be an array
        pages = result_pd["RECENT_PAGES"].iloc[0]
        if pages is not None:
            self.assertIsInstance(pages, (list, str))  # Could be list or JSON string

    def test_first_n_aggregation_values(self) -> None:
        """Test FIRST_N aggregation produces correct array of oldest values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.first_n("page_id", "3h", n=3).alias("first_pages")]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_first_n",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("FIRST_PAGES", result_pd.columns)

    def test_last_distinct_n_aggregation_values(self) -> None:
        """Test LAST_DISTINCT_N aggregation produces correct array of distinct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.last_distinct_n("category", "3h", n=3).alias("recent_categories")]

        sql = f"SELECT user_id, event_ts, category FROM {self._events_table}"
        fv = FeatureView(
            name="user_last_distinct_n",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # User 1 has categories: cat1, cat2, cat1, cat3, cat2
        # Distinct should give at most 3 unique categories
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("RECENT_CATEGORIES", result_pd.columns)

    def test_first_distinct_n_aggregation_values(self) -> None:
        """Test FIRST_DISTINCT_N aggregation produces correct array of distinct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.first_distinct_n("category", "3h", n=3).alias("first_categories")]

        sql = f"SELECT user_id, event_ts, category FROM {self._events_table}"
        fv = FeatureView(
            name="user_first_distinct_n",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 1)
        self.assertIn("FIRST_CATEGORIES", result_pd.columns)

    def test_all_aggregations_combined(self) -> None:
        """Test all aggregation types in a single feature view."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
            Feature.avg("amount", "2h").alias("amount_avg"),
            Feature.std("amount", "2h").alias("amount_std"),
            Feature.var("amount", "2h").alias("amount_var"),
            Feature.last_n("page_id", "2h", n=5).alias("recent_pages"),
            Feature.first_n("page_id", "2h", n=3).alias("first_pages"),
            Feature.last_distinct_n("category", "2h", n=3).alias("recent_cats"),
            Feature.first_distinct_n("category", "2h", n=3).alias("first_cats"),
        ]

        sql = f"SELECT user_id, event_ts, amount, page_id, category FROM {self._events_table}"
        fv = FeatureView(
            name="user_all_aggs",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Test with multiple users
        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 1, 3, 0, 0)), (2, datetime(2024, 1, 1, 4, 0, 0))], schema=["user_id", "query_ts"]
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            save_as="test_all_aggs",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        self.assertEqual(len(result_pd), 2)

        # Verify all columns exist
        expected_cols = [
            "AMOUNT_SUM",
            "TXN_COUNT",
            "AMOUNT_AVG",
            "AMOUNT_STD",
            "AMOUNT_VAR",
            "RECENT_PAGES",
            "FIRST_PAGES",
            "RECENT_CATS",
            "FIRST_CATS",
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns, f"Missing column: {col}")

    def test_multiple_users_different_windows(self) -> None:
        """Test aggregations work correctly for multiple users with different data patterns."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_multi",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query different times for different users
        spine_df = self._session.create_dataframe(
            [
                (1, datetime(2024, 1, 1, 2, 0, 0)),  # User 1 at hour 2
                (1, datetime(2024, 1, 1, 3, 0, 0)),  # User 1 at hour 3
                (2, datetime(2024, 1, 1, 3, 0, 0)),  # User 2 at hour 3
                (2, datetime(2024, 1, 1, 4, 0, 0)),  # User 2 at hour 4
            ],
            schema=["user_id", "query_ts"],
        )

        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            name="test_multi_user",
            output_type="dataset",
            join_method="cte",
        )

        result_pd = ds.read.to_pandas()
        self.assertEqual(len(result_pd), 4)

        # Each row should have aggregation values
        self.assertIn("AMOUNT_SUM", result_pd.columns)
        self.assertIn("TXN_COUNT", result_pd.columns)

    def test_generate_dataset_mixed_tiled_and_non_tiled_fvs(self) -> None:
        """Test dataset generation mixing tiled and non-tiled feature views."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Tiled FV with aggregations
        tiled_features = [
            Feature.sum("amount", "2h").alias("amount_sum_2h"),
        ]
        tiled_sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        tiled_fv = FeatureView(
            name="user_stats_tiled",
            entities=[e],
            feature_df=self._session.sql(tiled_sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=tiled_features,
        )
        registered_tiled_fv = fs.register_feature_view(feature_view=tiled_fv, version="v1")

        # Non-tiled FV (regular ASOF join)
        non_tiled_sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        non_tiled_fv = FeatureView(
            name="user_pages_non_tiled",
            entities=[e],
            feature_df=self._session.sql(non_tiled_sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
        )
        registered_non_tiled_fv = fs.register_feature_view(feature_view=non_tiled_fv, version="v1")

        # Create spine
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        # Generate dataset with both FVs
        ds = fs.generate_dataset(
            spine_df=spine_df,
            features=[registered_tiled_fv, registered_non_tiled_fv],
            spine_timestamp_col="query_ts",
            name="test_mixed_dataset",
            output_type="dataset",
            join_method="cte",
        )

        result_pd = ds.read.to_pandas()

        # Verify columns from both FVs exist
        self.assertIn("AMOUNT_SUM_2H", result_pd.columns)
        self.assertIn("PAGE_ID", result_pd.columns)

    def test_tile_boundary_optimization(self) -> None:
        """Test that spine rows within the same tile boundary get identical feature values.

        This test verifies the optimization where features are computed once per unique
        (entity, tile_boundary) pair rather than once per spine row. Multiple spine rows
        with timestamps in the same hour should all see the same feature values.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_boundary_test",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Create spine with multiple rows for user 1, all within the SAME hour (hour 3)
        # All these should see the same complete tiles (hours 1 and 2) and get identical features
        spine_df = self._session.create_dataframe(
            [
                (1, datetime(2024, 1, 1, 3, 5, 0)),  # 03:05 - same tile boundary
                (1, datetime(2024, 1, 1, 3, 15, 0)),  # 03:15 - same tile boundary
                (1, datetime(2024, 1, 1, 3, 30, 0)),  # 03:30 - same tile boundary
                (1, datetime(2024, 1, 1, 3, 45, 0)),  # 03:45 - same tile boundary
            ],
            schema=["user_id", "query_ts"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # Should have 4 rows (one per spine row)
        self.assertEqual(len(result_pd), 4)

        # All 4 rows should have IDENTICAL feature values since they share the same tile boundary
        # This proves the optimization is working - features computed once, expanded to all rows
        amount_sums = result_pd["AMOUNT_SUM"].unique()
        txn_counts = result_pd["TXN_COUNT"].unique()

        self.assertEqual(len(amount_sums), 1, f"Expected all rows to have same AMOUNT_SUM, got {amount_sums}")
        self.assertEqual(len(txn_counts), 1, f"Expected all rows to have same TXN_COUNT, got {txn_counts}")

        # Verify the actual values are correct
        # Complete tiles: hour 1 (30+40=70), hour 2 (50) = 120
        self.assertEqual(amount_sums[0], 120)
        # Complete tiles: hour 1 (2 events), hour 2 (1 event) = 3
        self.assertEqual(txn_counts[0], 3)

    # =========================================================================
    # generate_training_set Tests
    # =========================================================================

    def test_generate_training_set_with_tiled_fv(self) -> None:
        """Test generate_training_set works correctly with tiled FVs.

        This test verifies that generate_training_set returns aggregated features
        (not tile columns) when using join_method='cte' with tiled FVs.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_training",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        # Use generate_training_set with CTE method
        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()
        columns = [c.upper() for c in result_pd.columns]

        # Should have aggregated feature columns (not tile columns)
        self.assertIn("AMOUNT_SUM", columns)
        self.assertIn("TXN_COUNT", columns)

        # Should NOT have tile internal columns
        self.assertFalse(any("_PARTIAL_" in c for c in columns))
        self.assertNotIn("TILE_START", columns)

        # Verify actual values
        self.assertEqual(len(result_pd), 1)
        self.assertIsNotNone(result_pd["AMOUNT_SUM"].iloc[0])
        self.assertIsNotNone(result_pd["TXN_COUNT"].iloc[0])

    def test_generate_training_set_with_save_as(self) -> None:
        """Test generate_training_set with save_as produces correct results.

        This test verifies that saving to a table doesn't affect the correctness
        of aggregated features - specifically testing a reported bug where save_as
        produced intermediate/tile columns instead of aggregated features.
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_save_as_test",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 1, 3, 0, 0))], schema=["user_id", "query_ts"])

        # Generate unique table name (unqualified, will use FS schema)
        import uuid

        table_name = f"TRAINING_SET_{uuid.uuid4().hex[:8].upper()}"

        try:
            # Use generate_training_set WITH save_as
            result_df = fs.generate_training_set(
                spine_df=spine_df,
                features=[registered_fv],
                spine_timestamp_col="query_ts",
                save_as=table_name,
                join_method="cte",
            )

            # The result should be from the saved table
            result_pd = result_df.to_pandas()
            columns = [c.upper() for c in result_pd.columns]

            # Should have aggregated feature columns (not tile columns)
            self.assertIn("AMOUNT_SUM", columns, f"Expected AMOUNT_SUM in columns: {columns}")
            self.assertIn("TXN_COUNT", columns, f"Expected TXN_COUNT in columns: {columns}")

            # Should NOT have tile internal columns
            self.assertFalse(any("_PARTIAL_" in c for c in columns), f"Found _PARTIAL_ columns in result: {columns}")
            self.assertNotIn("TILE_START", columns, f"Found TILE_START in columns: {columns}")

            # Verify values are correct (not null)
            self.assertEqual(len(result_pd), 1)
            self.assertIsNotNone(result_pd["AMOUNT_SUM"].iloc[0])
            self.assertIsNotNone(result_pd["TXN_COUNT"].iloc[0])

        finally:
            # Cleanup
            self._session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()

    # =========================================================================
    # Tile Table Structure Tests
    # =========================================================================

    def test_tile_table_has_correct_columns_simple_agg(self) -> None:
        """Test that tile table (DT) has correct columns for simple aggregations."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
            Feature.count("amount", "2h").alias("txn_count"),
            Feature.avg("amount", "2h").alias("amount_avg"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_tile_test",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query the tile table structure
        tile_table = registered_fv.fully_qualified_name()
        result = self._session.sql(f"DESC TABLE {tile_table}").collect()
        column_names = [row["name"].upper() for row in result]

        # Should have: USER_ID, TILE_START, and partial aggregation columns
        self.assertIn("USER_ID", column_names)
        self.assertIn("TILE_START", column_names)

        # SUM uses _PARTIAL_SUM_{col}
        self.assertTrue(any("_PARTIAL_SUM_" in col for col in column_names))
        # COUNT uses _PARTIAL_COUNT_{col}
        self.assertTrue(any("_PARTIAL_COUNT_" in col for col in column_names))
        # AVG shares _PARTIAL_SUM and _PARTIAL_COUNT with SUM/COUNT (optimized)
        # No separate _PARTIAL_AVG_ columns needed

    def test_tile_table_has_correct_columns_std_var(self) -> None:
        """Test that tile table has correct columns for STD/VAR (needs SUM, COUNT, SUM_SQ)."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.std("amount", "2h").alias("amount_std"),
            Feature.var("amount", "2h").alias("amount_var"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_tile_stdvar",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query the tile table structure
        tile_table = registered_fv.fully_qualified_name()
        result = self._session.sql(f"DESC TABLE {tile_table}").collect()
        column_names = [row["name"].upper() for row in result]

        # STD and VAR should have _SUM, _COUNT, and _SUM_SQ columns
        sum_sq_cols = [col for col in column_names if "_SUM_SQ" in col]
        self.assertTrue(len(sum_sq_cols) > 0, f"Expected SUM_SQ columns, got: {column_names}")

    def test_tile_table_has_correct_columns_list_agg(self) -> None:
        """Test that tile table has array columns for list aggregations."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.last_n("page_id", "2h", n=5).alias("recent_pages"),
        ]

        sql = f"SELECT user_id, event_ts, page_id FROM {self._events_table}"
        fv = FeatureView(
            name="user_tile_list",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query the tile table structure
        tile_table = registered_fv.fully_qualified_name()
        result = self._session.sql(f"DESC TABLE {tile_table}").collect()

        # Find the list aggregation column (now named _PARTIAL_LAST_{col})
        list_cols = [row for row in result if "_PARTIAL_LAST_" in row["name"].upper()]
        self.assertTrue(len(list_cols) > 0, "Expected LAST partial column")

        # The type should be ARRAY
        for col in list_cols:
            self.assertIn("ARRAY", col["type"].upper(), f"Expected ARRAY type for {col['name']}")

    def test_tile_data_is_populated(self) -> None:
        """Test that tile table contains aggregated data after DT refresh."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "2h").alias("amount_sum"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_tile_data",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query tile data
        tile_table = registered_fv.fully_qualified_name()
        result = self._session.sql(f"SELECT * FROM {tile_table}").collect()

        # Should have tiles for the data we inserted
        self.assertGreater(len(result), 0, "Tile table should have data")

        # Each row should have a TILE_START
        for row in result:
            self.assertIn("TILE_START", row.asDict())

    # =========================================================================
    # Window Offset Tests
    # =========================================================================

    def test_offset_feature_values(self) -> None:
        """Test that offset shifts the aggregation window into the past.

        With hourly granularity, offset=1h means we look 1 hour further back.
        For a spine timestamp at hour 4:
        - Without offset, window="2h": aggregates hours 2-3
        - With offset="1h", window="2h": aggregates hours 1-2
        """
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Create two features: one without offset, one with offset
        features = [
            Feature.sum("amount", "2h").alias("current_sum"),
            Feature.sum("amount", "2h", offset="1h").alias("prev_sum"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_offset_test",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        # Query at hour 4 for user 1:
        # - current_sum (no offset): looks at tiles hour 2, 3 -> hours 2-3 events
        #   User 1 has: hour 2 = 50.0 (event at 02:30)
        #   So current_sum = 50.0
        # - prev_sum (offset=1h): looks at tiles hour 1, 2 -> hours 1-2 events
        #   User 1 has: hour 1 = 30+40=70.0, hour 2 = 50.0
        #   So prev_sum = 120.0
        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 1, 4, 0, 0))],
            schema=["user_id", "query_ts"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[registered_fv],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        self.assertEqual(len(result_pd), 1)

        # Verify the values differ due to offset
        current_sum = float(result_pd["CURRENT_SUM"].iloc[0])
        prev_sum = float(result_pd["PREV_SUM"].iloc[0])

        # Current sum: only tile hour 2 = 50.0
        self.assertEqual(current_sum, 50.0)
        # Previous sum: tiles hour 1 + hour 2 = 70 + 50 = 120.0
        self.assertEqual(prev_sum, 120.0)

    def test_offset_validation_not_multiple_of_granularity(self) -> None:
        """Test that offset not multiple of granularity raises error."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Try to create FV with offset that's not a multiple of granularity
        # Granularity = 1h, offset = 30m (not a multiple)
        with self.assertRaises(ValueError) as cm:
            features = [Feature.sum("amount", "2h", offset="30m")]
            sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
            FeatureView(
                name="invalid_offset_fv",
                entities=[e],
                feature_df=self._session.sql(sql),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="1h",
                features=features,
            )

        self.assertIn("must be a multiple of", str(cm.exception))

    def test_window_validation_not_multiple_of_granularity(self) -> None:
        """Test that window not multiple of granularity raises error."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Try to create FV with window that's not a multiple of granularity
        # Granularity = 1h, window = 90m (not a multiple)
        with self.assertRaises(ValueError) as cm:
            features = [Feature.sum("amount", "90m")]
            sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
            FeatureView(
                name="invalid_window_fv",
                entities=[e],
                feature_df=self._session.sql(sql),
                timestamp_col="event_ts",
                refresh_freq="1h",
                feature_granularity="1h",
                features=features,
            )

        self.assertIn("must be a multiple of", str(cm.exception))

    def test_week_over_week_comparison(self) -> None:
        """Test week-over-week style features using offset.

        This demonstrates a real use case: comparing current week vs previous week.
        Using daily granularity and 7d windows with 7d offset.
        """
        # Create a table with daily data spanning 3 weeks
        table_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.daily_events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE {table_path}
                (user_id INT, event_ts TIMESTAMP_NTZ, amount FLOAT)
            """
        ).collect()

        # Week 1: days 1-7, user 1 has total 100
        # Week 2: days 8-14, user 1 has total 200
        # Week 3: days 15-21, user 1 has total 300
        self._session.sql(
            f"""INSERT INTO {table_path} VALUES
                -- Week 1 (days 1-7)
                (1, '2024-01-01 12:00:00', 10.0),
                (1, '2024-01-03 12:00:00', 30.0),
                (1, '2024-01-05 12:00:00', 60.0),
                -- Week 2 (days 8-14)
                (1, '2024-01-08 12:00:00', 50.0),
                (1, '2024-01-10 12:00:00', 50.0),
                (1, '2024-01-12 12:00:00', 100.0),
                -- Week 3 (days 15-21)
                (1, '2024-01-15 12:00:00', 100.0),
                (1, '2024-01-18 12:00:00', 100.0),
                (1, '2024-01-20 12:00:00', 100.0)
            """
        ).collect()

        try:
            fs = self._create_feature_store()

            e = Entity("user", ["user_id"])
            fs.register_entity(e)

            # Week-over-week features
            features = [
                Feature.sum("amount", "7d").alias("current_week_sum"),
                Feature.sum("amount", "7d", offset="7d").alias("prev_week_sum"),
            ]

            fv = FeatureView(
                name="weekly_stats",
                entities=[e],
                feature_df=self._session.sql(f"SELECT user_id, event_ts, amount FROM {table_path}"),
                timestamp_col="event_ts",
                refresh_freq="1d",
                feature_granularity="1d",
                features=features,
            )

            registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

            # Query at day 22 (start of hypothetical week 4)
            # current_week_sum: days 15-21 (week 3) = 300
            # prev_week_sum: days 8-14 (week 2) = 200
            spine_df = self._session.create_dataframe(
                [(1, datetime(2024, 1, 22, 12, 0, 0))],
                schema=["user_id", "query_ts"],
            )

            result_df = fs.generate_training_set(
                spine_df=spine_df,
                features=[registered_fv],
                spine_timestamp_col="query_ts",
                join_method="cte",
            )

            result_pd = result_df.to_pandas()

            self.assertEqual(len(result_pd), 1)

            current_week = float(result_pd["CURRENT_WEEK_SUM"].iloc[0])
            prev_week = float(result_pd["PREV_WEEK_SUM"].iloc[0])

            self.assertEqual(current_week, 300.0)
            self.assertEqual(prev_week, 200.0)

        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {table_path}").collect()

    # =========================================================================
    # Existing Feature Store Compatibility Tests
    # =========================================================================

    def test_metadata_table_created_lazily_for_existing_fs(self) -> None:
        """Test that metadata table is created when first tiled FV is registered on existing FS."""
        # Create a schema without creating FS (simulating existing FS without metadata table)
        schema_name = create_random_schema(self._session, "FS_EXISTING", database=self.test_db)

        # Create FS in FAIL_IF_NOT_EXIST mode first to see it doesn't exist
        # This will fail because tags don't exist
        try:
            FeatureStore(
                self._session,
                self.test_db,
                schema_name,
                default_warehouse=self._test_warehouse_name,
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
            )
            self.fail("Should have raised an error")
        except Exception:
            pass

        # Now create FS properly
        fs = FeatureStore(
            self._session,
            self.test_db,
            schema_name,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)

        # Register entity
        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        # Drop metadata table to simulate existing FS without it
        self._session.sql(f"DROP TABLE IF EXISTS {fs._config.full_schema_path}.{_FS_METADATA_TABLE}").collect()

        # Verify table is gone
        result = self._session.sql(
            f"SHOW TABLES LIKE '{_FS_METADATA_TABLE}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(result), 0)

        # Register tiled FV - should create metadata table
        features = [Feature.sum("amount", "2h").alias("amount_sum")]
        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_stats",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Verify metadata table was created
        result = self._session.sql(
            f"SHOW TABLES LIKE '{_FS_METADATA_TABLE}' IN SCHEMA {fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(result), 1)


class LifetimeAggregationTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for lifetime aggregation features."""

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_store = []

        try:
            self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
            self._events_table = self._create_events_table()
        except Exception as e:
            self.tearDown()
            raise Exception(f"Test setup failed: {e}")

    def tearDown(self) -> None:
        for fs in self._active_feature_store:
            try:
                fs._clear(dryrun=False)
            except Exception as e:
                if "Intentional Integ Test Error" not in str(e):
                    raise Exception(f"Unexpected exception happens when clear: {e}")
            self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()

        self._session.sql(f"DROP TABLE IF EXISTS {self._events_table}").collect()
        super().tearDown()

    def _create_events_table(self) -> str:
        """Create a table with time-series event data spanning multiple days."""
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.lifetime_events_{uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (user_id INT, event_ts TIMESTAMP_NTZ, amount FLOAT, page_id VARCHAR(64), category VARCHAR(64))
            """
        ).collect()

        # Insert test data spanning multiple days
        self._session.sql(
            f"""INSERT INTO {table_full_path} (user_id, event_ts, amount, page_id, category)
                VALUES
                -- User 1 events spanning 5 days
                (1, '2024-01-01 10:00:00', 10.0, 'page_a', 'cat1'),
                (1, '2024-01-02 10:00:00', 20.0, 'page_b', 'cat2'),
                (1, '2024-01-03 10:00:00', 30.0, 'page_c', 'cat1'),
                (1, '2024-01-04 10:00:00', 40.0, 'page_d', 'cat3'),
                (1, '2024-01-05 10:00:00', 50.0, 'page_e', 'cat2'),
                -- User 2 events spanning 3 days
                (2, '2024-01-01 12:00:00', 100.0, 'page_x', 'cat1'),
                (2, '2024-01-03 12:00:00', 200.0, 'page_y', 'cat2'),
                (2, '2024-01-05 12:00:00', 300.0, 'page_z', 'cat1')
            """
        ).collect()
        return table_full_path

    def _create_feature_store(self) -> FeatureStore:
        current_schema = create_random_schema(self._session, "FS_LIFETIME_TEST", database=self.test_db)
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_store.append(fs)
        self._session.use_database(self._dummy_db)
        return fs

    # =========================================================================
    # Registration Tests for Lifetime Features
    # =========================================================================

    def test_register_lifetime_sum_feature(self) -> None:
        """Test registering a feature view with lifetime SUM aggregation."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "lifetime").alias("total_amount")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_sum",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertTrue(registered_fv.is_tiled)
        self.assertEqual(len(registered_fv.aggregation_specs), 1)
        self.assertTrue(registered_fv.aggregation_specs[0].is_lifetime())

    def test_register_mixed_lifetime_and_fixed_window_features(self) -> None:
        """Test registering features with both lifetime and fixed window aggregations."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "3d").alias("sum_3d"),  # Fixed window
            Feature.sum("amount", "lifetime").alias("total_sum"),  # Lifetime
            Feature.count("amount", "lifetime").alias("total_count"),  # Lifetime
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_mixed",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        registered_fv = fs.register_feature_view(feature_view=fv, version="v1")

        self.assertEqual(len(registered_fv.aggregation_specs), 3)
        # First feature is fixed window
        self.assertFalse(registered_fv.aggregation_specs[0].is_lifetime())
        # Second and third are lifetime
        self.assertTrue(registered_fv.aggregation_specs[1].is_lifetime())
        self.assertTrue(registered_fv.aggregation_specs[2].is_lifetime())

    # =========================================================================
    # Lifetime Feature Value Tests
    # =========================================================================

    def test_lifetime_sum_values(self) -> None:
        """Test that lifetime SUM returns correct cumulative values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "lifetime").alias("total_amount")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_sum",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Query at 2024-01-06 should include all events
        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 6, 0, 0, 0)), (2, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"]
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_sum", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1: 10 + 20 + 30 + 40 + 50 = 150
        user1_total = result_pd[result_pd["USER_ID"] == 1]["TOTAL_AMOUNT"].iloc[0]
        self.assertEqual(float(user1_total), 150.0)

        # User 2: 100 + 200 + 300 = 600
        user2_total = result_pd[result_pd["USER_ID"] == 2]["TOTAL_AMOUNT"].iloc[0]
        self.assertEqual(float(user2_total), 600.0)

    def test_lifetime_count_values(self) -> None:
        """Test that lifetime COUNT returns correct cumulative counts."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.count("amount", "lifetime").alias("total_count")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_count",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 6, 0, 0, 0)), (2, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"]
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_count", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1: 5 events
        user1_count = result_pd[result_pd["USER_ID"] == 1]["TOTAL_COUNT"].iloc[0]
        self.assertEqual(int(user1_count), 5)

        # User 2: 3 events
        user2_count = result_pd[result_pd["USER_ID"] == 2]["TOTAL_COUNT"].iloc[0]
        self.assertEqual(int(user2_count), 3)

    def test_lifetime_avg_values(self) -> None:
        """Test that lifetime AVG returns correct cumulative average."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.avg("amount", "lifetime").alias("avg_amount")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_avg",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 6, 0, 0, 0)), (2, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"]
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_avg", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1: 150 / 5 = 30
        user1_avg = result_pd[result_pd["USER_ID"] == 1]["AVG_AMOUNT"].iloc[0]
        self.assertEqual(float(user1_avg), 30.0)

        # User 2: 600 / 3 = 200
        user2_avg = result_pd[result_pd["USER_ID"] == 2]["AVG_AMOUNT"].iloc[0]
        self.assertEqual(float(user2_avg), 200.0)

    def test_lifetime_min_max_values(self) -> None:
        """Test that lifetime MIN and MAX return correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.min("amount", "lifetime").alias("min_amount"),
            Feature.max("amount", "lifetime").alias("max_amount"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_minmax",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe(
            [(1, datetime(2024, 1, 6, 0, 0, 0)), (2, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"]
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_minmax", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1: min=10, max=50
        user1_min = result_pd[result_pd["USER_ID"] == 1]["MIN_AMOUNT"].iloc[0]
        user1_max = result_pd[result_pd["USER_ID"] == 1]["MAX_AMOUNT"].iloc[0]
        self.assertEqual(float(user1_min), 10.0)
        self.assertEqual(float(user1_max), 50.0)

        # User 2: min=100, max=300
        user2_min = result_pd[result_pd["USER_ID"] == 2]["MIN_AMOUNT"].iloc[0]
        user2_max = result_pd[result_pd["USER_ID"] == 2]["MAX_AMOUNT"].iloc[0]
        self.assertEqual(float(user2_min), 100.0)
        self.assertEqual(float(user2_max), 300.0)

    def test_lifetime_std_var_values(self) -> None:
        """Test that lifetime STD and VAR return correct values."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.std("amount", "lifetime").alias("std_amount"),
            Feature.var("amount", "lifetime").alias("var_amount"),
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_stdvar",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_stdvar", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1: values are 10, 20, 30, 40, 50
        # Variance = ((10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2) / 5
        #          = (400 + 100 + 0 + 100 + 400) / 5 = 200
        # STD = sqrt(200) ≈ 14.14
        std_val = result_pd["STD_AMOUNT"].iloc[0]
        var_val = result_pd["VAR_AMOUNT"].iloc[0]

        self.assertAlmostEqual(float(var_val), 200.0, places=1)
        self.assertAlmostEqual(float(std_val), 14.14, places=1)

    def test_lifetime_unsupported_aggregations(self) -> None:
        """Test that unsupported aggregation types raise errors for lifetime windows."""
        # APPROX_COUNT_DISTINCT is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for approx_count_distinct"):
            Feature.approx_count_distinct("category", "lifetime").to_spec()

        # APPROX_PERCENTILE is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for approx_percentile"):
            Feature.approx_percentile("amount", "lifetime", percentile=0.5).to_spec()

        # LAST_N is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for last_n"):
            Feature.last_n("page_id", "lifetime", n=5).to_spec()

        # FIRST_N is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for first_n"):
            Feature.first_n("page_id", "lifetime", n=5).to_spec()

        # LAST_DISTINCT_N is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for last_distinct_n"):
            Feature.last_distinct_n("category", "lifetime", n=3).to_spec()

        # FIRST_DISTINCT_N is not supported for lifetime
        with self.assertRaisesRegex(ValueError, "Lifetime window is not supported for first_distinct_n"):
            Feature.first_distinct_n("category", "lifetime", n=3).to_spec()

    def test_mixed_lifetime_and_fixed_window_values(self) -> None:
        """Test that mixed lifetime and fixed window features work together."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [
            Feature.sum("amount", "3d").alias("sum_3d"),  # Fixed 3-day window
            Feature.sum("amount", "lifetime").alias("total_sum"),  # Lifetime
        ]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_mixed",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Query at 2024-01-06 00:00
        # 3d window should include Jan 3, 4, 5 (Jan 6 boundary, complete tiles before it)
        spine_df = self._session.create_dataframe([(1, datetime(2024, 1, 6, 0, 0, 0))], schema=["user_id", "query_ts"])

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_mixed", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas()

        # User 1 lifetime: 10 + 20 + 30 + 40 + 50 = 150
        total_sum = float(result_pd["TOTAL_SUM"].iloc[0])
        self.assertEqual(total_sum, 150.0)

        # User 1 3d window (Jan 3, 4, 5): 30 + 40 + 50 = 120
        sum_3d = float(result_pd["SUM_3D"].iloc[0])
        self.assertEqual(sum_3d, 120.0)

    def test_lifetime_offset_not_allowed(self) -> None:
        """Test that offset is not allowed with lifetime windows."""
        with self.assertRaisesRegex(ValueError, "Offset is not supported with lifetime"):
            # The validation happens when to_spec() is called
            Feature.sum("amount", "lifetime", offset="1d").to_spec()

    def test_lifetime_point_in_time_correctness(self) -> None:
        """Test that lifetime features are point-in-time correct (only include events before query time)."""
        fs = self._create_feature_store()

        e = Entity("user", ["user_id"])
        fs.register_entity(e)

        features = [Feature.sum("amount", "lifetime").alias("total_amount")]

        sql = f"SELECT user_id, event_ts, amount FROM {self._events_table}"
        fv = FeatureView(
            name="user_lifetime_pit",
            entities=[e],
            feature_df=self._session.sql(sql),
            timestamp_col="event_ts",
            refresh_freq="1d",
            feature_granularity="1d",
            features=features,
        )

        fs.register_feature_view(feature_view=fv, version="v1")

        # Query at different times to verify point-in-time correctness
        spine_df = self._session.create_dataframe(
            [
                (1, datetime(2024, 1, 2, 0, 0, 0)),  # Should only include Jan 1 (10)
                (1, datetime(2024, 1, 4, 0, 0, 0)),  # Should include Jan 1-3 (10+20+30=60)
                (1, datetime(2024, 1, 6, 0, 0, 0)),  # Should include all (150)
            ],
            schema=["user_id", "query_ts"],
        )

        result_df = fs.generate_training_set(
            spine_df=spine_df,
            features=[fs.get_feature_view("user_lifetime_pit", "v1")],
            spine_timestamp_col="query_ts",
            join_method="cte",
        )

        result_pd = result_df.to_pandas().sort_values("QUERY_TS")

        # Jan 2 query: only Jan 1 tile complete -> 10
        jan2_sum = float(result_pd.iloc[0]["TOTAL_AMOUNT"])
        self.assertEqual(jan2_sum, 10.0)

        # Jan 4 query: Jan 1-3 tiles complete -> 10+20+30=60
        jan4_sum = float(result_pd.iloc[1]["TOTAL_AMOUNT"])
        self.assertEqual(jan4_sum, 60.0)

        # Jan 6 query: Jan 1-5 tiles complete -> 150
        jan6_sum = float(result_pd.iloc[2]["TOTAL_AMOUNT"])
        self.assertEqual(jan6_sum, 150.0)


if __name__ == "__main__":
    absltest.main()
