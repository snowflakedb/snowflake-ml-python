"""Integration tests for spec-based Online Feature Table creation (Postgres store type).

Tests the ``WITH SPECIFICATION $$<json>$$`` OFT creation path for batch feature views
using ``OnlineStoreType.POSTGRES``.
"""

import json
import uuid

from absl.testing import absltest, parameterized
from common_utils import FS_INTEG_TEST_DATASET_SCHEMA
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import Feature, entity, feature_store, feature_view


class SpecBasedOFTTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for spec-based OFT creation (OnlineStoreType.POSTGRES)."""

    def setUp(self) -> None:
        super().setUp()
        self.warehouse = self._test_warehouse_name

        self.fs = feature_store.FeatureStore(
            session=self._session,
            database=self.test_db,
            name=self.test_schema,
            default_warehouse=self.warehouse,
            creation_mode=feature_store.CreationMode.CREATE_IF_NOT_EXIST,
        )

        # Create test entities
        self.user_entity = entity.Entity(name="user_entity", join_keys=["user_id"], desc="User entity")
        self.fs.register_entity(self.user_entity)

        self.product_entity = entity.Entity(name="product_entity", join_keys=["product_id"], desc="Product entity")
        self.fs.register_entity(self.product_entity)

        # Create test table with sample data using natural SQL types.
        # INT → LongType, FLOAT → DoubleType, VARCHAR → StringType,
        # TIMESTAMP_NTZ → TimestampType — all in _SUPPORTED_TYPES.
        self.test_table_name = f"TEST_SPEC_OFT_DATA_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{self.test_table_name} (
                user_id INT,
                product_id INT,
                purchase_amount FLOAT,
                purchase_time TIMESTAMP_NTZ
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {self.fs._config.full_schema_path}.{self.test_table_name} VALUES
            (1, 100, 10.5, '2023-01-01 00:00:00'::TIMESTAMP_NTZ),
            (2, 200, 20.0, '2023-01-02 00:00:00'::TIMESTAMP_NTZ),
            (3, 300, 30.5, '2023-01-03 00:00:00'::TIMESTAMP_NTZ)
        """
        ).collect()

        self.sample_data = self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}")

        # Create events table for tiled FV tests
        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}").collect()
        self._events_table = self._create_events_table()

    def tearDown(self) -> None:
        try:
            self.fs._clear(dryrun=False)
        except Exception:
            pass
        self._session.sql(f"DROP TABLE IF EXISTS {self._events_table}").collect()
        super().tearDown()

    def _create_events_table(self) -> str:
        """Create a table with time-series event data for tiled aggregation tests."""
        table_full_path = f"{self.test_db}.{FS_INTEG_TEST_DATASET_SCHEMA}.events_{uuid.uuid4().hex.upper()}"
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {table_full_path}
                (user_id INT, event_ts TIMESTAMP_NTZ, amount FLOAT)
            """
        ).collect()

        self._session.sql(
            f"""INSERT INTO {table_full_path} (user_id, event_ts, amount) VALUES
                (1, '2023-01-01 00:00:00'::TIMESTAMP_NTZ, 10.0),
                (1, '2023-01-01 01:00:00'::TIMESTAMP_NTZ, 20.0),
                (2, '2023-01-01 01:00:00'::TIMESTAMP_NTZ, 30.0),
                (2, '2023-01-01 02:00:00'::TIMESTAMP_NTZ, 40.0)
            """
        ).collect()

        return table_full_path

    def _get_events_df(self):
        return self._session.table(self._events_table)

    # =========================================================================
    # Non-tiled Batch FV with Postgres store type
    # =========================================================================

    def test_postgres_oft_basic_batch_fv(self) -> None:
        """Test spec-based OFT creation for a simple batch FV with Postgres store type."""
        fv_name = "spec_oft_basic"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="1m",
            desc="Basic batch FV with Postgres OFT",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="15s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Verify registration succeeded
        self.assertTrue(registered_fv.online)
        self.assertIsNotNone(registered_fv.online_config)
        self.assertEqual(registered_fv.online_config.target_lag, "15 seconds")

        # Verify online table was created
        online_name = registered_fv.fully_qualified_online_table_name()
        self.assertIsNotNone(online_name)
        self.assertIn("$ONLINE", online_name)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    def test_postgres_oft_batch_fv_without_timestamp(self) -> None:
        """Test spec-based OFT creation for a batch FV without a timestamp column."""
        fv_name = "spec_oft_no_ts"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            desc="Batch FV without timestamp with Postgres OFT",
            online_config=feature_view.OnlineConfig(
                enable=True,
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.online)
        self.assertIsNone(registered_fv.timestamp_col)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    def test_postgres_oft_batch_fv_multi_entity(self) -> None:
        """Test spec-based OFT creation for a batch FV with multiple entities."""
        fv_name = "spec_oft_multi_entity"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity, self.product_entity],
            feature_df=self.sample_data,
            timestamp_col="purchase_time",
            refresh_freq="5m",
            desc="Multi-entity batch FV with Postgres OFT",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="20s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.online)
        self.assertEqual(len(registered_fv.entities), 2)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    def test_postgres_oft_static_batch_fv(self) -> None:
        """Test spec-based OFT creation for a static (no refresh_freq) batch FV."""
        fv_name = "spec_oft_static"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            desc="Static batch FV with Postgres OFT",
            online_config=feature_view.OnlineConfig(
                enable=True,
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.online)
        self.assertIsNone(registered_fv.refresh_freq)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    # =========================================================================
    # Tiled Batch FV with Postgres store type
    # =========================================================================

    def test_postgres_oft_tiled_batch_fv(self) -> None:
        """Test spec-based OFT creation for a tiled (aggregated) batch FV."""
        fv_name = "spec_oft_tiled"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[Feature.sum("AMOUNT", "24h").alias("total_amount_24h")],
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="30s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.online)
        self.assertTrue(registered_fv.is_tiled)
        self.assertEqual(registered_fv.online_config.target_lag, "30 seconds")

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    def test_postgres_oft_tiled_multiple_features(self) -> None:
        """Test spec-based OFT creation for a tiled FV with multiple aggregation features."""
        fv_name = "spec_oft_tiled_multi"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self._get_events_df(),
            timestamp_col="event_ts",
            refresh_freq="1h",
            feature_granularity="1h",
            features=[
                Feature.sum("AMOUNT", "24h").alias("total_amount"),
                Feature.count("USER_ID", "24h").alias("event_count"),
                Feature.avg("AMOUNT", "24h").alias("avg_amount"),
            ],
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="30s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        self.assertTrue(registered_fv.online)
        self.assertTrue(registered_fv.is_tiled)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

    # =========================================================================
    # Lifecycle operations with Postgres store type
    # =========================================================================

    def test_postgres_oft_get_feature_view_preserves_config(self) -> None:
        """Test that get_feature_view preserves the Postgres online config."""
        fv_name = "spec_oft_get_fv"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="5m",
            desc="Test get FV with Postgres config",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="20s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        self.fs.register_feature_view(fv, "v1")

        # Retrieve and verify config is preserved.
        # Note: store_type is a client-side creation hint and is not persisted
        # in the backend metadata, so we only verify the operational fields.
        retrieved_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertTrue(retrieved_fv.online)
        self.assertIsNotNone(retrieved_fv.online_config)
        self.assertEqual(retrieved_fv.online_config.target_lag, "20 seconds")

    def test_postgres_oft_delete_feature_view(self) -> None:
        """Test that deleting a FV with Postgres OFT also deletes the OFT."""
        fv_name = "spec_oft_delete"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            online_config=feature_view.OnlineConfig(
                enable=True,
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online)

        # Delete
        self.fs.delete_feature_view(registered_fv)

        # Verify both FV and OFT are gone
        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.name == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 0)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 0)

    def test_postgres_oft_overwrite(self) -> None:
        """Test overwriting a FV with Postgres OFT recreates the OFT."""
        fv_name = "spec_oft_overwrite"

        # Register initial FV with Postgres OFT
        fv1 = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            desc="original",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="10s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )
        self.fs.register_feature_view(fv1, "v1")

        # Overwrite with different config
        fv2 = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="2h",
            desc="overwritten",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="30s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )
        self.fs.register_feature_view(fv2, "v1", overwrite=True)

        retrieved = self.fs.get_feature_view(fv_name, "v1")
        self.assertTrue(retrieved.online)
        self.assertEqual(retrieved.online_config.target_lag, "30 seconds")
        self.assertEqual(retrieved.desc, "overwritten")

    def test_postgres_oft_update_enable_disable(self) -> None:
        """Test enabling and disabling Postgres OFT via update_feature_view."""
        fv_name = "spec_oft_toggle"

        # Start with offline-only
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            online_config=feature_view.OnlineConfig(enable=False),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertFalse(registered_fv.online)

        # Enable online with Postgres store type
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="15s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )
        self.assertTrue(updated_fv.online)

        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)

        # Disable online
        disabled_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(enable=False),
        )
        self.assertFalse(disabled_fv.online)

        online_tables_after = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables_after), 0)

    def test_postgres_oft_suspend_resume(self) -> None:
        """Test suspend/resume on a FV with Postgres OFT."""
        fv_name = "spec_oft_suspend_resume"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="10m",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="15s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )
        self.fs.register_feature_view(fv, "v1")

        # Suspend
        suspended_fv = self.fs.suspend_feature_view(fv_name, "v1")
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Verify OFT still exists but is suspended
        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)
        self.assertEqual(online_tables[0]["scheduling_state"], "SUSPENDED")

        # Resume
        resumed_fv = self.fs.resume_feature_view(fv_name, "v1")
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        online_tables_after = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables_after), 1)
        self.assertIn(online_tables_after[0]["scheduling_state"], ["RUNNING"])

    # =========================================================================
    # Online config serialization with store_type
    # =========================================================================

    def test_online_config_postgres_serialization(self) -> None:
        """Test OnlineConfig serialization round-trips with Postgres store type."""
        config = feature_view.OnlineConfig(
            enable=True,
            target_lag="15s",
            store_type=feature_view.OnlineStoreType.POSTGRES,
        )
        json_str = config.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["store_type"], "postgres")

        reconstructed = feature_view.OnlineConfig.from_json(json_str)
        self.assertEqual(reconstructed.store_type, feature_view.OnlineStoreType.POSTGRES)
        self.assertEqual(reconstructed.enable, True)
        self.assertEqual(reconstructed.target_lag, "15s")

    def test_online_config_backward_compat_no_store_type(self) -> None:
        """Test that old configs without store_type deserialize to HYBRID_TABLE."""
        old_json = '{"enable": true, "target_lag": "10s"}'
        config = feature_view.OnlineConfig.from_json(old_json)
        self.assertEqual(config.store_type, feature_view.OnlineStoreType.HYBRID_TABLE)

    def test_list_feature_views_postgres_online_config(self) -> None:
        """Test list_feature_views shows Postgres store_type in online config."""
        fv_name = "spec_oft_list"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            online_config=feature_view.OnlineConfig(
                enable=True,
                target_lag="20s",
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )
        self.fs.register_feature_view(fv, "v1")

        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)

        online_config = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertTrue(online_config["enable"])

    # =========================================================================
    # Schema validation
    # =========================================================================

    def test_postgres_oft_rejects_unsupported_column_type(self) -> None:
        """Test that unsupported column types (e.g. DATE) are rejected with a clear error."""
        # Create a table with a DATE column — not supported for Postgres OFT
        bad_table_name = f"TEST_BAD_TYPES_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{bad_table_name} (
                user_id INT,
                birthday DATE,
                purchase_time TIMESTAMP_NTZ
            )
        """
        ).collect()

        bad_data = self._session.table(f"{self.fs._config.full_schema_path}.{bad_table_name}")

        fv = feature_view.FeatureView(
            name="spec_oft_bad_types",
            entities=[self.user_entity],
            feature_df=bad_data,
            timestamp_col="purchase_time",
            refresh_freq="1h",
            online_config=feature_view.OnlineConfig(
                enable=True,
                store_type=feature_view.OnlineStoreType.POSTGRES,
            ),
        )

        with self.assertRaises(Exception) as ctx:
            self.fs.register_feature_view(fv, "v1")

        self.assertIn("DateType", str(ctx.exception))
        self.assertIn("BIRTHDAY", str(ctx.exception))


if __name__ == "__main__":
    absltest.main()
