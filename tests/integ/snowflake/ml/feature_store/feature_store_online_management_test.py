import json
import uuid

import common_utils
from absl.testing import absltest, parameterized
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import entity, feature_store, feature_view


class FeatureStoreOnlineTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Test class for online feature store functionality (management/config/lifecycle) of feature view."""

    def setUp(self) -> None:
        """Set up test case with feature store, entities, and sample data."""
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
        self.user_entity = entity.Entity(name="user_entity", join_keys=["user_id"], desc="User entity for testing")
        self.fs.register_entity(self.user_entity)

        self.product_entity = entity.Entity(
            name="product_entity", join_keys=["product_id"], desc="Product entity for testing"
        )
        self.fs.register_entity(self.product_entity)
        # Create a real table for testing since create_dataframe with VALUES has limitations
        self.test_table_name = f"TEST_ONLINE_DATA_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{self.test_table_name} (
                user_id INT,
                product_id INT,
                purchase_amount INT,
                purchase_time TIMESTAMP
            )
        """
        ).collect()

        # Insert test data
        self._session.sql(
            f"""
            INSERT INTO {self.fs._config.full_schema_path}.{self.test_table_name} VALUES
            (1, 100, 1000, '2023-01-01 00:00:00'::TIMESTAMP),
            (2, 200, 2000, '2023-01-02 00:00:00'::TIMESTAMP),
            (3, 300, 3000, '2023-01-03 00:00:00'::TIMESTAMP)
        """
        ).collect()

        # Create DataFrame from table
        self.sample_data = self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}")

    def tearDown(self) -> None:
        """Clean up test case resources."""
        try:
            self.fs._clear(dryrun=False)
        except Exception:
            pass
        super().tearDown()

    def test_create_feature_view_with_online_default_config(self) -> None:
        """Test creating feature view with online enabled using default configuration."""
        fv_name = "test_online_fv_default"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="1m",
            desc="Test feature view with online support",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Validate default configuration
        self.assertTrue(registered_fv.online)
        self.assertIsNotNone(registered_fv.online_config)
        self.assertEqual(registered_fv.online_config.target_lag, "10 seconds")

        actual_online_name = registered_fv.fully_qualified_online_table_name()
        self.assertIsNotNone(actual_online_name)
        self.assertIn("$ONLINE", actual_online_name)

        # Verify online table exists in backend
        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.name == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        # Parse and verify online config
        self.assertIsNotNone(fv_rows[0]["ONLINE_CONFIG"])
        online_config = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertTrue(online_config["enable"])
        self.assertEqual(online_config["target_lag"], "10 seconds")

    def test_create_feature_view_with_custom_online_config(self) -> None:
        """Test creating feature view with custom online configuration."""
        fv_name = "test_online_fv_custom"

        config = feature_view.OnlineConfig(enable=True, target_lag="15s")

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="5m",
            desc="Test feature view with custom online config",
            online_config=config,
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Validate custom configuration
        self.assertTrue(registered_fv.online)
        self.assertIsNotNone(registered_fv.online_config)
        self.assertEqual(registered_fv.online_config.target_lag, "15 seconds")

        # Verify custom configuration
        self.assertTrue(registered_fv.online)
        self.assertIsNotNone(registered_fv.online_config)
        self.assertEqual(registered_fv.online_config.target_lag, "15 seconds")

    def test_create_feature_view_without_online(self) -> None:
        """Test creating feature view without online support."""
        fv_name = "test_offline_fv"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            desc="Test offline-only feature view",
            online_config=feature_view.OnlineConfig(enable=False),  # Explicitly disable online
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Verify no online support
        self.assertFalse(registered_fv.online)
        self.assertIsNotNone(registered_fv.online_config)  # Should still have config but with enable=False
        self.assertFalse(registered_fv.online_config.enable)
        # Should raise RuntimeError when trying to get online table name for offline feature view
        with self.assertRaises(RuntimeError):
            registered_fv.fully_qualified_online_table_name()

        # Verify in list result
        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.NAME == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)
        # Online config should show default disabled config when no online feature table exists
        self.assertIsNotNone(fv_rows[0]["ONLINE_CONFIG"])
        list_online_config = json.loads(fv_rows[0]["ONLINE_CONFIG"])
        self.assertFalse(list_online_config["enable"])
        self.assertEqual(list_online_config["target_lag"], "10 seconds")

        # Verify no online table exists
        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 0)

    def test_update_feature_view_enable_online(self) -> None:
        """Test enabling online storage for existing feature view."""
        fv_name = "test_update_enable_online_fv"

        # Create offline-only feature view
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            desc="Test update to enable online",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertFalse(registered_fv.online)

        # Enable online storage
        custom_config = feature_view.OnlineConfig(enable=True, target_lag="15s")
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=custom_config,
        )

        # Validate online is now enabled
        self.assertTrue(updated_fv.online)
        self.assertIsNotNone(updated_fv.online_config)
        self.assertEqual(updated_fv.online_config.target_lag, "15 seconds")

    def test_update_feature_view_disable_online(self) -> None:
        """Test disabling online storage for existing feature view."""
        fv_name = "test_update_disable_online_fv"

        # Create online-enabled feature view
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="20m",
            desc="Test update to disable online",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online)

        # Disable online storage
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        # Verify online is now disabled
        self.assertFalse(updated_fv.online)

    def test_update_online_configuration_only(self) -> None:
        """Test updating only online configuration without changing online status."""
        fv_name = "test_update_online_config_fv"

        # Create online-enabled feature view
        initial_config = feature_view.OnlineConfig(enable=True, target_lag="10s")
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="30m",
            desc="Test update online config only",
            online_config=initial_config,
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertEqual(registered_fv.online_config.target_lag, "10 seconds")

        # Update online configuration
        new_config = feature_view.OnlineConfig(enable=True, target_lag="12s")
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=new_config,
        )

        # Validate updated configuration
        self.assertTrue(updated_fv.online)
        self.assertEqual(updated_fv.online_config.target_lag, "12 seconds")

    def test_update_online_noop_when_target_lag_unspecified(self) -> None:
        """When already online, enable=True with no target_lag should not change current target_lag."""
        fv_name = "test_update_online_noop_no_target_lag"

        # Create online-enabled feature view with a non-default target_lag
        initial_config = feature_view.OnlineConfig(enable=True, target_lag="18s")
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="30m",
            desc="Test update online no-op when target_lag not provided",
            online_config=initial_config,
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online)
        self.assertEqual(registered_fv.online_config.target_lag, "18 seconds")

        # Call update with enable=True but without specifying target_lag
        noop_config = feature_view.OnlineConfig(enable=True)
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=noop_config,
        )

        # Validate target_lag remains unchanged
        self.assertTrue(updated_fv.online)
        self.assertIsNotNone(updated_fv.online_config)
        self.assertEqual(updated_fv.online_config.target_lag, "18 seconds")

    def test_delete_feature_view_with_online_table(self) -> None:
        """Test deleting feature view also deletes online table."""
        fv_name = "test_delete_online_fv"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="1h",
            desc="Test delete with online table",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online)

        # Verify feature view exists
        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.name == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 1)

        # Delete feature view
        self.fs.delete_feature_view(registered_fv)

        # Verify feature view and online table are both deleted
        list_result = self.fs.list_feature_views()
        fv_rows = list_result.filter(list_result.name == fv_name.upper()).collect()
        self.assertEqual(len(fv_rows), 0)

    def test_online_config_serialization(self) -> None:
        """Test OnlineConfig serialization and deserialization."""
        # Test with all fields
        config = feature_view.OnlineConfig(enable=True, target_lag="15s")
        json_str = config.to_json()
        reconstructed = feature_view.OnlineConfig.from_json(json_str)

        self.assertEqual(config.enable, reconstructed.enable)
        self.assertEqual(config.target_lag, reconstructed.target_lag)

        # Test with defaults
        default_config = feature_view.OnlineConfig()
        json_str = default_config.to_json()
        reconstructed = feature_view.OnlineConfig.from_json(json_str)

        self.assertEqual(default_config.enable, reconstructed.enable)
        self.assertEqual(default_config.target_lag, reconstructed.target_lag)

    def test_multi_entity_online_feature_view(self) -> None:
        """Test creating online feature view with multiple entities."""
        fv_name = "test_multi_entity_online_fv"

        # Create data with multiple join keys using a real table
        multi_table_name = f"TEST_MULTI_DATA_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{multi_table_name} (
                user_id INT,
                product_id INT,
                feature_value INT,
                event_time TIMESTAMP
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {self.fs._config.full_schema_path}.{multi_table_name} VALUES
            (1, 100, 500, '2023-01-01 00:00:00'::TIMESTAMP),
            (2, 200, 600, '2023-01-02 00:00:00'::TIMESTAMP),
            (3, 300, 700, '2023-01-03 00:00:00'::TIMESTAMP)
        """
        ).collect()

        multi_entity_data = self._session.table(f"{self.fs._config.full_schema_path}.{multi_table_name}")

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity, self.product_entity],
            feature_df=multi_entity_data,
            timestamp_col="event_time",
            refresh_freq="5m",
            desc="Test multi-entity online feature view",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="18s"),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Verify online functionality with multiple entities
        self.assertTrue(registered_fv.online)
        self.assertEqual(len(registered_fv.entities), 2)
        self.assertIsNotNone(registered_fv.fully_qualified_online_table_name())

    def test_online_feature_view_without_timestamp(self) -> None:
        """Test creating online feature view without timestamp column."""
        fv_name = "test_no_timestamp_online_fv"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            # No timestamp_col
            refresh_freq="1h",
            desc="Test online feature view without timestamp",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Should still work without timestamp
        self.assertTrue(registered_fv.online)
        self.assertIsNone(registered_fv.timestamp_col)

    def test_static_feature_view_with_online(self) -> None:
        """Test creating static feature view with online support."""
        fv_name = "test_static_online_fv"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            # No refresh_freq makes it static
            desc="Test static feature view with online",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Static feature view can still have online support
        self.assertTrue(registered_fv.online)
        self.assertIsNone(registered_fv.refresh_freq)
        self.assertEqual(registered_fv.online_config.target_lag, "10 seconds")

    def test_feature_view_get_with_online_info(self) -> None:
        """Test getting feature view reconstructs online information correctly."""
        fv_name = "test_get_online_fv"

        original_config = feature_view.OnlineConfig(enable=True, target_lag="17s")
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="45m",
            desc="Test get with online info",
            online_config=original_config,
        )

        self.fs.register_feature_view(fv, "v1")

        # Get feature view and verify online info is reconstructed
        retrieved_fv = self.fs.get_feature_view(fv_name, "v1")

        self.assertTrue(retrieved_fv.online)
        self.assertIsNotNone(retrieved_fv.online_config)
        self.assertIsNotNone(retrieved_fv.fully_qualified_online_table_name())

    def test_list_feature_views_online_columns(self) -> None:
        """Test list_feature_views includes online information in results."""
        fv_name_online = "test_list_online_fv"
        fv_name_offline = "test_list_offline_fv"

        # Create one online and one offline feature view
        online_fv = feature_view.FeatureView(
            name=fv_name_online,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            desc="Online feature view for list test",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="16s"),
        )

        offline_fv = feature_view.FeatureView(
            name=fv_name_offline,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="2h",
            desc="Offline feature view for list test",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        self.fs.register_feature_view(online_fv, "v1")
        self.fs.register_feature_view(offline_fv, "v1")

        # List feature views and check online columns
        list_result = self.fs.list_feature_views()
        columns = list_result.columns

        # Verify online config column is present
        self.assertIn("ONLINE_CONFIG", columns)

        # Check specific rows
        online_rows = list_result.filter(list_result.NAME == fv_name_online.upper()).collect()
        offline_rows = list_result.filter(list_result.NAME == fv_name_offline.upper()).collect()

        self.assertEqual(len(online_rows), 1)
        self.assertEqual(len(offline_rows), 1)

        # Verify online config - online feature view should have config, offline should not
        self.assertIsNotNone(online_rows[0]["ONLINE_CONFIG"])

        # Parse online config to verify it's enabled
        online_config = json.loads(online_rows[0]["ONLINE_CONFIG"])
        self.assertTrue(online_config["enable"])

        # Verify that runtime metadata fields are included for enabled online feature views
        self.assertIn("refresh_mode", online_config)
        self.assertIn("scheduling_state", online_config)
        self.assertEqual(online_config["target_lag"], "16 seconds")
        self.assertIn(online_config["refresh_mode"], ["AUTO", "FULL", "INCREMENTAL"])
        self.assertIn(online_config["scheduling_state"], ["RUNNING", "SUSPENDED"])

        # Offline feature view should have config but with enable=false
        self.assertIsNotNone(offline_rows[0]["ONLINE_CONFIG"])
        offline_config = json.loads(offline_rows[0]["ONLINE_CONFIG"])
        self.assertFalse(offline_config["enable"])
        self.assertNotIn("refresh_mode", offline_config)
        self.assertNotIn("scheduling_state", offline_config)
        self.assertEqual(offline_config["target_lag"], "10 seconds")

    def test_overwrite_feature_view_with_online(self) -> None:
        """Test overwriting feature view with different online configuration."""
        fv_name = "test_overwrite_online_fv"

        # Create initial feature view with online
        fv1 = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="1h",
            desc="Initial online feature view",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="10s"),
        )

        self.fs.register_feature_view(fv1, "v1")
        retrieved_fv1 = self.fs.get_feature_view(fv_name, "v1")
        self.assertTrue(retrieved_fv1.online)

        # Overwrite with different online configuration
        fv2 = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount"),
            refresh_freq="2h",
            desc="Overwritten online feature view",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="15s"),
        )

        self.fs.register_feature_view(fv2, "v1", overwrite=True)
        retrieved_fv2 = self.fs.get_feature_view(fv_name, "v1")

        # Verify overwrite worked
        self.assertTrue(retrieved_fv2.online)
        self.assertEqual(retrieved_fv2.online_config.target_lag, "15 seconds")
        self.assertEqual(retrieved_fv2.desc, "Overwritten online feature view")

    def test_suspend_resume_feature_view_with_online(self) -> None:
        """Test suspend and resume operations on feature views with online tables."""
        fv_name = "test_suspend_resume_online"

        # Create feature view with online enabled
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="10m",
            desc="Test suspend/resume with online feature view",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="15s"),  # Must be >= 10s
        )

        self.fs.register_feature_view(fv, "v1")

        # Verify feature view is initially active
        initial_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(initial_fv.status.value, "ACTIVE")
        self.assertTrue(initial_fv.online)

        # Test suspend operation - should suspend both offline and online
        suspended_fv = self.fs.suspend_feature_view(fv_name, "v1")
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Verify the feature view is suspended in the backend
        backend_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(backend_fv.status.value, "SUSPENDED")

        # Test resume operation - should resume both offline and online
        resumed_fv = self.fs.resume_feature_view(fv_name, "v1")
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        # Verify the feature view is active again in the backend
        final_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(final_fv.status.value, "ACTIVE")

        # Verify online table still exists and is properly configured
        self.assertTrue(final_fv.online)
        self.assertIsNotNone(final_fv.fully_qualified_online_table_name())

    def test_suspend_resume_feature_view_without_online(self) -> None:
        """Test suspend and resume operations on feature views without online tables."""
        fv_name = "test_suspend_resume_offline_only"

        # Create feature view without online enabled
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            desc="Test suspend/resume with offline-only feature view",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        self.fs.register_feature_view(fv, "v1")

        # Verify feature view is initially active and has no online table
        initial_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(initial_fv.status.value, "ACTIVE")
        self.assertFalse(initial_fv.online)
        with self.assertRaises(RuntimeError):
            initial_fv.fully_qualified_online_table_name()

        # Test suspend operation - should suspend only offline
        suspended_fv = self.fs.suspend_feature_view(fv_name, "v1")
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Test resume operation - should resume only offline
        resumed_fv = self.fs.resume_feature_view(fv_name, "v1")
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        # Verify feature view is back to active and still has no online table
        final_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(final_fv.status.value, "ACTIVE")
        self.assertFalse(final_fv.online)
        with self.assertRaises(RuntimeError):
            final_fv.fully_qualified_online_table_name()

    def test_suspend_resume_with_feature_view_object(self) -> None:
        """Test suspend and resume operations using FeatureView objects instead of name/version."""
        fv_name = "test_suspend_resume_obj"

        # Create feature view with online enabled
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="20m",
            desc="Test suspend/resume with FeatureView object",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Test suspend using FeatureView object
        suspended_fv = self.fs.suspend_feature_view(registered_fv)
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Test resume using FeatureView object
        resumed_fv = self.fs.resume_feature_view(suspended_fv)
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        # Verify final state
        final_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(final_fv.status.value, "ACTIVE")
        self.assertTrue(final_fv.online)

    def test_suspend_resume_explicit_online_offline_verification(self) -> None:
        """Test suspend/resume with explicit verification of both online and offline states."""
        fv_name = "test_explicit_suspend_resume"

        # Create feature view with online enabled
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="5m",
            desc="Test explicit suspend/resume verification",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="12s"),
        )

        self.fs.register_feature_view(fv, "v1")

        # Verify initial state - both offline and online should be active
        initial_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(initial_fv.status.value, "ACTIVE")
        self.assertTrue(initial_fv.online)

        # Verify we can read from both stores initially
        offline_data_initial = self.fs.read_feature_view(initial_fv, store_type=feature_view.StoreType.OFFLINE)
        online_data_initial = self.fs.read_feature_view(initial_fv, store_type=feature_view.StoreType.ONLINE)
        self.assertEqual(len(offline_data_initial.collect()), 3)
        self.assertEqual(len(online_data_initial.collect()), 3)

        # Verify initial online table state - should be RUNNING or ACTIVE
        initial_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(initial_online_tables), 1)
        initial_online_state = initial_online_tables[0]["scheduling_state"]
        self.assertIn(initial_online_state, ["RUNNING"])

        # Suspend - should suspend both offline and online
        suspended_fv = self.fs.suspend_feature_view(fv_name, "v1")
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Verify suspension affects both stores
        suspended_backend_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(suspended_backend_fv.status.value, "SUSPENDED")

        # Online table should still exist but be suspended - verify with SHOW command
        self.assertTrue(suspended_backend_fv.online)
        self.assertIsNotNone(suspended_backend_fv.fully_qualified_online_table_name())

        suspended_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(suspended_online_tables), 1)
        suspended_online_state = suspended_online_tables[0]["scheduling_state"]
        self.assertEqual(suspended_online_state, "SUSPENDED")

        # Resume - should resume both offline and online
        resumed_fv = self.fs.resume_feature_view(fv_name, "v1")
        self.assertEqual(resumed_fv.status.value, "ACTIVE")

        # Verify resumption restores both stores
        final_fv = self.fs.get_feature_view(fv_name, "v1")
        self.assertEqual(final_fv.status.value, "ACTIVE")
        self.assertTrue(final_fv.online)

        # Verify online table state is back to ACTIVE or RUNNING - verify with SHOW command
        resumed_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(resumed_online_tables), 1)
        resumed_online_state = resumed_online_tables[0]["scheduling_state"]
        self.assertIn(resumed_online_state, ["RUNNING"])

        # Verify online table configuration is preserved
        self.assertEqual(final_fv.online_config.target_lag, "12 seconds")

    def test_online_feature_table_with_different_session_schema(self) -> None:
        """Test OFT creation works when session schema differs from FS schema (SNOW-2430972).

        Scenario:
        1. FS created in: DB.FS_SCHEMA
        2. Session set to: DB.OTHER_SCHEMA (different)
        3. Register FV with online config
        4. Verify: OFT created successfully using FQDN

        Bug (unfixed): OFT references offline FV as unqualified "FV$v1" → searches wrong schema → fails
        Fix: OFT references offline FV as FQDN "DB.FS_SCHEMA.FV$v1" → finds it correctly → succeeds
        """
        # Create another schema for testing cross-schema scenarios
        other_schema = common_utils.create_random_schema(self._session, "OTHER_SCHEMA_FQDN", self.test_db)

        fv_name = "test_fqdn_cross_schema"
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="5m",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="10s"),
        )

        # Switch session to different schema (triggers the bug scenario)
        self._session.use_schema(f"{self.test_db}.{other_schema}")

        try:
            registered_fv = self.fs.register_feature_view(fv, "v1")

            self.assertTrue(registered_fv.online)
            self.assertIsNotNone(registered_fv.online_config)

            online_table_name = registered_fv.fully_qualified_online_table_name()
            self.assertIsNotNone(online_table_name)
            self.assertIn("$ONLINE", online_table_name)

            # Verify online table exists
            online_tables = self._session.sql(
                f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
            ).collect()
            self.assertEqual(len(online_tables), 1)

            # Verify can read from online table
            result = self.fs.read_feature_view(
                registered_fv, keys=[["1"]], store_type=feature_view.StoreType.ONLINE
            ).collect()
            self.assertEqual(len(result), 1)

        finally:
            self._session.use_schema(f"{self.test_db}.{self.test_schema}")

    def test_update_online_lag_without_enable_preserves_online_state(self) -> None:
        """Test that updating target_lag without specifying enable preserves online state.

        Reproduces SNOW-2432363: When updating OnlineConfig with only target_lag specified,
        the online feature view should stay enabled, not get disabled.
        """
        fv_name = "test_lag_update_preserve_online"

        # Step 1: Create a feature view with online ENABLED and initial lag
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            desc="Test lag update",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="5m"),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online, "Feature view should be online after registration")
        self.assertIn("5", registered_fv.online_config.target_lag)

        # Step 2: Update ONLY the target_lag WITHOUT specifying enable parameter
        # This should preserve the online state (stay enabled)
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(target_lag="10m"),  # Only lag, no enable specified
        )

        # Step 3: Verify online is still ENABLED (bug fix)
        self.assertTrue(updated_fv.online, "Online should stay ENABLED after lag update (SNOW-2432363 fix)")
        self.assertIn("10", updated_fv.online_config.target_lag, "Target lag should be updated to 10m")

    def test_update_online_lag_with_explicit_enable_true(self) -> None:
        """Test that explicitly setting enable=True while updating lag works correctly."""
        fv_name = "test_lag_update_explicit_enable_true"

        # Create offline feature view
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            desc="Test explicit enable=True",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertFalse(registered_fv.online, "Feature view should NOT be online initially")

        # Update with explicit enable=True and target_lag
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="8m"),
        )

        self.assertTrue(updated_fv.online, "Online should be ENABLED after explicit enable=True")
        self.assertIn("8", updated_fv.online_config.target_lag, "Target lag should be 8m")

    def test_update_online_with_explicit_enable_false_disables(self) -> None:
        """Test that explicitly setting enable=False still disables online storage."""
        fv_name = "test_lag_update_explicit_enable_false"

        # Create online feature view
        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="15m",
            desc="Test explicit enable=False",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="5m"),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")
        self.assertTrue(registered_fv.online, "Feature view should be online initially")

        # Update with explicit enable=False (should disable)
        updated_fv = self.fs.update_feature_view(
            name=fv_name,
            version="v1",
            online_config=feature_view.OnlineConfig(enable=False),
        )

        self.assertFalse(updated_fv.online, "Online should be DISABLED after explicit enable=False")

    def test_online_feature_table_creation_uses_fqdn_in_sql(self) -> None:
        """Verify offline and online tables use same database and schema (SNOW-2430972)."""
        fv_name = "test_fqdn_verification_mgmt"

        fv = feature_view.FeatureView(
            name=fv_name,
            entities=[self.user_entity],
            feature_df=self.sample_data.select("user_id", "purchase_amount", "purchase_time"),
            timestamp_col="purchase_time",
            refresh_freq="5m",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        fqdn_offline = registered_fv.fully_qualified_name()
        fqdn_online = registered_fv.fully_qualified_online_table_name()

        offline_parts = fqdn_offline.split(".")
        online_parts = fqdn_online.split(".")

        self.assertEqual(len(offline_parts), 3)
        self.assertEqual(len(online_parts), 3)

        # Verify same database and schema
        self.assertEqual(offline_parts[0], online_parts[0])
        self.assertEqual(offline_parts[1], online_parts[1])

        # Verify online table exists
        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%{fv_name.upper()}%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 1)


if __name__ == "__main__":
    absltest.main()
