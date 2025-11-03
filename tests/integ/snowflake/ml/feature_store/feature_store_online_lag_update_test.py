"""Integration tests for SNOW-2432363: Updating online target lag should not disable online."""

import uuid

import common_utils
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml.feature_store import entity, feature_store, feature_view
from snowflake.ml.utils import connection_params


class FeatureStoreOnlineLagUpdateTest(parameterized.TestCase):
    """Test class for verifying online lag updates preserve online state (SNOW-2432363 fix)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test class with feature store and entities."""
        cls._session_config = connection_params.SnowflakeLoginOptions()
        cls._session = snowpark.Session.builder.configs(cls._session_config).create()
        cls._active_feature_store = []

        try:
            cls._session.sql(f"CREATE DATABASE IF NOT EXISTS {common_utils.FS_INTEG_TEST_DUMMY_DB}").collect()
            cls._session.sql(f"CREATE DATABASE IF NOT EXISTS {common_utils.FS_INTEG_TEST_DB}").collect()
            cls._session.sql(
                f"CREATE SCHEMA IF NOT EXISTS {common_utils.FS_INTEG_TEST_DB}."
                f"{common_utils.FS_INTEG_TEST_DATASET_SCHEMA}"
            ).collect()
            common_utils.cleanup_temporary_objects(cls._session)

            cls.test_db = common_utils.FS_INTEG_TEST_DB
            cls.test_schema = common_utils.create_random_schema(cls._session, "LAG_UPDATE_TEST", cls.test_db)
            cls.warehouse = common_utils.get_test_warehouse_name(cls._session)
        except Exception as e:
            cls.tearDownClass()
            raise Exception(f"Test setup failed: {e}")

        cls.fs = feature_store.FeatureStore(
            session=cls._session,
            database=cls.test_db,
            name=cls.test_schema,
            default_warehouse=cls.warehouse,
            creation_mode=feature_store.CreationMode.CREATE_IF_NOT_EXIST,
        )
        cls._active_feature_store.append(cls.fs)

        # Create test entity
        cls.user_entity = entity.Entity(name="user_entity", join_keys=["user_id"], desc="User entity")
        cls.fs.register_entity(cls.user_entity)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test class resources."""
        try:
            if cls._active_feature_store:
                for fs in cls._active_feature_store:
                    try:
                        fs.delete_feature_store()
                    except Exception:
                        pass
        finally:
            try:
                common_utils.cleanup_temporary_objects(cls._session)
                cls._session.close()
            except Exception:
                pass

    def setUp(self) -> None:
        """Set up test case with sample data."""
        # Create a real table for testing since create_dataframe with VALUES has limitations
        self.test_table_name = f"TEST_LAG_UPDATE_DATA_{uuid.uuid4().hex.upper()[:8]}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{self.test_table_name} (
                user_id INT,
                purchase_amount INT,
                purchase_time TIMESTAMP
            )
        """
        ).collect()

        # Insert test data
        self._session.sql(
            f"""
            INSERT INTO {self.fs._config.full_schema_path}.{self.test_table_name} VALUES
            (1, 100, '2023-01-01 00:00:00'::TIMESTAMP),
            (2, 200, '2023-01-02 00:00:00'::TIMESTAMP),
            (3, 300, '2023-01-03 00:00:00'::TIMESTAMP)
        """
        ).collect()

        # Create DataFrame from table
        self.sample_data = self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}")

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


if __name__ == "__main__":
    absltest.main()
