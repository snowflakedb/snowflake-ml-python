"""Integration tests for Feature Store rollback and transaction safety logic."""

from __future__ import annotations

import typing
import unittest.mock as mock

import common_utils
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml.feature_store import entity, feature_store, feature_view
from snowflake.ml.utils import connection_params


class FeatureStoreRollbackTest(parameterized.TestCase):
    """Test rollback and transaction safety mechanisms in Feature Store operations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures, if any."""
        cls._session_config = connection_params.SnowflakeLoginOptions()
        cls._session = snowpark.Session.builder.configs(cls._session_config).create()
        cls._test_db = common_utils.FS_INTEG_TEST_DB
        cls._test_schema = common_utils.create_random_schema(cls._session, "ROLLBACK_TEST", cls._test_db)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down test fixtures, if any."""
        common_utils.cleanup_temporary_objects(cls._session)
        cls._session.close()

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self._test_warehouse = common_utils.get_test_warehouse_name(self._session)
        self.fs = feature_store.FeatureStore(
            session=self._session,
            database=self._test_db,
            name=self._test_schema,
            default_warehouse=self._test_warehouse,
            creation_mode=feature_store.CreationMode.CREATE_IF_NOT_EXIST,
        )

        # Create test entity
        self.test_entity = entity.Entity("user_id", ["user_id"])
        self.fs.register_entity(self.test_entity)

        # Create test table with sample data
        self.test_table_name = "rollback_test_data"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {self.fs._config.full_schema_path}.{self.test_table_name} (
                user_id INT,
                feature_value FLOAT,
                timestamp_col TIMESTAMP
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {self.fs._config.full_schema_path}.{self.test_table_name} VALUES
            (1, 10.5, '2023-01-01 00:00:00'::TIMESTAMP),
            (2, 20.5, '2023-01-02 00:00:00'::TIMESTAMP),
            (3, 30.5, '2023-01-03 00:00:00'::TIMESTAMP)
        """
        ).collect()

    def tearDown(self) -> None:
        """Clean up after each test method."""
        # Clean up test table
        try:
            self._session.sql(
                f"DROP TABLE IF EXISTS {self.fs._config.full_schema_path}.{self.test_table_name}"
            ).collect()
        except Exception:
            pass

    def test_register_feature_view_rollback_offline_failure(self) -> None:
        """Test rollback when offline feature view creation fails."""
        fv = feature_view.FeatureView(
            name="test_rollback_offline",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Test rollback on offline failure",
        )

        # Mock the dynamic table creation to fail
        with mock.patch.object(self.fs, "_create_dynamic_table") as mock_create_dt:
            mock_create_dt.side_effect = Exception("Simulated offline creation failure")

            # Attempt to register - should fail and rollback
            with self.assertRaises(Exception) as context:
                self.fs.register_feature_view(fv, "v1")

            self.assertIn("Failed to register feature view", str(context.exception))

            # Verify no residual objects exist
            with self.assertRaises((exceptions.SnowflakeMLException, ValueError)):
                self.fs.get_feature_view("test_rollback_offline", "v1")

    def test_register_feature_view_rollback_online_failure(self) -> None:
        """Test rollback when online feature table creation fails after successful offline creation."""
        fv = feature_view.FeatureView(
            name="test_rollback_online",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Test rollback on online failure",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="10s"),
        )

        # Mock online table creation to fail after offline succeeds
        with mock.patch.object(self.fs, "_create_online_feature_table") as mock_create_online:
            mock_create_online.side_effect = Exception("Simulated online creation failure")

            # Attempt to register - should fail and rollback both offline and online
            with self.assertRaises(Exception) as context:
                self.fs.register_feature_view(fv, "v1")

            self.assertIn("feature view", str(context.exception))
            self.assertIn("Failed", str(context.exception))

            # Verify no residual objects exist (offline should be rolled back too)
            with self.assertRaises((exceptions.SnowflakeMLException, ValueError)):
                self.fs.get_feature_view("test_rollback_online", "v1")

            # Verify no online table exists
            online_tables = self._session.sql(
                f"SHOW ONLINE FEATURE TABLES LIKE '%test_rollback_online%' IN SCHEMA {self.fs._config.full_schema_path}"
            ).collect()
            self.assertEqual(len(online_tables), 0)

    def test_update_feature_view_rollback_offline_failure(self) -> None:
        """Test rollback when update_feature_view offline update fails."""
        # First, create a successful feature view
        fv = feature_view.FeatureView(
            name="test_update_rollback",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Original description",
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Mock session.sql to fail on ALTER DYNAMIC TABLE
        original_sql = self._session.sql

        def mock_sql(query: str) -> typing.Any:
            if "ALTER DYNAMIC TABLE" in query and "TARGET_LAG = '2h'" in query:
                raise Exception("Simulated offline update failure")
            return original_sql(query)

        with mock.patch.object(self._session, "sql", side_effect=mock_sql):
            with self.assertRaises(Exception) as context:
                self.fs.update_feature_view(registered_fv, refresh_freq="2h", desc="Updated description")

            self.assertIn("Update feature view", str(context.exception))

        # Verify original values are preserved (rollback worked)
        updated_fv = self.fs.get_feature_view("test_update_rollback", "v1")
        self.assertIn("1", updated_fv.refresh_freq)  # Should remain original (format may vary)
        self.assertEqual(updated_fv.desc, "Original description")  # Should remain original

    def test_update_feature_view_rollback_online_enable_failure(self) -> None:
        """Test rollback when enabling online storage fails during update."""
        # Create offline-only feature view
        fv = feature_view.FeatureView(
            name="test_enable_online_rollback",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Test enable online rollback",
            online_config=feature_view.OnlineConfig(enable=False),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Mock online table creation to fail
        with mock.patch.object(self.fs, "_create_online_feature_table") as mock_create_online:
            mock_create_online.side_effect = Exception("Simulated online enable failure")

            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.update_feature_view(
                    registered_fv, online_config=feature_view.OnlineConfig(enable=True, target_lag="15s")
                )

        # Verify no online table was created (rollback worked)
        updated_fv = self.fs.get_feature_view("test_enable_online_rollback", "v1")
        with self.assertRaises(RuntimeError):
            updated_fv.fully_qualified_online_table_name()

        # Verify no orphaned online tables exist
        online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE '%test_enable_online_rollback%' IN SCHEMA "
            f"{self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(online_tables), 0)

    def test_update_feature_view_rollback_online_disable_failure(self) -> None:
        """Test rollback when disabling online storage fails during update."""
        # Create online-enabled feature view
        fv = feature_view.FeatureView(
            name="test_disable_online_rollback",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Test disable online rollback",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="10s"),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Verify online table exists
        self.assertIsNotNone(registered_fv.fully_qualified_online_table_name())

        # Mock session.sql to fail on DROP ONLINE FEATURE TABLE
        original_sql = self._session.sql

        def mock_sql(query: str) -> typing.Any:
            if "DROP ONLINE FEATURE TABLE" in query:
                raise Exception("Simulated online disable failure")
            return original_sql(query)

        with mock.patch.object(self._session, "sql", side_effect=mock_sql):
            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.update_feature_view(registered_fv, online_config=feature_view.OnlineConfig(enable=False))

        # Verify online table still exists (rollback worked - recreated the table)
        updated_fv = self.fs.get_feature_view("test_disable_online_rollback", "v1")
        self.assertIsNotNone(updated_fv.fully_qualified_online_table_name())

    def test_update_feature_view_rollback_mixed_operations(self) -> None:
        """Test rollback with multiple operations (offline + online config update)."""
        # Create online-enabled feature view
        fv = feature_view.FeatureView(
            name="test_mixed_rollback",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Original description",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="10s"),
        )
        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Mock to fail on online config update after offline update succeeds
        call_count = 0
        original_sql = self._session.sql

        def mock_sql(query: str) -> typing.Any:
            nonlocal call_count
            call_count += 1
            if "ALTER ONLINE FEATURE TABLE" in query and call_count > 1:
                raise Exception("Simulated online config update failure")
            return original_sql(query)

        with mock.patch.object(self._session, "sql", side_effect=mock_sql):
            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.update_feature_view(
                    registered_fv,
                    refresh_freq="2h",
                    desc="Updated description",
                    online_config=feature_view.OnlineConfig(enable=True, target_lag="30s"),
                )

        # Verify all changes were rolled back
        updated_fv = self.fs.get_feature_view("test_mixed_rollback", "v1")
        self.assertIn("1", updated_fv.refresh_freq)  # Should be reverted (format may vary)
        self.assertEqual(updated_fv.desc, "Original description")  # Should be reverted
        if updated_fv.online_config:
            self.assertIn("10", updated_fv.online_config.target_lag)  # Should be reverted (format may vary)

    def test_rollback_resource_cleanup_idempotent(self) -> None:
        """Test that rollback cleanup is idempotent and handles missing resources gracefully."""
        fv = feature_view.FeatureView(
            name="test_idempotent_rollback",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="1h",
            desc="Test idempotent rollback",
            online_config=feature_view.OnlineConfig(enable=True),
        )

        # Mock rollback to be called multiple times to test idempotency
        original_rollback = self.fs._rollback_created_resources
        rollback_call_count = 0

        def mock_rollback(resources: typing.Any) -> None:
            nonlocal rollback_call_count
            rollback_call_count += 1
            # Call original rollback multiple times to test idempotency
            for _ in range(2):
                original_rollback(resources)

        with mock.patch.object(self.fs, "_create_online_feature_table") as mock_create_online, mock.patch.object(
            self.fs, "_rollback_created_resources", side_effect=mock_rollback
        ):
            mock_create_online.side_effect = Exception("Simulated failure")

            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.register_feature_view(fv, "v1")

            # Verify rollback was called and handled gracefully
            self.assertEqual(rollback_call_count, 1)

    def test_atomic_suspend_rollback_on_failure(self) -> None:
        """Test that suspend operation is atomic and rolls back on failure."""
        # Create feature view with online enabled
        fv = feature_view.FeatureView(
            name="test_atomic_suspend",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="10m",
            desc="Test atomic suspend with rollback",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="15s"),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # Verify initial state - both offline and online should be active
        self.assertEqual(registered_fv.status.value, "ACTIVE")
        self.assertTrue(registered_fv.online)

        # Verify initial states using SHOW commands
        initial_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE 'TEST_ATOMIC_SUSPEND%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(initial_online_tables), 1)
        initial_online_state = initial_online_tables[0]["scheduling_state"]
        self.assertIn(initial_online_state, ["RUNNING", "ACTIVE"])

        # Mock the second operation (task) to fail, simulating partial failure
        original_sql = self._session.sql
        call_count = 0

        def mock_sql_with_failure(query: str) -> typing.Any:
            nonlocal call_count
            call_count += 1
            # Let first operation (ALTER DYNAMIC TABLE) succeed, but fail on second (ALTER TASK)
            if call_count == 2 and "ALTER TASK" in query and "SUSPEND" in query:
                raise Exception("Simulated task suspend failure")
            return original_sql(query)

        with mock.patch.object(self._session, "sql", side_effect=mock_sql_with_failure):
            # Attempt to suspend - should fail and rollback
            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.suspend_feature_view(registered_fv)

        # Verify that rollback worked - feature view should still be in original state
        final_fv = self.fs.get_feature_view("test_atomic_suspend", "v1")
        self.assertEqual(final_fv.status.value, "ACTIVE")  # Should be rolled back to ACTIVE
        self.assertTrue(final_fv.online)

        # Verify online table is still in original state
        final_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE 'TEST_ATOMIC_SUSPEND%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(final_online_tables), 1)
        final_online_state = final_online_tables[0]["scheduling_state"]
        self.assertIn(final_online_state, ["RUNNING", "ACTIVE"])  # Should be rolled back

    def test_atomic_resume_rollback_on_failure(self) -> None:
        """Test that resume operation is atomic and rolls back on failure."""
        # Create feature view with online enabled
        fv = feature_view.FeatureView(
            name="test_atomic_resume",
            entities=[self.test_entity],
            feature_df=self._session.table(f"{self.fs._config.full_schema_path}.{self.test_table_name}"),
            timestamp_col="timestamp_col",
            refresh_freq="10m",
            desc="Test atomic resume with rollback",
            online_config=feature_view.OnlineConfig(enable=True, target_lag="15s"),
        )

        registered_fv = self.fs.register_feature_view(fv, "v1")

        # First suspend successfully to set up the test
        suspended_fv = self.fs.suspend_feature_view(registered_fv)
        self.assertEqual(suspended_fv.status.value, "SUSPENDED")

        # Verify suspended states using SHOW commands
        suspended_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE 'TEST_ATOMIC_RESUME%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(suspended_online_tables), 1)
        suspended_online_state = suspended_online_tables[0]["scheduling_state"]
        self.assertEqual(suspended_online_state, "SUSPENDED")

        # Mock the online operation to fail, simulating partial failure during resume
        original_sql = self._session.sql
        call_count = 0

        def mock_sql_with_failure(query: str) -> typing.Any:
            nonlocal call_count
            call_count += 1
            # Let offline operations succeed, but fail on online operation
            if call_count == 3 and "ALTER ONLINE FEATURE TABLE" in query and "RESUME" in query:
                raise Exception("Simulated online resume failure")
            return original_sql(query)

        with mock.patch.object(self._session, "sql", side_effect=mock_sql_with_failure):
            # Attempt to resume - should fail and rollback
            with self.assertRaises((exceptions.SnowflakeMLException, RuntimeError)):
                self.fs.resume_feature_view(suspended_fv)

        # Verify that rollback worked - feature view should still be suspended
        final_fv = self.fs.get_feature_view("test_atomic_resume", "v1")
        self.assertEqual(final_fv.status.value, "SUSPENDED")  # Should be rolled back to SUSPENDED
        self.assertTrue(final_fv.online)

        # Verify online table is still suspended
        final_online_tables = self._session.sql(
            f"SHOW ONLINE FEATURE TABLES LIKE 'TEST_ATOMIC_RESUME%' IN SCHEMA {self.fs._config.full_schema_path}"
        ).collect()
        self.assertEqual(len(final_online_tables), 1)
        final_online_state = final_online_tables[0]["scheduling_state"]
        self.assertEqual(final_online_state, "SUSPENDED")  # Should be rolled back


if __name__ == "__main__":
    absltest.main()
