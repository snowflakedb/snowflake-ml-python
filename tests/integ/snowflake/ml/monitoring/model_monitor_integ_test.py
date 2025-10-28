import json
import uuid

from absl.testing import absltest, parameterized

from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.test_utils import db_manager, model_factory

INPUT_FEATURE_COLUMNS_NAMES = [f"input_feature_{i}" for i in range(64)]


class ModelMonitorRegistryIntegrationTest(parameterized.TestCase):
    def _get_monitor_segment_columns(self, monitor_name: str) -> list[str]:
        """Helper method to get segment columns from DESCRIBE MODEL MONITOR."""
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()

        # Access the columns column from the first row
        if not describe_result:
            raise AssertionError("DESCRIBE MODEL MONITOR returned empty result")

        columns_json_str = describe_result[0]["columns"]
        columns_json = json.loads(columns_json_str)
        return columns_json.get("segment_columns", [])

    def _create_test_table(
        self,
        fully_qualified_table_name: str,
        id_column_type: str = "STRING",
        segment_columns: list = None,
        custom_metric_columns: list = None,
    ) -> None:
        """Create a test table with optional segment columns for testing."""

        s = ", ".join([f"{i} FLOAT" for i in INPUT_FEATURE_COLUMNS_NAMES])

        # Build the segment columns part of the table definition
        segment_columns_def = ""
        if segment_columns:
            segment_columns_def = ", " + ", ".join([f"{col} STRING" for col in segment_columns])

        # Build the custom metric columns part of the table definition
        custom_metric_columns_def = ""
        if custom_metric_columns:
            custom_metric_columns_def = ", " + ", ".join([f"{col} FLOAT" for col in custom_metric_columns])

        self._session.sql(
            f"""CREATE OR REPLACE TABLE {fully_qualified_table_name}
            (label FLOAT, prediction FLOAT,
            {s}, id {id_column_type}, timestamp TIMESTAMP_NTZ{segment_columns_def}{custom_metric_columns_def})"""
        ).collect()

        # Needed to create DT against this table
        self._session.sql(f"ALTER TABLE {fully_qualified_table_name} SET CHANGE_TRACKING=TRUE").collect()

        # Build the segment columns part of the INSERT statement
        segment_columns_insert = ""
        segment_values_insert = ""
        if segment_columns:
            segment_columns_insert = ", " + ", ".join(segment_columns)
            segment_values_insert = ", " + ", ".join([f"'{col}_value'" for col in segment_columns])

        # Build the custom metric columns part of the INSERT statement
        custom_metric_columns_insert = ""
        custom_metric_values_insert = ""
        if custom_metric_columns:
            custom_metric_columns_insert = ", " + ", ".join(custom_metric_columns)
            custom_metric_values_insert = ", " + ", ".join(["1.23" for _ in custom_metric_columns])

        self._session.sql(
            f"""INSERT INTO {fully_qualified_table_name}
            (label, prediction, {", ".join(INPUT_FEATURE_COLUMNS_NAMES)}, id, timestamp{segment_columns_insert}
            {custom_metric_columns_insert}) VALUES (1, 1, {", ".join(["1"] * 64)}, '1', CURRENT_TIMESTAMP()
            {segment_values_insert}{custom_metric_values_insert})"""
        ).collect()

    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)
        self._schema_name = "PUBLIC"
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "TEST_MODEL_MONITORING"
        ).upper()
        # Time-travel is required for model monitoring.
        self._db_manager.create_database(self._db_name, data_retention_time_in_days=1)
        self._warehouse_name = "REGTEST_ML_SMALL"
        self._db_manager.set_warehouse(self._warehouse_name)

        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(
            self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
            options={"enable_monitoring": True},
        )

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._db_name)
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def _add_sample_model_version(self, model_name: str, version_name: str) -> model_version_impl.ModelVersion:
        model, features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        return self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=features,
        )

    def _add_sample_monitor(
        self,
        monitor_name: str,
        source: str,
        model_version: model_version_impl.ModelVersion,
        segment_columns=None,
        custom_metric_columns=None,
    ) -> model_monitor.ModelMonitor:
        return self.registry.add_monitor(
            name=monitor_name,
            source_config=model_monitor_config.ModelMonitorSourceConfig(
                source=source,
                prediction_score_columns=["prediction"],
                actual_class_columns=["label"],
                id_columns=["id"],
                timestamp_column="timestamp",
                segment_columns=segment_columns,
                custom_metric_columns=custom_metric_columns,
            ),
            model_monitor_config=model_monitor_config.ModelMonitorConfig(
                model_version=model_version,
                model_function_name="predict",
                background_compute_warehouse_name=self._warehouse_name,
            ),
        )

    def _create_sample_table_model_and_monitor(
        self, monitor_name: str, table_name: str, model_name: str, version_name: str = "V1"
    ):
        self._create_test_table(fully_qualified_table_name=f"{self._db_name}.{self._schema_name}.{table_name}")
        mv = self._add_sample_model_version(model_name=model_name, version_name=version_name)
        self._add_sample_monitor(monitor_name=monitor_name, source=table_name, model_version=mv)

    def test_show_and_get_monitors(self):
        # Test show returns empty initially
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 0)

        # Create monitor
        self._create_sample_table_model_and_monitor(
            monitor_name="monitor", table_name="source_table", model_name="model_name", version_name="V1"
        )

        # Test show returns the monitor with correct name
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "MONITOR")

        # Test get by name
        monitor = self.registry.get_monitor(name="monitor")
        self.assertEqual(monitor.name, "MONITOR")

        # Test get by model_version
        model_version = self.registry.get_model("model_name").version("V1")
        monitor2 = self.registry.get_monitor(model_version=model_version)
        self.assertEqual(monitor2.name, "MONITOR")

        # Test get by name, doesn't exist
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor 'non_existent_monitor'"):
            self.registry.get_monitor(name="non_existent_monitor")

        # Test get by model_version, doesn't exist
        model_version_not_monitored = self._add_sample_model_version(model_name="fake_model_name", version_name="V2")
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor for the given model version."):
            self.registry.get_monitor(model_version=model_version_not_monitored)

    def test_add_monitor_duplicate_fails(self):
        source_table_name = "source_table"
        model_name_original = "model_name"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_name}")
        mv = self._add_sample_model_version(
            model_name=model_name_original,
            version_name="V1",
        )
        self._add_sample_monitor(
            monitor_name="test_monitor_name",
            source="source_table",
            model_version=mv,
        )

        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            self._add_sample_monitor(
                monitor_name="test_monitor_name",
                source="source_table",
                model_version=mv,
            )

        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            self._add_sample_monitor(
                monitor_name="test_monitor_name2",
                source="source_table",
                model_version=mv,
            )

    def test_suspend_resume_monitor(self):
        self._create_sample_table_model_and_monitor(
            monitor_name="monitor", table_name="source_table", model_name="model_name"
        )
        monitor = self.registry.get_monitor(name="monitor")
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])

        monitor.resume()  # resume while already running
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])
        monitor.suspend()  # suspend after running
        self.assertEqual(self.registry.show_model_monitors()[0]["monitor_state"], "SUSPENDED")
        monitor.suspend()  # suspend while already suspended
        self.assertEqual(self.registry.show_model_monitors()[0]["monitor_state"], "SUSPENDED")
        monitor.resume()  # resume after suspending
        self.assertTrue(self.registry.show_model_monitors()[0]["monitor_state"] in ["ACTIVE", "RUNNING"])

    def test_delete_monitor(self) -> None:
        self._create_sample_table_model_and_monitor(
            monitor_name="monitor", table_name="source_table", model_name="model_name"
        )
        self.assertEqual(len(self.registry.show_model_monitors()), 1)
        self.registry.delete_monitor(name="monitor")
        with self.assertRaisesRegex(ValueError, "Unable to find model monitor 'monitor'"):
            self.registry.get_monitor(name="monitor")
        self.assertEqual(len(self.registry.show_model_monitors()), 0)

    def test_create_model_monitor_from_view(self):
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 0)

        source_table_name = "source_table"
        model_name_original = "model_name"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_name}")
        self._session.sql(
            f"""CREATE OR REPLACE VIEW {self._db_name}.{self._schema_name}.{source_table_name}_view
                AS SELECT * FROM {self._db_name}.{self._schema_name}.{source_table_name}"""
        ).collect()

        mv = self._add_sample_model_version(
            model_name=model_name_original,
            version_name="V1",
        )
        self._add_sample_monitor(
            monitor_name="monitor",
            source="source_table_view",
            model_version=mv,
        )

        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "MONITOR")

    def test_add_and_drop_segment_column(self):
        """Test adding and dropping a segment column, including duplicate add validation."""
        # Create table, model, and monitor
        source_table_name = "source_table_segment_drop"
        model_name = "model_segment_drop"
        monitor_name = "monitor_segment_drop"

        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )
        mv = self._add_sample_model_version(model_name=model_name, version_name="V1")
        monitor = self._add_sample_monitor(monitor_name=monitor_name, source=source_table_name, model_version=mv)

        # First add a segment column, then drop it
        monitor.add_segment_column("customer_segment")

        # Validate the segment column was added
        segment_columns = self._get_monitor_segment_columns(monitor_name)
        self.assertIn("CUSTOMER_SEGMENT", segment_columns, f"CUSTOMER_SEGMENT not found after add: {segment_columns}")

        # Test that adding the same segment column again should fail
        with self.assertRaisesRegex(SnowparkSQLException, ".*already exists.*"):
            monitor.add_segment_column("customer_segment")  # Should fail since already present

        # Now drop the segment column
        monitor.drop_segment_column("customer_segment")

        # Validate the segment column was removed
        segment_columns = self._get_monitor_segment_columns(monitor_name)
        self.assertNotIn(
            "CUSTOMER_SEGMENT", segment_columns, f"CUSTOMER_SEGMENT still found after drop: {segment_columns}"
        )

    def test_create_monitor_with_segment_columns_happy_path(self):
        """Test creating a monitor with valid segment_columns."""

        source_table_name = "source_table_with_segments"
        model_name = "model_with_segments"
        monitor_name = "monitor_with_segments"

        # Create table with segment columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )

        # Add additional segment column for testing multiple segments
        self._session.sql(
            f"ALTER TABLE {self._db_name}.{self._schema_name}.{source_table_name} ADD COLUMN region STRING"
        ).collect()
        self._session.sql(
            f"UPDATE {self._db_name}.{self._schema_name}.{source_table_name} SET region = 'us-west'"
        ).collect()

        # Create model version
        mv = self._add_sample_model_version(model_name=model_name, version_name="V1")

        # Create monitor with segment columns - this should succeed
        monitor = self._add_sample_monitor(
            monitor_name=monitor_name,
            source=source_table_name,
            model_version=mv,
            segment_columns=["customer_segment", "region"],
        )

        # Verify monitor was created successfully
        self.assertEqual(monitor.name, monitor_name.upper())

        # Verify it appears in the list of monitors
        monitors = self.registry.show_model_monitors()
        monitor_names = [m["name"] for m in monitors]
        self.assertIn(monitor_name.upper(), monitor_names)

    def test_create_monitor_with_segment_columns_missing_in_source(self):
        """Test creating a monitor with invalid segment_columns should fail."""

        source_table_name = "source_table_invalid_segments"
        model_name = "model_invalid_segments"
        monitor_name = "monitor_invalid_segments"

        # Create table with segment columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", segment_columns=["customer_segment"]
        )

        # Create model version
        mv = self._add_sample_model_version(model_name=model_name, version_name="V1")

        # Try to create monitor with invalid segment columns - this should fail
        with self.assertRaisesRegex(
            ValueError,
            "Segment column\\(s\\): \\['NONEXISTENT_COLUMN', 'ANOTHER_INVALID_COLUMN'\\] do not exist in source\\.",
        ):
            self._add_sample_monitor(
                monitor_name=monitor_name,
                source=source_table_name,
                model_version=mv,
                segment_columns=["nonexistent_column", "another_invalid_column"],
            )

        # Verify monitor was NOT created
        monitors = self.registry.show_model_monitors()
        monitor_names = [m["name"] for m in monitors]
        self.assertNotIn(monitor_name.upper(), monitor_names)

    def test_add_drop_custom_metric_columns(self):
        """Test adding and dropping custom metric columns."""

        self._session.sql("ALTER SESSION SET ENABLE_MODEL_MONITOR_CUSTOM_METRICS = TRUE").collect()
        source_table_name = "source_table_custom_metrics"
        model_name = "model_custom_metrics"
        monitor_name = "monitor_custom_metrics"

        # Create table with initial custom metric columns
        self._create_test_table(
            f"{self._db_name}.{self._schema_name}.{source_table_name}", custom_metric_columns=["initial_metric"]
        )

        # Create model version
        mv = self._add_sample_model_version(model_name=model_name, version_name="V1")

        # Create monitor with initial custom metric columns
        monitor = self._add_sample_monitor(
            monitor_name=monitor_name,
            source=source_table_name,
            model_version=mv,
            custom_metric_columns=["initial_metric"],
        )

        # Verify monitor was created successfully
        self.assertEqual(monitor.name, monitor_name.upper())

        # Verify monitor was created successfully with initial custom metric
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertIn("INITIAL_METRIC", describe_result[0]["columns"])

        # Add new custom metric columns
        monitor.add_custom_metric_column("input_feature_1")

        # Verify new custom metric columns were added
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertIn("INPUT_FEATURE_1", json.loads(describe_result[0]["columns"])["custom_metric_columns"])

        # Drop a custom metric column
        monitor.drop_custom_metric_column("initial_metric")

        # Verify the custom metric column was dropped
        describe_result = self._session.sql(
            f"DESCRIBE MODEL MONITOR {self._db_name}.{self._schema_name}.{monitor_name}"
        ).collect()
        self.assertNotIn("INITIAL_METRIC", json.loads(describe_result[0]["columns"])["custom_metric_columns"])
        self.assertIn("INITIAL_METRIC", json.loads(describe_result[0]["columns"])["numerical_columns"])


if __name__ == "__main__":
    absltest.main()
