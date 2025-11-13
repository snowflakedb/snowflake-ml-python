import json
import uuid

from absl.testing import parameterized

from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager, model_factory

NUM_FEATURES = 8
INPUT_FEATURE_COLUMNS_NAMES = [f"input_feature_{i}" for i in range(NUM_FEATURES)]


class ModelMonitorIntegrationTestBase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        cls._run_id = uuid.uuid4().hex
        cls._db_manager = db_manager.DBManager(cls._session)
        cls._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            cls._run_id, "TEST_MODEL_MONITORING"
        ).upper()
        cls._schema_name = "PUBLIC"
        cls._model_name = "TEST_MODEL"

        # Time-travel is required for model monitoring.
        cls._db_manager.create_database(cls._db_name, data_retention_time_in_days=1)
        cls._warehouse_name = "REGTEST_ML_SMALL"
        cls._db_manager.set_warehouse(cls._warehouse_name)

        cls._db_manager.cleanup_databases(expire_hours=6)

        cls.registry = registry.Registry(
            cls._session,
            database_name=cls._db_name,
            schema_name=cls._schema_name,
            options={"enable_monitoring": True},
        )

        cls._model, cls._features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        cls._model_version = cls.registry.log_model(
            model=cls._model,
            model_name=cls._model_name,
            version_name="V1",
            sample_input_data=cls._features,
        )

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

    def tearDown(self) -> None:
        # Clean up in dependency order: monitors -> tables
        self._drop_all_monitors()
        self._drop_all_tables()
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_database(cls._db_name)
        cls._session.close()

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
            {custom_metric_columns_insert}) VALUES (1, 1, {", ".join(["1"] * NUM_FEATURES)}, '1', CURRENT_TIMESTAMP()
            {segment_values_insert}{custom_metric_values_insert})"""
        ).collect()

    def _drop_all_tables(self) -> None:
        """Drop all tables in the current schema."""
        # Get all table names in the schema
        tables = self._session.sql(f"SHOW TABLES IN SCHEMA {self._db_name}.{self._schema_name}").collect()

        # Drop each table
        for table in tables:
            table_name = table["name"]
            self._session.sql(f"DROP TABLE IF EXISTS {self._db_name}.{self._schema_name}.{table_name}").collect()

    def _drop_all_monitors(self) -> None:
        """Drop all monitors in the current schema."""
        monitors = self.registry.show_model_monitors()
        for monitor in monitors:
            self.registry.delete_monitor(name=monitor["name"])

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

    def _create_sample_table_and_monitor(self, monitor_name: str, table_name: str):
        self._create_test_table(fully_qualified_table_name=f"{self._db_name}.{self._schema_name}.{table_name}")
        self._add_sample_monitor(monitor_name=monitor_name, source=table_name, model_version=self._model_version)
