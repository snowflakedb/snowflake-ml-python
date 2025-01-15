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
    def _create_test_table(self, fully_qualified_table_name: str, id_column_type: str = "STRING") -> None:
        s = ", ".join([f"{i} FLOAT" for i in INPUT_FEATURE_COLUMNS_NAMES])
        self._session.sql(
            f"""CREATE OR REPLACE TABLE {fully_qualified_table_name}
            (label FLOAT, prediction FLOAT,
            {s}, id {id_column_type}, timestamp TIMESTAMP)"""
        ).collect()

        # Needed to create DT against this table
        self._session.sql(f"ALTER TABLE {fully_qualified_table_name} SET CHANGE_TRACKING=TRUE").collect()

        self._session.sql(
            f"""INSERT INTO {fully_qualified_table_name}
            (label, prediction, {", ".join(INPUT_FEATURE_COLUMNS_NAMES)}, id, timestamp)
            VALUES (1, 1, {", ".join(["1"] * 64)}, '1', CURRENT_TIMESTAMP())"""
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
        self, monitor_name: str, source: str, model_version: model_version_impl.ModelVersion
    ) -> model_monitor.ModelMonitor:
        return self.registry.add_monitor(
            name=monitor_name,
            source_config=model_monitor_config.ModelMonitorSourceConfig(
                source=source,
                prediction_score_columns=["prediction"],
                actual_class_columns=["label"],
                id_columns=["id"],
                timestamp_column="timestamp",
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

    def test_show_model_monitors(self):
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 0)
        self._create_sample_table_model_and_monitor(
            monitor_name="monitor", table_name="source_table", model_name="model_name"
        )
        res = self.registry.show_model_monitors()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["name"], "MONITOR")

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

    def test_get_monitor(self):
        self._create_sample_table_model_and_monitor(
            monitor_name="monitor", table_name="source_table", model_name="model_name", version_name="V1"
        )
        # Test get by name.
        monitor = self.registry.get_monitor(name="monitor")
        self.assertEqual(monitor.name, "MONITOR")

        # Test get by model_version.
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


if __name__ == "__main__":
    absltest.main()
