import uuid

from absl.testing import absltest, parameterized

from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
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

    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)
        self._schema_name = "PUBLIC"
        # TODO(jfishbein): Investigate whether conversion to sql identifier requires uppercase.
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "monitor_registry"
        ).upper()
        self._session.sql(f"CREATE DATABASE IF NOT EXISTS {self._db_name}").collect()
        self.perm_stage = "@" + self._db_manager.create_stage(
            stage_name="model_registry_test_stage",
            schema_name=self._schema_name,
            db_name=self._db_name,
            sse_encrypted=True,
        )
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

    def _add_sample_model_version_and_monitor(
        self,
        monitor_registry: registry.Registry,
        source_table: str,
        model_name: str,
        version_name: str,
        monitor_name: str,
    ) -> model_monitor.ModelMonitor:
        model, features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        model_version: model_version_impl.ModelVersion = self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=features,
        )

        return monitor_registry.add_monitor(
            name=monitor_name,
            table_config=model_monitor_config.ModelMonitorTableConfig(
                source_table=source_table,
                prediction_columns=["prediction"],
                label_columns=["label"],
                id_columns=["id"],
                timestamp_column="timestamp",
            ),
            model_monitor_config=model_monitor_config.ModelMonitorConfig(
                model_version=model_version,
                model_function_name="predict",
                background_compute_warehouse_name=self._warehouse_name,
            ),
        )

    def test_model_monitor_unimplemented_in_registry(self):
        with self.assertRaisesRegex(NotImplementedError, registry._MODEL_MONITORING_UNIMPLEMENTED_ERROR):
            self.registry.show_model_monitors()


if __name__ == "__main__":
    absltest.main()
