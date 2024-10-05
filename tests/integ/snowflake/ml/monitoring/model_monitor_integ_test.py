import uuid

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring._client import model_monitor, monitor_sql_client
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
        self.registry = registry.Registry(self._session, database_name=self._db_name, schema_name=self._schema_name)

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

    def test_add_model_monitor(self) -> None:
        # Create an instance of the Registry class with Monitoring enabled.
        _monitor_registry = registry.Registry(
            session=self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
            options={"enable_monitoring": True},
        )

        source_table_name = "TEST_TABLE"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_name}")

        model_name = "TEST_MODEL"
        version_name = "TEST_VERSION"
        monitor_name = f"TEST_MONITOR_{model_name}_{version_name}_{self.run_id}"
        monitor = self._add_sample_model_version_and_monitor(
            _monitor_registry, source_table_name, model_name, version_name, monitor_name
        )

        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                FROM {self._db_name}.{self._schema_name}.{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE FULLY_QUALIFIED_MODEL_NAME = '{self._db_name}.{self._schema_name}.{model_name}' AND
                MODEL_VERSION_NAME = '{version_name}'"""
            ).count(),
            1,
        )

        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                        FROM {self._db_name}.{self._schema_name}.
                        _SNOWML_OBS_BASELINE_{model_name}_{version_name}"""
            ).count(),
            0,
        )

        table_columns = self._session.sql(
            f"""DESCRIBE TABLE
            {self._db_name}.{self._schema_name}._SNOWML_OBS_BASELINE_{model_name}_{version_name}"""
        ).collect()

        for col in table_columns:
            self.assertTrue(
                col["name"].upper()
                in ["PREDICTION", "LABEL", "ID", "TIMESTAMP", *[i.upper() for i in INPUT_FEATURE_COLUMNS_NAMES]]
            )

        df = self._session.create_dataframe(
            [
                (1.0, 1.0, *[1.0] * 64),
                (1.0, 1.0, *[1.0] * 64),
            ],
            ["LABEL", "PREDICTION", *[i.upper() for i in INPUT_FEATURE_COLUMNS_NAMES]],
        )
        monitor.set_baseline(df)
        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                        FROM {self._db_name}.{self._schema_name}.
                        _SNOWML_OBS_BASELINE_{model_name}_{version_name}"""
            ).count(),
            2,
        )

        pandas_cols = {
            "LABEL": [1.0, 2.0, 3.0],
            "PREDICTION": [1.0, 2.0, 3.0],
        }
        for i in range(64):
            pandas_cols[f"INPUT_FEATURE_{i}"] = [1.0, 2.0, 3.0]

        pandas_df = pd.DataFrame(pandas_cols)
        monitor.set_baseline(pandas_df)
        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                        FROM {self._db_name}.{self._schema_name}.
                        _SNOWML_OBS_BASELINE_{model_name}_{version_name}"""
            ).count(),
            3,
        )

        # create a snowpark dataframe that does not conform to the existing schema
        df = self._session.create_dataframe(
            [
                (1.0, "bad", *[1.0] * 64),
                (2.0, "very_bad", *[2.0] * 64),
            ],
            ["LABEL", "PREDICTION", *[i.upper() for i in INPUT_FEATURE_COLUMNS_NAMES]],
        )
        with self.assertRaises(ValueError) as e:
            monitor.set_baseline(df)

        expected_msg = "Ensure that the baseline dataframe columns match those provided in your monitored table"
        self.assertTrue(expected_msg in str(e.exception))
        expected_msg = "Numeric value 'bad' is not recognized"
        self.assertTrue(expected_msg in str(e.exception))

        # Delete monitor
        _monitor_registry.delete_monitor(monitor_name)

        # Validate that metadata entry is deleted
        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                FROM {self._db_name}.{self._schema_name}.{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE MONITOR_NAME = '{monitor.name}'"""
            ).count(),
            0,
        )

        # Validate that baseline table is deleted
        self.assertEqual(
            self._session.sql(
                f"""SHOW TABLES LIKE '%{self._db_name}.{self._schema_name}.
                    _SNOWML_OBS_BASELINE_{model_name}_{version_name}%'"""
            ).count(),
            0,
        )

        # Validate that dynamic tables are deleted
        self.assertEqual(
            self._session.sql(
                f"""SHOW TABLES LIKE '%{self._db_name}.{self._schema_name}.
                    _SNOWML_OBS_MONITORING_{model_name}_{version_name}%'"""
            ).count(),
            0,
        )
        self.assertEqual(
            self._session.sql(
                f"""SHOW TABLES LIKE '%{self._db_name}.{self._schema_name}.
                    _SNOWML_OBS_ACCURACY_{model_name}_{version_name}%'"""
            ).count(),
            0,
        )

    def test_add_model_monitor_varchar(self) -> None:
        _monitor_registry = registry.Registry(
            session=self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
            options={"enable_monitoring": True},
        )
        source_table = "TEST_TABLE"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table}", id_column_type="VARCHAR(64)")

        model_name = "TEST_MODEL"
        version_name = "TEST_VERSION"
        monitor_name = f"TEST_MONITOR_{model_name}_{version_name}_{self.run_id}"
        self._add_sample_model_version_and_monitor(
            _monitor_registry, source_table, model_name, version_name, monitor_name
        )

        self.assertEqual(
            self._session.sql(
                f"""SELECT *
                FROM {self._db_name}.{self._schema_name}.{monitor_sql_client.SNOWML_MONITORING_METADATA_TABLE_NAME}
                WHERE FULLY_QUALIFIED_MODEL_NAME = '{self._db_name}.{self._schema_name}.{model_name}' AND
                MODEL_VERSION_NAME = '{version_name}'"""
            ).count(),
            1,
        )

    def test_show_model_monitors(self) -> None:
        _monitor_registry = registry.Registry(
            session=self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
            options={"enable_monitoring": True},
        )
        source_table_1 = "TEST_TABLE_1"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_1}")

        source_table_2 = "TEST_TABLE_2"
        self._create_test_table(f"{self._db_name}.{self._schema_name}.{source_table_2}")

        model_1 = "TEST_MODEL_1"
        version_1 = "TEST_VERSION_1"
        monitor_1 = f"TEST_MONITOR_{model_1}_{version_1}_{self.run_id}"
        self._add_sample_model_version_and_monitor(_monitor_registry, source_table_1, model_1, version_1, monitor_1)

        model_2 = "TEST_MODEL_2"
        version_2 = "TEST_VERSION_2"
        monitor_2 = f"TEST_MONITOR_{model_2}_{version_2}_{self.run_id}"
        self._add_sample_model_version_and_monitor(_monitor_registry, source_table_2, model_2, version_2, monitor_2)

        stored_monitors = sorted(_monitor_registry.show_model_monitors(), key=lambda x: x["MONITOR_NAME"])
        self.assertEqual(len(stored_monitors), 2)
        row_1 = stored_monitors[0]
        self.assertEqual(row_1["MONITOR_NAME"], sql_identifier.SqlIdentifier(monitor_1))
        self.assertEqual(row_1["SOURCE_TABLE_NAME"], source_table_1)
        self.assertEqual(row_1["MODEL_VERSION_NAME"], version_1)
        self.assertEqual(row_1["IS_ENABLED"], True)
        row_2 = stored_monitors[1]
        self.assertEqual(row_2["MONITOR_NAME"], sql_identifier.SqlIdentifier(monitor_2))
        self.assertEqual(row_2["SOURCE_TABLE_NAME"], source_table_2)
        self.assertEqual(row_2["MODEL_VERSION_NAME"], version_2)
        self.assertEqual(row_2["IS_ENABLED"], True)


if __name__ == "__main__":
    absltest.main()
