import uuid

import pandas as pd
from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml._internal.utils import connection_params
from snowflake.ml.experiment import ExperimentTracking, _logging as experiment_logging
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class ExperimentModelRegistryIntegTest(absltest.TestCase):
    """Integration tests for the model_registry parameter on ExperimentTracking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        experiment_logging.ExperimentLogger.OUTPUT_DIRECTORY = "/tmp/experiment_tracking"

    def setUp(self) -> None:
        self.test_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)

        self._experiment_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.test_id, "EXPERIMENT_DB"
        ).upper()
        self._model_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.test_id, "MODEL_DB"
        ).upper()

        self._db_manager.create_database(self._experiment_db, data_retention_time_in_days=1)
        self._db_manager.create_database(self._model_db, data_retention_time_in_days=1)
        self._db_manager.cleanup_databases(expire_hours=6)

        ExperimentTracking._instance = None

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._experiment_db)
        self._db_manager.drop_database(self._model_db)
        ExperimentTracking._instance = None
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def test_log_model_to_separate_registry(self) -> None:
        """Model is logged to model_registry's db/schema, not the experiment's."""
        model_registry = Registry(
            self._session,
            database_name=self._model_db,
            schema_name="PUBLIC",
        )

        exp = ExperimentTracking(
            self._session,
            database_name=self._experiment_db,
            schema_name="PUBLIC",
            model_registry=model_registry,
        )
        exp.set_experiment("TEST_EXPERIMENT")

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        with exp.start_run(run_name="TEST_RUN"):
            model = LinearRegression()
            model.fit(X, y)
            mv = exp.log_model(
                model,
                model_name="TEST_MODEL",
                sample_input_data=X,
                target_platforms=["WAREHOUSE"],
            )

        # Model should be in the model db, not the experiment db
        models_in_model_db = self._session.sql(f"SHOW MODELS IN DATABASE {self._model_db}").collect()
        self.assertEqual(len(models_in_model_db), 1)
        self.assertEqual(models_in_model_db[0]["name"], "TEST_MODEL")
        self.assertEqual(models_in_model_db[0]["database_name"], self._model_db)

        models_in_experiment_db = self._session.sql(f"SHOW MODELS IN DATABASE {self._experiment_db}").collect()
        self.assertEqual(len(models_in_experiment_db), 0)

        # Model should still work
        actual = mv.run(X, function_name="predict")
        expected = pd.DataFrame({"output_feature_0": model.predict(X)})
        pd.testing.assert_frame_equal(actual, expected)

    def test_log_model_default_registry_when_none(self) -> None:
        """Without model_registry, model lands in the experiment's db/schema."""
        exp = ExperimentTracking(
            self._session,
            database_name=self._experiment_db,
            schema_name="PUBLIC",
        )
        exp.set_experiment("TEST_EXPERIMENT")

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        with exp.start_run(run_name="TEST_RUN"):
            model = LinearRegression()
            model.fit(X, y)
            exp.log_model(
                model,
                model_name="TEST_MODEL",
                sample_input_data=X,
                target_platforms=["WAREHOUSE"],
            )

        models = self._session.sql(f"SHOW MODELS IN DATABASE {self._experiment_db}").collect()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["database_name"], self._experiment_db)

    def test_model_registry_survives_set_experiment_db_change(self) -> None:
        """Changing the experiment db/schema via set_experiment does not affect model_registry."""
        model_registry = Registry(
            self._session,
            database_name=self._model_db,
            schema_name="PUBLIC",
        )

        exp = ExperimentTracking(
            self._session,
            database_name=self._experiment_db,
            schema_name="PUBLIC",
            model_registry=model_registry,
        )

        # Switch experiment to a different schema (still within experiment_db)
        self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._experiment_db}.OTHER_SCHEMA").collect()
        exp.set_experiment("TEST_EXPERIMENT_2", database_name=self._experiment_db, schema_name="OTHER_SCHEMA")

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        with exp.start_run(run_name="TEST_RUN"):
            model = LinearRegression()
            model.fit(X, y)
            exp.log_model(
                model,
                model_name="TEST_MODEL",
                sample_input_data=X,
                target_platforms=["WAREHOUSE"],
            )

        # Model should still be in the model db
        models = self._session.sql(f"SHOW MODELS IN DATABASE {self._model_db}").collect()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["database_name"], self._model_db)


if __name__ == "__main__":
    absltest.main()
