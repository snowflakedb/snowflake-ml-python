import json
import uuid

import pandas as pd
from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml import registry
from snowflake.ml._internal.utils import connection_params
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.experiment._entities import run as run_module
from snowflake.ml.model import ModelVersion
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class ExperimentLineageIntegrationTest(absltest.TestCase):
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
            self.run_id, "TEST_EXPERIMENT_LINEAGE"
        ).upper()
        self._db_manager.create_database(self._db_name, data_retention_time_in_days=1)
        self._db_manager.cleanup_databases(expire_hours=6)
        ExperimentTracking._instance = None  # Reset singleton for test
        self.exp = ExperimentTracking(
            self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
        )

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._db_name)
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def test_experiment_model_lineage(self) -> None:
        """Test that log_model persists lineage information from experiment to model version"""
        experiment_name = "LINEAGE_TEST_EXPERIMENT"
        run_name = "LINEAGE_TEST_RUN"
        model_name = "LINEAGE_TEST_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            model = LinearRegression()
            model.fit(X, y)
            mv = self.exp.log_model(
                model,
                model_name=model_name,
                sample_input_data=X,
            )

        # Test that the model version can be run and the output is correct
        actual = mv.run(X, function_name="predict")
        expected = pd.DataFrame({"output_feature_0": model.predict(X)})
        pd.testing.assert_frame_equal(actual, expected)

        # Test that lineage edge is correctly created
        experiment_fqn = self.exp._sql_client.fully_qualified_object_name(
            self.exp._database_name, self.exp._schema_name, self.exp._experiment.name
        )
        dgql_json = json.loads(
            self._session.sql(
                f"""select SYSTEM$DGQL('
                    {{
                        V(domain: EXPERIMENT, name:"{run_name}", parentName:"{experiment_fqn}")
                        {{
                            E(edgeType:[DATA_LINEAGE],direction:OUT)
                            {{
                                S {{domain, name, schema, db, properties}},
                                T {{domain, name, schema, db, properties}}
                            }}
                        }}
                    }}
            ')"""
            ).collect()[0][0]
        )
        lineage_edges = dgql_json.get("data", {}).get("V", {}).get("E", [])
        self.assertEqual(len(lineage_edges), 1, f"Expected 1 lineage edge, got {len(lineage_edges)}")
        # Confirm source is correct
        source = lineage_edges[0]["S"]
        self.assertEqual(source["domain"], "EXPERIMENT")
        self.assertEqual(source["name"], run_name)
        self.assertEqual(source["properties"]["parentName"], experiment_name)
        self.assertEqual(source["schema"], self._schema_name)
        self.assertEqual(source["db"], self._db_name)
        # Confirm target is correct
        target = lineage_edges[0]["T"]
        self.assertEqual(target["domain"], "MODEL")
        self.assertEqual(target["name"], mv.version_name)
        self.assertEqual(target["properties"]["parentName"], model_name)
        self.assertEqual(target["schema"], self._schema_name)
        self.assertEqual(target["db"], self._db_name)

    def test_list_model_versions(self) -> None:
        """Test that list_model_versions retrieves model versions logged under a run via lineage."""
        experiment_name = "LIST_MODEL_VERSIONS_EXPERIMENT"
        run_name = "LIST_MODEL_VERSIONS_RUN"
        model_name = "LIST_MODEL_VERSIONS_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            model = LinearRegression()
            model.fit(X, y)
            mv = self.exp.log_model(
                model,
                model_name=model_name,
                sample_input_data=X,
            )

        # Retrieve logged model versions by explicit run name
        logged_models = self.exp.list_model_versions(run_name=run_name)
        self.assertEqual(len(logged_models), 1, f"Expected 1 logged model, got {len(logged_models)}")
        logged_model = logged_models[0]
        self.assertIsInstance(logged_model, ModelVersion)
        self.assertEqual(logged_model.model_name, mv.model_name)
        self.assertEqual(logged_model.version_name, mv.version_name)

        # The retrieved model version should be runnable and produce the same output
        actual = logged_model.run(X, function_name="predict")
        expected = pd.DataFrame({"output_feature_0": model.predict(X)})
        pd.testing.assert_frame_equal(actual, expected)

    def test_list_model_versions_empty(self) -> None:
        """Test that list_model_versions returns an empty list for a run with no logged models."""
        experiment_name = "EMPTY_MODELS_EXPERIMENT"
        run_name = "EMPTY_MODELS_RUN"

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.exp.log_metric("accuracy", 0.9)

        logged_models = self.exp.list_model_versions(run_name=run_name)
        self.assertEqual(len(logged_models), 0, f"Expected 0 logged models, got {len(logged_models)}")

    def test_upstream_experiment_run(self) -> None:
        """Test that mv.lineage(direction='upstream') returns an ExperimentRun for the logging run."""
        experiment_name = "GET_RUN_EXPERIMENT"
        run_name = "GET_RUN_RUN"
        model_name = "GET_RUN_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            model = LinearRegression()
            model.fit(X, y)
            mv = self.exp.log_model(
                model,
                model_name=model_name,
                sample_input_data=X,
            )

        runs = mv.lineage(direction="upstream", domain_filter={"experiment"})
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertIsInstance(run, run_module.Run)
        assert isinstance(run, run_module.Run)
        self.assertEqual(run.database_name, self._db_name)
        self.assertEqual(run.schema_name, self._schema_name)
        self.assertEqual(run.experiment_name, experiment_name)
        self.assertEqual(run.name, run_name)

        # The returned names should lead back to the same model version
        self.exp.set_experiment(
            experiment_name=run.experiment_name,
            database_name=run.database_name,
            schema_name=run.schema_name,
        )
        logged_models = self.exp.list_model_versions(run_name=run.name)
        self.assertEqual([(m.model_name, m.version_name) for m in logged_models], [(mv.model_name, mv.version_name)])

    def test_upstream_experiment_run_duplicate_run_names(self) -> None:
        """Test that lineage reports the correct experiment when two experiments share a run name."""
        run_name = "SHARED_RUN"
        model_name = "SHARED_RUN_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        self.exp.set_experiment(experiment_name="DECOY_EXPERIMENT")
        with self.exp.start_run(run_name=run_name):
            self.exp.log_metric("accuracy", 0.9)

        self.exp.set_experiment(experiment_name="LOGGING_EXPERIMENT")
        with self.exp.start_run(run_name=run_name):
            model = LinearRegression()
            model.fit(X, y)
            mv = self.exp.log_model(
                model,
                model_name=model_name,
                sample_input_data=X,
            )

        runs = mv.lineage(direction="upstream", domain_filter={"experiment"})
        self.assertEqual(len(runs), 1)
        run = runs[0]
        assert isinstance(run, run_module.Run)
        self.assertEqual(run.experiment_name, "LOGGING_EXPERIMENT")
        self.assertEqual(run.name, run_name)

    def test_upstream_experiment_run_logged_outside_run(self) -> None:
        """Test that lineage returns no experiment runs for a model version not logged under a run."""
        model_name = "NO_RUN_MODEL"

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = [0, 1, 0]

        reg = registry.Registry(self._session, database_name=self._db_name, schema_name=self._schema_name)
        model = LinearRegression()
        model.fit(X, y)
        mv = reg.log_model(model, model_name=model_name, sample_input_data=X)

        runs = mv.lineage(direction="upstream", domain_filter={"experiment"})
        self.assertEqual(len(runs), 0)


if __name__ == "__main__":
    absltest.main()
