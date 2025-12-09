import json
import uuid

import pandas as pd
import xgboost as xgb
from absl.testing import absltest

from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.utils import connection_params
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
            model = xgb.XGBClassifier()
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
        self.assertEqual(target["domain"], "MODULE")
        self.assertEqual(target["name"], mv.version_name)
        self.assertEqual(target["properties"]["parentName"], model_name)
        self.assertEqual(target["schema"], self._schema_name)
        self.assertEqual(target["db"], self._db_name)


if __name__ == "__main__":
    absltest.main()
