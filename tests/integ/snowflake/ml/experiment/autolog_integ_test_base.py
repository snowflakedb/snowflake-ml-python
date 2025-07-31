import json
import uuid
from typing import Any

import pandas as pd

from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.experiment import experiment_tracking
from snowflake.ml.model import model_signature
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager

AUTOLOGGING_MIN_VERSION = (9, 20, 0)


class AutologIntegrationTest:
    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        if snowflake_env.get_current_snowflake_version(self._session).release < AUTOLOGGING_MIN_VERSION:
            self.skipTest(f"Skipping test because current Snowflake version is less than {AUTOLOGGING_MIN_VERSION}.")

        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)
        self._schema_name = "PUBLIC"
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "TEST_EXPERIMENT_TRACKING"
        ).upper()
        self._db_manager.create_database(self._db_name, data_retention_time_in_days=1)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.exp = experiment_tracking.ExperimentTracking(
            self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
        )
        self.X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.y = [0, 1, 0]
        self.num_steps = 5

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._db_name)
        super().tearDown()

    def _train_model(self, model_class: type[Any], callback: Any) -> None:
        pass

    def _test_autolog(self, model_class: type[Any], callback_class: type[Any], metric_name: str) -> None:
        """Test that autologging works."""
        experiment_name = "TEST_EXPERIMENT_AUTOLOG"
        model_name = "TEST_AUTOLOG_MODEL"

        model_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(name="a", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="b", dtype=model_signature.DataType.FLOAT),
            ],
            outputs=[model_signature.FeatureSpec(name="target", dtype=model_signature.DataType.INT8)],
        )

        callback = callback_class(
            self.exp,
            log_model=True,
            log_metrics=True,
            log_params=True,
            model_name=model_name,
            model_signature=model_sig,
        )
        self.exp.set_experiment(experiment_name=experiment_name)
        self._train_model(model_class, callback)

        # Verify all data was logged correctly
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 1)

        # Parse and verify metadata
        metadata = json.loads(runs[0]["metadata"])
        self.assertIn("metrics", metadata)
        metric_dict = {f"{m['name']}_step_{m['step']}": m for m in metadata["metrics"]}
        # Verify that the expected metric was logged for each step
        for step in range(self.num_steps):
            self.assertIn(f"{metric_name}_step_{step}", metric_dict)
        # Verify that params were logged
        self.assertIn("parameters", metadata)
        self.assertGreater(len(metadata["parameters"]), 0)
        # Verify that the model was logged
        models = self._session.sql(f"SHOW MODELS LIKE '{model_name}'").collect()
        self.assertEqual(len(models), 1)
