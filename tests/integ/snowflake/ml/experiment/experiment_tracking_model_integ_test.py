import pandas as pd
import xgboost as xgb
from absl.testing import absltest

from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentModelIntegTest(ExperimentTrackingIntegTestBase):
    def test_log_model(self) -> None:
        """Test that log_model works with experiment tracking"""
        experiment_name = "TEST_EXPERIMENT_LOG_MODEL"
        run_name = "TEST_RUN_LOG_MODEL"
        model_name = "TEST_MODEL"

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
                target_platforms=["WAREHOUSE"],
            )

        # Test that model exists
        models = self._session.sql(f"SHOW MODELS IN DATABASE {self._db_name}").collect()
        self.assertEqual(len(models), 1)
        self.assertEqual(model_name, models[0]["name"])
        self.assertEqual(self._schema_name, models[0]["schema_name"])
        self.assertEqual(self._db_name, models[0]["database_name"])
        self.assertIn(mv.version_name, models[0]["versions"])

        # Test that the model version can be run and the output is correct
        actual = mv.run(X, function_name="predict")
        expected = pd.DataFrame({"output_feature_0": model.predict(X)})
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    absltest.main()
