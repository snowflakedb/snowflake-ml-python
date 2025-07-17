from typing import Optional
from unittest.mock import ANY, MagicMock

from absl.testing import absltest, parameterized
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from snowflake.ml.experiment.callback import SnowflakeXgboostCallback
from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
from snowflake.ml.model.model_signature import DataType, FeatureSpec, ModelSignature


class SnowflakeCallbackTest(parameterized.TestCase):
    """Tests for Snowflake callback classes."""

    def setUp(self) -> None:
        self.experiment_tracking = MagicMock(spec=ExperimentTracking)

        # Create training data and parameters
        self.X = [[1, 2], [3, 4]]
        self.y = [0, 1]
        self.num_steps = 2

    @parameterized.parameters(XGBClassifier, XGBRegressor)  # type: ignore[misc]
    def test_xgboost_log_metrics(self, model_class: type[XGBModel]) -> None:
        """Test that metrics are autologged."""
        callback = SnowflakeXgboostCallback(
            experiment_tracking=self.experiment_tracking, log_model=False, log_metrics=True, log_params=False
        )
        model = model_class(callbacks=[callback], n_estimators=self.num_steps)
        model.fit(self.X, self.y, eval_set=[(self.X, self.y)])

        self.assertEqual(self.experiment_tracking.log_metric.call_count, self.num_steps)

    @parameterized.parameters(None, "custom_model_name")  # type: ignore[misc]
    def test_xgboost_log_model(self, model_name: Optional[str]) -> None:
        """Test that model is autologged."""
        model_signature = ModelSignature(
            inputs=[
                FeatureSpec(name="feature1", dtype=DataType.FLOAT),
                FeatureSpec(name="feature2", dtype=DataType.FLOAT),
            ],
            outputs=[FeatureSpec(name="target", dtype=DataType.INT8)],
        )
        callback = SnowflakeXgboostCallback(
            experiment_tracking=self.experiment_tracking,
            log_model=True,
            log_metrics=False,
            log_params=False,
            model_name=model_name,
            model_signature=model_signature,
        )

        # Set up mock experiment before calling model.fit
        mock_experiment = MagicMock()
        mock_experiment.name = "test_experiment"
        self.experiment_tracking._get_or_set_experiment.return_value = mock_experiment
        model = XGBClassifier(callbacks=[callback], n_estimators=self.num_steps)
        model.fit(self.X, self.y)

        expected_model_name = model_name or mock_experiment.name + "_model"
        self.experiment_tracking.log_model.assert_called_once_with(
            model=ANY, model_name=expected_model_name, signatures={"predict": model_signature}
        )

    @parameterized.parameters(XGBClassifier, XGBRegressor)  # type: ignore[misc]
    def test_xgboost_log_param(self, model_class: type[XGBModel]) -> None:
        """Test that params are autologged."""
        callback = SnowflakeXgboostCallback(
            experiment_tracking=self.experiment_tracking, log_model=False, log_metrics=False, log_params=True
        )
        model = model_class(callbacks=[callback], n_estimators=self.num_steps)
        model.fit(self.X, self.y)

        self.experiment_tracking.log_params.assert_called_once()


if __name__ == "__main__":
    absltest.main()
