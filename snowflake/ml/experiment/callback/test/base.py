from typing import Any, Optional
from unittest.mock import ANY, MagicMock

import numpy as np
from absl.testing import parameterized

from snowflake.ml.experiment import experiment_tracking
from snowflake.ml.model.model_signature import DataType, FeatureSpec, ModelSignature


class SnowflakeCallbackTest(parameterized.TestCase):
    """Base class for Snowflake callback tests."""

    def setUp(self) -> None:
        self.experiment_tracking = MagicMock(spec=experiment_tracking.ExperimentTracking)

        # Create training data and parameters
        self.X = np.array([[1, 2], [3, 4]])
        self.y = np.array([0, 1])
        self.num_steps = 2
        self.model_signature = ModelSignature(
            inputs=[
                FeatureSpec(name="feature1", dtype=DataType.FLOAT),
                FeatureSpec(name="feature2", dtype=DataType.FLOAT),
            ],
            outputs=[FeatureSpec(name="target", dtype=DataType.INT8)],
        )

    def _train_model(self, model_class: type[Any], callback: Any) -> None:
        pass

    def _get_callback(self, **kwargs: Any) -> Any:
        pass

    def _log_metrics(self, model_class: type[Any]) -> None:
        """Test that metrics are autologged."""
        callback = self._get_callback(
            experiment_tracking=self.experiment_tracking,
            log_model=False,
            log_metrics=True,
            log_params=False,
        )
        self._train_model(model_class=model_class, callback=callback)

        self.assertEqual(self.experiment_tracking.log_metric.call_count, self.num_steps)

    def _log_model(self, model_class: type[Any], model_name: Optional[str]) -> None:
        """Test that model is autologged."""

        callback = self._get_callback(
            experiment_tracking=self.experiment_tracking,
            log_model=True,
            log_metrics=False,
            log_params=False,
            model_name=model_name,
            model_signature=self.model_signature,
        )

        # Set up mock experiment before training the model
        mock_experiment = MagicMock()
        mock_experiment.name = "test_experiment"
        self.experiment_tracking._get_or_set_experiment.return_value = mock_experiment
        self._train_model(model_class=model_class, callback=callback)

        expected_model_name = model_name or mock_experiment.name + "_model"
        self.experiment_tracking.log_model.assert_called_once_with(
            model=ANY, model_name=expected_model_name, signatures={"predict": self.model_signature}
        )

    def _log_param(self, model_class: type[Any]) -> None:
        """Test that params are autologged."""
        callback = self._get_callback(
            experiment_tracking=self.experiment_tracking,
            log_model=False,
            log_metrics=False,
            log_params=True,
        )
        self._train_model(model_class=model_class, callback=callback)

        self.experiment_tracking.log_params.assert_called_once()
