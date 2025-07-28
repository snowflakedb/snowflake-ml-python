from typing import Any, Optional, Union
from unittest.mock import ANY, MagicMock

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from absl.testing import absltest, parameterized
from lightgbm import LGBMClassifier, LGBMModel, LGBMRegressor
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from snowflake.ml.experiment.callback import (
    SnowflakeLightgbmCallback,
    SnowflakeXgboostCallback,
)
from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
from snowflake.ml.model.model_signature import DataType, FeatureSpec, ModelSignature


class SnowflakeCallbackTest(parameterized.TestCase):
    """Tests for Snowflake callback classes."""

    supported_model_classes = [XGBClassifier, XGBRegressor, xgb.Booster, lgb.Booster, LGBMClassifier, LGBMRegressor]

    def setUp(self) -> None:
        self.experiment_tracking = MagicMock(spec=ExperimentTracking)

        # Create training data and parameters
        self.X = np.array([[1, 2], [3, 4]])
        self.y = np.array([0, 1])
        self.num_steps = 2

    def _train_model(
        self,
        model_class: type[Union[BaseEstimator, lgb.Booster, xgb.Booster]],
        callback: Union[SnowflakeXgboostCallback, SnowflakeLightgbmCallback],
    ) -> None:
        """Helper method to train a model using either the sklearn or XGBoost API depending on the model class."""
        if issubclass(model_class, XGBModel):
            assert isinstance(callback, SnowflakeXgboostCallback)
            model = model_class(n_estimators=self.num_steps, callbacks=[callback])
            model.fit(self.X, self.y, eval_set=[(self.X, self.y)])
        elif model_class is xgb.Booster:
            assert isinstance(callback, SnowflakeXgboostCallback)
            xgb_dtrain = xgb.DMatrix(self.X, label=self.y)
            xgb.train(
                {}, xgb_dtrain, evals=[(xgb_dtrain, "train")], num_boost_round=self.num_steps, callbacks=[callback]
            )
        elif issubclass(model_class, LGBMModel):
            assert isinstance(callback, SnowflakeLightgbmCallback)
            model = model_class(n_estimators=self.num_steps)
            model.fit(self.X, self.y, eval_set=[(self.X, self.y)], callbacks=[callback])
        elif model_class is lgb.Booster:
            assert isinstance(callback, SnowflakeLightgbmCallback)
            lgb_dtrain = lgb.Dataset(self.X, label=self.y)
            lgb.train({}, lgb_dtrain, valid_sets=[lgb_dtrain], num_boost_round=self.num_steps, callbacks=[callback])
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    def _get_callback(
        self, model_class: type[Union[XGBModel, xgb.Booster, lgb.Booster]], **kwargs: Any
    ) -> Union[SnowflakeXgboostCallback, SnowflakeLightgbmCallback]:
        """Helper method to create a SnowflakeXgboostCallback instance."""
        if issubclass(model_class, (XGBModel, xgb.Booster)):
            return SnowflakeXgboostCallback(**kwargs)
        elif issubclass(model_class, (LGBMModel, lgb.Booster)):
            return SnowflakeLightgbmCallback(**kwargs)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    @parameterized.parameters(*supported_model_classes)  # type: ignore[misc]
    def test_log_metrics(self, model_class: type[Union[BaseEstimator, xgb.Booster, lgb.Booster]]) -> None:
        """Test that metrics are autologged."""
        callback = self._get_callback(
            model_class,
            experiment_tracking=self.experiment_tracking,
            log_model=False,
            log_metrics=True,
            log_params=False,
        )
        self._train_model(model_class=model_class, callback=callback)

        self.assertEqual(self.experiment_tracking.log_metric.call_count, self.num_steps)

    @parameterized.product(
        model_class=supported_model_classes,
        model_name=[None, "custom_model_name"],
    )  # type: ignore[misc]
    def test_log_model(
        self, model_class: type[Union[BaseEstimator, xgb.Booster, lgb.Booster]], model_name: Optional[str]
    ) -> None:
        """Test that model is autologged."""
        model_signature = ModelSignature(
            inputs=[
                FeatureSpec(name="feature1", dtype=DataType.FLOAT),
                FeatureSpec(name="feature2", dtype=DataType.FLOAT),
            ],
            outputs=[FeatureSpec(name="target", dtype=DataType.INT8)],
        )
        callback = self._get_callback(
            model_class,
            experiment_tracking=self.experiment_tracking,
            log_model=True,
            log_metrics=False,
            log_params=False,
            model_name=model_name,
            model_signature=model_signature,
        )

        # Set up mock experiment before training the model
        mock_experiment = MagicMock()
        mock_experiment.name = "test_experiment"
        self.experiment_tracking._get_or_set_experiment.return_value = mock_experiment
        self._train_model(model_class=model_class, callback=callback)

        expected_model_name = model_name or mock_experiment.name + "_model"
        self.experiment_tracking.log_model.assert_called_once_with(
            model=ANY, model_name=expected_model_name, signatures={"predict": model_signature}
        )

    @parameterized.parameters(*supported_model_classes)  # type: ignore[misc]
    def test_log_param(self, model_class: type[Union[BaseEstimator, xgb.Booster, lgb.Booster]]) -> None:
        """Test that params are autologged."""
        callback = self._get_callback(
            model_class,
            experiment_tracking=self.experiment_tracking,
            log_model=False,
            log_metrics=False,
            log_params=True,
        )
        self._train_model(model_class=model_class, callback=callback)

        self.experiment_tracking.log_params.assert_called_once()


if __name__ == "__main__":
    absltest.main()
