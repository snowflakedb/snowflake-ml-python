from typing import Union

import xgboost as xgb
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.xgboost import SnowflakeXgboostCallback
from tests.integ.snowflake.ml.experiment.autolog_integ_test_base import (
    AutologIntegrationTest,
)


class AutologXgboostIntegrationTest(AutologIntegrationTest, parameterized.TestCase):
    def _train_model(
        self,
        model_class: type[Union[xgb.XGBModel, xgb.Booster]],
        callback: SnowflakeXgboostCallback,
    ) -> None:
        if issubclass(model_class, xgb.XGBModel):
            model = model_class(n_estimators=self.num_steps, callbacks=[callback])
            model.fit(self.X, self.y, eval_set=[(self.X, self.y)])
        elif model_class is xgb.Booster:
            dtrain = xgb.DMatrix(self.X, label=self.y)
            xgb.train({}, dtrain, evals=[(dtrain, "train")], num_boost_round=self.num_steps, callbacks=[callback])
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    @parameterized.parameters(
        (xgb.XGBClassifier, "validation_0:logloss"),
        (xgb.XGBRegressor, "validation_0:rmse"),
        (xgb.Booster, "train:rmse"),
    )  # type: ignore[misc]
    def test_autolog(self, model_class: type[Union[xgb.XGBModel, xgb.Booster]], metric_name: str) -> None:
        """Test that autologging works for XGBoost models."""
        self._test_autolog(
            model_class=model_class,
            callback_class=SnowflakeXgboostCallback,
            metric_name=metric_name,
        )


if __name__ == "__main__":
    absltest.main()
