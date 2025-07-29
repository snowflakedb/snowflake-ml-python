from typing import Any, Optional, Union

import xgboost as xgb
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.test.base import SnowflakeCallbackTest
from snowflake.ml.experiment.callback.xgboost import SnowflakeXgboostCallback


class SnowflakeXgboostCallbackTest(SnowflakeCallbackTest, parameterized.TestCase):

    supported_model_classes = [xgb.XGBClassifier, xgb.XGBRegressor, xgb.Booster]
    ModelClass = Union[xgb.XGBModel, xgb.Booster]

    def _train_model(
        self,
        model_class: type[ModelClass],
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

    def _get_callback(self, **kwargs: Any) -> SnowflakeXgboostCallback:
        return SnowflakeXgboostCallback(**kwargs)

    @parameterized.parameters(*supported_model_classes)  # type: ignore[misc]
    def test_log_metrics(self, model_class: type[ModelClass]) -> None:
        super()._log_metrics(model_class)

    @parameterized.product(
        model_class=supported_model_classes,
        model_name=[None, "custom_model_name"],
    )  # type: ignore[misc]
    def test_log_model(self, model_class: type[ModelClass], model_name: Optional[str] = None) -> None:
        super()._log_model(model_class, model_name)

    @parameterized.parameters(*supported_model_classes)  # type: ignore[misc]
    def test_log_param(self, model_class: type[ModelClass]) -> None:
        super()._log_param(model_class)


if __name__ == "__main__":
    absltest.main()
