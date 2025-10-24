from typing import Any, Optional, Union

import lightgbm as lgb
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.lightgbm import SnowflakeLightgbmCallback
from snowflake.ml.experiment.callback.test.base import SnowflakeCallbackTest


class SnowflakeLightgbmCallbackTest(SnowflakeCallbackTest, parameterized.TestCase):

    supported_model_classes = [lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.Booster]
    ModelClass = Union[lgb.LGBMModel, lgb.Booster]

    def _train_model(
        self,
        model_class: type[ModelClass],
        callback: SnowflakeLightgbmCallback,
    ) -> None:
        if issubclass(model_class, lgb.LGBMModel):
            assert isinstance(callback, SnowflakeLightgbmCallback)
            model = model_class(n_estimators=self.num_steps)
            model.fit(self.X, self.y, eval_set=[(self.X, self.y)], callbacks=[callback])
        elif model_class is lgb.Booster:
            assert isinstance(callback, SnowflakeLightgbmCallback)
            dtrain = lgb.Dataset(self.X, label=self.y)
            lgb.train({}, dtrain, valid_sets=[dtrain], num_boost_round=self.num_steps, callbacks=[callback])
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    def _get_callback(self, **kwargs: Any) -> SnowflakeLightgbmCallback:
        return SnowflakeLightgbmCallback(**kwargs)

    @parameterized.product(model_class=supported_model_classes, log_every_n_epochs=[1, 2])  # type: ignore[misc]
    def test_log_metrics(self, model_class: type[ModelClass], log_every_n_epochs: int) -> None:
        super()._log_metrics(model_class, log_every_n_epochs=log_every_n_epochs)

    @parameterized.product(
        model_class=supported_model_classes,
        model_name=[None, "custom_model_name"],
        version_name=[None, "v1"],
    )  # type: ignore[misc]
    def test_log_model(
        self, model_class: type[ModelClass], model_name: Optional[str] = None, version_name: Optional[str] = None
    ) -> None:
        super()._log_model(model_class, model_name, version_name)

    @parameterized.parameters(*supported_model_classes)  # type: ignore[misc]
    def test_log_param(self, model_class: type[ModelClass]) -> None:
        super()._log_param(model_class)


if __name__ == "__main__":
    absltest.main()
