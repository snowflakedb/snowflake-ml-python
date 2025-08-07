from typing import Union

import lightgbm as lgb
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.lightgbm import SnowflakeLightgbmCallback
from tests.integ.snowflake.ml.experiment.autolog_integ_test_base import (
    AutologIntegrationTest,
)


class AutologXgboostIntegrationTest(AutologIntegrationTest, parameterized.TestCase):
    def _train_model(
        self,
        model_class: type[Union[lgb.LGBMModel, lgb.Booster]],
        callback: SnowflakeLightgbmCallback,
    ) -> None:
        if issubclass(model_class, lgb.LGBMModel):
            model = model_class(n_estimators=self.num_steps)
            model.fit(self.X, self.y, eval_set=[(self.X, self.y)], callbacks=[callback])
        elif model_class is lgb.Booster:
            dtrain = lgb.Dataset(self.X, label=self.y)
            lgb.train({}, dtrain, valid_sets=[dtrain], num_boost_round=self.num_steps, callbacks=[callback])
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    @parameterized.parameters(
        (lgb.LGBMClassifier, "training:binary_logloss", 1),
        (lgb.LGBMRegressor, "training:l2", 2),
        (lgb.Booster, "training:l2", 1),
        (lgb.Booster, "training:l2", 3),
    )  # type: ignore[misc]
    def test_autolog(
        self, model_class: type[Union[lgb.LGBMModel, lgb.Booster]], metric_name: str, log_every_n_epochs: int
    ) -> None:
        """Test that autologging works for LightGBM models."""
        self._test_autolog(
            model_class=model_class,
            callback_class=SnowflakeLightgbmCallback,
            metric_name=metric_name,
            log_every_n_epochs=log_every_n_epochs,
        )


if __name__ == "__main__":
    absltest.main()
