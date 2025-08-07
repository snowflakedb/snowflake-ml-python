from typing import Any, Optional

import keras
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.keras import SnowflakeKerasCallback
from snowflake.ml.experiment.callback.test.base import SnowflakeCallbackTest


class SnowflakeKerasCallbackTest(SnowflakeCallbackTest, parameterized.TestCase):
    def _train_model(
        self,
        model_class: type[keras.Model],
        callback: SnowflakeKerasCallback,
    ) -> None:
        model = model_class()
        model.add(keras.layers.Dense(1))
        model.compile(loss="mean_squared_error")
        model.fit(self.X, self.y, epochs=self.num_steps, callbacks=[callback])

    def _get_callback(self, **kwargs: Any) -> SnowflakeKerasCallback:
        return SnowflakeKerasCallback(**kwargs)

    @parameterized.parameters(1, 2)  # type: ignore[misc]
    def test_log_metrics(self, log_every_n_epochs: int) -> None:
        super()._log_metrics(keras.Sequential, log_every_n_epochs=log_every_n_epochs)

    @parameterized.parameters(None, "custom_model_name")  # type: ignore[misc]
    def test_log_model(self, model_name: Optional[str] = None) -> None:
        super()._log_model(keras.Sequential, model_name)

    def test_log_param(self) -> None:
        super()._log_param(keras.Sequential)


if __name__ == "__main__":
    absltest.main()
