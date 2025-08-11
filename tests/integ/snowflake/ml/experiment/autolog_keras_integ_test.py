import keras
import numpy as np
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.keras import SnowflakeKerasCallback
from tests.integ.snowflake.ml.experiment.autolog_integ_test_base import (
    AutologIntegrationTest,
)


class AutologKerasIntegrationTest(AutologIntegrationTest, parameterized.TestCase):
    def _train_model(
        self,
        model_class: type[keras.Model],
        callback: SnowflakeKerasCallback,
    ) -> None:
        model = model_class()
        model.add(keras.layers.Dense(1))
        model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"])
        model.fit(self.X.values, np.array(self.y), epochs=self.num_steps, callbacks=[callback])

    @parameterized.parameters(
        (keras.Sequential, "loss", 1),
        (keras.Sequential, "loss", 2),
        (keras.Sequential, "mean_absolute_error", 1),
        (keras.Sequential, "mean_absolute_error", 3),
    )  # type: ignore[misc]
    def test_autolog(self, model_class: type[keras.Model], metric_name: str, log_every_n_epochs: int) -> None:
        """Test that autologging works for Keras models."""
        self._test_autolog(
            model_class=model_class,
            callback_class=SnowflakeKerasCallback,
            metric_name=metric_name,
            log_every_n_epochs=log_every_n_epochs,
        )


if __name__ == "__main__":
    absltest.main()
