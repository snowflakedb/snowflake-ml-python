import math
from typing import Any
from unittest.mock import ANY, MagicMock

import catboost
import numpy as np
from absl.testing import absltest, parameterized

from snowflake.ml.experiment import experiment_tracking
from snowflake.ml.experiment.callback.catboost import SnowflakeCatboostCallback

_MODEL_CLASSES = [catboost.CatBoostClassifier, catboost.CatBoostRegressor, catboost.CatBoost]
_NUM_STEPS = 3
# With metric_period=2 CatBoost recomputes metrics on iterations 1, 3, 5 and again on the final
# iteration 6, i.e. at 0-based epochs 0, 2, 4, 5.
_NUM_METRIC_PERIOD_STEPS = 6


def _train(
    model_class: type[catboost.CatBoost],
    callback: SnowflakeCatboostCallback,
    *,
    iterations: int = _NUM_STEPS,
    metric_period: int = 1,
) -> catboost.CatBoost:
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    params: dict[str, Any] = {
        "iterations": iterations,
        "metric_period": metric_period,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if model_class is catboost.CatBoost:
        model = model_class(params)
    else:
        model = model_class(**params)
    model.fit(X, y, callbacks=[callback])
    return model


class SnowflakeCatboostCallbackTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.exp = MagicMock(spec=experiment_tracking.ExperimentTracking)

    @parameterized.product(model_class=_MODEL_CLASSES, log_every_n_epochs=[1, 2])  # type: ignore[misc]
    def test_log_metrics(self, model_class: type[catboost.CatBoost], log_every_n_epochs: int) -> None:
        callback = SnowflakeCatboostCallback(self.exp, log_every_n_epochs=log_every_n_epochs)
        _train(model_class, callback)

        expected_call_count = math.ceil(_NUM_STEPS / log_every_n_epochs)
        self.assertEqual(self.exp.log_metric.call_count, expected_call_count)
        for epoch in range(0, _NUM_STEPS, log_every_n_epochs):
            self.exp.log_metric.assert_any_call(key=ANY, value=ANY, step=epoch)

    @parameterized.parameters(*_MODEL_CLASSES)  # type: ignore[misc]
    def test_log_metrics_with_metric_period(self, model_class: type[catboost.CatBoost]) -> None:
        """A metric_period above 1 leaves the previous value in place, which must not be logged again."""
        callback = SnowflakeCatboostCallback(self.exp)
        model = _train(model_class, callback, iterations=_NUM_METRIC_PERIOD_STEPS, metric_period=2)

        # Every value CatBoost actually computed is logged exactly once, and nothing else is.
        computed_metrics = model.get_evals_result()["learn"]
        expected_values = [value for values in computed_metrics.values() for value in values]
        logged_values = [call.kwargs["value"] for call in self.exp.log_metric.call_args_list]
        self.assertCountEqual(logged_values, expected_values)

        # Each metric is logged at most once per step, and only on steps where CatBoost recomputed it.
        logged_steps = [call.kwargs["step"] for call in self.exp.log_metric.call_args_list]
        self.assertEqual(sorted(set(logged_steps)), [0, 2, 4, 5])
        self.assertEqual(len(logged_steps), len(set(logged_steps)) * len(computed_metrics))

    @parameterized.parameters(*_MODEL_CLASSES)  # type: ignore[misc]
    def test_log_params(self, model_class: type[catboost.CatBoost]) -> None:
        callback = SnowflakeCatboostCallback(self.exp, params={"learning_rate": 0.1, "depth": 6})
        _train(model_class, callback)

        self.exp.log_params.assert_called_once()

    @parameterized.parameters(*_MODEL_CLASSES)  # type: ignore[misc]
    def test_no_params_or_model_by_default(self, model_class: type[catboost.CatBoost]) -> None:
        callback = SnowflakeCatboostCallback(self.exp)
        _train(model_class, callback)

        self.exp.log_model.assert_not_called()
        self.exp.log_params.assert_not_called()

    def test_log_every_n_epochs_must_be_positive_integer(self) -> None:
        invalid_values: list[Any] = [0, -1, 1.5, True]
        for invalid_value in invalid_values:
            with self.subTest(log_every_n_epochs=invalid_value):
                with self.assertRaises(ValueError):
                    SnowflakeCatboostCallback(self.exp, log_every_n_epochs=invalid_value)


if __name__ == "__main__":
    absltest.main()
