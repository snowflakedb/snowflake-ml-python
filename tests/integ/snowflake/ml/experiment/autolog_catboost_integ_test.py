from typing import Any

import catboost
from absl.testing import absltest, parameterized

from snowflake.ml.experiment.callback.catboost import SnowflakeCatboostCallback
from tests.integ.snowflake.ml.experiment.autolog_integ_test_base import (
    AutologIntegrationTest,
)


class AutologCatboostIntegrationTest(AutologIntegrationTest, parameterized.TestCase):
    def _build_and_train(
        self,
        model_class: type[catboost.CatBoost],
        callback_kwargs: dict[str, Any],
    ) -> None:
        params: dict[str, Any] = {"iterations": self.num_steps, "verbose": 0, "allow_writing_files": False}
        if model_class is catboost.CatBoost:
            model = model_class(params)
        elif issubclass(model_class, catboost.CatBoost):
            model = model_class(**params)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")
        callback = SnowflakeCatboostCallback(self.exp, params=model.get_params(), **callback_kwargs)
        model.fit(self.X, self.y, callbacks=[callback])

    @parameterized.parameters(
        (catboost.CatBoostClassifier, "learn:Logloss", 1),
        (catboost.CatBoostRegressor, "learn:RMSE", 2),
        (catboost.CatBoost, "learn:RMSE", 1),
        (catboost.CatBoost, "learn:RMSE", 3),
    )  # type: ignore[misc]
    def test_autolog(self, model_class: type[catboost.CatBoost], metric_name: str, log_every_n_epochs: int) -> None:
        experiment_name = "TEST_EXPERIMENT_AUTOLOG"

        self.exp.set_experiment(experiment_name=experiment_name)
        self._build_and_train(model_class, {"log_every_n_epochs": log_every_n_epochs})

        experiment_fqn = f"{self._db_name}.{self._schema_name}.{experiment_name}"
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {experiment_fqn}").collect()
        self.assertEqual(len(runs), 1)
        run_name = runs[0]["name"]

        metrics = self._session.sql(
            f"SELECT * FROM TABLE(SYSTEM$GET_EXPERIMENT_RUN_METRICS('{experiment_fqn}', '{run_name}'))"
        ).collect()
        metric_set = {(m["METRIC_NAME"], m["STEP"]) for m in metrics}
        for epoch in range(0, self.num_steps, log_every_n_epochs):
            self.assertIn((metric_name, epoch), metric_set)
        for metric in metrics:
            self.assertIn(metric["STEP"], range(0, self.num_steps, log_every_n_epochs))

        # Verify that params were logged
        parameters = self._session.sql(f"SHOW RUN PARAMETERS IN EXPERIMENT {experiment_fqn} RUN {run_name}").collect()
        self.assertGreater(len(parameters), 0)


if __name__ == "__main__":
    absltest.main()
