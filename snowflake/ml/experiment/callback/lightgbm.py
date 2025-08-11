from typing import TYPE_CHECKING, Optional
from warnings import warn

import lightgbm as lgb

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
    from snowflake.ml.model.model_signature import ModelSignature


class SnowflakeLightgbmCallback(lgb.callback._RecordEvaluationCallback):
    def __init__(
        self,
        experiment_tracking: "ExperimentTracking",
        log_model: bool = True,
        log_metrics: bool = True,
        log_params: bool = True,
        log_every_n_epochs: int = 1,
        model_name: Optional[str] = None,
        model_signature: Optional["ModelSignature"] = None,
    ) -> None:
        self._experiment_tracking = experiment_tracking
        self.log_model = log_model
        self.log_metrics = log_metrics
        self.log_params = log_params
        if log_every_n_epochs < 1:
            raise ValueError("`log_every_n_epochs` must be positive.")
        self.log_every_n_epochs = log_every_n_epochs
        self.model_name = model_name
        self.model_signature = model_signature

        super().__init__(eval_result={})

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        if self.log_params:
            if env.iteration == env.begin_iteration:  # Log params only at the first iteration
                self._experiment_tracking.log_params(env.params)

        if self.log_metrics and env.iteration % self.log_every_n_epochs == 0:
            super().__call__(env)
            for dataset_name, metrics in self.eval_result.items():
                for metric_name, log in metrics.items():
                    metric_key = dataset_name + ":" + metric_name
                    self._experiment_tracking.log_metric(key=metric_key, value=log[-1], step=env.iteration)

        if self.log_model:
            if env.iteration == env.end_iteration - 1:  # Log model only at the last iteration
                if self.model_signature:
                    model_name = self.model_name or self._experiment_tracking._get_or_set_experiment().name + "_model"
                    self._experiment_tracking.log_model(  # type: ignore[call-arg]
                        model=env.model,
                        model_name=model_name,
                        signatures={"predict": self.model_signature},
                    )
                else:
                    warn(
                        "Model will not be logged because model signature is missing. To autolog the model, "
                        "please specify `model_signature` when constructing SnowflakeLightgbmCallback."
                    )
