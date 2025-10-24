import json
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

import xgboost as xgb

from snowflake.ml.experiment import utils

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
    from snowflake.ml.model.model_signature import ModelSignature


class SnowflakeXgboostCallback(xgb.callback.TrainingCallback):
    def __init__(
        self,
        experiment_tracking: "ExperimentTracking",
        log_model: bool = True,
        log_metrics: bool = True,
        log_params: bool = True,
        log_every_n_epochs: int = 1,
        model_name: Optional[str] = None,
        version_name: Optional[str] = None,
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
        self.version_name = version_name
        self.model_signature = model_signature

    def before_training(self, model: xgb.Booster) -> xgb.Booster:
        if self.log_params:
            params = json.loads(model.save_config())
            self._experiment_tracking.log_params(utils.flatten_nested_params(params))

        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: dict[str, dict[str, Any]]) -> bool:
        if self.log_metrics and epoch % self.log_every_n_epochs == 0:
            for dataset_name, metrics in evals_log.items():
                for metric_name, log in metrics.items():
                    metric_key = dataset_name + ":" + metric_name
                    self._experiment_tracking.log_metric(key=metric_key, value=log[-1], step=epoch)

        return False

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if self.log_model:
            if not self.model_signature:
                warn(
                    "Model will not be logged because model signature is missing. "
                    "To autolog the model, please specify `model_signature` when constructing SnowflakeXgboostCallback."
                )
                return model

            model_name = self.model_name or self._experiment_tracking._get_or_set_experiment().name + "_model"
            self._experiment_tracking.log_model(  # type: ignore[call-arg]
                model=model,
                model_name=model_name,
                version_name=self.version_name,
                signatures={"predict": self.model_signature},
            )

        return model
