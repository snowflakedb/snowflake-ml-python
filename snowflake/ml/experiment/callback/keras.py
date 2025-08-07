import json
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

import keras

from snowflake.ml.experiment import utils

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
    from snowflake.ml.model.model_signature import ModelSignature


class SnowflakeKerasCallback(keras.callbacks.Callback):
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

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        if self.log_params:
            params = json.loads(self.model.to_json())
            self._experiment_tracking.log_params(utils.flatten_nested_params(params))

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if self.log_metrics and logs and epoch % self.log_every_n_epochs == 0:
            for key, value in logs.items():
                try:
                    value = float(value)
                except Exception:
                    pass
                else:
                    self._experiment_tracking.log_metric(key=key, value=value, step=epoch)

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        if self.log_model:
            if not self.model_signature:
                warn(
                    "Model will not be logged because model signature is missing. "
                    "To autolog the model, please specify `model_signature` when constructing SnowflakeKerasCallback."
                )
                return
            model_name = self.model_name or self._experiment_tracking._get_or_set_experiment().name + "_model"
            self._experiment_tracking.log_model(  # type: ignore[call-arg]
                model=self.model,
                model_name=model_name,
                signatures={"predict": self.model_signature},
            )
