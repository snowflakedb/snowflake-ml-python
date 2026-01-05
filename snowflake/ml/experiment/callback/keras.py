import json
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

import keras

from snowflake.ml.experiment import utils

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
    from snowflake.ml.model.model_signature import ModelSignature


class SnowflakeKerasCallback(keras.callbacks.Callback):
    """Keras callback for automatically logging to a Snowflake ML Experiment."""

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
        """
        Creates a new Keras callback.

        Args:
            experiment_tracking (snowflake.ml.experiment.ExperimentTracking): The Experiment Tracking instance
                to use for logging.
            log_model (bool): Whether to log the model at the end of training. Default is True.
            log_metrics (bool): Whether to log metrics during training. Default is True.
            log_params (bool): Whether to log model parameters at the start of training. Default is True.
            log_every_n_epochs (int): Frequency with which to log metrics. Must be positive.
                Default is 1, logging after every epoch.
            model_name (Optional[str]): The model name to use when logging the model.
                If None, the model name will be derived from the experiment name.
            version_name (Optional[str]): The model version name to use when logging the model.
                If None, the version name will be randomly generated.
            model_signature (Optional[snowflake.ml.model.model_signature.ModelSignature]): The model signature to use
                when logging the model. This is required if ``log_model`` is set to True.

        Raises:
            ValueError: When ``log_every_n_epochs`` is not a positive integer.
        """
        self._experiment_tracking = experiment_tracking
        self.log_model = log_model
        self.log_metrics = log_metrics
        self.log_params = log_params
        if not (utils.is_integer(log_every_n_epochs) and log_every_n_epochs > 0):
            raise ValueError("`log_every_n_epochs` must be a positive integer.")
        self.log_every_n_epochs = log_every_n_epochs
        self.model_name = model_name
        self.version_name = version_name
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
                version_name=self.version_name,
                signatures={"predict": self.model_signature},
            )
