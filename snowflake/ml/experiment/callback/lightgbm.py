from typing import TYPE_CHECKING, Optional
from warnings import warn

import lightgbm as lgb

from snowflake.ml.experiment import utils

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
    from snowflake.ml.model.model_signature import ModelSignature


class SnowflakeLightgbmCallback(lgb.callback._RecordEvaluationCallback):
    """LightGBM callback for automatically logging to a Snowflake ML Experiment."""

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
        Creates a new LightGBM callback.

        Args:
            experiment_tracking (snowflake.ml.experiment.ExperimentTracking): The Experiment Tracking instance
                to use for logging.
            log_model (bool): Whether to log the model at the end of training. Default is True.
            log_metrics (bool): Whether to log metrics during training. Default is True.
            log_params (bool): Whether to log model parameters at the start of training. Default is True.
            log_every_n_epochs (int): Frequency with which to log metrics. Must be positive.
                Default is 1, logging after every iteration.
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
                        version_name=self.version_name,
                        signatures={"predict": self.model_signature},
                    )
                else:
                    warn(
                        "Model will not be logged because model signature is missing. To autolog the model, "
                        "please specify `model_signature` when constructing SnowflakeLightgbmCallback."
                    )
