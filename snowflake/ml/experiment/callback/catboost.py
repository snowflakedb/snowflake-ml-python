from typing import TYPE_CHECKING, Any

from snowflake.ml.experiment import utils

if TYPE_CHECKING:
    from snowflake.ml.experiment.experiment_tracking import ExperimentTracking


class SnowflakeCatboostCallback:
    """CatBoost callback that logs training metrics and parameters to a Snowflake ML Experiment.

    CatBoost's callback protocol only exposes iteration number and metrics, not the model itself.
    This callback logs metrics during training and, optionally, model parameters. To log the trained
    model, call ``ExperimentTracking.log_model`` after ``fit`` returns::

        callback = SnowflakeCatboostCallback(
            exp,
            params=model.get_params(),
        )
        model.fit(X, y, callbacks=[callback])
        exp.log_model(model, model_name="my_model", signatures={"predict": sig})
    """

    def __init__(
        self,
        experiment_tracking: "ExperimentTracking",
        *,
        params: dict[str, Any] | None = None,
        log_every_n_epochs: int = 1,
    ) -> None:
        """
        Initialize the callback.

        Args:
            experiment_tracking: The Experiment Tracking instance to use for logging.
            params: Model parameters to log at the start of training. Pass
                ``model.get_params()`` before calling ``fit``. If None, no parameters are logged.
            log_every_n_epochs: Frequency with which to log metrics. Must be a positive integer.
                Default is 1, logging after every iteration. A metric is only logged for iterations
                where CatBoost recomputed it, so a model trained with ``metric_period`` greater than 1
                logs less often than this setting alone implies.

        Raises:
            ValueError: When ``log_every_n_epochs`` is not a positive integer.
        """
        self._experiment_tracking = experiment_tracking
        self._params = params
        if not (utils.is_integer(log_every_n_epochs) and log_every_n_epochs > 0):
            raise ValueError("`log_every_n_epochs` must be a positive integer.")
        self.log_every_n_epochs = log_every_n_epochs
        self._params_logged = False
        self._metric_value_counts: dict[str, int] = {}

    def after_iteration(self, info: Any) -> bool:
        # CatBoost numbers iterations from 1; normalize to 0-based to match the other callbacks.
        epoch = info.iteration - 1

        if self._params is not None and not self._params_logged:
            self._experiment_tracking.log_params(utils.flatten_nested_params(self._params))
            self._params_logged = True

        should_log = epoch % self.log_every_n_epochs == 0
        for dataset_name, metrics in info.metrics.items():
            for metric_name, values in metrics.items():
                metric_key = dataset_name + ":" + metric_name
                # CatBoost runs callbacks on every iteration but only appends a metric value on the
                # iterations where it recomputes one, which `metric_period` can make less frequent. A
                # list that did not grow means the trailing value belongs to an earlier iteration, so
                # logging it again would attribute a stale value to this step.
                recomputed = len(values) > self._metric_value_counts.get(metric_key, 0)
                self._metric_value_counts[metric_key] = len(values)
                if should_log and recomputed:
                    self._experiment_tracking.log_metric(key=metric_key, value=values[-1], step=epoch)

        return True
