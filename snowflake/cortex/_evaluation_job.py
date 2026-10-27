import math

from snowflake import snowpark
from snowflake.cortex import _evaluation_result, _experiment_job

_EVALUATION_RUN_PREFIX = "EVAL_"


class EvaluationJob(_experiment_job.ExperimentJob):
    """Asynchronous AI function evaluation experiment."""

    def __init__(self, session: snowpark.Session, experiment: str) -> None:
        super().__init__(session, experiment, job_type="Evaluation")
        self._result: _evaluation_result.EvaluationResult | None = None

    def result(self, *, timeout: float = -1) -> _evaluation_result.EvaluationResult:
        if self._result is None:
            if self._aborted:
                raise RuntimeError(f"Evaluation experiment {self._experiment!r} was aborted")
            self.wait(timeout=timeout)
            if error_message := self.error_message:
                raise RuntimeError(f"Evaluation experiment {self._experiment!r} failed: {error_message}")
            self._raise_for_failed_runs()
            self._result = self._load_result()
        return self._result

    def _raise_for_failed_runs(self) -> None:
        # The backend writes a run's score before uploading its details and committing it, so a run committed as
        # failed can still carry a score. Any failed run fails the evaluation rather than being averaged in.
        failed_runs = self._get_failed_runs(run_prefix=_EVALUATION_RUN_PREFIX)
        if not failed_runs:
            return
        run_name = failed_runs[0]
        error_message = self._get_run_error_message(run_name)
        detail = f": {error_message}" if error_message else ""
        raise RuntimeError(f"Evaluation experiment {self._experiment!r} run {run_name!r} failed{detail}")

    def _load_result(self) -> _evaluation_result.EvaluationResult:
        rows = self._get_client().show_run_metrics_in_experiment(experiment_name=self._experiment_name)
        scores_by_run: dict[str, float] = {}
        for row in rows:
            attributes = row.as_dict()
            run_name = attributes.get("run_name")
            if (
                isinstance(run_name, str)
                and run_name.startswith(_EVALUATION_RUN_PREFIX)
                and attributes.get("name") == "score"
                and run_name not in scores_by_run
            ):
                try:
                    score = float(attributes["value"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"Evaluation experiment {self._experiment!r} returned an invalid score"
                    ) from error
                scores_by_run[run_name] = score
        if scores_by_run:
            return _evaluation_result.EvaluationResult(score=math.fsum(scores_by_run.values()) / len(scores_by_run))
        raise RuntimeError(f"Evaluation experiment {self._experiment!r} did not return a score")
