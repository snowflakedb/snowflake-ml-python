import logging
from typing import Any

from snowflake import snowpark
from snowflake.cortex import _candidate, _experiment_job, _optimization_result
from snowflake.ml.experiment._client import experiment_tracking_sql_client

logger = logging.getLogger(__name__)

_SEED_RUN_NAME = "SEED"
_ITERATION_RUN_PREFIX = "ITER_"
_FRONTIER_VALUES = frozenset({"1", "1.0", "true"})
_NON_FRONTIER_VALUES = frozenset({"0", "0.0", "false"})


class OptimizationJob(_experiment_job.ExperimentJob):
    """Asynchronous AI function optimization experiment."""

    def __init__(self, session: snowpark.Session, experiment: str) -> None:
        super().__init__(session, experiment, job_type="Optimization")
        self._result: _optimization_result.OptimizationResult | None = None

    def result(self, *, timeout: float = -1) -> _optimization_result.OptimizationResult:
        if self._result is None:
            if self._aborted:
                raise RuntimeError(f"Optimization experiment {self._experiment!r} was aborted")
            self.wait(timeout=timeout)
            summary = self._get_summary()
            if error_message := summary.get("error_message"):
                raise RuntimeError(f"Optimization experiment {self._experiment!r} failed: {error_message}")
            self._result = self._load_result(summary)
        return self._result

    def _load_result(self, summary: dict[str, Any]) -> _optimization_result.OptimizationResult:
        client = self._get_client()
        run_names = [
            str(row.as_dict()[experiment_tracking_sql_client.RUN_NAME_COL_NAME])
            for row in client.show_runs_in_experiment(experiment_name=self._experiment_name)
        ]
        metrics = experiment_tracking_sql_client.group_run_attributes(
            client.show_run_metrics_in_experiment(experiment_name=self._experiment_name)
        )
        parameters = experiment_tracking_sql_client.group_run_attributes(
            client.show_run_parameters_in_experiment(experiment_name=self._experiment_name)
        )
        optimization_run_names = sorted(
            (
                run_name
                for run_name in run_names
                if run_name == _SEED_RUN_NAME or run_name.startswith(_ITERATION_RUN_PREFIX)
            ),
            key=_run_order,
        )
        if not optimization_run_names:
            raise RuntimeError(f"Optimization experiment {self._experiment!r} did not return any candidates")
        frontier_names = summary.get("pareto_frontier_runs")
        if frontier_names is None:
            frontier_names = [
                run_name
                for run_name in optimization_run_names
                if self._is_frontier_run(run_name, metrics.get(run_name, {}).get("is_frontier"))
            ]
        unknown_frontier_names = set(frontier_names).difference(optimization_run_names)
        if unknown_frontier_names:
            raise RuntimeError(
                f"Optimization experiment {self._experiment!r} identified unknown frontier candidates "
                f"{sorted(unknown_frontier_names)!r}"
            )
        if not frontier_names:
            raise RuntimeError(
                f"Optimization experiment {self._experiment!r} did not identify any candidates on the "
                "cost-quality frontier"
            )
        candidates: dict[str, _candidate.Candidate] = {}
        for run_name in optimization_run_names:
            try:
                candidates[run_name] = _candidate.from_experiment_run(
                    run_name, metrics.get(run_name, {}), parameters.get(run_name, {}), self._experiment
                )
            except _candidate.IncompleteCandidateError:
                if run_name in frontier_names or run_name == _SEED_RUN_NAME:
                    raise
                logger.warning(
                    f"Optimization experiment {self._experiment!r} run {run_name!r} did not report its model or "
                    "function body; leaving it out of the result's candidates."
                )
        frontier = [candidates[run_name] for run_name in frontier_names]
        baseline = candidates.get(_SEED_RUN_NAME)
        for candidate in frontier if baseline is None else [*frontier, baseline]:
            if candidate.val_score is None or candidate.relative_cost is None:
                raise RuntimeError(
                    f"Optimization experiment {self._experiment!r} returned an incomplete candidate "
                    f"{candidate.run_name!r}"
                )
        metric = _optimization_result.selection_metric(frontier, baseline)
        return _optimization_result.OptimizationResult(
            sorted(
                frontier,
                key=lambda candidate: (
                    _optimization_result.cost_of(candidate),
                    -_optimization_result.score_of(candidate, metric),
                ),
            ),
            baseline=baseline,
            candidates=candidates.values(),
        )

    def _is_frontier_run(self, run_name: str, is_frontier: Any) -> bool:
        if is_frontier is None:
            return False
        normalized_value = str(is_frontier).strip().lower()
        if normalized_value in _FRONTIER_VALUES:
            return True
        if normalized_value in _NON_FRONTIER_VALUES:
            return False
        raise RuntimeError(
            f"Optimization experiment {self._experiment!r} returned an unrecognized is_frontier value "
            f"{is_frontier!r} for run {run_name!r}"
        )


def _run_order(run_name: str) -> tuple[int, int, str]:
    # SEED first, then ITER_<N> by N, then any iteration run whose suffix is not a number.
    if run_name == _SEED_RUN_NAME:
        return (0, 0, run_name)
    suffix = run_name[len(_ITERATION_RUN_PREFIX) :]
    return (1, int(suffix), run_name) if suffix.isdigit() else (2, 0, run_name)
