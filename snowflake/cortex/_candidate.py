import dataclasses
import math
from typing import Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class Candidate:
    """An AI function candidate produced by an optimization run.

    Frontier candidates always have a ``val_score`` and a ``relative_cost``. Other candidates, such as ones the
    optimizer rejected, may be missing any score or cost the backend did not record.

    Attributes:
        model: Model used by the candidate function.
        body: Candidate scalar SQL expression produced by the backend.
        run_name: Experiment run that produced the candidate.
        val_score: Score on the validation data, or ``None`` if the backend did not record one.
        test_score: Score on the held-out test data, or ``None`` if the candidate was not test-evaluated.
        relative_cost: Cost relative to the seed function, whose cost is 1.0, or ``None`` if not recorded.
        run_type: Role the backend recorded for the run, such as ``"seed"``, ``"accepted"`` or ``"rejected"``.
        parent_candidate: Run of the candidate this one was derived from, if any.
    """

    model: str
    body: str
    run_name: str
    val_score: float | None = None
    test_score: float | None = None
    relative_cost: float | None = None
    run_type: str | None = None
    parent_candidate: str | None = None


class IncompleteCandidateError(RuntimeError):
    """Raised when a run did not report the model or function body that define its candidate."""


def from_experiment_run(
    run_name: str,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    experiment: str,
) -> Candidate:
    """Build a candidate from the metrics and parameters the backend recorded on its run.

    A reported metric that is not a finite number raises ``RuntimeError`` (see :func:`parse_metric`).

    Args:
        run_name: Run that produced the candidate.
        metrics: Metric values reported by the run, keyed by metric name.
        parameters: Parameter values reported by the run, keyed by parameter name.
        experiment: Experiment that contains the run.

    Returns:
        The candidate. Scores and cost the run did not report are ``None``.

    Raises:
        IncompleteCandidateError: If the run did not report its model or function body.
    """
    model = parameters.get("model")
    body = parameters.get("function_impl")
    if not isinstance(model, str) or not model or not isinstance(body, str) or not body:
        raise IncompleteCandidateError(
            f"Optimization experiment {experiment!r} returned an incomplete candidate {run_name!r}"
        )

    def metric(name: str) -> float | None:
        return parse_metric(metrics, name, run_name=run_name, experiment=experiment)

    def parameter(name: str) -> str | None:
        value = parameters.get(name)
        return str(value) if value else None

    return Candidate(
        model=model,
        body=body,
        run_name=run_name,
        val_score=metric("val_score"),
        test_score=metric("test_score"),
        relative_cost=metric("cost_compared_to_seed"),
        run_type=parameter("run_type"),
        parent_candidate=parameter("parent_candidate"),
    )


def parse_metric(metrics: dict[str, Any], name: str, *, run_name: str, experiment: str) -> float | None:
    """Read a run metric as a finite number.

    Args:
        metrics: Metric values reported by the run, keyed by metric name.
        name: Name of the metric to read.
        run_name: Run that reported the metrics.
        experiment: Experiment that contains the run.

    Returns:
        The metric value, or ``None`` if the run did not report the metric.

    Raises:
        RuntimeError: If the run reported the metric but its value is not a finite number.
    """
    value = metrics.get(name)
    if value is None:
        return None
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        parsed_value = math.nan
    if not math.isfinite(parsed_value):
        raise RuntimeError(
            f"Optimization experiment {experiment!r} returned an invalid {name} value {value!r} for run {run_name!r}"
        )
    return parsed_value
