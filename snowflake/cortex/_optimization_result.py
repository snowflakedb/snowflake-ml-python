import dataclasses
from collections.abc import Iterable
from typing import cast

from snowflake.cortex import _candidate


@dataclasses.dataclass(frozen=True)
class OptimizationResult:
    """Results produced by a completed AI function optimization.

    Attributes:
        frontier: Candidates on the cost-quality frontier.
        baseline: The seed candidate the optimization started from.
        candidates: Every candidate the optimization produced, including ones the optimizer rejected, in run order.
    """

    frontier: tuple[_candidate.Candidate, ...]
    baseline: _candidate.Candidate | None = None
    candidates: tuple[_candidate.Candidate, ...] = ()

    def __init__(
        self,
        frontier: Iterable[_candidate.Candidate],
        *,
        baseline: _candidate.Candidate | None = None,
        candidates: Iterable[_candidate.Candidate] = (),
    ) -> None:
        frontier_candidates = tuple(frontier)
        if not frontier_candidates:
            raise ValueError("OptimizationResult requires at least one frontier candidate")
        for candidate in _scored_candidates(frontier_candidates, baseline):
            if candidate.val_score is None or candidate.relative_cost is None:
                raise ValueError(
                    f"Frontier and baseline candidates require a val_score and a relative_cost; "
                    f"{candidate.run_name!r} is missing one"
                )
        object.__setattr__(self, "frontier", frontier_candidates)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "candidates", tuple(candidates))

    @property
    def selection_metric(self) -> str:
        """Score the selectors compare.

        ``"test_score"`` when every frontier candidate and the baseline have one, otherwise ``"val_score"``, so
        candidates are never compared across different scores.

        Returns:
            ``"test_score"`` or ``"val_score"``.
        """
        return selection_metric(self.frontier, self.baseline)

    def best_quality(self) -> _candidate.Candidate:
        metric = self.selection_metric
        return max(self.frontier, key=lambda candidate: score_of(candidate, metric))

    def cheapest(self) -> _candidate.Candidate:
        return min(self.frontier, key=cost_of)

    def best_under_cost(self, max_relative_cost: float) -> _candidate.Candidate:
        """Return the highest-scoring candidate within a cost ratio of the seed.

        A relative cost of 1.0 represents the same estimated cost as the seed function.

        Args:
            max_relative_cost: Maximum cost relative to the seed function.

        Returns:
            The highest-scoring candidate within the cost limit.

        Raises:
            ValueError: If no candidate is within the cost limit.
        """
        metric = self.selection_metric
        candidates = [candidate for candidate in self.frontier if cost_of(candidate) <= max_relative_cost]
        if not candidates:
            raise ValueError(f"No candidate has relative_cost <= {max_relative_cost}")
        return max(candidates, key=lambda candidate: score_of(candidate, metric))

    def cheapest_above_quality(self, min_quality: float) -> _candidate.Candidate:
        metric = self.selection_metric
        candidates = [candidate for candidate in self.frontier if score_of(candidate, metric) >= min_quality]
        if not candidates:
            raise ValueError(f"No candidate has {metric} >= {min_quality}")
        return min(candidates, key=cost_of)


def selection_metric(frontier: Iterable[_candidate.Candidate], baseline: _candidate.Candidate | None) -> str:
    """Return the score that frontier candidates and the baseline are compared on.

    Args:
        frontier: Candidates on the cost-quality frontier.
        baseline: The seed candidate, if any.

    Returns:
        ``"test_score"`` when every one of them has a test score, otherwise ``"val_score"``.
    """
    if all(candidate.test_score is not None for candidate in _scored_candidates(tuple(frontier), baseline)):
        return "test_score"
    return "val_score"


def score_of(candidate: _candidate.Candidate, metric: str) -> float:
    """Return a frontier or baseline candidate's ``metric`` score, as chosen by :func:`selection_metric`."""
    # Frontier and baseline candidates are validated to have a val_score, and selection_metric only picks
    # test_score when all of them have one.
    return cast(float, candidate.test_score if metric == "test_score" else candidate.val_score)


def cost_of(candidate: _candidate.Candidate) -> float:
    """Return a frontier or baseline candidate's relative cost."""
    # Frontier and baseline candidates are validated to have a relative_cost.
    return cast(float, candidate.relative_cost)


def _scored_candidates(
    frontier: tuple[_candidate.Candidate, ...], baseline: _candidate.Candidate | None
) -> tuple[_candidate.Candidate, ...]:
    return frontier if baseline is None else (*frontier, baseline)
