import dataclasses

from snowflake import snowpark


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    """Results produced by a completed AI function evaluation."""

    score: float
    details: snowpark.DataFrame | None = dataclasses.field(default=None, compare=False, repr=False)

    __hash__ = None  # type: ignore[assignment]
