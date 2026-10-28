import dataclasses
import enum


class RunStatus(enum.Enum):
    """Status of an experiment run.

    ``RUNNING`` is the only status that accepts metrics, parameters, and artifacts. ``PAUSED`` can
    return to ``RUNNING``. ``FINISHED``, ``FAILED``, and ``CANCELLED`` are terminal and cannot be left.
    """

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        """Return whether a run in this status can no longer be changed."""
        return self in {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.CANCELLED}


@dataclasses.dataclass
class Metric:
    name: str
    value: float
    step: int

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Param:
    name: str
    value: str

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return dataclasses.asdict(self)
