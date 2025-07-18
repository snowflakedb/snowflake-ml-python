import dataclasses
import enum
import typing


class RunStatus(str, enum.Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


@dataclasses.dataclass
class Metric:
    name: str
    value: float
    step: int


@dataclasses.dataclass
class Param:
    name: str
    value: str


@dataclasses.dataclass
class RunMetadata:
    status: RunStatus
    metrics: list[Metric]
    parameters: list[Param]

    @classmethod
    def from_dict(
        cls,
        metadata: dict,  # type: ignore[type-arg]
    ) -> "RunMetadata":
        return RunMetadata(
            status=RunStatus(metadata.get("status", RunStatus.UNKNOWN.value)),
            metrics=[Metric(**m) for m in metadata.get("metrics", [])],
            parameters=[Param(**p) for p in metadata.get("parameters", [])],
        )

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return dataclasses.asdict(self)

    def set_metric(
        self,
        key: str,
        value: float,
        step: int,
    ) -> None:
        for metric in self.metrics:
            if metric.name == key and metric.step == step:
                metric.value = value
                break
        else:
            self.metrics.append(Metric(name=key, value=value, step=step))

    def set_param(
        self,
        key: str,
        value: typing.Any,
    ) -> None:
        for parameter in self.parameters:
            if parameter.name == key:
                parameter.value = str(value)
                break
        else:
            self.parameters.append(Param(name=key, value=str(value)))
