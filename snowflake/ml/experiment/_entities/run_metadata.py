import dataclasses


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
