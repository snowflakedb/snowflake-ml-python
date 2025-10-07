import os
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Iterator, Literal, Optional, Protocol, Union, runtime_checkable

JOB_STATUS = Literal[
    "PENDING",
    "RUNNING",
    "FAILED",
    "DONE",
    "CANCELLING",
    "CANCELLED",
    "INTERNAL_ERROR",
    "DELETED",
]


@runtime_checkable
class PayloadPath(Protocol):
    """A protocol for path-like objects used in this module, covering methods from pathlib.Path and StagePath."""

    @property
    def name(self) -> str:
        ...

    @property
    def suffix(self) -> str:
        ...

    @property
    def parent(self) -> "PayloadPath":
        ...

    @property
    def root(self) -> str:
        ...

    def exists(self) -> bool:
        ...

    def is_file(self) -> bool:
        ...

    def is_absolute(self) -> bool:
        ...

    def absolute(self) -> "PayloadPath":
        ...

    def joinpath(self, *other: Union[str, os.PathLike[str]]) -> "PayloadPath":
        ...

    def as_posix(self) -> str:
        ...

    def is_relative_to(self, *other: Union[str, os.PathLike[str]]) -> bool:
        ...

    def relative_to(self, *other: Union[str, os.PathLike[str]]) -> PurePath:
        ...

    def __fspath__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...


@dataclass
class PayloadSpec:
    """Represents a payload item to be uploaded."""

    source_path: PayloadPath
    remote_relative_path: Optional[PurePath] = None

    def __iter__(self) -> Iterator[Union[PayloadPath, Optional[PurePath]]]:
        return iter((self.source_path, self.remote_relative_path))


@dataclass(frozen=True)
class PayloadEntrypoint:
    file_path: PayloadPath
    main_func: Optional[str]


@dataclass(frozen=True)
class UploadedPayload:
    # TODO: Include manifest of payload files for validation
    stage_path: PurePath
    entrypoint: list[Union[str, PurePath]]
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeResources:
    cpu: float  # Number of vCPU cores
    memory: float  # Memory in GiB
    gpu: int = 0  # Number of GPUs
    gpu_type: Optional[str] = None


@dataclass(frozen=True)
class ImageSpec:
    resource_requests: ComputeResources
    resource_limits: ComputeResources
    container_image: str


@dataclass(frozen=True)
class ServiceInfo:
    database_name: str
    schema_name: str
    status: str
    compute_pool: str
    target_instances: int
