import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ComputeResources:
    cpu: float  # Number of vCPU cores
    memory: float  # Memory in GiB
    gpu: int = 0  # Number of GPUs
    gpu_type: Optional[str] = None


@dataclass(frozen=True)
class ImageSpec:
    repo: str
    image_name: str
    image_tag: str
    resource_requests: ComputeResources
    resource_limits: ComputeResources

    @property
    def full_name(self) -> str:
        return f"{self.repo}/{self.image_name}:{self.image_tag}"
