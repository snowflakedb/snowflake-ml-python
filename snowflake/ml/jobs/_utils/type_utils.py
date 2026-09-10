import os
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Literal, Protocol, runtime_checkable

from typing_extensions import Self

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
    def parts(self) -> tuple[str, ...]:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def stem(self) -> str:
        ...

    @property
    def suffix(self) -> str:
        ...

    @property
    def parent(self) -> Self:
        ...

    @property
    def root(self) -> str:
        ...

    def exists(self) -> bool:
        ...

    def is_file(self) -> bool:
        ...

    def is_dir(self) -> bool:
        ...

    def is_absolute(self) -> bool:
        ...

    def absolute(self) -> Self:
        ...

    def joinpath(self, *other: str | os.PathLike[str]) -> Self:
        ...

    def as_posix(self) -> str:
        ...

    def is_relative_to(self, *other: str | os.PathLike[str]) -> bool:
        ...

    def relative_to(self, *other: str | os.PathLike[str]) -> PurePath:
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
    remote_relative_path: PurePath | None = None
    compress: bool = False


@dataclass(frozen=True)
class PayloadEntrypoint:
    file_path: PayloadPath
    main_func: str | None


@dataclass(frozen=True)
class UploadedPayload:
    # TODO: Include manifest of payload files for validation
    stage_path: PurePath
    entrypoint: list[str | PurePath]
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeResources:
    cpu: float  # Number of vCPU cores
    memory: float  # Memory in GiB
    gpu: int = 0  # Number of GPUs
    gpu_type: str | None = None


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


@dataclass
class JobOptions:
    external_access_integrations: list[str] | None = None
    query_warehouse: str | None = None
    target_instances: int | None = None
    min_instances: int | None = None
    use_async: bool | None = True
    generate_suffix: bool | None = True
    comment: str | None = None


@dataclass
class SpecOptions:
    stage_path: str
    args: list[str] | None = None
    env_vars: dict[str, str] | None = None
    enable_metrics: bool | None = None
    spec_overrides: dict[str, Any] | None = None
    runtime: str | None = None
    enable_stage_mount_v2: bool | None = True
    artifact_repositories: list[str] | None = None
