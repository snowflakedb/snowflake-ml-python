from dataclasses import dataclass
from pathlib import PurePath
from typing import List, Literal, Optional, Union

JOB_STATUS = Literal[
    "PENDING",
    "RUNNING",
    "FAILED",
    "DONE",
    "INTERNAL_ERROR",
]


@dataclass(frozen=True)
class PayloadEntrypoint:
    file_path: PurePath
    main_func: Optional[str]


@dataclass(frozen=True)
class UploadedPayload:
    # TODO: Include manifest of payload files for validation
    stage_path: PurePath
    entrypoint: List[Union[str, PurePath]]


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
