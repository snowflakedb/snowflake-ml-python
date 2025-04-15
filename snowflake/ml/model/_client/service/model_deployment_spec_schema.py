from typing import TypedDict

from typing_extensions import NotRequired, Required


class ModelDict(TypedDict):
    name: Required[str]
    version: Required[str]


class ImageBuildDict(TypedDict):
    compute_pool: Required[str]
    image_repo: Required[str]
    force_rebuild: Required[bool]
    external_access_integrations: NotRequired[list[str]]


class ServiceDict(TypedDict):
    name: Required[str]
    compute_pool: Required[str]
    cpu: NotRequired[str]
    memory: NotRequired[str]
    gpu: NotRequired[str]
    num_workers: NotRequired[int]
    max_batch_rows: NotRequired[int]
    ingress_enabled: Required[bool]
    max_instances: Required[int]


class JobDict(TypedDict):
    name: Required[str]
    compute_pool: Required[str]
    cpu: NotRequired[str]
    memory: NotRequired[str]
    gpu: NotRequired[str]
    num_workers: NotRequired[int]
    max_batch_rows: NotRequired[int]
    warehouse: Required[str]
    target_method: Required[str]
    input_table_name: Required[str]
    output_table_name: Required[str]


class ModelServiceDeploymentSpecDict(TypedDict):
    models: Required[list[ModelDict]]
    image_build: Required[ImageBuildDict]
    service: Required[ServiceDict]


class ModelJobDeploymentSpecDict(TypedDict):
    models: Required[list[ModelDict]]
    image_build: Required[ImageBuildDict]
    job: Required[JobDict]
