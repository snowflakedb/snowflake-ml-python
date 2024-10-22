from typing import List, TypedDict

from typing_extensions import NotRequired, Required


class ModelDict(TypedDict):
    name: Required[str]
    version: Required[str]


class ImageBuildDict(TypedDict):
    compute_pool: Required[str]
    image_repo: Required[str]
    force_rebuild: Required[bool]
    external_access_integrations: NotRequired[List[str]]


class ServiceDict(TypedDict):
    name: Required[str]
    compute_pool: Required[str]
    ingress_enabled: Required[bool]
    max_instances: Required[int]
    cpu: NotRequired[str]
    memory: NotRequired[str]
    gpu: NotRequired[str]
    num_workers: NotRequired[int]
    max_batch_rows: NotRequired[int]


class ModelDeploymentSpecDict(TypedDict):
    models: Required[List[ModelDict]]
    image_build: Required[ImageBuildDict]
    service: Required[ServiceDict]
