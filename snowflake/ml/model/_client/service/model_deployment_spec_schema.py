from typing import List, TypedDict

from typing_extensions import NotRequired, Required


class ModelDict(TypedDict):
    name: Required[str]
    version: Required[str]


class ImageBuildDict(TypedDict):
    compute_pool: Required[str]
    image_repo: Required[str]
    image_name: NotRequired[str]
    force_rebuild: Required[bool]
    external_access_integrations: Required[List[str]]


class ServiceDict(TypedDict):
    name: Required[str]
    compute_pool: Required[str]
    ingress_enabled: Required[bool]
    min_instances: Required[int]
    max_instances: Required[int]
    gpu: NotRequired[str]


class ModelDeploymentSpecDict(TypedDict):
    models: Required[List[ModelDict]]
    image_build: Required[ImageBuildDict]
    service: Required[ServiceDict]
