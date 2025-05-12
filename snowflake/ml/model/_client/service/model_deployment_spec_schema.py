from typing import Optional

from pydantic import BaseModel


class Model(BaseModel):
    name: str
    version: str


class ImageBuild(BaseModel):
    compute_pool: str
    image_repo: str
    force_rebuild: bool
    external_access_integrations: Optional[list[str]] = None


class Service(BaseModel):
    name: str
    compute_pool: str
    ingress_enabled: bool
    max_instances: int
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[str] = None
    num_workers: Optional[int] = None
    max_batch_rows: Optional[int] = None


class Job(BaseModel):
    name: str
    compute_pool: str
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[str] = None
    num_workers: Optional[int] = None
    max_batch_rows: Optional[int] = None
    warehouse: str
    target_method: str
    input_table_name: str
    output_table_name: str


class LogModelArgs(BaseModel):
    pip_requirements: Optional[list[str]] = None
    conda_dependencies: Optional[list[str]] = None
    target_platforms: Optional[list[str]] = None
    comment: Optional[str] = None
    warehouse: Optional[str] = None


class HuggingFaceModel(BaseModel):
    hf_model_name: str
    task: Optional[str] = None
    tokenizer: Optional[str] = None
    hf_token: Optional[str] = None
    trust_remote_code: Optional[bool] = False
    revision: Optional[str] = None
    hf_model_kwargs: Optional[str] = "{}"


class ModelLogging(BaseModel):
    log_model_args: Optional[LogModelArgs] = None
    hf_model: Optional[HuggingFaceModel] = None


class ModelServiceDeploymentSpec(BaseModel):
    models: list[Model]
    image_build: ImageBuild
    service: Service
    model_loggings: Optional[list[ModelLogging]] = None


class ModelJobDeploymentSpec(BaseModel):
    models: list[Model]
    image_build: ImageBuild
    job: Job
    model_loggings: Optional[list[ModelLogging]] = None
