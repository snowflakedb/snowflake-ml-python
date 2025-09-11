from typing import Optional

from pydantic import BaseModel

BaseModel.model_config["protected_namespaces"] = ()


class Model(BaseModel):
    name: str
    version: str


class InferenceEngineSpec(BaseModel):
    inference_engine_name: str
    inference_engine_args: Optional[list[str]] = None


class ImageBuild(BaseModel):
    compute_pool: Optional[str] = None
    image_repo: Optional[str] = None
    force_rebuild: Optional[bool] = None
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
    inference_engine_spec: Optional[InferenceEngineSpec] = None


class Input(BaseModel):
    input_stage_location: str
    input_file_pattern: str


class Output(BaseModel):
    output_stage_location: str
    completion_filename: str


class Job(BaseModel):
    name: str
    compute_pool: str
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[str] = None
    num_workers: Optional[int] = None
    max_batch_rows: Optional[int] = None
    warehouse: Optional[str] = None
    function_name: str
    input: Input
    output: Output
    replicas: Optional[int] = None


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
    token: Optional[str] = None
    trust_remote_code: Optional[bool] = False
    revision: Optional[str] = None
    hf_model_kwargs: Optional[str] = "{}"


class ModelLogging(BaseModel):
    log_model_args: Optional[LogModelArgs] = None
    hf_model: Optional[HuggingFaceModel] = None


class ModelServiceDeploymentSpec(BaseModel):
    models: list[Model]
    image_build: Optional[ImageBuild] = None
    service: Service
    model_loggings: Optional[list[ModelLogging]] = None


class ModelJobDeploymentSpec(BaseModel):
    models: list[Model]
    image_build: Optional[ImageBuild] = None
    job: Job
    model_loggings: Optional[list[ModelLogging]] = None
