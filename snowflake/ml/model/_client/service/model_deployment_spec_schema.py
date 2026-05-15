import enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

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


class FeatureRetrievalType(str, enum.Enum):
    """Retriever backend used to look up feature columns at inference time.

    The string values are the wire format and must stay in sync with the GS-side
    ``ModelDeploymentSpecFeatureLookup`` ``type`` field; OFT_VNEXT is the only
    value GS currently accepts. Inheriting from ``str`` keeps YAML/JSON
    serialization as the plain string for backwards compatibility.
    """

    OFT_VNEXT = "oft_vnext"


class FeatureLookup(BaseModel):
    # Serialize the enum to its string value so the wire format stays "oft_vnext"
    # rather than the Python repr of the enum member.
    model_config = ConfigDict(use_enum_values=True)

    source: str
    type: FeatureRetrievalType
    version: str


class FeatureRetrievalConfig(BaseModel):
    lookups: dict[str, list[FeatureLookup]]


class Service(BaseModel):
    name: str
    compute_pool: str
    ingress_enabled: bool
    min_instances: int
    max_instances: int
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[str] = None
    num_workers: Optional[int] = None
    max_batch_rows: Optional[int] = None
    autocapture: Optional[bool] = None
    inference_engine_spec: Optional[InferenceEngineSpec] = None
    feature_retrieval: Optional[FeatureRetrievalConfig] = None


class Input(BaseModel):
    input_stage_location: Optional[str] = None
    input_file_pattern: str
    column_handling: Optional[str] = None
    params: Optional[str] = None
    partition_columns: Optional[list[str]] = None


class Output(BaseModel):
    output_stage_location: Optional[str] = None
    base_stage_location: Optional[str] = None
    completion_filename: str


class Job(BaseModel):
    name: Optional[str] = None
    name_prefix: Optional[str] = None
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
    sync: Optional[bool] = None
    inference_engine_spec: Optional[InferenceEngineSpec] = None


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
    token_secret_object: Optional[str] = None
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
