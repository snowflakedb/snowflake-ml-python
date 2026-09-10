import enum

from pydantic import BaseModel, ConfigDict

BaseModel.model_config["protected_namespaces"] = ()


class Model(BaseModel):
    name: str
    version: str


class InferenceEngineSpec(BaseModel):
    inference_engine_name: str
    inference_engine_args: list[str] | None = None


class ImageBuild(BaseModel):
    compute_pool: str | None = None
    image_repo: str | None = None
    force_rebuild: bool | None = None
    external_access_integrations: list[str] | None = None


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
    cpu: str | None = None
    memory: str | None = None
    gpu: str | None = None
    num_workers: int | None = None
    max_batch_rows: int | None = None
    autocapture: bool | None = None
    inference_engine_spec: InferenceEngineSpec | None = None
    feature_retrieval: FeatureRetrievalConfig | None = None


class LogModelArgs(BaseModel):
    pip_requirements: list[str] | None = None
    conda_dependencies: list[str] | None = None
    target_platforms: list[str] | None = None
    comment: str | None = None
    warehouse: str | None = None


class HuggingFaceModel(BaseModel):
    hf_model_name: str
    task: str | None = None
    tokenizer: str | None = None
    token: str | None = None
    token_secret_object: str | None = None
    trust_remote_code: bool | None = False
    revision: str | None = None
    hf_model_kwargs: str | None = "{}"


class ModelLogging(BaseModel):
    log_model_args: LogModelArgs | None = None
    hf_model: HuggingFaceModel | None = None


class ModelServiceDeploymentSpec(BaseModel):
    models: list[Model]
    image_build: ImageBuild | None = None
    service: Service
    model_loggings: list[ModelLogging] | None = None
