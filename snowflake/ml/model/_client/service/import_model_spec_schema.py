from pydantic import BaseModel

from snowflake.ml.model._client.service import model_deployment_spec_schema

BaseModel.model_config["protected_namespaces"] = ()


class ModelName(BaseModel):
    model_name: str
    version_name: str


class ModelSpec(BaseModel):
    name: ModelName
    hf_model: model_deployment_spec_schema.HuggingFaceModel | None = None
    log_model_args: model_deployment_spec_schema.LogModelArgs | None = None


class ImportModelSpec(BaseModel):
    compute_pool: str
    models: list[ModelSpec]
