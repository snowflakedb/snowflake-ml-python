from typing import Optional

from pydantic import BaseModel

from snowflake.ml.model._client.service import model_deployment_spec_schema

BaseModel.model_config["protected_namespaces"] = ()


class ModelName(BaseModel):
    model_name: str
    version_name: str


class ModelSpec(BaseModel):
    name: ModelName
    hf_model: Optional[model_deployment_spec_schema.HuggingFaceModel] = None
    log_model_args: Optional[model_deployment_spec_schema.LogModelArgs] = None


class ImportModelSpec(BaseModel):
    compute_pool: str
    models: list[ModelSpec]
