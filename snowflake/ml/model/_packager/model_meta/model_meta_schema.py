# This files contains schema definition of what will be written into model.yml
# Changing this file should lead to a change of the schema version.
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import NotRequired, Required

from snowflake.ml.model import type_hints

MODEL_METADATA_VERSION = "2023-12-01"
MODEL_METADATA_MIN_SNOWPARK_ML_VERSION = "1.0.12"


class FunctionProperties(Enum):
    PARTITIONED = "PARTITIONED"


class ModelRuntimeDependenciesDict(TypedDict):
    conda: Required[str]
    pip: Required[str]


class ModelRuntimeDict(TypedDict):
    imports: Required[List[str]]
    dependencies: Required[ModelRuntimeDependenciesDict]


class ModelEnvDict(TypedDict):
    conda: Required[str]
    pip: Required[str]
    python_version: Required[str]
    cuda_version: NotRequired[Optional[str]]
    snowpark_ml_version: Required[str]


class BaseModelBlobOptions(TypedDict):
    ...


class CatBoostModelBlobOptions(BaseModelBlobOptions):
    catboost_estimator_type: Required[str]


class HuggingFacePipelineModelBlobOptions(BaseModelBlobOptions):
    task: Required[str]
    batch_size: Required[int]


class LightGBMModelBlobOptions(BaseModelBlobOptions):
    lightgbm_estimator_type: Required[str]


class MLFlowModelBlobOptions(BaseModelBlobOptions):
    artifact_path: Required[str]


class XgboostModelBlobOptions(BaseModelBlobOptions):
    xgb_estimator_type: Required[str]


ModelBlobOptions = Union[
    BaseModelBlobOptions,
    HuggingFacePipelineModelBlobOptions,
    MLFlowModelBlobOptions,
    XgboostModelBlobOptions,
]


class ExplainabilityMetadataDict(TypedDict):
    algorithm: Required[str]


class ModelBlobMetadataDict(TypedDict):
    name: Required[str]
    model_type: Required[type_hints.SupportedModelHandlerType]
    path: Required[str]
    handler_version: Required[str]
    function_properties: NotRequired[Dict[str, Dict[str, Any]]]
    artifacts: NotRequired[Dict[str, str]]
    options: NotRequired[ModelBlobOptions]


class ModelMetadataDict(TypedDict):
    creation_timestamp: Required[str]
    env: Required[ModelEnvDict]
    runtimes: NotRequired[Dict[str, ModelRuntimeDict]]
    metadata: NotRequired[Optional[Dict[str, str]]]
    model_type: Required[type_hints.SupportedModelHandlerType]
    models: Required[Dict[str, ModelBlobMetadataDict]]
    name: Required[str]
    signatures: Required[Dict[str, Dict[str, Any]]]
    version: Required[str]
    min_snowpark_ml_version: Required[str]
    task: Required[str]
    explainability: NotRequired[Optional[ExplainabilityMetadataDict]]
    function_properties: NotRequired[Dict[str, Dict[str, Any]]]


class ModelExplainAlgorithm(Enum):
    SHAP = "shap"
