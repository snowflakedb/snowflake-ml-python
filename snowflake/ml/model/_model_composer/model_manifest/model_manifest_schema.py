# This files contains schema definition of what will be written into MANIFEST.yml

from typing import Any, Dict, List, Literal, TypedDict

from typing_extensions import NotRequired, Required

from snowflake.ml.model import model_signature

MODEL_MANIFEST_VERSION = "1.0"

MANIFEST_CLIENT_DATA_KEY_NAME = "snowpark_ml_data"
MANIFEST_CLIENT_DATA_SCHEMA_VERSION = "2024-02-01"


class ModelRuntimeDependenciesDict(TypedDict):
    conda: Required[str]


class ModelRuntimeDict(TypedDict):
    language: Required[Literal["PYTHON"]]
    version: Required[str]
    imports: Required[List[str]]
    dependencies: Required[ModelRuntimeDependenciesDict]


class ModelMethodSignatureField(TypedDict):
    type: Required[str]


class ModelMethodSignatureFieldWithName(ModelMethodSignatureField):
    name: Required[str]


class ModelFunctionMethodDict(TypedDict):
    name: Required[str]
    runtime: Required[str]
    type: Required[Literal["FUNCTION"]]
    handler: Required[str]
    inputs: Required[List[ModelMethodSignatureFieldWithName]]
    outputs: Required[List[ModelMethodSignatureField]]


ModelMethodDict = ModelFunctionMethodDict


class ModelFunctionInfo(TypedDict):
    """Function information.

    Attributes:
        name: Name of the function to be called via SQL.
        target_method: actual target method name to be called.
        signature: The signature of the model method.
    """

    name: Required[str]
    target_method: Required[str]
    signature: Required[model_signature.ModelSignature]


class ModelFunctionInfoDict(TypedDict):
    name: Required[str]
    target_method: Required[str]
    signature: Required[Dict[str, Any]]


class SnowparkMLDataDict(TypedDict):
    schema_version: Required[str]
    functions: Required[List[ModelFunctionInfoDict]]


class ModelManifestDict(TypedDict):
    manifest_version: Required[str]
    runtimes: Required[Dict[str, ModelRuntimeDict]]
    methods: Required[List[ModelMethodDict]]
    user_data: NotRequired[Dict[str, Any]]
