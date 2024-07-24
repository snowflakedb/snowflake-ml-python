# This files contains schema definition of what will be written into MANIFEST.yml
import enum
from typing import Any, Dict, List, Literal, TypedDict, Union

from typing_extensions import NotRequired, Required

from snowflake.ml.model import model_signature

MODEL_MANIFEST_VERSION = "1.0"

MANIFEST_CLIENT_DATA_KEY_NAME = "snowpark_ml_data"
MANIFEST_CLIENT_DATA_SCHEMA_VERSION = "2024-02-01"


class ModelMethodFunctionTypes(enum.Enum):
    FUNCTION = "FUNCTION"
    TABLE_FUNCTION = "TABLE_FUNCTION"


class ModelRuntimeDependenciesDict(TypedDict):
    conda: NotRequired[str]
    pip: NotRequired[str]


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
    type: Required[str]
    handler: Required[str]
    inputs: Required[List[ModelMethodSignatureFieldWithName]]
    outputs: Required[Union[List[ModelMethodSignatureField], List[ModelMethodSignatureFieldWithName]]]


ModelMethodDict = ModelFunctionMethodDict


class ModelFunctionInfo(TypedDict):
    """Function information.

    Attributes:
        name: Name of the function to be called via SQL.
        target_method: actual target method name to be called.
        target_method_function_type: target method function type (FUNCTION or TABLE_FUNCTION).
        signature: The signature of the model method.
    """

    name: Required[str]
    target_method: Required[str]
    target_method_function_type: Required[str]
    signature: Required[model_signature.ModelSignature]


class ModelFunctionInfoDict(TypedDict):
    name: Required[str]
    target_method: Required[str]
    signature: Required[Dict[str, Any]]


class SnowparkMLDataDict(TypedDict):
    schema_version: Required[str]
    functions: Required[List[ModelFunctionInfoDict]]


class LineageSourceTypes(enum.Enum):
    DATASET = "DATASET"


class LineageSourceDict(TypedDict):
    type: Required[str]
    entity: Required[str]
    version: NotRequired[str]


class ModelManifestDict(TypedDict):
    manifest_version: Required[str]
    runtimes: Required[Dict[str, ModelRuntimeDict]]
    methods: Required[List[ModelMethodDict]]
    user_data: NotRequired[Dict[str, Any]]
    lineage_sources: NotRequired[List[LineageSourceDict]]
