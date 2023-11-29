# This files contains schema definition of what will be written into MANIFEST.yml

from typing import Dict, List, Literal, TypedDict

from typing_extensions import NotRequired, Required

MODEL_MANIFEST_VERSION = "1.0"


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


class ModelManifestDict(TypedDict):
    manifest_version: Required[str]
    runtimes: Required[Dict[str, ModelRuntimeDict]]
    methods: Required[List[ModelMethodDict]]
    user_data: NotRequired[Dict[str, str]]
