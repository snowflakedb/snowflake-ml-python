# mypy: disable-error-code="import"
from enum import Enum
from typing import TYPE_CHECKING, Dict, Literal, Sequence, TypedDict, TypeVar, Union

import numpy.typing as npt
from typing_extensions import NotRequired

if TYPE_CHECKING:
    import catboost
    import lightgbm
    import mlflow
    import numpy as np
    import pandas as pd
    import sentence_transformers
    import sklearn.base
    import sklearn.pipeline
    import tensorflow
    import torch
    import transformers
    import xgboost

    import snowflake.ml.model.custom_model
    import snowflake.ml.model.models.huggingface_pipeline
    import snowflake.snowpark
    from snowflake.ml.modeling.framework import base  # noqa: F401


_SupportedBuiltins = Union[int, float, bool, str, bytes, "_SupportedBuiltinsList"]
_SupportedNumpyDtype = Union[
    "np.int8",
    "np.int16",
    "np.int32",
    "np.int64",
    "np.float32",
    "np.float64",
    "np.uint8",
    "np.uint16",
    "np.uint32",
    "np.uint64",
    "np.bool_",
    "np.str_",
    "np.bytes_",
    "np.datetime64",
]
_SupportedNumpyArray = npt.NDArray[_SupportedNumpyDtype]
_SupportedBuiltinsList = Sequence[_SupportedBuiltins]
_SupportedArrayLike = Union[_SupportedNumpyArray, "torch.Tensor", "tensorflow.Tensor", "tensorflow.Variable"]

SupportedLocalDataType = Union[
    "pd.DataFrame", _SupportedNumpyArray, Sequence[_SupportedArrayLike], _SupportedBuiltinsList
]

SupportedDataType = Union[SupportedLocalDataType, "snowflake.snowpark.DataFrame"]

_DataType = TypeVar("_DataType", bound=SupportedDataType)

CustomModelType = TypeVar("CustomModelType", bound="snowflake.ml.model.custom_model.CustomModel")

SupportedRequireSignatureModelType = Union[
    "catboost.CatBoost",
    "lightgbm.LGBMModel",
    "lightgbm.Booster",
    "snowflake.ml.model.custom_model.CustomModel",
    "sklearn.base.BaseEstimator",
    "sklearn.pipeline.Pipeline",
    "xgboost.XGBModel",
    "xgboost.Booster",
    "torch.nn.Module",
    "torch.jit.ScriptModule",  # type:ignore[name-defined]
    "tensorflow.Module",
]

SupportedNoSignatureRequirementsModelType = Union[
    "base.BaseEstimator",
    "mlflow.pyfunc.PyFuncModel",
    "transformers.Pipeline",
    "sentence_transformers.SentenceTransformer",
    "snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel",
]

SupportedModelType = Union[
    SupportedRequireSignatureModelType,
    SupportedNoSignatureRequirementsModelType,
]
"""This is defined as the type that Snowflake native model packaging could accept.
Here is all acceptable types of Snowflake native model packaging and its handler file in _model_handlers/ folder.

| Type                            | Handler File | Handler             |
|---------------------------------|--------------|---------------------|
| catboost.CatBoost       | catboost.py   | _CatBoostModelHandler    |
| snowflake.ml.model.custom_model.CustomModel | custom.py    | _CustomModelHandler |
| sklearn.base.BaseEstimator      | sklearn.py   | _SKLModelHandler    |
| sklearn.pipeline.Pipeline       | sklearn.py   | _SKLModelHandler    |
| xgboost.XGBModel       | xgboost.py   | _XGBModelHandler    |
| xgboost.Booster        | xgboost.py   | _XGBModelHandler    |
| lightgbm.LGBMModel       | lightgbm.py   | _LGBMModelHandler    |
| lightgbm.Booster        | lightgbm.py   | _LGBMModelHandler    |
| snowflake.ml.framework.base.BaseEstimator      | snowmlmodel.py   | _SnowMLModelHandler    |
| torch.nn.Module      | pytroch.py   | _PyTorchHandler    |
| torch.jit.ScriptModule      | torchscript.py   | _TorchScriptHandler    |
| tensorflow.Module     | tensorflow.py   | _TensorFlowHandler    |
| mlflow.pyfunc.PyFuncModel | mlflow.py   | _MLFlowHandler |
| transformers.Pipeline | huggingface_pipeline.py | _HuggingFacePipelineHandler |
| huggingface_pipeline.HuggingFacePipelineModel | huggingface_pipeline.py | _HuggingFacePipelineHandler |
| sentence_transformers.SentenceTransformer | sentence_transformers.py | _SentenceTransformerHandler |
"""

SupportedModelHandlerType = Literal[
    "catboost",
    "custom",
    "huggingface_pipeline",
    "lightgbm",
    "mlflow",
    "pytorch",
    "sentence_transformers",
    "sklearn",
    "snowml",
    "tensorflow",
    "torchscript",
    "xgboost",
]

_ModelType = TypeVar("_ModelType", bound=SupportedModelType)


class ModelMethodSaveOptions(TypedDict):
    case_sensitive: NotRequired[bool]
    max_batch_size: NotRequired[int]
    function_type: NotRequired[Literal["FUNCTION", "TABLE_FUNCTION"]]


class BaseModelSaveOption(TypedDict):
    """Options for saving the model.

    embed_local_ml_library: Embedding local SnowML into the code directory of the folder.
    relax_version: Whether or not relax the version constraints of the dependencies if unresolvable in Warehouse.
        It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to True.
    """

    embed_local_ml_library: NotRequired[bool]
    relax_version: NotRequired[bool]
    function_type: NotRequired[Literal["FUNCTION", "TABLE_FUNCTION"]]
    method_options: NotRequired[Dict[str, ModelMethodSaveOptions]]
    enable_explainability: NotRequired[bool]


class CatBoostModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class CustomModelSaveOption(BaseModelSaveOption):
    cuda_version: NotRequired[str]


class SKLModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class XGBModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class LGBMModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class SNOWModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class PyTorchSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class TorchScriptSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class TensorflowSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class MLFlowSaveOptions(BaseModelSaveOption):
    model_uri: NotRequired[str]
    ignore_mlflow_metadata: NotRequired[bool]
    ignore_mlflow_dependencies: NotRequired[bool]


class HuggingFaceSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


class SentenceTransformersSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


ModelSaveOption = Union[
    BaseModelSaveOption,
    CatBoostModelSaveOptions,
    CustomModelSaveOption,
    LGBMModelSaveOptions,
    SKLModelSaveOptions,
    XGBModelSaveOptions,
    SNOWModelSaveOptions,
    PyTorchSaveOptions,
    TorchScriptSaveOptions,
    TensorflowSaveOptions,
    MLFlowSaveOptions,
    HuggingFaceSaveOptions,
    SentenceTransformersSaveOptions,
]


class BaseModelLoadOption(TypedDict):
    """Options for loading the model."""

    ...


class CatBoostModelLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]


class CustomModelLoadOption(BaseModelLoadOption):
    ...


class SKLModelLoadOptions(BaseModelLoadOption):
    ...


class XGBModelLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]


class LGBMModelLoadOptions(BaseModelLoadOption):
    ...


class SNOWModelLoadOptions(BaseModelLoadOption):
    ...


class PyTorchLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]


class TorchScriptLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]


class TensorflowLoadOptions(BaseModelLoadOption):
    ...


class MLFlowLoadOptions(BaseModelLoadOption):
    ...


class HuggingFaceLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]
    device_map: NotRequired[str]
    device: NotRequired[Union[str, int]]


class SentenceTransformersLoadOptions(BaseModelLoadOption):
    use_gpu: NotRequired[bool]
    device: NotRequired[str]


ModelLoadOption = Union[
    BaseModelLoadOption,
    CatBoostModelLoadOptions,
    CustomModelLoadOption,
    LGBMModelLoadOptions,
    SKLModelLoadOptions,
    XGBModelLoadOptions,
    SNOWModelLoadOptions,
    PyTorchLoadOptions,
    TorchScriptLoadOptions,
    TensorflowLoadOptions,
    MLFlowLoadOptions,
    HuggingFaceLoadOptions,
    SentenceTransformersLoadOptions,
]


class Task(Enum):
    UNKNOWN = "UNKNOWN"
    TABULAR_BINARY_CLASSIFICATION = "TABULAR_BINARY_CLASSIFICATION"
    TABULAR_MULTI_CLASSIFICATION = "TABULAR_MULTI_CLASSIFICATION"
    TABULAR_REGRESSION = "TABULAR_REGRESSION"
    TABULAR_RANKING = "TABULAR_RANKING"
