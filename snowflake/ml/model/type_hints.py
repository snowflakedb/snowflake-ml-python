# mypy: disable-error-code="import"
from typing import TYPE_CHECKING, Sequence, TypedDict, TypeVar, Union

import numpy.typing as npt
from typing_extensions import NotRequired, TypeAlias

if TYPE_CHECKING:
    import mlflow
    import numpy as np
    import pandas as pd
    import sklearn.base
    import sklearn.pipeline
    import tensorflow
    import torch
    import xgboost

    import snowflake.ml.model.custom_model
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
    "np.bool8",
    "np.str0",
    "np.bytes0",
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
    "snowflake.ml.model.custom_model.CustomModel",
    "sklearn.base.BaseEstimator",
    "sklearn.pipeline.Pipeline",
    "xgboost.XGBModel",
    "xgboost.Booster",
    "torch.nn.Module",
    "torch.jit.ScriptModule",  # type:ignore[name-defined]
    "tensorflow.Module",
]

SupportedNoSignatureRequirementsModelType: TypeAlias = Union["base.BaseEstimator", "mlflow.pyfunc.PyFuncModel"]

SupportedModelType = Union[
    SupportedRequireSignatureModelType,
    SupportedNoSignatureRequirementsModelType,
]
"""This is defined as the type that Snowflake native model packaging could accept.
Here is all acceptable types of Snowflake native model packaging and its handler file in _handlers/ folder.

| Type                            | Handler File | Handler             |
|---------------------------------|--------------|---------------------|
| snowflake.ml.model.custom_model.CustomModel | custom.py    | _CustomModelHandler |
| sklearn.base.BaseEstimator      | sklearn.py   | _SKLModelHandler    |
| sklearn.pipeline.Pipeline       | sklearn.py   | _SKLModelHandler    |
| xgboost.XGBModel       | xgboost.py   | _XGBModelHandler    |
| xgboost.Booster        | xgboost.py   | _XGBModelHandler    |
| snowflake.ml.framework.base.BaseEstimator      | snowmlmodel.py   | _SnowMLModelHandler    |
| torch.nn.Module      | pytroch.py   | _PyTorchHandler    |
| torch.jit.ScriptModule      | torchscript.py   | _TorchScriptHandler    |
| tensorflow.Module     | tensorflow.py   | _TensorFlowHandler    |
"""


_ModelType = TypeVar("_ModelType", bound=SupportedModelType)


class DeployOptions(TypedDict):
    """Common Options for deploying to Snowflake.

    keep_order: Whether or not preserve the row order when predicting. Only available for dataframe has fewer than 2**64
        rows. Defaults to True.
    output_with_input_features: Whether or not preserve the input columns in the output when predicting.
        Defaults to False.
    """

    keep_order: NotRequired[bool]
    output_with_input_features: NotRequired[bool]


class WarehouseDeployOptions(DeployOptions):
    """Options for deploying to the Snowflake Warehouse.


    permanent_udf_stage_location: A Snowflake stage option where the UDF should be persisted. If specified, the model
        will be deployed as a permanent UDF, otherwise temporary.
    relax_version: Whether or not relax the version constraints of the dependencies if unresolvable. Defaults to False.
    replace_udf: Flag to indicate when deploying model as permanent UDF, whether overwriting existed UDF is allowed.
        Default to False.
    """

    permanent_udf_stage_location: NotRequired[str]
    relax_version: NotRequired[bool]
    replace_udf: NotRequired[bool]


class SnowparkContainerServiceDeployOptions(DeployOptions):
    """Deployment options for deploying to SnowService.
    When type hint is updated, please ensure the concrete class is updated accordingly at:
    //snowflake/ml/model/_deploy_client/snowservice/_deploy_options

    compute_pool[REQUIRED]: SnowService compute pool name. Please refer to official doc for how to create a
        compute pool: https://docs.snowflake.com/LIMITEDACCESS/snowpark-containers/reference/compute-pool
    image_repo: SnowService image repo path. e.g. "<image_registry>/<db>/<schema>/<repo>". Default to auto
        inferred based on session information.
    min_instances: Minimum number of service replicas. Default to 1.
    max_instances: Maximum number of service replicas. Default to 1.
    endpoint: The specific name of the endpoint that the service function will communicate with. This option is
        useful when the service has multiple endpoints. Default to “predict”.
    prebuilt_snowflake_image: When provided, the image-building step is skipped, and the pre-built image from
        Snowflake is used as is. This option is for users who consistently use the same image for multiple use
        cases, allowing faster deployment. The snowflake image used for deployment is logged to the console for
        future use. Default to None.
    use_gpu: When set to True, a CUDA-enabled Docker image will be used to provide a runtime CUDA environment.
        Default to False.
    """

    compute_pool: str
    image_repo: NotRequired[str]
    min_instances: NotRequired[int]
    max_instances: NotRequired[int]
    endpoint: NotRequired[str]
    prebuilt_snowflake_image: NotRequired[str]
    use_gpu: NotRequired[bool]


class BaseModelSaveOption(TypedDict):
    """Options for saving the model.

    embed_local_ml_library: Embedding local SnowML into the code directory of the folder.
    allow_overwritten_stage_file: Flag to indicate when saving the model as a stage file, whether overwriting existed
        file is allowed. Default to False.
    """

    embed_local_ml_library: NotRequired[bool]
    allow_overwritten_stage_file: NotRequired[bool]


class CustomModelSaveOption(BaseModelSaveOption):
    ...


class SKLModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class XGBModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class SNOWModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class PyTorchSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class TorchScriptSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class TensorflowSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class MLFlowSaveOptions(BaseModelSaveOption):
    model_uri: NotRequired[str]
    ignore_mlflow_metadata: NotRequired[bool]
    ignore_mlflow_dependencies: NotRequired[bool]


ModelSaveOption = Union[
    BaseModelSaveOption,
    CustomModelSaveOption,
    SKLModelSaveOptions,
    XGBModelSaveOptions,
    SNOWModelSaveOptions,
    PyTorchSaveOptions,
    TorchScriptSaveOptions,
    TensorflowSaveOptions,
    MLFlowSaveOptions,
]
