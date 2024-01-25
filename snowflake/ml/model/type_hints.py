# mypy: disable-error-code="import"
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

import numpy.typing as npt
from typing_extensions import NotRequired, Required

from snowflake.ml.model import deploy_platforms
from snowflake.ml.model._signatures import core

if TYPE_CHECKING:
    import mlflow
    import numpy as np
    import pandas as pd
    import sklearn.base
    import sklearn.pipeline
    import tensorflow
    import torch
    import transformers
    import xgboost

    import snowflake.ml.model.custom_model
    import snowflake.ml.model.models.huggingface_pipeline
    import snowflake.ml.model.models.llm
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

SupportedNoSignatureRequirementsModelType = Union[
    "base.BaseEstimator",
    "mlflow.pyfunc.PyFuncModel",
    "transformers.Pipeline",
    "snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel",
    "snowflake.ml.model.models.llm.LLM",
]

SupportedModelType = Union[
    SupportedRequireSignatureModelType,
    SupportedNoSignatureRequirementsModelType,
]
"""This is defined as the type that Snowflake native model packaging could accept.
Here is all acceptable types of Snowflake native model packaging and its handler file in _model_handlers/ folder.

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
| mlflow.pyfunc.PyFuncModel | mlflow.py   | _MLFlowHandler |
| transformers.Pipeline | huggingface_pipeline.py | _HuggingFacePipelineHandler |
| huggingface_pipeline.HuggingFacePipelineModel | huggingface_pipeline.py | _HuggingFacePipelineHandler |
"""

SupportedModelHandlerType = Literal[
    "custom",
    "huggingface_pipeline",
    "mlflow",
    "pytorch",
    "sklearn",
    "snowml",
    "tensorflow",
    "torchscript",
    "xgboost",
    "llm",
]

_ModelType = TypeVar("_ModelType", bound=SupportedModelType)


class DeployOptions(TypedDict):
    """Common Options for deploying to Snowflake."""

    ...


class WarehouseDeployOptions(DeployOptions):
    """Options for deploying to the Snowflake Warehouse.


    permanent_udf_stage_location: A Snowflake stage option where the UDF should be persisted. If specified, the model
        will be deployed as a permanent UDF, otherwise temporary.
    relax_version: Whether or not relax the version constraints of the dependencies if unresolvable. It detects any
        ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to False.
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
    prebuilt_snowflake_image: When provided, the image-building step is skipped, and the pre-built image from
        Snowflake is used as is. This option is for users who consistently use the same image for multiple use
        cases, allowing faster deployment. The snowflake image used for deployment is logged to the console for
        future use. Default to None.
    num_gpus: Number of GPUs to be used for the service. Default to 0.
    num_workers: Number of workers used for model inference. Please ensure that the number of workers is set lower than
        the total available memory divided by the size of model to prevent memory-related issues. Default is number of
        CPU cores * 2 + 1.
    enable_remote_image_build: When set to True, will enable image build on a remote SnowService job. Default is True.
    force_image_build: When set to True, an image rebuild will occur. The default is False, which means the system
        will automatically check whether a previously built image can be reused
    model_in_image: When set to True, image would container full model weights. The default if False, which
                means image without model weights and we do stage mount to access weights.
    debug_mode: When set to True, deployment artifacts will be persisted in a local temp directory.
    enable_ingress: When set to True, will expose HTTP endpoint for access to the predict method of the created
        service.
    """

    compute_pool: str
    image_repo: NotRequired[str]
    min_instances: NotRequired[int]
    max_instances: NotRequired[int]
    prebuilt_snowflake_image: NotRequired[str]
    num_gpus: NotRequired[int]
    num_workers: NotRequired[int]
    enable_remote_image_build: NotRequired[bool]
    force_image_build: NotRequired[bool]
    model_in_image: NotRequired[bool]
    debug_mode: NotRequired[bool]
    enable_ingress: NotRequired[bool]


class ModelMethodSaveOptions(TypedDict):
    case_sensitive: NotRequired[bool]
    max_batch_size: NotRequired[int]


class BaseModelSaveOption(TypedDict):
    """Options for saving the model.

    embed_local_ml_library: Embedding local SnowML into the code directory of the folder.
    relax_version: Whether or not relax the version constraints of the dependencies if unresolvable. It detects any
        ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to False.
    """

    embed_local_ml_library: NotRequired[bool]
    relax_version: NotRequired[bool]
    _legacy_save: NotRequired[bool]
    method_options: NotRequired[Dict[str, ModelMethodSaveOptions]]


class CustomModelSaveOption(BaseModelSaveOption):
    cuda_version: NotRequired[str]


class SKLModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class XGBModelSaveOptions(BaseModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
    cuda_version: NotRequired[str]


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


class LLMSaveOptions(BaseModelSaveOption):
    cuda_version: NotRequired[str]


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
    HuggingFaceSaveOptions,
    LLMSaveOptions,
]


class ModelLoadOption(TypedDict):
    """Options for loading the model.

    use_gpu: Enable GPU-specific loading logic.
    """

    use_gpu: NotRequired[bool]


class SnowparkContainerServiceDeployDetails(TypedDict):
    """
    Attributes:
        service_info: A snowpark row containing the result of "describe service"
        service_function_sql: SQL for service function creation.
    """

    service_info: Optional[Dict[str, Any]]
    service_function_sql: str


class WarehouseDeployDetails(TypedDict):
    ...


DeployDetails = Union[
    SnowparkContainerServiceDeployDetails,
    WarehouseDeployDetails,
]


class Deployment(TypedDict):
    """Deployment information.

    Attributes:
        name: Name of the deployment.
        platform: Target platform to deploy the model.
        target_method: Target method name.
        signature: The signature of the model method.
        options: Additional options when deploying the model.
    """

    name: Required[str]
    platform: Required[deploy_platforms.TargetPlatform]
    target_method: Required[str]
    signature: core.ModelSignature
    options: Required[DeployOptions]
    details: NotRequired[DeployDetails]
