# mypy: disable-error-code="import"
from typing import TYPE_CHECKING, Sequence, TypedDict, TypeVar, Union

import numpy.typing as npt
from typing_extensions import NotRequired, TypeAlias

from snowflake.ml.modeling.framework import base

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import sklearn.base
    import sklearn.pipeline
    import xgboost

    import snowflake.ml.model.custom_model
    import snowflake.snowpark


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

SupportedLocalDataType = Union[
    "pd.DataFrame", _SupportedNumpyArray, Sequence[_SupportedNumpyArray], _SupportedBuiltinsList
]

SupportedDataType = Union[SupportedLocalDataType, "snowflake.snowpark.DataFrame"]

_DataType = TypeVar("_DataType", bound=SupportedDataType)

CustomModelType = TypeVar("CustomModelType", bound="snowflake.ml.model.custom_model.CustomModel")

SupportedLocalModelType = Union[
    "snowflake.ml.model.custom_model.CustomModel",
    "sklearn.base.BaseEstimator",
    "sklearn.pipeline.Pipeline",
    "xgboost.XGBModel",
    "xgboost.Booster",
]

SupportedSnowMLModelType: TypeAlias = base.BaseEstimator

SupportedModelType = Union[
    SupportedLocalModelType,
    SupportedSnowMLModelType,
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
"""


_ModelType = TypeVar("_ModelType", bound=SupportedModelType)


class DeployOptions(TypedDict):
    """Common Options for deploying to Snowflake.

    output_with_input_features: Whether or not preserve the input columns in the output when predicting.
        Defaults to False.
    keep_order: Whether or not preserve the row order when predicting. Only available for dataframe has fewer than 2**64
        rows. Defaults to True.

    Internal-only options
    _use_local_snowml: Use local SnowML when as the execution library of the deployment. If set to True, local SnowML
        would be packed and uploaded to 1) session stage, if it is a temporary deployment, or 2) the provided stage path
        if it is a permanent deployment. It should be set to True before SnowML available in Snowflake Anaconda Channel.
        Default to False.
    """

    _use_local_snowml: NotRequired[bool]
    output_with_input_features: NotRequired[bool]
    keep_order: NotRequired[bool]


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


class ModelSaveOption(TypedDict):
    """Options for saving the model.

    allow_overwritten_stage_file: Flag to indicate when saving the model as a stage file, whether overwriting existed
        file is allowed. Default to False.
    """

    allow_overwritten_stage_file: NotRequired[bool]


class CustomModelSaveOption(TypedDict):
    ...


class SKLModelSaveOptions(ModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class XGBModelSaveOptions(ModelSaveOption):
    target_methods: NotRequired[Sequence[str]]


class SNOWModelSaveOptions(ModelSaveOption):
    target_methods: NotRequired[Sequence[str]]
