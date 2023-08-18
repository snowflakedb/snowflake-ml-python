import warnings
from typing import Any, List, Literal, Optional, Sequence, Type

import numpy as np
import pandas as pd

import snowflake.snowpark
import snowflake.snowpark.types as spt
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import formatting, identifier
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import (
    base_handler,
    builtins_handler as builtins_handler,
    core,
    numpy_handler,
    pandas_handler,
    pytorch_handler,
    snowpark_handler,
    tensorflow_handler,
    utils,
)

DataType = core.DataType
BaseFeatureSpec = core.BaseFeatureSpec
FeatureSpec = core.FeatureSpec
FeatureGroupSpec = core.FeatureGroupSpec
ModelSignature = core.ModelSignature


_LOCAL_DATA_HANDLERS: List[Type[base_handler.BaseDataHandler[Any]]] = [
    pandas_handler.PandasDataFrameHandler,
    numpy_handler.NumpyArrayHandler,
    builtins_handler.ListOfBuiltinHandler,
    numpy_handler.SeqOfNumpyArrayHandler,
    pytorch_handler.SeqOfPyTorchTensorHandler,
    tensorflow_handler.SeqOfTensorflowTensorHandler,
]
_ALL_DATA_HANDLERS = _LOCAL_DATA_HANDLERS + [snowpark_handler.SnowparkDataFrameHandler]


def _truncate_data(data: model_types.SupportedDataType) -> model_types.SupportedDataType:
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            row_count = handler.count(data)
            if row_count <= handler.SIG_INFER_ROWS_COUNT_LIMIT:
                return data

            warnings.warn(
                formatting.unwrap(
                    f"""
                    The sample input has {row_count} rows, thus a truncation happened before inferring signature.
                    This might cause inaccurate signature inference.
                    If that happens, consider specifying signature manually.
                    """
                ),
                category=UserWarning,
            )
            return handler.truncate(data)
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_IMPLEMENTED,
        original_exception=NotImplementedError(
            f"Unable to infer model signature: Un-supported type provided {type(data)} for data truncate."
        ),
    )


def _infer_signature(
    data: model_types.SupportedLocalDataType, role: Literal["input", "output"]
) -> Sequence[core.BaseFeatureSpec]:
    """Infer the inputs/outputs signature given a data that could be dataframe, numpy array or list.
        Dispatching is used to separate logic for different types.
        (Not using Python's singledispatch for unsupported feature of union dispatching in 3.8)

    Args:
        data: The data that we want to infer signature from.
        role: a flag indicating that if this is to infer an input or output feature.

    Raises:
        SnowflakeMLException: NotImplementedError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            return handler.infer_signature(data, role)
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_IMPLEMENTED,
        original_exception=NotImplementedError(
            f"Unable to infer model signature: Un-supported type provided {type(data)} for X type inference."
        ),
    )


def _validate_pandas_df(data: pd.DataFrame, features: Sequence[core.BaseFeatureSpec]) -> None:
    """It validates pandas dataframe with provided features.

    Args:
        data: A pandas dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.

    Raises:
        SnowflakeMLException: NotImplementedError: FeatureGroupSpec is not supported.
        SnowflakeMLException: ValueError: Raised when a feature cannot be found.
        SnowflakeMLException: ValueError: Raised when feature is scalar but confront list element.
        SnowflakeMLException: ValueError: Raised when feature type is not aligned in list element.
        SnowflakeMLException: ValueError: Raised when feature shape is not aligned in list element.
        SnowflakeMLException: ValueError: Raised when feature is scalar but confront array element.
        SnowflakeMLException: ValueError: Raised when feature type is not aligned in numpy array element.
        SnowflakeMLException: ValueError: Raised when feature shape is not aligned in numpy array element.
        SnowflakeMLException: ValueError: Raised when feature type is not aligned in string element.
        SnowflakeMLException: ValueError: Raised when feature type is not aligned in bytes element.
    """
    for feature in features:
        ft_name = feature.name
        try:
            data_col = data[ft_name]
        except KeyError:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(f"Data Validation Error: feature {ft_name} does not exist in data."),
            )

        df_col_dtype = data_col.dtype
        if isinstance(feature, core.FeatureGroupSpec):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
            )

        assert isinstance(feature, core.FeatureSpec)  # assert for mypy.
        ft_type = feature._dtype
        ft_shape = feature._shape
        if df_col_dtype != np.dtype("O"):
            if ft_type != core.DataType.from_numpy_type(df_col_dtype):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by all elements in {data_col}."
                    ),
                )
            elif ft_shape is not None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + "Feature is a array type feature while scalar data is provided."
                    ),
                )
        else:
            if isinstance(data_col[0], list):
                if not ft_shape:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + "Feature is a scalar feature while list data is provided."
                        ),
                    )

                converted_data_list = [utils.convert_list_to_ndarray(data_row) for data_row in data_col]

                if not all(
                    core.DataType.from_numpy_type(converted_data.dtype) == ft_type
                    for converted_data in converted_data_list
                ):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature type {ft_type} is not met by all elements in {data_col}."
                        ),
                    )

                if ft_shape and ft_shape != (-1,):
                    if not all(np.shape(converted_data) == ft_shape for converted_data in converted_data_list):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error in feature {ft_name}: "
                                + f"Feature shape {ft_shape} is not met by all elements in {data_col}."
                            ),
                        )

            elif isinstance(data_col[0], np.ndarray):
                if not ft_shape:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + "Feature is a scalar feature while array data is provided."
                        ),
                    )

                if not all(core.DataType.from_numpy_type(data_row.dtype) == ft_type for data_row in data_col):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature type {ft_type} is not met by all elements in {data_col}."
                        ),
                    )

                ft_shape = feature._shape
                if ft_shape and ft_shape != (-1,):
                    if not all(np.shape(data_row) == ft_shape for data_row in data_col):
                        ft_shape = (-1,)
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error in feature {ft_name}: "
                                + f"Feature shape {ft_shape} is not met by all elements in {data_col}."
                            ),
                        )

            elif isinstance(data_col[0], str):
                if ft_shape is not None:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + "Feature is a array type feature while scalar data is provided."
                        ),
                    )

                if ft_type != core.DataType.STRING:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature type {ft_type} is not met by all elements in {data_col}."
                        ),
                    )

            elif isinstance(data_col[0], bytes):
                if ft_shape is not None:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + "Feature is a array type feature while scalar data is provided."
                        ),
                    )

                if ft_type != core.DataType.BYTES:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature type {ft_type} is not met by all elements in {data_col}."
                        ),
                    )


def _validate_snowpark_data(data: snowflake.snowpark.DataFrame, features: Sequence[core.BaseFeatureSpec]) -> None:
    """Validate Snowpark DataFrame as input

    Args:
        data: A snowpark dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.

    Raises:
        SnowflakeMLException: NotImplementedError: FeatureGroupSpec is not supported.
        SnowflakeMLException: ValueError: Raised when confronting invalid feature.
        SnowflakeMLException: ValueError: Raised when a feature cannot be found.
    """
    schema = data.schema
    for feature in features:
        ft_name = feature.name
        found = False
        for field in schema.fields:
            name = identifier.get_unescaped_names(field.name)
            if name == ft_name:
                found = True
                if field.nullable:
                    warnings.warn(
                        f"Warn in feature {ft_name}: Nullable column {field.name} provided,"
                        + " inference might fail if there is null value.",
                        category=RuntimeWarning,
                    )
                if isinstance(feature, core.FeatureGroupSpec):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.NOT_IMPLEMENTED,
                        original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                    )
                assert isinstance(feature, core.FeatureSpec)  # mypy
                ft_type = feature._dtype
                field_data_type = field.datatype
                if isinstance(field_data_type, spt.ArrayType):
                    if feature._shape is None:
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error in feature {ft_name}: "
                                + f"Feature is a array feature, while {field.name} is not."
                            ),
                        )
                    warnings.warn(
                        f"Warn in feature {ft_name}: Feature is a array feature, type validation cannot happen.",
                        category=RuntimeWarning,
                    )
                else:
                    if feature._shape:
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error in feature {ft_name}: "
                                + f"Feature is a scalar feature, while {field.name} is not."
                            ),
                        )
                    if not ft_type.is_same_snowpark_type(field_data_type):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error in feature {ft_name}: "
                                + f"Feature type {ft_type} is not met by column {field.name}."
                            ),
                        )
        if not found:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(f"Data Validation Error: feature {ft_name} does not exist in data."),
            )


def _convert_local_data_to_df(data: model_types.SupportedLocalDataType) -> pd.DataFrame:
    """Convert local data to pandas DataFrame or Snowpark DataFrame

    Args:
        data: The provided data.

    Raises:
        SnowflakeMLException: NotImplementedError: Raised when data cannot be handled by any data handler.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = None
    for handler in _LOCAL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            df = handler.convert_to_df(data, ensure_serializable=False)
            break
    if df is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_IMPLEMENTED,
            original_exception=NotImplementedError(f"Data Validation Error: Un-supported type {type(data)} provided."),
        )

    return df


def _convert_and_validate_local_data(
    data: model_types.SupportedLocalDataType, features: Sequence[core.BaseFeatureSpec]
) -> pd.DataFrame:
    """Validate the data with features in model signature and convert to DataFrame

    Args:
        features: A list of feature specs that the data should follow.
        data: The provided data.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = _convert_local_data_to_df(data)
    df = utils.rename_pandas_df(df, features)
    _validate_pandas_df(df, features)
    df = pandas_handler.PandasDataFrameHandler.convert_to_df(df, ensure_serializable=True)

    return df


def infer_signature(
    input_data: model_types.SupportedLocalDataType,
    output_data: model_types.SupportedLocalDataType,
    input_feature_names: Optional[List[str]] = None,
    output_feature_names: Optional[List[str]] = None,
) -> core.ModelSignature:
    """Infer model signature from given input and output sample data.

    Currently, we support infer the model signature from example input/output data in the following cases:
        - Pandas data frame whose column could have types of supported data types,
            list (including list of supported data types, list of numpy array of supported data types, and nested list),
            and numpy array of supported data types.
            - Does not support DataFrame with CategoricalIndex column index.
            - Does not support DataFrame with column of variant length list or numpy array.
        - Numpy array of supported data types.
        - List of Numpy array of supported data types.
        - List of supported data types, or nested list of supported data types.
            - Does not support list of list of variant length list.

    When a ValueError is raised when inferring the signature, it indicates that the data is ill and it is impossible to
    create a signature reflecting that.
    When a NotImplementedError is raised, it indicates that it might be possible to create a signature reflecting the
    provided data, however, we could not infer it.

    Args:
        input_data: Sample input data for the model.
        output_data: Sample output data for the model.
        input_feature_names: Name for input features. Defaults to None.
        output_feature_names: Name for output features. Defaults to None.

    Returns:
        A model signature.
    """
    inputs = _infer_signature(input_data, role="input")
    inputs = utils.rename_features(inputs, input_feature_names)
    outputs = _infer_signature(output_data, role="output")
    outputs = utils.rename_features(outputs, output_feature_names)
    return core.ModelSignature(inputs, outputs)
