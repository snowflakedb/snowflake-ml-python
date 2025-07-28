import enum
import json
import warnings
from typing import Any, Literal, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from typing_extensions import Never

import snowflake.snowpark
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as spt
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import formatting, identifier, sql_identifier
from snowflake.ml.model import type_hints
from snowflake.ml.model._signatures import (
    base_handler,
    builtins_handler,
    core,
    dmatrix_handler,
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


_LOCAL_DATA_HANDLERS: list[type[base_handler.BaseDataHandler[Any]]] = [
    pandas_handler.PandasDataFrameHandler,
    numpy_handler.NumpyArrayHandler,
    builtins_handler.ListOfBuiltinHandler,
    numpy_handler.SeqOfNumpyArrayHandler,
    pytorch_handler.PyTorchTensorHandler,
    pytorch_handler.SeqOfPyTorchTensorHandler,
    tensorflow_handler.TensorflowTensorHandler,
    tensorflow_handler.SeqOfTensorflowTensorHandler,
    dmatrix_handler.XGBoostDMatrixHandler,
]
_ALL_DATA_HANDLERS = _LOCAL_DATA_HANDLERS + [snowpark_handler.SnowparkDataFrameHandler]

_TELEMETRY_PROJECT = "MLOps"
_MODEL_TELEMETRY_SUBPROJECT = "ModelSignature"


def _truncate_data(
    data: type_hints.SupportedDataType,
    length: Optional[int] = 100,
) -> type_hints.SupportedDataType:
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            # If length is None, return the original data
            if length is None:
                return data

            row_count = handler.count(data)
            if row_count <= length:
                return data

            warnings.warn(
                formatting.unwrap(
                    f"""
                    The sample input has {row_count} rows. Using the first 100 rows to define the inputs and outputs
                    of the model and the data types of each. Use `signatures` parameter to specify model inputs and
                    outputs manually if the automatic inference is not correct.
                    """
                ),
                category=UserWarning,
                stacklevel=1,
            )
            return handler.truncate(data, length)
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_IMPLEMENTED,
        original_exception=NotImplementedError(
            f"Unable to infer model signature: Un-supported type provided {type(data)} for data truncate."
        ),
    )


def _infer_signature(
    data: type_hints.SupportedLocalDataType, role: Literal["input", "output"], use_snowflake_identifiers: bool = False
) -> Sequence[core.BaseFeatureSpec]:
    """Infer the inputs/outputs signature given a data that could be dataframe, numpy array or list.
        Dispatching is used to separate logic for different types.
        (Not using Python's singledispatch for unsupported feature of union dispatching in 3.8)

    Args:
        data: The data that we want to infer signature from.
        role: a flag indicating that if this is to infer an input or output feature.
        use_snowflake_identifiers: a flag indicating whether to ensure the signature names are
            valid snowflake identifiers.

    Raises:
        SnowflakeMLException: NotImplementedError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    signature = None
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            signature = handler.infer_signature(data, role)
            break

    if signature is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_IMPLEMENTED,
            original_exception=NotImplementedError(
                f"Unable to infer model signature: Un-supported type provided {type(data)} for X type inference."
            ),
        )

    if use_snowflake_identifiers:
        signature = _rename_signature_with_snowflake_identifiers(signature)

    return signature


def _rename_signature_with_snowflake_identifiers(
    signature: Sequence[core.BaseFeatureSpec],
) -> Sequence[core.BaseFeatureSpec]:
    inferred_names = []
    for feature_spec in signature:
        name = identifier.rename_to_valid_snowflake_identifier(feature_spec.name)
        inferred_names.append(name)

    signature = utils.rename_features(signature, inferred_names)

    return signature


def _validate_array_or_series_type(
    arr: Union[type_hints._SupportedNumpyArray, pd.Series], feature_type: core.DataType, strict: bool = False
) -> bool:
    original_dtype = arr.dtype
    dtype = arr.dtype
    if isinstance(
        dtype,
        (
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
            pd.BooleanDtype,
        ),
    ):
        dtype = dtype.type
    elif isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    elif isinstance(dtype, pd.StringDtype):
        dtype = np.str_
    if feature_type in [
        core.DataType.INT8,
        core.DataType.INT16,
        core.DataType.INT32,
        core.DataType.INT64,
        core.DataType.UINT8,
        core.DataType.UINT16,
        core.DataType.UINT32,
        core.DataType.UINT64,
    ]:
        if not (np.issubdtype(dtype, np.integer)):
            return False
        if not strict:
            return True
        if isinstance(original_dtype, pd.CategoricalDtype):
            min_v, max_v = arr.cat.as_ordered().min(), arr.cat.as_ordered().min()  # type: ignore[union-attr]
        else:
            min_v, max_v = arr.min(), arr.max()
        return bool(max_v <= np.iinfo(feature_type._numpy_type).max and min_v >= np.iinfo(feature_type._numpy_type).min)
    elif feature_type in [core.DataType.FLOAT, core.DataType.DOUBLE]:
        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
            return False
        if not strict:
            return True
        min_v, max_v = arr.min(), arr.max()
        return bool(
            max_v <= np.finfo(feature_type._numpy_type).max  # type: ignore[arg-type]
            and min_v >= np.finfo(feature_type._numpy_type).min  # type: ignore[arg-type]
        )
    elif feature_type in [core.DataType.TIMESTAMP_NTZ]:
        return np.issubdtype(arr.dtype, np.datetime64)
    else:
        return np.can_cast(dtype, feature_type._numpy_type, casting="no")


def _validate_pandas_df(data: pd.DataFrame, features: Sequence[core.BaseFeatureSpec], strict: bool = False) -> None:
    """It validates pandas dataframe with provided features.

    Args:
        data: A pandas dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.
        strict: Enable strict validation, this includes value range based validation

    Raises:
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
                original_exception=ValueError(
                    f"Data Validation Error: feature {ft_name} does not exist in data. "
                    f"Available columns are {data.columns}."
                ),
            )

        if data_col.isnull().any():
            data_col = utils.series_dropna(data_col)
        df_col_dtype = data_col.dtype

        if isinstance(feature, core.FeatureGroupSpec):
            if df_col_dtype != np.dtype("O"):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        f"Data Validation Error in feature group {ft_name}: "
                        + f"It needs to be a dictionary or list of dictionary, but get {df_col_dtype}."
                    ),
                )
            continue

        assert isinstance(feature, core.FeatureSpec)  # assert for mypy.
        ft_type = feature._dtype
        ft_shape = feature._shape
        if isinstance(df_col_dtype, pd.CategoricalDtype):
            df_col_dtype = df_col_dtype.categories.dtype
        if df_col_dtype != np.dtype("O"):
            if not _validate_array_or_series_type(data_col, ft_type, strict=strict):
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
            if utils.check_if_series_is_empty(data_col):
                continue
            if isinstance(data_col.iloc[0], list):
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
                    _validate_array_or_series_type(converted_data, ft_type, strict=strict)
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

            elif isinstance(data_col.iloc[0], np.ndarray):
                if not ft_shape:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + "Feature is a scalar feature while array data is provided."
                        ),
                    )

                if not all(_validate_array_or_series_type(data_row, ft_type, strict=strict) for data_row in data_col):
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

            elif isinstance(data_col.iloc[0], str):
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

            elif isinstance(data_col.iloc[0], bytes):
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


def assert_never(arg: Never) -> Never:
    raise AssertionError("Expected code to be unreachable")


class SnowparkIdentifierRule(enum.Enum):
    INFERRED = "inferred"
    NORMALIZED = "normalized"

    def get_identifier_from_feature(self, ft_name: str) -> str:
        if self == SnowparkIdentifierRule.INFERRED:
            return identifier.get_inferred_name(ft_name)
        elif self == SnowparkIdentifierRule.NORMALIZED:
            return identifier.resolve_identifier(ft_name)
        else:
            assert_never(self)

    def get_sql_identifier_from_feature(self, ft_name: str) -> sql_identifier.SqlIdentifier:
        if self == SnowparkIdentifierRule.INFERRED:
            return sql_identifier.SqlIdentifier(ft_name, case_sensitive=True)
        elif self == SnowparkIdentifierRule.NORMALIZED:
            return sql_identifier.SqlIdentifier(ft_name, case_sensitive=False)
        else:
            assert_never(self)


def _get_dataframe_values_range(
    df: snowflake.snowpark.DataFrame,
) -> dict[str, Union[tuple[int, int], tuple[float, float]]]:
    columns = [
        F.array_construct(F.min(field.name), F.max(field.name)).as_(field.name)
        for field in df.schema.fields
        if isinstance(field.datatype, spt._NumericType)
    ]
    if not columns:
        return {}
    res = df.select(columns).collect()
    if len(res) != 1:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=ValueError(f"Unable to get the value range of fields {df.columns}"),
        )
    return cast(
        dict[str, Union[tuple[int, int], tuple[float, float]]],
        {
            sql_identifier.SqlIdentifier(k, case_sensitive=True).identifier(): (json.loads(v)[0], json.loads(v)[1])
            for k, v in res[0].as_dict().items()
        },
    )


def _validate_snowpark_data(
    data: snowflake.snowpark.DataFrame, features: Sequence[core.BaseFeatureSpec], strict: bool = False
) -> SnowparkIdentifierRule:
    """Validate Snowpark DataFrame as input. It will try to map both normalized name or inferred name.

    Args:
        data: A snowpark dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.
        strict: Enable strict validation, this includes value range based validation.

    Raises:
        SnowflakeMLException: ValueError: Raised when confronting invalid feature.
        SnowflakeMLException: ValueError: Raised when a feature cannot be found.

    Returns:
        Identifier rule to use.
        - inferred: signature `a` - Snowpark DF `"a"`, use `get_inferred_name`
        - normalized: signature `a` - Snowpark DF `A`, use `resolve_identifier`
    """
    errors: dict[SnowparkIdentifierRule, list[Exception]] = {
        SnowparkIdentifierRule.INFERRED: [],
        SnowparkIdentifierRule.NORMALIZED: [],
    }
    schema = data.schema
    if strict:
        values_range = _get_dataframe_values_range(data)
    else:
        values_range = {}
    for identifier_rule in errors.keys():
        for feature in features:
            try:
                ft_name = identifier_rule.get_identifier_from_feature(feature.name)
            except ValueError as e:
                errors[identifier_rule].append(e)
                continue
            found = False
            for field in schema.fields:
                if field.name == ft_name:
                    found = True
                    if isinstance(feature, core.FeatureGroupSpec):
                        if not isinstance(field.datatype, (spt.ArrayType, spt.StructType, spt.VariantType)):
                            errors[identifier_rule].append(
                                ValueError(
                                    f"Data Validation Error in feature group {feature.name}: "
                                    + f"Feature expects {feature.as_snowpark_type()},"
                                    + f" while {field.name} has type {field.datatype}."
                                ),
                            )
                        continue
                    assert isinstance(feature, core.FeatureSpec)  # mypy
                    ft_type = feature._dtype
                    field_data_type = field.datatype
                    if isinstance(field_data_type, spt.ArrayType):
                        if feature._shape is None:
                            errors[identifier_rule].append(
                                ValueError(
                                    f"Data Validation Error in feature {feature.name}: "
                                    + f"Feature is a scalar feature, while {field.name} is not."
                                ),
                            )
                        warnings.warn(
                            (f"Feature {feature.name} type cannot be validated: feature is an array feature."),
                            category=RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        if feature._shape:
                            errors[identifier_rule].append(
                                ValueError(
                                    f"Data Validation Error in feature {feature.name}: "
                                    + f"Feature is an array feature, while {field.name} is not."
                                ),
                            )
                            continue
                        try:
                            _validate_snowpark_type_feature(
                                data, field, ft_type, feature.name, values_range.get(field.name, None), strict=strict
                            )
                        except snowml_exceptions.SnowflakeMLException as e:
                            errors[identifier_rule].append(e.original_exception)
                    break
            if not found:
                errors[identifier_rule].append(
                    ValueError(f"Data Validation Error: feature {feature.name} does not exist in data."),
                )
    if all(len(error_list) != 0 for error_list in errors.values()):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_DATA,
            original_exception=ValueError(
                f"""
Data Validation Error when validating your Snowpark DataFrame.
If using the normalized names from model signatures, there are the following errors:
{errors[SnowparkIdentifierRule.NORMALIZED]}

If using the inferred names from model signatures, there are the following errors:
{errors[SnowparkIdentifierRule.INFERRED]}
"""
            ),
        )
    else:
        return (
            SnowparkIdentifierRule.INFERRED
            if len(errors[SnowparkIdentifierRule.INFERRED]) == 0
            else SnowparkIdentifierRule.NORMALIZED
        )


def _validate_snowpark_type_feature(
    df: snowflake.snowpark.DataFrame,
    field: spt.StructField,
    ft_type: DataType,
    ft_name: str,
    value_range: Optional[Union[tuple[int, int], tuple[float, float]]],
    strict: bool = False,
) -> None:
    field_data_type = field.datatype
    col_name = identifier.get_unescaped_names(field.name)

    if ft_type in [
        core.DataType.INT8,
        core.DataType.INT16,
        core.DataType.INT32,
        core.DataType.INT64,
        core.DataType.UINT8,
        core.DataType.UINT16,
        core.DataType.UINT32,
        core.DataType.UINT64,
    ]:
        if not (
            isinstance(field_data_type, spt._IntegralType)
            or (isinstance(field_data_type, spt.DecimalType) and field_data_type.scale == 0)
        ):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name} "
                    f"because of its original type {field_data_type}"
                ),
            )
        if not strict:
            return
        if value_range is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name} "
                    f"because of its original type {field_data_type} is non-Numeric."
                ),
            )
        min_v, max_v = value_range
        if max_v > np.iinfo(ft_type._numpy_type).max or min_v < np.iinfo(ft_type._numpy_type).min:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name} "
                    f"because it overflows with min or max"
                ),
            )
    elif ft_type in [core.DataType.FLOAT, core.DataType.DOUBLE]:
        if not (
            isinstance(
                field_data_type,
                (spt._IntegralType, spt.FloatType, spt.DoubleType, spt.DecimalType),
            )
        ):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    + f"Feature type {ft_type} is not met by column {col_name}."
                ),
            )
        if isinstance(field_data_type, spt.DecimalType) and field_data_type.scale > 0:
            warnings.warn(
                (
                    f"Type {field_data_type} is being automatically converted to DOUBLE in the Snowpark DataFrame. "
                    "This automatic conversion may lead to potential precision loss and rounding errors. "
                    "If you wish to prevent this conversion, you should manually perform "
                    "the necessary data type conversion."
                ),
                category=UserWarning,
                stacklevel=2,
            )

        if not strict:
            return
        if value_range is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name} "
                    f"because of its original type {field_data_type} is non-Numeric."
                ),
            )
        min_v, max_v = value_range
        if (
            max_v > np.finfo(ft_type._numpy_type).max  # type: ignore[arg-type]
            or min_v < np.finfo(ft_type._numpy_type).min  # type: ignore[arg-type]
        ):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name}."
                    f"because it overflows with min or max"
                ),
            )
    else:
        if not (isinstance(field_data_type, ft_type._snowpark_type)):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    f"Feature type {ft_type} is not met by column {col_name}."
                ),
            )


def _convert_local_data_to_df(
    data: type_hints.SupportedLocalDataType, ensure_serializable: bool = False
) -> pd.DataFrame:
    """Convert local data to pandas DataFrame or Snowpark DataFrame

    Args:
        data: The provided data.
        ensure_serializable: Ensure the data is serializable. Defaults to False.

    Raises:
        SnowflakeMLException: NotImplementedError: Raised when data cannot be handled by any data handler.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = None
    for handler in _LOCAL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            df = handler.convert_to_df(data, ensure_serializable=ensure_serializable)
            break
    if df is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_IMPLEMENTED,
            original_exception=NotImplementedError(f"Data Validation Error: Un-supported type {type(data)} provided."),
        )

    return df


def _convert_and_validate_local_data(
    data: type_hints.SupportedLocalDataType, features: Sequence[core.BaseFeatureSpec], strict: bool = False
) -> pd.DataFrame:
    """Validate the data with features in model signature and convert to DataFrame

    Args:
        features: A list of feature specs that the data should follow.
        data: The provided data.
        strict: Enable strict validation.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = _convert_local_data_to_df(data)
    df = utils.rename_pandas_df(df, features)
    _validate_pandas_df(df, features, strict=strict)

    return df


@telemetry.send_api_usage_telemetry(
    project=_TELEMETRY_PROJECT,
    subproject=_MODEL_TELEMETRY_SUBPROJECT,
)
def infer_signature(
    input_data: type_hints.SupportedLocalDataType,
    output_data: type_hints.SupportedLocalDataType,
    input_feature_names: Optional[list[str]] = None,
    output_feature_names: Optional[list[str]] = None,
    input_data_limit: Optional[int] = 100,
    output_data_limit: Optional[int] = 100,
) -> core.ModelSignature:
    """
    Infer model signature from given input and output sample data.

    Currently supports inferring model signatures from the following data types:

        - Pandas DataFrame with columns of supported data types, lists (including nested lists) of supported data types,
            and NumPy arrays of supported data types.
            - Does not support DataFrame with CategoricalIndex column index.
        - NumPy arrays of supported data types.
        - Lists of NumPy arrays of supported data types.
        - Lists of supported data types or nested lists of supported data types.

    When inferring the signature, a ValueError indicates that the data is insufficient or invalid.

    When it might be possible to create a signature reflecting the provided data, but it could not be inferred,
    a NotImplementedError is raised

    Args:
        input_data: Sample input data for the model.
        output_data: Sample output data for the model.
        input_feature_names: Names for input features. Defaults to None.
        output_feature_names: Names for output features. Defaults to None.
        input_data_limit: Limit the number of rows to be used in signature inference in the input data. Defaults to 100.
            If None, all rows are used. If the number of rows in the input data is less than the limit, all rows are
            used.
        output_data_limit: Limit the number of rows to be used in signature inference in the output data. Defaults to
            100. If None, all rows are used. If the number of rows in the output data is less than the limit, all rows
            are used.

    Returns:
        A model signature inferred from the given input and output sample data.
    """
    inputs = _infer_signature(_truncate_data(input_data, input_data_limit), role="input")
    inputs = utils.rename_features(inputs, input_feature_names)
    outputs = _infer_signature(_truncate_data(output_data, output_data_limit), role="output")
    outputs = utils.rename_features(outputs, output_feature_names)
    return core.ModelSignature(inputs, outputs)
