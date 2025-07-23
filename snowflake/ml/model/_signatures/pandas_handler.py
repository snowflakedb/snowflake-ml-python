import warnings
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core, utils


class PandasDataFrameHandler(base_handler.BaseDataHandler[pd.DataFrame]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Union[pd.DataFrame, pd.Series]]:
        return isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)

    @staticmethod
    def count(data: pd.DataFrame) -> int:
        return len(data.index)

    @staticmethod
    def truncate(data: pd.DataFrame, length: int) -> pd.DataFrame:
        return data.head(min(PandasDataFrameHandler.count(data), length))

    @staticmethod
    def validate(data: Union[pd.DataFrame, pd.Series]) -> None:
        if isinstance(data, pd.Series):
            # check if the series is empty and throw error
            if data.empty:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Empty data is found."),
                )
            # convert the series to a dataframe
            data = data.to_frame()

        df_cols = data.columns

        if df_cols.has_duplicates:  # Rule out categorical index with duplicates
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Duplicate column index is found."),
            )

        if not all(hasattr(data[col], "dtype") for col in data.columns):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(f"Unknown column confronted in {data}"),
            )

        if len(df_cols) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Empty data is found."),
            )

        if df_cols.dtype not in [
            np.int64,
            np.uint64,
            np.float64,
            np.object_,
        ]:  # To keep compatibility with Pandas 2.x and 1.x
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Unsupported column index type is found."),
            )

        df_col_dtypes = [data[col].dtype for col in data.columns]
        for df_col, df_col_dtype in zip(df_cols, df_col_dtypes):
            df_col_data = data[df_col]
            if df_col_data.isnull().any():
                warnings.warn(
                    (
                        f"Null value detected in column {df_col}, model signature inference might not accurate, "
                        "or your prediction might fail if your model does not support null input. If this is not "
                        "expected, please check your input dataframe."
                    ),
                    category=UserWarning,
                    stacklevel=2,
                )

                df_col_data = utils.series_dropna(df_col_data)
                df_col_dtype = df_col_data.dtype

            if utils.check_if_series_is_empty(df_col_data):
                continue

            if df_col_dtype == np.dtype("O"):
                # Check if all objects have the same type
                if not all(isinstance(data_row, type(df_col_data.iloc[0])) for data_row in df_col_data):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            "Data Validation Error: "
                            + f"Inconsistent type of element in object found in column data {df_col_data}."
                        ),
                    )

                if isinstance(df_col_data.iloc[0], np.ndarray):
                    arr_dtype = core.DataType.from_numpy_type(df_col_data.iloc[0].dtype)

                    if not all(core.DataType.from_numpy_type(data_row.dtype) == arr_dtype for data_row in df_col_data):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                "Data Validation Error: "
                                + f"Inconsistent type of element in object found in column data {df_col_data}."
                            ),
                        )
                elif not isinstance(df_col_data.iloc[0], (str, bytes, dict, list)):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error: Unsupported type confronted in {df_col_data}"
                        ),
                    )

    @staticmethod
    def infer_signature(
        data: Union[pd.DataFrame, pd.Series],
        role: Literal["input", "output"],
    ) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{PandasDataFrameHandler.FEATURE_PREFIX}_"
        if isinstance(data, pd.Series):
            data = data.to_frame()
        df_cols = data.columns
        role_prefix = (
            PandasDataFrameHandler.INPUT_PREFIX if role == "input" else PandasDataFrameHandler.OUTPUT_PREFIX
        ) + "_"
        if df_cols.dtype in [np.int64, np.uint64, np.float64]:
            ft_names = [f"{role_prefix}{feature_prefix}{i}" for i in df_cols]
        else:
            ft_names = list(map(str, data.columns.to_list()))

        df_col_dtypes = [data[col].dtype for col in data.columns]

        specs = []
        for df_col, df_col_dtype, ft_name in zip(df_cols, df_col_dtypes, ft_names):
            df_col_data = data[df_col]

            if df_col_data.isnull().all():
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        "Data Validation Error: "
                        f"There is no non-null data in column {df_col} so the signature cannot be inferred."
                    ),
                )
            if df_col_data.isnull().any():
                df_col_data = utils.series_dropna(df_col_data)
            df_col_dtype = df_col_data.dtype

            if df_col_dtype == np.dtype("O"):
                if isinstance(df_col_data.iloc[0], list):
                    spec_0 = utils.infer_list(ft_name, df_col_data.iloc[0])
                    for i in range(1, len(df_col_data)):
                        spec = utils.infer_list(ft_name, df_col_data.iloc[i])
                        if spec._shape != spec_0._shape:
                            spec_0._shape = (-1,)
                            spec._shape = (-1,)
                        if spec != spec_0:
                            raise snowml_exceptions.SnowflakeMLException(
                                error_code=error_codes.INVALID_DATA,
                                original_exception=ValueError(
                                    "Unable to construct signature: "
                                    f"Ragged nested or Unsupported list-like data {df_col_data} confronted."
                                ),
                            )
                    specs.append(spec_0)
                elif isinstance(df_col_data.iloc[0], dict):
                    specs.append(utils.infer_dict(ft_name, df_col_data.iloc[0]))
                elif isinstance(df_col_data.iloc[0], np.ndarray):
                    arr_dtype = core.DataType.from_numpy_type(df_col_data.iloc[0].dtype)
                    ft_shape = np.shape(df_col_data.iloc[0])

                    if not all(np.shape(data_row) == ft_shape for data_row in df_col_data):
                        ft_shape = (-1,)

                    specs.append(core.FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
                elif isinstance(df_col_data.iloc[0], str):
                    specs.append(core.FeatureSpec(dtype=core.DataType.STRING, name=ft_name))
                elif isinstance(df_col_data.iloc[0], bytes):
                    specs.append(core.FeatureSpec(dtype=core.DataType.BYTES, name=ft_name))
            elif isinstance(df_col_dtype, pd.CategoricalDtype):
                category_dtype = df_col_dtype.categories.dtype
                if category_dtype == np.dtype("O"):
                    if isinstance(df_col_dtype.categories[0], str):
                        specs.append(core.FeatureSpec(dtype=core.DataType.STRING, name=ft_name))
                    elif isinstance(df_col_dtype.categories[0], bytes):
                        specs.append(core.FeatureSpec(dtype=core.DataType.BYTES, name=ft_name))
                    else:
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                f"Data Validation Error: Unsupported type confronted in {df_col_dtype.categories[0]}"
                            ),
                        )
                else:
                    specs.append(core.FeatureSpec(dtype=core.DataType.from_numpy_type(category_dtype), name=ft_name))
            elif isinstance(data[df_col].iloc[0], np.datetime64):
                specs.append(core.FeatureSpec(dtype=core.DataType.TIMESTAMP_NTZ, name=ft_name))
            else:
                specs.append(core.FeatureSpec(dtype=core.DataType.from_numpy_type(df_col_dtype), name=ft_name))
        return specs

    @staticmethod
    def convert_to_df(data: pd.DataFrame, ensure_serializable: bool = True) -> pd.DataFrame:
        if not ensure_serializable:
            return data
        # This convert is necessary since numpy dataframe cannot be correctly handled when provided as an element of
        # a list when creating Snowpark Dataframe.
        df = data.copy()
        df_cols = df.columns
        df_col_dtypes = [df[col].dtype for col in df.columns]
        for df_col, df_col_dtype in zip(df_cols, df_col_dtypes):
            if df_col_dtype == np.dtype("O"):
                if isinstance(df[df_col].iloc[0], np.ndarray):
                    df[df_col] = df[df_col].map(np.ndarray.tolist)
        return df
