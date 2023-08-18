from typing import Literal, Sequence

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
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[pd.DataFrame]:
        return isinstance(data, pd.DataFrame)

    @staticmethod
    def count(data: pd.DataFrame) -> int:
        return len(data.index)

    @staticmethod
    def truncate(data: pd.DataFrame) -> pd.DataFrame:
        return data.head(min(PandasDataFrameHandler.count(data), PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT))

    @staticmethod
    def validate(data: pd.DataFrame) -> None:
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
            if df_col_dtype == np.dtype("O"):
                # Check if all objects have the same type
                if not all(isinstance(data_row, type(data[df_col][0])) for data_row in data[df_col]):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error: Inconsistent type of object found in column data {data[df_col]}."
                        ),
                    )

                if isinstance(data[df_col][0], list):
                    arr = utils.convert_list_to_ndarray(data[df_col][0])
                    arr_dtype = core.DataType.from_numpy_type(arr.dtype)

                    converted_data_list = [utils.convert_list_to_ndarray(data_row) for data_row in data[df_col]]

                    if not all(
                        core.DataType.from_numpy_type(converted_data.dtype) == arr_dtype
                        for converted_data in converted_data_list
                    ):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                "Data Validation Error: "
                                + f"Inconsistent type of element in object found in column data {data[df_col]}."
                            ),
                        )

                elif isinstance(data[df_col][0], np.ndarray):
                    arr_dtype = core.DataType.from_numpy_type(data[df_col][0].dtype)

                    if not all(core.DataType.from_numpy_type(data_row.dtype) == arr_dtype for data_row in data[df_col]):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_DATA,
                            original_exception=ValueError(
                                "Data Validation Error: "
                                + f"Inconsistent type of element in object found in column data {data[df_col]}."
                            ),
                        )
                elif not isinstance(data[df_col][0], (str, bytes)):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Data Validation Error: Unsupported type confronted in {data[df_col]}"
                        ),
                    )

    @staticmethod
    def infer_signature(data: pd.DataFrame, role: Literal["input", "output"]) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{PandasDataFrameHandler.FEATURE_PREFIX}_"
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
            if df_col_dtype == np.dtype("O"):
                if isinstance(data[df_col][0], list):
                    arr = utils.convert_list_to_ndarray(data[df_col][0])
                    arr_dtype = core.DataType.from_numpy_type(arr.dtype)
                    ft_shape = np.shape(data[df_col][0])

                    converted_data_list = [utils.convert_list_to_ndarray(data_row) for data_row in data[df_col]]

                    if not all(np.shape(converted_data) == ft_shape for converted_data in converted_data_list):
                        ft_shape = (-1,)

                    specs.append(core.FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
                elif isinstance(data[df_col][0], np.ndarray):
                    arr_dtype = core.DataType.from_numpy_type(data[df_col][0].dtype)
                    ft_shape = np.shape(data[df_col][0])

                    if not all(np.shape(data_row) == ft_shape for data_row in data[df_col]):
                        ft_shape = (-1,)

                    specs.append(core.FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
                elif isinstance(data[df_col][0], str):
                    specs.append(core.FeatureSpec(dtype=core.DataType.STRING, name=ft_name))
                elif isinstance(data[df_col][0], bytes):
                    specs.append(core.FeatureSpec(dtype=core.DataType.BYTES, name=ft_name))
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
                if isinstance(df[df_col][0], np.ndarray):
                    df[df_col] = df[df_col].map(np.ndarray.tolist)
        return df
