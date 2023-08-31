from collections import abc
from typing import List, Literal, Sequence

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core


class NumpyArrayHandler(base_handler.BaseDataHandler[model_types._SupportedNumpyArray]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._SupportedNumpyArray]:
        return isinstance(data, np.ndarray)

    @staticmethod
    def count(data: model_types._SupportedNumpyArray) -> int:
        return data.shape[0]

    @staticmethod
    def truncate(data: model_types._SupportedNumpyArray) -> model_types._SupportedNumpyArray:
        return data[: min(NumpyArrayHandler.count(data), NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)]

    @staticmethod
    def validate(data: model_types._SupportedNumpyArray) -> None:
        if data.shape == (0,):
            # Empty array
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Empty data is found."),
            )

        if data.shape == ():
            # scalar
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Scalar data is found."),
            )

    @staticmethod
    def infer_signature(
        data: model_types._SupportedNumpyArray, role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{NumpyArrayHandler.FEATURE_PREFIX}_"
        dtype = core.DataType.from_numpy_type(data.dtype)
        role_prefix = (NumpyArrayHandler.INPUT_PREFIX if role == "input" else NumpyArrayHandler.OUTPUT_PREFIX) + "_"
        if len(data.shape) == 1:
            return [core.FeatureSpec(dtype=dtype, name=f"{role_prefix}{feature_prefix}0")]
        else:
            # For high-dimension array, 0-axis is for batch, 1-axis is for column, further more is details of columns.
            features = []
            n_cols = data.shape[1]
            ft_names = [f"{role_prefix}{feature_prefix}{i}" for i in range(n_cols)]
            for col_data, ft_name in zip(data[0], ft_names):
                if isinstance(col_data, np.ndarray):
                    ft_shape = np.shape(col_data)
                    features.append(core.FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
                else:
                    features.append(core.FeatureSpec(dtype=dtype, name=ft_name))
            return features

    @staticmethod
    def convert_to_df(data: model_types._SupportedNumpyArray, ensure_serializable: bool = True) -> pd.DataFrame:
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        n_cols = data.shape[1]
        if len(data.shape) == 2:
            return pd.DataFrame(data)
        else:
            n_rows = data.shape[0]
            if ensure_serializable:
                return pd.DataFrame(data={i: [data[k, i].tolist() for k in range(n_rows)] for i in range(n_cols)})
            return pd.DataFrame(data={i: [list(data[k, i]) for k in range(n_rows)] for i in range(n_cols)})


class SeqOfNumpyArrayHandler(base_handler.BaseDataHandler[Sequence[model_types._SupportedNumpyArray]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Sequence[model_types._SupportedNumpyArray]]:
        if not isinstance(data, abc.Sequence):
            return False
        if len(data) == 0:
            return False
        if isinstance(data[0], np.ndarray):
            return all(isinstance(data_col, np.ndarray) for data_col in data)
        return False

    @staticmethod
    def count(data: Sequence[model_types._SupportedNumpyArray]) -> int:
        return min(NumpyArrayHandler.count(data_col) for data_col in data)

    @staticmethod
    def truncate(data: Sequence[model_types._SupportedNumpyArray]) -> Sequence[model_types._SupportedNumpyArray]:
        return [
            data_col[: min(SeqOfNumpyArrayHandler.count(data), SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence[model_types._SupportedNumpyArray]) -> None:
        for data_col in data:
            NumpyArrayHandler.validate(data_col)

    @staticmethod
    def infer_signature(
        data: Sequence[model_types._SupportedNumpyArray], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{SeqOfNumpyArrayHandler.FEATURE_PREFIX}_"
        features: List[core.BaseFeatureSpec] = []
        role_prefix = (
            SeqOfNumpyArrayHandler.INPUT_PREFIX if role == "input" else SeqOfNumpyArrayHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = core.DataType.from_numpy_type(data_col.dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(
        data: Sequence[model_types._SupportedNumpyArray], ensure_serializable: bool = True
    ) -> pd.DataFrame:
        if ensure_serializable:
            return pd.DataFrame(data={i: data_col.tolist() for i, data_col in enumerate(data)})
        return pd.DataFrame(data={i: list(data_col) for i, data_col in enumerate(data)})
