from collections import abc
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml._internal import type_utils
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core

if TYPE_CHECKING:
    import tensorflow


class SeqOfTensorflowTensorHandler(
    base_handler.BaseDataHandler[Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]]
):
    @staticmethod
    def can_handle(
        data: model_types.SupportedDataType,
    ) -> TypeGuard[Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]]:
        if not isinstance(data, abc.Sequence):
            return False
        if len(data) == 0:
            return False
        if type_utils.LazyType("tensorflow.Tensor").isinstance(data[0]) or type_utils.LazyType(
            "tensorflow.Variable"
        ).isinstance(data[0]):
            return all(
                type_utils.LazyType("tensorflow.Tensor").isinstance(data_col)
                or type_utils.LazyType("tensorflow.Variable").isinstance(data_col)
                for data_col in data
            )
        return False

    @staticmethod
    def count(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> int:
        import tensorflow as tf

        rows = []
        for data_col in data:
            shapes = data_col.shape.as_list()
            if data_col.shape == tf.TensorShape(None) or (not shapes) or (shapes[0] is None):
                # Unknown shape array
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Unknown shape data is found."),
                )
            # Make mypy happy
            assert isinstance(shapes[0], int)

            rows.append(shapes[0])

        return min(rows)

    @staticmethod
    def truncate(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]
    ) -> Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        return [
            data_col[
                : min(SeqOfTensorflowTensorHandler.count(data), SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)
            ]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> None:
        import tensorflow as tf

        for data_col in data:
            if data_col.shape == tf.TensorShape(None) or any(dim is None for dim in data_col.shape.as_list()):
                # Unknown shape array
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Unknown shape data is found."),
                )

            if data_col.shape == tf.TensorShape([0]):
                # Empty array
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Empty data is found."),
                )

            if data_col.shape == tf.TensorShape([1]) or data_col.shape == tf.TensorShape([]):
                # scalar
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Scalar data is found."),
                )

    @staticmethod
    def infer_signature(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{SeqOfTensorflowTensorHandler.FEATURE_PREFIX}_"
        features: List[core.BaseFeatureSpec] = []
        role_prefix = (
            SeqOfTensorflowTensorHandler.INPUT_PREFIX if role == "input" else SeqOfTensorflowTensorHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = core.DataType.from_numpy_type(data_col.dtype.as_numpy_dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], ensure_serializable: bool = True
    ) -> pd.DataFrame:
        if ensure_serializable:
            return pd.DataFrame({i: data_col.numpy().tolist() for i, data_col in enumerate(iterable=data)})
        return pd.DataFrame({i: list(data_col.numpy()) for i, data_col in enumerate(iterable=data)})

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[core.BaseFeatureSpec]] = None
    ) -> Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        import tensorflow as tf

        res = []
        if features:
            for feature in features:
                if isinstance(feature, core.FeatureGroupSpec):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.NOT_IMPLEMENTED,
                        original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                    )
                assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
                res.append(
                    tf.convert_to_tensor(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type))
                )
            return res
        return [tf.convert_to_tensor(np.stack(df[col].to_numpy())) for col in df]
