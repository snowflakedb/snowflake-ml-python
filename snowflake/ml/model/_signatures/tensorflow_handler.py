from collections import abc
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml._internal import type_utils
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core, numpy_handler

if TYPE_CHECKING:
    import tensorflow


class TensorflowTensorHandler(base_handler.BaseDataHandler[Union["tensorflow.Tensor", "tensorflow.Variable"]]):
    @staticmethod
    def can_handle(
        data: model_types.SupportedDataType,
    ) -> TypeGuard[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        return type_utils.LazyType("tensorflow.Tensor").isinstance(data) or type_utils.LazyType(
            "tensorflow.Variable"
        ).isinstance(data)

    @staticmethod
    def count(data: Union["tensorflow.Tensor", "tensorflow.Variable"]) -> int:
        return numpy_handler.NumpyArrayHandler.count(data.numpy())

    @staticmethod
    def truncate(
        data: Union["tensorflow.Tensor", "tensorflow.Variable"], length: int
    ) -> Union["tensorflow.Tensor", "tensorflow.Variable"]:
        return data[: min(TensorflowTensorHandler.count(data), length)]

    @staticmethod
    def validate(data: Union["tensorflow.Tensor", "tensorflow.Variable"]) -> None:
        numpy_handler.NumpyArrayHandler.validate(data.numpy())

    @staticmethod
    def infer_signature(
        data: Union["tensorflow.Tensor", "tensorflow.Variable"], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        return numpy_handler.NumpyArrayHandler.infer_signature(data.numpy(), role=role)

    @staticmethod
    def convert_to_df(
        data: Union["tensorflow.Tensor", "tensorflow.Variable"], ensure_serializable: bool = True
    ) -> pd.DataFrame:
        return numpy_handler.NumpyArrayHandler.convert_to_df(data.numpy(), ensure_serializable=ensure_serializable)

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[core.BaseFeatureSpec]] = None
    ) -> Union["tensorflow.Tensor", "tensorflow.Variable"]:
        import tensorflow as tf

        if features is None:
            if any(dtype == np.dtype("O") for dtype in df.dtypes):
                return tf.convert_to_tensor(np.array(df.to_numpy().tolist()))
            return tf.convert_to_tensor(df.to_numpy())

        assert isinstance(features[0], core.FeatureSpec)
        return tf.convert_to_tensor(np.array(df.to_numpy().tolist()), dtype=features[0]._dtype._numpy_type)


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

        return all(TensorflowTensorHandler.can_handle(data_col) for data_col in data)

    @staticmethod
    def count(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> int:
        return min(TensorflowTensorHandler.count(data_col) for data_col in data)

    @staticmethod
    def truncate(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], length: int
    ) -> Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        return [data_col[: min(SeqOfTensorflowTensorHandler.count(data), length)] for data_col in data]

    @staticmethod
    def validate(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> None:
        for data_col in data:
            TensorflowTensorHandler.validate(data_col)

    @staticmethod
    def infer_signature(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        return numpy_handler.SeqOfNumpyArrayHandler.infer_signature([data_col.numpy() for data_col in data], role=role)

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
                        error_code=error_codes.INVALID_DATA_TYPE,
                        original_exception=NotImplementedError(
                            "FeatureGroupSpec is not supported when converting to Tensorflow tensor."
                        ),
                    )
                assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
                res.append(
                    tf.convert_to_tensor(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type))
                )
            return res
        return [tf.convert_to_tensor(np.stack(df[col].to_numpy())) for col in df]
