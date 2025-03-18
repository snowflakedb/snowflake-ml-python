from collections import abc
from typing import TYPE_CHECKING, Literal, Optional, Sequence

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
    import torch


class PyTorchTensorHandler(base_handler.BaseDataHandler["torch.Tensor"]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard["torch.Tensor"]:
        return type_utils.LazyType("torch.Tensor").isinstance(data)

    @staticmethod
    def count(data: "torch.Tensor") -> int:
        return data.shape[0]

    @staticmethod
    def truncate(data: "torch.Tensor", length: int) -> "torch.Tensor":
        return data[: min(PyTorchTensorHandler.count(data), length)]

    @staticmethod
    def validate(data: "torch.Tensor") -> None:
        return numpy_handler.NumpyArrayHandler.validate(data.detach().cpu().numpy())

    @staticmethod
    def infer_signature(data: "torch.Tensor", role: Literal["input", "output"]) -> Sequence[core.BaseFeatureSpec]:
        return numpy_handler.NumpyArrayHandler.infer_signature(data.detach().cpu().numpy(), role=role)

    @staticmethod
    def convert_to_df(data: "torch.Tensor", ensure_serializable: bool = True) -> pd.DataFrame:
        return numpy_handler.NumpyArrayHandler.convert_to_df(
            data.detach().cpu().numpy(), ensure_serializable=ensure_serializable
        )

    @staticmethod
    def convert_from_df(df: pd.DataFrame, features: Optional[Sequence[core.BaseFeatureSpec]] = None) -> "torch.Tensor":
        import torch

        if features is None:
            if any(dtype == np.dtype("O") for dtype in df.dtypes):
                return torch.from_numpy(np.array(df.to_numpy().tolist()))
            return torch.from_numpy(df.to_numpy())

        assert isinstance(features[0], core.FeatureSpec)
        return torch.from_numpy(
            np.array(df.to_numpy().tolist(), dtype=features[0]._dtype._numpy_type),
        )


class SeqOfPyTorchTensorHandler(base_handler.BaseDataHandler[Sequence["torch.Tensor"]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Sequence["torch.Tensor"]]:
        if not isinstance(data, abc.Sequence):
            return False
        if len(data) == 0:
            return False
        return all(PyTorchTensorHandler.can_handle(data_col) for data_col in data)

    @staticmethod
    def count(data: Sequence["torch.Tensor"]) -> int:
        return min(PyTorchTensorHandler.count(data_col) for data_col in data)

    @staticmethod
    def truncate(data: Sequence["torch.Tensor"], length: int) -> Sequence["torch.Tensor"]:
        return [data_col[: min(SeqOfPyTorchTensorHandler.count(data), length)] for data_col in data]

    @staticmethod
    def validate(data: Sequence["torch.Tensor"]) -> None:
        for data_col in data:
            PyTorchTensorHandler.validate(data_col)

    @staticmethod
    def infer_signature(
        data: Sequence["torch.Tensor"], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        return numpy_handler.SeqOfNumpyArrayHandler.infer_signature(
            [data_col.detach().cpu().numpy() for data_col in data], role=role
        )

    @staticmethod
    def convert_to_df(data: Sequence["torch.Tensor"], ensure_serializable: bool = True) -> pd.DataFrame:
        # Use list(...) instead of .tolist() to ensure that
        # the content is still numpy array so that the type could be preserved.
        # But that would not serializable and cannot use as UDF input and output.
        if ensure_serializable:
            return pd.DataFrame({i: data_col.detach().cpu().numpy().tolist() for i, data_col in enumerate(data)})
        return pd.DataFrame({i: list(data_col.detach().cpu().numpy()) for i, data_col in enumerate(data)})

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[core.BaseFeatureSpec]] = None
    ) -> Sequence["torch.Tensor"]:
        import torch

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
                res.append(torch.from_numpy(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type)))
            return res
        return [torch.from_numpy(np.stack(df[col].to_numpy())) for col in df]
