from collections import abc
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence

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
    import torch


class SeqOfPyTorchTensorHandler(base_handler.BaseDataHandler[Sequence["torch.Tensor"]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Sequence["torch.Tensor"]]:
        if not isinstance(data, abc.Sequence):
            return False
        if len(data) == 0:
            return False
        if type_utils.LazyType("torch.Tensor").isinstance(data[0]):
            return all(type_utils.LazyType("torch.Tensor").isinstance(data_col) for data_col in data)
        return False

    @staticmethod
    def count(data: Sequence["torch.Tensor"]) -> int:
        return min(data_col.shape[0] for data_col in data)

    @staticmethod
    def truncate(data: Sequence["torch.Tensor"]) -> Sequence["torch.Tensor"]:
        return [
            data_col[: min(SeqOfPyTorchTensorHandler.count(data), SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence["torch.Tensor"]) -> None:
        import torch

        for data_col in data:
            if data_col.shape == torch.Size([0]):
                # Empty array
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Empty data is found."),
                )

            if data_col.shape == torch.Size([1]):
                # scalar
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError("Data Validation Error: Scalar data is found."),
                )

    @staticmethod
    def infer_signature(
        data: Sequence["torch.Tensor"], role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{SeqOfPyTorchTensorHandler.FEATURE_PREFIX}_"
        features: List[core.BaseFeatureSpec] = []
        role_prefix = (
            SeqOfPyTorchTensorHandler.INPUT_PREFIX if role == "input" else SeqOfPyTorchTensorHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = core.DataType.from_torch_type(data_col.dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(core.FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(data: Sequence["torch.Tensor"], ensure_serializable: bool = True) -> pd.DataFrame:
        # Use list(...) instead of .tolist() to ensure that
        # the content is still numpy array so that the type could be preserved.
        # But that would not serializable and cannot use as UDF input and output.
        if ensure_serializable:
            return pd.DataFrame({i: data_col.detach().to("cpu").numpy().tolist() for i, data_col in enumerate(data)})
        return pd.DataFrame({i: list(data_col.detach().to("cpu").numpy()) for i, data_col in enumerate(data)})

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
                        error_code=error_codes.NOT_IMPLEMENTED,
                        original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                    )
                assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
                res.append(torch.from_numpy(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type)))
            return res
        return [torch.from_numpy(np.stack(df[col].to_numpy())) for col in df]
