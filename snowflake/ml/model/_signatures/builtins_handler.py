import datetime
from collections import abc
from typing import Literal, Sequence

import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core, pandas_handler


class ListOfBuiltinHandler(base_handler.BaseDataHandler[model_types._SupportedBuiltinsList]):
    @staticmethod
    def _can_handle_element(
        element: model_types._SupportedBuiltins,
    ) -> TypeGuard[model_types._SupportedBuiltins]:
        if isinstance(element, abc.Sequence) and not isinstance(element, str):
            for sub_element in element:
                if not ListOfBuiltinHandler._can_handle_element(sub_element):
                    return False
            return True
        elif isinstance(element, abc.Mapping):
            for key, value in element.items():
                if not isinstance(key, str):
                    return False
                if not ListOfBuiltinHandler._can_handle_element(value):
                    return False
            return True
        else:
            return isinstance(element, (int, float, bool, str, datetime.datetime))

    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._SupportedBuiltinsList]:
        if not isinstance(data, abc.Sequence) or isinstance(data, str):
            return False
        if len(data) == 0:
            return False
        return ListOfBuiltinHandler._can_handle_element(data)

    @staticmethod
    def count(data: model_types._SupportedBuiltinsList) -> int:
        return len(data)

    @staticmethod
    def truncate(data: model_types._SupportedBuiltinsList, length: int) -> model_types._SupportedBuiltinsList:
        return data[: min(ListOfBuiltinHandler.count(data), length)]

    @staticmethod
    def validate(data: model_types._SupportedBuiltinsList) -> None:
        if not all(isinstance(data_row, type(data[0])) for data_row in data):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Data Validation Error: Inconsistent type of object found in data {data}."
                ),
            )
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(f"Data Validation Error: Ill-shaped list data {data} confronted."),
            )

    @staticmethod
    def infer_signature(
        data: model_types._SupportedBuiltinsList, role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        return pandas_handler.PandasDataFrameHandler.infer_signature(pd.DataFrame(data), role)

    @staticmethod
    def convert_to_df(
        data: model_types._SupportedBuiltinsList,
        ensure_serializable: bool = True,
    ) -> pd.DataFrame:
        return pd.DataFrame(data)
