from typing import Literal, Sequence

import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import base_handler, core, pandas_handler


class ListOfBuiltinHandler(base_handler.BaseDataHandler[model_types._SupportedBuiltinsList]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._SupportedBuiltinsList]:
        return (
            isinstance(data, list)
            and len(data) > 0
            and all(isinstance(data_col, (int, float, bool, str, bytes, list)) for data_col in data)
        )

    @staticmethod
    def count(data: model_types._SupportedBuiltinsList) -> int:
        return len(data)

    @staticmethod
    def truncate(data: model_types._SupportedBuiltinsList) -> model_types._SupportedBuiltinsList:
        return data[: min(ListOfBuiltinHandler.count(data), ListOfBuiltinHandler.SIG_INFER_ROWS_COUNT_LIMIT)]

    @staticmethod
    def validate(data: model_types._SupportedBuiltinsList) -> None:
        if not all(isinstance(data_row, type(data[0])) for data_row in data):
            raise ValueError(f"Data Validation Error: Inconsistent type of object found in data {data}.")
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            raise ValueError(f"Data Validation Error: Ill-shaped list data {data} confronted.")

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
