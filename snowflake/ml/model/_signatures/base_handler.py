from abc import ABC, abstractmethod
from typing import Final, Generic, Literal, Sequence

import pandas as pd
from typing_extensions import TypeGuard

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import core


class BaseDataHandler(ABC, Generic[model_types._DataType]):
    FEATURE_PREFIX: Final[str] = "feature"
    INPUT_PREFIX: Final[str] = "input"
    OUTPUT_PREFIX: Final[str] = "output"

    @staticmethod
    @abstractmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._DataType]:
        ...

    @staticmethod
    @abstractmethod
    def count(data: model_types._DataType) -> int:
        ...

    @staticmethod
    @abstractmethod
    def truncate(data: model_types._DataType, length: int) -> model_types._DataType:
        ...

    @staticmethod
    @abstractmethod
    def validate(data: model_types._DataType) -> None:
        ...

    @staticmethod
    @abstractmethod
    def infer_signature(
        data: model_types._DataType, role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        ...

    @staticmethod
    @abstractmethod
    def convert_to_df(data: model_types._DataType, ensure_serializable: bool = True) -> pd.DataFrame:
        ...
