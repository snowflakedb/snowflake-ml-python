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
from snowflake.ml.model._signatures import base_handler, core

if TYPE_CHECKING:
    import xgboost


class XGBoostDMatrixHandler(base_handler.BaseDataHandler["xgboost.DMatrix"]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard["xgboost.DMatrix"]:
        return type_utils.LazyType("xgboost.DMatrix").isinstance(data)

    @staticmethod
    def count(data: "xgboost.DMatrix") -> int:
        return data.num_row()

    @staticmethod
    def truncate(data: "xgboost.DMatrix", length: int) -> "xgboost.DMatrix":

        num_rows = min(
            XGBoostDMatrixHandler.count(data),
            length,
        )
        return data.slice(list(range(num_rows)))

    @staticmethod
    def validate(data: "xgboost.DMatrix") -> None:
        if data.num_row() == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Data Validation Error: Empty data is found."),
            )

    @staticmethod
    def infer_signature(data: "xgboost.DMatrix", role: Literal["input", "output"]) -> Sequence[core.BaseFeatureSpec]:
        feature_prefix = f"{XGBoostDMatrixHandler.FEATURE_PREFIX}_"
        features: list[core.BaseFeatureSpec] = []
        role_prefix = (
            XGBoostDMatrixHandler.INPUT_PREFIX if role == "input" else XGBoostDMatrixHandler.OUTPUT_PREFIX
        ) + "_"

        feature_names = data.feature_names or []
        feature_types = data.feature_types or []

        for i, (feature_name, dtype) in enumerate(zip(feature_names, feature_types)):
            if not feature_name:
                ft_name = f"{role_prefix}{feature_prefix}{i}"
            else:
                ft_name = feature_name

            features.append(core.FeatureSpec(dtype=core.DataType.from_numpy_type(np.dtype(dtype)), name=ft_name))
        return features

    @staticmethod
    def convert_to_df(data: "xgboost.DMatrix", ensure_serializable: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(data.get_data().toarray(), columns=data.feature_names)

        feature_types = data.feature_types or []

        if feature_types:
            for idx, col in enumerate(df.columns):
                dtype = feature_types[idx]
                df[col] = df[col].astype(dtype)

        return df

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[core.BaseFeatureSpec]] = None
    ) -> "xgboost.DMatrix":
        import xgboost as xgb

        enable_categorical = False
        for col, d_type in df.dtypes.items():
            if pd.api.extensions.ExtensionDtype.is_dtype(d_type):
                continue
            if not np.issubdtype(d_type, np.number):
                df[col] = df[col].astype("category")
                enable_categorical = True

        if not features:
            return xgb.DMatrix(df, enable_categorical=enable_categorical)
        else:
            feature_names = []
            feature_types = []
            for feature in features:
                if isinstance(feature, core.FeatureGroupSpec):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.NOT_IMPLEMENTED,
                        original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                    )
                assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
                feature_names.append(feature.name)
                feature_types.append(feature._dtype._numpy_type)
            return xgb.DMatrix(
                df,
                feature_names=feature_names,
                feature_types=feature_types,
                enable_categorical=enable_categorical,
            )
