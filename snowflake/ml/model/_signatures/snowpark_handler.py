import json
from typing import Literal, Optional, Sequence, cast

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

import snowflake.snowpark
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as spt
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._deploy_client.warehouse import infer_template
from snowflake.ml.model._signatures import base_handler, core, pandas_handler


class SnowparkDataFrameHandler(base_handler.BaseDataHandler[snowflake.snowpark.DataFrame]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[snowflake.snowpark.DataFrame]:
        return isinstance(data, snowflake.snowpark.DataFrame)

    @staticmethod
    def count(data: snowflake.snowpark.DataFrame) -> int:
        return data.count()

    @staticmethod
    def truncate(data: snowflake.snowpark.DataFrame) -> snowflake.snowpark.DataFrame:
        return cast(snowflake.snowpark.DataFrame, data.limit(SnowparkDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT))

    @staticmethod
    def validate(data: snowflake.snowpark.DataFrame) -> None:
        schema = data.schema
        for field in schema.fields:
            data_type = field.datatype
            try:
                core.DataType.from_snowpark_type(data_type)
            except snowml_exceptions.SnowflakeMLException:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        f"Data Validation Error: Unsupported data type {field.datatype} in column {field.name}."
                    ),
                )

    @staticmethod
    def infer_signature(
        data: snowflake.snowpark.DataFrame, role: Literal["input", "output"]
    ) -> Sequence[core.BaseFeatureSpec]:
        return pandas_handler.PandasDataFrameHandler.infer_signature(
            SnowparkDataFrameHandler.convert_to_df(data.limit(n=1)), role=role
        )

    @staticmethod
    def convert_to_df(
        data: snowflake.snowpark.DataFrame,
        ensure_serializable: bool = True,
        features: Optional[Sequence[core.BaseFeatureSpec]] = None,
    ) -> pd.DataFrame:
        # This method do things on top of to_pandas, to make sure the local dataframe got is in correct shape.
        dtype_map = {}
        if features:
            for feature in features:
                if isinstance(feature, core.FeatureGroupSpec):
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.NOT_IMPLEMENTED,
                        original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                    )
                assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
                dtype_map[feature.name] = feature.as_dtype()
        df_local = data.to_pandas()
        # This is because Array will become string (Even though the correct schema is set)
        # and object will become variant type and requires an additional loads
        # to get correct data otherwise it would be string.
        for field in data.schema.fields:
            if isinstance(field.datatype, spt.ArrayType):
                df_local[identifier.get_unescaped_names(field.name)] = df_local[
                    identifier.get_unescaped_names(field.name)
                ].map(json.loads)
        # Only when the feature is not from inference, we are confident to do the type casting.
        # Otherwise, dtype_map will be empty
        df_local = df_local.astype(dtype=dtype_map)
        return df_local

    @staticmethod
    def convert_from_df(
        session: snowflake.snowpark.Session, df: pd.DataFrame, keep_order: bool = False
    ) -> snowflake.snowpark.DataFrame:
        # This method is necessary to create the Snowpark Dataframe in correct schema.
        # However, in this case, the order could not be preserved. Thus, a _ID column has to be added,
        # if keep_order is True.
        # Although in this case, the column with array type can get correct ARRAY type, however, the element
        # type is not preserved, and will become string type. This affect the implementation of convert_from_df.
        df = pandas_handler.PandasDataFrameHandler.convert_to_df(df)
        df_cols = df.columns
        if df_cols.dtype != np.object_:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=ValueError("Cannot convert a Pandas DataFrame whose column index is not a string"),
            )
        features = pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input")
        # Role will be no effect on the column index. That is to say, the feature name is the actual column name.
        sp_df = session.create_dataframe(df)
        column_names = []
        columns = []
        for feature in features:
            if isinstance(feature, core.FeatureGroupSpec):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_IMPLEMENTED,
                    original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                )
            assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
            column_names.append(identifier.get_inferred_name(feature.name))
            columns.append(F.col(identifier.get_inferred_name(feature.name)).cast(feature.as_snowpark_type()))

        sp_df = sp_df.with_columns(column_names, columns)

        if keep_order:
            sp_df = sp_df.with_column(infer_template._KEEP_ORDER_COL_NAME, F.monotonically_increasing_id())

        return sp_df
