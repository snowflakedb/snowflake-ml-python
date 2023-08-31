import json
from typing import List, Literal, Optional, Sequence, cast

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard

import snowflake.snowpark
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
            if isinstance(data_type, spt.ArrayType):
                actual_data_type = data_type.element_type
            else:
                actual_data_type = data_type
            if not any(type.is_same_snowpark_type(actual_data_type) for type in core.DataType):
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
        features: List[core.BaseFeatureSpec] = []
        schema = data.schema
        for field in schema.fields:
            name = identifier.get_unescaped_names(field.name)
            if isinstance(field.datatype, spt.ArrayType):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_IMPLEMENTED,
                    original_exception=NotImplementedError(
                        "Cannot infer model signature from Snowpark DataFrame with Array Type."
                    ),
                )
            else:
                features.append(core.FeatureSpec(name=name, dtype=core.DataType.from_snowpark_type(field.datatype)))
        return features

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
        # Snowpark ignore the schema argument when providing a pandas DataFrame.
        # However, in this case, if a cell of the original Dataframe is some array type,
        # they will be inferred as VARIANT.
        # To make sure Snowpark get the correct schema, we have to provide in a list of records.
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
        schema_list = []
        for feature in features:
            if isinstance(feature, core.FeatureGroupSpec):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_IMPLEMENTED,
                    original_exception=NotImplementedError("FeatureGroupSpec is not supported."),
                )
            assert isinstance(feature, core.FeatureSpec), "Invalid feature kind."
            schema_list.append(
                spt.StructField(
                    identifier.get_inferred_name(feature.name),
                    feature.as_snowpark_type(),
                    nullable=df[feature.name].isnull().any(),
                )
            )

        data = df.rename(columns=identifier.get_inferred_name).to_dict("records")
        if keep_order:
            for idx, data_item in enumerate(data):
                data_item[infer_template._KEEP_ORDER_COL_NAME] = idx
            schema_list.append(spt.StructField(infer_template._KEEP_ORDER_COL_NAME, spt.LongType(), nullable=False))
        sp_df = session.create_dataframe(
            data,  # To make sure the schema can be used, otherwise, array will become variant.
            spt.StructType(schema_list),
        )
        return sp_df
