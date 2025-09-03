import json
from typing import Any, Literal, Optional, Sequence, cast

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
from snowflake.ml.model._signatures import base_handler, core, pandas_handler

_KEEP_ORDER_COL_NAME = "_ID"


class SnowparkDataFrameHandler(base_handler.BaseDataHandler[snowflake.snowpark.DataFrame]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[snowflake.snowpark.DataFrame]:
        return isinstance(data, snowflake.snowpark.DataFrame)

    @staticmethod
    def count(data: snowflake.snowpark.DataFrame) -> int:
        return data.count()

    @staticmethod
    def truncate(data: snowflake.snowpark.DataFrame, length: int) -> snowflake.snowpark.DataFrame:
        return cast(snowflake.snowpark.DataFrame, data.limit(length))

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
            SnowparkDataFrameHandler.convert_to_df(data), role=role
        )

    @staticmethod
    def convert_to_df(
        data: snowflake.snowpark.DataFrame,
        ensure_serializable: bool = True,
        features: Optional[Sequence[core.BaseFeatureSpec]] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        # This method do things on top of to_pandas, to make sure the local dataframe got is in correct shape.
        dtype_map = {}

        if features:
            quoted_identifiers_ignore_case = SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                data.session, statement_params
            )
            for feature in features:
                feature_name = feature.name.upper() if quoted_identifiers_ignore_case else feature.name
                dtype_map[feature_name] = feature.as_dtype()

        df_local = data.to_pandas()

        # This is because Array will become string (Even though the correct schema is set)
        # and object will become variant type and requires an additional loads
        # to get correct data otherwise it would be string.
        def load_if_not_null(x: str) -> Optional[Any]:
            if x is None:
                return None
            return json.loads(x)

        for field in data.schema.fields:
            if isinstance(field.datatype, (spt.ArrayType, spt.MapType, spt.StructType)):
                df_local[identifier.get_unescaped_names(field.name)] = df_local[
                    identifier.get_unescaped_names(field.name)
                ].map(load_if_not_null)
        # Only when the feature is not from inference, we are confident to do the type casting.
        # Otherwise, dtype_map will be empty.
        # Errors are ignored to make sure None won't be converted and won't raise Error
        df_local = df_local.astype(dtype=dtype_map)
        return df_local

    @staticmethod
    def convert_from_df(
        session: snowflake.snowpark.Session,
        df: pd.DataFrame,
        keep_order: bool = False,
        features: Optional[Sequence[core.BaseFeatureSpec]] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> snowflake.snowpark.DataFrame:
        # This method is necessary to create the Snowpark Dataframe in correct schema.
        # However, in this case, the order could not be preserved. Thus, a _ID column has to be added,
        # if keep_order is True.
        # Although in this case, the column with array type can get correct ARRAY type, however, the element
        # type is not preserved, and will become string type. This affect the implementation of convert_from_df.
        df = pandas_handler.PandasDataFrameHandler.convert_to_df(df)
        quoted_identifiers_ignore_case = SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
            session, statement_params
        )
        if quoted_identifiers_ignore_case:
            df.columns = [str(col).upper() for col in df.columns]

        df_cols = df.columns
        if df_cols.dtype != np.object_:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=ValueError("Cannot convert a Pandas DataFrame whose column index is not a string"),
            )
        if not features:
            features = pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input")
        # Role will be no effect on the column index. That is to say, the feature name is the actual column name.
        if keep_order:
            df = df.reset_index(drop=True)
            df[_KEEP_ORDER_COL_NAME] = df.index
        sp_df = session.create_dataframe(df)
        column_names = []
        columns = []
        for feature in features:
            feature_name = identifier.get_inferred_name(feature.name)
            if quoted_identifiers_ignore_case:
                feature_name = feature_name.upper()
            column_names.append(feature_name)
            columns.append(F.col(feature_name).cast(feature.as_snowpark_type()))

        sp_df = sp_df.with_columns(column_names, columns)

        return sp_df

    @staticmethod
    def _is_quoted_identifiers_ignore_case_enabled(
        session: snowflake.snowpark.Session, statement_params: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        Check if QUOTED_IDENTIFIERS_IGNORE_CASE parameter is enabled.

        Args:
            session: Snowpark session to check parameter for
            statement_params: Optional statement parameters to check first

        Returns:
            bool: True if QUOTED_IDENTIFIERS_IGNORE_CASE is enabled, False otherwise
            Returns False if the parameter cannot be retrieved (e.g., in stored procedures)
        """
        if statement_params is not None:
            for key, value in statement_params.items():
                if key.upper() == "QUOTED_IDENTIFIERS_IGNORE_CASE":
                    parameter_value = str(value)
                    return parameter_value.lower() == "true"

        try:
            result = session.sql(
                "SHOW PARAMETERS LIKE 'QUOTED_IDENTIFIERS_IGNORE_CASE' IN SESSION",
                _emit_ast=False,
            ).collect(_emit_ast=False)

            parameter_value = str(result[0].value)
            return parameter_value.lower() == "true"

        except Exception:
            # Parameter query can fail in certain environments (e.g., in stored procedures)
            # In that case, assume default behavior (case-sensitive)
            return False
