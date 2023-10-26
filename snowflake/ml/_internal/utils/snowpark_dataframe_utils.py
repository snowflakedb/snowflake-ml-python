import logging
import warnings

from snowflake import snowpark
from snowflake.snowpark import functions, types


def cast_snowpark_dataframe(df: snowpark.DataFrame) -> snowpark.DataFrame:
    """Cast columns in the dataframe to types that are compatible with tensor.

    It assists FileSet.make() in performing implicit data casting.

    Args:
        df: A snowpark dataframe.

    Returns:
        A snowpark dataframe whose data type has been casted.
    """

    fields = df.schema.fields
    selected_cols = []
    for field in fields:
        src = field.column_identifier.quoted_name
        if isinstance(field.datatype, types.DecimalType):
            if field.datatype.scale:
                dest: types.DataType = types.FloatType()
            else:
                dest = types.LongType()
            selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
        elif isinstance(field.datatype, types.DoubleType):
            dest = types.FloatType()
            selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
        elif isinstance(field.datatype, types.ByteType):
            # Snowpark maps ByteType to BYTEINT, which will not do the casting job when unloading to parquet files.
            # We will use SMALLINT instead until this issue got fixed.
            # Investigate JIRA filed: SNOW-725041
            dest = types.ShortType()
            selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
        elif field.datatype in (types.ShortType(), types.IntegerType(), types.LongType()):
            dest = field.datatype
            selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
        else:
            if field.datatype in (types.DateType(), types.TimestampType(), types.TimeType()):
                logging.warning(
                    "A Column with DATE or TIMESTAMP data type detected. "
                    "It might not be able to get converted to tensors. "
                    "Please consider handle it in feature engineering."
                )
            elif (
                isinstance(field.datatype, types.ArrayType)
                or isinstance(field.datatype, types.MapType)
                or isinstance(field.datatype, types.VariantType)
            ):
                logging.warning(
                    "A Column with semi-structured data type (variant, array or object) was detected. "
                    "It might not be able to get converted to tensors. "
                    "Please consider handling it in feature engineering."
                )
            selected_cols.append(functions.col(src))
    df = df.select(selected_cols)
    return df


def cast_snowpark_dataframe_column_types(df: snowpark.DataFrame) -> snowpark.DataFrame:
    """Cast columns in the dataframe to types that are compatible with pandas DataFrame.

    It assists modeling API (fit, predict, ...) in performing implicit data casting.
    The reason for casting: snowpark dataframe would transform as pandas dataframe
        to compute within sproc.

    Args:
        df: A snowpark dataframe.

    Returns:
        A snowpark dataframe whose data type has been casted.
    """
    fields = df.schema.fields
    selected_cols = []
    for field in fields:
        src = field.column_identifier.quoted_name
        # Handle DecimalType: Numbers up to 38 digits, with an optional precision and scale
        # By default, precision is 38 and scale is 0 (i.e. NUMBER(38, 0))
        if isinstance(field.datatype, types.DecimalType):
            # If datatype has scale; convert into float/double type
            # In snowflake, DOUBLE is the same as FLOAT, provides precision up to 18.
            if field.datatype.scale:
                dest_dtype: types.DataType = types.DoubleType()
                warnings.warn(
                    f"Warning: The Decimal({field.datatype.precision}, {field.datatype.scale}) data type"
                    " is being automatically converted to DoubleType in the Snowpark DataFrame. "
                    "This automatic conversion may lead to potential precision loss and rounding errors. "
                    "If you wish to prevent this conversion, you should manually perform "
                    "the necessary data type conversion."
                )
            else:
                # IntegerType default as NUMBER(38, 0), but
                # snowpark dataframe would automatically transform to LongType in function `convert_sf_to_sp_type`
                # To align with snowpark, set all the decimal without scale as LongType
                dest_dtype = types.LongType()
                warnings.warn(
                    f"Warning: The Decimal({field.datatype.precision}, 0) data type"
                    " is being automatically converted to LongType in the Snowpark DataFrame. "
                    "This automatic conversion may lead to potential precision loss and rounding errors. "
                    "If you wish to prevent this conversion, you should manually perform "
                    "the necessary data type conversion."
                )
            selected_cols.append(functions.cast(functions.col(src), dest_dtype).alias(src))
        # TODO: add more type handling or error message
        else:
            selected_cols.append(functions.col(src))
    df = df.select(selected_cols)
    return df
