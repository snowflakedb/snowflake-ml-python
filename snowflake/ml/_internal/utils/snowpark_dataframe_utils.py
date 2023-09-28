import logging

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
