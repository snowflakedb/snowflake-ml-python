"""
A helper script cleans open taxi data (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
and store into snowflake database.

Download yellow trip data(2016 Jan): https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
Download wine data:
https://www.google.com/url?q=https://github.com/snowflakedb/snowml/blob/main/snowflake/ml/feature_store/notebooks/customer_demo/winequality-red.csv&sa=D&source=docs&ust=1702084016573738&usg=AOvVaw3r_muH0_LKBDr45C1Gj3cb
"""

from absl.logging import logging

from snowflake.ml._internal.utils import identifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark.types import FloatType, IntegerType, StructField, StructType

# TODO these global parameters should be shared with those defined in feature_store/tests/common_utils.py
FS_INTEG_TEST_DB = "SNOWML_FEATURE_STORE_TEST_DB"
FS_INTEG_TEST_DATASET_SCHEMA = "TEST_DATASET"
FS_INTEG_TEST_YELLOW_TRIP_DATA = "yellow_tripdata_2016_01"
FS_INTEG_TEST_WINE_QUALITY_DATA = "wine_quality_data"


TRIPDATA_NAME = "yellow_tripdata_2016-01.parquet"
WINEDATA_NAME = "winequality-red.csv"
FILE_LOCAL_PATH = "file://~/Downloads/"

logger = logging.getLogger(__name__)


def create_tripdata(sess: Session, overwrite_mode: str) -> None:
    sess.file.put(f"{FILE_LOCAL_PATH}/{TRIPDATA_NAME}", sess.get_session_stage())
    df = sess.read.parquet(f"{sess.get_session_stage()}/{TRIPDATA_NAME}")
    for old_col_name in df.columns:
        df = df.with_column_renamed(old_col_name, identifier.get_unescaped_names(old_col_name))

    full_table_name = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{FS_INTEG_TEST_YELLOW_TRIP_DATA}"
    df.write.mode(overwrite_mode).save_as_table(full_table_name)
    rows_count = sess.sql(f"SELECT COUNT(*) FROM {full_table_name}").collect()[0][0]

    logger.info(f"{full_table_name} has total {rows_count} rows.")


def create_winedata(sess: Session, overwrite_mode: str) -> None:
    sess.file.put(f"{FILE_LOCAL_PATH}/{WINEDATA_NAME}", sess.get_session_stage())
    input_schema = StructType(
        [
            StructField("fixed_acidity", FloatType()),
            StructField("volatile_acidity", FloatType()),
            StructField("citric_acid", FloatType()),
            StructField("residual_sugar", FloatType()),
            StructField("chlorides", FloatType()),
            StructField("free_sulfur_dioxide", IntegerType()),
            StructField("total_sulfur_dioxide", IntegerType()),
            StructField("density", FloatType()),
            StructField("pH", FloatType()),
            StructField("sulphates", FloatType()),
            StructField("alcohol", FloatType()),
            StructField("quality", IntegerType()),
        ]
    )

    full_table_name = f"{FS_INTEG_TEST_DB}.{FS_INTEG_TEST_DATASET_SCHEMA}.{FS_INTEG_TEST_WINE_QUALITY_DATA}"
    df = (
        sess.read.options({"field_delimiter": ";", "skip_header": 1})
        .schema(input_schema)
        .csv(f"{sess.get_session_stage()}/{WINEDATA_NAME}")
    )
    df.write.mode(overwrite_mode).save_as_table(full_table_name)
    rows_count = sess.sql(f"SELECT COUNT(*) FROM {full_table_name}").collect()[0][0]

    logger.info(f"{full_table_name} has total {rows_count} rows.")


if __name__ == "__main__":
    sess = Session.builder.configs(SnowflakeLoginOptions()).create()
    sess.sql(f"USE DATABASE {FS_INTEG_TEST_DB}").collect()

    create_tripdata(sess, "overwrite")
    create_winedata(sess, "overwrite")

    logger.info("Script completes successfully.")
