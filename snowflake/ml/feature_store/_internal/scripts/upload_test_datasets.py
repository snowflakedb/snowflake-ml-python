# A helper script cleans open taxi data (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
# and store into snowflake database.

from snowflake.ml._internal.utils import identifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

PARQUET_FILE_NAME = "yellow_tripdata_2016-01.parquet"

PARQUET_FILE_LOCAL_PATH = f"file://~/Downloads/{PARQUET_FILE_NAME}"


def get_destination_table_name(original_file_name: str) -> str:
    return original_file_name.split(".")[0].replace("-", "_").upper()


if __name__ == "__main__":
    sess = Session.builder.configs(SnowflakeLoginOptions()).create()
    current_db = "SNOWML_FEATURE_STORE_TEST_DB"
    current_schema = "TEST_DATASET"

    sess.file.put(PARQUET_FILE_LOCAL_PATH, sess.get_session_stage())
    df = sess.read.parquet(f"{sess.get_session_stage()}/{PARQUET_FILE_NAME}")
    for old_col_name in df.columns:
        df = df.with_column_renamed(old_col_name, identifier.get_unescaped_names(old_col_name))

    table_name = get_destination_table_name(PARQUET_FILE_NAME)
    full_table_name = f"{current_db}.{current_schema}.{table_name}"
    df.write.mode("ignore").save_as_table(full_table_name)
    rows_count = sess.sql(f"SELECT COUNT(*) FROM {full_table_name}").collect()[0][0]

    print(f"{full_table_name} has total {rows_count} rows.")
    print("Script completes successfully.")
