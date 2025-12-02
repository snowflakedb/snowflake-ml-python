from typing import Any, Callable, Optional
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal

from snowflake.ml.feature_store.feature_view import FeatureView
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

# Database used for feature store integration test
FS_INTEG_TEST_DB = "SNOWML_FEATURE_STORE_TEST_DB"

# Dummy database for feature store integration test to make sure feature store code
# is resilient to session pointing to different location
FS_INTEG_TEST_DUMMY_DB = "SNOWML_FEATURE_STORE_DUMMY_DB"

# Schema with test datasets used for feature store integration test
FS_INTEG_TEST_DATASET_SCHEMA = "TEST_DATASET"

# Yellow trip dataset
FS_INTEG_TEST_YELLOW_TRIP_DATA = "yellow_tripdata_2016_01"

# Wine quality dataset
FS_INTEG_TEST_WINE_QUALITY_DATA = "wine_quality_data"

# If object live time is greater than specified hours it will be deleted.
DB_OBJECT_EXPIRE_HOURS = 24


def create_random_schema(
    session: Session, prefix: str, database: str = FS_INTEG_TEST_DB, additional_options: str = ""
) -> str:
    schema = prefix + "_" + uuid4().hex.upper()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema} {additional_options}").collect()
    return schema


def compare_dataframe(
    actual_df: pd.DataFrame, target_data: dict[str, Any], sort_cols: list[str], exclude_cols: Optional[list[str]] = None
) -> None:
    if exclude_cols is not None:
        for c in exclude_cols:
            assert c.upper() in actual_df, f"{c.upper()} is missing in actual_df"
        actual_df = actual_df.drop([c.upper() for c in exclude_cols], axis=1)

    target_df = pd.DataFrame(target_data).sort_values(by=sort_cols)
    assert_frame_equal(
        actual_df.sort_values(by=sort_cols).reset_index(drop=True), target_df.reset_index(drop=True), check_dtype=False
    )


def compare_feature_views(actual_fvs: list[FeatureView], target_fvs: list[FeatureView]) -> None:
    assert len(actual_fvs) == len(target_fvs)
    for actual_fv, target_fv in zip(actual_fvs, target_fvs):
        assert actual_fv == target_fv, f"{actual_fv.name} doesn't match {target_fv.name}"


def create_mock_session(trouble_query: str, exception: Exception, config: Optional[dict[str, str]] = None) -> Any:
    def side_effect(session: Session) -> Callable[[Any], Any]:
        original_sql = session.sql

        def dispatch(*args: Any) -> Any:
            if trouble_query in args[0]:
                raise exception
            return original_sql(*args)

        return dispatch

    config = config or SnowflakeLoginOptions()
    session = Session.builder.configs(config).create()
    session.sql = Mock(side_effect=side_effect(session))
    return session


def create_mock_table(
    session: Session, database: Optional[str] = None, schema: Optional[str] = None, table_prefix: str = "TEST_TABLE"
) -> str:
    test_table = f"{table_prefix}_{uuid4().hex.upper()}"
    if schema:
        test_table = schema + "." + test_table
    if database:
        assert bool(schema)
        test_table = database + "." + test_table
    session.sql(
        f"""CREATE TABLE IF NOT EXISTS {test_table}
            (name VARCHAR(64), id INT, title VARCHAR(128), age INT, dept VARCHAR(64), ts INT)
        """
    ).collect()
    session.sql(
        f"""INSERT OVERWRITE INTO {test_table} (name, id, title, age, dept, ts)
            VALUES
            ('john', 1, 'boss', 20, 'sales', 100),
            ('porter', 2, 'manager', 30, 'engineer', 200)
        """
    ).collect()
    return test_table


def get_test_warehouse_name(session: Session) -> str:
    session_warehouse = session.get_current_warehouse()
    if not session_warehouse:
        raise RuntimeError("No warehouse is configured in the current session.")
    # Sanitize any accidental surrounding quotes to avoid generating SQL like """WH"""
    return session_warehouse.strip('"')
