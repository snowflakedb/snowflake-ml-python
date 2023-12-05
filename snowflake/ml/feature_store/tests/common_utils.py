from typing import Any, Callable, Dict, List
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


def create_random_schema(session: Session, prefix: str, database: str = FS_INTEG_TEST_DB) -> str:
    schema = prefix + "_" + uuid4().hex.upper()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()
    return schema


def compare_dataframe(actual_df: pd.DataFrame, target_data: Dict[str, Any], sort_cols: List[str]) -> None:
    target_df = pd.DataFrame(target_data).sort_values(by=sort_cols)
    assert_frame_equal(
        actual_df.sort_values(by=sort_cols).reset_index(drop=True), target_df.reset_index(drop=True), check_dtype=False
    )


def compare_feature_views(actual_fvs: List[FeatureView], target_fvs: List[FeatureView]) -> None:
    assert len(actual_fvs) == len(target_fvs)
    for actual_fv, target_fv in zip(actual_fvs, target_fvs):
        assert actual_fv == target_fv, f"{actual_fv.name} doesn't match {target_fv.name}"


def create_mock_session(trouble_query: str, exception: Exception) -> Any:
    def side_effect(session: Session) -> Callable[[Any], Any]:
        original_sql = session.sql

        def dispatch(*args: Any) -> Any:
            if trouble_query in args[0]:
                raise exception
            return original_sql(*args)

        return dispatch

    session = Session.builder.configs(SnowflakeLoginOptions()).create()
    session.sql = Mock(side_effect=side_effect(session))
    return session


def get_test_warehouse_name(session: Session) -> str:
    session_warehouse = session.get_current_warehouse()
    return session_warehouse if session_warehouse else "REGTEST_ML_4XL_MULTI"
