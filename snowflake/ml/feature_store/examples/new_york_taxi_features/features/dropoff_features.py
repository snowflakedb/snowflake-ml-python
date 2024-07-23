from typing import List

from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.new_york_taxi_features.entities import (
    trip_dropoff,
)
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(session: Session, source_dfs: List[DataFrame], source_tables: List[str]) -> FeatureView:
    """Create a draft feature view."""
    feature_df = session.sql(
        f"""
        select
            TPEP_DROPOFF_DATETIME as TS,
            DOLOCATIONID,
            count(FARE_AMOUNT) over (
                partition by DOLOCATIONID
                order by TPEP_DROPOFF_DATETIME
                range between interval '1 hours' preceding and current row
            ) TRIP_COUNT_1H,
            count(FARE_AMOUNT) over (
                partition by DOLOCATIONID
                order by TPEP_DROPOFF_DATETIME
                range between interval '5 hours' preceding and current row
            ) TRIP_COUNT_5H
        from {source_tables[0]}
    """
    )

    return FeatureView(
        name="f_trip_dropoff",  # name of feature view
        entities=[trip_dropoff],  # entities
        feature_df=feature_df,  # definition query
        refresh_freq="12h",  # the frequency this feature view re-compute
        timestamp_col="TS",  # timestamp column. Used when generate training data
        desc="Managed feature view trip dropoff refreshed every 12 hours.",
    )
