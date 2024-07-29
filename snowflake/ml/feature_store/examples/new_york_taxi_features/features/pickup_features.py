from typing import List

from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.new_york_taxi_features.entities import (
    trip_pickup,
)
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(session: Session, source_dfs: List[DataFrame], source_tables: List[str]) -> FeatureView:
    """Create a draft feature view."""
    feature_df = session.sql(
        f"""
        with
            cte_1 (TS, PULOCATIONID, TRIP_COUNT_2H, TRIP_COUNT_5H, TRIP_FARE_SUM_2H, TRIP_FARE_SUM_5H) as (
                select
                    TPEP_PICKUP_DATETIME as TS,
                    PULOCATIONID,
                    count(FARE_AMOUNT) over (
                        partition by PULOCATIONID
                        order by TPEP_PICKUP_DATETIME
                        range between interval '2 hours' preceding and current row
                    ) TRIP_COUNT_2H,
                    count(FARE_AMOUNT) over (
                        partition by PULOCATIONID
                        order by TPEP_PICKUP_DATETIME
                        range between interval '5 hours' preceding and current row
                    ) TRIP_COUNT_5H,
                    sum(FARE_AMOUNT) over (
                        partition by PULOCATIONID
                        order by TPEP_PICKUP_DATETIME
                        range between interval '2 hours' preceding and current row
                    ) TRIP_FARE_SUM_2H,
                    count(FARE_AMOUNT) over (
                        partition by PULOCATIONID
                        order by TPEP_PICKUP_DATETIME
                        range between interval '5 hours' preceding and current row
                    ) TRIP_FARE_SUM_5H
                from {source_tables[0]}
            )
        select
            TS,
            PULOCATIONID,
            TRIP_FARE_SUM_2H / TRIP_COUNT_2H as MEAN_FARE_2H,
            TRIP_FARE_SUM_5H / TRIP_COUNT_5H as MEAN_FARE_5H,
        from cte_1
    """
    )

    return FeatureView(
        name="f_trip_pickup",  # name of feature view
        entities=[trip_pickup],  # entities
        feature_df=feature_df,  # definition query
        refresh_freq="1d",  # the frequency this feature view re-compute
        timestamp_col="TS",  # timestamp column. Used when generate training data
        desc="Managed feature view trip pickup refreshed everyday.",
    )
