from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.new_york_taxi_features.entities import (
    location_id,
)
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a draft feature view."""
    feature_df = session.sql(
        f"""
        select
            TPEP_DROPOFF_DATETIME as TS,
            DOLOCATIONID,
            avg(FARE_AMOUNT) over (
                partition by DOLOCATIONID
                order by TPEP_DROPOFF_DATETIME
                range between interval '1 hours' preceding and current row
            ) AVG_FARE_1H,
            avg(FARE_AMOUNT) over (
                partition by DOLOCATIONID
                order by TPEP_DROPOFF_DATETIME
                range between interval '10 hours' preceding and current row
            ) AVG_FARE_10h
        from {database}.{schema}.{source_tables[0]}
    """
    )

    return FeatureView(
        name="f_location",  # name of feature view
        entities=[location_id],  # entities
        feature_df=feature_df,  # definition query
        refresh_freq="12h",  # the frequency this feature view re-compute
        timestamp_col="TS",  # timestamp column. Used when generate training data
        desc="Features aggregated by location id and refreshed every 12 hours.",
    ).attach_feature_desc(
        {
            "AVG_FARE_1H": "Averaged fare in past 1 hour window aggregated by location.",
            "AVG_FARE_10H": "Averaged fare in past 10 hours aggregated by location.",
        }
    )
