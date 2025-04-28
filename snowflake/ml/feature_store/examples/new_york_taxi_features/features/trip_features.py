from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.new_york_taxi_features.entities import trip_id
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a draft feature view."""
    feature_df = session.sql(
        f"""
        select
            TRIP_ID,
            PASSENGER_COUNT,
            TRIP_DISTANCE,
            FARE_AMOUNT
        from
            {database}.{schema}.{source_tables[0]}
        """
    )

    return FeatureView(
        name="f_trip",  # name of feature view
        entities=[trip_id],  # entities
        feature_df=feature_df,  # definition query
        refresh_freq="1d",  # the frequency this feature view re-compute
        timestamp_col=None,  # timestamp column. Used when generate training data
        desc="Features per trip refreshed every day.",
    ).attach_feature_desc(
        {
            "PASSENGER_COUNT": "The count of passenger of a trip.",
            "TRIP_DISTANCE": "The distance of a trip.",
            "FARE_AMOUNT": "The fare of a trip.",
        }
    )
