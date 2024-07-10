from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.simple_features.entities import end_station_id
from snowflake.snowpark import Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(session: Session, source_schema: str) -> FeatureView:
    """Create a feature view about trip station."""
    query = session.sql(
        f"""
        select
            end_station_id,
            count(end_station_id) as f_count,
            avg(tripduration) as f_avg_tripduration
        from {source_schema}.citibike_trips
        group by end_station_id
        """
    )

    return FeatureView(
        name="f_station_static",  # name of feature view
        entities=[end_station_id],  # entities
        feature_df=query,  # definition query
        refresh_freq=None,  # refresh frequency. None indicates it never refresh
        desc="Static feature view about trip station.",
    )
