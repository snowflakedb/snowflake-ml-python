from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.citibike_trip_features.entities import (
    end_station_id,
)
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a feature view about trip station."""
    query = session.sql(
        f"""
        select
            end_station_id,
            count(end_station_id) as f_count,
            avg(end_station_latitude) as f_avg_latitude,
            avg(end_station_longitude) as f_avg_longtitude
        from {database}.{schema}.{source_tables[0]}
        group by end_station_id
        """
    )

    return FeatureView(
        name="f_station",  # name of feature view
        entities=[end_station_id],  # entities
        feature_df=query,  # definition query
        refresh_freq="1d",  # refresh frequency. '1d' means it refreshes everyday
        desc="Station features refreshed every day.",
    ).attach_feature_desc(
        {
            "f_count": "How many times this station appears in 1 day.",
            "f_avg_latitude": "Averaged latitude of a station.",
            "f_avg_longtitude": "Averaged longtitude of a station.",
        }
    )
