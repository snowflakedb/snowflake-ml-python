from typing import List

from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.airline_features.entities import zipcode_entity
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(session: Session, source_dfs: List[DataFrame], source_tables: List[str]) -> FeatureView:
    """Create a feature view about airport weather."""
    query = session.sql(
        """
        select
            DATETIME_UTC AS TS,
            AIRPORT_ZIP_CODE,
            sum(RAIN_MM_H) over (
                partition by AIRPORT_ZIP_CODE
                order by DATETIME_UTC
                range between interval '30 minutes' preceding and current row
            ) RAIN_SUM_30M,
            sum(RAIN_MM_H) over (
                partition by AIRPORT_ZIP_CODE
                order by DATETIME_UTC
                range between interval '1 day' preceding and current row
            ) RAIN_SUM_60M
        from AIRPORT_WEATHER_STATION
        """
    )

    return FeatureView(
        name="f_weather",  # name of feature view
        entities=[zipcode_entity],  # entities
        feature_df=query,  # definition query
        timestamp_col="TS",  # timestamp column
        refresh_freq="1d",  # refresh frequency
        desc="Airport weather features refreshed every day.",
    ).attach_feature_desc(
        {
            "RAIN_SUM_30M": "The sum of rain fall over past 30 minutes for one zipcode.",
            "RAIN_SUM_60M": "The sum of rain fall over past 1 day for one zipcode.",
        }
    )
