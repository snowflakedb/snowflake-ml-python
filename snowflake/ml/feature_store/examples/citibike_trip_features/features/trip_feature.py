from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.citibike_trip_features.entities import trip_id
from snowflake.snowpark import DataFrame, Session, functions as F


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a feature view about trip."""
    feature_df = source_dfs[0].select(
        "trip_id",
        F.col("birth_year").alias("f_birth_year"),
        F.col("gender").alias("f_gender"),
        F.col("bikeid").alias("f_bikeid"),
    )

    return FeatureView(
        name="f_trip",  # name of feature view
        entities=[trip_id],  # entities
        feature_df=feature_df,  # definition query
        refresh_freq=None,  # refresh frequency. None indicates it never refresh
        desc="Static trip features",
    ).attach_feature_desc(
        {
            "f_birth_year": "The birth year of a trip passenger.",
            "f_gender": "The gender of a trip passenger.",
            "f_bikeid": "The bike id of a trip passenger.",
        }
    )
