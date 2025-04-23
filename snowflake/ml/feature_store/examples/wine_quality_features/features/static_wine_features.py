from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.wine_quality_features.entities import wine_id
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a feature view about trip station."""
    feature_df = source_dfs[0].select("WINE_ID", "SULPHATES", "ALCOHOL")

    return FeatureView(
        name="EXTRA_WINE_FEATURES",  # name of feature view
        entities=[wine_id],  # entities
        feature_df=feature_df,  # feature dataframe
        refresh_freq=None,  # refresh frequency. None means it never refresh
        desc="Static features about wine quality which never refresh.",
    ).attach_feature_desc(
        {
            "SULPHATES": "Sulphates.",
            "ALCOHOL": "Alcohol.",
        }
    )
