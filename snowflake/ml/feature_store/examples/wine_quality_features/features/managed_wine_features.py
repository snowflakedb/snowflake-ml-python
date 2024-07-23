from typing import List

from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.wine_quality_features.entities import (
    wine_entity,
)
from snowflake.snowpark import DataFrame, Session, functions as F


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(session: Session, source_dfs: List[DataFrame], source_tables: List[str]) -> FeatureView:
    """Create a feature view about trip station."""
    feature_df = source_dfs[0].select(
        "WINE_ID",
        "FIXED_ACIDITY",
        "CITRIC_ACID",
        "CHLORIDES",
        "TOTAL_SULFUR_DIOXIDE",
        "PH",
        (F.col("FIXED_ACIDITY") * F.col("CITRIC_ACID")).alias("MY_NEW_FEATURE"),
    )

    return FeatureView(
        name="WINE_FEATURES",  # name of feature view
        entities=[wine_entity],  # entities
        feature_df=feature_df,  # definition dataframe
        refresh_freq="1d",  # refresh frequency. '1d' means it refreshes everyday
        desc="Managed feature view about wine quality which refreshes everyday.",
    )
