from snowflake.ml.feature_store import FeatureView
from snowflake.ml.feature_store.examples.airline_features.entities import plane_entity
from snowflake.snowpark import DataFrame, Session


# This function will be invoked by example_helper.py. Do not change the name.
def create_draft_feature_view(
    session: Session, source_dfs: list[DataFrame], source_tables: list[str], database: str, schema: str
) -> FeatureView:
    """Create a feature view about airplane model."""
    query = session.sql(
        f"""
        select
           PLANE_MODEL,
           SEATING_CAPACITY
        from
            {database}.{schema}.PLANE_MODEL_ATTRIBUTES
        """
    )

    return FeatureView(
        name="f_plane",  # name of feature view
        entities=[plane_entity],  # entities
        feature_df=query,  # definition query
        refresh_freq=None,  # refresh frequency
        desc="Plane features never refresh.",
    ).attach_feature_desc(
        {
            "SEATING_CAPACITY": "The seating capacity of a plane.",
        }
    )
