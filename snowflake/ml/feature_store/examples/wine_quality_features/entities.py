from snowflake.ml.feature_store import Entity

wine_id = Entity(
    name="WINE",
    join_keys=["WINE_ID"],
    desc="Wine ID.",
)


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> list[Entity]:
    return [wine_id]
