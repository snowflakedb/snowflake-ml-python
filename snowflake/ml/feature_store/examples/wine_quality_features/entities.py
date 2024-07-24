from typing import List

from snowflake.ml.feature_store import Entity

wine_entity = Entity(
    name="WINE",
    join_keys=["WINE_ID"],
    desc="Wine ID column.",
)


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> List[Entity]:
    return [wine_entity]
