from typing import List

from snowflake.ml.feature_store import Entity

trip_pickup = Entity(name="TRIP_PICKUP", join_keys=["PULOCATIONID"], desc="Trip pickup entity.")

trip_dropoff = Entity(name="TRIP_DROPOFF", join_keys=["DOLOCATIONID"], desc="Trip dropoff entity.")


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> List[Entity]:
    return [trip_pickup, trip_dropoff]
