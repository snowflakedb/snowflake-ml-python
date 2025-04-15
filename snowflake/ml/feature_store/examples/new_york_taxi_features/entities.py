from snowflake.ml.feature_store import Entity

trip_id = Entity(name="TRIP_ID", join_keys=["TRIP_ID"], desc="Trip id.")

location_id = Entity(name="DOLOCATIONID", join_keys=["DOLOCATIONID"], desc="Drop off location id.")


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> list[Entity]:
    return [trip_id, location_id]
