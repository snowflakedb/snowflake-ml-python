from snowflake.ml.feature_store import Entity

end_station_id = Entity(
    name="end_station_id",
    join_keys=["end_station_id"],
    desc="The id of an end station.",
)

trip_id = Entity(
    name="trip_id",
    join_keys=["trip_id"],
    desc="The id of a trip.",
)


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> list[Entity]:
    return [end_station_id, trip_id]
