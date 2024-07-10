from snowflake.ml.feature_store import Entity

end_station_id = Entity(
    name="end_station_id",
    join_keys=["end_station_id"],
    desc="End Station Id",
)


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> None:
    return [end_station_id]
