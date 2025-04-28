from snowflake.ml.feature_store import Entity

zipcode_entity = Entity(
    name="AIRPORT_ZIP_CODE",
    join_keys=["AIRPORT_ZIP_CODE"],
    desc="Zip code of the airport.",
)

plane_entity = Entity(name="PLANE_MODEL", join_keys=["PLANE_MODEL"], desc="The model of an airplane.")


# This will be invoked by example_helper.py. Do not change function name.
def get_all_entities() -> list[Entity]:
    return [zipcode_entity, plane_entity]
