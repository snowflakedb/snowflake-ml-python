from snowflake import snowpark


def get_attribute_map(session: snowpark.Session, requested_attributes: dict[str, int]) -> dict[str, int]:
    metadata = session._conn._cursor.description
    for index in range(len(metadata)):
        if metadata[index].name in requested_attributes.keys():
            requested_attributes[metadata[index].name] = index
    return requested_attributes
