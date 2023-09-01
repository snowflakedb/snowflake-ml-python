#
# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
#
import sys
from io import BytesIO
from typing import Any

import cloudpickle

import snowflake.snowpark._internal.utils as snowpark_utils
from snowflake import snowpark

_RESULT_SIZE_THRESHOLD = 5 * (1024**2)  # 5MB


# This module handles serialization, uploading, downloading, and deserialization of stored
# procedure results. If the results are too large to be returned from a stored procedure,
# the result will be uploaded. The client can then retrieve and deserialize the result if
# it was uploaded.


def serialize(session: snowpark.Session, result: Any) -> bytes:
    """
    Serialize a tuple containing the result (or None) and the result object filepath
    if the result was uploaded to a stage (or None).

    Args:
        session: Snowpark session.
        result: Object to be serialized.

    Returns:
        Cloudpickled string of bytes of the result tuple.
    """
    result_object_filepath = None
    result_bytes = cloudpickle.dumps(result)
    if sys.getsizeof(result_bytes) > _RESULT_SIZE_THRESHOLD:
        stage_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        session.sql(f"CREATE TEMPORARY STAGE {stage_name}").collect()
        result_object_filepath = f"@{stage_name}/{snowpark_utils.generate_random_alphanumeric()}"
        session.file.put_stream(BytesIO(result_bytes), result_object_filepath)
        result_object_filepath = f"{result_object_filepath}.gz"

    if result_object_filepath is not None:
        return cloudpickle.dumps((None, result_object_filepath))  # type: ignore[no-any-return]

    return cloudpickle.dumps((result, None))  # type: ignore[no-any-return]


def deserialize(session: snowpark.Session, result_bytes: bytes) -> Any:
    """
    Loads and/or deserializes the (maybe uploaded) result.

    Args:
        session: Snowpark session.
        result_bytes: String of bytes returned by serialize method.

    Returns:
        The deserialized result (any type).
    """
    result_object, result_object_filepath = cloudpickle.loads(result_bytes)
    if result_object_filepath is not None:
        result_object_bytes_io = session.file.get_stream(result_object_filepath, decompress=True)
        result_bytes = result_object_bytes_io.read()
        result_object = cloudpickle.loads(result_bytes)

    return result_object
