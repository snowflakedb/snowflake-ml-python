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


class SnowflakeResult:
    """
    Handles serialization, uploading, downloading, and deserialization of stored procedure results. If the results
    are too large to be returned from a stored procedure, the result will be uploaded. The client can then retrieve
    and deserialize the result if it was uploaded.
    """

    def __init__(self, session: snowpark.Session, result: Any) -> None:
        self.result = result
        self.session = session
        self.result_object_filepath = None
        result_bytes = cloudpickle.dumps(self.result)
        if sys.getsizeof(result_bytes) > _RESULT_SIZE_THRESHOLD:
            stage_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
            session.sql(f"CREATE TEMPORARY STAGE {stage_name}").collect()
            result_object_filepath = f"@{stage_name}/{snowpark_utils.generate_random_alphanumeric()}"
            session.file.put_stream(BytesIO(result_bytes), result_object_filepath)
            self.result_object_filepath = f"{result_object_filepath}.gz"

    def serialize(self) -> bytes:
        """
        Serialize a tuple containing the result (or None) and the result object filepath
        if the result was uploaded to a stage (or None).

        Returns:
            Cloudpickled string of bytes of the result tuple.
        """
        if self.result_object_filepath is not None:
            return cloudpickle.dumps((None, self.result_object_filepath))  # type: ignore[no-any-return]
        return cloudpickle.dumps((self.result, None))  # type: ignore[no-any-return]

    @staticmethod
    def load_result_from_filepath(session: snowpark.Session, result_object_filepath: str) -> Any:
        """
        Loads and deserializes the uploaded result.

        Args:
            session: Snowpark session.
            result_object_filepath: Stage filepath of the result object returned by serialize method.

        Returns:
            The original serialized result (any type).
        """
        result_object_bytes_io = session.file.get_stream(result_object_filepath, decompress=True)
        result_bytes = result_object_bytes_io.read()
        return cloudpickle.loads(result_bytes)
