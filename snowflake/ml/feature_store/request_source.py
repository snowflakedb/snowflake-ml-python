"""Request source definition for realtime feature views.

This module provides the RequestSource class for defining request-time
input schemas for realtime feature views.
"""

from dataclasses import dataclass

from snowflake.snowpark.types import StructType


@dataclass(frozen=True)
class RequestSource:
    """Defines request-time input for realtime feature views.

    RequestSource describes the schema of data that will be provided at
    inference time (request time) rather than being pre-computed.

    Args:
        schema: Expected schema of request-time inputs as a Snowpark StructType.

    Example::

        >>> from snowflake.snowpark.types import (
        ...     StructType, StructField, FloatType, StringType,
        ... )
        >>> request = RequestSource(
        ...     schema=StructType([
        ...         StructField("transaction_amount", FloatType()),
        ...         StructField("merchant_id", StringType()),
        ...     ])
        ... )
    """

    schema: StructType
