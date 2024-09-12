"""Disables the snowpark observability tracer when running modeling fit"""

from snowflake.ml.modeling._internal.snowpark_implementations import snowpark_trainer

snowpark_trainer._ENABLE_TRACER = False
