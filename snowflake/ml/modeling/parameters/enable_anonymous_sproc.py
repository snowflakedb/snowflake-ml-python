"""Enables the anonymous stored procedures for running modeling fit"""

from snowflake.ml.modeling._internal.snowpark_implementations import snowpark_trainer

snowpark_trainer._ENABLE_ANONYMOUS_SPROC = True
