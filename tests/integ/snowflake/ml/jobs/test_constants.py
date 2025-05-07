from snowflake.ml._internal.utils import snowflake_env

_TEST_COMPUTE_POOL = "E2E_TEST_POOL"
_TEST_SCHEMA = "ML_JOB_TEST_SCHEMA"
_SUPPORTED_CLOUDS = {
    snowflake_env.SnowflakeCloudType.AWS,
    snowflake_env.SnowflakeCloudType.AZURE,
}
