from snowflake.ml.jobs._utils import constants as mljob_constants

# Constants defining the shutdown signal actor configuration.
SHUTDOWN_ACTOR_NAME = "ShutdownSignal"
SHUTDOWN_ACTOR_NAMESPACE = "default"
SHUTDOWN_RPC_TIMEOUT_SECONDS = 5.0


# The followings are Inherited from snowflake.ml.jobs._utils.constants
# We need to copy them here since snowml package on the server side does
# not have the latest version of the code

# Log start and end messages
LOG_START_MSG = getattr(
    mljob_constants,
    "LOG_START_MSG",
    "--------------------------------\nML job started\n--------------------------------",
)
LOG_END_MSG = getattr(
    mljob_constants,
    "LOG_END_MSG",
    "--------------------------------\nML job finished\n--------------------------------",
)

# min_instances environment variable name
MIN_INSTANCES_ENV_VAR = getattr(mljob_constants, "MIN_INSTANCES_ENV_VAR", "MLRS_MIN_INSTANCES")
