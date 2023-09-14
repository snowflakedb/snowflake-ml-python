"""
- *XXX: category
    - 0XXX: undefined
    - 1XXX: system
    - 2XXX: user
    - 9XXX: internal test
- X*XX: source
    - X0XX: undefined
    - X1XX: Python built-in
    - X2XX: snowml (e.g. FileSetError)
    - X3XX: Snowpark (e.g. SnowparkClientException)
    - X4XX: Python connector (e.g. DatabaseError)
    - X5XX: Snowflake API (e.g. HTTP Error with Snowflake Image Repo)
    - X9XX: 3p dependency
- XX**: cause
"""

# INTERNAL
# Indicates an intentional error for internal error handling testing.
INTERNAL_TEST = "9000"

# UNDEFINED
# Indicates a failure that is not raised by Snowpark ML is caught by telemetry, and therefore undefined to the library,
# which can be caused by dependency APIs, unknown internal errors, etc.
UNDEFINED = "0000"

# SYSTEM
# Indicates an internal failure raising a Python built-in error with an ambiguous cause, such as invoking an unexpected
# private API, catching an error with an unknown cause, etc.
INTERNAL_PYTHON_ERROR = "1100"
# Indicates an internal failure raising a Snowpark ML error with an ambiguous cause, such as invoking an unexpected
# private API, catching an error with an unknown cause, etc.
INTERNAL_SNOWML_ERROR = "1200"
# Indicates an internal failure raising a Snowpark error with an ambiguous cause, such as invalid queries, invalid
# permission, catching an error with an unknown cause, etc.
INTERNAL_SNOWPARK_ERROR = "1300"
# Indicates an internal failure raising a error when using SPCS with an ambiguous cause, such as invalid queries,
# invalid permission, catching an error with an unknown cause, etc.
INTERNAL_SNOWPARK_CONTAINER_SERVICE_ERROR = "1301"
# Indicates an internal failure raising a Snowflake API error with an ambiguous cause, such as invalid queries, invalid
# permission, catching an error with an unknown cause, etc.
INTERNAL_SNOWFLAKE_API_ERROR = "1500"
# Indicates an internal HTTP or non-HTTP failure raising a error when interacting with Snowflake Container Services
INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR = "1501"

# Indicates an internal failure raising a error when using docker with an ambiguous cause, such as invalid queries,
# invalid permission, catching an error with an unknown cause, etc.
INTERNAL_DOCKER_ERROR = "1901"

# USER
# Indicates the incompatibility of local dependency versions with the target requirements. For example, an API added in
# a later version is called with an older dependency installed.
DEPENDENCY_VERSION_ERROR = "2100"
# Indicates the resource is missing: not whether the absence is temporary or permanent.
NOT_FOUND = "2101"
# The method is known but is not supported by the target resource. For example, calling `to_xgboost` is not allowed by
# Snowpark ML models based on scikit-learn.
METHOD_NOT_ALLOWED = "2102"
# Not implemented.
NOT_IMPLEMENTED = "2103"

# Calling an API with unsupported keywords/values.
INVALID_ARGUMENT = "2110"
# Object has invalid attributes caused by invalid/unsupported value, unsupported data type, size mismatch, etc.
INVALID_ATTRIBUTE = "2111"
# Missing and invalid data caused by null value, unexpected value (e.g. division by 0), out of range value, etc.
INVALID_DATA = "2112"
# Invalid data type in the processed data. For example, an API handling numeric columns gets a string column.
INVALID_DATA_TYPE = "2113"
# Calling an API with unsupported value type, or perform actions on objects with incorrect types.
INVALID_TYPE = "2114"
# Trying to create an object already exists.
OBJECT_ALREADY_EXISTS = "2115"

# Indicates the creation of underlying resources (files, stages, tables, etc) failed, which can be caused by duplicated
# name, invalid permission, etc.
SNOWML_CREATE_FAILED = "2200"
# Indicates the read of underlying resources (files, stages, tables, etc) failed, which can be caused by duplicated
# name, invalid permission, etc.
SNOWML_READ_FAILED = "2201"
# Indicates the update of underlying resources (files, stages, tables, etc) failed, which can be caused by duplicated
# name, invalid permission, etc.
SNOWML_UPDATE_FAILED = "2202"
# Indicates the deletion of underlying resources (files, stages, tables, etc) failed, which can be caused by duplicated
# name, invalid permission, etc.
SNOWML_DELETE_FAILED = "2203"
# Indicates the Snowflake resource is missing: not whether the absence is temporary or permanent.
SNOWML_NOT_FOUND = "2204"
# Indicates the access of a stage failed, which can be caused by invalid name, invalid permission, etc.
SNOWML_INVALID_STAGE = "2210"
# Invalid query caused by syntax error, invalid source, etc.
SNOWML_INVALID_QUERY = "2211"

# Invalid Snowpark Session (Missing information) in Snowpark Session that is required.
INVALID_SNOWPARK_SESSION = "2301"

# Incorrect local Python environment when trying to do some actions that require specific dependency or versions.
LOCAL_ENVIRONMENT_ERROR = "2501"
# Unfeasible dependencies requirement when trying to do some actions that require specific environments.
UNFEASIBLE_ENVIRONMENT_ERROR = "2502"

# Missing required client side dependency.
CLIENT_DEPENDENCY_MISSING_ERROR = "2511"
