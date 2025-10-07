from snowflake.ml._internal.utils.snowflake_env import SnowflakeCloudType
from snowflake.ml.jobs._utils.types import ComputeResources

# SPCS specification constants
DEFAULT_CONTAINER_NAME = "main"
MEMORY_VOLUME_NAME = "dshm"
STAGE_VOLUME_NAME = "stage-volume"

# Environment variables
STAGE_MOUNT_PATH_ENV_VAR = "MLRS_STAGE_MOUNT_PATH"
PAYLOAD_DIR_ENV_VAR = "MLRS_PAYLOAD_DIR"
RESULT_PATH_ENV_VAR = "MLRS_RESULT_PATH"
MIN_INSTANCES_ENV_VAR = "MLRS_MIN_INSTANCES"
TARGET_INSTANCES_ENV_VAR = "SNOWFLAKE_JOBS_COUNT"
INSTANCES_MIN_WAIT_ENV_VAR = "MLRS_INSTANCES_MIN_WAIT"
INSTANCES_TIMEOUT_ENV_VAR = "MLRS_INSTANCES_TIMEOUT"
INSTANCES_CHECK_INTERVAL_ENV_VAR = "MLRS_INSTANCES_CHECK_INTERVAL"
RUNTIME_IMAGE_TAG_ENV_VAR = "MLRS_CONTAINER_IMAGE_TAG"

# Stage mount paths
STAGE_VOLUME_MOUNT_PATH = "/mnt/job_stage"
APP_STAGE_SUBPATH = "app"
SYSTEM_STAGE_SUBPATH = "system"
OUTPUT_STAGE_SUBPATH = "output"
RESULT_PATH_DEFAULT_VALUE = f"{OUTPUT_STAGE_SUBPATH}/mljob_result"

# Default container image information
DEFAULT_IMAGE_REPO = "/snowflake/images/snowflake_images"
DEFAULT_IMAGE_CPU = "st_plat/runtime/x86/runtime_image/snowbooks"
DEFAULT_IMAGE_GPU = "st_plat/runtime/x86/generic_gpu/runtime_image/snowbooks"
DEFAULT_IMAGE_TAG = "1.8.0"
DEFAULT_ENTRYPOINT_PATH = "func.py"

# Percent of container memory to allocate for /dev/shm volume
MEMORY_VOLUME_SIZE = 0.3

# Ray port configuration
RAY_PORTS = {
    "HEAD_CLIENT_SERVER_PORT": "10001",
    "HEAD_GCS_PORT": "12001",
    "HEAD_DASHBOARD_GRPC_PORT": "12002",
    "HEAD_DASHBOARD_PORT": "12003",
    "OBJECT_MANAGER_PORT": "12011",
    "NODE_MANAGER_PORT": "12012",
    "RUNTIME_ENV_AGENT_PORT": "12013",
    "DASHBOARD_AGENT_GRPC_PORT": "12014",
    "DASHBOARD_AGENT_LISTEN_PORT": "12015",
    "MIN_WORKER_PORT": "12031",
    "MAX_WORKER_PORT": "13000",
}

# Node health check configuration
# TODO(SNOW-1937020): Revisit the health check configuration
ML_RUNTIME_HEALTH_CHECK_PORT = "5001"
ENABLE_HEALTH_CHECKS_ENV_VAR = "ENABLE_HEALTH_CHECKS"
ENABLE_HEALTH_CHECKS = "false"

# Job status polling constants
JOB_POLL_INITIAL_DELAY_SECONDS = 0.1
JOB_POLL_MAX_DELAY_SECONDS = 30

# Log start and end messages
LOG_START_MSG = "--------------------------------\nML job started\n--------------------------------"
LOG_END_MSG = "--------------------------------\nML job finished\n--------------------------------"

# Default setting for verbose logging in get_log function
DEFAULT_VERBOSE_LOG = False

# Compute pool resource information
# TODO: Query Snowflake for resource information instead of relying on this hardcoded
#       table from https://docs.snowflake.com/en/sql-reference/sql/create-compute-pool
COMMON_INSTANCE_FAMILIES = {
    "CPU_X64_XS": ComputeResources(cpu=1, memory=6),
    "CPU_X64_S": ComputeResources(cpu=3, memory=13),
    "CPU_X64_M": ComputeResources(cpu=6, memory=28),
    "CPU_X64_L": ComputeResources(cpu=28, memory=116),
    "HIGHMEM_X64_S": ComputeResources(cpu=6, memory=58),
}
AWS_INSTANCE_FAMILIES = {
    "HIGHMEM_X64_M": ComputeResources(cpu=28, memory=240),
    "HIGHMEM_X64_L": ComputeResources(cpu=124, memory=984),
    "GPU_NV_S": ComputeResources(cpu=6, memory=27, gpu=1, gpu_type="A10G"),
    "GPU_NV_M": ComputeResources(cpu=44, memory=178, gpu=4, gpu_type="A10G"),
    "GPU_NV_L": ComputeResources(cpu=92, memory=1112, gpu=8, gpu_type="A100"),
}
AZURE_INSTANCE_FAMILIES = {
    "HIGHMEM_X64_M": ComputeResources(cpu=28, memory=244),
    "HIGHMEM_X64_L": ComputeResources(cpu=92, memory=654),
    "GPU_NV_XS": ComputeResources(cpu=3, memory=26, gpu=1, gpu_type="T4"),
    "GPU_NV_SM": ComputeResources(cpu=32, memory=424, gpu=1, gpu_type="A10"),
    "GPU_NV_2M": ComputeResources(cpu=68, memory=858, gpu=2, gpu_type="A10"),
    "GPU_NV_3M": ComputeResources(cpu=44, memory=424, gpu=2, gpu_type="A100"),
    "GPU_NV_SL": ComputeResources(cpu=92, memory=858, gpu=4, gpu_type="A100"),
}
CLOUD_INSTANCE_FAMILIES = {
    SnowflakeCloudType.AWS: AWS_INSTANCE_FAMILIES,
    SnowflakeCloudType.AZURE: AZURE_INSTANCE_FAMILIES,
}
