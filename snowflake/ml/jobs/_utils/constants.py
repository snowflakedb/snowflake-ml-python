from snowflake.ml._internal.utils.snowflake_env import SnowflakeCloudType
from snowflake.ml.jobs._utils.types import ComputeResources

# SPCS specification constants
DEFAULT_CONTAINER_NAME = "main"
PAYLOAD_DIR_ENV_VAR = "MLRS_PAYLOAD_DIR"
MEMORY_VOLUME_NAME = "dshm"
STAGE_VOLUME_NAME = "stage-volume"
STAGE_VOLUME_MOUNT_PATH = "/mnt/app"

# Default container image information
DEFAULT_IMAGE_REPO = "/snowflake/images/snowflake_images"
DEFAULT_IMAGE_CPU = "st_plat/runtime/x86/runtime_image/snowbooks"
DEFAULT_IMAGE_GPU = "st_plat/runtime/x86/generic_gpu/runtime_image/snowbooks"
DEFAULT_IMAGE_TAG = "0.9.2"
DEFAULT_ENTRYPOINT_PATH = "func.py"

# Percent of container memory to allocate for /dev/shm volume
MEMORY_VOLUME_SIZE = 0.3

# Job status polling constants
JOB_POLL_INITIAL_DELAY_SECONDS = 0.1
JOB_POLL_MAX_DELAY_SECONDS = 1

# Magic attributes
IS_MLJOB_REMOTE_ATTR = "_is_mljob_remote_callable"

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
