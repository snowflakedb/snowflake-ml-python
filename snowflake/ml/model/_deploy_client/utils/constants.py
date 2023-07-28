from enum import Enum


class ResourceType(Enum):
    SERVICE = "service"
    JOB = "job"


"""
Potential SnowService status based on existing ResourceSetStatus proto:

github.com/snowflakedb/snowflake/blob/main/GlobalServices/src/main/protobuf/snowservices_resourceset_reconciler.proto
"""


class ResourceStatus(Enum):
    UNKNOWN = "UNKNOWN"  # status is unknown because we have not received enough data from K8s yet.
    PENDING = "PENDING"  # resource set is being created, can't be used yet
    READY = "READY"  # resource set has been deployed.
    DELETING = "DELETING"  # resource set is being deleted
    FAILED = "FAILED"  # resource set has failed and cannot be used anymore
    DONE = "DONE"  # resource set has finished running
    NOT_FOUND = "NOT_FOUND"  # not found or deleted
    INTERNAL_ERROR = "INTERNAL_ERROR"  # there was an internal service error.


RESOURCE_TO_STATUS_FUNCTION_MAPPING = {
    ResourceType.SERVICE: "SYSTEM$GET_SNOWSERVICE_STATUS",
    ResourceType.JOB: "SYSTEM$GET_JOB_STATUS",
}

PREDICT = "predict"
STAGE = "stage"
COMPUTE_POOL = "compute_pool"
MIN_INSTANCES = "min_instances"
MAX_INSTANCES = "max_instances"
GPU_COUNT = "gpu"
OVERRIDDEN_BASE_IMAGE = "image"
ENDPOINT = "endpoint"
SERVICE_SPEC = "service_spec"
INFERENCE_SERVER_CONTAINER = "inference-server"

"""Image build related constants"""
SNOWML_IMAGE_REPO = "snowml_repo"
MODEL_DIR = "model_dir"
INFERENCE_SERVER_DIR = "inference_server"
ENTRYPOINT_SCRIPT = "gunicorn_run.sh"
PROD_IMAGE_REGISTRY_DOMAIN = "snowflakecomputing.com"
PROD_IMAGE_REGISTRY_SUBDOMAIN = "registry"
DEV_IMAGE_REGISTRY_SUBDOMAIN = "registry-dev"
MODEL_ENV_FOLDER = "env"
CONDA_FILE = "conda.yaml"
