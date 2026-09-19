"""Shared ``func_params_to_log`` allowlists for model public APIs.

Log configuration, identifiers, and resource knobs. Do not log data payloads,
free-text comments/values, local filesystem paths, raw SQL, or FeatureView
objects (their ``repr`` dumps internal state).
"""

CREATE_SERVICE_FUNC_PARAMS_TO_LOG = [
    "service_name",
    "image_build_compute_pool",
    "service_compute_pool",
    "image_repo",
    "ingress_enabled",
    "min_instances",
    "max_instances",
    "cpu_requests",
    "memory_requests",
    "gpu_requests",
    "num_workers",
    "max_batch_rows",
    "force_rebuild",
    "build_external_access_integrations",
    "block",
    "autocapture",
    "inference_engine_options",
]

RUN_FUNC_PARAMS_TO_LOG = [
    "function_name",
    "service_name",
    "params",
    "partition_column",
    "strict_input_validation",
]

RUN_BATCH_FUNC_PARAMS_TO_LOG = [
    "compute_pool",
    "input_spec",
    "output_spec",
    "resources_spec",
    "inference_spec",
    "image_build_spec",
    "replicas",
    "function_name",
    "job_name",
    "async_",
    "input_stage_location",
]

BATCH_INFERENCE_TASK_FUNC_PARAMS_TO_LOG = [
    "name",
    "compute_pool",
    "input_spec",
    "output_spec",
    "resources_spec",
    "inference_spec",
    "image_build_spec",
    "replicas",
    "function_name",
    "input_stage_location",
]

LOG_MODEL_FUNC_PARAMS_TO_LOG = [
    "model_name",
    "version_name",
    "comment",
    "metrics",
    "conda_dependencies",
    "pip_requirements",
    "artifact_repository_map",
    "resource_constraint",
    "target_platforms",
    "python_version",
    "signatures",
    "task",
    "options",
]

HF_LOG_MODEL_AND_CREATE_SERVICE_FUNC_PARAMS_TO_LOG = [
    "model_name",
    "version_name",
    "pip_requirements",
    "conda_dependencies",
] + CREATE_SERVICE_FUNC_PARAMS_TO_LOG
