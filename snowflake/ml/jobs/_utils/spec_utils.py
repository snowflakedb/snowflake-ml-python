import logging
import os
import re
import sys
from math import ceil
from pathlib import PurePath
from typing import Any, Literal, Optional, Union

from snowflake import snowpark
from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.jobs._utils import constants, feature_flags, query_helper, types
from snowflake.ml.jobs._utils.runtime_env_utils import RuntimeEnvironmentsDict

_OCI_TAG_REGEX = re.compile("^[a-zA-Z0-9._-]{1,128}$")


def _get_node_resources(session: snowpark.Session, compute_pool: str) -> types.ComputeResources:
    """Extract resource information for the specified compute pool"""
    # Get the instance family
    rows = query_helper.run_query(
        session,
        "show compute pools like ?",
        params=[compute_pool],
    )
    if not rows:
        raise ValueError(f"Compute pool '{compute_pool}' not found")
    instance_family: str = rows[0]["instance_family"]
    cloud = snowflake_env.get_current_cloud(session, default=snowflake_env.SnowflakeCloudType.AWS)

    return (
        constants.COMMON_INSTANCE_FAMILIES.get(instance_family)
        or constants.CLOUD_INSTANCE_FAMILIES[cloud][instance_family]
    )


def _get_runtime_image(session: snowpark.Session, target_hardware: Literal["CPU", "GPU"]) -> Optional[str]:
    rows = query_helper.run_query(session, "CALL SYSTEM$NOTEBOOKS_FIND_LABELED_RUNTIMES()")
    if not rows:
        return None
    try:
        runtime_envs = RuntimeEnvironmentsDict.model_validate_json(rows[0][0])
        spcs_container_runtimes = runtime_envs.get_spcs_container_runtimes()
    except Exception as e:
        logging.warning(f"Failed to parse runtime image name from {rows[0][0]}, error: {e}")
        return None

    selected_runtime = next(
        (
            runtime
            for runtime in spcs_container_runtimes
            if (
                runtime.hardware_type.lower() == target_hardware.lower()
                and runtime.python_version.major == sys.version_info.major
                and runtime.python_version.minor == sys.version_info.minor
            )
        ),
        None,
    )
    return selected_runtime.runtime_container_image if selected_runtime else None


def _check_image_tag_valid(tag: Optional[str]) -> bool:
    if tag is None:
        return False

    return _OCI_TAG_REGEX.fullmatch(tag) is not None


def _get_image_spec(
    session: snowpark.Session, compute_pool: str, runtime_environment: Optional[str] = None
) -> types.ImageSpec:
    """
    Resolve image specification (container image and resources) for the job.

    Behavior:
    - If `runtime_environment` is empty or the feature flag is disabled, use the
      default image tag and image name.
    - If `runtime_environment` is a valid image tag, use that tag with the default
      repository/name.
    - If `runtime_environment` is a full image URL, use it directly.
    - If the feature flag is enabled and `runtime_environment` is not provided,
      select an ML Runtime image matching the local Python major.minor
    - When multiple inputs are provided, `runtime_environment` takes priority.

    Args:
        session: Snowflake session.
        compute_pool: Compute pool used to infer CPU/GPU resources.
        runtime_environment: Optional image tag or full image URL to override.

    Returns:
        Image spec including container image and resource requests/limits.
    """
    # Retrieve compute pool node resources
    resources = _get_node_resources(session, compute_pool=compute_pool)
    hardware = "GPU" if resources.gpu > 0 else "CPU"
    image_tag = _get_runtime_image_tag()
    image_repo = constants.DEFAULT_IMAGE_REPO
    image_name = constants.DEFAULT_IMAGE_GPU if resources.gpu > 0 else constants.DEFAULT_IMAGE_CPU

    # Use MLRuntime image
    container_image = None
    if runtime_environment:
        if _check_image_tag_valid(runtime_environment):
            image_tag = runtime_environment
        else:
            container_image = runtime_environment
    elif feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS.is_enabled():
        container_image = _get_runtime_image(session, hardware)  # type: ignore[arg-type]

    container_image = container_image or f"{image_repo}/{image_name}:{image_tag}"
    # TODO: Should each instance consume the entire pod?
    return types.ImageSpec(
        resource_requests=resources,
        resource_limits=resources,
        container_image=container_image,
    )


def generate_spec_overrides(
    environment_vars: Optional[dict[str, str]] = None,
    custom_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate a dictionary of service specification overrides.

    Args:
        environment_vars: Environment variables to set in primary container
        custom_overrides: Custom service specification overrides

    Returns:
        Resulting service specifiation patch dict. Empty if no overrides were supplied.
    """
    # Generate container level overrides
    container_spec: dict[str, Any] = {
        "name": constants.DEFAULT_CONTAINER_NAME,
    }

    if environment_vars:
        # TODO: Validate environment variables
        container_spec["env"] = environment_vars

    # Build container override spec only if any overrides were supplied
    spec = {}
    if len(container_spec) > 1:
        spec = {
            "spec": {
                "containers": [container_spec],
            }
        }

    # Apply custom overrides
    if custom_overrides:
        spec = merge_patch(spec, custom_overrides, display_name="custom_overrides")

    return spec


def generate_service_spec(
    session: snowpark.Session,
    compute_pool: str,
    payload: types.UploadedPayload,
    args: Optional[list[str]] = None,
    target_instances: int = 1,
    min_instances: int = 1,
    enable_metrics: bool = False,
    runtime_environment: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a service specification for a job.

    Args:
        session: Snowflake session
        compute_pool: Compute pool for job execution
        payload: Uploaded job payload
        args: Arguments to pass to entrypoint script
        target_instances: Number of instances for multi-node job
        enable_metrics: Enable platform metrics for the job
        min_instances: Minimum number of instances required to start the job
        runtime_environment: The runtime image to use. Only support image tag or full image URL.

    Returns:
        Job service specification
    """
    image_spec = _get_image_spec(session, compute_pool, runtime_environment)

    # Set resource requests/limits, including nvidia.com/gpu quantity if applicable
    resource_requests: dict[str, Union[str, int]] = {
        "cpu": f"{int(image_spec.resource_requests.cpu * 1000)}m",
        "memory": f"{image_spec.resource_limits.memory}Gi",
    }
    resource_limits: dict[str, Union[str, int]] = {
        "cpu": f"{int(image_spec.resource_requests.cpu * 1000)}m",
        "memory": f"{image_spec.resource_limits.memory}Gi",
    }
    if image_spec.resource_limits.gpu > 0:
        resource_requests["nvidia.com/gpu"] = image_spec.resource_requests.gpu
        resource_limits["nvidia.com/gpu"] = image_spec.resource_limits.gpu

    # Add local volumes for ephemeral logs and artifacts
    volumes: list[dict[str, str]] = []
    volume_mounts: list[dict[str, str]] = []
    for volume_name, mount_path in [
        ("system-logs", "/var/log/managedservices/system/mlrs"),
        ("user-logs", "/var/log/managedservices/user/mlrs"),
    ]:
        volume_mounts.append(
            {
                "name": volume_name,
                "mountPath": mount_path,
            }
        )
        volumes.append(
            {
                "name": volume_name,
                "source": "local",
            }
        )

    # Mount 30% of memory limit as a memory-backed volume
    memory_volume_size = min(
        ceil(image_spec.resource_limits.memory * constants.MEMORY_VOLUME_SIZE),
        image_spec.resource_requests.memory,
    )
    volume_mounts.append(
        {
            "name": constants.MEMORY_VOLUME_NAME,
            "mountPath": "/dev/shm",
        }
    )
    volumes.append(
        {
            "name": constants.MEMORY_VOLUME_NAME,
            "source": "memory",
            "size": f"{memory_volume_size}Gi",
        }
    )

    # Mount payload as volume
    stage_mount = PurePath(constants.STAGE_VOLUME_MOUNT_PATH)
    volume_mounts.append(
        {
            "name": constants.STAGE_VOLUME_NAME,
            "mountPath": stage_mount.as_posix(),
        }
    )
    volumes.append(
        {
            "name": constants.STAGE_VOLUME_NAME,
            "source": payload.stage_path.as_posix(),
        }
    )

    # TODO: Add hooks for endpoints for integration with TensorBoard etc

    env_vars = payload.env_vars
    endpoints: list[dict[str, Any]] = []

    if target_instances > 1:
        # Update environment variables for multi-node job
        env_vars.update(constants.RAY_PORTS)
        env_vars[constants.ENABLE_HEALTH_CHECKS_ENV_VAR] = constants.ENABLE_HEALTH_CHECKS
        env_vars[constants.MIN_INSTANCES_ENV_VAR] = str(min_instances)

        # Define Ray endpoints for intra-service instance communication
        ray_endpoints: list[dict[str, Any]] = [
            {"name": "ray-client-server-endpoint", "port": 10001, "protocol": "TCP"},
            {"name": "ray-gcs-endpoint", "port": 12001, "protocol": "TCP"},
            {"name": "ray-dashboard-grpc-endpoint", "port": 12002, "protocol": "TCP"},
            {"name": "ray-dashboard-endpoint", "port": 12003, "protocol": "TCP"},
            {"name": "ray-object-manager-endpoint", "port": 12011, "protocol": "TCP"},
            {"name": "ray-node-manager-endpoint", "port": 12012, "protocol": "TCP"},
            {"name": "ray-runtime-agent-endpoint", "port": 12013, "protocol": "TCP"},
            {"name": "ray-dashboard-agent-grpc-endpoint", "port": 12014, "protocol": "TCP"},
            {"name": "ephemeral-port-range", "portRange": "32768-60999", "protocol": "TCP"},
            {"name": "ray-worker-port-range", "portRange": "12031-13000", "protocol": "TCP"},
        ]
        endpoints.extend(ray_endpoints)

    metrics = []
    if enable_metrics:
        # https://docs.snowflake.com/en/developer-guide/snowpark-container-services/monitoring-services#label-spcs-available-platform-metrics
        metrics = [
            "system",
            "status",
            "network",
            "storage",
        ]

    spec_dict = {
        "containers": [
            {
                "name": constants.DEFAULT_CONTAINER_NAME,
                "image": image_spec.container_image,
                "command": ["/usr/local/bin/_entrypoint.sh"],
                "args": [
                    (stage_mount.joinpath(v).as_posix() if isinstance(v, PurePath) else v) for v in payload.entrypoint
                ]
                + (args or []),
                "env": env_vars,
                "volumeMounts": volume_mounts,
                "resources": {
                    "requests": resource_requests,
                    "limits": resource_limits,
                },
            },
        ],
        "volumes": volumes,
    }

    if target_instances > 1:
        spec_dict.update(
            {
                "resourceManagement": {
                    "controlPolicy": {
                        "startupOrder": {
                            "type": "FirstInstance",
                        },
                    },
                },
            }
        )
    if endpoints:
        spec_dict["endpoints"] = endpoints
    if metrics:
        spec_dict.update(
            {
                "platformMonitor": {
                    "metricConfig": {
                        "groups": metrics,
                    },
                },
            }
        )

    # Assemble into service specification dict
    spec = {"spec": spec_dict}

    return spec


def merge_patch(base: Any, patch: Any, display_name: str = "") -> Any:
    """
    Implements a modified RFC7386 JSON Merge Patch
    https://datatracker.ietf.org/doc/html/rfc7386

    Behavior differs from the RFC in the following ways:
      1. Empty nested dictionaries resulting from the patch are treated as None and are pruned
      2. Attempts to merge lists of dicts using a merge key (default "name").
         See _merge_lists_of_dicts for details on list merge behavior.

    Args:
        base: The base object to patch.
        patch: The patch object.
        display_name: The name of the patch object for logging purposes.

    Returns:
        The patched object.
    """
    if type(base) is not type(patch):
        if base is not None:
            logging.warning(f"Type mismatch while merging {display_name} (base={type(base)}, patch={type(patch)})")
        return patch
    elif isinstance(patch, list) and all(isinstance(v, dict) for v in base + patch):
        # TODO: Should we prune empty lists?
        return _merge_lists_of_dicts(base, patch, display_name=display_name)
    elif not isinstance(patch, dict) or len(patch) == 0:
        return patch

    result = dict(base)  # Shallow copy
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            merge_result = merge_patch(result.get(key, None), value, display_name=f"{display_name}.{key}")
            if isinstance(merge_result, dict) and len(merge_result) == 0:
                result.pop(key, None)
            else:
                result[key] = merge_result

    return result


def _merge_lists_of_dicts(
    base: list[dict[str, Any]],
    patch: list[dict[str, Any]],
    merge_key: str = "name",
    display_name: str = "",
) -> list[dict[str, Any]]:
    """
    Attempts to merge lists of dicts by matching on a merge key (default "name").
    - If the merge key is missing, the behavior falls back to overwriting the list.
    - If the merge key is present, the behavior is to match the list elements based on the
        merge key and preserving any unmatched elements from the base list.
    - Matched entries may be dropped in the following way(s):
        1. The matching patch entry has a None key entry, e.g. { "name": "foo", None: None }.

    Args:
        base: The base list of dicts.
        patch: The patch list of dicts.
        merge_key: The key to use for merging.
        display_name: The name of the patch object for logging purposes.

    Returns:
        The merged list of dicts if merging successful, else returns the patch list.
    """
    if any(merge_key not in d for d in base + patch):
        logging.warning(f"Missing merge key {merge_key} in {display_name}. Falling back to overwrite behavior.")
        return patch

    # Build mapping of merge key values to list elements for the base list
    result = {d[merge_key]: d for d in base}
    if len(result) != len(base):
        logging.warning(f"Duplicate merge key {merge_key} in {display_name}. Falling back to overwrite behavior.")
        return patch

    # Apply patches
    for d in patch:
        key = d[merge_key]

        # Removal case 1: `None` key in patch entry
        if None in d:
            result.pop(key, None)
            continue

        # Apply patch
        if key in result:
            d = merge_patch(
                result[key],
                d,
                display_name=f"{display_name}[{merge_key}={d[merge_key]}]",
            )
            # TODO: Should we drop the item if the patch result is empty save for the merge key?
            #       Can check `d.keys() <= {merge_key}`
        result[key] = d

    return list(result.values())


def _get_runtime_image_tag() -> str:
    """
    Detect runtime image tag from container environment.

    Checks in order:
    1. Environment variable MLRS_CONTAINER_IMAGE_TAG
    2. Falls back to hardcoded default

    Returns:
        str: The runtime image tag to use for job containers
    """
    env_tag = os.environ.get(constants.RUNTIME_IMAGE_TAG_ENV_VAR)
    if env_tag:
        logging.debug(f"Using runtime image tag from environment: {env_tag}")
        return env_tag

    # Fall back to default
    logging.debug(f"Using default runtime image tag: {constants.DEFAULT_IMAGE_TAG}")
    return constants.DEFAULT_IMAGE_TAG
