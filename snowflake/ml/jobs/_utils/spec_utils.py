import logging
import sys
from typing import Literal, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.jobs._utils import constants, query_helper, types
from snowflake.ml.jobs._utils.runtime_env_utils import RuntimeEnvironmentsDict


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
