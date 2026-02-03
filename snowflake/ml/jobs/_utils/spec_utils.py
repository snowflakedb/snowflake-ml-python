from snowflake import snowpark
from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.jobs._utils import constants, query_helper, types


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
