import json
import sys
from typing import cast

from snowflake import snowpark
from snowflake.ml.jobs._utils import feature_flags, query_helper


def get_runtime_image(session: snowpark.Session, compute_pool: str, runtime_environment: str | None = None) -> str:
    runtime_environment = runtime_environment or ""
    rows = query_helper.run_query(session, f"CALL SYSTEM$GET_ML_JOB_RUNTIME('{compute_pool}', '{runtime_environment}')")
    if not rows or not rows[0][0]:
        raise ValueError("Failed to get any available runtime image")
    image = cast(str, rows[0][0])
    # for cre reference , return the reference as is - the CRE resolution will be handled in GS side
    if image.lower().startswith("cre@"):
        return image
    if ":" not in image:
        raise ValueError(f"image {image} is not a valid runtime image")
    return image


def resolve_runtime_image(session: snowpark.Session, compute_pool: str, runtime_environment: str | None = None) -> str:
    """Resolve the concrete runtime image to run on from an optional runtime selector.

    Args:
        session: The Snowpark session to use.
        compute_pool: The compute pool the image must be available for.
        runtime_environment: The requested runtime, as an image tag or a full image URL. When runtime
            versions are enabled, it is sent alongside the local Python version so the server can pick
            a matching image.

    Returns:
        The resolved runtime image.
    """
    if feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS.is_enabled():
        runtime_dict = {"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}"}
        if runtime_environment is not None:
            runtime_dict["runtimeEnvironment"] = runtime_environment
        runtime_environment = json.dumps(runtime_dict)

    return get_runtime_image(session, compute_pool, runtime_environment)
