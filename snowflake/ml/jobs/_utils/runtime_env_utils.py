from typing import Optional, cast

from snowflake import snowpark
from snowflake.ml.jobs._utils import query_helper


def get_runtime_image(session: snowpark.Session, compute_pool: str, runtime_environment: Optional[str] = None) -> str:
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
