from typing import Optional, cast

from snowflake import snowpark
from snowflake.ml.jobs._utils import query_helper


def get_runtime_image(
    session: snowpark.Session, compute_pool: str, runtime_environment: Optional[str] = None
) -> Optional[str]:
    runtime_environment = runtime_environment if runtime_environment else ""
    rows = query_helper.run_query(session, f"CALL SYSTEM$GET_ML_JOB_RUNTIME('{compute_pool}', '{runtime_environment}')")
    if not rows:
        raise ValueError("Failed to get any available runtime image")
    image = rows[0][0]
    url, tag = image.rsplit(":", 1)
    if url is None or tag is None:
        raise ValueError(f"image {image} is not a valid runtime image")
    return cast(str, image) if image else None
