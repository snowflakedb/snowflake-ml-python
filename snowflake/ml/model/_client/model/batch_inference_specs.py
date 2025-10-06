from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SaveMode(str, Enum):
    """Save mode options for batch inference output.

    Determines the behavior when files already exist in the output location.

    OVERWRITE: Remove existing files and write new results.

    ERROR: Raise an error if files already exist in the output location.
    """

    OVERWRITE = "overwrite"
    ERROR = "error"


class OutputSpec(BaseModel):
    stage_location: str
    mode: SaveMode = SaveMode.ERROR


class JobSpec(BaseModel):
    image_repo: Optional[str] = None
    job_name: Optional[str] = None
    num_workers: Optional[int] = None
    function_name: Optional[str] = None
    force_rebuild: bool = False
    max_batch_rows: int = 1024
    warehouse: Optional[str] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    gpu_requests: Optional[str] = None
    replicas: Optional[int] = None
