from typing import Optional, Union

from pydantic import BaseModel


class InputSpec(BaseModel):
    stage_location: str


class OutputSpec(BaseModel):
    stage_location: str


class JobSpec(BaseModel):
    image_repo: Optional[str] = None
    job_name: Optional[str] = None
    num_workers: Optional[int] = None
    function_name: Optional[str] = None
    gpu: Optional[Union[str, int]] = None
    force_rebuild: bool = False
    max_batch_rows: int = 1024
    warehouse: Optional[str] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    replicas: Optional[int] = None
