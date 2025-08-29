from typing import Optional, Union

from pydantic import BaseModel


class InputSpec(BaseModel):
    input_stage_location: str
    input_file_pattern: str = "*"


class OutputSpec(BaseModel):
    output_stage_location: str
    output_file_prefix: Optional[str] = None
    completion_filename: str = "_SUCCESS"


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
