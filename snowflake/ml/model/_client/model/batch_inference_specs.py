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
    """Specification for batch inference output.

    Defines where the inference results should be written and how to handle
    existing files at the output location.

    Attributes:
        stage_location (str): The stage path where batch inference results will be saved.
            This should be a full path including the stage with @ prefix. For example,
            '@My_DB.PUBLIC.MY_STAGE/someth/path/'. A non-existent directory will be re-created.
            Only Snowflake internal stages are supported at this moment.
        mode (SaveMode): The save mode that determines behavior when files already exist
            at the output location. Defaults to SaveMode.ERROR which raises an error
            if files exist. Can be set to SaveMode.OVERWRITE to replace existing files.

    Example:
        >>> output_spec = OutputSpec(
        ...     stage_location="@My_DB.PUBLIC.MY_STAGE/someth/path/",
        ...     mode=SaveMode.OVERWRITE
        ... )
    """

    stage_location: str
    mode: SaveMode = SaveMode.ERROR


class JobSpec(BaseModel):
    """Specification for batch inference job execution.

    Defines the compute resources, job settings, and execution parameters
    for running batch inference jobs in Snowflake.

    Attributes:
        image_repo (Optional[str]): Container image repository for the inference job.
            If not specified, uses the default repository.
        job_name (Optional[str]): Custom name for the batch inference job.
            If not provided, a name will be auto-generated in the form of "BATCH_INFERENCE_<UUID>".
        num_workers (Optional[int]): The number of workers to run the inference service for handling
            requests in parallel within an instance of the service. By default, it is set to 2*vCPU+1
            of the node for CPU based inference and 1 for GPU based inference. For GPU based inference,
            please see best practices before playing with this value.
        function_name (Optional[str]): Name of the specific function to call for inference.
            Required when the model has multiple inference functions.
        force_rebuild (bool): Whether to force rebuilding the container image even if
            it already exists. Defaults to False.
        max_batch_rows (int): Maximum number of rows to process in a single batch.
            Defaults to 1024. Larger values may improve throughput.
        warehouse (Optional[str]): Snowflake warehouse to use for the batch inference job.
            If not specified, uses the session's current warehouse.
        cpu_requests (Optional[str]): The cpu limit for CPU based inference. Can be an integer,
            fractional or string values. If None, we attempt to utilize all the vCPU of the node.
        memory_requests (Optional[str]): The memory limit for inference. Can be an integer
            or a fractional value, but requires a unit (GiB, MiB). If None, we attempt to utilize all
            the memory of the node.
        gpu_requests (Optional[str]): The gpu limit for GPU based inference. Can be integer or
            string values. Use CPU if None.
        replicas (Optional[int]): Number of job replicas to run for high availability.
            If not specified, defaults to 1 replica.

    Example:
        >>> job_spec = JobSpec(
        ...     job_name="my_inference_job",
        ...     num_workers=4,
        ...     cpu_requests="2",
        ...     memory_requests="8Gi",
        ...     max_batch_rows=2048
        ... )
    """

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
