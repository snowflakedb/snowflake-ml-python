from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, model_validator
from typing_extensions import TypedDict


class SaveMode(str, Enum):
    """Save mode options for batch inference output.

    Determines the behavior when files already exist in the output location.

    OVERWRITE: Remove existing files and write new results.

    ERROR: Raise an error if files already exist in the output location.
    """

    OVERWRITE = "overwrite"
    ERROR = "error"


class InputFormat(str, Enum):
    """The format of the input column data."""

    FULL_STAGE_PATH = "full_stage_path"


class FileEncoding(str, Enum):
    """The encoding of the file content that will be passed to the custom model."""

    RAW_BYTES = "raw_bytes"
    BASE64 = "base64"
    BASE64_DATA_URL = "base64_data_url"


class ColumnHandlingOptions(TypedDict):
    """Options for handling specific columns during run_batch for file I/O."""

    input_format: InputFormat
    convert_to: FileEncoding


class InputSpec(BaseModel):
    """Specification for batch inference input options.

    Defines optional configuration for processing input data during batch inference.

    Attributes:
        params (Optional[dict[str, Any]]): Optional dictionary of model inference parameters
            (e.g., temperature, top_k for LLMs). These are passed as keyword arguments to the
            model's inference method. Defaults to None.
        column_handling (Optional[dict[str, ColumnHandlingOptions]]): Optional dictionary
            specifying how to handle specific columns during file I/O. Maps column names to their
            input format and file encoding configuration.
        partition_column (Optional[str]): Optional column name to use for partitioning the input
            data. When set, the batch inference job will partition the data by this column.
            Defaults to None.

    Example:
        >>> input_spec = InputSpec(
        ...     params={"temperature": 0.7, "top_k": 50},
        ...     column_handling={
        ...         "image_col": {
        ...             "input_format": InputFormat.FULL_STAGE_PATH,
        ...             "convert_to": FileEncoding.BASE64
        ...         }
        ...     }
        ... )
    """

    params: Optional[dict[str, Any]] = None
    column_handling: Optional[dict[str, ColumnHandlingOptions]] = None
    partition_column: Optional[str] = None


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
            Mutually exclusive with job_name_prefix.
        job_name_prefix (Optional[str]): Prefix for auto-generated job names. When set, the job name
            will be generated as "<PREFIX>_<UUID>". This is useful for task integration where each
            repeated execution needs a unique name with a recognizable prefix.
            Mutually exclusive with job_name.
        num_workers (Optional[int]): The number of workers to run the inference service for handling
            requests in parallel within an instance of the service. By default, it is set to 2*vCPU+1
            of the node for CPU based inference and 1 for GPU based inference. For GPU based inference,
            please see best practices before playing with this value.
        function_name (Optional[str]): Name of the specific function to call for inference.
            Required when the model has multiple inference functions.
        force_rebuild (bool): Whether to force rebuilding the container image even if
            it already exists. Defaults to False.
        max_batch_rows (Optional[int]): Maximum number of rows to process in a single batch.
            Auto determined if None. Larger values may improve throughput.
        warehouse (Optional[str]): Snowflake warehouse to use for the batch inference job.
            If not specified, uses the session's current warehouse.
        cpu_requests (Optional[str]): The cpu limit for CPU based inference. Can be an integer,
            fractional or string values. If None, we attempt to utilize all the vCPU of the node.
        memory_requests (Optional[str]): The memory limit for inference. Can be an integer
            or a fractional value, but requires a unit (GiB, MiB). If None, we attempt to utilize all
            the memory of the node.
        gpu_requests (Optional[str]): The gpu limit for GPU based inference. Can be integer or
            string values. Use CPU if None.
        replicas (Optional[int]): Number of SPCS job nodes used for distributed inference.
            If not specified, defaults to 1 replica.
        block (bool): Whether the SPCS batch inference job runs synchronously
            or asynchronously. When True, the call blocks until the job completes. When
            False, the call returns immediately after job creation. Defaults to False.

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
    job_name_prefix: Optional[str] = None
    num_workers: Optional[int] = None
    function_name: Optional[str] = None
    force_rebuild: bool = False
    max_batch_rows: Optional[int] = None
    warehouse: Optional[str] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    gpu_requests: Optional[str] = None
    replicas: Optional[int] = None
    block: bool = False

    @model_validator(mode="after")
    def _validate_job_name_exclusivity(self) -> "JobSpec":
        if self.job_name is not None and self.job_name_prefix is not None:
            raise ValueError("job_name and job_name_prefix are mutually exclusive. Please specify only one or neither.")
        return self
