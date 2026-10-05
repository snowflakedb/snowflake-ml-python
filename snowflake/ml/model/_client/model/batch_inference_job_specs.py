from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import TypedDict

from snowflake.ml.model import inference_engine as inference_engine_module


class SaveMode(str, Enum):
    """Save mode options for batch inference output.

    Determines the behavior when files already exist in the output stage location.

    OVERWRITE: Remove existing files and write new results.

    ERROR: Raise an error if files already exist in the output stage location.
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
    """Input block of the batch inference job specification.

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
    """

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] | None = None
    column_handling: dict[str, ColumnHandlingOptions] | None = None
    partition_column: str | None = None


class OutputSpec(BaseModel):
    """Output block of the batch inference job specification.

    Results are written under ``<stage_location>/<job_name>/``.

    Attributes:
        stage_location (str): The stage path under which batch inference results will be saved.
            This should be a full path including the stage with @ prefix. For example,
            '@My_DB.PUBLIC.MY_STAGE/some/path/'. Only Snowflake internal stages are supported.
        mode (SaveMode): The save mode that determines behavior when files already exist
            at the output stage location. Defaults to SaveMode.ERROR.
    """

    model_config = ConfigDict(extra="forbid")

    stage_location: str
    mode: SaveMode = SaveMode.ERROR


class ResourcesSpec(BaseModel):
    """Resources block of the batch inference job specification.

    Attributes:
        cpu_requests (Optional[str]): The cpu limit for CPU based inference. Can be an integer,
            fractional or string values. If None, we attempt to utilize all the vCPU of the node.
        memory_requests (Optional[str]): The memory limit for inference. Can be an integer
            or a fractional value, but requires a unit (GiB, MiB). If None, we attempt to utilize
            all the memory of the node.
        gpu_requests (Optional[str]): The gpu limit for GPU based inference. Can be integer or
            string values. Use CPU if None.
    """

    model_config = ConfigDict(extra="forbid")

    cpu_requests: str | None = None
    memory_requests: str | None = None
    gpu_requests: str | None = None


class EngineOptions(BaseModel):
    """``inference.engine_options`` sub-block of the job specification.

    Attributes:
        engine (Optional[InferenceEngine]): The inference engine to use.
        engine_args_override (Optional[list[str]]): Arguments passed through to the inference engine.
    """

    model_config = ConfigDict(extra="forbid")

    engine: inference_engine_module.InferenceEngine | None = None
    engine_args_override: list[str] | None = None

    @field_validator("engine", mode="before")
    @classmethod
    def _coerce_engine(cls, value: Any) -> inference_engine_module.InferenceEngine | None:
        # Accept an InferenceEngine member or a case-insensitive value/name string
        # (e.g. "vllm", "VLLM", "python_generic") rather than only the exact enum value.
        if value is None:
            return None
        return inference_engine_module.InferenceEngine.from_value(value)

    @field_serializer("engine")
    def _serialize_engine(self, engine: inference_engine_module.InferenceEngine | None) -> str | None:
        # Emit the enum member name (e.g. "VLLM") rather than its lower-case value,
        # to match the canonical spelling used in the EXECUTE INFERENCE JOB SERVICE spec.
        return engine.name if engine is not None else None


class AdapterSpec(BaseModel):
    """Serialized adapter identity nested under ``inference.adapters``.

    ``alias`` is optional. Dict keys from the Python API set it; list items and
    adapter-as-target omit it so the server can default to ``{name}/VERSIONS/{version}``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    alias: str | None = None


class InferenceSpec(BaseModel):
    """Inference block of the batch inference job specification.

    Attributes:
        num_workers (Optional[int]): The number of workers to run the inference service for handling
            requests in parallel within an instance of the service. Auto determined if None.
        max_batch_rows (Optional[int]): Maximum number of rows to process in a single batch.
            Auto determined if None. Larger values may improve throughput.
        engine_options (Optional[EngineOptions]): Options for a custom inference engine.
        adapters (Optional[dict[str, ModelVersion] | list[ModelVersion]]): Adapter versions for
            this job. A dict maps job alias (``input.params.model``) to adapter versions; keys
            become ``alias`` on each emitted list item. A nonempty list and adapter-as-target omit
            ``alias``. Empty ``{}`` / ``[]`` are treated as unset so the YAML key is omitted.
            ``run_batch`` serializes these to name/version identities before writing the job spec.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    num_workers: int | None = None
    max_batch_rows: int | None = None
    engine_options: EngineOptions | None = None
    adapters: dict[str, Any] | list[Any] | None = None

    @field_validator("adapters", mode="before")
    @classmethod
    def _empty_adapters_to_none(cls, value: Any) -> Any:
        if value == [] or value == {}:
            return None
        return value

    @field_serializer("adapters")
    def _serialize_adapters(self, value: Any) -> list[dict[str, Any]] | None:
        if not value:
            return None
        if not isinstance(value, list) or not all(isinstance(item, AdapterSpec) for item in value):
            raise ValueError(
                "batch inference: adapter versions cannot be written into the job spec until they are "
                "resolved to name and version."
            )
        return [item.model_dump(mode="json", exclude_none=True) for item in value]


def split_inference_adapters(
    inference_spec: InferenceSpec | None,
) -> tuple[InferenceSpec | None, dict[str, Any] | list[Any] | None]:
    """Peel ``adapters`` off ``inference_spec`` without mutating the caller object.

    Args:
        inference_spec: Existing inference block, or None.

    Returns:
        The spec with ``adapters`` cleared (or None if that leaves an empty spec), and the
        peeled adapters value (or None when unset).
    """
    if inference_spec is None or inference_spec.adapters is None:
        return inference_spec, None
    raw = inference_spec.adapters
    stripped = inference_spec.model_copy(update={"adapters": None})
    if not stripped.model_dump(exclude_none=True):
        return None, raw
    return stripped, raw


def inference_spec_with_adapters(
    inference_spec: InferenceSpec | None,
    adapters: dict[str, Any] | list[Any] | None,
) -> InferenceSpec | None:
    """Return ``inference_spec`` with ``adapters`` set when nonempty.

    Args:
        inference_spec: Existing inference block, or None.
        adapters: Adapter versions or serialized identities. Empty or None leaves
            ``inference_spec`` unchanged.

    Returns:
        The original spec when ``adapters`` is empty, otherwise a spec that includes them.
    """
    if not adapters:
        return inference_spec
    if inference_spec is None:
        return InferenceSpec(adapters=adapters)
    return inference_spec.model_copy(update={"adapters": adapters})


class ImageBuildSpec(BaseModel):
    """Image-build block of the batch inference job specification.

    Attributes:
        image_repo (Optional[str]): Container image repository for the inference job.
            If not specified, uses the default repository.
        force_rebuild (bool): Whether to force rebuilding the container image even if
            it already exists. Defaults to False.
    """

    model_config = ConfigDict(extra="forbid")

    image_repo: str | None = None
    force_rebuild: bool = False
