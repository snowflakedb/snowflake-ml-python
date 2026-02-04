import io
import logging
import os
import traceback
from pathlib import PurePath
from typing import Any, Callable, Optional

import pydantic

from snowflake import snowpark
from snowflake.ml.jobs._interop import data_utils, exception_utils, legacy, protocols
from snowflake.ml.jobs._interop.dto_schema import (
    ExceptionMetadata,
    PayloadDTO,
    ResultDTO,
    ResultDTOAdapter,
    ResultMetadata,
)
from snowflake.ml.jobs._interop.results import ExecutionResult, LoadedExecutionResult
from snowflake.snowpark import exceptions as sp_exceptions

DEFAULT_CODEC = data_utils.JsonDtoCodec
DEFAULT_PROTOCOL = protocols.AutoProtocol()
DEFAULT_PROTOCOL.try_register_protocol(protocols.CloudPickleProtocol)
DEFAULT_PROTOCOL.try_register_protocol(protocols.ArrowTableProtocol)
DEFAULT_PROTOCOL.try_register_protocol(protocols.PandasDataFrameProtocol)
DEFAULT_PROTOCOL.try_register_protocol(protocols.NumpyArrayProtocol)

# Constants for argument encoding
_MAX_INLINE_SIZE = 1024 * 1024  # 1MB - https://docs.snowflake.com/en/user-guide/query-size-limits

logger = logging.getLogger(__name__)


def save(
    value: Any,
    path: str,
    session: Optional[snowpark.Session] = None,
    max_inline_size: int = 0,
) -> Optional[bytes]:
    """
    Serialize a value. Returns inline bytes if small enough, else writes to file.

    Args:
        value: The value to serialize. If ExecutionResult, creates ResultDTO with success flag.
        path: Full file path for writing the DTO (if needed). Protocol data saved to path's parent.
        session: Snowpark session for stage operations.
        max_inline_size: Max bytes for inline return. 0 = always write to file.

    Returns:
        Encoded bytes if <= max_inline_size, else None (written to file).

    Raises:
        Exception: If session validation fails during serialization.
    """
    if isinstance(value, ExecutionResult):
        dto: PayloadDTO = ResultDTO(success=value.success, value=value.value)
        raw_value = value.value
    else:
        dto = PayloadDTO(value=value)
        raw_value = value

    try:
        payload = DEFAULT_CODEC.encode(dto)
    except TypeError:
        dto.value = None  # Remove raw value to avoid serialization error
        if isinstance(dto, ResultDTO):
            # Metadata enables client fallback display when result can't be deserialized (protocol mismatch)..
            dto.metadata = _get_metadata(raw_value)
        try:
            path_dir = PurePath(path).parent.as_posix()
            protocol_info = DEFAULT_PROTOCOL.save(raw_value, path_dir, session=session)
            dto.protocol = protocol_info

        except Exception as e:
            logger.warning(f"Error dumping result value: {repr(e)}")
            # We handle serialization failures differently based on the DTO type:
            # 1. Job Results (ResultDTO): Allow a "soft-fail."
            #    Since the job has already executed, we return the serialization error
            #    to the client so they can debug the output or update their protocol version.
            # 2. Input Arguments: Trigger a "hard-fail."
            #    If arguments cannot be saved, the job script cannot run. We raise
            #    an immediate exception to prevent execution with invalid state.
            if not isinstance(dto, ResultDTO):
                raise
            dto.serialize_error = repr(e)

        # Encode the modified DTO
        payload = DEFAULT_CODEC.encode(dto)

    if not isinstance(dto, ResultDTO) and len(payload) <= max_inline_size:
        return payload

    with data_utils.open_stream(path, "wb", session=session) as stream:
        stream.write(payload)
    return None


save_result = save  # Backwards compatibility


def load(
    path_or_data: str,
    session: Optional[snowpark.Session] = None,
    path_transform: Optional[Callable[[str], str]] = None,
) -> Any:
    """Load data from a file path or inline string."""

    try:
        with data_utils.open_stream(path_or_data, "r", session=session) as stream:
            # Load the DTO as a dict for easy fallback to legacy loading if necessary
            dto_dict = DEFAULT_CODEC.decode(stream, as_dict=True)
    # the exception could be OSError or BlockingIOError(the file name is too long)
    except OSError as e:
        # path_or_data might be inline data
        try:
            dto_dict = DEFAULT_CODEC.decode(io.StringIO(path_or_data), as_dict=True)
        except Exception:
            raise e
    except UnicodeDecodeError:
        # Path may be a legacy result file (cloudpickle)
        assert session is not None
        return legacy.load_legacy_result(session, path_or_data)

    try:
        dto = ResultDTOAdapter.validate_python(dto_dict)
    except pydantic.ValidationError as e:
        if "success" in dto_dict:
            assert session is not None
            if path_or_data.endswith(".json"):
                path_or_data = os.path.splitext(path_or_data)[0] + ".pkl"
            return legacy.load_legacy_result(session, path_or_data, result_json=dto_dict)
        raise ValueError("Invalid result schema") from e

    # Try loading data from file using the protocol info
    payload_value = None
    data_load_error = None
    if dto.protocol is not None:
        try:
            logger.debug(f"Loading result value with protocol {dto.protocol}")
            payload_value = DEFAULT_PROTOCOL.load(dto.protocol, session=session, path_transform=path_transform)
        except sp_exceptions.SnowparkSQLException:
            raise  # Data retrieval errors should be bubbled up
        except Exception as e:
            logger.debug(f"Error loading result value with protocol {dto.protocol}: {repr(e)}")
            # Error handling strategy depends on the DTO type:
            # 1. ResultDTO: Soft-fail. The job has already finished.
            #    We package the load error into the result so the client can
            #    debug or adjust their protocol version to retrieve the output.
            # 2. PayloadDTO : Raise a hard error. If arguments cannot be
            #    loaded, the job cannot run. We abort early to prevent execution.
            if not isinstance(dto, ResultDTO):
                raise
            data_load_error = e

    # Prepare to assemble the final result
    if not isinstance(dto, ResultDTO):
        return payload_value if payload_value is not None else dto.value

    if dto.serialize_error:
        serialize_error = TypeError("Original result serialization failed with error: " + dto.serialize_error)
        if data_load_error:
            data_load_error.__context__ = serialize_error
        else:
            data_load_error = serialize_error

    result_value = payload_value if payload_value is not None else dto.value
    if not dto.success and result_value is None:
        # Try to reconstruct exception from metadata if available
        if isinstance(dto.metadata, ExceptionMetadata):
            logger.debug(f"Reconstructing exception from metadata {dto.metadata}")
            result_value = exception_utils.build_exception(
                type_str=dto.metadata.type,
                message=dto.metadata.message,
                traceback=dto.metadata.traceback,
                original_repr=dto.metadata.repr,
            )
        # Generate a generic error if we still don't have a value,
        # attaching the data load error if any
        if result_value is None:
            result_value = exception_utils.RemoteError("Unknown remote error")
            result_value.__cause__ = data_load_error

    return LoadedExecutionResult(
        success=dto.success,
        value=result_value,
        load_error=data_load_error,
    )


def _get_metadata(value: Any) -> ResultMetadata:
    type_name = f"{type(value).__module__}.{type(value).__name__}"
    if isinstance(value, BaseException):
        return ExceptionMetadata(
            type=type_name,
            repr=repr(value),
            message=str(value),
            traceback="".join(traceback.format_tb(value.__traceback__)),
        )
    return ResultMetadata(
        type=type_name,
        repr=repr(value),
    )
