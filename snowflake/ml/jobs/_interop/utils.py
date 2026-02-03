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
    ResultDTO,
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


logger = logging.getLogger(__name__)


def save_result(result: ExecutionResult, path: str, session: Optional[snowpark.Session] = None) -> None:
    """
    Save the result to a file.
    """
    result_dto = ResultDTO(
        success=result.success,
        value=result.value,
    )

    try:
        # Try to encode result directly
        payload = DEFAULT_CODEC.encode(result_dto)
    except TypeError:
        result_dto.value = None  # Remove raw value to avoid serialization error
        result_dto.metadata = _get_metadata(result.value)  # Add metadata for client fallback on protocol mismatch
        try:
            path_dir = PurePath(path).parent.as_posix()
            protocol_info = DEFAULT_PROTOCOL.save(result.value, path_dir, session=session)
            result_dto.protocol = protocol_info

        except Exception as e:
            logger.warning(f"Error dumping result value: {repr(e)}")
            result_dto.serialize_error = repr(e)

        # Encode the modified result DTO
        payload = DEFAULT_CODEC.encode(result_dto)

    with data_utils.open_stream(path, "wb", session=session) as stream:
        stream.write(payload)


def load_result(
    path: str, session: Optional[snowpark.Session] = None, path_transform: Optional[Callable[[str], str]] = None
) -> ExecutionResult:
    """Load the result from a file on a Snowflake stage."""
    try:
        with data_utils.open_stream(path, "r", session=session) as stream:
            # Load the DTO as a dict for easy fallback to legacy loading if necessary
            dto_dict = DEFAULT_CODEC.decode(stream, as_dict=True)
    except UnicodeDecodeError:
        # Path may be a legacy result file (cloudpickle)
        # TODO: Re-use the stream
        assert session is not None
        return legacy.load_legacy_result(session, path)

    try:
        dto = ResultDTO.model_validate(dto_dict)
    except pydantic.ValidationError as e:
        if "success" in dto_dict:
            assert session is not None
            if path.endswith(".json"):
                path = os.path.splitext(path)[0] + ".pkl"
            return legacy.load_legacy_result(session, path, result_json=dto_dict)
        raise ValueError("Invalid result schema") from e

    # Try loading data from file using the protocol info
    result_value = None
    data_load_error = None
    if dto.protocol is not None:
        try:
            logger.debug(f"Loading result value with protocol {dto.protocol}")
            result_value = DEFAULT_PROTOCOL.load(dto.protocol, session=session, path_transform=path_transform)
        except sp_exceptions.SnowparkSQLException:
            raise  # Data retrieval errors should be bubbled up
        except Exception as e:
            logger.debug(f"Error loading result value with protocol {dto.protocol}: {repr(e)}")
            data_load_error = e

    # Wrap serialize_error in a TypeError
    if dto.serialize_error:
        serialize_error = TypeError("Original result serialization failed with error: " + dto.serialize_error)
        if data_load_error:
            data_load_error.__context__ = serialize_error
        else:
            data_load_error = serialize_error

    # Prepare to assemble the final result
    result_value = result_value if result_value is not None else dto.value
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
