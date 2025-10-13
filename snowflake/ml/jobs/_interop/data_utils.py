import io
import json
from typing import Any, Literal, Optional, Protocol, Union, cast, overload

from snowflake import snowpark
from snowflake.ml.jobs._interop import dto_schema


class StageFileWriter(io.IOBase):
    """
    A context manager IOBase implementation that proxies writes to an internal BytesIO
    and uploads to Snowflake stage on close.
    """

    def __init__(self, session: snowpark.Session, path: str) -> None:
        self._session = session
        self._path = path
        self._buffer = io.BytesIO()
        self._closed = False
        self._exception_occurred = False

    def write(self, data: Union[bytes, bytearray]) -> int:
        """Write data to the internal buffer."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self._buffer.write(data)

    def close(self, write_contents: bool = True) -> None:
        """Close the file and upload the buffer contents to the stage."""
        if not self._closed:
            # Only upload if buffer has content and no exception occurred
            if write_contents and self._buffer.tell() > 0:
                self._buffer.seek(0)
                self._session.file.put_stream(self._buffer, self._path)
            self._buffer.close()
            self._closed = True

    def __enter__(self) -> "StageFileWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        exception_occurred = exc_type is not None
        self.close(write_contents=not exception_occurred)

    @property
    def closed(self) -> bool:
        return self._closed

    def writable(self) -> bool:
        return not self._closed

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return not self._closed


def _is_stage_path(path: str) -> bool:
    return path.startswith("@") or path.startswith("snow://")


def open_stream(path: str, mode: str = "rb", session: Optional[snowpark.Session] = None) -> io.IOBase:
    if _is_stage_path(path):
        if session is None:
            raise ValueError("Session is required when opening a stage path")
        if "r" in mode:
            stream: io.IOBase = session.file.get_stream(path)  # type: ignore[assignment]
            return stream
        elif "w" in mode:
            return StageFileWriter(session, path)
        else:
            raise ValueError(f"Unsupported mode '{mode}' for stage path")
    else:
        result: io.IOBase = open(path, mode)  # type: ignore[assignment]
        return result


class DtoCodec(Protocol):
    @overload
    @staticmethod
    def decode(stream: io.IOBase, as_dict: Literal[True]) -> dict[str, Any]:
        ...

    @overload
    @staticmethod
    def decode(stream: io.IOBase, as_dict: Literal[False] = False) -> dto_schema.ResultDTO:
        ...

    @staticmethod
    def decode(stream: io.IOBase, as_dict: bool = False) -> Union[dto_schema.ResultDTO, dict[str, Any]]:
        pass

    @staticmethod
    def encode(dto: dto_schema.ResultDTO) -> bytes:
        pass


class JsonDtoCodec(DtoCodec):
    @overload
    @staticmethod
    def decode(stream: io.IOBase, as_dict: Literal[True]) -> dict[str, Any]:
        ...

    @overload
    @staticmethod
    def decode(stream: io.IOBase, as_dict: Literal[False] = False) -> dto_schema.ResultDTO:
        ...

    @staticmethod
    def decode(stream: io.IOBase, as_dict: bool = False) -> Union[dto_schema.ResultDTO, dict[str, Any]]:
        data = cast(dict[str, Any], json.load(stream))
        if as_dict:
            return data
        return dto_schema.ResultDTO.model_validate(data)

    @staticmethod
    def encode(dto: dto_schema.ResultDTO) -> bytes:
        # Temporarily extract the value to avoid accidentally applying model_dump() on it
        result_value = dto.value
        dto.value = None  # Clear value to avoid serializing it in the model_dump
        result_dict = dto.model_dump()
        result_dict["value"] = result_value  # Put back the value
        return json.dumps(result_dict).encode("utf-8")
