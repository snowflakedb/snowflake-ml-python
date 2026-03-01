import io
from typing import Any, Iterable, Iterator, Optional, TextIO


class OutputTee(TextIO):
    """A class that duplicates string writes to multiple file-like objects."""

    def __init__(self, *streams: TextIO) -> None:
        """Initialize the OutputTee with a list of output streams."""
        super().__init__()
        self.streams = streams
        if not self.writable():
            raise ValueError("All inputs to OutputTee must be writable.")

    def close(self) -> None:
        """Does not do anything. It is implemented this way because some of the streams may be in use elsewhere.
        It is the responsibility of the caller to close any streams that need to be closed."""
        pass

    def fileno(self) -> int:
        raise io.UnsupportedOperation("OutputTee does not support fileno")

    def flush(self) -> None:
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False

    def read(self, n: int = -1) -> str:
        raise io.UnsupportedOperation("OutputTee does not support reading")

    def readable(self) -> bool:
        """OutputTee is not readable."""
        return False

    def readline(self, limit: int = -1) -> str:
        raise io.UnsupportedOperation("OutputTee does not support reading")

    def readlines(self, hint: int = -1) -> list[str]:
        raise io.UnsupportedOperation("OutputTee does not support reading")

    def seek(self, offset: int, whence: int = 0) -> int:
        raise io.UnsupportedOperation("OutputTee does not support seek")

    def seekable(self) -> bool:
        """OutputTee is not seekable."""
        return False

    def tell(self) -> int:
        raise io.UnsupportedOperation("OutputTee does not support tell")

    def truncate(self, size: Optional[int] = None) -> int:
        raise io.UnsupportedOperation("OutputTee does not support truncate")

    def writable(self) -> bool:
        """OutputTee is writable if and only if all streams are writable."""
        return all(stream.writable() for stream in self.streams)

    def write(self, data: str) -> int:
        """Write to all streams."""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def writelines(self, lines: Iterable[str]) -> None:
        """Write lines to all streams."""
        lines_list = list(lines)
        for stream in self.streams:
            stream.writelines(lines_list)

    def __enter__(self) -> "OutputTee":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def __iter__(self) -> Iterator[str]:
        raise io.UnsupportedOperation("OutputTee does not support reading")

    def __next__(self) -> str:
        raise io.UnsupportedOperation("OutputTee does not support reading")
