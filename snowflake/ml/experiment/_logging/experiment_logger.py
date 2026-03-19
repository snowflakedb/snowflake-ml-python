import io
import json
import os
from typing import Any, Iterable, Iterator, TextIO


class ExperimentLogger(TextIO):
    OUTPUT_DIRECTORY = "/var/log/services/external/experiment_tracking"

    def __init__(
        self,
        exp_id: int,
        run_id: int,
        stream: str,
    ) -> None:
        super().__init__()
        filepath = os.path.join(self.OUTPUT_DIRECTORY, str(exp_id), str(run_id), f"{stream}.log")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "a", buffering=1)  # Line buffering
        self.exp_id = exp_id
        self.run_id = run_id
        self.stream = stream
        self._buffer = ""  # Buffer to store incomplete lines

    def _write_line(self, line: str) -> None:
        log_message = {
            "body": line,
            "attributes": {
                "snow.experiment.id": self.exp_id,
                "snow.experiment.run.id": self.run_id,
                "snow.experiment.stream": self.stream,
            },
        }
        json_data = json.dumps(log_message)
        self.file.write(json_data + "\n")

    def close(self) -> None:
        self.flush()
        self.file.close()

    def fileno(self) -> int:
        return self.file.fileno()

    def flush(self) -> None:
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        self.file.flush()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def read(self, n: int = -1) -> str:
        raise io.UnsupportedOperation("read")

    def readline(self, limit: int = -1) -> str:
        raise io.UnsupportedOperation("readline")

    def readlines(self, hint: int = -1) -> list[str]:
        raise io.UnsupportedOperation("readlines")

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        raise io.UnsupportedOperation("seek")

    def tell(self) -> int:
        raise io.UnsupportedOperation("tell")

    def truncate(self, size: int | None = None) -> int:
        raise io.UnsupportedOperation("truncate")

    def writable(self) -> bool:
        return True

    def write(self, data: str) -> int:
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line)
        return len(data)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[str]:
        raise io.UnsupportedOperation("__iter__")

    def __next__(self) -> str:
        raise io.UnsupportedOperation("__next__")
