import concurrent.futures
import contextlib
import logging
import os
import pathlib
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Generator, Optional, Union
from urllib import parse

from snowflake import snowpark
from snowflake.ml._internal import file_utils

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 8
DISK_SAFETY_MARGIN = 0.9
DEFAULT_DISK_BUDGET_WAIT_TIMEOUT_SECONDS = 3600
_LAZY_UPLOAD_TEMP_PREFIX = "snowml_hf_lazy_upload_"


def _lazy_upload_temp_root() -> pathlib.Path:
    """Return the filesystem root where lazy-upload temp directories are created."""
    return pathlib.Path(tempfile.gettempdir())


def _validate_repo_relative_path(path: str) -> None:
    """Reject repo-relative paths that could escape the intended upload root."""
    if not path or path.startswith("/"):
        raise ValueError(
            "model upload: invalid HuggingFace repository file path. "
            f"Expected a relative path within the repository, got {path!r}."
        )
    parts = pathlib.PurePosixPath(path).parts
    if ".." in parts:
        raise ValueError(
            "model upload: invalid HuggingFace repository file path. "
            f"Path must not contain parent-directory segments: {path!r}."
        )


@dataclass(frozen=True)
class LazyHFUpload:
    """Deferred HuggingFace repository upload metadata."""

    download_kwargs: dict[str, Any]
    files: list[str]
    file_sizes: dict[str, int]
    relative_stage_dir: pathlib.PurePosixPath
    download_token: Optional[str] = None


class DiskBudget:
    """Tracks available disk bytes for concurrent HuggingFace downloads."""

    def __init__(
        self,
        available_bytes: int,
        *,
        acquire_timeout_seconds: float = DEFAULT_DISK_BUDGET_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        self._available_bytes = available_bytes
        self._condition = threading.Condition()
        self._acquire_timeout_seconds = acquire_timeout_seconds

    @staticmethod
    def from_files(
        file_sizes: dict[str, int],
        files: list[str],
    ) -> "DiskBudget":
        """Validate disk space and return a budget for concurrent downloads."""
        temp_root = _lazy_upload_temp_root()
        usage = shutil.disk_usage(temp_root)
        budget_bytes = int(usage.free * DISK_SAFETY_MARGIN)

        if not files:
            return DiskBudget(budget_bytes)

        largest_file = max(files, key=lambda filename: file_sizes.get(filename, 0))
        largest_file_size = file_sizes.get(largest_file, 0)
        if largest_file_size > budget_bytes:
            raise ValueError(
                "model upload: insufficient disk space to download HuggingFace model files. "
                f"The largest file ({largest_file}) requires {_format_size(largest_file_size)} "
                f"but only {_format_size(budget_bytes)} is available."
            )

        logger.info(
            "HuggingFace lazy upload disk budget: %s available at %s (%s total free)",
            _format_size(budget_bytes),
            temp_root,
            _format_size(usage.free),
        )
        return DiskBudget(budget_bytes)

    def acquire(self, nbytes: int) -> None:
        """Block until nbytes of disk budget are available, then reserve them."""
        with self._condition:
            deadline = time.monotonic() + self._acquire_timeout_seconds
            while self._available_bytes < nbytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "model upload: timed out waiting for disk space to download HuggingFace model files. "
                        f"Required {_format_size(nbytes)} but only {_format_size(self._available_bytes)} "
                        "was available."
                    )
                self._condition.wait(timeout=remaining)
            self._available_bytes -= nbytes

    def release(self, nbytes: int) -> None:
        """Return nbytes to the disk budget."""
        with self._condition:
            self._available_bytes += nbytes
            self._condition.notify_all()

    @contextlib.contextmanager
    def reserve(self, nbytes: int) -> Generator[None, None, None]:
        """Reserve disk budget for the duration of the context."""
        self.acquire(nbytes)
        try:
            yield
        finally:
            self.release(nbytes)


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes} B"


def _upload_single_file(
    *,
    session: snowpark.Session,
    stage_path: Union[pathlib.PurePosixPath, parse.ParseResult],
    lazy: LazyHFUpload,
    filename: str,
    disk_budget: DiskBudget,
    statement_params: Optional[dict[str, Any]],
) -> None:
    """Download one HuggingFace file, upload it to stage, and free local disk space."""
    import huggingface_hub as hf_hub

    _validate_repo_relative_path(filename)

    file_size = lazy.file_sizes.get(filename, 0)
    if file_size == 0:
        logger.warning(
            "HuggingFace file %s has no known size; disk budget will not be reserved for it.",
            filename,
        )

    relative_path = lazy.relative_stage_dir / pathlib.PurePosixPath(filename)
    tmp_dir = tempfile.mkdtemp(prefix=_LAZY_UPLOAD_TEMP_PREFIX, dir=str(_lazy_upload_temp_root()))
    try:
        stage_dir_path = file_utils._resolve_stage_dir_path(stage_path, relative_path)
        with disk_budget.reserve(file_size):
            local_path = hf_hub.hf_hub_download(
                filename=filename,
                local_dir=tmp_dir,
                token=lazy.download_token,
                **lazy.download_kwargs,
            )
            file_utils.upload_file_to_stage(
                session,
                local_path,
                stage_dir_path,
                statement_params=statement_params,
            )
            if os.path.isfile(local_path):
                os.remove(local_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def stream_upload(
    session: snowpark.Session,
    stage_path: Union[pathlib.PurePosixPath, parse.ParseResult],
    lazy: LazyHFUpload,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    statement_params: Optional[dict[str, Any]] = None,
) -> None:
    """Download HuggingFace repo files in parallel and upload each to stage.

    Args:
        session: Snowpark Session.
        stage_path: Base path in the stage.
        lazy: Metadata describing which files to upload and where.
        max_workers: Maximum number of concurrent download-and-upload workers.
        statement_params: Statement Params.

    Raises:
        Exception: If disk space is insufficient or any file download or stage upload fails.
    """
    total_files = len(lazy.files)
    if total_files == 0:
        return

    for filename in lazy.files:
        _validate_repo_relative_path(filename)

    disk_budget = DiskBudget.from_files(lazy.file_sizes, lazy.files)
    logger.info(
        "Starting parallel upload of %s HuggingFace files with %s workers",
        total_files,
        max_workers,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _upload_single_file,
                session=session,
                stage_path=stage_path,
                lazy=lazy,
                filename=filename,
                disk_budget=disk_budget,
                statement_params=statement_params,
            ): filename
            for filename in lazy.files
        }
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            file_size = lazy.file_sizes.get(filename, 0)
            logger.info(
                "Uploaded HuggingFace file %s (size: %s)",
                filename,
                _format_size(file_size),
            )
