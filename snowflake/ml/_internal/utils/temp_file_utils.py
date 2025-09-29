import logging
import os
import shutil
import tempfile
from typing import Iterable, Union

logger = logging.getLogger(__name__)


def get_temp_file_path(prefix: str = "") -> str:
    """Returns a new random temp file path.

    Args:
        prefix: A prefix to the temp file path, this can help add stored file information. Defaults to None.

    Returns:
        A new temp file path.
    """
    # TODO(snandamuri): Use in-memory filesystem for temp files.
    local_file = tempfile.NamedTemporaryFile(prefix=prefix, delete=True)
    local_file_name = local_file.name
    local_file.close()
    return local_file_name


def cleanup_temp_files(file_paths: Union[str, Iterable[str]]) -> None:
    """Deletes all the temp files.

    Args:
        file_paths: Files paths to be deleted.

    Raises:
        TypeError: Unhandled type for `file_paths`.
    """
    files_to_delete = []
    if type(file_paths) is str:
        files_to_delete = [file_paths]
    elif type(file_paths) is list:
        files_to_delete = file_paths
    elif type(file_paths) in [range, set, tuple]:
        files_to_delete = list(file_paths)
    else:
        raise TypeError(f"Could not convert {file_paths} to list")

    for file_to_delete in files_to_delete:
        try:
            if os.path.isdir(file_to_delete):
                shutil.rmtree(path=file_to_delete, ignore_errors=True)
            else:
                os.remove(file_to_delete)
        except FileNotFoundError:
            logger.warn(f"Failed to cleanup file {file_to_delete}")
