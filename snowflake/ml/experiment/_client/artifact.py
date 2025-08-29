import os
import posixpath
from typing import NamedTuple


class ArtifactInfo(NamedTuple):
    name: str
    size: int
    md5: str
    last_modified: str


def get_put_path_pairs(local_path: str, artifact_path: str) -> list[tuple[str, str]]:
    """Enumerate files to upload and their destination subdirectories.

    Expands a local path (file or directory) into a list of pairs used for uploading
    artifacts to the stage.

    Args:
        local_path: Absolute or relative path to a local file or directory to upload.
        artifact_path: Destination subdirectory under the run's artifact root in the
            stage. If empty, files are uploaded to the root. When uploading a
            directory, this value is prepended to each file's relative path.

    Returns:
        A list of tuples ``(local_file_path, destination_artifact_subdir)``. For a
        single file, the list contains one entry with ``destination_artifact_subdir``
        set to ``artifact_path``. For directories, one entry per file is produced and
        subdirectories are preserved using POSIX separators.

    Raises:
        FileNotFoundError: If ``local_path`` does not exist.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    if os.path.isfile(local_path):
        return [(local_path, artifact_path)]

    pairs: list[tuple[str, str]] = []
    base_dir = local_path
    for root, _, files in os.walk(base_dir):
        if not files:
            continue
        rel_dir = os.path.relpath(root, base_dir)
        rel_dir_posix = "" if rel_dir in (".", "") else rel_dir.replace(os.sep, "/")
        dest_artifact_path = posixpath.join(artifact_path, rel_dir_posix) if rel_dir_posix else artifact_path
        for filename in files:
            file_path = os.path.join(root, filename)
            pairs.append((file_path, dest_artifact_path))
    return pairs


def get_download_path_pairs(artifacts: list[ArtifactInfo], base_target_dir: str) -> list[tuple[str, str]]:
    """Given artifact metadata entries, computes where each artifact should be written
    locally.

    Args:
        artifacts: List of artifact metadata where ``name`` is a POSIX-style path
            relative to the run's artifact root (e.g., ``"file.txt"`` or
            ``"nested/dir/file.txt"``).
        base_target_dir: Local base directory to download into. If empty, the
            current working directory is used. Directories will be created as
            needed.

    Returns:
        A list of tuples ``(artifact_relative_path, local_directory)``. Each tuple
        provides the relative path to request from the stage and the local directory
        where the file should be written.
    """
    planned: list[tuple[str, str]] = []
    for info in artifacts:
        rel_dir = os.path.dirname(info.name)
        local_dir = base_target_dir if rel_dir in ("", ".") else os.path.join(base_target_dir, rel_dir)
        planned.append((info.name, local_dir))
    return planned
