import contextlib
import hashlib
import importlib
import io
import os
import pathlib
import shutil
import tempfile
import zipfile
from typing import IO, Generator, Optional, Tuple, Union

from snowflake.snowpark import session as snowpark_session

GENERATED_PY_FILE_EXT = (".pyc", ".pyo", ".pyd", ".pyi")


def copy_file_or_tree(src: str, dst_dir: str) -> None:
    """Copy file or directory into target directory.

    Args:
        src: Source file or directory path.
        dst_dir: Destination directory path.
    """
    if os.path.isfile(src):
        shutil.copy(src=src, dst=dst_dir)
    else:
        dir_name = os.path.basename(os.path.abspath(src))
        dst_path = os.path.join(dst_dir, dir_name)
        shutil.copytree(src=src, dst=dst_path, ignore=shutil.ignore_patterns("__pycache__"))


@contextlib.contextmanager
def zip_file_or_directory_to_stream(
    path: str,
    leading_path: Optional[str] = None,
    ignore_generated_py_file: bool = True,
) -> Generator[io.BytesIO, None, None]:
    """This is a temporary fixed version of snowflake.snowpark._internal.utils.zip_file_or_directory_to_stream function.
    It compresses the file or directory as a zip file to a binary stream. The zip file could be later imported as a
    Python package.

    The original version did not implement correctly as it did not add folder record for those directory level between
    the leading_path and path. In this case, the generated zip file could not be imported as a Python namespace package.

    The original version wrongly believe that __init__.py is needed for all directories along the import path when
    importing a module as a zip file. However, it is not necessary as modern Python has already support namespace
    package where __init__.py is no longer required.

    Args:
        path: The absolute path to a file or directory.
        leading_path: This argument is used to determine where directory should
            start in the zip file. Basically, this argument works as the role
            of `start` argument in os.path.relpath(path, start), i.e.,
            absolute path = [leading path]/[relative path]. For example,
            when the path is "/tmp/dir1/dir2/test.py", and the leading path
            is "/tmp/dir1", the generated filesystem structure in the zip file
            will be "dir2/" and "dir2/test.py". Defaults to None.
        ignore_generated_py_file: Whether to ignore some generated python files
            in the directory. Defaults to True.

    Raises:
        FileNotFoundError: Raised when the given path does not exist.
        ValueError: Raised when the leading path is not a actual leading path of path

    Yields:
        A bytes IO stream containing the zip file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is not found")
    if leading_path and not path.startswith(leading_path):
        raise ValueError(f"{leading_path} doesn't lead to {path}")
    # if leading_path is not provided, just use the parent path,
    # and the compression will start from the parent directory
    start_path = leading_path if leading_path else os.path.join(path, "..")

    with io.BytesIO() as input_stream:
        with zipfile.ZipFile(input_stream, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if os.path.realpath(path) != os.path.realpath(start_path):
                cur_path = os.path.dirname(path)
                while os.path.realpath(cur_path) != os.path.realpath(start_path):
                    zf.writestr(f"{os.path.relpath(cur_path, start_path)}/", "")
                    cur_path = os.path.dirname(cur_path)

            if os.path.isdir(path):
                for dirname, _, files in os.walk(path):
                    # ignore __pycache__
                    if ignore_generated_py_file and "__pycache__" in dirname:
                        continue
                    zf.write(dirname, os.path.relpath(dirname, start_path))
                    for file in files:
                        # ignore generated python files
                        if ignore_generated_py_file and file.endswith(GENERATED_PY_FILE_EXT):
                            continue
                        filename = os.path.join(dirname, file)
                        zf.write(filename, os.path.relpath(filename, start_path))
            else:
                zf.write(path, os.path.relpath(path, start_path))

        yield input_stream


@contextlib.contextmanager
def unzip_stream_in_temp_dir(stream: IO[bytes], temp_root: Optional[str] = None) -> Generator[str, None, None]:
    """Unzip an IO stream into a temporary directory.

    Args:
        stream: The input stream.
        temp_root: The root directory where the temporary directory should created in. Defaults to None.

    Yields:
        The path to the created temporary directory.
    """
    with tempfile.TemporaryDirectory(dir=temp_root) as tempdir:
        with zipfile.ZipFile(stream, mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.extractall(path=tempdir)
        yield tempdir


@contextlib.contextmanager
def zip_snowml() -> Generator[Tuple[io.BytesIO, str], None, None]:
    """Zip the snowflake-ml source code as a zip-file for import.

    Yields:
        A bytes IO stream containing the zip file.
    """
    snowml_path = list(importlib.import_module("snowflake.ml").__path__)[0]
    root_path = os.path.normpath(os.path.join(snowml_path, os.pardir, os.pardir))
    with zip_file_or_directory_to_stream(snowml_path, root_path) as stream:
        yield stream, hash_directory(snowml_path)


def hash_directory(directory: Union[str, pathlib.Path]) -> str:
    """Hash the **content** of a folder recursively using SHA-1.

    Args:
        directory: The path to the directory to be hashed.

    Returns:
        The hexdigest form of the hash result.
    """

    def _update_hash_from_dir(directory: Union[str, pathlib.Path], hash: "hashlib._Hash") -> "hashlib._Hash":
        assert pathlib.Path(directory).is_dir(), "Provided path is not a directory."
        for path in sorted(pathlib.Path(directory).iterdir(), key=lambda p: str(p).lower()):
            hash.update(path.name.encode())
            if path.is_file():
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(64 * 1024), b""):
                        hash.update(chunk)
            elif path.is_dir():
                hash = _update_hash_from_dir(path, hash)
        return hash

    return _update_hash_from_dir(directory, hashlib.sha1()).hexdigest()


def upload_snowml(session: snowpark_session.Session, stage_location: Optional[str] = None) -> str:
    """Upload the SnowML local code into a stage if provided, or a session stage.
    It will label the file name using the SHA-1 of the snowflake.ml folder, so that if the source code does not change,
    it won't reupload. Any changes will, however, result a new zip file.

    Args:
        session: Snowpark connection session.
        stage_location: The path to the stage location where the uploaded SnowML should be. Defaults to None.

    Returns:
        The path to the uploaded SnowML zip file.
    """
    with zip_snowml() as (stream, hash_str):
        if stage_location is None:
            stage_location = session.get_session_stage()
        file_location = os.path.join(stage_location, f"snowml_{hash_str}.zip")
        session.file.put_stream(stream, stage_location=file_location, auto_compress=False, overwrite=False)
    return file_location
