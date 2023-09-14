import contextlib
import hashlib
import io
import os
import pathlib
import pkgutil
import shutil
import tarfile
import tempfile
import zipfile
from typing import IO, Generator, List, Optional, Union

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
        ValueError: Raised when the arcname cannot be encoded using ASCII.

    Yields:
        A bytes IO stream containing the zip file.
    """
    # TODO(SNOW-862576): Should remove check on ASCII encoding after SNOW-862576 fixed.
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
                    arcname = os.path.relpath(cur_path, start_path)
                    if not _able_ascii_encode(arcname):
                        raise ValueError(f"File name {arcname} cannot be encoded using ASCII. Please rename.")
                    zf.write(cur_path, arcname)
                    cur_path = os.path.dirname(cur_path)

            if os.path.isdir(path):
                for dirpath, _, files in os.walk(path):
                    # ignore __pycache__
                    if ignore_generated_py_file and "__pycache__" in dirpath:
                        continue
                    arcname = os.path.relpath(dirpath, start_path)
                    if not _able_ascii_encode(arcname):
                        raise ValueError(f"File name {arcname} cannot be encoded using ASCII. Please rename.")
                    zf.write(dirpath, arcname)
                    for file in files:
                        # ignore generated python files
                        if ignore_generated_py_file and file.endswith(GENERATED_PY_FILE_EXT):
                            continue
                        file_path = os.path.join(dirpath, file)
                        arcname = os.path.relpath(file_path, start_path)
                        if not _able_ascii_encode(arcname):
                            raise ValueError(f"File name {arcname} cannot be encoded using ASCII. Please rename.")
                        zf.write(file_path, arcname)
            else:
                arcname = os.path.relpath(path, start_path)
                if not _able_ascii_encode(arcname):
                    raise ValueError(f"File name {arcname} cannot be encoded using ASCII. Please rename.")
                zf.write(path, arcname)

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


def hash_directory(
    directory: Union[str, pathlib.Path], *, ignore_hidden: bool = False, excluded_files: Optional[List[str]] = None
) -> str:
    """Hash the **content** of a folder recursively using SHA-1.

    Args:
        directory: The path to the directory to be hashed.
        ignore_hidden: Whether to ignore hidden file. Defaults to False.
        excluded_files: List of file names to be excluded from the hashing.

    Returns:
        The hexdigest form of the hash result.
    """
    if not excluded_files:
        excluded_files = []

    def _update_hash_from_dir(
        directory: Union[str, pathlib.Path], hash: "hashlib._Hash", *, ignore_hidden: bool, excluded_files: List[str]
    ) -> "hashlib._Hash":
        assert pathlib.Path(directory).is_dir(), "Provided path is not a directory."
        for path in sorted(pathlib.Path(directory).iterdir(), key=lambda p: str(p).lower()):
            if ignore_hidden and path.name.startswith("."):
                continue
            if path.name in excluded_files:
                continue
            hash.update(path.name.encode())
            if path.is_file():
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(64 * 1024), b""):
                        hash.update(chunk)
            elif path.is_dir():
                hash = _update_hash_from_dir(path, hash, ignore_hidden=ignore_hidden, excluded_files=excluded_files)
        return hash

    return _update_hash_from_dir(
        directory, hashlib.sha1(), ignore_hidden=ignore_hidden, excluded_files=excluded_files
    ).hexdigest()


def get_all_modules(dirname: str, prefix: str = "") -> List[str]:
    modules = [mod.name for mod in pkgutil.iter_modules([dirname], prefix=prefix)]
    subdirs = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for sub_dirname in subdirs:
        basename = os.path.basename(sub_dirname)
        sub_dir_namespace = f"{prefix}{basename}"
        if sub_dir_namespace not in modules:
            modules.append(sub_dir_namespace)
        modules.extend(get_all_modules(sub_dirname, prefix=f"{sub_dir_namespace}."))
    return modules


def _able_ascii_encode(s: str) -> bool:
    try:
        s.encode("ascii", errors="strict")
        return True
    except UnicodeEncodeError:
        return False


@contextlib.contextmanager
def _create_tar_gz_stream(source_dir: str, arcname: Optional[str] = None) -> Generator[io.BytesIO, None, None]:
    """
    Create a compressed tarball (.tar.gz) of the source directory and return an input stream as a context
    manager.

    Args:
        source_dir (str): The path to the directory to compress.
        arcname: Alternative name for a file in the archive

    Yields:
        io.BytesIO: An input stream containing the compressed tarball.
    """
    with io.BytesIO() as output_stream:
        with tarfile.open(fileobj=output_stream, mode="w:gz") as tar:
            tar.add(source_dir, arcname=arcname)
        output_stream.seek(0)
        yield output_stream
