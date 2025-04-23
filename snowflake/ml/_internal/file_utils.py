import contextlib
import hashlib
import importlib
import io
import logging
import os
import pathlib
import pkgutil
import shutil
import sys
import tarfile
import tempfile
import zipfile
from typing import Any, Callable, Generator, Literal, Optional, Union
from urllib import parse

import cloudpickle

from snowflake import snowpark
from snowflake.ml._internal.exceptions import exceptions
from snowflake.snowpark import exceptions as snowpark_exceptions

GENERATED_PY_FILE_EXT = (".pyc", ".pyo", ".pyd", ".pyi")


def copytree(
    src: "Union[str, os.PathLike[str]]",
    dst: "Union[str, os.PathLike[str]]",
    ignore: Optional[Callable[..., set[str]]] = None,
    dirs_exist_ok: bool = False,
) -> "Union[str, os.PathLike[str]]":
    """This is a forked version of shutil.copytree that remove all copystat, to make sure it works in Sproc.

    Args:
        src: Path to source file or directory
        dst: Path to destination file or directory
        ignore: Ignore pattern. Defaults to None.
        dirs_exist_ok: Flag to indicate if it is okay when creating dir of destination it has existed.
            Defaults to False.

    Raises:
        Error: Raised when there is any errors when copying.

    Returns:
        Path to destination file or directory
    """
    sys.audit("shutil.copytree", src, dst)
    with os.scandir(src) as itr:
        entries = list(itr)

    if ignore is not None:
        ignored_names = ignore(os.fspath(src), [x.name for x in entries])
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []

    for srcentry in entries:
        if srcentry.name in ignored_names:
            continue
        srcname = os.path.join(src, srcentry.name)
        dstname = os.path.join(dst, srcentry.name)
        try:
            if srcentry.is_dir():
                copytree(srcentry, dstname, ignore, dirs_exist_ok)
            else:
                # Will raise a SpecialFileError for unsupported file types
                shutil.copy(srcentry, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    if errors:
        raise shutil.Error(errors)
    return dst


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
        copytree(src=src, dst=dst_path, ignore=shutil.ignore_patterns("__pycache__"))


def make_archive(
    target_path: str,
    root_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
    owner: Optional[str] = None,
    group: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    target_file = pathlib.Path(target_path)
    ext = "".join(target_file.suffixes)
    basename = str(target_file.parent / target_file.name.replace(ext, ""))
    EXT_TO_FORMAT_MAPPING = {".zip": "zip", ".tar": "tar", ".tar.gz": "gztar", ".tar.bz2": "bztar", ".tar.xz": "xztar"}
    shutil.make_archive(
        basename,
        EXT_TO_FORMAT_MAPPING[ext],
        root_dir=root_dir,
        base_dir=base_dir,
        verbose=verbose,
        dry_run=dry_run,
        owner=owner,
        group=group,
        logger=logger,
    )


def zip_python_package(zipfile_path: str, package_name: str, ignore_generated_py_file: bool = True) -> None:
    import importlib_resources
    from importlib_resources import abc as importlib_resources_abc

    _, pkg_start_path = get_package_path(package_name)
    if os.path.isfile(pkg_start_path):
        shutil.copy(pkg_start_path, zipfile_path)
        return

    base_dirs = package_name.split(".")
    assert len(base_dirs) >= 1, "Invalid package name."

    with zipfile.ZipFile(zipfile_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(len(base_dirs)):
            arcname = pathlib.PurePosixPath(*base_dirs[: i + 1])
            zf.writestr(str(arcname) + "/", "")
        base_arcname = pathlib.PurePosixPath(*base_dirs)

        def _add_to_zip(
            zf: zipfile.ZipFile, path_info: importlib_resources_abc.Traversable, base_arcname: pathlib.PurePosixPath
        ) -> None:
            if path_info.is_file():
                if ignore_generated_py_file and path_info.name.endswith(GENERATED_PY_FILE_EXT):
                    return
                arcname = base_arcname / path_info.name
                if not _able_ascii_encode(str(arcname)):
                    raise ValueError(f"File name {arcname} cannot be encoded using ASCII. Please rename.")
                zf.writestr(str(arcname), path_info.read_bytes())
            elif path_info.is_dir():
                arcname = base_arcname / path_info.name
                zf.writestr(str(arcname) + "/", "")
                for sub_path_info in path_info.iterdir():
                    _add_to_zip(zf, sub_path_info, arcname)

        for sub_path_info in importlib_resources.files(package_name).iterdir():
            _add_to_zip(zf, sub_path_info, base_arcname)


def hash_directory(
    directory: Union[str, pathlib.Path], *, ignore_hidden: bool = False, excluded_files: Optional[list[str]] = None
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
        directory: Union[str, pathlib.Path], hash: "hashlib._Hash", *, ignore_hidden: bool, excluded_files: list[str]
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


def get_all_modules(dirname: str, prefix: str = "") -> list[str]:
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


def get_package_path(package_name: str, strategy: Literal["first", "last"] = "first") -> tuple[str, str]:
    """[Obsolete]Return the path to where a package is defined and its start location.
    Example 1: snowflake.ml -> path/to/site-packages/snowflake/ml, path/to/site-packages
    Example 2: zip_imported_module -> path/to/some/zipfile.zip/zip_imported_module, path/to/some/zipfile.zip

    Args:
        package_name: Qualified package name, like `snowflake.ml`
        strategy: Pick first or last one in sys.path. First is in most cases, the one being used. Last is, in most
            cases, the first to get imported from site-packages or even builtins.

    Returns:
        A tuple of the path to the package and start path
    """
    levels = len(package_name.split("."))
    pkg_path = list(importlib.import_module(package_name).__path__)[0 if strategy == "first" else -1]
    pkg_start_path = os.path.abspath(os.path.join(pkg_path, *([os.pardir] * levels)))
    return pkg_path, pkg_start_path


def stage_object(session: snowpark.Session, object: object, stage_location: str) -> list[snowpark.PutResult]:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    with open(temp_file_path, "wb") as file:
        cloudpickle.dump(object, file)
    put_res = session.file.put(temp_file_path, stage_location, auto_compress=False, overwrite=True)
    os.remove(temp_file_path)
    return put_res


def stage_file_exists(
    session: snowpark.Session, stage_location: str, file_name: str, statement_params: dict[str, Any]
) -> bool:
    try:
        res = session.sql(f"list {stage_location}/{file_name}").collect(statement_params=statement_params)
        return len(res) > 0
    except exceptions.SnowflakeMLException:
        return False


def _retry_on_sql_error(exception: Exception) -> bool:
    return isinstance(exception, snowpark_exceptions.SnowparkSQLException)


def upload_directory_to_stage(
    session: snowpark.Session,
    local_path: pathlib.Path,
    stage_path: Union[pathlib.PurePosixPath, parse.ParseResult],
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> None:
    """Upload a local folder recursively to a stage and keep the structure.

    Args:
        session: Snowpark Session.
        local_path: Local path to upload.
        stage_path: Base path in the stage.
        statement_params: Statement Params.
    """
    import retrying

    file_operation = snowpark.FileOperation(session=session)

    for root, _, filenames in os.walk(local_path):
        root_path = pathlib.Path(root)
        for filename in filenames:
            local_file_path = root_path / filename
            relative_path = pathlib.PurePosixPath(local_file_path.relative_to(local_path).as_posix())

            if isinstance(stage_path, parse.ParseResult):
                relative_stage_path = (pathlib.PosixPath(stage_path.path) / relative_path).parent
                new_url = parse.ParseResult(
                    scheme=stage_path.scheme,
                    netloc=stage_path.netloc,
                    path=str(relative_stage_path),
                    params=stage_path.params,
                    query=stage_path.query,
                    fragment=stage_path.fragment,
                )
                stage_dir_path = parse.urlunparse(new_url)
            else:
                stage_dir_path = str((stage_path / relative_path).parent)

            retrying.retry(
                retry_on_exception=_retry_on_sql_error,
                stop_max_attempt_number=5,
                wait_exponential_multiplier=100,
                wait_exponential_max=10000,
            )(file_operation.put)(
                str(local_file_path),
                str(stage_dir_path),
                auto_compress=False,
                overwrite=False,
                statement_params=statement_params,
            )


def download_directory_from_stage(
    session: snowpark.Session,
    stage_path: pathlib.PurePosixPath,
    local_path: pathlib.Path,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> None:
    """Upload a folder in stage recursively to a folder in local and keep the structure.

    Args:
        session: Snowpark Session.
        stage_path: Stage path to download from.
        local_path: Local path as the base of destination.
        statement_params: Statement Params.
    """
    import retrying

    file_operation = file_operation = snowpark.FileOperation(session=session)
    file_list = [
        pathlib.PurePosixPath(stage_path.parts[0], *pathlib.PurePosixPath(row.name).parts[1:])
        for row in session.sql(f"ls {stage_path}").collect()
    ]
    for stage_file_path in file_list:
        local_file_dir = local_path / stage_file_path.relative_to(stage_path).parent
        local_file_dir.mkdir(parents=True, exist_ok=True)
        retrying.retry(
            retry_on_exception=_retry_on_sql_error,
            stop_max_attempt_number=5,
            wait_exponential_multiplier=100,
            wait_exponential_max=10000,
        )(file_operation.get)(str(stage_file_path), str(local_file_dir), statement_params=statement_params)


def open_file(path: str, *args: Any, **kwargs: Any) -> Any:
    """This function is a wrapper on top of the Python built-in "open" function, with a few added default values
    to ensure successful execution across different platforms.

    Args:
        path: file path
        *args: arguments.
        **kwargs: key arguments.

    Returns:
        Open file and return a stream.
    """
    kwargs.setdefault("newline", "\n")
    kwargs.setdefault("encoding", "utf-8")
    return open(path, *args, **kwargs)
