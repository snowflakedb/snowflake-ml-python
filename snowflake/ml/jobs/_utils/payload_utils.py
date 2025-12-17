import functools
import importlib
import inspect
import io
import itertools
import logging
import pickle
import sys
import textwrap
from importlib.abc import Traversable
from pathlib import Path, PurePath
from types import ModuleType
from typing import IO, Any, Callable, Optional, Union, cast, get_args, get_origin

import cloudpickle as cp
from packaging import version

from snowflake import snowpark
from snowflake.ml.jobs._utils import (
    constants,
    function_payload_utils,
    query_helper,
    stage_utils,
    types,
)
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark._internal import code_generation
from snowflake.snowpark._internal.utils import zip_file_or_directory_to_stream

logger = logging.getLogger(__name__)

cp.register_pickle_by_value(function_payload_utils)
ImportType = Union[str, Path, ModuleType]

_SUPPORTED_ARG_TYPES = {str, int, float}
_SUPPORTED_ENTRYPOINT_EXTENSIONS = {".py"}
_ENTRYPOINT_FUNC_NAME = "func"
_STARTUP_SCRIPT_PATH = PurePath("startup.sh")


def _compress_and_upload_file(
    session: snowpark.Session, source_path: Path, stage_path: PurePath, import_path: Optional[str] = None
) -> None:
    absolute_source_path = source_path.absolute()
    leading_path = absolute_source_path.as_posix()[: -len(import_path)] if import_path else None
    filename = f"{source_path.name}.zip" if source_path.is_dir() or source_path.suffix == ".py" else source_path.name
    with zip_file_or_directory_to_stream(source_path.absolute().as_posix(), leading_path) as stream:
        session.file.put_stream(
            cast(IO[bytes], stream),
            stage_path.joinpath(filename).as_posix(),
            auto_compress=False,
            overwrite=True,
        )


def _upload_directory(session: snowpark.Session, source_path: Path, payload_stage_path: PurePath) -> None:
    # Manually traverse the directory and upload each file, since Snowflake PUT
    # can't handle directories. Reduce the number of PUT operations by using
    # wildcard patterns to batch upload files with the same extension.
    upload_path_patterns = set()
    for p in source_path.rglob("*"):
        if p.is_dir():
            continue
        # Skip python cache files
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.name.startswith("."):
            # Hidden files: use .* pattern for batch upload
            if p.suffix:
                upload_path_patterns.add(p.parent.joinpath(f".*{p.suffix}"))
            else:
                upload_path_patterns.add(p.parent.joinpath(".*"))
        else:
            # Regular files: use * pattern for batch upload
            if p.suffix:
                upload_path_patterns.add(p.parent.joinpath(f"*{p.suffix}"))
            else:
                upload_path_patterns.add(p)

    for path in upload_path_patterns:
        session.file.put(
            str(path),
            payload_stage_path.joinpath(path.parent.relative_to(source_path)).as_posix(),
            overwrite=True,
            auto_compress=False,
        )


def upload_payloads(session: snowpark.Session, stage_path: PurePath, *payload_specs: types.PayloadSpec) -> None:
    for spec in payload_specs:
        source_path = spec.source_path
        remote_relative_path = spec.remote_relative_path
        compress = spec.compress
        payload_stage_path = stage_path.joinpath(remote_relative_path) if remote_relative_path else stage_path
        if isinstance(source_path, stage_utils.StagePath):
            # only copy files into one stage directory from another stage directory, not from stage file
            # due to incomplete of StagePath functionality
            if source_path.as_posix().endswith(".py"):
                session.sql(f"copy files into {stage_path.as_posix()}/ from {source_path.as_posix()}").collect()
            else:
                session.sql(
                    f"copy files into {payload_stage_path.as_posix()}/ from {source_path.as_posix()}/"
                ).collect()
        elif isinstance(source_path, Path):
            if source_path.is_dir():
                if compress:
                    _compress_and_upload_file(
                        session,
                        source_path,
                        stage_path,
                        remote_relative_path.as_posix() if remote_relative_path else None,
                    )
                else:
                    _upload_directory(session, source_path, payload_stage_path)

            elif source_path.is_file():
                if compress and source_path.suffix == ".py":
                    _compress_and_upload_file(
                        session,
                        source_path,
                        stage_path,
                        remote_relative_path.as_posix() if remote_relative_path else None,
                    )
                else:
                    session.file.put(
                        str(source_path.resolve()),
                        payload_stage_path.as_posix(),
                        overwrite=True,
                        auto_compress=False,
                    )


def upload_system_resources(session: snowpark.Session, stage_path: PurePath) -> None:
    resource_ref = importlib.resources.files(__package__).joinpath("scripts")

    def upload_dir(ref: Traversable, relative_path: str = "") -> None:
        for item in ref.iterdir():
            current_path = Path(relative_path) / item.name if relative_path else Path(item.name)
            if item.is_dir():
                # Recursively process subdirectories
                upload_dir(item, str(current_path))
            elif item.is_file():
                content = item.read_bytes()
                session.file.put_stream(
                    io.BytesIO(content),
                    stage_path.joinpath(current_path).as_posix(),
                    auto_compress=False,
                    overwrite=True,
                )

    upload_dir(resource_ref)


def resolve_source(
    source: Union[types.PayloadPath, Callable[..., Any]]
) -> Union[types.PayloadPath, Callable[..., Any]]:
    if callable(source):
        return source
    elif isinstance(source, types.PayloadPath):
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")
        return source.absolute()
    else:
        raise ValueError("Unsupported source type. Source must be a stage, file, directory, or callable.")


def resolve_entrypoint(
    source: Union[types.PayloadPath, Callable[..., Any]],
    entrypoint: Optional[Union[types.PayloadPath, list[str]]],
) -> Union[types.PayloadEntrypoint, list[str]]:
    """Resolve and validate the entrypoint for a job payload.

    Args:
        source: The source path or callable for the job payload.
        entrypoint: The entrypoint specification. Can be:
            - A path (str or Path) to a Python script file
            - A list of strings representing a custom command (passed through as-is)
            - None (inferred from source if source is a file)

    Returns:
        Either a PayloadEntrypoint object for file-based entrypoints, or the list
        of strings passed through unchanged for custom command entrypoints.

    Raises:
        ValueError: If the entrypoint is invalid or cannot be resolved.
        FileNotFoundError: If the entrypoint file does not exist.
    """
    # If entrypoint is a list, pass it through without resolution/validation
    # This allows users to specify custom entrypoints (e.g., installed CLI tools)
    if isinstance(entrypoint, (list, tuple)):
        return entrypoint

    if callable(source):
        # Entrypoint is generated for callable payloads
        return types.PayloadEntrypoint(
            file_path=entrypoint or Path(constants.DEFAULT_ENTRYPOINT_PATH),
            main_func=_ENTRYPOINT_FUNC_NAME,
        )

    # Resolve entrypoint path for file-based payloads
    parent = source.absolute()
    if entrypoint is None:
        if parent.is_file():
            # Infer entrypoint from source
            entrypoint = parent
        else:
            raise ValueError("Entrypoint must be provided when source is a directory")
    elif entrypoint.is_absolute():
        # Absolute path - validate it's a subpath of source dir
        if not entrypoint.is_relative_to(parent):
            raise ValueError(f"Entrypoint must be a subpath of {parent}, got: {entrypoint}")
    else:
        # Relative path
        if (abs_entrypoint := entrypoint.absolute()).is_relative_to(parent) and abs_entrypoint.is_file():
            # Relative to working dir iff path is relative to source dir and exists
            entrypoint = abs_entrypoint
        else:
            # Relative to source dir
            entrypoint = parent.joinpath(entrypoint)

    # Validate resolved entrypoint file
    if not entrypoint.is_file():
        raise FileNotFoundError(
            "Entrypoint not found. Ensure the entrypoint is a valid file and is under"
            f" the source directory (source={parent}, entrypoint={entrypoint})"
        )

    if entrypoint.suffix not in _SUPPORTED_ENTRYPOINT_EXTENSIONS:
        raise ValueError(
            "Unsupported entrypoint type:"
            f" supported={','.join(_SUPPORTED_ENTRYPOINT_EXTENSIONS)} got={entrypoint.suffix}"
        )

    return types.PayloadEntrypoint(
        file_path=entrypoint,  # entrypoint is an absolute path at this point
        main_func=None,
    )


def get_zip_file_from_path(path: types.PayloadPath) -> types.PayloadPath:
    """Finds the path of the outermost zip archive from a given file path.

    Examples:
        >>> get_zip_file_from_path("/path/to/archive.zip/nested_file.py")
        "/path/to/archive.zip"
        >>> get_zip_file_from_path("/path/to/archive.zip")
        "/path/to/archive.zip"
        >>> get_zip_file_from_path("/path/to/regular_file.py")
        "/path/to/regular_file.py"

    Args:
        path: The file path to inspect.

    Returns:
        str: The path to the outermost zip file, or the original path if
            none is found.
    """

    path_str = path.as_posix()

    index = path_str.rfind(".zip/")
    if index != -1:
        return stage_utils.resolve_path(path_str[: index + 4])
    return path


def _finalize_payload_pair(
    p: types.PayloadPath, base_import_path: Optional[str]
) -> tuple[types.PayloadPath, Optional[str]]:
    """Finalize the `(payload_path, import_path)` pair based on source type.

    - Zip file: ignore import path (returns `(p, None)`).
    - Python file: if `base_import_path` is provided, append ".py"; otherwise None.
    - Directory: preserve `base_import_path` as-is.
    - Stage file: use `base_import_path` as-is since we do not compress stage files.
    - Other files: ignore import path (None).

    Args:
        p (types.PayloadPath): The resolved source path
        base_import_path (Optional[str]): Slash-separated import path

    Returns:
        tuple[types.PayloadPath, Optional[str]]: `(p, final_import_path)` where:
            - `final_import_path` is None for zip archives and non-Python files.
            - `final_import_path` is `base_import_path + ".py"` for Python files when
              `base_import_path` is provided; otherwise None.
            - `final_import_path` is `base_import_path` for directories.

    """
    if p.suffix == ".zip":
        final_import_path = None
    elif isinstance(p, stage_utils.StagePath):
        final_import_path = base_import_path
    elif p.is_file():
        if p.suffix == ".py":
            final_import_path = (base_import_path + ".py") if base_import_path else None
        else:
            final_import_path = None
    else:
        final_import_path = base_import_path

    validate_import_path(p, final_import_path)
    return (p, None) if p.suffix == ".zip" else (p, final_import_path)


def resolve_import_path(
    path: Union[types.PayloadPath, ModuleType],
    import_path: Optional[str] = None,
) -> list[tuple[types.PayloadPath, Optional[str]]]:
    """
    Resolve and normalize the import path for modules, Python files, or zip payloads.

    Args:
        path (Union[types.PayloadPath, ModuleType]): The source path or module to resolve.
            - If a directory is provided, it is compressed as a zip archive preserving its structure.
            - If a single Python file is provided, the file itself is zipped.
            - If a module is provided, it is treated as a directory or Python file.
            - If a zip file is provided, it is uploaded as it is.
            - If a stage file is provided, we only support stage file when the import path is provided
        import_path (Optional[str], optional): Explicit import path to use. If None,
            the function infers it from `path`.

    Returns:
        list[tuple[types.PayloadPath, Optional[str]]]: A list of tuples where each tuple
        contains the resolved payload path and its corresponding import path (if any).

    Raises:
        FileNotFoundError: If the provided `path` does not exist.
        NotImplementedError: If the stage file is provided without an import path.
        ValueError: If the import path cannot be resolved or is invalid.
    """
    if import_path is None:
        import_path = path.stem if isinstance(path, types.PayloadPath) else path.__name__
    import_path = import_path.strip().replace(".", "/") if import_path else None
    if isinstance(path, Path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is not found")
        return [_finalize_payload_pair(path.absolute(), import_path)]
    elif isinstance(path, stage_utils.StagePath):
        if import_path:
            return [_finalize_payload_pair(path.absolute(), import_path)]
        raise NotImplementedError("We only support stage file when the import path is provided")
    elif isinstance(path, ModuleType):
        if hasattr(path, "__path__"):
            paths = [get_zip_file_from_path(stage_utils.resolve_path(p).absolute()) for p in path.__path__]
            return [_finalize_payload_pair(p, import_path) for p in paths]
        elif hasattr(path, "__file__") and path.__file__:
            p = get_zip_file_from_path(Path(path.__file__).absolute())
            return [_finalize_payload_pair(p, import_path)]
        else:
            raise ValueError(f"Module {path} is not a valid module")
    else:
        raise ValueError(f"Module {path} is not a valid imports")


def validate_import_path(source: Union[str, types.PayloadPath], import_path: Optional[str]) -> None:
    """Validate the import path for local python file or directory."""
    if import_path is None:
        return

    source_path = stage_utils.resolve_path(source) if isinstance(source, str) else source
    if isinstance(source_path, stage_utils.StagePath):
        if not source_path.as_posix().endswith(import_path + ".py"):
            raise ValueError(f"Import path {import_path} must end with the source name {source_path}")
    elif (source_path.is_file() and source_path.suffix == ".py") or source_path.is_dir():
        if not source_path.as_posix().endswith(import_path):
            raise ValueError(f"Import path {import_path} must end with the source name {source_path}")


def upload_imports(
    imports: Optional[list[Union[str, Path, ModuleType, tuple[Union[str, Path, ModuleType], Optional[str]]]]],
    session: snowpark.Session,
    stage_path: PurePath,
) -> None:
    """Resolve paths and upload imports for ML Jobs.

    Args:
        imports: Optional list of paths/modules, or tuples of
            ``(path_or_module, import_path)``. The path can be a local
            directory, a local ``.py`` file, a local ``.zip`` file, or a stage
            path (for example, ``@stage/path``). If a tuple is provided and the
            first element is a local directory or ``.py`` file, the second
            element denotes the Python import path (dot or slash separated) to
            which the content should be mounted. If not provided for local
            sources, it defaults to the stem of the path/module. For stage
            paths or non-Python local files, the import path is ignored.
        session: Active Snowpark session used to upload files.
        stage_path: Destination stage subpath where payloads will be uploaded.

    Raises:
        ValueError: If a import has an invalid format or the
            provided import path is incompatible with the source.

    """
    if not imports:
        return
    for additional_payload in imports:
        if isinstance(additional_payload, tuple):
            source, import_path = additional_payload
        elif isinstance(additional_payload, str) or isinstance(additional_payload, ModuleType):
            source = additional_payload
            import_path = None
        else:
            raise ValueError(f"Invalid import format: {additional_payload}")
        resolved_imports = resolve_import_path(
            stage_utils.resolve_path(source) if not isinstance(source, ModuleType) else source, import_path
        )
        for source_path, import_path in resolved_imports:
            # TODO(SNOW-2467038): support import path for stage files or directories
            if isinstance(source_path, stage_utils.StagePath):
                remote = None
                compress = False
            elif source_path.as_posix().endswith(".zip"):
                remote = None
                compress = False
            elif source_path.is_dir() or source_path.suffix == ".py":
                remote = PurePath(import_path) if import_path else None
                compress = True
            else:
                # if the file is not a python file, ignore the import path
                remote = None
                compress = False

            upload_payloads(session, stage_path, types.PayloadSpec(source_path, remote, compress=compress))


class JobPayload:
    def __init__(
        self,
        source: Union[str, Path, Callable[..., Any]],
        entrypoint: Optional[Union[str, Path, list[str]]] = None,
        *,
        pip_requirements: Optional[list[str]] = None,
        imports: Optional[list[Union[ImportType, tuple[ImportType, Optional[str]]]]] = None,
    ) -> None:
        """Initialize a job payload.

        Args:
            source: The source for the job payload. Can be a file path, directory path,
                stage path, or a callable.
            entrypoint: The entrypoint for job execution. Can be:
                - A path (str or Path) to a Python script file
                - A list of strings representing a custom command (e.g., ["arctic_training"])
                  which is passed through as-is without resolution or validation
                - None (inferred from source if source is a file)
            pip_requirements: A list of pip requirements for the job.
            imports: A list of additional imports/payloads for the job.
        """
        # for stage path like snow://domain....., Path(path) will remove duplicate /, it will become snow:/ domain...
        self.source = stage_utils.resolve_path(source) if isinstance(source, str) else source
        if isinstance(entrypoint, list):
            self.entrypoint: Optional[Union[types.PayloadPath, list[str]]] = entrypoint
        else:
            self.entrypoint = stage_utils.resolve_path(entrypoint) if isinstance(entrypoint, str) else entrypoint
        self.pip_requirements = pip_requirements
        self.imports = imports

    def upload(self, session: snowpark.Session, stage_path: Union[str, PurePath]) -> types.UploadedPayload:
        # Prepare local variables
        stage_path = PurePath(stage_path) if isinstance(stage_path, str) else stage_path
        source = resolve_source(self.source)
        entrypoint = resolve_entrypoint(source, self.entrypoint)
        pip_requirements = self.pip_requirements or []

        # Create stage if necessary
        stage_name = stage_path.parts[0].lstrip("@")
        # Explicitly check if stage exists first since we may not have CREATE STAGE privilege
        try:
            query_helper.run_query(session, "describe stage identifier(?)", params=[stage_name])
        except sp_exceptions.SnowparkSQLException:
            query_helper.run_query(
                session,
                "create stage if not exists identifier(?)"
                " encryption = ( type = 'SNOWFLAKE_SSE' )"
                " comment = 'Created by snowflake.ml.jobs Python API'",
                params=[stage_name],
            )

        # Upload payload to stage - organize into app/ subdirectory
        app_stage_path = stage_path.joinpath(constants.APP_STAGE_SUBPATH)
        upload_imports(self.imports, session, app_stage_path)

        # Handle list entrypoints (custom commands like ["arctic_training"])
        if isinstance(entrypoint, (list, tuple)):
            payload_name = entrypoint[0] if entrypoint else None
            # For list entrypoints, still upload source if it's a path
            if isinstance(source, Path):
                upload_payloads(session, app_stage_path, types.PayloadSpec(source, None))
            elif isinstance(source, stage_utils.StagePath):
                upload_payloads(session, app_stage_path, types.PayloadSpec(source, None))
            python_entrypoint: list[Union[str, PurePath]] = list(entrypoint)
        else:
            # Standard file-based entrypoint handling
            payload_name = None
            if not isinstance(source, types.PayloadPath):
                if isinstance(source, function_payload_utils.FunctionPayload):
                    payload_name = source.function.__name__

                source_code = generate_python_code(source, source_code_display=True)
                _ = session.file.put_stream(
                    io.BytesIO(source_code.encode()),
                    stage_location=app_stage_path.joinpath(entrypoint.file_path).as_posix(),
                    auto_compress=False,
                    overwrite=True,
                )
                source = Path(entrypoint.file_path.parent)

            elif isinstance(source, stage_utils.StagePath):
                payload_name = entrypoint.file_path.stem
                # copy payload to stage
                if source == entrypoint.file_path:
                    source = source.parent
                upload_payloads(session, app_stage_path, types.PayloadSpec(source, None))

            elif isinstance(source, Path):
                payload_name = entrypoint.file_path.stem
                upload_payloads(session, app_stage_path, types.PayloadSpec(source, None))
                if source.is_file():
                    source = source.parent

            python_entrypoint = [
                PurePath(
                    constants.STAGE_VOLUME_MOUNT_PATH,
                    constants.APP_STAGE_SUBPATH,
                    entrypoint.file_path.relative_to(source).as_posix(),
                ),
            ]
            if entrypoint.main_func:
                python_entrypoint += ["--script_main_func", entrypoint.main_func]

        if pip_requirements:
            session.file.put_stream(
                io.BytesIO("\n".join(pip_requirements).encode()),
                stage_location=app_stage_path.joinpath("requirements.txt").as_posix(),
                auto_compress=False,
                overwrite=True,
            )

        # Upload system scripts and other assets to system/ directory
        system_stage_path = stage_path.joinpath(constants.SYSTEM_STAGE_SUBPATH)
        system_pip_requirements = []
        if not any(r.startswith("cloudpickle") for r in pip_requirements):
            system_pip_requirements.append(f"cloudpickle~={version.parse(cp.__version__).major}.0")
        if system_pip_requirements:
            # Upload requirements.txt to system path in stage
            session.file.put_stream(
                io.BytesIO("\n".join(system_pip_requirements).encode()),
                stage_location=system_stage_path.joinpath("requirements.txt").as_posix(),
                auto_compress=False,
                overwrite=True,
            )
        upload_system_resources(session, system_stage_path)

        env_vars = {
            constants.STAGE_MOUNT_PATH_ENV_VAR: constants.STAGE_VOLUME_MOUNT_PATH,
            constants.PAYLOAD_DIR_ENV_VAR: constants.APP_STAGE_SUBPATH,
            constants.RESULT_PATH_ENV_VAR: constants.RESULT_PATH_DEFAULT_VALUE,
        }

        return types.UploadedPayload(
            stage_path=stage_path,
            entrypoint=[
                "bash",
                f"{constants.STAGE_VOLUME_MOUNT_PATH}/{constants.SYSTEM_STAGE_SUBPATH}/{_STARTUP_SCRIPT_PATH}",
                *python_entrypoint,
            ],
            env_vars=env_vars,
            payload_name=payload_name,
        )


def _get_parameter_type(param: inspect.Parameter) -> Optional[type[object]]:
    # Unwrap Optional type annotations
    param_type = param.annotation
    if get_origin(param_type) is Union and len(get_args(param_type)) == 2 and type(None) in get_args(param_type):
        param_type = next(t for t in get_args(param_type) if t is not type(None))

    # Return None for empty type annotations
    if param_type == inspect.Parameter.empty:
        return None
    return cast(type[object], param_type)


def _validate_parameter_type(param_type: type[object], param_name: str) -> None:
    # Validate param_type is a supported type
    if param_type not in _SUPPORTED_ARG_TYPES:
        raise ValueError(
            f"Unsupported argument type {param_type} for '{param_name}'."
            f" Supported types: {', '.join(t.__name__ for t in _SUPPORTED_ARG_TYPES)}"
        )


def _generate_source_code_comment(func: Callable[..., Any]) -> str:
    """Generate a comment string containing the source code of a function for readability."""
    try:
        if isinstance(func, functools.partial):
            # Unwrap functools.partial and generate source code comment from the original function
            comment = code_generation.generate_source_code(func.func)  # type: ignore[arg-type]
            args = itertools.chain((repr(a) for a in func.args), (f"{k}={v!r}" for k, v in func.keywords.items()))

            # Update invocation comment to show arguments passed via functools.partial
            comment = comment.replace(
                f"= {func.func.__name__}",
                "= functools.partial({}({}))".format(
                    func.func.__name__,
                    ", ".join(args),
                ),
            )
            return comment
        else:
            return code_generation.generate_source_code(func)  # type: ignore[arg-type]
    except Exception as exc:
        error_msg = f"Source code comment could not be generated for {func} due to error {exc}."
        return code_generation.comment_source_code(error_msg)


def _serialize_callable(func: Callable[..., Any]) -> bytes:
    try:
        func_bytes: bytes = cp.dumps(func)
        return func_bytes
    except (pickle.PicklingError, TypeError) as e:
        if isinstance(e, TypeError) and "_thread.lock" in str(e):
            raise RuntimeError(
                "Unable to pickle an object that internally holds a reference to a Session object, "
                "such as a Snowpark DataFrame."
            ) from e
        if isinstance(func, functools.partial):
            # Try to find which part of the partial isn't serializable for better debuggability
            objects = [
                ("function", func.func),
                *((f"positional arg {i}", a) for i, a in enumerate(func.args)),
                *((f"keyword arg '{k}'", v) for k, v in func.keywords.items()),
            ]
            for name, obj in objects:
                try:
                    cp.dumps(obj)
                except pickle.PicklingError:
                    raise ValueError(f"Unable to serialize {name}: {obj}") from e
        raise ValueError(f"Unable to serialize function: {func}") from e


def _generate_param_handler_code(signature: inspect.Signature, output_name: str = "kwargs") -> str:
    # Generate argparse logic for argument handling (type coercion, default values, etc)
    argparse_code = ["import argparse", "", "parser = argparse.ArgumentParser()"]
    argparse_postproc = []
    for name, param in signature.parameters.items():
        opts = {}

        param_type = _get_parameter_type(param)
        if param_type is not None:
            _validate_parameter_type(param_type, name)
            opts["type"] = param_type.__name__

        if param.default != inspect.Parameter.empty:
            opts["default"] = f"'{param.default}'" if isinstance(param.default, str) else param.default

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Keyword argument
            argparse_code.append(
                f"parser.add_argument('--{name}', required={'default' not in opts},"
                f" {', '.join(f'{k}={v}' for k, v in opts.items())})"
            )
        else:
            # Positional argument. Use `argparse.add_mutually_exclusive_group()`
            # to allow passing positional args by name as well
            group_name = f"{name}_group"
            argparse_code.append(
                f"{group_name} = parser.add_mutually_exclusive_group(required={'default' not in opts})"
            )
            argparse_code.append(
                f"{group_name}.add_argument('pos-{name}', metavar='{name}', nargs='?',"
                f" {', '.join(f'{k}={v}' for k, v in opts.items() if k != 'default')})"
            )
            argparse_code.append(
                f"{group_name}.add_argument('--{name}', {', '.join(f'{k}={v}' for k, v in opts.items())})"
            )
            argparse_code.append("")  # Add newline for readability
            argparse_postproc.append(
                f"args.{name} = {name} if ({name} := args.__dict__.pop('pos-{name}')) is not None else args.{name}"
            )
    argparse_code.append("args = parser.parse_args()")
    param_code = "\n".join(argparse_code + argparse_postproc)
    param_code += f"\n{output_name} = vars(args)"

    return param_code


def generate_python_code(payload: Callable[..., Any], source_code_display: bool = False) -> str:
    """Generate an entrypoint script from a Python function."""

    if isinstance(payload, function_payload_utils.FunctionPayload):
        function = payload.function
    else:
        function = payload

    signature = inspect.signature(function)
    if any(
        p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for p in signature.parameters.values()
    ):
        raise NotImplementedError("Function must not have unpacking arguments (* or **)")

    # Mirrored from Snowpark generate_python_code() function
    # https://github.com/snowflakedb/snowpark-python/blob/main/src/snowflake/snowpark/_internal/udf_utils.py
    source_code_comment = _generate_source_code_comment(function) if source_code_display else ""

    arg_dict_name = "kwargs"
    if isinstance(payload, function_payload_utils.FunctionPayload):
        param_code = f"{arg_dict_name} = {{}}"
    else:
        param_code = _generate_param_handler_code(signature, arg_dict_name)
    return f"""
import sys
import pickle

try:
    {textwrap.indent(source_code_comment, '    ')}
    {_ENTRYPOINT_FUNC_NAME} = pickle.loads(bytes.fromhex('{_serialize_callable(payload).hex()}'))
except (TypeError, pickle.PickleError):
    if sys.version_info.major != {sys.version_info.major} or sys.version_info.minor != {sys.version_info.minor}:
        raise RuntimeError(
            "Failed to deserialize function due to Python version mismatch."
            f" Runtime environment is Python {{sys.version_info.major}}.{{sys.version_info.minor}}"
            " but function was serialized using Python {sys.version_info.major}.{sys.version_info.minor}."
        ) from None
    raise
except AttributeError as e:
    if 'cloudpickle' in str(e):
        import cloudpickle as cp
        raise RuntimeError(
            "Failed to deserialize function due to cloudpickle version mismatch."
            f" Runtime environment uses cloudpickle=={{cp.__version__}}"
            " but job was serialized using cloudpickle=={cp.__version__}."
        ) from e
    raise

if __name__ == '__main__':
{textwrap.indent(param_code, '    ')}

    __return__ = {_ENTRYPOINT_FUNC_NAME}(**{arg_dict_name})
"""


def create_function_payload(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> function_payload_utils.FunctionPayload:
    signature = inspect.signature(func)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    session_argument = ""
    session = None
    for name, val in list(bound.arguments.items()):
        if isinstance(val, snowpark.Session):
            if session:
                raise TypeError(f"Expected only one Session-type argument, but got both {session_argument} and {name}.")
            session = val
            session_argument = name
            del bound.arguments[name]
    payload = function_payload_utils.FunctionPayload(func, session, session_argument, *bound.args, **bound.kwargs)

    return payload
