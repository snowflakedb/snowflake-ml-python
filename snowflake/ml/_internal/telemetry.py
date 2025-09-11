#!/usr/bin/env python3
import contextvars
import enum
import functools
import inspect
import operator
import os
import sys
import time
import traceback
import types
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar, Union, cast

from typing_extensions import ParamSpec

from snowflake import connector
from snowflake.connector import connect, telemetry as connector_telemetry, time_util
from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.snowpark import dataframe, exceptions as snowpark_exceptions, session
from snowflake.snowpark._internal import server_connection, utils

_log_counter = 0
_FLUSH_SIZE = 10

# Prepopulate allowed connection types for type checking later since getattr is slow on large modules
_CONNECTION_TYPES = {
    conn_type: getattr(connector, conn_type)
    for conn_type in ["SnowflakeConnection", "StoredProcConnection"]
    if hasattr(connector, conn_type)
}

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")


def _get_login_token() -> Union[str, bytes]:
    with open("/snowflake/session/token") as f:
        return f.read()


def _get_snowflake_connection() -> Optional[connector.SnowflakeConnection]:
    conn = None
    if os.getenv("SNOWFLAKE_HOST") is not None and os.getenv("SNOWFLAKE_ACCOUNT") is not None:
        try:
            conn = connect(
                host=os.getenv("SNOWFLAKE_HOST"),
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                token=_get_login_token(),
                authenticator="oauth",
            )
        except Exception:
            # Failed to get a new SnowflakeConnection in SPCS. Fall back to using the active session.
            # This will work in some cases once SPCS enables multiple authentication modes, and users select any auth.
            pass

    if conn is None:
        try:
            active_session = next(iter(session._get_active_sessions()))
            conn = active_session._conn._conn if active_session.telemetry_enabled else None
        except snowpark_exceptions.SnowparkSessionException:
            # Failed to get an active session. No connection available.
            pass

    return conn


@enum.unique
class TelemetryProject(enum.Enum):
    MLOPS = "MLOps"
    MODELING = "ModelDevelopment"
    # TODO: Update with remaining projects.


@enum.unique
class TelemetrySubProject(enum.Enum):
    MONITORING = "Monitoring"
    REGISTRY = "ModelManagement"
    # TODO: Update with remaining subprojects.


@enum.unique
class TelemetryField(enum.Enum):
    # constants
    NAME = "name"
    # types of telemetry
    TYPE_FUNCTION_USAGE = "function_usage"
    TYPE_SNOWML_SPCS_USAGE = "snowml_spcs_usage"
    TYPE_SNOWML_PIPELINE_USAGE = "snowml_pipeline_usage"
    # message keys for telemetry
    KEY_PROJECT = "project"
    KEY_SUBPROJECT = "subproject"
    KEY_FUNC_NAME = "func_name"
    KEY_FUNC_PARAMS = "func_params"
    KEY_ERROR_INFO = "error_info"
    KEY_ERROR_CODE = "error_code"
    KEY_STACK_TRACE = "stack_trace"
    KEY_DURATION = "duration"
    KEY_VERSION = "version"
    KEY_PYTHON_VERSION = "python_version"
    KEY_OS = "operating_system"
    KEY_DATA = "data"
    KEY_CATEGORY = "category"
    KEY_API_CALLS = "api_calls"
    KEY_SFQIDS = "sfqids"
    KEY_CUSTOM_TAGS = "custom_tags"
    # function categories
    FUNC_CAT_USAGE = "usage"


class _TelemetrySourceType(enum.Enum):
    # Automatically inferred telemetry/statement parameters
    AUTO_TELEMETRY = "SNOWML_AUTO_TELEMETRY"
    # Mixture of manual and automatic telemetry/statement parameters
    AUGMENT_TELEMETRY = "SNOWML_AUGMENT_TELEMETRY"


_statement_params_context_var: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("statement_params")


class _StatementParamsPatchManager:
    def __init__(self) -> None:
        self._patch_cache: set[server_connection.ServerConnection] = set()
        self._context_var: contextvars.ContextVar[dict[str, str]] = _statement_params_context_var

    def apply_patches(self) -> None:
        try:
            # Apply patching to all active sessions in case of multiple
            for sess in session._get_active_sessions():
                # Check patch cache here to avoid unnecessary context switches
                if self._get_target(sess) not in self._patch_cache:
                    self._patch_session(sess)
        except snowpark_exceptions.SnowparkSessionException:
            pass

    def set_statement_params(self, statement_params: dict[str, str]) -> None:
        # Only set value if not already set in context
        if not self._context_var.get({}):
            self._context_var.set(statement_params)

    def _get_target(self, session: session.Session) -> server_connection.ServerConnection:
        return cast(server_connection.ServerConnection, session._conn)

    def _patch_session(self, session: session.Session, throw_on_patch_fail: bool = False) -> None:
        # Extract target
        try:
            target = self._get_target(session)
        except AttributeError:
            if throw_on_patch_fail:
                raise
            # TODO: Log a warning, this probably means there was a breaking change in Snowpark/SnowflakeConnection
            return

        # Check if session has already been patched
        if target in self._patch_cache:
            return
        self._patch_cache.add(target)

        functions = [
            ("execute_and_notify_query_listener", "_statement_params"),
            ("execute_async_and_notify_query_listener", "_statement_params"),
        ]

        for func, param_name in functions:
            try:
                self._patch_with_statement_params(target, func, param_name=param_name)
            except AttributeError:
                if throw_on_patch_fail:  # primarily used for testing
                    raise
                # TODO: Log a warning, this probably means there was a breaking change in Snowpark/SnowflakeConnection

    def _patch_with_statement_params(
        self, target: object, function_name: str, param_name: str = "statement_params"
    ) -> None:
        func = getattr(target, function_name)
        assert callable(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Retrieve context level statement parameters
            context_params = self._context_var.get(dict())
            if not context_params:
                # Exit early if not in SnowML (decorator) context
                return func(*args, **kwargs)

            # Extract any explicitly provided statement parameters
            orig_kwargs = dict(kwargs)
            in_params = kwargs.pop(param_name, None) or {}

            # Inject a special flag to statement parameters so we can filter out these patched logs if necessary
            # Calls that include SnowML telemetry are tagged with "SNOWML_AUGMENT_TELEMETRY"
            # and calls without SnowML telemetry are tagged with "SNOWML_AUTO_TELEMETRY"
            if TelemetryField.KEY_PROJECT.value in in_params:
                context_params["snowml_telemetry_type"] = _TelemetrySourceType.AUGMENT_TELEMETRY.value
            else:
                context_params["snowml_telemetry_type"] = _TelemetrySourceType.AUTO_TELEMETRY.value

            # Apply any explicitly provided statement parameters and result into function call
            context_params.update(in_params)
            kwargs[param_name] = context_params

            try:
                return func(*args, **kwargs)
            except TypeError as e:
                if str(e).endswith(f"unexpected keyword argument '{param_name}'"):
                    # TODO: Log warning that this patch is invalid
                    # Unwrap function for future invocations
                    setattr(target, function_name, func)
                    return func(*args, **orig_kwargs)
                else:
                    raise

        setattr(target, function_name, wrapper)

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # unpickling does not call __init__ by default, do it manually here
        self.__init__()  # type: ignore[misc]


_patch_manager = _StatementParamsPatchManager()


def get_statement_params(
    project: str, subproject: Optional[str] = None, class_name: Optional[str] = None
) -> dict[str, Any]:
    """
    Get telemetry statement parameters.

    Args:
        project: Project name.
        subproject: Subproject name.
        class_name: Caller's module name.

    Returns:
        dictionary of parameters to log to event table.
    """
    frame = inspect.currentframe()
    return get_function_usage_statement_params(
        project=project,
        subproject=subproject,
        function_name=get_statement_params_full_func_name(frame.f_back if frame else None, class_name),
    )


def add_statement_params_custom_tags(
    statement_params: Optional[dict[str, Any]], custom_tags: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Add custom_tags to existing statement_params.  Overwrite keys in custom_tags dict that already exist.
    If existing statement_params are not provided, do nothing as the information cannot be effectively tracked.

    Args:
        statement_params: Existing statement_params dictionary.
        custom_tags: Dictionary of existing k/v pairs to add as custom_tags

    Returns:
        new statement_params dictionary with all keys and an updated custom_tags field.
    """
    if not statement_params:
        return {}
    existing_custom_tags: dict[str, Any] = statement_params.pop(TelemetryField.KEY_CUSTOM_TAGS.value, {})
    existing_custom_tags.update(custom_tags)
    # NOTE: This can be done with | operator after upgrade from py3.8
    return {
        **statement_params,
        TelemetryField.KEY_CUSTOM_TAGS.value: existing_custom_tags,
    }


# TODO: we can merge this with get_statement_params after code clean up
def get_statement_params_full_func_name(frame: Optional[types.FrameType], class_name: Optional[str] = None) -> str:
    """
    Get the class-level or module-level full function name to be logged in statement parameters.
    The full function name is in the form of "module_name.class_name.function_name" (class-level)
    or "module_name.function_name" (module-level) when `class_name` is None.

    Args:
        frame: Frame object for the caller’s stack frame.
        class_name: Class name.

    Returns:
        Full function name.

    Examples:

        >>> import inspect

        >>> func_name = get_statement_params_full_func_name(inspect.currentframe(), "ClassName")
        >>> statement_params = get_function_usage_statement_params(function_name=func_name, ...)
    """
    module = inspect.getmodule(frame)
    module_name = module.__name__ if module else None
    function_name = frame.f_code.co_name if frame else None
    func_name = ".".join([name for name in [module_name, class_name, function_name] if name])
    return func_name


def get_function_usage_statement_params(
    project: Optional[str] = None,
    subproject: Optional[str] = None,
    *,
    function_category: str = TelemetryField.FUNC_CAT_USAGE.value,
    function_name: Optional[str] = None,
    function_parameters: Optional[dict[str, Any]] = None,
    api_calls: Optional[
        list[
            Union[
                dict[str, Union[Callable[..., Any], str]],
                Union[Callable[..., Any], str],
            ]
        ]
    ] = None,
    custom_tags: Optional[dict[str, Union[bool, int, str, float]]] = None,
) -> dict[str, Any]:
    """
    Get function usage statement parameters.

    Args:
        project: Project.
        subproject: Subproject.
        function_category: Function category.
        function_name: Function name.
        function_parameters: Function parameters.
        api_calls: API calls in the function.
        custom_tags: Custom tags.

    Returns:
        Statement parameters.

    Examples:

        >>> statement_params = get_function_usage_statement_params(...)
        >>> df.collect(statement_params=statement_params)
    """
    telemetry_type = f"{env.SOURCE.lower()}_{TelemetryField.TYPE_FUNCTION_USAGE.value}"
    statement_params: dict[str, Any] = {
        connector_telemetry.TelemetryField.KEY_SOURCE.value: env.SOURCE,
        TelemetryField.KEY_PROJECT.value: project,
        TelemetryField.KEY_SUBPROJECT.value: subproject,
        TelemetryField.KEY_OS.value: env.OS,
        TelemetryField.KEY_VERSION.value: snowml_version.VERSION,
        TelemetryField.KEY_PYTHON_VERSION.value: env.PYTHON_VERSION,
        connector_telemetry.TelemetryField.KEY_TYPE.value: telemetry_type,
        TelemetryField.KEY_CATEGORY.value: function_category,
    }

    if function_name:
        statement_params[TelemetryField.KEY_FUNC_NAME.value] = function_name
    if function_parameters:
        statement_params[TelemetryField.KEY_FUNC_PARAMS.value] = function_parameters
    if api_calls:
        statement_params[TelemetryField.KEY_API_CALLS.value] = []
        for api_call in api_calls:
            if isinstance(api_call, dict):
                telemetry_api_call = api_call.copy()
                # convert Callable to str
                for field, api in api_call.items():
                    if callable(api):
                        telemetry_api_call[field] = _get_full_func_name(api)
                statement_params[TelemetryField.KEY_API_CALLS.value].append(telemetry_api_call)
            elif callable(api_call):
                func_name = _get_full_func_name(api_call)
                statement_params[TelemetryField.KEY_API_CALLS.value].append({TelemetryField.NAME.value: func_name})
            else:
                statement_params[TelemetryField.KEY_API_CALLS.value].append({TelemetryField.NAME.value: api_call})
    if custom_tags:
        statement_params[TelemetryField.KEY_CUSTOM_TAGS.value] = custom_tags
    # Snowpark doesn't support None value in statement_params from version 1.29
    for k in statement_params:
        if statement_params[k] is None:
            statement_params[k] = ""
    return statement_params


# TODO: fix the type hints here. It should use TypeVars.
def suppress_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator used in the telemetry client to suppress exceptions raised by client functions.
    We should continue the business logic when telemetry fails.

    Args:
        func: Telemetry client function.

    Returns:
        Decorator that suppresses exceptions raised by the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    return wrapper


def send_custom_usage(
    project: str,
    *,
    telemetry_type: str,
    subproject: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    conn = _get_snowflake_connection()

    # Send telemetry if Snowflake connection is available.
    if conn is not None:
        client = _SourceTelemetryClient(conn=conn, project=project, subproject=subproject)
        common_metrics = client._create_basic_telemetry_data(telemetry_type=telemetry_type)
        data = {**common_metrics, TelemetryField.KEY_DATA.value: data, **kwargs}
        client._send(msg=data)


def send_api_usage_telemetry(
    project: str,
    subproject: Optional[str] = None,
    *,
    func_params_to_log: Optional[Iterable[str]] = None,
    conn_attr_name: Optional[str] = None,
    api_calls_extractor: Optional[
        Callable[
            ...,
            list[
                Union[
                    dict[str, Union[Callable[..., Any], str]],
                    Union[Callable[..., Any], str],
                ]
            ],
        ]
    ] = None,
    sfqids_extractor: Optional[Callable[..., list[str]]] = None,
    subproject_extractor: Optional[Callable[[Any], str]] = None,
    custom_tags: Optional[dict[str, Union[bool, int, str, float]]] = None,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, _ReturnValue]]:
    """
    Decorator that sends API usage telemetry and adds function usage statement parameters to the dataframe returned by
    the function.

    Args:
        project: Project.
        subproject: Subproject.
        func_params_to_log: Function parameters to log.
        conn_attr_name: Name of the SnowflakeConnection attribute in `self`.
        api_calls_extractor: Extract API calls from `self`.
        sfqids_extractor: Extract sfqids from `self`.
        subproject_extractor: Extract subproject at runtime from `self`.
        custom_tags: Custom tags.

    Returns:
        Decorator that sends function usage telemetry for any call to the decorated function.

    Raises:
        TypeError: If `conn_attr_name` is provided but the conn attribute is not of type SnowflakeConnection.
        ValueError: If both `subproject` and `subproject_extractor` are provided

    # noqa: DAR402
    """
    start_time = time.perf_counter()

    if subproject is not None and subproject_extractor is not None:
        raise ValueError("Specifying both subproject and subproject_extractor is not allowed")

    def decorator(func: Callable[_Args, _ReturnValue]) -> Callable[_Args, _ReturnValue]:
        @functools.wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> _ReturnValue:
            params = _get_func_params(func, func_params_to_log, args, kwargs) if func_params_to_log else None

            api_calls: list[Union[dict[str, Union[Callable[..., Any], str]], Callable[..., Any], str]] = []
            if api_calls_extractor:
                extracted_api_calls = api_calls_extractor(args[0])
                for api_call in extracted_api_calls:
                    if isinstance(api_call, str):
                        api_calls.append({TelemetryField.NAME.value: api_call})
                    elif callable(api_call):
                        api_calls.append({TelemetryField.NAME.value: _get_full_func_name(api_call)})
                    else:
                        api_calls.append(api_call)
            api_calls.append({TelemetryField.NAME.value: _get_full_func_name(func)})

            sfqids = None
            if sfqids_extractor:
                sfqids = sfqids_extractor(args[0])

            subproject_name = subproject
            if subproject_extractor is not None:
                subproject_name = subproject_extractor(args[0])

            statement_params = get_function_usage_statement_params(
                project=project,
                subproject=subproject_name,
                function_category=TelemetryField.FUNC_CAT_USAGE.value,
                function_name=_get_full_func_name(func),
                function_parameters=params,
                api_calls=api_calls,
                custom_tags=custom_tags,
            )

            def update_stmt_params_if_snowpark_df(obj: _ReturnValue, statement_params: dict[str, Any]) -> _ReturnValue:
                """
                Update SnowML function usage statement parameters to the object if it is a Snowpark DataFrame.
                Used to track APIs returning a Snowpark DataFrame.

                Args:
                    obj: Object to check and update.
                    statement_params: Statement parameters.

                Returns:
                    Updated object.
                """
                if isinstance(obj, dataframe.DataFrame):
                    if hasattr(obj, "_statement_params") and obj._statement_params:
                        obj._statement_params.update(statement_params)
                    else:
                        obj._statement_params = statement_params  # type: ignore[assignment]
                return obj

            # Set up framework-level credit usage instrumentation
            ctx = contextvars.copy_context()
            _patch_manager.apply_patches()

            # This function should be executed with ctx.run()
            def execute_func_with_statement_params() -> _ReturnValue:
                _patch_manager.set_statement_params(statement_params)
                result = func(*args, **kwargs)
                return update_stmt_params_if_snowpark_df(result, statement_params)

            # prioritize `conn_attr_name` over the active session
            if conn_attr_name:
                # raise AttributeError if conn attribute does not exist in `self`
                conn = operator.attrgetter(conn_attr_name)(args[0])
                if not isinstance(conn, _CONNECTION_TYPES.get(type(conn).__name__, connector.SnowflakeConnection)):
                    raise TypeError(
                        f"Expected a conn object of type {' or '.join(_CONNECTION_TYPES.keys())} but got {type(conn)}"
                    )
            else:
                conn = _get_snowflake_connection()

            if conn is None:
                # Telemetry not enabled, just execute without our additional telemetry logic
                try:
                    return ctx.run(execute_func_with_statement_params)
                except snowml_exceptions.SnowflakeMLException as e:
                    raise e.original_exception from e

            # TODO(hayu): [SNOW-750287] Optimize telemetry client to a singleton.
            telemetry = _SourceTelemetryClient(conn=conn, project=project, subproject=subproject_name)
            telemetry_args = dict(
                func_name=_get_full_func_name(func),
                function_category=TelemetryField.FUNC_CAT_USAGE.value,
                func_params=params,
                api_calls=api_calls,
                sfqids=sfqids,
                custom_tags=custom_tags,
            )
            try:
                return ctx.run(execute_func_with_statement_params)
            except Exception as e:
                if not isinstance(e, snowml_exceptions.SnowflakeMLException):
                    # already handled via a nested decorated function
                    if getattr(e, "_snowflake_ml_handled", False):
                        raise
                    if isinstance(e, snowpark_exceptions.SnowparkClientException):
                        me = snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INTERNAL_SNOWPARK_ERROR, original_exception=e
                        )
                    else:
                        me = snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.UNDEFINED, original_exception=e
                        )
                else:
                    me = e

                telemetry_args["error"] = repr(me)
                telemetry_args["error_code"] = me.error_code
                # exclude telemetry frames
                excluded_frames = 2
                tb = traceback.extract_tb(sys.exc_info()[2])
                formatted_tb = "".join(traceback.format_list(tb[excluded_frames:]))
                formatted_exception = traceback.format_exception_only(*sys.exc_info()[:2])[0]  # error type + message
                telemetry_args["stack_trace"] = formatted_tb + formatted_exception

                me.original_exception._snowflake_ml_handled = True  # type: ignore[attr-defined]
                if e is not me:
                    raise  # Directly raise non-wrapped exceptions to preserve original stacktrace
                elif me.suppress_source_trace:
                    raise me.original_exception from None
                else:
                    raise me.original_exception from e
            finally:
                telemetry_args["duration"] = time.perf_counter() - start_time  # type: ignore[assignment]
                telemetry.send_function_usage_telemetry(**telemetry_args)
                global _log_counter
                _log_counter += 1
                if _log_counter >= _FLUSH_SIZE or "error" in telemetry_args:
                    telemetry.send_batch()
                    _log_counter = 0

        return cast(Callable[_Args, _ReturnValue], wrap)

    return decorator


def _get_full_func_name(func: Callable[..., Any]) -> str:
    """
    Get the full function name with module and qualname.

    Args:
        func: Function.

    Returns:
        Full function name.
    """
    module = func.__module__ if hasattr(func, "__module__") else None
    qualname = func.__qualname__
    func_name = f"{module}.{qualname}" if module else qualname
    return func_name


def _get_func_params(
    func: Callable[..., Any], func_params_to_log: Optional[Iterable[str]], args: Any, kwargs: Any
) -> dict[str, Any]:
    """
    Get function parameters.

    Args:
        func: Function.
        func_params_to_log: Function parameters to log..
        args: Arguments.
        kwargs: Keyword arguments.

    Returns:
        Function parameters.
    """
    params = {}
    if func_params_to_log:
        spec = inspect.getfullargspec(func)
        for field in func_params_to_log:
            found, extracted_value = _extract_arg_value(field, spec, args, kwargs)
            if not found:
                pass
            else:
                params[field] = repr(extracted_value)
    return params


def _extract_arg_value(field: str, func_spec: inspect.FullArgSpec, args: Any, kwargs: Any) -> tuple[bool, Any]:
    """
    Function to extract a specified argument value.

    Args:
        field: Target function argument name to extract.
        func_spec: Full argument spec for the function.
        args: `args` for the invoked function.
        kwargs: `kwargs` for the invoked function.

    Returns:
        Tuple: First value indicates if `field` exists.
        Second value is the extracted value if existed.
    """
    if field in func_spec.args:
        idx = func_spec.args.index(field)
        if idx < len(args):
            return True, args[idx]
        elif field in kwargs:
            return True, kwargs[field]
        else:
            if func_spec.defaults:
                required_len = len(func_spec.args) - len(func_spec.defaults)
                return True, func_spec.defaults[idx - required_len]
            return False, None
    elif func_spec.kwonlydefaults and field in func_spec.kwonlyargs:
        if field in kwargs:
            return True, kwargs[field]
        elif field in func_spec.kwonlydefaults:
            return True, func_spec.kwonlydefaults[field]
        else:
            return False, None
    else:
        return False, None


class _SourceTelemetryClient:
    def __init__(
        self,
        conn: connector.SnowflakeConnection,
        project: Optional[str] = None,
        subproject: Optional[str] = None,
    ) -> None:
        """
        Universal telemetry client for the source using Python connector TelemetryClient.

        Args:
            conn: SnowflakeConnection.
            project: Project.
            subproject: Subproject.

        Attributes:
            _telemetry: Python connector TelemetryClient.
            source: Source.
            version: Library version.
            python_version: Python version.
            os: Operating system.
        """
        # TODO(hayu): [SNOW-750111] Support telemetry when libraries are used in SProc.
        self._telemetry: Optional[connector_telemetry.TelemetryClient] = (
            None if utils.is_in_stored_procedure() else conn._telemetry  # type: ignore[no-untyped-call]
        )
        self.source: str = env.SOURCE
        self.project: Optional[str] = project
        self.subproject: Optional[str] = subproject
        self.version = snowml_version.VERSION
        self.python_version: str = env.PYTHON_VERSION
        self.os: str = env.OS

    def _send(self, msg: dict[str, Any], timestamp: Optional[int] = None) -> None:
        """
        Add telemetry data to a batch in connector client.

        Args:
            msg: Telemetry message.
            timestamp: Timestamp.
        """
        if self._telemetry:
            if not timestamp:
                timestamp = time_util.get_time_millis()
            telemetry_data = connector_telemetry.TelemetryData(message=msg, timestamp=timestamp)
            self._telemetry.try_add_log_to_batch(telemetry_data)

    def _create_basic_telemetry_data(self, telemetry_type: str) -> dict[str, Any]:
        message = {
            connector_telemetry.TelemetryField.KEY_SOURCE.value: self.source,
            TelemetryField.KEY_PROJECT.value: self.project,
            TelemetryField.KEY_SUBPROJECT.value: self.subproject,
            TelemetryField.KEY_VERSION.value: self.version,
            TelemetryField.KEY_PYTHON_VERSION.value: self.python_version,
            TelemetryField.KEY_OS.value: self.os,
            connector_telemetry.TelemetryField.KEY_TYPE.value: telemetry_type,
        }
        return message

    @suppress_exceptions
    def send_function_usage_telemetry(
        self,
        func_name: str,
        function_category: str,
        duration: float,
        func_params: Optional[dict[str, Any]] = None,
        api_calls: Optional[list[dict[str, Any]]] = None,
        sfqids: Optional[list[Any]] = None,
        custom_tags: Optional[dict[str, Union[bool, int, str, float]]] = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        """
        Send function usage telemetry message.

        Args:
            func_name: Function name.
            function_category: Function category.
            duration: Function duration.
            func_params: Function parameters.
            api_calls: API calls.
            sfqids: Snowflake query IDs.
            custom_tags: Custom tags.
            error: Error.
            error_code: Error code.
            stack_trace: Error stack trace.
        """
        data: dict[str, Any] = {
            TelemetryField.KEY_FUNC_NAME.value: func_name,
            TelemetryField.KEY_CATEGORY.value: function_category,
        }
        if func_params:
            data[TelemetryField.KEY_FUNC_PARAMS.value] = func_params
        if api_calls:
            data[TelemetryField.KEY_API_CALLS.value] = api_calls
        if sfqids:
            data[TelemetryField.KEY_SFQIDS.value] = sfqids
        if custom_tags:
            data[TelemetryField.KEY_CUSTOM_TAGS.value] = custom_tags

        telemetry_type = f"{self.source.lower()}_{TelemetryField.TYPE_FUNCTION_USAGE.value}"
        message: dict[str, Any] = {
            **self._create_basic_telemetry_data(telemetry_type),
            TelemetryField.KEY_DATA.value: data,
            TelemetryField.KEY_DURATION.value: duration,
        }

        if error:
            message[TelemetryField.KEY_ERROR_INFO.value] = error
            message[TelemetryField.KEY_ERROR_CODE.value] = error_code
            message[TelemetryField.KEY_STACK_TRACE.value] = stack_trace

        self._send(message)

    @suppress_exceptions
    def send_batch(self) -> None:
        """Send the telemetry data batch immediately."""
        if self._telemetry:
            self._telemetry.send_batch()


def get_sproc_statement_params_kwargs(sproc: Callable[..., Any], statement_params: dict[str, Any]) -> dict[str, Any]:
    """
    Get statement_params keyword argument for sproc call.

    Args:
        sproc: sproc function
        statement_params: dictionary to be passed as statement params, if possible

    Returns:
        Keyword arguments dict
    """
    sproc_argspec = inspect.getfullargspec(sproc)
    kwargs = {}
    if "statement_params" in sproc_argspec.args:
        kwargs["statement_params"] = statement_params

    return kwargs
