#!/usr/bin/env python3
import enum
import functools
import inspect
import operator
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import ParamSpec

from snowflake import connector
from snowflake.connector import telemetry as connector_telemetry, time_util
from snowflake.ml._internal import env
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.snowpark import dataframe, exceptions as snowpark_exceptions, session
from snowflake.snowpark._internal import utils

_log_counter = 0
_FLUSH_SIZE = 10

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")


@enum.unique
class TelemetryField(enum.Enum):
    # constants
    NAME = "name"
    # types of telemetry
    TYPE_FUNCTION_USAGE = "function_usage"
    # message keys for telemetry
    KEY_PROJECT = "project"
    KEY_SUBPROJECT = "subproject"
    KEY_FUNC_NAME = "func_name"
    KEY_FUNC_PARAMS = "func_params"
    KEY_ERROR_INFO = "error_info"
    KEY_ERROR_CODE = "error_code"
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


def get_statement_params(
    project: str, subproject: Optional[str] = None, class_name: Optional[str] = None
) -> Dict[str, Any]:
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
    function_parameters: Optional[Dict[str, Any]] = None,
    api_calls: Optional[
        List[
            Union[
                Dict[str, Union[Callable[..., Any], str]],
                Union[Callable[..., Any], str],
            ]
        ]
    ] = None,
    custom_tags: Optional[Dict[str, Union[bool, int, str, float]]] = None,
) -> Dict[str, Any]:
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
    statement_params: Dict[str, Any] = {
        connector_telemetry.TelemetryField.KEY_SOURCE.value: env.SOURCE,
        TelemetryField.KEY_PROJECT.value: project,
        TelemetryField.KEY_SUBPROJECT.value: subproject,
        TelemetryField.KEY_OS.value: env.OS,
        TelemetryField.KEY_VERSION.value: env.VERSION,
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
            if isinstance(api_call, Dict):
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


def send_api_usage_telemetry(
    project: str,
    subproject: Optional[str] = None,
    *,
    func_params_to_log: Optional[Iterable[str]] = None,
    conn_attr_name: Optional[str] = None,
    api_calls_extractor: Optional[
        Callable[
            ...,
            List[
                Union[
                    Dict[str, Union[Callable[..., Any], str]],
                    Union[Callable[..., Any], str],
                ]
            ],
        ]
    ] = None,
    sfqids_extractor: Optional[Callable[..., List[str]]] = None,
    custom_tags: Optional[Dict[str, Union[bool, int, str, float]]] = None,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, _ReturnValue]]:
    """
    Decorator that sends API usage telemetry.

    Args:
        project: Project.
        subproject: Subproject.
        func_params_to_log: Function parameters to log.
        conn_attr_name: Name of the SnowflakeConnection attribute in `self`.
        api_calls_extractor: Extract API calls from `self`.
        sfqids_extractor: Extract sfqids from `self`.
        custom_tags: Custom tags.

    Returns:
        Decorator that sends function usage telemetry for any call to the decorated function.

    Raises:
        TypeError: If `conn_attr_name` is provided but the conn attribute is not of type SnowflakeConnection.

    # noqa: DAR402
    """

    def decorator(func: Callable[_Args, _ReturnValue]) -> Callable[_Args, _ReturnValue]:
        @functools.wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> _ReturnValue:
            params = _get_func_params(func, func_params_to_log, args, kwargs) if func_params_to_log else None

            # prioritize `conn_attr_name` over the active session
            if conn_attr_name:
                # raise AttributeError if conn attribute does not exist in `self`
                conn = operator.attrgetter(conn_attr_name)(args[0])
                if not isinstance(conn, connector.SnowflakeConnection):
                    raise TypeError(f"Expected a conn object of type SnowflakeConnection, but got {type(conn)}")
            # get an active session
            else:
                try:
                    active_session = next(iter(session._get_active_sessions()))
                # server no default session
                except snowpark_exceptions.SnowparkSessionException:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if isinstance(e, snowml_exceptions.SnowflakeMLException):
                            e = e.original_exception
                        # suppress SnowparkSessionException from telemetry in the stack trace
                        raise e from None

                conn = active_session._conn._conn
                if (not active_session.telemetry_enabled) or (conn is None):
                    try:
                        return func(*args, **kwargs)
                    except snowml_exceptions.SnowflakeMLException as e:
                        raise e.original_exception from e

            api_calls: List[Dict[str, Any]] = []
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

            # TODO(hayu): [SNOW-750287] Optimize telemetry client to a singleton.
            telemetry = _SourceTelemetryClient(conn=conn, project=project, subproject=subproject)
            telemetry_args = dict(
                func_name=_get_full_func_name(func),
                function_category=TelemetryField.FUNC_CAT_USAGE.value,
                func_params=params,
                api_calls=api_calls,
                sfqids=sfqids,
                custom_tags=custom_tags,
            )
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, snowml_exceptions.SnowflakeMLException):
                    # already handled via a nested decorated function
                    if hasattr(e, "_snowflake_ml_handled") and e._snowflake_ml_handled:
                        raise e
                    if isinstance(e, snowpark_exceptions.SnowparkClientException):
                        e = snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INTERNAL_SNOWPARK_ERROR, original_exception=e
                        )
                    else:
                        e = snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.UNDEFINED, original_exception=e
                        )
                telemetry_args["error"] = repr(e)
                telemetry_args["error_code"] = e.error_code
                e.original_exception._snowflake_ml_handled = True  # type: ignore[attr-defined]
                raise e.original_exception from e
            else:
                return res
            finally:
                telemetry.send_function_usage_telemetry(**telemetry_args)
                global _log_counter
                _log_counter += 1
                if _log_counter >= _FLUSH_SIZE or "error" in telemetry_args:
                    telemetry.send_batch()
                    _log_counter = 0

        return cast(Callable[_Args, _ReturnValue], wrap)

    return decorator


def add_stmt_params_to_df(
    project: str,
    subproject: Optional[str] = None,
    *,
    function_category: str = TelemetryField.FUNC_CAT_USAGE.value,
    func_params_to_log: Optional[Iterable[str]] = None,
    api_calls: Optional[
        List[
            Union[
                Dict[str, Union[Callable[..., Any], str]],
                Union[Callable[..., Any], str],
            ]
        ]
    ] = None,
    custom_tags: Optional[Dict[str, Union[bool, int, str, float]]] = None,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, _ReturnValue]]:
    """
    Decorator that adds function usage statement parameters to the dataframe returned by the function.

    Args:
        project: Project.
        subproject: Subproject.
        function_category: Function category.
        func_params_to_log: Function parameters to log.
        api_calls: API calls in the function.
        custom_tags: Custom tags.

    Returns:
        Decorator that adds function usage statement parameters to the dataframe returned by the decorated function.
    """

    def decorator(func: Callable[_Args, _ReturnValue]) -> Callable[_Args, _ReturnValue]:
        @functools.wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> _ReturnValue:
            params = _get_func_params(func, func_params_to_log, args, kwargs) if func_params_to_log else None
            statement_params = get_function_usage_statement_params(
                project=project,
                subproject=subproject,
                function_category=function_category,
                function_name=_get_full_func_name(func),
                function_parameters=params,
                api_calls=api_calls,
                custom_tags=custom_tags,
            )

            try:
                res = func(*args, **kwargs)
                if isinstance(res, dataframe.DataFrame):
                    if hasattr(res, "_statement_params") and res._statement_params:
                        res._statement_params.update(statement_params)
                    else:
                        res._statement_params = statement_params  # type: ignore[assignment]
            except Exception:
                raise
            else:
                return res

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
) -> Dict[str, Any]:
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


def _extract_arg_value(field: str, func_spec: inspect.FullArgSpec, args: Any, kwargs: Any) -> Tuple[bool, Any]:
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
        else:
            return True, func_spec.kwonlydefaults[field]
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
        self.version = env.VERSION
        self.python_version: str = env.PYTHON_VERSION
        self.os: str = env.OS

    def _send(self, msg: Dict[str, Any], timestamp: Optional[int] = None) -> None:
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

    def _create_basic_telemetry_data(self, telemetry_type: str) -> Dict[str, Any]:
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
        func_params: Optional[Dict[str, Any]] = None,
        api_calls: Optional[List[Dict[str, Any]]] = None,
        sfqids: Optional[List[Any]] = None,
        custom_tags: Optional[Dict[str, Union[bool, int, str, float]]] = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Send function usage telemetry message.

        Args:
            func_name: Function name.
            function_category: Function category.
            func_params: Function parameters.
            api_calls: API calls.
            sfqids: Snowflake query IDs.
            custom_tags: Custom tags.
            error: Error.
            error_code: Error code.
        """
        data: Dict[str, Any] = {
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
        message: Dict[str, Any] = {
            **self._create_basic_telemetry_data(telemetry_type),
            TelemetryField.KEY_DATA.value: data,
        }

        if error:
            message[TelemetryField.KEY_ERROR_INFO.value] = error
            message[TelemetryField.KEY_ERROR_CODE.value] = error_code

        self._send(message)

    @suppress_exceptions
    def send_batch(self) -> None:
        """Send the telemetry data batch immediately."""
        if self._telemetry:
            self._telemetry.send_batch()
