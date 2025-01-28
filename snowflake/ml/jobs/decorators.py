import copy
import functools
import inspect
from typing import Callable, Dict, List, Optional, TypeVar

from typing_extensions import ParamSpec

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.jobs import job as jb, manager as jm
from snowflake.ml.jobs._utils import payload_utils

_PROJECT = "MLJob"

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")


@snowpark._internal.utils.private_preview(version="1.7.4")
@telemetry.send_api_usage_telemetry(project=_PROJECT)
def remote(
    compute_pool: str,
    stage_name: str,
    pip_requirements: Optional[List[str]] = None,
    external_access_integrations: Optional[List[str]] = None,
    query_warehouse: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    session: Optional[snowpark.Session] = None,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, jb.MLJob]]:
    """
    Submit a job to the compute pool.

    Args:
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        env_vars: Environment variables to set in container
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        Decorator that dispatches invocations of the decorated function as remote jobs.
    """

    def decorator(func: Callable[_Args, _ReturnValue]) -> Callable[_Args, jb.MLJob]:
        # Copy the function to avoid modifying the original
        # We need to modify the line number of the function to exclude the
        # decorator from the copied source code
        wrapped_func = copy.copy(func)
        wrapped_func.__code__ = wrapped_func.__code__.replace(co_firstlineno=func.__code__.co_firstlineno + 1)

        # Validate function arguments based on signature
        signature = inspect.signature(func)
        pos_arg_names = []
        for name, param in signature.parameters.items():
            param_type = payload_utils.get_parameter_type(param)
            if param_type is not None:
                payload_utils.validate_parameter_type(param_type, name)
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                pos_arg_names.append(name)

        @functools.wraps(func)
        def wrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> jb.MLJob:
            # Validate positional args
            for i, arg in enumerate(args):
                arg_name = pos_arg_names[i] if i < len(pos_arg_names) else f"args[{i}]"
                payload_utils.validate_parameter_type(type(arg), arg_name)

            # Validate keyword args
            for k, v in kwargs.items():
                payload_utils.validate_parameter_type(type(v), k)

            arg_list = [str(v) for v in args] + [x for k, v in kwargs.items() for x in (f"--{k}", str(v))]
            job = jm._submit_job(
                source=wrapped_func,
                args=arg_list,
                stage_name=stage_name,
                compute_pool=compute_pool,
                pip_requirements=pip_requirements,
                external_access_integrations=external_access_integrations,
                query_warehouse=query_warehouse,
                env_vars=env_vars,
                session=session,
            )
            assert isinstance(job, jb.MLJob)
            return job

        return wrapper

    return decorator
