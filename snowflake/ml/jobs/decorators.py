import copy
import functools
from typing import Callable, Optional, TypeVar

from typing_extensions import ParamSpec

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.jobs import job as jb, manager as jm
from snowflake.ml.jobs._utils import constants

_PROJECT = "MLJob"

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def remote(
    compute_pool: str,
    *,
    stage_name: str,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    query_warehouse: Optional[str] = None,
    env_vars: Optional[dict[str, str]] = None,
    num_instances: Optional[int] = None,
    enable_metrics: bool = False,
    session: Optional[snowpark.Session] = None,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, jb.MLJob[_ReturnValue]]]:
    """
    Submit a job to the compute pool.

    Args:
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        query_warehouse: The query warehouse to use. Defaults to session warehouse.
        env_vars: Environment variables to set in container
        num_instances: The number of nodes in the job. If none specified, create a single node job.
        enable_metrics: Whether to enable metrics publishing for the job.
        session: The Snowpark session to use. If none specified, uses active session.

    Returns:
        Decorator that dispatches invocations of the decorated function as remote jobs.
    """

    def decorator(func: Callable[_Args, _ReturnValue]) -> Callable[_Args, jb.MLJob[_ReturnValue]]:
        # Copy the function to avoid modifying the original
        # We need to modify the line number of the function to exclude the
        # decorator from the copied source code
        wrapped_func = copy.copy(func)
        wrapped_func.__code__ = wrapped_func.__code__.replace(co_firstlineno=func.__code__.co_firstlineno + 1)

        @functools.wraps(func)
        def wrapper(*args: _Args.args, **kwargs: _Args.kwargs) -> jb.MLJob[_ReturnValue]:
            payload = functools.partial(func, *args, **kwargs)
            setattr(payload, constants.IS_MLJOB_REMOTE_ATTR, True)
            job = jm._submit_job(
                source=payload,
                stage_name=stage_name,
                compute_pool=compute_pool,
                pip_requirements=pip_requirements,
                external_access_integrations=external_access_integrations,
                query_warehouse=query_warehouse,
                env_vars=env_vars,
                num_instances=num_instances,
                enable_metrics=enable_metrics,
                session=session,
            )
            assert isinstance(job, jb.MLJob), f"Unexpected job type: {type(job)}"
            return job

        return wrapper

    return decorator
