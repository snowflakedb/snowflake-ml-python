import copy
import functools
from typing import Any, Callable, Optional, TypeVar

from typing_extensions import ParamSpec

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.jobs import job as jb, manager as jm
from snowflake.ml.jobs._utils import payload_utils

_PROJECT = "MLJob"

_Args = ParamSpec("_Args")
_ReturnValue = TypeVar("_ReturnValue")


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def remote(
    compute_pool: str,
    *,
    stage_name: str,
    target_instances: int = 1,
    pip_requirements: Optional[list[str]] = None,
    external_access_integrations: Optional[list[str]] = None,
    session: Optional[snowpark.Session] = None,
    **kwargs: Any,
) -> Callable[[Callable[_Args, _ReturnValue]], Callable[_Args, jb.MLJob[_ReturnValue]]]:
    """
    Submit a job to the compute pool.

    Args:
        compute_pool: The compute pool to use for the job.
        stage_name: The name of the stage where the job payload will be uploaded.
        target_instances: The number of nodes in the job. If none specified, create a single node job.
        pip_requirements: A list of pip requirements for the job.
        external_access_integrations: A list of external access integrations.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments. Supported arguments:
            database (str): The database to use for the job.
            schema (str): The schema to use for the job.
            min_instances (int): The minimum number of nodes required to start the job.
                If none specified, defaults to target_instances. If set, the job
                will not start until the minimum number of nodes is available.
            env_vars (dict): Environment variables to set in container.
            enable_metrics (bool): Whether to enable metrics publishing for the job.
            query_warehouse (str): The query warehouse to use. Defaults to session warehouse.
            spec_overrides (dict): A dictionary of overrides for the service spec.

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
        def wrapper(*_args: _Args.args, **_kwargs: _Args.kwargs) -> jb.MLJob[_ReturnValue]:
            payload = payload_utils.create_function_payload(func, *_args, **_kwargs)
            job = jm._submit_job(
                source=payload,
                stage_name=stage_name,
                compute_pool=compute_pool,
                target_instances=target_instances,
                pip_requirements=pip_requirements,
                external_access_integrations=external_access_integrations,
                session=payload.session or session,
                **kwargs,
            )
            assert isinstance(job, jb.MLJob), f"Unexpected job type: {type(job)}"
            return job

        return wrapper

    return decorator
