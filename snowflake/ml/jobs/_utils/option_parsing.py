import dataclasses
import logging
import os
from typing import Any

from snowflake.ml.jobs._utils import constants

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CommonOptions:
    """Options accepted by every entry point that creates an ML Jobs object on a compute pool."""

    target_instances: int
    min_instances: int
    runtime_environment: str | None
    artifact_repositories: list[str] | None
    env_vars: dict[str, str] | None
    enable_metrics: bool
    imports: list[Any] | None
    database: str | None
    schema: str | None


def pop_common_options(kwargs: dict[str, Any], target_instances: int) -> CommonOptions:
    """Pop the commonly accepted options out of a caller's keyword arguments.

    Recognized keys are removed from ``kwargs`` in place, so the caller can pop the options specific
    to it and then pass the remainder to ``warn_unknown_kwargs``. Nothing is validated here: callers
    run ``check_database_schema`` and ``check_instance_counts`` at the point their own error order
    calls for.

    Args:
        kwargs: The caller's keyword arguments.
        target_instances: The requested instance count, which callers take as an explicit parameter
            or pop from ``kwargs`` themselves.

    Returns:
        The options that were popped.
    """
    min_instances = kwargs.pop("min_instances", target_instances)

    artifact_repositories = kwargs.pop("artifact_repositories", None)
    if artifact_repositories is not None:
        logger.warning("'artifact_repositories' is in public preview since 1.50.0.")

    database = kwargs.pop("database", None)
    schema = kwargs.pop("schema", None)

    return CommonOptions(
        target_instances=target_instances,
        min_instances=min_instances,
        runtime_environment=kwargs.pop(
            "runtime_environment", os.environ.get(constants.RUNTIME_IMAGE_TAG_ENV_VAR, None)
        ),
        artifact_repositories=artifact_repositories,
        env_vars=kwargs.pop("env_vars", None),
        enable_metrics=kwargs.pop("enable_metrics", True),
        imports=kwargs.pop("imports", None),
        database=database,
        schema=schema,
    )


def check_database_schema(options: CommonOptions) -> None:
    """Reject a database given without a schema.

    Args:
        options: The popped options.

    Raises:
        ValueError: If a database is specified without a schema.
    """
    if options.database and not options.schema:
        raise ValueError("Schema must be specified if database is specified.")


def check_instance_counts(options: CommonOptions) -> None:
    """Reject instance counts that are out of range.

    Args:
        options: The popped options.

    Raises:
        ValueError: If ``target_instances`` is below 1, or ``min_instances`` is not between 1 and
            ``target_instances``.
    """
    if options.target_instances < 1:
        raise ValueError("target_instances must be greater than 0.")
    if not (0 < options.min_instances <= options.target_instances):
        raise ValueError("min_instances must be greater than 0 and less than or equal to target_instances.")


def warn_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    """Warn about keyword arguments left unconsumed by the caller.

    Args:
        kwargs: The caller's keyword arguments, after every recognized key has been popped.
    """
    if kwargs:
        logger.warning(f"Ignoring unknown kwargs: {kwargs.keys()}")
