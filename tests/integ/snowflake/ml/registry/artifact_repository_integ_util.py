"""Helpers to provision an external test PyPI artifact repository for integ tests.

Creates a Snowflake SECRET (from ``PRIVATE_PYPI_USERNAME`` / ``PRIVATE_PYPI_PASSWORD``),
an API INTEGRATION, and a PYPI ARTIFACT REPOSITORY pointing at the shared external
test PyPI index. Intended for registry integ tests that need a private/external index
rather than ``snowflake.snowpark.pypi_shared_repository``.
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

PRIVATE_PYPI_USERNAME_ENV = "PRIVATE_PYPI_USERNAME"
PRIVATE_PYPI_PASSWORD_ENV = "PRIVATE_PYPI_PASSWORD"

EXTERNAL_TEST_PYPI_INDEX_URL = "https://snowflake-inc.repo.sonatype.app/repository/pypi-test-hosted/simple/"
EXTERNAL_TEST_PYPI_ALLOWED_PREFIXES: tuple[str, ...] = (
    "https://snowflake-inc.repo.sonatype.app",
    # External test PyPI serves artifacts through S3.
    "https://sonatype-repo-usw2-p-snowflake-inc-primary.s3.us-west-2.amazonaws.com",
)

_TEST_OBJECT_PREFIX = "snowml_test_"


def _test_object_name(run_id: str, suffix: str) -> str:
    """Build a ``snowml_test_<timestamp>_<run_id>_<suffix>`` object name."""
    return f"{_TEST_OBJECT_PREFIX}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{run_id}_{suffix}"


def _sql_string_literal(value: str) -> str:
    """Return a single-quoted SQL string literal with quotes escaped."""
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class ExternalTestPypiArtifactRepository:
    """Fully qualified objects for an external test PyPI artifact repository."""

    secret_fqn: str
    api_integration_name: str
    artifact_repository_fqn: str

    @property
    def artifact_repository_map(self) -> dict[str, str]:
        """Map suitable for ``log_model`` / ``create_service`` ``artifact_repository_map``."""
        return {"pip": self.artifact_repository_fqn}


def get_private_pypi_credentials() -> Optional[tuple[str, str]]:
    """Read private PyPI credentials from the environment.

    Returns:
        ``(username, password)`` when both ``PRIVATE_PYPI_USERNAME`` and
        ``PRIVATE_PYPI_PASSWORD`` are set; otherwise ``None``.
    """
    username = os.getenv(PRIVATE_PYPI_USERNAME_ENV)
    password = os.getenv(PRIVATE_PYPI_PASSWORD_ENV)
    if not username or not password:
        return None
    return username, password


def create_external_test_pypi_artifact_repository(
    session: Any,
    *,
    database: str,
    schema: str,
    run_id: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> ExternalTestPypiArtifactRepository:
    """Create secret, API integration, and external test PyPI artifact repository.

    Object names are derived from ``run_id`` using the standard ``snowml_test_``
    prefix. The secret and artifact repository are schema-scoped under
    ``database.schema``; the API integration is account-scoped and must be dropped
    explicitly (see ``drop_external_test_pypi_artifact_repository``).

    Args:
        session: Active Snowpark session with privileges to create the objects.
        database: Database for the secret and artifact repository.
        schema: Schema for the secret and artifact repository.
        run_id: Unique id used to name account- and schema-scoped objects.
        username: Optional override for ``PRIVATE_PYPI_USERNAME``.
        password: Optional override for ``PRIVATE_PYPI_PASSWORD``.

    Returns:
        Created object names and a ready-to-use ``artifact_repository_map``.

    Raises:
        ValueError: If credentials are not provided and env vars are unset.
    """
    if username is None or password is None:
        credentials = get_private_pypi_credentials()
        if credentials is None:
            raise ValueError(
                f"Private PyPI credentials required: set {PRIVATE_PYPI_USERNAME_ENV} and "
                f"{PRIVATE_PYPI_PASSWORD_ENV}, or pass username and password."
            )
        env_username, env_password = credentials
        if username is None:
            username = env_username
        if password is None:
            password = env_password

    secret_name = _test_object_name(run_id, "ext_pypi_secret").upper()
    api_integration_name = _test_object_name(run_id, "ext_pypi_api").upper()
    repository_name = _test_object_name(run_id, "ext_pypi_repo").upper()

    secret_fqn = f"{database}.{schema}.{secret_name}"
    artifact_repository_fqn = f"{database}.{schema}.{repository_name}"
    allowed_prefixes_sql = ",\n            ".join(
        _sql_string_literal(prefix) for prefix in EXTERNAL_TEST_PYPI_ALLOWED_PREFIXES
    )

    logger.info("Creating external test PyPI secret %s", secret_fqn)
    session.sql(
        f"""
        CREATE OR REPLACE SECRET {secret_fqn}
          TYPE = PASSWORD
          USERNAME = {_sql_string_literal(username)}
          PASSWORD = {_sql_string_literal(password)}
        """
    ).collect()

    logger.info("Creating external test PyPI API integration %s", api_integration_name)
    session.sql(
        f"""
        CREATE OR REPLACE API INTEGRATION {api_integration_name}
          API_PROVIDER = ARTIFACT_REPOSITORY_API
          API_ALLOWED_PREFIXES = (
            {allowed_prefixes_sql}
          )
          ALLOWED_AUTHENTICATION_SECRETS = ({secret_fqn})
          ENABLED = TRUE
        """
    ).collect()

    logger.info("Creating external test PyPI artifact repository %s", artifact_repository_fqn)
    session.sql(
        f"""
        CREATE OR REPLACE ARTIFACT REPOSITORY {artifact_repository_fqn}
          API_INTEGRATION = {api_integration_name}
          INDEX_URL = {_sql_string_literal(EXTERNAL_TEST_PYPI_INDEX_URL)}
          AUTHENTICATION_SECRET = {secret_fqn}
          TYPE = PYPI
        """
    ).collect()

    return ExternalTestPypiArtifactRepository(
        secret_fqn=secret_fqn,
        api_integration_name=api_integration_name,
        artifact_repository_fqn=artifact_repository_fqn,
    )


def drop_external_test_pypi_artifact_repository(
    session: Any,
    repo: ExternalTestPypiArtifactRepository,
    *,
    if_exists: bool = True,
) -> None:
    """Drop artifact repository, API integration, and secret created by the helper.

    Drop order respects dependencies (repository -> API integration -> secret).
    Best-effort when ``if_exists`` is True: failures are logged and ignored.

    Args:
        session: Active Snowpark session.
        repo: Objects previously returned by ``create_external_test_pypi_artifact_repository``.
        if_exists: When True, ignore missing-object errors during cleanup.

    Raises:
        Exception: Propagated from Snowflake when ``if_exists`` is False and a drop fails.
    """
    if_exists_sql = " IF EXISTS" if if_exists else ""
    drop_statements = (
        f"DROP ARTIFACT REPOSITORY{if_exists_sql} {repo.artifact_repository_fqn}",
        f"DROP API INTEGRATION{if_exists_sql} {repo.api_integration_name}",
        f"DROP SECRET{if_exists_sql} {repo.secret_fqn}",
    )
    for statement in drop_statements:
        try:
            session.sql(statement).collect()
        except Exception:
            if not if_exists:
                raise
            logger.warning("Best-effort cleanup failed for: %s", statement, exc_info=True)


def maybe_create_external_test_pypi_artifact_repository(
    session: Any,
    *,
    database: str,
    schema: str,
    run_id: str,
) -> Optional[ExternalTestPypiArtifactRepository]:
    """Create the external test PyPI repo when credentials are available.

    Args:
        session: Active Snowpark session with privileges to create the objects.
        database: Database for the secret and artifact repository.
        schema: Schema for the secret and artifact repository.
        run_id: Unique id used to name account- and schema-scoped objects.

    Returns:
        Created repository handle, or ``None`` if ``PRIVATE_PYPI_USERNAME`` /
        ``PRIVATE_PYPI_PASSWORD`` are unset.
    """
    if get_private_pypi_credentials() is None:
        logger.info(
            "Skipping external test PyPI artifact repository setup: %s / %s not set",
            PRIVATE_PYPI_USERNAME_ENV,
            PRIVATE_PYPI_PASSWORD_ENV,
        )
        return None
    return create_external_test_pypi_artifact_repository(
        session,
        database=database,
        schema=schema,
        run_id=run_id,
    )


@contextmanager
def external_test_pypi_artifact_repository(
    session: Any,
    *,
    database: str,
    schema: str,
    run_id: str,
) -> Iterator[ExternalTestPypiArtifactRepository]:
    """Context manager that creates then drops the external test PyPI repository.

    Args:
        session: Active Snowpark session with privileges to create the objects.
        database: Database for the secret and artifact repository.
        schema: Schema for the secret and artifact repository.
        run_id: Unique id used to name account- and schema-scoped objects.

    Yields:
        Created repository handle.
    """
    repo = create_external_test_pypi_artifact_repository(
        session,
        database=database,
        schema=schema,
        run_id=run_id,
    )
    try:
        yield repo
    finally:
        drop_external_test_pypi_artifact_repository(session, repo)
