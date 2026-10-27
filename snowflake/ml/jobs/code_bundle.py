import hashlib
import io
import logging
import re
import time
from pathlib import Path
from typing import Any, cast

import yaml

from snowflake import snowpark
from snowflake.connector import errors as connector_errors
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.mixins import SerializableSessionMixin
from snowflake.ml.jobs import job as jb
from snowflake.ml.jobs._utils import (
    constants,
    object_naming,
    option_parsing,
    payload_utils,
    query_helper,
    runtime_env_utils,
    stage_utils,
    type_utils,
)
from snowflake.snowpark import context as sp_context, exceptions as sp_exceptions

_PROJECT = "MLJob"
_CODE_BUNDLE_MANIFEST_FILENAME = "code_bundle.yaml"
_PUBLIC_PREVIEW_WARNING = "Code bundles are in Public Preview since 2.6.0. Do not use them in production."
_STAGE_SUBPATH_RE = re.compile(r"^[\w.-]+$")

logger = logging.getLogger(__name__)


class MLCodeBundle(SerializableSessionMixin):
    """A handle to a registered code bundle."""

    def __init__(
        self,
        name: str,
        *,
        session: snowpark.Session | None = None,
        stage_path: str | None = None,
    ) -> None:
        self._name = name
        self._session = session or sp_context.get_active_session()
        self._stage_path = stage_path

    @property
    def name(self) -> str:
        """The fully qualified name of the code bundle (``database.schema.name``)."""
        return self._name

    def __repr__(self) -> str:
        return f"MLCodeBundle('{self._name}')"

    def delete(self) -> None:
        """Drop this code bundle."""
        delete_code_bundle(self, session=self._session)


def _build_code_bundle_manifest(
    compute_pool: str,
    result_stage: str,
    runtime_environment: str,
    *,
    query_warehouse: str | None = None,
    target_instances: int = 1,
    min_instances: int = 1,
    enable_metrics: bool = True,
    **extra_bundle_fields: Any,
) -> str:
    """Build the ``code_bundle.yaml`` manifest content for an ML-type code bundle.

    Args:
        compute_pool: The compute pool jobs run on.
        result_stage: The stage jobs write results to.
        runtime_environment: The resolved runtime image jobs run on.
        query_warehouse: The warehouse jobs run queries against.
        target_instances: The number of instances to use for jobs.
        min_instances: The minimum number of instances required to start a job.
        enable_metrics: Whether to enable metrics publishing for jobs.
        extra_bundle_fields: Additional top-level bundle fields, such as ``artifact_repositories``,
            ``secrets``, ``external_access_integrations``, and ``env_vars``. Fields with empty values
            are omitted from the manifest.

    Returns:
        The rendered ``code_bundle.yaml`` content.
    """
    compute_options: dict[str, Any] = {
        "compute_pool": compute_pool,
        "query_warehouse": query_warehouse,
        "target_instances": target_instances,
        "min_instances": min_instances,
        # The manifest field is named runtime_version but accepts a runtime image tag or URL.
        "runtime_version": runtime_environment,
    }
    bundle: dict[str, Any] = {
        "type": "ml",
        "compute_type": "compute_pool",
        "language": "python",
        "compute_options": {key: value for key, value in compute_options.items() if value is not None},
        "properties": {
            "enable_metrics": enable_metrics,
            "result_stage": result_stage,
        },
    }
    for field_name, value in extra_bundle_fields.items():
        if value:
            bundle[field_name] = value

    return yaml.dump({"bundle": bundle}, sort_keys=False)


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    func_params_to_log=["pip_requirements", "target_instances"],
)
def register_code_bundle(
    source: str | Path,
    compute_pool: str,
    result_stage: str,
    *,
    stage_name: str,
    name: str | None = None,
    pip_requirements: list[str] | None = None,
    external_access_integrations: list[str] | None = None,
    target_instances: int = 1,
    session: snowpark.Session | None = None,
    **kwargs: Any,
) -> MLCodeBundle:
    """Package source into a stage and create a code bundle from it.

    Since there is no server-side manifest generation for this path, this function authors
    ``code_bundle.yaml`` itself from the arguments below and uploads it alongside the source,
    before issuing ``CREATE OR REPLACE CODE BUNDLE``.

    Args:
        source: The local file/directory path or stage path containing the code bundle source.
        compute_pool: The compute pool jobs run on when triggered from this bundle.
        result_stage: The stage jobs write results to. Must be a separate, writeable stage --
            the bundle's own stage is mounted read-only.
        stage_name: The stage the source is uploaded to, into a directory named after the bundle. An
            unqualified stage is resolved in the code bundle's database and schema, and is created if it
            does not exist. Re-registering a bundle replaces that directory's previous contents.
        name: The name of the code bundle. If not specified, a name is generated from ``source``.
        pip_requirements: A list of pip requirements to include with the bundle.
        external_access_integrations: A list of external access integrations.
        target_instances: The number of instances to use for jobs triggered from this bundle.
        session: The Snowpark session to use. If none specified, uses active session.
        kwargs: Additional keyword arguments. Supported arguments:
            database (str): The database to create the code bundle in. If not specified, uses the
                current database.
            schema (str): The schema to create the code bundle in. If not specified, uses the
                current schema.
            min_instances (int): The minimum number of instances required to start a job. If none
                specified, defaults to target_instances.
            env_vars (dict): Environment variables to set in the container.
            enable_metrics (bool): Whether to enable metrics publishing for jobs. Defaults to True.
            imports (list[Union[tuple[str, str], tuple[str]]]): A list of additional payloads used by
                the bundle.
            runtime_environment (str): The runtime image to use. Only support image tag or full image
                URL, e.g. "1.7.1" or "image_repo/image_name:image_tag". When it refers to a full
                image URL, it should contain image repository, image name and image tag.
            query_warehouse (str): The warehouse jobs run queries against. If not specified, uses the
                warehouse of the current session.
            secrets (list[str]): A list of secret objects to inject at runtime.
            artifact_repositories (list[str]): (Public Preview) A list of artifact repositories to use
                for resolving pip requirements. Each entry is the name of a Snowflake artifact
                repository.

    Returns:
        A handle to the registered code bundle.

    Raises:
        ValueError: If the instance counts are out of range, if ``source`` is a callable, if database
            is specified without schema, if database/schema cannot be resolved from either the
            arguments or the session context, if ``result_stage`` or ``stage_name`` is not a valid
            stage path, or if ``source`` is a stage path inside the bundle's own payload directory.
    """
    logger.warning(_PUBLIC_PREVIEW_WARNING)
    session = session or sp_context.get_active_session()

    options = option_parsing.pop_common_options(kwargs, target_instances)
    option_parsing.check_instance_counts(options)
    option_parsing.check_database_schema(options)
    secrets = kwargs.pop("secrets", None)
    query_warehouse = kwargs.pop("query_warehouse", None) or session.get_current_warehouse()
    option_parsing.warn_unknown_kwargs(kwargs)

    if callable(source):
        raise ValueError(
            "register_code_bundle does not support callable payloads;"
            " provide a local file, directory, or stage path."
        )

    database, schema = options.database, options.schema

    if name:
        parsed_database, parsed_schema, parsed_name = identifier.parse_schema_level_object_identifier(name)
        database = parsed_database or database
        schema = parsed_schema or schema
        name = parsed_name
        bundle_object_name = identifier.resolve_identifier(name)
    else:
        # Derived names may require SQL identifier quoting. Preserve the raw name here because the
        # payload directory is chosen separately from the SQL object name below.
        name = payload_utils.get_payload_name(str(source))
        bundle_object_name = identifier.get_inferred_name(name)

    database, schema = object_naming.resolve_database_schema(session, database, schema)
    qualified_result_stage = _qualify_stage(result_stage, database, schema)
    bundle_fqn = identifier.get_schema_level_object_identifier(database, schema, bundle_object_name)

    # Resolve the payload stage in the bundle's own database and schema, so bundles with the same name
    # in different schemas never share a payload directory.
    stage_subpath = _get_code_bundle_stage_subpath(name, bundle_fqn)
    stage_path = object_naming.build_object_stage_path(_qualify_stage(stage_name, database, schema), stage_subpath)

    resolved_source = cast(
        type_utils.PayloadPath,
        payload_utils.resolve_source(stage_utils.resolve_path(source) if isinstance(source, str) else source),
    )

    safe_stage_path = stage_utils.try_resolve_stage_path(stage_path.as_posix())
    if safe_stage_path is None:
        raise ValueError(f"{stage_path} is not a valid stage path.")
    if _is_inside_stage_directory(session, resolved_source, safe_stage_path):
        raise ValueError(
            f"The source {resolved_source} is inside the code bundle's payload directory {safe_stage_path}. "
            "Use a source outside that directory."
        )

    # Create stage if necessary
    stage_name_root = stage_path.parts[0].lstrip("@")
    # Explicitly check if the stage exists first since we may not have CREATE STAGE privilege
    try:
        query_helper.run_query(session, "describe stage identifier(?)", params=[stage_name_root])
    except sp_exceptions.SnowparkSQLException:
        query_helper.run_query(
            session,
            "create stage if not exists identifier(?)"
            " encryption = ( type = 'SNOWFLAKE_SSE' )"
            " comment = 'Created by snowflake.ml.jobs Python API'",
            params=[stage_name_root],
        )

    # CREATE OR REPLACE CODE BUNDLE snapshots everything in the payload directory, so clear what an earlier
    # registration under this name left there. The trailing slash keeps REMOVE from also matching sibling
    # directories that share this name as a prefix. Owner's-rights stored procedures reject REMOVE; there the
    # upload goes ahead without clearing, so files removed from the source since the last registration remain.
    try:
        query_helper.run_query(session, f"REMOVE {safe_stage_path}/")
    except sp_exceptions.SnowparkSQLException:
        pass

    payload_utils.upload_payloads(session, stage_path, type_utils.PayloadSpec(resolved_source, None), overwrite=True)
    payload_utils.upload_imports(options.imports, session, stage_path)

    if pip_requirements:
        session.file.put_stream(
            io.BytesIO("\n".join(pip_requirements).encode()),
            stage_location=stage_path.joinpath("requirements.txt").as_posix(),
            auto_compress=False,
            overwrite=True,
        )

    runtime_environment = runtime_env_utils.resolve_runtime_image(session, compute_pool, options.runtime_environment)

    manifest = _build_code_bundle_manifest(
        compute_pool,
        qualified_result_stage,
        runtime_environment,
        query_warehouse=query_warehouse,
        target_instances=options.target_instances,
        min_instances=options.min_instances,
        enable_metrics=options.enable_metrics,
        artifact_repositories=options.artifact_repositories,
        secrets=secrets,
        external_access_integrations=external_access_integrations,
        env_vars=[{key: value} for key, value in options.env_vars.items()] if options.env_vars else None,
    )
    session.file.put_stream(
        io.BytesIO(manifest.encode()),
        stage_location=stage_path.joinpath(_CODE_BUNDLE_MANIFEST_FILENAME).as_posix(),
        auto_compress=False,
        overwrite=True,
    )

    # The FROM clause only accepts a stage literal, not a bind parameter, so the validated path is
    # passed as an escaped string literal while the bundle name stays bound.
    query_helper.run_query(
        session,
        f"CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '{_escape_string_literal(safe_stage_path)}'",
        params=[bundle_fqn],
    )

    return MLCodeBundle(bundle_fqn, session=session, stage_path=safe_stage_path)


def _is_inside_stage_directory(session: snowpark.Session, source: type_utils.PayloadPath, directory: str) -> bool:
    """Whether a stage source lies inside the given fully qualified stage directory.

    An unqualified source stage resolves in the session's current database and schema, which is where
    Snowflake reads it from; without that context it cannot refer to the bundle's directory.

    Args:
        session: The session whose current database and schema an unqualified source resolves in.
        source: The resolved payload source.
        directory: The fully qualified payload directory of the code bundle.

    Returns:
        True if ``source`` is a stage path at or below ``directory``.
    """
    if not isinstance(source, stage_utils.StagePath) or not source.as_posix().startswith("@"):
        return False
    parsed_database, parsed_schema, _, _ = identifier.parse_snowflake_stage_path(source.as_posix().lstrip("@"))
    database = parsed_database or session.get_current_database()
    schema = parsed_schema or session.get_current_schema()
    if not database or not schema:
        return False
    qualified_source = stage_utils.StagePath(
        _qualify_stage(
            source.as_posix(), identifier.resolve_identifier(database), identifier.resolve_identifier(schema)
        )
    )
    return qualified_source.is_relative_to(stage_utils.StagePath(directory))


def _qualify_stage(stage: str, database: str, schema: str) -> str:
    """Fully qualify a stage path, filling in the database and schema the bundle is created in."""
    try:
        parsed_database, parsed_schema, stage_name, subpath = identifier.parse_snowflake_stage_path(stage.lstrip("@"))
    except ValueError:
        raise ValueError(f"{stage} is not a valid stage path.")
    return "@{}.{}.{}{}".format(
        identifier.resolve_identifier(parsed_database) if parsed_database else database,
        identifier.resolve_identifier(parsed_schema) if parsed_schema else schema,
        identifier.resolve_identifier(stage_name),
        subpath,
    )


def _get_code_bundle_stage_subpath(name: str, bundle_fqn: str) -> str:
    """Return a stable stage-directory name that is valid independently of SQL identifier quoting."""
    if name not in {".", ".."} and _STAGE_SUBPATH_RE.fullmatch(name):
        return name
    name_digest = hashlib.sha256(bundle_fqn.encode()).hexdigest()[:16]
    return f"code_bundle_{name_digest}"


def _escape_string_literal(value: str) -> str:
    """Escape a value for embedding in a single-quoted SQL string literal.

    Mirrors the escaping the Snowflake connector applies to bound string values. It is kept local
    because snowflake.connector.converter, which exposed this, was removed in connector 5, where
    the replacement lives under a private path.

    Args:
        value: The raw string to embed between single quotes.

    Returns:
        The escaped string, without the surrounding quotes.
    """
    # Backslash first: the replacements below introduce backslashes that must not be doubled.
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("'", "\\'")


@telemetry.send_api_usage_telemetry(project=_PROJECT, func_params_to_log=["timeout"])
def register_job_from_code_bundle(
    code_bundle: str | MLCodeBundle,
    *,
    entrypoint: str,
    args: list[str] | None = None,
    session: snowpark.Session | None = None,
    timeout: float = constants.CODE_BUNDLE_JOB_ID_TIMEOUT_SECONDS,
) -> jb.MLJob[None]:
    """Trigger a job from a registered code bundle.

    Submits ``EXECUTE CODE BUNDLE`` asynchronously and returns as soon as the ML Job id backing
    the execution can be recovered, rather than blocking until the job completes. A single code
    bundle can back any number of job executions.

    Args:
        code_bundle: The code bundle to run, as a name or an ``MLCodeBundle`` handle.
        entrypoint: The path to the entrypoint file within the code bundle.
        args: A list of arguments to pass to the job.
        session: The Snowpark session to use. If none specified, uses active session (or the
            session the ``MLCodeBundle`` was registered with, if one is passed).
        timeout: Maximum seconds to wait for the job id to appear in history.

    Returns:
        An object representing the triggered job.

    Raises:
        TimeoutError: If no job id appears within ``timeout`` seconds.  # noqa: DAR402
        SnowparkSQLException: If the code bundle execution fails before a job is created.  # noqa: DAR402
        ValueError: If ``timeout`` is not greater than 0.
    """
    logger.warning(_PUBLIC_PREVIEW_WARNING)
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0.")

    bundle_fqn, session, _ = _resolve_code_bundle(code_bundle, session)

    # ENTRYPOINT only accepts a string literal, not a bind parameter.
    query = f"EXECUTE CODE BUNDLE IDENTIFIER(?) ENTRYPOINT = '{_escape_string_literal(entrypoint)}'"
    params: list[Any] = [bundle_fqn]
    if args:
        placeholders = ", ".join(["?"] * len(args))
        query += f" ARGUMENTS = ({placeholders})"
        params.extend(args)

    bundle_database = identifier.parse_schema_level_object_identifier(bundle_fqn)[0] or session.get_current_database()

    async_job = session.sql(query, params=params).collect_nowait()
    job_id = _wait_for_job_id(session, async_job, database=bundle_database, timeout=timeout)
    return jb.MLJob[None](job_id, session=session)


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def delete_code_bundle(
    code_bundle: str | MLCodeBundle,
    *,
    session: snowpark.Session | None = None,
) -> None:
    """Drop a registered code bundle.

    The code bundle object is dropped first. If the handle still knows the payload stage used at
    registration, those files are then removed on a best-effort basis.

    Args:
        code_bundle: The code bundle to drop, as a name or an ``MLCodeBundle`` handle.
        session: The Snowpark session to use. If none specified, uses the session the
            handle was registered with, or the active session.
    """
    bundle_fqn, session, stage_path = _resolve_code_bundle(code_bundle, session)
    # Drop before removing files so a failed drop never leaves a bundle whose payload is gone.
    query_helper.run_query(session, "DROP CODE BUNDLE IDENTIFIER(?)", params=[bundle_fqn])
    if stage_path:
        safe_stage_path = stage_utils.try_resolve_stage_path(stage_path)
        if safe_stage_path is None:
            logger.warning(
                f"Skipping stage cleanup for code bundle {bundle_fqn}: {stage_path!r} is not a valid stage path."
            )
        else:
            try:
                session.sql(f"REMOVE {safe_stage_path}/").collect()
                logger.debug(f"Successfully cleaned up stage files for code bundle {bundle_fqn} at {safe_stage_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up stage files for code bundle {bundle_fqn}: {e}")


def _resolve_code_bundle(
    code_bundle: str | MLCodeBundle, session: snowpark.Session | None
) -> tuple[str, snowpark.Session, str | None]:
    """Resolve a code bundle name, session, and optional payload stage path."""
    if isinstance(code_bundle, MLCodeBundle):
        return code_bundle.name, session or code_bundle._session, code_bundle._stage_path

    parsed_database, parsed_schema, parsed_name = identifier.parse_schema_level_object_identifier(code_bundle)
    bundle_fqn = identifier.get_schema_level_object_identifier(
        identifier.resolve_identifier(parsed_database) if parsed_database else None,
        identifier.resolve_identifier(parsed_schema) if parsed_schema else None,
        identifier.resolve_identifier(parsed_name),
    )
    return bundle_fqn, session or sp_context.get_active_session(), None


def _wait_for_job_id(
    session: snowpark.Session,
    async_job: snowpark.AsyncJob,
    *,
    database: str | None,
    timeout: float = constants.CODE_BUNDLE_JOB_ID_TIMEOUT_SECONDS,
) -> str:
    """Poll CODE_BUNDLE_HISTORY for the ML Job id created by an EXECUTE CODE BUNDLE query.

    The history can list the job id shortly before the job's service exists, so once the id appears
    this waits up to ``CODE_BUNDLE_SERVICE_WAIT_SECONDS`` for the service before returning it.

    Args:
        session: The Snowpark session to use.
        async_job: The asynchronous EXECUTE CODE BUNDLE submission.
        database: The database the code bundle belongs to, whose INFORMATION_SCHEMA holds its history.
            If None, the session's current database is used.
        timeout: Maximum time to wait, in seconds.

    Returns:
        The ML Job id backing this execution.

    Raises:
        SnowparkSQLException: If the code bundle execution fails before a job is created.
        TimeoutError: If no job id appears within ``timeout`` seconds.
    """
    query_id = async_job.query_id
    information_schema = f"{database}.INFORMATION_SCHEMA" if database else "INFORMATION_SCHEMA"
    history_query = f"SELECT RUNTIME_NAME FROM TABLE({information_schema}.CODE_BUNDLE_HISTORY(QUERY_ID => ?))"

    deadline = time.monotonic() + timeout
    delay = float(constants.JOB_POLL_INITIAL_DELAY_SECONDS)
    while True:
        rows = query_helper.run_query(session, history_query, params=[query_id])
        if rows and rows[0][0]:
            job_id = cast(str, rows[0][0])
            _wait_for_job_service(session, job_id, max_wait=constants.CODE_BUNDLE_SERVICE_WAIT_SECONDS)
            return job_id
        if async_job.is_failed():
            # A statement that fails before creating a job never shows up in history; raise its error.
            try:
                async_job.result()
            except connector_errors.ProgrammingError as e:
                raise sp_exceptions.SnowparkSQLException(
                    e.msg or str(e), error_code=str(e.errno), conn_error=e, sfqid=e.sfqid, query=e.query
                ) from e
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"code bundle execution: timed out after {timeout}s waiting for the job started by query {query_id}."
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.2, constants.JOB_POLL_MAX_DELAY_SECONDS)


def _wait_for_job_service(session: snowpark.Session, job_id: str, *, max_wait: float) -> None:
    """Check every ``CODE_BUNDLE_SERVICE_POLL_SECONDS``, up to ``max_wait`` seconds, until the job's service exists.

    The job has already been created at this point, so this only delays returning it: any error other
    than the service not existing yet ends the wait instead of failing the call.

    Args:
        session: The Snowpark session to use.
        job_id: The fully qualified ML Job id.
        max_wait: Maximum time to wait, in seconds.
    """
    wait_deadline = time.monotonic() + max_wait
    while True:
        try:
            query_helper.run_query(session, "DESCRIBE SERVICE IDENTIFIER(?)", params=[job_id])
            return
        except sp_exceptions.SnowparkSQLException as e:
            if e.sql_error_code != 2003:
                return
        remaining = wait_deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(constants.CODE_BUNDLE_SERVICE_POLL_SECONDS, remaining))
