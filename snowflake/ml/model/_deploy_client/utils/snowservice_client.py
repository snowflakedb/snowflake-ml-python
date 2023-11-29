import json
import logging
import textwrap
import time
from typing import Optional

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import log_stream_processor, uri
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class SnowServiceClient:
    """
    SnowService client implementation: a Python wrapper for SnowService SQL queries.
    """

    def __init__(self, session: Session) -> None:
        """Initialization

        Args:
            session: Snowpark session
        """
        self.session = session

    def create_image_repo(self, repo_name: str) -> None:
        self.session.sql(f"CREATE IMAGE REPOSITORY IF NOT EXISTS {repo_name}").collect()

    def create_or_replace_service(
        self,
        service_name: str,
        compute_pool: str,
        spec_stage_location: str,
        *,
        min_instances: Optional[int] = 1,
        max_instances: Optional[int] = 1,
    ) -> None:
        """Create or replace service. Since SnowService doesn't support the CREATE OR REPLACE service syntax, we will
        first attempt to drop the service if it exists, and then create the service. Please note that this approach may
        have side effects due to the lack of transaction support.

        Args:
            service_name: Name of the service.
            min_instances: Minimum number of service replicas.
            max_instances: Maximum number of service replicas.
            compute_pool: Name of the compute pool.
            spec_stage_location: Stage path for the service spec.
        """
        stage, path = uri.get_stage_and_path(spec_stage_location)
        self._drop_service_if_exists(service_name)
        sql = textwrap.dedent(
            f"""
             CREATE SERVICE {service_name}
                IN COMPUTE POOL {compute_pool}
                FROM {stage}
                SPEC = '{path}'
                MIN_INSTANCES={min_instances}
                MAX_INSTANCES={max_instances}
            """
        )
        logger.info(f"Creating service {service_name}")
        logger.debug(f"Create service with SQL: \n {sql}")
        self.session.sql(sql).collect()

    def create_job(self, compute_pool: str, spec_stage_location: str) -> None:
        """Execute the job creation SQL command. Note that the job creation is synchronous, hence we execute it in a
        async way so that we can query the log in the meantime.

        Upon job failure, full job container log will be logged.

        Args:
            compute_pool: name of the compute pool
            spec_stage_location: path to the stage location where the spec is located at.

        """
        stage, path = uri.get_stage_and_path(spec_stage_location)
        sql = textwrap.dedent(
            f"""
            EXECUTE SERVICE
            IN COMPUTE POOL {compute_pool}
            FROM {stage}
            SPEC = '{path}'
            """
        )
        logger.debug(f"Create job with SQL: \n {sql}")
        cur = self.session._conn._conn.cursor()
        cur.execute_async(sql)
        job_id = cur._sfqid
        self.block_until_resource_is_ready(
            resource_name=str(job_id),
            resource_type=constants.ResourceType.JOB,
            container_name=constants.KANIKO_CONTAINER_NAME,
            max_retries=240,
            retry_interval_secs=15,
        )

    def _drop_service_if_exists(self, service_name: str) -> None:
        """Drop service if it already exists.

        Args:
            service_name: Name of the service.
        """
        self.session.sql(f"DROP SERVICE IF EXISTS {service_name}").collect()

    def create_or_replace_service_function(
        self,
        service_func_name: str,
        service_name: str,
        *,
        endpoint_name: str = constants.PREDICT,
        path_at_service_endpoint: str = constants.PREDICT,
        max_batch_rows: Optional[int] = None,
    ) -> str:
        """Create or replace service function.

        Args:
            service_func_name: Name of the service function.
            service_name: Name of the service.
            endpoint_name: Name the service endpoint, declared in the service spec, indicating the listening port.
            path_at_service_endpoint: Specify the path/route at the service endpoint. Multiple paths can exist for a
                given endpoint. For example, an inference server listening on port 5000 may have paths like "/predict"
                and "/monitoring
            max_batch_rows: Specify the MAX_BATCH_ROWS property of the service function, if None, leave unset

        Returns:
            The actual SQL for service function creation.
        """
        max_batch_rows_sql = ""
        if max_batch_rows:
            max_batch_rows_sql = f"MAX_BATCH_ROWS = {max_batch_rows}"

        sql = textwrap.dedent(
            f"""
            CREATE OR REPLACE FUNCTION {service_func_name}(input OBJECT)
                RETURNS OBJECT
                SERVICE={service_name}
                ENDPOINT={endpoint_name}
                {max_batch_rows_sql}
                AS '/{path_at_service_endpoint}'
            """
        )
        logger.debug(f"Create service function with SQL: \n {sql}")
        self.session.sql(sql).collect()
        logger.debug(f"Successfully created service function: {service_func_name}")
        return sql

    def block_until_resource_is_ready(
        self,
        resource_name: str,
        resource_type: constants.ResourceType,
        *,
        max_retries: int = 180,
        container_name: str = constants.INFERENCE_SERVER_CONTAINER,
        retry_interval_secs: int = 10,
    ) -> None:
        """Blocks execution until the specified resource is ready.
        Note that this is a best-effort approach because when launching a service, it's possible for it to initially
        fail due to a system error. However, SnowService may automatically retry and recover the service, leading to
        potential false-negative information.

        Args:
            resource_name: Name of the resource.
            resource_type: Type of the resource.
            container_name: The container to query the log from.
            max_retries: The maximum number of retries to check the resource readiness (default: 60).
            retry_interval_secs: The number of seconds to wait between each retry (default: 10).

        Raises:
            SnowflakeMLException: If the resource received the following status [failed, not_found, internal_error,
                deleting]
            SnowflakeMLException: If the resource does not reach the ready/done state within the specified number
                of retries.
        """
        assert resource_type == constants.ResourceType.SERVICE or resource_type == constants.ResourceType.JOB
        query_command = ""
        if resource_type == constants.ResourceType.SERVICE:
            query_command = f"CALL SYSTEM$GET_SERVICE_LOGS('{resource_name}', '0', '{container_name}')"
        elif resource_type == constants.ResourceType.JOB:
            query_command = f"CALL SYSTEM$GET_JOB_LOGS('{resource_name}', '{container_name}')"
        logger.warning(
            f"Best-effort log streaming from SPCS will be enabled when python logging level is set to INFO."
            f"Alternatively, you can also query the logs by running the query '{query_command}'"
        )
        lsp = log_stream_processor.LogStreamProcessor()

        for attempt_idx in range(max_retries):
            if logger.level <= logging.INFO:
                resource_log = self.get_resource_log(
                    resource_name=resource_name,
                    resource_type=resource_type,
                    container_name=container_name,
                )
                lsp.process_new_logs(resource_log, log_level=logging.INFO)

            status = self.get_resource_status(resource_name=resource_name, resource_type=resource_type)

            if resource_type == constants.ResourceType.JOB and status == constants.ResourceStatus.DONE:
                return
            elif resource_type == constants.ResourceType.SERVICE and status == constants.ResourceStatus.READY:
                return

            if (
                status
                in [
                    constants.ResourceStatus.FAILED,
                    constants.ResourceStatus.NOT_FOUND,
                    constants.ResourceStatus.INTERNAL_ERROR,
                    constants.ResourceStatus.DELETING,
                ]
                or attempt_idx >= max_retries - 1
            ):
                if logger.level > logging.INFO:
                    resource_log = self.get_resource_log(
                        resource_name=resource_name,
                        resource_type=resource_type,
                        container_name=container_name,
                    )
                    # Show full error log when logging level is above INFO level. For INFO level and below, we already
                    # show the log through logStreamProcessor above.
                    logger.error(resource_log)

                error_message = "failed"
                if attempt_idx >= max_retries - 1:
                    error_message = "does not reach ready/done status"

                if resource_type == constants.ResourceType.SERVICE:
                    self._drop_service_if_exists(service_name=resource_name)

                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_CONTAINER_SERVICE_ERROR,
                    original_exception=RuntimeError(
                        f"{resource_type} {resource_name} {error_message}." f"\nStatus: {status if status else ''} \n"
                    ),
                )
            time.sleep(retry_interval_secs)

    def get_resource_log(
        self, resource_name: str, resource_type: constants.ResourceType, container_name: str
    ) -> Optional[str]:
        if resource_type == constants.ResourceType.SERVICE:
            try:
                row = self.session.sql(
                    f"CALL SYSTEM$GET_SERVICE_LOGS('{resource_name}', '0', '{container_name}')"
                ).collect()
                return str(row[0]["SYSTEM$GET_SERVICE_LOGS"])
            except Exception:
                return None
        elif resource_type == constants.ResourceType.JOB:
            try:
                row = self.session.sql(f"CALL SYSTEM$GET_JOB_LOGS('{resource_name}', '{container_name}')").collect()
                return str(row[0]["SYSTEM$GET_JOB_LOGS"])
            except Exception:
                return None
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=NotImplementedError(
                    f"{resource_type.name} is not yet supported in get_resource_log function"
                ),
            )

    def get_resource_status(
        self, resource_name: str, resource_type: constants.ResourceType
    ) -> Optional[constants.ResourceStatus]:
        """Get resource status.

        Args:
            resource_name: Name of the resource.
            resource_type: Type of the resource.

        Raises:
            SnowflakeMLException: If resource type does not have a corresponding system function for querying status.
            SnowflakeMLException: If corresponding status call failed.

        Returns:
            Optional[constants.ResourceStatus]: The status of the resource, or None if the resource status is empty.
        """
        if resource_type not in constants.RESOURCE_TO_STATUS_FUNCTION_MAPPING:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Status querying is not supported for resources of type '{resource_type}'."
                ),
            )
        status_func = constants.RESOURCE_TO_STATUS_FUNCTION_MAPPING[resource_type]
        try:
            row = self.session.sql(f"CALL {status_func}('{resource_name}');").collect()
        except Exception:
            # Silent fail as SPCS status call is not guaranteed to return in time. Will rely on caller to retry.
            return None

        resource_metadata = json.loads(row[0][status_func])[0]
        logger.debug(f"Resource status metadata: {resource_metadata}")
        if resource_metadata and resource_metadata["status"]:
            try:
                status = resource_metadata["status"]
                return constants.ResourceStatus(status)
            except ValueError:
                logger.warning(f"Unknown status returned: {status}")
        return None
