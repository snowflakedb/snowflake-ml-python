import json
import logging
import time
from typing import Optional

from snowflake.ml.model._deploy_client.utils import constants
from snowflake.snowpark import Session


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
        self.session.sql(f"CREATE OR REPLACE IMAGE REPOSITORY {repo_name}").collect()

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
        assert spec_stage_location.startswith("@"), f"stage path should start with @, actual: {spec_stage_location}"
        self._drop_service_if_exists(service_name)
        sql = f"""
             CREATE SERVICE {service_name}
                 MIN_INSTANCES={min_instances}
                 MAX_INSTANCES={max_instances}
                 COMPUTE_POOL={compute_pool}
                 SPEC={spec_stage_location}
         """
        logging.info(f"Create service with SQL: \n {sql}")
        self.session.sql(sql).collect()

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
    ) -> None:
        """Create or replace service function.

        Args:
            service_func_name: Name of the service function.
            service_name: Name of the service.
            endpoint_name: Name the service endpoint, declared in the service spec, indicating the listening port.
            path_at_service_endpoint: Specify the path/route at the service endpoint. Multiple paths can exist for a
                given endpoint. For example, an inference server listening on port 5000 may have paths like "/predict"
                and "/monitoring

        """
        sql = f"""
            CREATE OR REPLACE FUNCTION {service_func_name}(input OBJECT)
                RETURNS OBJECT
                SERVICE={service_name}
                ENDPOINT={endpoint_name}
                AS '/{path_at_service_endpoint}'
            """
        logging.info(f"Create service function with SQL: \n {sql}")
        self.session.sql(sql).collect()
        logging.info(f"Successfully created service function: {service_func_name}")

    def block_until_resource_is_ready(
        self,
        resource_name: str,
        resource_type: constants.ResourceType,
        *,
        max_retries: int = 60,
        retry_interval_secs: int = 5,
    ) -> None:
        """Blocks execution until the specified resource is ready.
        Note that this is a best-effort approach because when launching a service, it's possible for it to initially
        fail due to a system error. However, SnowService may automatically retry and recover the service, leading to
        potential false-negative information.

        Args:
            resource_name: Name of the resource.
            resource_type: Type of the resource.
            max_retries: The maximum number of retries to check the resource readiness (default: 60).
            retry_interval_secs: The number of seconds to wait between each retry (default: 5).

        Raises:
            RuntimeError: If the resource received the following status [failed, not_found, internal_error, deleting]
            RuntimeError: If the resource does not reach the ready/done state within the specified number of retries.
        """
        for _ in range(max_retries):
            status = self.get_resource_status(resource_name=resource_name, resource_type=resource_type)
            if status in [constants.ResourceStatus.READY, constants.ResourceStatus.DONE]:
                return
            elif status in [
                constants.ResourceStatus.FAILED,
                constants.ResourceStatus.NOT_FOUND,
                constants.ResourceStatus.INTERNAL_ERROR,
                constants.ResourceStatus.DELETING,
            ]:
                error_log = self.get_resource_log(
                    resource_name=resource_name,
                    resource_type=resource_type,
                    container_name=constants.INFERENCE_SERVER_CONTAINER,
                )
                raise RuntimeError(f"{resource_type} {resource_name} failed. \n {error_log if error_log else ''}")
            time.sleep(retry_interval_secs)

        raise RuntimeError("Resource never reached the ready/done state.")

    def get_resource_log(
        self, resource_name: str, resource_type: constants.ResourceType, container_name: str
    ) -> Optional[str]:
        if resource_type != constants.ResourceType.SERVICE:
            raise NotImplementedError(f"{resource_type.name} is not yet supported in get_resource_log function")
        try:
            row = self.session.sql(
                f"CALL SYSTEM$GET_SNOWSERVICE_LOGS('{resource_name}', '0', '{container_name}')"
            ).collect()
            return str(row[0]["SYSTEM$GET_SNOWSERVICE_LOGS"])
        except Exception:
            return None

    def get_resource_status(
        self, resource_name: str, resource_type: constants.ResourceType
    ) -> Optional[constants.ResourceStatus]:
        """Get resource status.

        Args:
            resource_name: Name of the resource.
            resource_type: Type of the resource.

        Raises:
            ValueError: If resource type does not have a corresponding system function for querying status.
            RuntimeError: If corresponding status call failed.

        Returns:
            Optional[constants.ResourceStatus]: The status of the resource, or None if the resource status is empty.
        """
        if resource_type not in constants.RESOURCE_TO_STATUS_FUNCTION_MAPPING:
            raise ValueError(f"Status querying is not supported for resources of type '{resource_type}'.")
        status_func = constants.RESOURCE_TO_STATUS_FUNCTION_MAPPING[resource_type]
        try:
            row = self.session.sql(f"CALL {status_func}('{resource_name}');").collect()
        except Exception as e:
            raise RuntimeError(f"Error while querying the {resource_type} {resource_name} status: {str(e)}")
        resource_metadata = json.loads(row[0][status_func])[0]
        logging.info(f"Resource status metadata: {resource_metadata}")
        if resource_metadata and resource_metadata["status"]:
            try:
                status = resource_metadata["status"]
                return constants.ResourceStatus(status)
            except ValueError:
                logging.warning(f"Unknown status returned: {status}")
        return None
