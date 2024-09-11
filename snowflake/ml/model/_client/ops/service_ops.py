import dataclasses
import hashlib
import logging
import pathlib
import queue
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

from snowflake import snowpark
from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model._client.sql import service as service_sql, stage as stage_sql
from snowflake.snowpark import exceptions, row, session
from snowflake.snowpark._internal import utils as snowpark_utils


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s [%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger(__name__)
logger.propagate = False


@dataclasses.dataclass
class ServiceLogInfo:
    service_name: str
    container_name: str
    instance_id: str = "0"


class ServiceOperator:
    """Service operator for container services logic."""

    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name
        self._workspace = tempfile.TemporaryDirectory()
        self._service_client = service_sql.ServiceSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._stage_client = stage_sql.StageSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._model_deployment_spec = model_deployment_spec.ModelDeploymentSpec(
            workspace_path=pathlib.Path(self._workspace.name)
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ServiceOperator):
            return False
        return self._service_client == __value._service_client

    @property
    def workspace_path(self) -> pathlib.Path:
        return pathlib.Path(self._workspace.name)

    def create_service(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        service_database_name: Optional[sql_identifier.SqlIdentifier],
        service_schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        image_build_compute_pool_name: sql_identifier.SqlIdentifier,
        service_compute_pool_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        ingress_enabled: bool,
        max_instances: int,
        gpu_requests: Optional[str],
        num_workers: Optional[int],
        force_rebuild: bool,
        build_external_access_integration: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        # create a temp stage
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        self._stage_client.create_tmp_stage(
            database_name=database_name,
            schema_name=schema_name,
            stage_name=stage_name,
            statement_params=statement_params,
        )
        stage_path = self._stage_client.fully_qualified_object_name(database_name, schema_name, stage_name)

        self._model_deployment_spec.save(
            database_name=database_name or self._database_name,
            schema_name=schema_name or self._schema_name,
            model_name=model_name,
            version_name=version_name,
            service_database_name=service_database_name,
            service_schema_name=service_schema_name,
            service_name=service_name,
            image_build_compute_pool_name=image_build_compute_pool_name,
            service_compute_pool_name=service_compute_pool_name,
            image_repo_database_name=image_repo_database_name,
            image_repo_schema_name=image_repo_schema_name,
            image_repo_name=image_repo_name,
            ingress_enabled=ingress_enabled,
            max_instances=max_instances,
            gpu=gpu_requests,
            num_workers=num_workers,
            force_rebuild=force_rebuild,
            external_access_integration=build_external_access_integration,
        )
        file_utils.upload_directory_to_stage(
            self._session,
            local_path=self.workspace_path,
            stage_path=pathlib.PurePosixPath(stage_path),
            statement_params=statement_params,
        )

        # check if the inference service is already running
        try:
            model_inference_service_status, _ = self._service_client.get_service_status(
                service_name=service_name,
                include_message=False,
                statement_params=statement_params,
            )
            model_inference_service_exists = model_inference_service_status == service_sql.ServiceStatus.READY
        except exceptions.SnowparkSQLException:
            model_inference_service_exists = False

        # deploy the model service
        query_id, async_job = self._service_client.deploy_model(
            stage_path=stage_path,
            model_deployment_spec_file_rel_path=model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH,
            statement_params=statement_params,
        )

        # stream service logs in a thread
        services = [
            ServiceLogInfo(service_name=self._get_model_build_service_name(query_id), container_name="model-build"),
            ServiceLogInfo(service_name=service_name, container_name="model-inference"),
        ]
        exception_queue: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        log_thread = self._start_service_log_streaming(
            async_job, services, model_inference_service_exists, exception_queue, statement_params
        )
        log_thread.join()

        try:
            # non-blocking check for an exception
            exception = exception_queue.get(block=False)
            if exception:
                raise exception
        except queue.Empty:
            pass

        return service_name

    def _start_service_log_streaming(
        self,
        async_job: snowpark.AsyncJob,
        services: List[ServiceLogInfo],
        model_inference_service_exists: bool,
        exception_queue: queue.Queue,  # type: ignore[type-arg]
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> threading.Thread:
        """Start the service log streaming in a separate thread."""
        log_thread = threading.Thread(
            target=self._stream_service_logs,
            args=(async_job, services, model_inference_service_exists, exception_queue, statement_params),
        )
        log_thread.start()
        return log_thread

    def _stream_service_logs(
        self,
        async_job: snowpark.AsyncJob,
        services: List[ServiceLogInfo],
        model_inference_service_exists: bool,
        exception_queue: queue.Queue,  # type: ignore[type-arg]
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stream service logs while the async job is running."""

        def fetch_logs(service_name: str, container_name: str, offset: int) -> Tuple[str, int]:
            service_logs = self._service_client.get_service_logs(
                service_name=service_name,
                container_name=container_name,
                statement_params=statement_params,
            )

            # return only new logs starting after the offset
            if len(service_logs) > offset:
                new_logs = service_logs[offset:]
                new_offset = len(service_logs)
            else:
                new_logs = ""
                new_offset = offset

            return new_logs, new_offset

        is_model_build_service_done = False
        log_offset = 0
        model_build_service, model_inference_service = services[0], services[1]
        service_name, container_name = model_build_service.service_name, model_build_service.container_name
        # BuildJobName
        service_logger = get_logger(service_name)
        service_logger.propagate = False
        while not async_job.is_done():
            if model_inference_service_exists:
                time.sleep(5)
                continue

            try:
                block_size = 180
                service_status, message = self._service_client.get_service_status(
                    service_name=service_name, include_message=True, statement_params=statement_params
                )
                logger.info(f"Inference service {service_name} is {service_status.value}.")

                new_logs, new_offset = fetch_logs(service_name, container_name, log_offset)
                if new_logs:
                    service_logger.info(new_logs)
                    log_offset = new_offset

                # check if model build service is done
                if not is_model_build_service_done:
                    service_status, _ = self._service_client.get_service_status(
                        service_name=model_build_service.service_name,
                        include_message=False,
                        statement_params=statement_params,
                    )

                    if service_status == service_sql.ServiceStatus.DONE:
                        is_model_build_service_done = True
                        log_offset = 0
                        service_name = model_inference_service.service_name
                        container_name = model_inference_service.container_name
                        # InferenceServiceName-InstanceId
                        service_logger = get_logger(f"{service_name}-{model_inference_service.instance_id}")
                        service_logger.propagate = False
                        logger.info(f"Model build service {model_build_service.service_name} complete.")
                        logger.info("-" * block_size)
            except ValueError:
                logger.warning(f"Unknown service status: {service_status.value}")
            except Exception as ex:
                logger.warning(f"Caught an exception when logging: {repr(ex)}")

            time.sleep(5)

        if model_inference_service_exists:
            logger.info(f"Inference service {model_inference_service.service_name} is already RUNNING.")
        else:
            self._finalize_logs(service_logger, services[-1], log_offset, statement_params)

        # catch exceptions from the deploy model execution
        try:
            res = cast(List[row.Row], async_job.result())
            logger.info(f"Model deployment for inference service {model_inference_service.service_name} complete.")
            logger.info(res[0][0])
        except Exception as ex:
            exception_queue.put(ex)

    def _finalize_logs(
        self,
        service_logger: logging.Logger,
        service: ServiceLogInfo,
        offset: int,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fetch service logs after the async job is done to ensure no logs are missed."""
        try:
            service_logs = self._service_client.get_service_logs(
                service_name=service.service_name,
                container_name=service.container_name,
                statement_params=statement_params,
            )

            if len(service_logs) > offset:
                service_logger.info(service_logs[offset:])
        except Exception as ex:
            logger.warning(f"Caught an exception when logging: {repr(ex)}")

    @staticmethod
    def _get_model_build_service_name(query_id: str) -> str:
        """Get the model build service name through the server-side logic."""
        most_significant_bits = uuid.UUID(query_id).int >> 64
        md5_hash = hashlib.md5(str(most_significant_bits).encode()).hexdigest()
        identifier = md5_hash[:6]
        return ("model_build_" + identifier).upper()
