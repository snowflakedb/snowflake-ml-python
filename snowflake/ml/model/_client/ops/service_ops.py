import dataclasses
import enum
import hashlib
import logging
import pathlib
import re
import tempfile
import threading
import time
import warnings
from typing import Any, Optional, Union, cast

from snowflake import snowpark
from snowflake.ml import jobs
from snowflake.ml._internal import file_utils, platform_capabilities as pc
from snowflake.ml._internal.utils import identifier, service_logger, sql_identifier
from snowflake.ml.model import inference_engine as inference_engine_module, type_hints
from snowflake.ml.model._client.model import batch_inference_specs
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model._client.sql import service as service_sql, stage as stage_sql
from snowflake.snowpark import async_job, exceptions, row, session
from snowflake.snowpark._internal import utils as snowpark_utils

module_logger = service_logger.get_logger(__name__, service_logger.LogColor.GREY)
module_logger.propagate = False


class DeploymentStep(enum.Enum):
    MODEL_BUILD = ("model-build", "model_build_")
    MODEL_INFERENCE = ("model-inference", None)
    MODEL_LOGGING = ("model-logging", "model_logging_")

    def __init__(self, container_name: str, service_name_prefix: Optional[str]) -> None:
        self._container_name = container_name
        self._service_name_prefix = service_name_prefix

    @property
    def container_name(self) -> str:
        """Get the container name for the deployment step."""
        return self._container_name

    @property
    def service_name_prefix(self) -> Optional[str]:
        """Get the service name prefix for the deployment step."""
        return self._service_name_prefix


@dataclasses.dataclass
class ServiceLogInfo:
    database_name: Optional[sql_identifier.SqlIdentifier]
    schema_name: Optional[sql_identifier.SqlIdentifier]
    service_name: sql_identifier.SqlIdentifier
    deployment_step: DeploymentStep
    instance_id: str = "0"
    log_color: service_logger.LogColor = service_logger.LogColor.GREY

    def __post_init__(self) -> None:
        # service name used in logs for display
        self.display_service_name = sql_identifier.get_fully_qualified_name(
            self.database_name,
            self.schema_name,
            self.service_name,
        )

    def fetch_logs(
        self,
        service_client: service_sql.ServiceSQLClient,
        offset: int,
        statement_params: Optional[dict[str, Any]],
    ) -> tuple[str, int]:
        service_logs = service_client.get_service_logs(
            database_name=self.database_name,
            schema_name=self.schema_name,
            service_name=self.service_name,
            container_name=self.deployment_step.container_name,
            statement_params=statement_params,
        )

        # return only new logs starting after the offset
        new_logs = service_logs[offset:]
        new_offset = max(offset, len(service_logs))

        return new_logs, new_offset


@dataclasses.dataclass
class ServiceLogMetadata:
    service_logger: logging.Logger
    service: ServiceLogInfo
    service_status: Optional[service_sql.ServiceStatus]
    is_model_build_service_done: bool
    is_model_logger_service_done: bool
    log_offset: int

    def transition_service_log_metadata(
        self,
        to_service: ServiceLogInfo,
        msg: str,
        is_model_build_service_done: bool,
        is_model_logger_service_done: bool,
        operation_id: str,
        propagate: bool = False,
    ) -> None:
        to_service_logger = service_logger.get_logger(
            f"{to_service.display_service_name}-{to_service.instance_id}",
            to_service.log_color,
            operation_id=operation_id,
        )
        to_service_logger.propagate = propagate
        self.service_logger = to_service_logger
        self.service = to_service
        self.service_status = None
        self.is_model_build_service_done = is_model_build_service_done
        self.is_model_logger_service_done = is_model_logger_service_done
        self.log_offset = 0
        block_size = 180
        module_logger.info(msg)
        module_logger.info("-" * block_size)


@dataclasses.dataclass
class HFModelArgs:
    hf_model_name: str
    hf_task: Optional[str] = None
    hf_tokenizer: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_token: Optional[str] = None
    hf_trust_remote_code: bool = False
    hf_model_kwargs: Optional[dict[str, Any]] = None
    pip_requirements: Optional[list[str]] = None
    conda_dependencies: Optional[list[str]] = None
    comment: Optional[str] = None
    warehouse: Optional[str] = None


@dataclasses.dataclass
class InferenceEngineArgs:
    inference_engine: inference_engine_module.InferenceEngine
    inference_engine_args_override: Optional[list[str]] = None


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
        self._use_inlined_deployment_spec = pc.PlatformCapabilities.get_instance().is_inlined_deployment_spec_enabled()
        if self._use_inlined_deployment_spec:
            self._workspace = None
            self._model_deployment_spec = model_deployment_spec.ModelDeploymentSpec()
        else:
            self._workspace = tempfile.TemporaryDirectory()
            self._model_deployment_spec = model_deployment_spec.ModelDeploymentSpec(
                workspace_path=pathlib.Path(self._workspace.name)
            )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ServiceOperator):
            return False
        return self._service_client == __value._service_client

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
        image_repo_name: Optional[str],
        ingress_enabled: bool,
        max_instances: int,
        cpu_requests: Optional[str],
        memory_requests: Optional[str],
        gpu_requests: Optional[Union[int, str]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        build_external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        block: bool,
        progress_status: type_hints.ProgressStatus,
        statement_params: Optional[dict[str, Any]] = None,
        # hf model
        hf_model_args: Optional[HFModelArgs] = None,
        # inference engine model
        inference_engine_args: Optional[InferenceEngineArgs] = None,
    ) -> Union[str, async_job.AsyncJob]:

        # Generate operation ID for this deployment
        operation_id = service_logger.get_operation_id()

        # Fall back to the registry's database and schema if not provided
        database_name = database_name or self._database_name
        schema_name = schema_name or self._schema_name

        # Fall back to the model's database and schema if not provided then to the registry's database and schema
        service_database_name = service_database_name or database_name
        service_schema_name = service_schema_name or schema_name

        image_repo_fqn = ServiceOperator._get_image_repo_fqn(image_repo_name, database_name, schema_name)

        # There may be more conditions to enable image build in the future
        # For now, we only enable image build if inference engine is not specified
        is_enable_image_build = inference_engine_args is None

        # Step 1: Preparing deployment artifacts
        progress_status.update("preparing deployment artifacts...")
        progress_status.increment()

        if self._workspace:
            stage_path = self._create_temp_stage(database_name, schema_name, statement_params)
        else:
            stage_path = None
        self._model_deployment_spec.clear()
        self._model_deployment_spec.add_model_spec(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
        )

        if is_enable_image_build:
            self._model_deployment_spec.add_image_build_spec(
                image_build_compute_pool_name=image_build_compute_pool_name,
                fully_qualified_image_repo_name=image_repo_fqn,
                force_rebuild=force_rebuild,
                external_access_integrations=build_external_access_integrations,
            )

        self._model_deployment_spec.add_service_spec(
            service_database_name=service_database_name,
            service_schema_name=service_schema_name,
            service_name=service_name,
            inference_compute_pool_name=service_compute_pool_name,
            ingress_enabled=ingress_enabled,
            max_instances=max_instances,
            cpu=cpu_requests,
            memory=memory_requests,
            gpu=gpu_requests,
            num_workers=num_workers,
            max_batch_rows=max_batch_rows,
        )
        if hf_model_args:
            # hf model
            self._model_deployment_spec.add_hf_logger_spec(
                hf_model_name=hf_model_args.hf_model_name,
                hf_task=hf_model_args.hf_task,
                hf_token=(
                    # when using inlined deployment spec, we need to use QMARK_RESERVED_TOKEN
                    # to avoid revealing the token while calling the SYSTEM$DEPLOY_MODEL function
                    # noop if using file-based deployment spec or token is not provided
                    service_sql.QMARK_RESERVED_TOKEN
                    if hf_model_args.hf_token and self._use_inlined_deployment_spec
                    else hf_model_args.hf_token
                ),
                hf_tokenizer=hf_model_args.hf_tokenizer,
                hf_revision=hf_model_args.hf_revision,
                hf_trust_remote_code=hf_model_args.hf_trust_remote_code,
                pip_requirements=hf_model_args.pip_requirements,
                conda_dependencies=hf_model_args.conda_dependencies,
                comment=hf_model_args.comment,
                warehouse=hf_model_args.warehouse,
                **(hf_model_args.hf_model_kwargs if hf_model_args.hf_model_kwargs else {}),
            )

        if inference_engine_args:
            self._model_deployment_spec.add_inference_engine_spec(
                inference_engine=inference_engine_args.inference_engine,
                inference_engine_args=inference_engine_args.inference_engine_args_override,
            )

        spec_yaml_str_or_path = self._model_deployment_spec.save()

        # Step 2: Uploading deployment artifacts
        progress_status.update("uploading deployment artifacts...")
        progress_status.increment()

        if self._workspace:
            assert stage_path is not None
            file_utils.upload_directory_to_stage(
                self._session,
                local_path=pathlib.Path(self._workspace.name),
                stage_path=pathlib.PurePosixPath(stage_path),
                statement_params=statement_params,
            )

        # check if the inference service is already running/suspended
        model_inference_service_exists = self._check_if_service_exists(
            database_name=service_database_name,
            schema_name=service_schema_name,
            service_name=service_name,
            service_status_list_if_exists=[
                service_sql.ServiceStatus.RUNNING,
                service_sql.ServiceStatus.SUSPENDING,
                service_sql.ServiceStatus.SUSPENDED,
            ],
            statement_params=statement_params,
        )

        # Step 3: Initiating model deployment
        progress_status.update("initiating model deployment...")
        progress_status.increment()

        # deploy the model service
        query_id, async_job = self._service_client.deploy_model(
            stage_path=stage_path if self._workspace else None,
            model_deployment_spec_file_rel_path=(
                model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH if self._workspace else None
            ),
            model_deployment_spec_yaml_str=None if self._workspace else spec_yaml_str_or_path,
            query_params=(
                # when using inlined deployment spec, we need to add the token to the query params
                # to avoid revealing the token while calling the SYSTEM$DEPLOY_MODEL function
                # noop if using file-based deployment spec or token is not provided
                [hf_model_args.hf_token]
                if (self._use_inlined_deployment_spec and hf_model_args and hf_model_args.hf_token)
                else []
            ),
            statement_params=statement_params,
        )

        model_build_service: Optional[ServiceLogInfo] = None
        if is_enable_image_build:
            # stream service logs in a thread
            model_build_service_name = sql_identifier.SqlIdentifier(
                self._get_service_id_from_deployment_step(query_id, DeploymentStep.MODEL_BUILD)
            )
            model_build_service = ServiceLogInfo(
                database_name=service_database_name,
                schema_name=service_schema_name,
                service_name=model_build_service_name,
                deployment_step=DeploymentStep.MODEL_BUILD,
                log_color=service_logger.LogColor.GREEN,
            )

        model_inference_service = ServiceLogInfo(
            database_name=service_database_name,
            schema_name=service_schema_name,
            service_name=service_name,
            deployment_step=DeploymentStep.MODEL_INFERENCE,
            log_color=service_logger.LogColor.BLUE,
        )

        model_logger_service: Optional[ServiceLogInfo] = None
        if hf_model_args:
            model_logger_service_name = sql_identifier.SqlIdentifier(
                self._get_service_id_from_deployment_step(query_id, DeploymentStep.MODEL_LOGGING)
            )

            model_logger_service = ServiceLogInfo(
                database_name=service_database_name,
                schema_name=service_schema_name,
                service_name=model_logger_service_name,
                deployment_step=DeploymentStep.MODEL_LOGGING,
                log_color=service_logger.LogColor.ORANGE,
            )

        # start service log streaming
        log_thread = self._start_service_log_streaming(
            async_job=async_job,
            model_logger_service=model_logger_service,
            model_build_service=model_build_service,
            model_inference_service=model_inference_service,
            model_inference_service_exists=model_inference_service_exists,
            force_rebuild=force_rebuild,
            operation_id=operation_id,
            statement_params=statement_params,
        )

        if block:
            try:
                # Step 4: Starting model build: waits for build to start
                progress_status.update("starting model image build...")
                progress_status.increment()

                # Poll for model build to start if not using existing service
                if not model_inference_service_exists and model_build_service:
                    self._wait_for_service_status(
                        model_build_service_name,
                        service_sql.ServiceStatus.RUNNING,
                        service_database_name,
                        service_schema_name,
                        async_job,
                        statement_params,
                    )

                # Step 5: Building model image
                progress_status.update("building model image...")
                progress_status.increment()

                # Poll for model build completion
                if not model_inference_service_exists and model_build_service:
                    self._wait_for_service_status(
                        model_build_service_name,
                        service_sql.ServiceStatus.DONE,
                        service_database_name,
                        service_schema_name,
                        async_job,
                        statement_params,
                    )

                # Step 6: Deploying model service (push complete, starting inference service)
                progress_status.update("deploying model service...")
                progress_status.increment()

                log_thread.join()

                res = cast(str, cast(list[row.Row], async_job.result())[0][0])
                module_logger.info(f"Inference service {service_name} deployment complete: {res}")
                return res

            except RuntimeError as e:
                # Handle service creation/deployment failures
                error_msg = f"Model service deployment failed: {str(e)}"
                module_logger.error(error_msg)

                # Update progress status to show failure
                progress_status.update(error_msg, state="error")

                # Stop the log thread if it's running
                if "log_thread" in locals() and log_thread.is_alive():
                    log_thread.join(timeout=5)  # Give it a few seconds to finish gracefully

                # Re-raise the exception to propagate the error
                raise RuntimeError(error_msg) from e

        return async_job

    @staticmethod
    def _get_image_repo_fqn(
        image_repo_name: Optional[str],
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> Optional[str]:
        """Get the fully qualified name of the image repository."""
        if image_repo_name is None or image_repo_name.strip() == "":
            return None
        # Parse image repo
        (
            image_repo_database_name,
            image_repo_schema_name,
            image_repo_name,
        ) = sql_identifier.parse_fully_qualified_name(image_repo_name)
        image_repo_database_name = image_repo_database_name or database_name
        image_repo_schema_name = image_repo_schema_name or schema_name
        return identifier.get_schema_level_object_identifier(
            db=image_repo_database_name.identifier(),
            schema=image_repo_schema_name.identifier(),
            object_name=image_repo_name.identifier(),
        )

    def _start_service_log_streaming(
        self,
        async_job: snowpark.AsyncJob,
        model_logger_service: Optional[ServiceLogInfo],
        model_build_service: Optional[ServiceLogInfo],
        model_inference_service: ServiceLogInfo,
        model_inference_service_exists: bool,
        force_rebuild: bool,
        operation_id: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> threading.Thread:
        """Start the service log streaming in a separate thread."""
        # TODO: create a DAG of services and stream logs in the order of the DAG
        log_thread = threading.Thread(
            target=self._stream_service_logs,
            args=(
                async_job,
                model_logger_service,
                model_build_service,
                model_inference_service,
                model_inference_service_exists,
                force_rebuild,
                operation_id,
                statement_params,
            ),
        )
        log_thread.start()
        return log_thread

    def _fetch_log_and_update_meta(
        self,
        force_rebuild: bool,
        service_log_meta: ServiceLogMetadata,
        model_build_service: Optional[ServiceLogInfo],
        model_inference_service: ServiceLogInfo,
        operation_id: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Helper function to fetch logs and update the service log metadata if needed.

        This function checks the service status and fetches logs if the service exists.
        It also updates the service log metadata with the
        new service status and logs.
        If the service is done, it transitions the service log metadata.

        Args:
            force_rebuild: Whether to force rebuild the model build image.
            service_log_meta: The ServiceLogMetadata holds the state of the service log metadata.
            model_build_service: The ServiceLogInfo for the model build service.
            model_inference_service: The ServiceLogInfo for the model inference service.
            operation_id: The operation ID for the service, e.g. "model_deploy_a1b2c3d4_1703875200"
            statement_params: The statement parameters to use for the service client.
        """

        service = service_log_meta.service
        # check if using an existing model build image
        if (
            service.deployment_step == DeploymentStep.MODEL_BUILD
            and not force_rebuild
            and service_log_meta.is_model_logger_service_done
            and not service_log_meta.is_model_build_service_done
        ):
            model_build_service_exists = self._check_if_service_exists(
                database_name=service.database_name,
                schema_name=service.schema_name,
                service_name=service.service_name,
                statement_params=statement_params,
            )
            new_model_inference_service_exists = self._check_if_service_exists(
                database_name=model_inference_service.database_name,
                schema_name=model_inference_service.schema_name,
                service_name=model_inference_service.service_name,
                statement_params=statement_params,
            )
            if not model_build_service_exists and new_model_inference_service_exists:
                service_log_meta.transition_service_log_metadata(
                    model_inference_service,
                    "Model build is not rebuilding the inference image, but using a previously built image.",
                    is_model_build_service_done=True,
                    is_model_logger_service_done=service_log_meta.is_model_logger_service_done,
                    operation_id=operation_id,
                )

        try:
            statuses = self._service_client.get_service_container_statuses(
                database_name=service.database_name,
                schema_name=service.schema_name,
                service_name=service.service_name,
                include_message=True,
                statement_params=statement_params,
            )
            service_status = statuses[0].service_status
        except exceptions.SnowparkSQLException:
            # If the service is not found, log that the service is not found
            # and wait for a few seconds before returning
            module_logger.info(f"Service status for service {service.display_service_name} not found.")
            time.sleep(5)
            return

        # Case 1: service_status is PENDING or the service_status changed
        if (service_status != service_sql.ServiceStatus.RUNNING) or (service_status != service_log_meta.service_status):
            service_log_meta.service_status = service_status

            if service.deployment_step == DeploymentStep.MODEL_BUILD:
                module_logger.info(
                    f"Image build service {service.display_service_name} is "
                    f"{service_log_meta.service_status.value}."
                )
            elif service.deployment_step == DeploymentStep.MODEL_INFERENCE:
                module_logger.info(
                    f"Inference service {service.display_service_name} is {service_log_meta.service_status.value}."
                )
            elif service.deployment_step == DeploymentStep.MODEL_LOGGING:
                module_logger.info(
                    f"Model logger service {service.display_service_name} is "
                    f"{service_log_meta.service_status.value}."
                )
            for status in statuses:
                if status.instance_id is not None:
                    instance_status, container_status = None, None
                    if status.instance_status is not None:
                        instance_status = status.instance_status.value
                    if status.container_status is not None:
                        container_status = status.container_status.value
                    module_logger.info(
                        f"Instance[{status.instance_id}]: "
                        f"instance status: {instance_status}, "
                        f"container status: {container_status}, "
                        f"message: {status.message}"
                    )
            time.sleep(5)

        # Case 2: service_status is RUNNING
        # stream logs and update the log offset
        if service_status == service_sql.ServiceStatus.RUNNING:
            new_logs, new_offset = service.fetch_logs(
                self._service_client,
                service_log_meta.log_offset,
                statement_params=statement_params,
            )
            if new_logs:
                service_log_meta.service_logger.info(new_logs)
                service_log_meta.log_offset = new_offset

        # Case 3: service_status is DONE
        if service_status == service_sql.ServiceStatus.DONE:
            # check if model logger service is done
            # and transition the service log metadata to the model image build service
            if service.deployment_step == DeploymentStep.MODEL_LOGGING:
                if model_build_service:
                    # building the inference image, transition to the model build service
                    service_log_meta.transition_service_log_metadata(
                        model_build_service,
                        f"Model Logger service {service.display_service_name} complete.",
                        is_model_build_service_done=False,
                        is_model_logger_service_done=service_log_meta.is_model_logger_service_done,
                        operation_id=operation_id,
                    )
                else:
                    # no model build service, transition to the model inference service
                    service_log_meta.transition_service_log_metadata(
                        model_inference_service,
                        f"Model Logger service {service.display_service_name} complete.",
                        is_model_build_service_done=True,
                        is_model_logger_service_done=service_log_meta.is_model_logger_service_done,
                        operation_id=operation_id,
                    )
            # check if model build service is done
            # and transition the service log metadata to the model inference service
            elif service.deployment_step == DeploymentStep.MODEL_BUILD:
                service_log_meta.transition_service_log_metadata(
                    model_inference_service,
                    f"Image build service {service.display_service_name} complete.",
                    is_model_build_service_done=True,
                    is_model_logger_service_done=service_log_meta.is_model_logger_service_done,
                    operation_id=operation_id,
                )
            elif service.deployment_step == DeploymentStep.MODEL_INFERENCE:
                module_logger.info(f"Inference service {service.display_service_name} is deployed.")
            else:
                module_logger.warning(f"Service {service.display_service_name} is done, but not transitioning.")

    def _enforce_save_mode(self, output_mode: batch_inference_specs.SaveMode, output_stage_location: str) -> None:
        """Enforce the save mode for the output stage location.

        Args:
            output_mode: The output mode
            output_stage_location: The output stage location to check/clean.

        Raises:
            FileExistsError: When ERROR mode is specified and files exist in the output location.
            RuntimeError: When operations fail (checking files or removing files).
            ValueError: When an invalid SaveMode is specified.
        """
        list_results = self._stage_client.list_stage(output_stage_location)

        if output_mode == batch_inference_specs.SaveMode.ERROR:
            if len(list_results) > 0:
                raise FileExistsError(
                    f"Output stage location '{output_stage_location}' is not empty. "
                    f"Found {len(list_results)} existing files. When using ERROR mode, the output location "
                    f"must be empty. Please clear the existing files or use OVERWRITE mode."
                )
        elif output_mode == batch_inference_specs.SaveMode.OVERWRITE:
            if len(list_results) > 0:
                warnings.warn(
                    f"Output stage location '{output_stage_location}' is not empty. "
                    f"Found {len(list_results)} existing files. OVERWRITE mode will remove all existing files "
                    f"in the output location before running the batch inference job.",
                    stacklevel=2,
                )
                try:
                    self._session.sql(f"REMOVE {output_stage_location}").collect()
                except Exception as e:
                    raise RuntimeError(
                        f"OVERWRITE was specified. However, failed to remove existing files in output stage "
                        f"{output_stage_location}: {e}. Please clear up the existing files manually and retry "
                        f"the operation."
                    )
        else:
            valid_modes = list(batch_inference_specs.SaveMode)
            raise ValueError(f"Invalid SaveMode: {output_mode}. Must be one of {valid_modes}")

    def _stream_service_logs(
        self,
        async_job: snowpark.AsyncJob,
        model_logger_service: Optional[ServiceLogInfo],
        model_build_service: Optional[ServiceLogInfo],
        model_inference_service: ServiceLogInfo,
        model_inference_service_exists: bool,
        force_rebuild: bool,
        operation_id: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Stream service logs while the async job is running."""

        if model_build_service:
            model_build_service_logger = service_logger.get_logger(
                model_build_service.display_service_name,  # BuildJobName
                model_build_service.log_color,
                operation_id=operation_id,
            )
            service_log_meta = ServiceLogMetadata(
                service_logger=model_build_service_logger,
                service=model_build_service,
                service_status=None,
                is_model_build_service_done=False,
                is_model_logger_service_done=True,
                log_offset=0,
            )
        elif model_logger_service:
            model_logger_service_logger = service_logger.get_logger(
                model_logger_service.display_service_name,  # ModelLoggerName
                model_logger_service.log_color,
                operation_id=operation_id,
            )

            service_log_meta = ServiceLogMetadata(
                service_logger=model_logger_service_logger,
                service=model_logger_service,
                service_status=None,
                is_model_build_service_done=False,
                is_model_logger_service_done=False,
                log_offset=0,
            )
        else:
            model_inference_service_logger = service_logger.get_logger(
                model_inference_service.display_service_name,  # ModelInferenceName
                model_inference_service.log_color,
                operation_id=operation_id,
            )
            service_log_meta = ServiceLogMetadata(
                service_logger=model_inference_service_logger,
                service=model_inference_service,
                service_status=None,
                is_model_build_service_done=False,
                is_model_logger_service_done=False,
                log_offset=0,
            )

        while not async_job.is_done():
            if model_inference_service_exists:
                time.sleep(5)
                continue

            try:
                # fetch logs for the service
                # (model logging, model build, or model inference)
                # upon completion, transition to the next service if any
                self._fetch_log_and_update_meta(
                    service_log_meta=service_log_meta,
                    force_rebuild=force_rebuild,
                    model_build_service=model_build_service,
                    model_inference_service=model_inference_service,
                    operation_id=operation_id,
                    statement_params=statement_params,
                )
            except Exception as ex:
                pattern = r"002003 \(02000\)"  # error code: service does not exist
                is_snowpark_sql_exception = isinstance(ex, exceptions.SnowparkSQLException)
                contains_msg = any(msg in str(ex) for msg in ["Pending scheduling", "Waiting to start"])
                matches_pattern = service_log_meta.service_status is None and re.search(pattern, str(ex)) is not None

                if not (is_snowpark_sql_exception and (contains_msg or matches_pattern)):
                    module_logger.warning(f"Caught an exception when logging: {repr(ex)}")
                time.sleep(5)

        if model_inference_service_exists:
            module_logger.info(
                f"Inference service {model_inference_service.display_service_name} has already been deployed."
            )
        else:
            self._finalize_logs(
                service_log_meta.service_logger,
                service_log_meta.service,
                service_log_meta.log_offset,
                statement_params,
            )

    def _finalize_logs(
        self,
        service_logger: logging.Logger,
        service: ServiceLogInfo,
        offset: int,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fetch service logs after the async job is done to ensure no logs are missed."""
        try:
            time.sleep(5)  # wait for complete service logs
            service_logs = self._service_client.get_service_logs(
                database_name=service.database_name,
                schema_name=service.schema_name,
                service_name=service.service_name,
                container_name=service.deployment_step.container_name,
                statement_params=statement_params,
            )

            if len(service_logs) > offset:
                service_logger.info(service_logs[offset:])
        except Exception as ex:
            module_logger.warning(f"Caught an exception when logging: {repr(ex)}")

    def _wait_for_service_status(
        self,
        service_name: sql_identifier.SqlIdentifier,
        target_status: service_sql.ServiceStatus,
        service_database_name: Optional[sql_identifier.SqlIdentifier],
        service_schema_name: Optional[sql_identifier.SqlIdentifier],
        async_job: snowpark.AsyncJob,
        statement_params: Optional[dict[str, Any]] = None,
        timeout_minutes: int = 30,
    ) -> None:
        """Wait for service to reach the specified status while monitoring async job for failures.

        Args:
            service_name: The service to monitor
            target_status: The target status to wait for
            service_database_name: Database containing the service
            service_schema_name: Schema containing the service
            async_job: The async job to monitor for completion/failure
            statement_params: SQL statement parameters
            timeout_minutes: Maximum time to wait before timing out

        Raises:
            RuntimeError: If service fails, times out, or enters an error state
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        service_seen_before = False

        while True:
            # Check if async job has failed (but don't return on success - we need specific service status)
            if async_job.is_done():
                try:
                    async_job.result()
                    # Async job completed successfully, but we're waiting for a specific service status
                    # This might mean the service completed and was cleaned up
                    module_logger.debug(
                        f"Async job completed but we're still waiting for {service_name} to reach {target_status.value}"
                    )
                except Exception as e:
                    raise RuntimeError(f"Service deployment failed: {e}")

            try:
                statuses = self._service_client.get_service_container_statuses(
                    database_name=service_database_name,
                    schema_name=service_schema_name,
                    service_name=service_name,
                    include_message=True,
                    statement_params=statement_params,
                )

                if statuses:
                    service_seen_before = True
                    current_status = statuses[0].service_status

                    # Check if we've reached the target status
                    if current_status == target_status:
                        return

                    # Check for failure states
                    if current_status in [service_sql.ServiceStatus.FAILED, service_sql.ServiceStatus.INTERNAL_ERROR]:
                        error_msg = f"Service {service_name} failed with status {current_status.value}"
                        if statuses[0].message:
                            error_msg += f": {statuses[0].message}"
                        raise RuntimeError(error_msg)

            except exceptions.SnowparkSQLException as e:
                # Service might not exist yet - this is expected during initial deployment
                if "does not exist" in str(e) or "002003" in str(e):
                    # If we're waiting for DONE status and we've seen the service before,
                    # it likely completed and was cleaned up
                    if target_status == service_sql.ServiceStatus.DONE and service_seen_before:
                        module_logger.debug(
                            f"Service {service_name} disappeared after being seen, "
                            f"assuming it reached {target_status.value} and was cleaned up"
                        )
                        return

                    module_logger.debug(f"Service {service_name} not found yet, continuing to wait...")
                else:
                    # Re-raise unexpected SQL exceptions
                    raise RuntimeError(f"Error checking service status: {e}")
            except Exception as e:
                # Re-raise unexpected exceptions instead of masking them
                raise RuntimeError(f"Unexpected error while waiting for service status: {e}")

            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise RuntimeError(
                    f"Timeout waiting for service {service_name} to reach status {target_status.value} "
                    f"after {timeout_minutes} minutes"
                )

            time.sleep(2)  # Poll every 2 seconds

    @staticmethod
    def _get_service_id_from_deployment_step(query_id: str, deployment_step: DeploymentStep) -> str:
        """Get the service ID through the server-side logic."""
        uuid = query_id.replace("-", "")
        big_int = int(uuid, 16)
        md5_hash = hashlib.md5(str(big_int).encode()).hexdigest()
        identifier = md5_hash[:8]
        service_name_prefix = deployment_step.service_name_prefix
        if service_name_prefix is None:
            # raise an exception if the service name prefix is None
            raise ValueError(f"Service name prefix is {service_name_prefix} for deployment step {deployment_step}.")
        return (service_name_prefix + identifier).upper()

    def _check_if_service_exists(
        self,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        service_status_list_if_exists: Optional[list[service_sql.ServiceStatus]] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> bool:
        if service_status_list_if_exists is None:
            service_status_list_if_exists = [
                service_sql.ServiceStatus.PENDING,
                service_sql.ServiceStatus.RUNNING,
                service_sql.ServiceStatus.SUSPENDING,
                service_sql.ServiceStatus.SUSPENDED,
                service_sql.ServiceStatus.DONE,
                service_sql.ServiceStatus.FAILED,
            ]
        try:
            statuses = self._service_client.get_service_container_statuses(
                database_name=database_name,
                schema_name=schema_name,
                service_name=service_name,
                include_message=False,
                statement_params=statement_params,
            )
            service_status = statuses[0].service_status
            return any(service_status == status for status in service_status_list_if_exists)
        except exceptions.SnowparkSQLException:
            return False

    def invoke_batch_job_method(
        self,
        *,
        function_name: str,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        job_name: str,
        compute_pool_name: sql_identifier.SqlIdentifier,
        warehouse: sql_identifier.SqlIdentifier,
        image_repo_name: Optional[str],
        input_stage_location: str,
        input_file_pattern: str,
        output_stage_location: str,
        completion_filename: str,
        force_rebuild: bool,
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        cpu_requests: Optional[str],
        memory_requests: Optional[str],
        gpu_requests: Optional[str],
        replicas: Optional[int],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> jobs.MLJob[Any]:
        database_name = self._database_name
        schema_name = self._schema_name

        job_database_name, job_schema_name, job_name = sql_identifier.parse_fully_qualified_name(job_name)
        job_database_name = job_database_name or database_name
        job_schema_name = job_schema_name or schema_name

        self._model_deployment_spec.clear()

        self._model_deployment_spec.add_model_spec(
            database_name=database_name,
            schema_name=schema_name,
            model_name=model_name,
            version_name=version_name,
        )

        self._model_deployment_spec.add_job_spec(
            job_database_name=job_database_name,
            job_schema_name=job_schema_name,
            job_name=job_name,
            inference_compute_pool_name=compute_pool_name,
            num_workers=num_workers,
            max_batch_rows=max_batch_rows,
            input_stage_location=input_stage_location,
            input_file_pattern=input_file_pattern,
            output_stage_location=output_stage_location,
            completion_filename=completion_filename,
            function_name=function_name,
            warehouse=warehouse,
            cpu=cpu_requests,
            memory=memory_requests,
            gpu=gpu_requests,
            replicas=replicas,
        )

        self._model_deployment_spec.add_image_build_spec(
            image_build_compute_pool_name=compute_pool_name,
            fully_qualified_image_repo_name=self._get_image_repo_fqn(image_repo_name, database_name, schema_name),
            force_rebuild=force_rebuild,
        )

        spec_yaml_str_or_path = self._model_deployment_spec.save()

        if self._workspace:
            module_logger.info("using workspace")
            stage_path = self._create_temp_stage(database_name, schema_name, statement_params)
            file_utils.upload_directory_to_stage(
                self._session,
                local_path=pathlib.Path(self._workspace.name),
                stage_path=pathlib.PurePosixPath(stage_path),
                statement_params=statement_params,
            )
        else:
            module_logger.info("not using workspace")
            stage_path = None

        _, async_job = self._service_client.deploy_model(
            stage_path=stage_path if self._workspace else None,
            model_deployment_spec_file_rel_path=(
                model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH if self._workspace else None
            ),
            model_deployment_spec_yaml_str=None if self._workspace else spec_yaml_str_or_path,
            statement_params=statement_params,
        )

        # Block until the async job is done
        async_job.result()

        return jobs.MLJob(
            id=sql_identifier.get_fully_qualified_name(job_database_name, job_schema_name, job_name),
            session=self._session,
        )

    def _create_temp_stage(
        self,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> str:
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        self._stage_client.create_tmp_stage(
            database_name=database_name,
            schema_name=schema_name,
            stage_name=stage_name,
            statement_params=statement_params,
        )
        return self._stage_client.fully_qualified_object_name(database_name, schema_name, stage_name)  # stage path
