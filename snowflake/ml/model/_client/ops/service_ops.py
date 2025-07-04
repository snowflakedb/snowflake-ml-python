import dataclasses
import hashlib
import logging
import pathlib
import re
import tempfile
import threading
import time
from typing import Any, Optional, Union, cast

from snowflake import snowpark
from snowflake.ml._internal import file_utils, platform_capabilities as pc
from snowflake.ml._internal.utils import identifier, service_logger, sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model._client.sql import service as service_sql, stage as stage_sql
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import async_job, dataframe, exceptions, row, session
from snowflake.snowpark._internal import utils as snowpark_utils

module_logger = service_logger.get_logger(__name__, service_logger.LogColor.GREY)
module_logger.propagate = False


@dataclasses.dataclass
class ServiceLogInfo:
    database_name: Optional[sql_identifier.SqlIdentifier]
    schema_name: Optional[sql_identifier.SqlIdentifier]
    service_name: sql_identifier.SqlIdentifier
    container_name: str
    instance_id: str = "0"

    def __post_init__(self) -> None:
        # service name used in logs for display
        self.display_service_name = sql_identifier.get_fully_qualified_name(
            self.database_name, self.schema_name, self.service_name
        )


@dataclasses.dataclass
class ServiceLogMetadata:
    service_logger: logging.Logger
    service: ServiceLogInfo
    service_status: Optional[service_sql.ServiceStatus]
    is_model_build_service_done: bool
    log_offset: int


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
        if pc.PlatformCapabilities.get_instance().is_inlined_deployment_spec_enabled():
            self._workspace = None
            self._model_deployment_spec = model_deployment_spec.ModelDeploymentSpec()
        else:
            self._workspace = tempfile.TemporaryDirectory()
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
        cpu_requests: Optional[str],
        memory_requests: Optional[str],
        gpu_requests: Optional[Union[int, str]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        build_external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        block: bool,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> Union[str, async_job.AsyncJob]:

        # Fall back to the registry's database and schema if not provided
        database_name = database_name or self._database_name
        schema_name = schema_name or self._schema_name

        # Fall back to the model's database and schema if not provided then to the registry's database and schema
        service_database_name = service_database_name or database_name or self._database_name
        service_schema_name = service_schema_name or schema_name or self._schema_name

        image_repo_database_name = image_repo_database_name or database_name or self._database_name
        image_repo_schema_name = image_repo_schema_name or schema_name or self._schema_name
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
        self._model_deployment_spec.add_image_build_spec(
            image_build_compute_pool_name=image_build_compute_pool_name,
            image_repo_database_name=image_repo_database_name,
            image_repo_schema_name=image_repo_schema_name,
            image_repo_name=image_repo_name,
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
        spec_yaml_str_or_path = self._model_deployment_spec.save()
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

        # deploy the model service
        query_id, async_job = self._service_client.deploy_model(
            stage_path=stage_path if self._workspace else None,
            model_deployment_spec_file_rel_path=(
                model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH if self._workspace else None
            ),
            model_deployment_spec_yaml_str=None if self._workspace else spec_yaml_str_or_path,
            statement_params=statement_params,
        )

        # stream service logs in a thread
        model_build_service_name = sql_identifier.SqlIdentifier(self._get_model_build_service_name(query_id))
        model_build_service = ServiceLogInfo(
            database_name=service_database_name,
            schema_name=service_schema_name,
            service_name=model_build_service_name,
            container_name="model-build",
        )
        model_inference_service = ServiceLogInfo(
            database_name=service_database_name,
            schema_name=service_schema_name,
            service_name=service_name,
            container_name="model-inference",
        )
        services = [model_build_service, model_inference_service]
        log_thread = self._start_service_log_streaming(
            async_job, services, model_inference_service_exists, force_rebuild, statement_params
        )

        if block:
            log_thread.join()

            res = cast(str, cast(list[row.Row], async_job.result())[0][0])
            module_logger.info(f"Inference service {service_name} deployment complete: {res}")
            return res
        else:
            return async_job

    def _start_service_log_streaming(
        self,
        async_job: snowpark.AsyncJob,
        services: list[ServiceLogInfo],
        model_inference_service_exists: bool,
        force_rebuild: bool,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> threading.Thread:
        """Start the service log streaming in a separate thread."""
        log_thread = threading.Thread(
            target=self._stream_service_logs,
            args=(
                async_job,
                services,
                model_inference_service_exists,
                force_rebuild,
                statement_params,
            ),
        )
        log_thread.start()
        return log_thread

    def _stream_service_logs(
        self,
        async_job: snowpark.AsyncJob,
        services: list[ServiceLogInfo],
        model_inference_service_exists: bool,
        force_rebuild: bool,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Stream service logs while the async job is running."""

        def fetch_logs(service: ServiceLogInfo, offset: int) -> tuple[str, int]:
            service_logs = self._service_client.get_service_logs(
                database_name=service.database_name,
                schema_name=service.schema_name,
                service_name=service.service_name,
                container_name=service.container_name,
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

        def set_service_log_metadata_to_model_inference(
            meta: ServiceLogMetadata, inference_service: ServiceLogInfo, msg: str
        ) -> None:
            model_inference_service_logger = service_logger.get_logger(  # InferenceServiceName-InstanceId
                f"{inference_service.display_service_name}-{inference_service.instance_id}",
                service_logger.LogColor.BLUE,
            )
            model_inference_service_logger.propagate = False
            meta.service_logger = model_inference_service_logger
            meta.service = inference_service
            meta.service_status = None
            meta.is_model_build_service_done = True
            meta.log_offset = 0
            block_size = 180
            module_logger.info(msg)
            module_logger.info("-" * block_size)

        model_build_service, model_inference_service = services[0], services[1]
        model_build_service_logger = service_logger.get_logger(  # BuildJobName
            model_build_service.display_service_name, service_logger.LogColor.GREEN
        )
        model_build_service_logger.propagate = False
        service_log_meta = ServiceLogMetadata(
            service_logger=model_build_service_logger,
            service=model_build_service,
            service_status=None,
            is_model_build_service_done=False,
            log_offset=0,
        )
        while not async_job.is_done():
            if model_inference_service_exists:
                time.sleep(5)
                continue

            try:
                # check if using an existing model build image
                if not force_rebuild and not service_log_meta.is_model_build_service_done:
                    model_build_service_exists = self._check_if_service_exists(
                        database_name=model_build_service.database_name,
                        schema_name=model_build_service.schema_name,
                        service_name=model_build_service.service_name,
                        statement_params=statement_params,
                    )
                    new_model_inference_service_exists = self._check_if_service_exists(
                        database_name=model_inference_service.database_name,
                        schema_name=model_inference_service.schema_name,
                        service_name=model_inference_service.service_name,
                        statement_params=statement_params,
                    )
                    if not model_build_service_exists and new_model_inference_service_exists:
                        set_service_log_metadata_to_model_inference(
                            service_log_meta,
                            model_inference_service,
                            (
                                "Model Inference image build is not rebuilding the image, but using a previously built "
                                "image."
                            ),
                        )
                        continue

                statuses = self._service_client.get_service_container_statuses(
                    database_name=service_log_meta.service.database_name,
                    schema_name=service_log_meta.service.schema_name,
                    service_name=service_log_meta.service.service_name,
                    include_message=True,
                    statement_params=statement_params,
                )
                service_status = statuses[0].service_status
                if (service_status != service_sql.ServiceStatus.RUNNING) or (
                    service_status != service_log_meta.service_status
                ):
                    service_log_meta.service_status = service_status
                    module_logger.info(
                        f"{'Inference' if service_log_meta.is_model_build_service_done else 'Image build'} service "
                        f"{service_log_meta.service.display_service_name} is "
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

                new_logs, new_offset = fetch_logs(
                    service_log_meta.service,
                    service_log_meta.log_offset,
                )
                if new_logs:
                    service_log_meta.service_logger.info(new_logs)
                    service_log_meta.log_offset = new_offset

                # check if model build service is done
                if not service_log_meta.is_model_build_service_done:
                    statuses = self._service_client.get_service_container_statuses(
                        database_name=model_build_service.database_name,
                        schema_name=model_build_service.schema_name,
                        service_name=model_build_service.service_name,
                        include_message=False,
                        statement_params=statement_params,
                    )
                    service_status = statuses[0].service_status

                    if service_status == service_sql.ServiceStatus.DONE:
                        set_service_log_metadata_to_model_inference(
                            service_log_meta,
                            model_inference_service,
                            f"Image build service {model_build_service.display_service_name} complete.",
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
                service_log_meta.service_logger, service_log_meta.service, service_log_meta.log_offset, statement_params
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
                container_name=service.container_name,
                statement_params=statement_params,
            )

            if len(service_logs) > offset:
                service_logger.info(service_logs[offset:])
        except Exception as ex:
            module_logger.warning(f"Caught an exception when logging: {repr(ex)}")

    @staticmethod
    def _get_model_build_service_name(query_id: str) -> str:
        """Get the model build service name through the server-side logic."""
        uuid = query_id.replace("-", "")
        big_int = int(uuid, 16)
        md5_hash = hashlib.md5(str(big_int).encode()).hexdigest()
        identifier = md5_hash[:8]
        return ("model_build_" + identifier).upper()

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

    def invoke_job_method(
        self,
        target_method: str,
        signature: model_signature.ModelSignature,
        X: Union[type_hints.SupportedDataType, dataframe.DataFrame],
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        job_database_name: Optional[sql_identifier.SqlIdentifier],
        job_schema_name: Optional[sql_identifier.SqlIdentifier],
        job_name: sql_identifier.SqlIdentifier,
        compute_pool_name: sql_identifier.SqlIdentifier,
        warehouse_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        output_table_database_name: Optional[sql_identifier.SqlIdentifier],
        output_table_schema_name: Optional[sql_identifier.SqlIdentifier],
        output_table_name: sql_identifier.SqlIdentifier,
        cpu_requests: Optional[str],
        memory_requests: Optional[str],
        gpu_requests: Optional[Union[int, str]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        build_external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> Union[type_hints.SupportedDataType, dataframe.DataFrame]:
        # fall back to the registry's database and schema if not provided
        database_name = database_name or self._database_name
        schema_name = schema_name or self._schema_name

        # fall back to the model's database and schema if not provided then to the registry's database and schema
        job_database_name = job_database_name or database_name or self._database_name
        job_schema_name = job_schema_name or schema_name or self._schema_name

        image_repo_database_name = image_repo_database_name or database_name or self._database_name
        image_repo_schema_name = image_repo_schema_name or schema_name or self._schema_name

        input_table_database_name = job_database_name
        input_table_schema_name = job_schema_name
        output_table_database_name = output_table_database_name or database_name or self._database_name
        output_table_schema_name = output_table_schema_name or schema_name or self._schema_name

        if self._workspace:
            stage_path = self._create_temp_stage(database_name, schema_name, statement_params)
        else:
            stage_path = None

        # validate and prepare input
        if not isinstance(X, dataframe.DataFrame):
            keep_order = True
            output_with_input_features = False
            df = model_signature._convert_and_validate_local_data(X, signature.inputs)
            s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
                self._session, df, keep_order=keep_order, features=signature.inputs, statement_params=statement_params
            )
        else:
            keep_order = False
            output_with_input_features = True
            s_df = X

        # only write the index and feature input columns
        cols = [snowpark_handler._KEEP_ORDER_COL_NAME] if snowpark_handler._KEEP_ORDER_COL_NAME in s_df.columns else []
        cols += [
            sql_identifier.SqlIdentifier(feature.name, case_sensitive=True).identifier() for feature in signature.inputs
        ]
        s_df = s_df.select(cols)
        original_cols = s_df.columns

        # input/output tables
        fq_output_table_name = identifier.get_schema_level_object_identifier(
            output_table_database_name.identifier(),
            output_table_schema_name.identifier(),
            output_table_name.identifier(),
        )
        tmp_input_table_id = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
        )
        fq_tmp_input_table_name = identifier.get_schema_level_object_identifier(
            job_database_name.identifier(),
            job_schema_name.identifier(),
            tmp_input_table_id.identifier(),
        )
        s_df.write.save_as_table(
            table_name=fq_tmp_input_table_name,
            mode="errorifexists",
            statement_params=statement_params,
        )

        try:
            self._model_deployment_spec.clear()
            # save the spec
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
                cpu=cpu_requests,
                memory=memory_requests,
                gpu=gpu_requests,
                num_workers=num_workers,
                max_batch_rows=max_batch_rows,
                warehouse=warehouse_name,
                target_method=target_method,
                input_table_database_name=input_table_database_name,
                input_table_schema_name=input_table_schema_name,
                input_table_name=tmp_input_table_id,
                output_table_database_name=output_table_database_name,
                output_table_schema_name=output_table_schema_name,
                output_table_name=output_table_name,
            )

            self._model_deployment_spec.add_image_build_spec(
                image_build_compute_pool_name=compute_pool_name,
                image_repo_database_name=image_repo_database_name,
                image_repo_schema_name=image_repo_schema_name,
                image_repo_name=image_repo_name,
                force_rebuild=force_rebuild,
                external_access_integrations=build_external_access_integrations,
            )

            spec_yaml_str_or_path = self._model_deployment_spec.save()
            if self._workspace:
                assert stage_path is not None
                file_utils.upload_directory_to_stage(
                    self._session,
                    local_path=pathlib.Path(self._workspace.name),
                    stage_path=pathlib.PurePosixPath(stage_path),
                    statement_params=statement_params,
                )

            # deploy the job
            query_id, async_job = self._service_client.deploy_model(
                stage_path=stage_path if self._workspace else None,
                model_deployment_spec_file_rel_path=(
                    model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH if self._workspace else None
                ),
                model_deployment_spec_yaml_str=None if self._workspace else spec_yaml_str_or_path,
                statement_params=statement_params,
            )

            while not async_job.is_done():
                time.sleep(5)
        finally:
            self._session.table(fq_tmp_input_table_name).drop_table()

        # handle the output
        df_res = self._session.table(fq_output_table_name)
        if keep_order:
            df_res = df_res.sort(
                snowpark_handler._KEEP_ORDER_COL_NAME,
                ascending=True,
            )
            df_res = df_res.drop(snowpark_handler._KEEP_ORDER_COL_NAME)

        if not output_with_input_features:
            df_res = df_res.drop(*original_cols)

        # get final result
        if not isinstance(X, dataframe.DataFrame):
            return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(
                df_res, features=signature.outputs, statement_params=statement_params
            )
        else:
            return df_res

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
