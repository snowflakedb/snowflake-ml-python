import pathlib
import tempfile
from typing import Any, Dict, Optional

from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.model._client.sql import service as service_sql, stage as stage_sql
from snowflake.snowpark import session
from snowflake.snowpark._internal import utils as snowpark_utils


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
        image_name: Optional[sql_identifier.SqlIdentifier],
        ingress_enabled: bool,
        min_instances: int,
        max_instances: int,
        gpu_requests: Optional[str],
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
            image_name=image_name,
            ingress_enabled=ingress_enabled,
            min_instances=min_instances,
            max_instances=max_instances,
            gpu=gpu_requests,
            force_rebuild=force_rebuild,
            external_access_integration=build_external_access_integration,
        )
        file_utils.upload_directory_to_stage(
            self._session,
            local_path=self.workspace_path,
            stage_path=pathlib.PurePosixPath(stage_path),
            statement_params=statement_params,
        )

        # deploy the model service
        self._service_client.deploy_model(
            stage_path=stage_path,
            model_deployment_spec_file_rel_path=model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH,
            statement_params=statement_params,
        )

        return service_name
