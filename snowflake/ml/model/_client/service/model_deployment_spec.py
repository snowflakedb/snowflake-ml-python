import pathlib
from typing import Any, Optional, Union, overload

import yaml

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model._client.service import model_deployment_spec_schema


class ModelDeploymentSpec:
    """Class to construct deploy.yml file for Model container services deployment.

    Attributes:
        workspace_path: A local path where model related files should be dumped to.
    """

    DEPLOY_SPEC_FILE_REL_PATH = "deploy.yml"

    def __init__(self, workspace_path: Optional[pathlib.Path] = None) -> None:
        self.workspace_path = workspace_path

    @overload
    def save(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        service_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_name: sql_identifier.SqlIdentifier,
        inference_compute_pool_name: sql_identifier.SqlIdentifier,
        image_build_compute_pool_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        cpu: Optional[str],
        memory: Optional[str],
        gpu: Optional[Union[str, int]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        # service spec
        ingress_enabled: bool,
        max_instances: int,
    ) -> str:
        ...

    @overload
    def save(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        job_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_name: sql_identifier.SqlIdentifier,
        inference_compute_pool_name: sql_identifier.SqlIdentifier,
        image_build_compute_pool_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        cpu: Optional[str],
        memory: Optional[str],
        gpu: Optional[Union[str, int]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        # job spec
        warehouse: sql_identifier.SqlIdentifier,
        target_method: str,
        input_table_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        input_table_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        input_table_name: sql_identifier.SqlIdentifier,
        output_table_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        output_table_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        output_table_name: sql_identifier.SqlIdentifier,
    ) -> str:
        ...

    def save(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        service_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_name: Optional[sql_identifier.SqlIdentifier] = None,
        inference_compute_pool_name: sql_identifier.SqlIdentifier,
        image_build_compute_pool_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        cpu: Optional[str],
        memory: Optional[str],
        gpu: Optional[Union[str, int]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]],
        # service spec
        ingress_enabled: Optional[bool] = None,
        max_instances: Optional[int] = None,
        # job spec
        warehouse: Optional[sql_identifier.SqlIdentifier] = None,
        target_method: Optional[str] = None,
        input_table_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        input_table_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        input_table_name: Optional[sql_identifier.SqlIdentifier] = None,
        output_table_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        output_table_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        output_table_name: Optional[sql_identifier.SqlIdentifier] = None,
    ) -> str:
        # create the deployment spec
        # models spec
        fq_model_name = identifier.get_schema_level_object_identifier(
            database_name.identifier(), schema_name.identifier(), model_name.identifier()
        )
        model = model_deployment_spec_schema.Model(name=fq_model_name, version=version_name.identifier())

        # image_build spec
        saved_image_repo_database = image_repo_database_name or database_name
        saved_image_repo_schema = image_repo_schema_name or schema_name
        fq_image_repo_name = identifier.get_schema_level_object_identifier(
            db=saved_image_repo_database.identifier(),
            schema=saved_image_repo_schema.identifier(),
            object_name=image_repo_name.identifier(),
        )

        image_build = model_deployment_spec_schema.ImageBuild(
            compute_pool=image_build_compute_pool_name.identifier(),
            image_repo=fq_image_repo_name,
            force_rebuild=force_rebuild,
            external_access_integrations=(
                [eai.identifier() for eai in external_access_integrations] if external_access_integrations else None
            ),
        )

        # universal base inference spec in service and job
        base_inference_spec: dict[str, Any] = {}
        if cpu:
            base_inference_spec["cpu"] = cpu
        if memory:
            base_inference_spec["memory"] = memory
        if gpu:
            if isinstance(gpu, int):
                gpu_str = str(gpu)
            else:
                gpu_str = gpu
            base_inference_spec["gpu"] = gpu_str
        if num_workers:
            base_inference_spec["num_workers"] = num_workers
        if max_batch_rows:
            base_inference_spec["max_batch_rows"] = max_batch_rows

        if service_name:  # service spec
            assert ingress_enabled, "ingress_enabled is required for service spec"
            assert max_instances, "max_instances is required for service spec"
            saved_service_database = service_database_name or database_name
            saved_service_schema = service_schema_name or schema_name
            fq_service_name = identifier.get_schema_level_object_identifier(
                saved_service_database.identifier(), saved_service_schema.identifier(), service_name.identifier()
            )
            service = model_deployment_spec_schema.Service(
                name=fq_service_name,
                compute_pool=inference_compute_pool_name.identifier(),
                ingress_enabled=ingress_enabled,
                max_instances=max_instances,
                **base_inference_spec,
            )

            # model deployment spec
            model_deployment_spec: Union[
                model_deployment_spec_schema.ModelServiceDeploymentSpec,
                model_deployment_spec_schema.ModelJobDeploymentSpec,
            ] = model_deployment_spec_schema.ModelServiceDeploymentSpec(
                models=[model],
                image_build=image_build,
                service=service,
            )
        else:  # job spec
            assert job_name, "job_name is required for job spec"
            assert warehouse, "warehouse is required for job spec"
            assert target_method, "target_method is required for job spec"
            assert input_table_name, "input_table_name is required for job spec"
            assert output_table_name, "output_table_name is required for job spec"
            saved_job_database = job_database_name or database_name
            saved_job_schema = job_schema_name or schema_name
            input_table_database_name = input_table_database_name or database_name
            input_table_schema_name = input_table_schema_name or schema_name
            output_table_database_name = output_table_database_name or database_name
            output_table_schema_name = output_table_schema_name or schema_name
            fq_job_name = identifier.get_schema_level_object_identifier(
                saved_job_database.identifier(), saved_job_schema.identifier(), job_name.identifier()
            )
            fq_input_table_name = identifier.get_schema_level_object_identifier(
                input_table_database_name.identifier(),
                input_table_schema_name.identifier(),
                input_table_name.identifier(),
            )
            fq_output_table_name = identifier.get_schema_level_object_identifier(
                output_table_database_name.identifier(),
                output_table_schema_name.identifier(),
                output_table_name.identifier(),
            )
            job = model_deployment_spec_schema.Job(
                name=fq_job_name,
                compute_pool=inference_compute_pool_name.identifier(),
                warehouse=warehouse.identifier(),
                target_method=target_method,
                input_table_name=fq_input_table_name,
                output_table_name=fq_output_table_name,
                **base_inference_spec,
            )

            # model deployment spec
            model_deployment_spec = model_deployment_spec_schema.ModelJobDeploymentSpec(
                models=[model],
                image_build=image_build,
                job=job,
            )

        if self.workspace_path is None:
            return yaml.safe_dump(model_deployment_spec.model_dump(exclude_none=True))

        # save the yaml
        file_path = self.workspace_path / self.DEPLOY_SPEC_FILE_REL_PATH
        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(model_deployment_spec.model_dump(exclude_none=True), f)
        return str(file_path.resolve())
