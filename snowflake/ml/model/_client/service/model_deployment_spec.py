import pathlib
from typing import List, Optional, Union

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

    def save(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
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
        cpu: Optional[str],
        memory: Optional[str],
        gpu: Optional[Union[str, int]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
        force_rebuild: bool,
        external_access_integrations: Optional[List[sql_identifier.SqlIdentifier]],
    ) -> str:
        # create the deployment spec
        # models spec
        fq_model_name = identifier.get_schema_level_object_identifier(
            database_name.identifier(), schema_name.identifier(), model_name.identifier()
        )
        model_dict = model_deployment_spec_schema.ModelDict(name=fq_model_name, version=version_name.identifier())

        # image_build spec
        saved_image_repo_database = image_repo_database_name or database_name
        saved_image_repo_schema = image_repo_schema_name or schema_name
        fq_image_repo_name = identifier.get_schema_level_object_identifier(
            saved_image_repo_database.identifier(), saved_image_repo_schema.identifier(), image_repo_name.identifier()
        )
        image_build_dict: model_deployment_spec_schema.ImageBuildDict = {
            "compute_pool": image_build_compute_pool_name.identifier(),
            "image_repo": fq_image_repo_name,
            "force_rebuild": force_rebuild,
        }
        if external_access_integrations is not None:
            image_build_dict["external_access_integrations"] = [
                eai.identifier() for eai in external_access_integrations
            ]

        # service spec
        saved_service_database = service_database_name or database_name
        saved_service_schema = service_schema_name or schema_name
        fq_service_name = identifier.get_schema_level_object_identifier(
            saved_service_database.identifier(), saved_service_schema.identifier(), service_name.identifier()
        )
        service_dict = model_deployment_spec_schema.ServiceDict(
            name=fq_service_name,
            compute_pool=service_compute_pool_name.identifier(),
            ingress_enabled=ingress_enabled,
            max_instances=max_instances,
        )
        if cpu:
            service_dict["cpu"] = cpu

        if memory:
            service_dict["memory"] = memory

        if gpu:
            if isinstance(gpu, int):
                gpu_str = str(gpu)
            else:
                gpu_str = gpu
            service_dict["gpu"] = gpu_str

        if num_workers:
            service_dict["num_workers"] = num_workers

        if max_batch_rows:
            service_dict["max_batch_rows"] = max_batch_rows

        # model deployment spec
        model_deployment_spec_dict = model_deployment_spec_schema.ModelDeploymentSpecDict(
            models=[model_dict],
            image_build=image_build_dict,
            service=service_dict,
        )

        # Anchors are not supported in the server, avoid that.
        yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
        if self.workspace_path is None:
            return yaml.safe_dump(model_deployment_spec_dict)
        # save the yaml
        file_path = self.workspace_path / self.DEPLOY_SPEC_FILE_REL_PATH
        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(model_deployment_spec_dict, f)
        return str(file_path.resolve())
