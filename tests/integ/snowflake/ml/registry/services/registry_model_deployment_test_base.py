import inspect
import os
import pathlib
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import yaml
from absl.testing import absltest

from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import snowflake_env, sql_identifier
from snowflake.ml.model import ModelVersion, type_hints as model_types
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.registry import registry
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)


@pytest.mark.spcs_deployment_image
@absltest.skipUnless(
    test_env_utils.get_current_snowflake_cloud_type() == snowflake_env.SnowflakeCloudType.AWS,
    "SPCS only available in AWS",
)
class RegistryModelDeploymentTestBase(common_test_base.CommonTestBase):
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _SPCS_EAI = "SPCS_EGRESS_ACCESS_INTEGRATION"
    _TEST_SPCS_WH = "REGTEST_ML_SMALL"

    BUILDER_IMAGE_PATH = os.getenv("BUILDER_IMAGE_PATH", None)
    BASE_CPU_IMAGE_PATH = os.getenv("BASE_CPU_IMAGE_PATH", None)
    BASE_GPU_IMAGE_PATH = os.getenv("BASE_GPU_IMAGE_PATH", None)

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

        self._run_id = uuid.uuid4().hex[:2]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = "PUBLIC"
        self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "image_repo"
        ).upper()

        self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_image_repo(self._test_image_repo)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _deploy_model_with_image_override(
        self,
        mv: ModelVersion,
        service_name: str,
        gpu_requests: Optional[str] = None,
    ) -> None:
        """Deploy model with image override."""
        is_gpu = gpu_requests is not None
        image_path = self.BASE_GPU_IMAGE_PATH if is_gpu else self.BASE_CPU_IMAGE_PATH
        build_compute_pool = sql_identifier.SqlIdentifier(self._TEST_CPU_COMPUTE_POOL)
        service_compute_pool = sql_identifier.SqlIdentifier(
            self._TEST_GPU_COMPUTE_POOL if is_gpu else self._TEST_CPU_COMPUTE_POOL
        )

        # create a temp stage
        database_name = sql_identifier.SqlIdentifier(self._test_db)
        schema_name = sql_identifier.SqlIdentifier(self._test_schema)
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        image_repo_name = sql_identifier.SqlIdentifier(self._test_image_repo)

        mv._service_ops._stage_client.create_tmp_stage(
            database_name=database_name, schema_name=schema_name, stage_name=stage_name
        )
        stage_path = mv._service_ops._stage_client.fully_qualified_object_name(database_name, schema_name, stage_name)

        deploy_spec_file_rel_path = model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH

        mv._service_ops._model_deployment_spec.save(
            database_name=database_name,
            schema_name=schema_name,
            model_name=mv._model_name,
            version_name=mv._version_name,
            service_database_name=database_name,
            service_schema_name=schema_name,
            service_name=sql_identifier.SqlIdentifier(service_name),
            image_build_compute_pool_name=build_compute_pool,
            service_compute_pool_name=service_compute_pool,
            image_repo_database_name=database_name,
            image_repo_schema_name=schema_name,
            image_repo_name=image_repo_name,
            ingress_enabled=False,
            max_instances=1,
            num_workers=None,
            max_batch_rows=None,
            gpu=gpu_requests,
            force_rebuild=True,
            external_access_integration=sql_identifier.SqlIdentifier(self._SPCS_EAI),
        )

        with (mv._service_ops.workspace_path / deploy_spec_file_rel_path).open("r", encoding="utf-8") as f:
            deploy_spec_dict = yaml.safe_load(f)

        deploy_spec_dict["image_build"]["builder_image"] = self.BUILDER_IMAGE_PATH
        deploy_spec_dict["image_build"]["base_image"] = image_path

        with (mv._service_ops.workspace_path / deploy_spec_file_rel_path).open("w", encoding="utf-8") as f:
            yaml.dump(deploy_spec_dict, f)

        file_utils.upload_directory_to_stage(
            self.session,
            local_path=mv._service_ops.workspace_path,
            stage_path=pathlib.PurePosixPath(stage_path),
        )

        # deploy the model service
        mv._service_ops._service_client.deploy_model(
            stage_path=stage_path, model_deployment_spec_file_rel_path=deploy_spec_file_rel_path
        )

    def _test_registry_model_deployment(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: Dict[str, Tuple[Any, Callable[[Any], Any]]],
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        gpu_requests: Optional[str] = None,
    ) -> None:
        if self.BUILDER_IMAGE_PATH and self.BASE_CPU_IMAGE_PATH and self.BASE_GPU_IMAGE_PATH:
            with_image_override = True
        elif not self.BUILDER_IMAGE_PATH and not self.BASE_CPU_IMAGE_PATH and not self.BASE_GPU_IMAGE_PATH:
            with_image_override = False
        else:
            raise ValueError(
                "Please set or unset BUILDER_IMAGE_PATH, BASE_CPU_IMAGE_PATH, and BASE_GPU_IMAGE_PATH at the same time."
            )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(
                self.session, "snowflake-snowpark-python!=1.12.0, <1.21.1"
            )
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            options=options,
        )

        service = f"service_{inspect.stack()[1].function}_{self._run_id}"
        if with_image_override:
            self._deploy_model_with_image_override(
                mv,
                service_name=service,
                gpu_requests=gpu_requests,
            )
        else:
            mv.create_service(
                service_name=service,
                image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                service_compute_pool=(
                    self._TEST_CPU_COMPUTE_POOL if gpu_requests is None else self._TEST_GPU_COMPUTE_POOL
                ),
                image_repo=self._test_image_repo,
                gpu_requests=gpu_requests,
                force_rebuild=True,
                build_external_access_integration=self._SPCS_EAI,
            )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, function_name=target_method, service_name=service)
            check_func(res)
