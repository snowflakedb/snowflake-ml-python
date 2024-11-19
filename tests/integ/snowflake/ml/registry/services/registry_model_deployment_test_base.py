import http
import inspect
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pytest
import requests
import retrying
import yaml
from absl.testing import absltest
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from packaging import version

from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import (
    identifier,
    jwt_generator,
    snowflake_env,
    sql_identifier,
)
from snowflake.ml.model import ModelVersion, model_signature, type_hints as model_types
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.registry import registry
from snowflake.ml.utils import authentication
from snowflake.snowpark import row
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
    _TEST_SPCS_WH = "REGTEST_ML_SMALL"

    BUILDER_IMAGE_PATH = os.getenv("BUILDER_IMAGE_PATH", None)
    BASE_CPU_IMAGE_PATH = os.getenv("BASE_CPU_IMAGE_PATH", None)
    BASE_GPU_IMAGE_PATH = os.getenv("BASE_GPU_IMAGE_PATH", None)

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

        with open(self.session._conn._lower_case_parameters["private_key_path"], "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=backends.default_backend()
            )

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
        service_compute_pool: str,
        gpu_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = True,
    ) -> None:
        """Deploy model with image override."""
        is_gpu = gpu_requests is not None
        image_path = self.BASE_GPU_IMAGE_PATH if is_gpu else self.BASE_CPU_IMAGE_PATH
        build_compute_pool = sql_identifier.SqlIdentifier(self._TEST_CPU_COMPUTE_POOL)

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
            service_compute_pool_name=sql_identifier.SqlIdentifier(service_compute_pool),
            image_repo_database_name=database_name,
            image_repo_schema_name=schema_name,
            image_repo_name=image_repo_name,
            ingress_enabled=True,
            max_instances=max_instances,
            num_workers=num_workers,
            max_batch_rows=max_batch_rows,
            cpu=None,
            memory=None,
            gpu=gpu_requests,
            force_rebuild=force_rebuild,
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
        query_id, async_job = mv._service_ops._service_client.deploy_model(
            stage_path=stage_path, model_deployment_spec_file_rel_path=deploy_spec_file_rel_path
        )

        # TODO(hayu): Remove the version check after Snowflake 8.37.0 release
        if snowflake_env.get_current_snowflake_version(self.session) >= version.parse("8.37.0"):
            # stream service logs in a thread
            model_build_service_name = sql_identifier.SqlIdentifier(
                mv._service_ops._get_model_build_service_name(query_id)
            )
            model_build_service = service_ops.ServiceLogInfo(
                database_name=database_name,
                schema_name=schema_name,
                service_name=model_build_service_name,
                container_name="model-build",
            )
            model_inference_service = service_ops.ServiceLogInfo(
                database_name=database_name,
                schema_name=schema_name,
                service_name=sql_identifier.SqlIdentifier(service_name),
                container_name="model-inference",
            )
            services = [model_build_service, model_inference_service]
            log_thread = mv._service_ops._start_service_log_streaming(async_job, services, False, True)
            log_thread.join()
        else:
            while not async_job.is_done():
                time.sleep(5)

        res = cast(str, cast(List[row.Row], async_job.result())[0][0])
        logging.info(f"Inference service {service_name} deployment complete: {res}")

    def _test_registry_model_deployment(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: Dict[str, Tuple[Any, Callable[[Any], Any]]],
        service_name: Optional[str] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        gpu_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_instances: int = 1,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = True,
    ) -> ModelVersion:
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

        if service_name is None:
            service_name = f"service_{inspect.stack()[1].function}_{self._run_id}"
        if service_compute_pool is None:
            service_compute_pool = self._TEST_CPU_COMPUTE_POOL if gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

        if with_image_override:
            self._deploy_model_with_image_override(
                mv,
                service_name=service_name,
                service_compute_pool=sql_identifier.SqlIdentifier(service_compute_pool),
                gpu_requests=gpu_requests,
                num_workers=num_workers,
                max_instances=max_instances,
                max_batch_rows=max_batch_rows,
                force_rebuild=force_rebuild,
            )
        else:
            mv.create_service(
                service_name=service_name,
                image_build_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                service_compute_pool=service_compute_pool,
                image_repo=self._test_image_repo,
                gpu_requests=gpu_requests,
                force_rebuild=force_rebuild,
                num_workers=num_workers,
                max_instances=max_instances,
                max_batch_rows=max_batch_rows,
                ingress_enabled=True,
            )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, function_name=target_method, service_name=service_name)
            check_func(res)

        endpoint = RegistryModelDeploymentTestBase._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res_df = self._inference_using_rest_api(
                test_input, endpoint=endpoint, jwt_token_generator=jwt_token_generator, target_method=target_method
            )
            check_func(res_df)

        return mv

    @staticmethod
    def retry_if_result_status_retriable(result: requests.Response) -> bool:
        if result.status_code in [
            http.HTTPStatus.SERVICE_UNAVAILABLE,
            http.HTTPStatus.TOO_MANY_REQUESTS,
            http.HTTPStatus.GATEWAY_TIMEOUT,
        ]:
            return True
        return False

    @staticmethod
    def _ensure_ingress_url(mv: ModelVersion) -> str:
        while True:
            endpoint = mv.list_services().loc[0, "inference_endpoint"]
            if endpoint is not None:
                break
            time.sleep(10)
        return endpoint

    def _get_jwt_token_generator(self) -> jwt_generator.JWTGenerator:
        account = identifier.get_unescaped_names(self.session.get_current_account())
        user = identifier.get_unescaped_names(self.session.get_current_user())
        if not account or not user:
            raise ValueError("Account and user must be set.")

        return authentication.get_jwt_token_generator(
            account,
            user,
            self.private_key,
        )

    def _inference_using_rest_api(
        self,
        test_input: pd.DataFrame,
        *,
        endpoint: str,
        jwt_token_generator: jwt_generator.JWTGenerator,
        target_method: str,
    ) -> pd.DataFrame:
        test_input_arr = model_signature._convert_local_data_to_df(test_input).values
        test_input_arr = np.column_stack([range(test_input_arr.shape[0]), test_input_arr])
        res = retrying.retry(
            wait_exponential_multiplier=100,
            wait_exponential_max=4000,
            retry_on_result=RegistryModelDeploymentTestBase.retry_if_result_status_retriable,
        )(requests.post)(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json={"data": test_input_arr.tolist()},
            auth=authentication.SnowflakeJWTTokenAuth(
                jwt_token_generator=jwt_token_generator,
                role=identifier.get_unescaped_names(self.session.get_current_role()),
                endpoint=endpoint,
            ),
        )
        res.raise_for_status()
        return pd.DataFrame([x[1] for x in res.json()["data"]])
