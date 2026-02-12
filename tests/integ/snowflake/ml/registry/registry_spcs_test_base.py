"""Common base class for registry SPCS integration tests."""

import logging
import os
import pathlib
import tempfile
import uuid
from typing import Any, Optional

import pytest
import yaml
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake.ml._internal import file_utils, platform_capabilities as pc
from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model import ModelVersion
from snowflake.ml.model._client.service import model_deployment_spec
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


@pytest.mark.spcs_deployment_image
class RegistrySPCSTestBase(common_test_base.CommonTestBase):
    """Base class for registry SPCS integration tests with common setup logic."""

    # bazel test --test_env CPU_COMPUTE_POOL=<CPU_POOL> --test_env GPU_COMPUTE_POOL=<GPU_POOL>
    # --test_env WAREHOUSE=<WAREHOUSE> //...
    _TEST_CPU_COMPUTE_POOL = os.getenv("CPU_COMPUTE_POOL", "REGTEST_INFERENCE_CPU_POOL")
    _TEST_GPU_COMPUTE_POOL = os.getenv("GPU_COMPUTE_POOL", "REGTEST_INFERENCE_GPU_POOL")
    _TEST_SPCS_WH = os.getenv("WAREHOUSE", "REGTEST_ML_SMALL")

    BUILDER_IMAGE_PATH = os.getenv("BUILDER_IMAGE_PATH", None)
    BASE_CPU_IMAGE_PATH = os.getenv("BASE_CPU_IMAGE_PATH", None)
    BASE_GPU_IMAGE_PATH = os.getenv("BASE_GPU_IMAGE_PATH", None)
    BASE_BATCH_CPU_IMAGE_PATH = os.getenv("BASE_BATCH_CPU_IMAGE_PATH", None)
    BASE_BATCH_GPU_IMAGE_PATH = os.getenv("BASE_BATCH_GPU_IMAGE_PATH", None)
    RAY_ORCHESTRATOR_PATH = os.getenv("RAY_ORCHESTRATOR_PATH", None)
    PROXY_IMAGE_PATH = os.getenv("PROXY_IMAGE_PATH", None)
    MODEL_LOGGER_PATH = os.getenv("MODEL_LOGGER_PATH", None)
    VLLM_IMAGE_PATH = os.getenv("VLLM_IMAGE_PATH", None)
    INFERENCE_IMAGE_BUILDER_PATH = os.getenv("INFERENCE_IMAGE_BUILDER_PATH", None)

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        # Get login options BEFORE session creation (which clears password for security)
        login_options = connection_params.SnowflakeLoginOptions()

        # Capture password from login options before session creation clears it
        pat_token = login_options.get("password")

        # Now create session (this will clear password in session._conn._lower_case_parameters)
        super().setUp()

        # Set log level to INFO so that service logs are visible
        logging.basicConfig(level=logging.INFO)

        # Read private_key_path from session connection parameters (after session creation)
        conn_params = self.session._conn._lower_case_parameters
        private_key_path = conn_params.get("private_key_path")

        if private_key_path:
            # Try to load private key for JWT authentication
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=backends.default_backend()
                )
            self.pat_token = None
        elif pat_token:
            # Use PAT token from password parameter
            self.private_key = None
            self.pat_token = pat_token
        else:
            # No authentication credentials available
            self.private_key = None
            self.pat_token = None
            raise ValueError("No authentication credentials found: neither private_key_path nor password parameter set")

        self.snowflake_account_url = self.session._conn._lower_case_parameters.get("host", None)
        if self.snowflake_account_url:
            self.snowflake_account_url = f"https://{self.snowflake_account_url}"

        self._run_id = uuid.uuid4().hex[:4]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = "PUBLIC"
        self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "image_repo"
        ).upper()
        self._test_stage = "TEST_STAGE"

        if not self.session.get_current_warehouse():
            self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._create_stage()
        self._db_manager.create_image_repo(self._test_image_repo)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def _create_stage(self) -> None:
        """Create stage with appropriate settings. Can be overridden by subclasses."""
        self._db_manager.create_stage(self._test_stage)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _add_common_model_deployment_spec_options(
        self,
        *,
        mv: ModelVersion,
        database: sql_identifier.SqlIdentifier,
        schema: sql_identifier.SqlIdentifier,
        force_rebuild=False,
    ) -> None:
        image_repo_name = sql_identifier.SqlIdentifier(self._test_image_repo)

        # add model spec
        mv._service_ops._model_deployment_spec.add_model_spec(
            database_name=mv._model_ops._model_version_client._database_name,
            schema_name=mv._model_ops._model_version_client._schema_name,
            model_name=mv._model_name,
            version_name=mv._version_name,
        )

        # Add image build spec
        image_repo_fqn = identifier.get_schema_level_object_identifier(
            database.identifier(), schema.identifier(), image_repo_name.identifier()
        )
        build_compute_pool = sql_identifier.SqlIdentifier(self._TEST_CPU_COMPUTE_POOL)
        mv._service_ops._model_deployment_spec.add_image_build_spec(
            image_build_compute_pool_name=build_compute_pool,
            fully_qualified_image_repo_name=image_repo_fqn,
            force_rebuild=force_rebuild,
            external_access_integrations=None,
        )
        return

    def _get_fully_qualified_service_or_job_name(
        self, service_name: str
    ) -> tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier,]:
        database_name_id, schema_name_id, name_id = sql_identifier.parse_fully_qualified_name(service_name)
        database_name_id = database_name_id or sql_identifier.SqlIdentifier(self._test_db)
        schema_name_id = schema_name_id or sql_identifier.SqlIdentifier(self._test_schema)
        assert database_name_id is not None, "Database name should not be None"
        assert schema_name_id is not None, "Schema name should not be None"

        return database_name_id, schema_name_id, name_id

    def _deploy_override_model(
        self,
        *,
        mv: ModelVersion,
        database: sql_identifier.SqlIdentifier,
        schema: sql_identifier.SqlIdentifier,
        inference_image: str,
        is_batch_inference: bool,
        override_vllm_image: bool = False,
        builder_image_path: Optional[str] = None,
    ) -> tuple[str, Any]:

        deploy_spec = mv._service_ops._model_deployment_spec.save()

        if mv._service_ops._model_deployment_spec.workspace_path:
            with pathlib.Path(deploy_spec).open("r", encoding="utf-8") as f:
                deploy_spec_dict = yaml.safe_load(f)
        else:
            deploy_spec_dict = yaml.safe_load(deploy_spec)

        # Use provided builder_image_path or fall back to BUILDER_IMAGE_PATH (kaniko)
        effective_builder_image = builder_image_path or self.BUILDER_IMAGE_PATH
        deploy_spec_dict["image_build"]["builder_image"] = effective_builder_image
        deploy_spec_dict["image_build"]["base_image"] = inference_image
        if not is_batch_inference:
            deploy_spec_dict["service"]["proxy_image"] = self.PROXY_IMAGE_PATH
        if not is_batch_inference and override_vllm_image:
            deploy_spec_dict["service"]["inference_engine_spec"]["inference_engine_image"] = self.VLLM_IMAGE_PATH

        inline_deploy_spec_enabled = pc.PlatformCapabilities.get_instance().is_inlined_deployment_spec_enabled()
        # Deploy the model
        if inline_deploy_spec_enabled:
            # Use inline deployment spec
            deploy_spec_yaml_str = yaml.dump(deploy_spec_dict)
            query_id, async_job = mv._service_ops._service_client.deploy_model(
                model_deployment_spec_yaml_str=deploy_spec_yaml_str,
            )
        else:
            stage = sql_identifier.SqlIdentifier(
                snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
            )
            # Use file-based deployment spec
            temp_dir = tempfile.TemporaryDirectory()
            workspace_path = pathlib.Path(temp_dir.name)
            deploy_spec_file_rel_path = model_deployment_spec.ModelDeploymentSpec.DEPLOY_SPEC_FILE_REL_PATH
            stage_path = mv._service_ops._stage_client.create_tmp_stage(
                database_name=database,
                schema_name=schema,
                stage_name=stage,
            )
            with (workspace_path / deploy_spec_file_rel_path).open("w", encoding="utf-8") as f:
                yaml.dump(deploy_spec_dict, f)
            file_utils.upload_directory_to_stage(
                self.session,
                local_path=workspace_path,
                stage_path=pathlib.PurePosixPath(stage_path),
            )
            query_id, async_job = mv._service_ops._service_client.deploy_model(
                stage_path=stage_path,
                model_deployment_spec_file_rel_path=deploy_spec_file_rel_path,
            )

        return query_id, async_job
