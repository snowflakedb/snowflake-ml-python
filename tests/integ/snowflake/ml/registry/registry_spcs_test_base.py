"""Common base class for registry SPCS integration tests."""

import logging
import os
import uuid

import pytest
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
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
    PROXY_IMAGE_PATH = os.getenv("PROXY_IMAGE_PATH", None)
    MODEL_LOGGER_PATH = os.getenv("MODEL_LOGGER_PATH", None)

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
