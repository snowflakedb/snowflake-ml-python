"""Common base class for registry SPCS integration tests."""

import logging
import os
import uuid
from typing import Any, Optional

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
    BASE_BATCH_CPU_IMAGE_PATH = os.getenv("BASE_BATCH_CPU_IMAGE_PATH", None)
    BASE_BATCH_GPU_IMAGE_PATH = os.getenv("BASE_BATCH_GPU_IMAGE_PATH", None)
    RAY_ORCHESTRATOR_PATH = os.getenv("RAY_ORCHESTRATOR_PATH", None)
    PROXY_IMAGE_PATH = os.getenv("PROXY_IMAGE_PATH", None)
    MODEL_LOGGER_PATH = os.getenv("MODEL_LOGGER_PATH", None)
    VLLM_IMAGE_PATH = os.getenv("VLLM_IMAGE_PATH", None)

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

    def list_job_services(
        self,
        *,
        name_like: Optional[str] = None,
        compute_pool: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List job services in the current account via ``SHOW SERVICES``.

        Filters the rows returned by ``SHOW SERVICES`` to only those where ``is_job`` is true,
        i.e. SPCS job services (as opposed to long-running services).

        Args:
            name_like: Optional SQL ``LIKE`` pattern to match against the service name
                (e.g. ``"MY_JOB_%"``). Forwarded to ``SHOW SERVICES LIKE ...``.
            compute_pool: Optional compute pool name to scope the listing to
                (``SHOW SERVICES IN COMPUTE POOL <pool>``).

        Returns:
            A list of dicts, one per matching job service, with the columns from ``SHOW SERVICES``
            (e.g. ``name``, ``database_name``, ``schema_name``, ``status``, ``compute_pool``,
            ``is_job``, ...).
        """
        query = "SHOW SERVICES"
        if name_like is not None:
            escaped = name_like.replace("'", "''")
            query += f" LIKE '{escaped}'"
        if compute_pool is not None:
            query += f" IN COMPUTE POOL {compute_pool}"

        rows = self.session.sql(query).collect()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            row_dict = row.as_dict()
            is_job = row_dict.get("is_job")
            if isinstance(is_job, str):
                is_job_bool = is_job.strip().lower() in ("true", "t", "yes", "y", "1")
            else:
                is_job_bool = bool(is_job)
            if is_job_bool:
                jobs.append(row_dict)
        return jobs

    def get_job_service_spec(
        self,
        job_name: str,
        *,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> str:
        """Return the YAML spec of a job service via ``DESCRIBE SERVICE``.

        Args:
            job_name: The job service name. May be unqualified or fully qualified.
            database: Optional database to qualify the job with. Defaults to ``self._test_db``
                when neither ``database`` nor a qualified ``job_name`` is provided.
            schema: Optional schema to qualify the job with. Defaults to ``self._test_schema``
                when ``database`` is supplied (or defaulted) and ``job_name`` is unqualified.

        Returns:
            The ``spec`` column of the ``DESCRIBE SERVICE`` row, which contains the service YAML.

        Raises:
            ValueError: If ``DESCRIBE SERVICE`` returns no rows for the given job.
        """
        if "." in job_name:
            fully_qualified = job_name
        else:
            db = database if database is not None else self._test_db
            sch = schema if schema is not None else self._test_schema
            fully_qualified = f"{db}.{sch}.{job_name}"

        rows = self.session.sql(f"DESCRIBE SERVICE {fully_qualified}").collect()
        if not rows:
            raise ValueError(f"DESCRIBE SERVICE {fully_qualified} returned no rows")
        return rows[0].as_dict()["spec"]
