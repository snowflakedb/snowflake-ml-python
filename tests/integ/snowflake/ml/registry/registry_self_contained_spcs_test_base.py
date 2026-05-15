"""Test base that builds, pushes, and deploys SPCS images from Bazel ``data`` deps.

Extends :class:`RegistrySPCSTestBase` so that each test class pushes its
images once in ``setUpClass`` (expensive) and each test method only applies
session overrides (cheap).

Each test class creates its own non-temporary database (owned by the
executor role) containing a fixed-name image repo (``IMAGE_REPO``).
Because the executor owns the database and all objects inside it, Docker
push succeeds even though it authenticates via a separate connection.

The database is dropped in ``tearDownClass``; stale databases are
garbage-collected by ``db_manager.cleanup_databases``.

Subclasses declare which images they need via :attr:`IMAGE_TARGETS`::

    class MyTest(RegistrySelfContainedSPCSTestBase):
        IMAGE_TARGETS = {
            image_operations.BUILDER_SESSION_PARAM:
                "model_container_services_deployment/kaniko/image.image_ref",
            image_operations.BASE_CPU_INFERENCE_SESSION_PARAM:
                "model_container_services_deployment/inference_server/image_cpu.image_ref",
            image_operations.PROXY_SESSION_PARAM:
                "model_container_services_deployment/proxy/image_amd64.image_ref",
        }
"""

from __future__ import annotations

import logging
import os
import pathlib
import uuid
from typing import Optional

import pytest
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from model_container_services_deployment.ci import image_operations
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]

_IMAGE_REPO_SCHEMA = "PUBLIC"
_IMAGE_REPO_NAME = "IMAGE_REPO"


@pytest.mark.spcs_deployment_image
class RegistrySelfContainedSPCSTestBase(registry_spcs_test_base.RegistrySPCSTestBase):
    """SPCS test base that pushes locally-built images from Bazel data deps.

    The heavy lifting (registry login, tag, push) happens once per class in
    ``setUpClass``.  Each test method only applies ``ALTER SESSION SET``
    overrides so the Snowflake backend picks up the freshly-pushed images.

    A test-specific database is created and owned by the executor role.
    A fixed-name image repo (``IMAGE_REPO``) is created inside it so that
    Docker push (which authenticates via a separate connection) can access
    it through role ownership.
    """

    IMAGE_TARGETS: dict[str, str] = {}

    _cls_connection_name: Optional[str] = None
    _cls_run_id: str = ""
    _cls_test_db: str = ""
    _cls_test_schema: str = _IMAGE_REPO_SCHEMA
    _cls_image_tag: str = ""
    _cls_image_overrides: dict[str, str] = {}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls._cls_connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME", "")
        cls._cls_run_id = uuid.uuid4().hex[:4]

        cls._cls_test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(cls._cls_run_id, "db").upper()
        cls._cls_test_schema = _IMAGE_REPO_SCHEMA

        session = test_env_utils.get_available_session()
        try:
            warehouse = os.getenv("WAREHOUSE", "REGTEST_ML_SMALL")
            if not session.get_current_warehouse():
                session.sql(f"USE WAREHOUSE {warehouse}").collect()

            cls_db_manager = db_manager.DBManager(session)
            cls_db_manager.create_database(cls._cls_test_db)
            cls_db_manager.create_stage("TEST_STAGE")
            cls_db_manager.create_image_repo(_IMAGE_REPO_NAME)
            cls_db_manager.cleanup_databases(expire_hours=6)

            if not cls.IMAGE_TARGETS:
                return

            runfiles_dir = pathlib.Path(os.environ["TEST_SRCDIR"]) / "_main"
            image_refs: dict[str, str] = {}
            for session_param, ref_runfile_path in cls.IMAGE_TARGETS.items():
                resolved = runfiles_dir / ref_runfile_path
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Image ref file not found at {resolved}. "
                        "Ensure the docker_image target is listed in the test's 'data' deps."
                    )
                local_ref = image_operations.read_image_ref(resolved)
                image_refs[session_param] = local_ref
                logger.info("Image %s -> local ref %s", session_param, local_ref)

            registry_domain = image_operations.resolve_registry_domain_via_session(session, _IMAGE_REPO_NAME)
            repo_path = f"{cls._cls_test_db}/{_IMAGE_REPO_SCHEMA}/{_IMAGE_REPO_NAME}"

            cls._cls_image_tag = image_operations.generate_tag(_REPO_ROOT)

            login_options = connection_params.SnowflakeLoginOptions()
            login_conn = image_operations.spcs_registry_login_via_session(login_options, registry_domain)

            try:
                repo_prefix = f"{registry_domain}/{repo_path}".lower()
                overrides: dict[str, str] = {}
                for session_param, local_ref in image_refs.items():
                    image_name = local_ref.split(":")[0]
                    remote_ref = f"{repo_prefix}/{image_name}:{cls._cls_image_tag}"
                    image_operations.tag_and_push(local_ref, remote_ref)
                    overrides[session_param] = f"/{repo_path}/{image_name}:{cls._cls_image_tag}"

                cls._cls_image_overrides = overrides
                logger.info("Pushed %d images, overrides: %s", len(overrides), overrides)
            finally:
                login_conn.close()
        finally:
            session.close()

    def setUp(self) -> None:
        login_options = connection_params.SnowflakeLoginOptions()
        pat_token = login_options.get("password")
        common_test_base.CommonTestBase.setUp(self)

        logging.basicConfig(level=logging.INFO)
        self._setup_auth(pat_token)

        self._run_id = self._cls_run_id
        self._test_db = self._cls_test_db
        self._test_schema = self._cls_test_schema
        self._test_image_repo = _IMAGE_REPO_NAME
        self._test_stage = "TEST_STAGE"

        if not self.session.get_current_warehouse():
            self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        self.session.sql(f"USE DATABASE {self._test_db}").collect()
        self.session.sql(f"USE SCHEMA {self._test_schema}").collect()
        self.registry = registry.Registry(self.session)

        if self._cls_image_overrides:
            image_operations.apply_session_overrides(self.session, self._cls_image_overrides)

    def _setup_auth(self, pat_token: Optional[str]) -> None:
        """Configure authentication credentials from the session connection.

        Args:
            pat_token: PAT token extracted before session creation.

        Raises:
            ValueError: If no authentication credentials are available.
        """
        conn_params = self.session._conn._lower_case_parameters
        private_key_path = conn_params.get("private_key_path")

        if private_key_path:
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=backends.default_backend()
                )
            self.pat_token = None
        elif pat_token:
            self.private_key = None
            self.pat_token = pat_token
        else:
            self.private_key = None
            self.pat_token = None
            raise ValueError("No authentication credentials found: neither private_key_path nor password parameter set")

        self.snowflake_account_url = conn_params.get("host", None)
        if self.snowflake_account_url:
            self.snowflake_account_url = f"https://{self.snowflake_account_url}"

    def tearDown(self) -> None:
        common_test_base.CommonTestBase.tearDown(self)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._cls_test_db:
            try:
                session = test_env_utils.get_available_session()
                try:
                    cls_db_manager = db_manager.DBManager(session)
                    cls_db_manager.drop_database(cls._cls_test_db)
                finally:
                    session.close()
            except Exception:
                logger.warning(
                    "Failed to drop test database %s; lazy GC will clean it up.",
                    cls._cls_test_db,
                    exc_info=True,
                )
        super().tearDownClass()
