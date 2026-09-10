"""Shared base class for AI SQL BYOM (Bring-Your-Own-Model) SPCS integration tests.

Handles the deploy-once-per-class pattern, sentinel-based failure propagation,
session parameter application, and shared teardown so that individual AI SQL
function test classes only need to declare their session parameters and implement
the deployment logic.
"""

import logging
import tempfile

import snowflake.snowpark.exceptions
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import db_manager

logger = logging.getLogger(__name__)

# Privilege-denial errors for AISQL against an SPCS service. Includes Cortex Model
# RBAC (behavior change bundle 2026_07, SQL 399217).
AISQL_BYOM_DENY_ERROR_RE = (
    r"Insufficient privileges|"
    r"does not exist or not authorized|"
    r"does not exist or is not authorized|"
    r"invalid argument|"
    r"unavailable|"
    r"unknown model|"
    r"Not authorized to access model"
)


class AISQLBYOMTestBase(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Base for AI SQL BYOM integration tests that deploy a shared SPCS service.

    Subclasses must:
      - Declare _SESSION_PARAMS (dict[str, str]) listing the ALTER SESSION parameters
        required by their AI SQL function (key=param name, value=SQL literal value).
      - Implement _do_deploy() to log the model and create the SPCS service.
        _do_deploy() must set type(self)._service_name to the fully-qualified service
        name once the service is ready.
      - Override setUpClass() to save and set model-cache env vars, calling
        super().setUpClass() first so the shared cache_dir is created.
      - Override tearDownClass() to restore env vars, calling super().tearDownClass()
        last so the DB and cache_dir are cleaned up after env vars are restored.
    """

    # Subclasses declare session parameters as a dict of {PARAM_NAME: sql_value}.
    # setUp applies them all via ALTER SESSION SET key=value before each test.
    _SESSION_PARAMS: dict[str, str] = {}

    # Class-level sentinel per subclass:
    #   None          — not deployed yet
    #   "DEPLOYING"   — deployment in progress (another setUp is running _do_deploy)
    #   "DEPLOY_FAILED" — deployment failed; subsequent tests will be skipped
    #   <fq_name>     — fully-qualified service name, service is ready
    _service_name: str | None = None
    _model_fq_name: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Create the shared model-cache temp directory.

        Subclasses must call super().setUpClass() first, then set their
        model-cache environment variables to cls.cache_dir.name.
        """
        cls.cache_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        """Drop the test database and clean up the shared cache directory.

        Subclasses must restore their environment variables before calling
        super().tearDownClass().
        """
        if hasattr(cls, "_db_manager") and hasattr(cls, "_test_db"):
            cls._db_manager.drop_database(cls._test_db)
        cls.cache_dir.cleanup()

    def setUp(self) -> None:
        """Apply session parameters, then deploy the service on the first call."""
        super().setUp()
        for key, value in self._SESSION_PARAMS.items():
            self.session.sql(f"ALTER SESSION SET {key}={value}").collect()
        if type(self)._service_name is None:
            self._deploy_test_service()
        if type(self)._service_name == "DEPLOY_FAILED":
            self.skipTest("Service deployment failed in a previous setUp — skipping.")

    def tearDown(self) -> None:
        # Per-test teardown is skipped — the shared service and DB are cleaned up
        # in tearDownClass after all test methods complete.
        pass

    def _aisql_byom_make_limited_role(self, suffix: str, *, service_fqn: str | None = None) -> str:
        """Create a minimal role for AISQL BYOM permission tests.

        Grants only the structural privileges needed to execute queries. Callers
        are responsible for adding object-level grants (USAGE ON SERVICE, SERVICE
        ROLE, USAGE ON MODEL, etc.) before switching into this role.

        When service_fqn is provided and lives in a different database than
        self._test_db (which happens when the service was deployed once on the
        first setUp but subsequent test methods get a fresh _run_id and _test_db),
        USAGE on that database and schema is also granted.

        Args:
            suffix: Short label appended to the auto-generated role name.
            service_fqn: Optional fully-qualified service name
                (e.g. "DB.SCHEMA.SERVICE"). When provided, grants cross-DB access
                if the service lives outside self._test_db.

        Returns:
            The fully-qualified role name that was created.
        """
        role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, suffix).upper()
        warehouse = self.session.get_current_warehouse()
        current_user = self.session.get_current_user().strip('"')

        self._db_manager.create_role(role)
        self.session.sql(f"GRANT ROLE {role} TO USER {current_user}").collect()
        self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {role}").collect()
        self.session.sql(f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {role}").collect()
        try:
            self.session.sql(f"GRANT USAGE ON WAREHOUSE {warehouse} TO ROLE {role}").collect()
        except snowflake.snowpark.exceptions.SnowparkSQLException:
            logger.warning("Could not grant warehouse %s to role %s.", warehouse, role)

        if service_fqn and service_fqn not in ("DEPLOYING", "DEPLOY_FAILED"):
            svc_parts = service_fqn.split(".")
            if len(svc_parts) >= 2:
                svc_db, svc_schema = svc_parts[0], svc_parts[1]
                if svc_db.upper() != self._test_db.upper():
                    self.session.sql(f"GRANT USAGE ON DATABASE {svc_db} TO ROLE {role}").collect()
                    self.session.sql(f"GRANT USAGE ON SCHEMA {svc_db}.{svc_schema} TO ROLE {role}").collect()

        # Log whether PUBLIC has CORTEX_USER so failures are easy to diagnose.
        # All roles (including ephemeral test roles) inherit CORTEX_USER via PUBLIC
        # when regtest_env.sql has been run. Without it, AI_COMPLETE with SPCS
        # service names returns "Unknown function".
        cortex_grants = self.session.sql("SHOW GRANTS OF DATABASE ROLE SNOWFLAKE.CORTEX_USER").collect()
        public_has_cortex = any(r.as_dict().get("grantee_name", "") == "PUBLIC" for r in cortex_grants)
        logger.info("PUBLIC has SNOWFLAKE.CORTEX_USER: %s", public_has_cortex)
        if not public_has_cortex:
            logger.warning(
                "PUBLIC does not have SNOWFLAKE.CORTEX_USER — AI_COMPLETE with SPCS "
                "service names will fail for role %s. Re-run regtest_env.sql to fix.",
                role,
            )

        return role

    def _deploy_test_service(self) -> None:
        """Sentinel wrapper: marks DEPLOYING, calls _do_deploy, marks DEPLOY_FAILED on error.

        This prevents multiple setUp calls from each attempting to deploy when the
        first deployment is still running or has already failed.

        Raises:
            Exception: Re-raises any exception from _do_deploy after marking DEPLOY_FAILED.
        """
        type(self)._service_name = "DEPLOYING"
        try:
            self._do_deploy()
        except Exception:
            type(self)._service_name = "DEPLOY_FAILED"
            raise

    def _do_deploy(self) -> None:
        """Deploy the model and SPCS service.

        Implementations must log the model, create the service, wait for it to
        be ready, and set type(self)._service_name to the fully-qualified service
        name before returning.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError
