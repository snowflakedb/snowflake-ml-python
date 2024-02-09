import unittest
import uuid

from absl.testing import absltest

from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils


@unittest.skipUnless(
    test_env_utils.get_current_snowflake_cloud_type() == snowflake_env.SnowflakeCloudType.AWS,
    "SPCS only available in AWS",
)
class SpcsIntegTestBase(absltest.TestCase):
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _SPCS_EAIS = ["SPCS_EGRESS_ACCESS_INTEGRATION"]

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        login_options = connection_params.SnowflakeLoginOptions()

        self._run_id = uuid.uuid4().hex[:2]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()
        self._test_stage = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "stage").upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()
        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.create_stage(self._test_stage, self._test_schema, self._test_db, sse_encrypted=True)
        self._db_manager.cleanup_databases(expire_hours=6)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()
