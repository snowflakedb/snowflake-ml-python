import uuid
from unittest import SkipTest

from absl.testing import absltest

from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class SpcsIntegTestBase(absltest.TestCase):
    _SNOWSERVICE_CONNECTION_NAME = "regtest"
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        try:
            login_options = connection_params.SnowflakeLoginOptions(connection_name=self._SNOWSERVICE_CONNECTION_NAME)
        except KeyError:
            raise SkipTest(
                "SnowService connection parameters not present: skipping "
                "TestModelRegistryIntegWithSnowServiceDeployment."
            )

        _run_id = uuid.uuid4().hex[:2]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_run_id, "schema").upper()
        _test_stage = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_run_id, "stage").upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()
        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.create_stage(_test_stage, self._test_schema, self._test_db, sse_encrypted=True)
        self._db_manager.cleanup_databases(expire_hours=6)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()
