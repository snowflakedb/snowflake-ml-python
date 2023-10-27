import uuid
from unittest import SkipTest

from absl.testing import absltest

from snowflake.ml._internal.utils import identifier
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class SpcsIntegTestBase(absltest.TestCase):
    _SNOWSERVICE_CONNECTION_NAME = "regtest"
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _RUN_ID = uuid.uuid4().hex[:2]
    _TEST_DB = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "db").upper()
    _TEST_SCHEMA = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "schema").upper()
    _TEST_STAGE = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "stage").upper()
    _TEST_IMAGE_REPO = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "repo").upper()

    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        try:
            login_options = connection_params.SnowflakeLoginOptions(connection_name=cls._SNOWSERVICE_CONNECTION_NAME)
        except KeyError:
            raise SkipTest(
                "SnowService connection parameters not present: skipping "
                "TestModelRegistryIntegWithSnowServiceDeployment."
            )
        cls._session = Session.builder.configs(login_options).create()
        prev_session_db = identifier._get_unescaped_name(cls._session.get_current_database())
        prev_session_schema = identifier._get_unescaped_name(cls._session.get_current_schema())

        cls._db_manager = db_manager.DBManager(cls._session)
        cls._db_manager.create_database(cls._TEST_DB)
        cls._db_manager.create_schema(cls._TEST_SCHEMA)
        cls._db_manager.create_stage(cls._TEST_STAGE, cls._TEST_SCHEMA, cls._TEST_DB, sse_encrypted=True)
        cls._db_manager.cleanup_databases(expire_hours=6)

        # When creating a new database or schema within a session, the current session's database or schema will
        # automatically switch to the newly created one. To replicate a more realistic user experience, we ensure that
        # the original session's database or schema remains unchanged from the registry's database or schema.
        cls._db_manager.use_database(prev_session_db)
        cls._db_manager.use_schema(prev_session_schema)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_database(cls._TEST_DB)
        cls._session.close()
