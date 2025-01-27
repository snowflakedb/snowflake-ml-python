from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import spec_utils
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils


class SpecUtilsIntegTests(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.db = cls.session.get_current_database()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.close()
        super().tearDownClass()

    def test_get_node_resources(self) -> None:
        with self.assertRaises(ValueError):
            spec_utils._get_node_resources(self.session, "dummy_pool_that_doesnt_exist")

        pools = self.session.sql("SHOW COMPUTE POOLS").collect()
        if not pools:
            self.skipTest("No compute pools available")
        rst = spec_utils._get_node_resources(self.session, pools[0]["name"])
        self.assertIsNotNone(rst)


if __name__ == "__main__":
    absltest.main()
