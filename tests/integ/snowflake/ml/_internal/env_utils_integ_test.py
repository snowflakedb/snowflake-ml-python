from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class TestEnvUtils(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_validate_requirement_in_snowflake_conda_channel(self) -> None:
        res = env_utils.get_matched_package_versions_in_information_schema(
            session=self._session, reqs=[requirements.Requirement("xgboost")], python_version=snowml_env.PYTHON_VERSION
        )
        self.assertNotEmpty(res["xgboost"])

        res = env_utils.get_matched_package_versions_in_information_schema(
            session=self._session,
            reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
            python_version=snowml_env.PYTHON_VERSION,
        )
        self.assertNotEmpty(res["xgboost"])
        self.assertNotEmpty(res["pytorch"])

        res = env_utils.get_matched_package_versions_in_information_schema(
            session=self._session,
            reqs=[requirements.Requirement("xgboost==1.0.*")],
            python_version=snowml_env.PYTHON_VERSION,
        )
        self.assertEmpty(res["xgboost"])

        res = env_utils.get_matched_package_versions_in_information_schema(
            session=self._session,
            reqs=[requirements.Requirement("python-package")],
            python_version=snowml_env.PYTHON_VERSION,
        )
        self.assertNotIn("python-package", res)


if __name__ == "__main__":
    absltest.main()
