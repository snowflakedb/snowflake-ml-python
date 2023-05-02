#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#


from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class TestEnvUtils(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_validate_requirement_in_snowflake_conda_channel(self) -> None:
        res = env_utils.validate_requirements_in_snowflake_conda_channel(
            session=self._session, reqs=[requirements.Requirement("xgboost")]
        )
        self.assertNotEmpty(res)

        res = env_utils.validate_requirements_in_snowflake_conda_channel(
            session=self._session, reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")]
        )
        self.assertNotEmpty(res)

        self.assertIsNone(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=self._session, reqs=[requirements.Requirement("xgboost<1.3")]
            )
        )

        self.assertIsNone(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=self._session, reqs=[requirements.Requirement("python-package")]
            )
        )


if __name__ == "__main__":
    absltest.main()
