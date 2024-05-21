from unittest import mock

from absl.testing import absltest

from snowflake.ml.modeling._internal.snowpark_implementations import snowpark_trainer
from snowflake.snowpark import DataFrame, Session


class EnableAnonymousSPROC(absltest.TestCase):
    def test_disable_distributed_hpo(self) -> None:
        mock_session = mock.MagicMock(spec=Session)
        mock_dataframe = mock.MagicMock(spec=DataFrame)
        mock_dataframe._session = mock_session

        self.assertFalse(snowpark_trainer._ENABLE_ANONYMOUS_SPROC)

        # Disable distributed HPO
        import snowflake.ml.modeling.parameters.enable_anonymous_sproc  # noqa: F401

        self.assertTrue(snowpark_trainer._ENABLE_ANONYMOUS_SPROC)


if __name__ == "__main__":
    absltest.main()
