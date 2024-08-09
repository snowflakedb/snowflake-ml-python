from absl.testing import absltest

from snowflake.ml.utils import sql_client


class SqlClientTest(absltest.TestCase):
    def test_creation_mode_if_not_exists(self) -> None:
        creation_mode = sql_client.CreationMode(if_not_exists=True)
        self.assertEqual(
            creation_mode.get_ddl_phrases()[sql_client.CreationOption.CREATE_IF_NOT_EXIST], " IF NOT EXISTS"
        )

        creation_mode = sql_client.CreationMode(if_not_exists=False)
        self.assertEqual(creation_mode.get_ddl_phrases()[sql_client.CreationOption.CREATE_IF_NOT_EXIST], "")

    def test_creation_mode_or_replace(self) -> None:
        creation_mode = sql_client.CreationMode(or_replace=True)
        self.assertEqual(creation_mode.get_ddl_phrases()[sql_client.CreationOption.OR_REPLACE], " OR REPLACE")

        creation_mode = sql_client.CreationMode(or_replace=False)
        self.assertEqual(creation_mode.get_ddl_phrases()[sql_client.CreationOption.OR_REPLACE], "")


if __name__ == "__main__":
    absltest.main()
