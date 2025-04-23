from typing import cast

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.utils import table_manager
from snowflake.ml.test_utils import mock_data_frame, mock_session


class TableManagerTest(absltest.TestCase):
    """Testing table manager util functions."""

    def setUp(self) -> None:
        """Creates Snowpark environments for testing."""
        self._session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    def test_get_fully_qualified_schema_name(self) -> None:
        test_cases = [
            ("testdb", "testschema", "testdb.testschema"),
            ('"testdb"', '"testschema"', '"testdb"."testschema"'),
        ]
        for database_name, schema_name, expected_res in test_cases:
            with self.subTest():
                self.assertEqual(
                    table_manager.get_fully_qualified_schema_name(database_name, schema_name),
                    expected_res,
                )

    def test_get_fully_qualified_table_name(self) -> None:
        test_cases = [
            ("testdb", "testschema", "table", "testdb.testschema.table"),
            ('"testdb"', '"testschema"', '"table"', '"testdb"."testschema"."table"'),
            ("testdb", "testschema", '"table"', 'testdb.testschema."table"'),
        ]
        for database_name, schema_name, table_name, expected_res in test_cases:
            with self.subTest():
                self.assertEqual(
                    table_manager.get_fully_qualified_table_name(database_name, schema_name, table_name),
                    expected_res,
                )

    def test_create_single_table(self) -> None:
        schema_list = [("ID", "VARCHAR"), ("TYPE", "VARCHAR")]
        database_name = "testdb"
        schema_name = "testschema"
        table_name = "testtable"
        self._session.add_mock_sql(
            query=f"CREATE TABLE IF NOT EXISTS {database_name}.{schema_name}.{table_name} (ID VARCHAR, TYPE VARCHAR)",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"Table {table_name} successfully created.")],
            ),
        )
        table_manager.create_single_table(
            cast(snowpark.Session, self._session),
            database_name,
            schema_name,
            table_name,
            schema_list,
            {},
        )

    def test_insert_table_entry(self) -> None:
        table_name = "testtable"
        insert_query = f"INSERT INTO {table_name} ( ID,TYPE ) SELECT 1,'a' "
        self._session.add_mock_sql(
            query=insert_query,
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )
        table_manager.insert_table_entry(cast(snowpark.Session, self._session), table_name, {"ID": 1, "TYPE": "a"})

    def test_validate_table_exist(self) -> None:
        table_name = "testtable"
        schema_name = "testschema"
        empty_row_list: list[snowpark.Row] = []
        test_cases = [
            (empty_row_list, False),
            ([snowpark.Row(**{"number of rows inserted": 1})], True),
        ]
        for snowpark_res, expected_res in test_cases:
            with self.subTest():
                self._session.add_mock_sql(
                    query=f"SHOW TABLES LIKE '{table_name}' IN {schema_name}",
                    result=mock_data_frame.MockDataFrame(snowpark_res),
                )
                self.assertEqual(
                    table_manager.validate_table_exist(cast(snowpark.Session, self._session), table_name, schema_name),
                    expected_res,
                )


if __name__ == "__main__":
    absltest.main()
