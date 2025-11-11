from typing import Any, cast

from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal.utils import table_manager
from snowflake.ml.test_utils import mock_data_frame, mock_session


class TableManagerTest(parameterized.TestCase):
    """Testing table manager util functions."""

    def setUp(self) -> None:
        """Creates Snowpark environments for testing."""
        self._session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    @parameterized.parameters(  # type: ignore[misc]
        ("testdb", "testschema", "testdb.testschema"),
        ('"testdb"', '"testschema"', '"testdb"."testschema"'),
    )
    def test_get_fully_qualified_schema_name(self, database_name: str, schema_name: str, expected_res: str) -> None:
        self.assertEqual(
            table_manager.get_fully_qualified_schema_name(database_name, schema_name),
            expected_res,
        )

    @parameterized.parameters(  # type: ignore[misc]
        ("testdb", "testschema", "table", "testdb.testschema.table"),
        ('"testdb"', '"testschema"', '"table"', '"testdb"."testschema"."table"'),
        ("testdb", "testschema", '"table"', 'testdb.testschema."table"'),
    )
    def test_get_fully_qualified_table_name(
        self, database_name: str, schema_name: str, table_name: str, expected_res: str
    ) -> None:
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

    @parameterized.parameters(  # type: ignore[misc]
        ([], False),
        ([{"number of rows inserted": 1}], True),
    )
    def test_validate_table_exist(self, row_data: list[dict[str, Any]], expected_res: bool) -> None:
        table_name = "testtable"
        schema_name = "testschema"
        snowpark_res = [snowpark.Row(**data) for data in row_data]
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
