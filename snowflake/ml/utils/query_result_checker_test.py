from typing import List

import query_result_checker
from absl.testing.absltest import TestCase, main

from snowflake.connector import DataError
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark.row import Row


class SnowflakeQueryResultCheckerTest(TestCase):
    def test_result_dimension_matcher(self) -> None:
        """Test result_dimension_matcher()."""
        row1 = Row(name1=1, name2=2)
        row2 = Row(name1=3, name2=4)
        self.assertTrue(query_result_checker.result_dimension_matcher(2, 2, [row1, row2]))
        self.assertTrue(query_result_checker.result_dimension_matcher(None, 2, [row1, row2]))
        self.assertTrue(query_result_checker.result_dimension_matcher(2, None, [row1, row2]))
        self.assertRaises(DataError, query_result_checker.result_dimension_matcher, 2, 1, [row1, row2])
        self.assertRaises(DataError, query_result_checker.result_dimension_matcher, 1, 2, [row1, row2])

    def test_column_name_matcher(self) -> None:
        """Test column_name_matcher()."""
        row1 = Row(name1=1, name2=2)
        row2 = Row(name1=3, name2=4)
        self.assertTrue(query_result_checker.column_name_matcher("name1", [row1, row2]))
        self.assertRaises(DataError, query_result_checker.column_name_matcher, "name1", [])
        self.assertRaises(DataError, query_result_checker.column_name_matcher, "name3", [row1, row2])

    def test_cell_value_partial_matcher(self) -> None:
        """Test cell_value_partial_matcher()."""
        row1 = Row(name1=1, name2="foo")
        row2 = Row(name1=2, name2="bar")
        self.assertTrue(query_result_checker.cell_value_partial_matcher(0, 0, 1, [row1, row2]))
        self.assertRaises(DataError, query_result_checker.cell_value_partial_matcher, 0, 0, 2, [row1, row2])
        self.assertTrue(query_result_checker.cell_value_partial_matcher(0, 1, "foo", [row1, row2]))
        self.assertRaises(DataError, query_result_checker.cell_value_partial_matcher, 1, 1, "foo", [row1, row2])

    def test_result_validator_dimensions_partial_ok(self) -> None:
        """Use the base ResultValidator to verify the dimensions and value match of an operation result."""
        expected_result = [Row("number of rows updated=1, number of multi-joined rows updated=0")]
        actual_result = (
            query_result_checker.ResultValidator(result=expected_result)
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_value_match(row_idx=0, col_idx=0, expected_value="number of rows updated=1")
            .validate()
        )
        self.assertEqual(expected_result, actual_result)

    def test_sql_result_validator_dimensions_partial_ok(self) -> None:
        """Use SqlResultValidator to check dimension and value match of the result."""
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "UPDATE TABLE SET COL = 'value'"
        sql_result = [Row("number of rows updated=1, number of multi-joined rows updated=0")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_value_match(row_idx=0, col_idx=0, expected_value="number of rows updated=1")
            .validate()
        )
        self.assertEqual(sql_result, actual_result)

    def test_sql_result_validator_dimensions_rows_cols_separately_ok(self) -> None:
        """Use SqlResultValidator to check dimensions (rows and cols) separately."""
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "UPDATE TABLE SET COL = 'value'"
        sql_result = [Row("number of rows updated=1, number of multi-joined rows updated=0")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .has_dimensions(expected_rows=1)
            .has_dimensions(expected_cols=1)
            .validate()
        )
        self.assertEqual(sql_result, actual_result)

    def test_sql_result_validator_dimensions_ok_partial_fail(self) -> None:
        """Use SqlResultValidator to check dimension and value match of the result. We expect the dimensions check to
        succeed and the value match to fail."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "UPDATE TABLE SET COL = 'value'"
        sql_result = [Row("number of rows updated=0, number of multi-joined rows updated=0")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .has_dimensions(expected_rows=1, expected_cols=1)
                .has_value_match(row_idx=0, col_idx=0, expected_value="number of rows updated=1")
                .validate()
            )
            self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_dimensions_fail(self) -> None:
        """Use SqlResultValidator to find a mismatch between the expected and actual result dimensions."""
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "CREATE TABLE TEMP"
        sql_result: List[Row] = []
        expected_result = mock_data_frame.MockDataFrame(collect_result=sql_result)
        expected_result.add_operation(operation="collect", result=sql_result)
        session.add_mock_sql(query=query, result=expected_result)
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .has_dimensions(expected_rows=1, expected_cols=1)
                .validate()
            )
            self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_column_ok(self) -> None:
        """Use SqlResultValidator to check that a specific column exists in the result."""
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "CREATE TABLE TEMP"
        sql_result = [Row(status="Table TEMP successfully created.")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .has_column(expected_col_name="status")
            .validate()
        )
        self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_column_fail(self) -> None:
        """Use SqlResultValidator to check that a specific column exists in the result but we the column is missing."""
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "CREATE TABLE TEMP"
        sql_result = [Row(result="Table TEMP successfully created.")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .has_column(expected_col_name="status")
                .validate()
            )
            self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_insertion_success_ok(self) -> None:
        """Use SqlResultValidator to check that an INSERT INTO command had the desired result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "INSERT INTO TABLE ( KEY ) SELECT 'key'"
        sql_result = [Row(**{"number of rows inserted": 1})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .insertion_success(expected_num_rows=1)
            .validate()
        )
        self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_insertion_success_fail(self) -> None:
        """Use SqlResultValidator to check that an INSERT INTO command had the desired result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "INSERT INTO TABLE ( KEY ) SELECT 'key'"
        sql_result = [Row(**{"number of rows inserted": 0})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .insertion_success(expected_num_rows=1)
                .validate()
            )
            self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_named_value_match_ok(self) -> None:
        """Use SqlResultValidator to check the value in a named column in a given row of the result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "DELETE FROM my_table WHERE KEY='value'"
        sql_result = [Row(**{"number of rows deleted": 1})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .has_named_value_match(row_idx=0, col_name="number of rows deleted", expected_value=1)
            .validate()
        )
        self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_named_value_match_fail(self) -> None:
        """Use SqlResultValidator to check the value in a named column in a given row of the result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "DELETE FROM my_table WHERE KEY='value'"
        sql_result = [Row(**{"number of rows deleted": 0})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .has_named_value_match(row_idx=0, col_name="number of rows deleted", expected_value=1)
                .validate()
            )
            self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_deletion_success_ok(self) -> None:
        """Use SqlResultValidator to check that an DELETE FROM command had the desired result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "DELETE FROM my_table WHERE KEY='value'"
        sql_result = [Row(**{"number of rows deleted": 1})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = (
            query_result_checker.SqlResultValidator(session=session, query=query)
            .deletion_success(expected_num_rows=1)
            .validate()
        )
        self.assertEqual(actual_result, sql_result)

    def test_sql_result_validator_deletion_success_fail(self) -> None:
        """Use SqlResultValidator to check that an DELETE FROM command had the desired result."""

        session = mock_session.MockSession(conn=None, test_case=self)
        query = "DELETE FROM my_table WHERE KEY='value'"
        sql_result = [Row(**{"number of rows deleted": 0})]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        with self.assertRaises(DataError):
            actual_result = (
                query_result_checker.SqlResultValidator(session=session, query=query)
                .deletion_success(expected_num_rows=1)
                .validate()
            )
            self.assertEqual(actual_result, sql_result)


if __name__ == "__main__":
    main()
