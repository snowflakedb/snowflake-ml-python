#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

from typing import List

from absl.testing import parameterized
from absl.testing.absltest import main

from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.ml.utils.parallelize import map_dataframe_by_column
from snowflake.snowpark import DataFrame, Session


class TestParallelize(parameterized.TestCase):
    def setUp(self):
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    @parameterized.parameters(
        {"partition_size": 2},
        {"partition_size": 3},
        {"partition_size": 4},
        {"partition_size": 8},
    )
    def test_map_dataframe_by_column_by_partition_size(self, partition_size):
        schema = ["a", "b", "c", "d", "_exclude"]
        dataset = self._session.create_dataframe([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], schema=schema)

        def _negate(df: DataFrame, col_subset: List[str]) -> DataFrame:
            return df.select_expr([f"{col}*(-1)" for col in col_subset])

        results = map_dataframe_by_column(
            df=dataset,
            cols=["a", "b", "c", "d"],
            map_func=_negate,
            partition_size=partition_size,
        )

        assert results == [[0, -1, -2, -3], [-5, -6, -7, -8]]

    def test_map_dataframe_by_column_multiple_output_columns(self):
        schema = ["a", "b", "c", "d"]
        dataset = self._session.create_dataframe([[0, 1, 2, 3], [5, 6, 7, 8]], schema=schema)

        def _increment_and_negate(df: DataFrame, col_subset: List[str]) -> DataFrame:
            unflattened = [(f"{col}+1", f"{col}*(-1)") for col in col_subset]
            return df.select_expr([val for tup in unflattened for val in tup])

        results = map_dataframe_by_column(
            df=dataset,
            cols=schema,
            map_func=_increment_and_negate,
            partition_size=2,
        )

        assert results == [[1, 0, 2, -1, 3, -2, 4, -3], [6, -5, 7, -6, 8, -7, 9, -8]]

    def test_map_dataframe_by_column_nonpositive_partition_size_not_allowed(self):
        schema = ["a", "b", "c", "d", "_exclude"]
        dataset = self._session.create_dataframe([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], schema=schema)

        def _negate(df: DataFrame, col_subset: List[str]) -> DataFrame:
            return df.select_expr([f"{col}*(-1)" for col in col_subset])

        with self.assertRaisesRegex(Exception, "must be a positive integer"):
            map_dataframe_by_column(
                df=dataset,
                cols=["a", "b", "c", "d"],
                map_func=_negate,
                partition_size=0,
            )

    def test_map_dataframe_by_column_variable_column_output_not_allowed(self):
        schema = ["1", "2", "3", "4"]
        dataset = self._session.create_dataframe([["a", "b", "c", "d"]], schema=schema)

        def _unroll_columns(df: DataFrame, col_subset: List[str]) -> DataFrame:
            unflattened = [["'value'"] * int(col) for col in col_subset]
            return df.select_expr([f"{val} AS VALUE_COL_{idx}" for tup in unflattened for (idx, val) in enumerate(tup)])

        with self.assertRaisesRegex(Exception, "must contain the same number of columns"):
            map_dataframe_by_column(
                df=dataset,
                cols=schema,
                map_func=_unroll_columns,
                partition_size=1,
            )

    def test_map_dataframe_by_column_variable_row_output_not_allowed(self):
        schema = ["1", "2", "3", "4"]
        dataset = self._session.create_dataframe([["a", "b", "c", "d"]], schema=schema)

        def _unroll_rows(df: DataFrame, col_subset: List[str]) -> DataFrame:
            unioned_df = df.select_expr("1")
            for col in col_subset:
                for _ in range(int(col)):
                    unioned_df = unioned_df.union_all(df.select_expr("1"))

            return unioned_df

        with self.assertRaisesRegex(Exception, "must return the same number of rows"):
            map_dataframe_by_column(
                df=dataset,
                cols=schema,
                map_func=_unroll_rows,
                partition_size=2,
            )

    @parameterized.parameters(
        {"cols": ["b", "c"]},
        {"cols": []},
    )
    def test_map_dataframe_by_column_mismatched_schema(self, cols):
        dataset = self._session.create_dataframe([[1, 2]], schema=["a", "b"])

        def _negate(df: DataFrame, col_subset: List[str]) -> DataFrame:
            return df.select_expr([f"{col}*(-1)" for col in col_subset])

        with self.assertRaisesRegex(Exception, "does not index into the dataset"):
            map_dataframe_by_column(
                df=dataset,
                cols=cols,
                map_func=_negate,
                partition_size=2,
            )


if __name__ == "__main__":
    main()
