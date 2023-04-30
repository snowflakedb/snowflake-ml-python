import numpy as np
import pandas as pd
import sparse
from absl.testing import absltest
from pandas.api import types as pandas_types


class SparseTest(absltest.TestCase):
    """Testing sparse utility functions."""

    def test_to_sparse_pandas_normal(self) -> None:
        """Tests the conversion of data frames."""

        # Normal operation with valid integer and string inputs.
        df_expected = pd.DataFrame(
            {
                "str1": ["a", "b", "c", "d"],
                "sparse1_0": [0, 2, 0, 0],
                "sparse1_1": [0, 3, 0, 0],
                "sparse1_2": [0, 0, 0, 0],
                "sparse1_3": [1, 0, 0, 1],
                "sparse2_0": [np.nan, "zz", np.nan, np.nan],
                "sparse2_1": ["xy", np.nan, np.nan, np.nan],
                "sparse2_2": [np.nan, np.nan, "", "xyz"],
            }
        )
        df = pd.DataFrame(
            {
                "str1": ["a", "b", "c", "d"],
                "sparse1": [
                    '{"3": 1, "array_length": 4}',
                    '{"0": 2, "1": 3, "array_length": 4}',
                    None,
                    '{"3": 1, "array_length": 4}',
                ],
                "sparse2": [
                    '{"1": "xy", "array_length": 3}',
                    '{"0": "zz", "array_length": 3}',
                    '{"2": "", "array_length": 3}',
                    '{"2": "xyz", "array_length": 3}',
                ],
            }
        )
        df_actual = sparse._pandas_to_sparse_pandas(df, ["sparse1", "sparse2"])
        self.assertTrue(df_expected.compare(df_actual).empty)
        self.assertTrue(
            pandas_types.is_sparse(df_actual["sparse1_0"].dtype)
            & pandas_types.is_sparse(df_actual["sparse1_1"].dtype)
            & pandas_types.is_sparse(df_actual["sparse1_2"].dtype)
            & pandas_types.is_sparse(df_actual["sparse1_3"].dtype)
            & pandas_types.is_sparse(df_actual["sparse2_0"].dtype)
            & pandas_types.is_sparse(df_actual["sparse2_1"].dtype)
            & pandas_types.is_sparse(df_actual["sparse2_2"].dtype)
        )

    def test_to_sparse_pandas_array_length_mismatch(self) -> None:
        # array_length mismatch
        df = pd.DataFrame(
            {
                "sparse1": [
                    '{"3": 1, "array_length": 5}',
                    '{"0": 2, "1": 3, "array_length": 4}',
                    '{"3": 1, "array_length": 4}',
                ]
            }
        )
        with self.assertRaises(ValueError):
            sparse._pandas_to_sparse_pandas(df, ["sparse1"])

    def test_to_sparse_pandas_missing_array_length(self) -> None:
        # missing array_length
        df = pd.DataFrame(
            {"sparse1": ['{"3": 1}', '{"0": 2, "1": 3, "array_length": 4}', '{"3": 1, "array_length": 4}']}
        )
        with self.assertRaises(KeyError):
            sparse._pandas_to_sparse_pandas(df, ["sparse1"])

    def test_to_sparse_pandas_value_type_mismatch(self) -> None:
        # value type mismatch
        df = pd.DataFrame(
            {
                "sparse1": [
                    '{"3": 1, "array_length": 4}',
                    '{"0": "a", "1": 3, "array_length": 4}',
                    '{"3": 1, "array_length": 4}',
                ]
            }
        )
        with self.assertRaises(ValueError):
            sparse._pandas_to_sparse_pandas(df, ["sparse1"])

    def test_to_sparse_pandas_index_greater_than_array_length(self) -> None:
        # index greater than or equal to array_length
        df = pd.DataFrame(
            {
                "sparse1": [
                    '{"3": 1, "array_length": 4}',
                    '{"0": 2, "4": 3, "array_length": 4}',
                    '{"3": 1, "array_length": 4}',
                ]
            }
        )
        with self.assertRaises(ValueError):
            sparse._pandas_to_sparse_pandas(df, ["sparse1"])

    def test_to_sparse_pandas_none_integer_key(self) -> None:
        # none-integer key
        df = pd.DataFrame(
            {
                "sparse1": [
                    '{"3": 1, "array_length": 4}',
                    '{"3x": 2, "4": 3, "array_length": 4}',
                    '{"3": 1, "array_length": 4}',
                ]
            }
        )
        with self.assertRaises(ValueError):
            sparse._pandas_to_sparse_pandas(df, ["sparse1"])


if __name__ == "__main__":
    absltest.main()
