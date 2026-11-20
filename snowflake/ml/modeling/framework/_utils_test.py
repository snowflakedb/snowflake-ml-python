import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.modeling.framework import _utils


class StringNaToNoneTest(absltest.TestCase):
    def test_string_dtype_missing_becomes_none(self) -> None:
        series = pd.Series(["a", pd.NA, "b"], dtype="string")
        converted = _utils.string_na_to_none(series)
        self.assertEqual(converted.dtype, object)
        self.assertEqual(converted.tolist(), ["a", None, "b"])

    def test_object_series_unchanged(self) -> None:
        series = pd.Series(["a", None, "b"], dtype=object)
        converted = _utils.string_na_to_none(series)
        self.assertIs(converted, series)

    def test_numeric_series_unchanged(self) -> None:
        series = pd.Series([1.0, np.nan, 2.0])
        converted = _utils.string_na_to_none(series)
        self.assertIs(converted, series)
        self.assertTrue(np.isnan(converted.iloc[1]))

    def test_dataframe_converts_only_string_columns(self) -> None:
        df = pd.DataFrame(
            {
                "cat": pd.Series(["a", pd.NA], dtype="string"),
                "num": pd.Series([1.0, np.nan]),
            }
        )
        converted = _utils.dataframe_string_na_to_none(df)
        self.assertEqual(converted["cat"].tolist(), ["a", None])
        self.assertTrue(np.isnan(converted["num"].iloc[1]))
        self.assertEqual(df["cat"].tolist()[0], "a")
        self.assertTrue(pd.isna(df["cat"].iloc[1]))

    def test_dataframe_without_string_columns_is_unchanged(self) -> None:
        df = pd.DataFrame({"num": [1.0, np.nan]})
        self.assertIs(_utils.dataframe_string_na_to_none(df), df)


if __name__ == "__main__":
    absltest.main()
