import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import core, pandas_handler
from snowflake.ml.test_utils import exception_utils


class PandasDataFrameHandlerTest(absltest.TestCase):
    def test_validate_pd_DataFrame(self) -> None:
        df = pd.DataFrame([])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Empty data is found."
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, 4]], columns=["a", "a"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Duplicate column index is found"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        sub_df = pd.DataFrame([2.5, 6.8])
        df = pd.DataFrame([[1, sub_df], [2, sub_df]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unsupported type confronted in"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame(
            [[1, 2.0, 1, 2.0, 1, 2.0], [2, 4.0, 2, 4.0, 2, 4.0]],
            columns=pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
        )
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Duplicate column index is found"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, 4]], columns=["a", "a"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Duplicate column index is found"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, "Hello"], [2, [2, 6]]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Inconsistent type of object"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, [2, 6]]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Inconsistent type of object"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2, [6]]], [2, [2, 6]]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Ragged nested or Unsupported list-like data"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2, 6]], [2, [2, [6]]]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Ragged nested or Unsupported list-like data"
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2, 6]]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Inconsistent type of element in object found in column data",
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2, 6])]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Inconsistent type of element in object found in column data",
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, 6]], columns=["a", "b"])
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Inconsistent type of object found in column data",
        ):
            pandas_handler.PandasDataFrameHandler.validate(df)

    def test_trunc_pd_DataFrame(self) -> None:
        df = pd.DataFrame([1] * (pandas_handler.PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))

        pd.testing.assert_frame_equal(
            pd.DataFrame([1] * (pandas_handler.PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT)),
            pandas_handler.PandasDataFrameHandler.truncate(df),
        )

        df = pd.DataFrame([1] * (pandas_handler.PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))

        pd.testing.assert_frame_equal(
            df,
            pandas_handler.PandasDataFrameHandler.truncate(df),
        )

    def test_infer_signature_pd_DataFrame(self) -> None:
        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [core.FeatureSpec("a", core.DataType.INT64)],
        )

        df = pd.DataFrame(["a", "b", "c", "d"], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [core.FeatureSpec("a", core.DataType.STRING)],
        )

        df = pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [core.FeatureSpec("a", core.DataType.BYTES)],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64),
                core.FeatureSpec("input_feature_1", core.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5]]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.DOUBLE, shape=(-1,)),
            ],
        )

        df = pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.DOUBLE, shape=(2, 1)),
            ],
        )

        a = np.array([2.5, 6.8])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5])]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.DOUBLE, shape=(-1,)),
            ],
        )

        a = np.array([[2, 5], [6, 8]])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT64),
                core.FeatureSpec("b", core.DataType.INT64, shape=(2, 2)),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3]))
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("2000Q1", core.DataType.INT64),
                core.FeatureSpec("2002Q3", core.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.date_range("2020-01-06", "2020-03-03", freq="MS"))
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("2020-02-01 00:00:00", core.DataType.INT64),
                core.FeatureSpec("2020-03-01 00:00:00", core.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame(
            [[1, 2.0], [2, 4.0]], columns=pd.TimedeltaIndex(data=["1 days 02:00:00", "1 days 06:05:01.000030"])
        )
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("1 days 02:00:00", core.DataType.INT64),
                core.FeatureSpec("1 days 06:05:01.000030", core.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.interval_range(start=0, end=2))
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("(0, 1]", core.DataType.INT64),
                core.FeatureSpec("(1, 2]", core.DataType.DOUBLE),
            ],
        )

        arrays = [[1, 2], ["red", "blue"]]
        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.MultiIndex.from_arrays(arrays, names=("number", "color")))
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("(1, 'red')", core.DataType.INT64),
                core.FeatureSpec("(2, 'blue')", core.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="output"),
            [core.FeatureSpec("output_feature_0", core.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="output"),
            [core.FeatureSpec("a", core.DataType.INT64)],
        )

        df = pd.DataFrame(["a", "b", "c", "d"], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="output"),
            [core.FeatureSpec("a", core.DataType.STRING)],
        )

        df = pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="output"),
            [core.FeatureSpec("a", core.DataType.BYTES)],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]])
        self.assertListEqual(
            pandas_handler.PandasDataFrameHandler.infer_signature(df, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64),
                core.FeatureSpec("output_feature_1", core.DataType.DOUBLE),
            ],
        )

    def test_convert_to_df_pd_DataFrame(self) -> None:
        a = np.array([[2, 5], [6, 8]])
        li = [[2, 5], [6, 8]]
        df1 = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        df2 = pd.DataFrame([[1, li], [2, li]], columns=["a", "b"])
        pd.testing.assert_frame_equal(pandas_handler.PandasDataFrameHandler.convert_to_df(df1), df2)


if __name__ == "__main__":
    absltest.main()
