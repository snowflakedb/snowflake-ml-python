import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.test_utils import exception_utils


class ModelSignatureMiscTest(absltest.TestCase):
    def test_infer_signature(self) -> None:
        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, None, 4])
        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(arr, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        d1 = datetime.datetime(year=2024, month=6, day=21, hour=1, minute=1, second=1)
        d2 = datetime.datetime(year=2024, month=7, day=11, hour=1, minute=1, second=1)
        df_dates = pd.DataFrame([d1, d2])
        self.assertListEqual(
            model_signature._infer_signature(df_dates, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.TIMESTAMP_NTZ)],
        )

        arr_dates = np.array([d1, d2], dtype=np.datetime64)
        self.assertListEqual(
            model_signature._infer_signature(arr_dates, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.TIMESTAMP_NTZ)],
        )

        lt_dates = [d1, d2]
        self.assertListEqual(
            model_signature._infer_signature(lt_dates, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.TIMESTAMP_NTZ)],
        )

        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            model_signature._infer_signature(lt1, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        lt2 = [[1, 2], [3, 4]]
        self.assertListEqual(
            model_signature._infer_signature(lt2, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        lt3 = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature(lt3, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt3, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        torch_tensor = torch.LongTensor([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(torch_tensor, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
            ],
        )
        lt4 = [torch_tensor, torch_tensor]
        self.assertListEqual(
            model_signature._infer_signature(lt4, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt4, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt4, role="input", use_snowflake_identifiers=True),
            [
                model_signature.FeatureSpec('"input_feature_0"', model_signature.DataType.INT64),
                model_signature.FeatureSpec('"input_feature_1"', model_signature.DataType.INT64),
            ],
        )

        tf_tensor = tf.constant([1, 2, 3, 4], dtype=tf.int64)
        self.assertListEqual(
            model_signature._infer_signature(tf_tensor, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
            ],
        )
        lt5 = [tf_tensor, tf_tensor]
        self.assertListEqual(
            model_signature._infer_signature(lt5, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt5, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        # categorical column
        df = pd.DataFrame({"column_0": ["a", "b", "c", "d"], "column_1": [1, 2, 3, 4]})
        df["column_0"] = df["column_0"].astype("category")
        df["column_1"] = df["column_1"].astype("category")

        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("column_0", model_signature.DataType.STRING),
                model_signature.FeatureSpec("column_1", model_signature.DataType.INT64),
            ],
        )

        series = pd.Series(["a", "b", "c", "d"], name="column_0")
        series = series.astype("category")
        self.assertListEqual(
            model_signature._infer_signature(series, role="input"),
            [model_signature.FeatureSpec("column_0", model_signature.DataType.STRING)],
        )

        df = pd.DataFrame([1, 2, 3, 4])
        lt = [df, arr]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=NotImplementedError, expected_regex="Un-supported type provided"
        ):
            model_signature._infer_signature(lt, role="input")

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Inconsistent type of object found in data",
        ):
            model_signature._infer_signature([True, 1], role="input")

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=NotImplementedError, expected_regex="Un-supported type provided"
        ):
            model_signature._infer_signature(1, role="input")

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=NotImplementedError, expected_regex="Un-supported type provided"
        ):
            model_signature._infer_signature([], role="input")

        df = pd.DataFrame([[{"a": 1}], [{"a": 1}]])
        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [
                model_signature.FeatureGroupSpec(
                    "input_feature_0",
                    [
                        model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                    ],
                )
            ],
        )

        df = pd.DataFrame([[[{"a": 1}]], [[{"a": 1}]]])
        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [
                model_signature.FeatureGroupSpec(
                    "input_feature_0",
                    [
                        model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                    ],
                    shape=(-1,),
                )
            ],
        )

        lt = [[{"a": 1}], [{"a": 1}]]
        self.assertListEqual(
            model_signature._infer_signature(lt, role="input"),
            [
                model_signature.FeatureGroupSpec(
                    "input_feature_0",
                    [
                        model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                    ],
                )
            ],
        )

        lt = [[[{"a": 1}]], [[{"a": 1}]]]
        self.assertListEqual(
            model_signature._infer_signature(lt, role="input"),
            [
                model_signature.FeatureGroupSpec(
                    "input_feature_0",
                    [
                        model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                    ],
                    shape=(-1,),
                )
            ],
        )

    def test_validate_pandas_df(self) -> None:
        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT8),
            model_signature.FeatureSpec("b", model_signature.DataType.UINT64),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)
        model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"], index=[1, 2]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(pd.DataFrame([[257, 5], [6, 8]], columns=["a", "b"]), fts, strict=True)

        model_signature._validate_pandas_df(pd.DataFrame([[257, 5], [6, 8]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(pd.DataFrame([[2, -5], [6, 8]], columns=["a", "b"]), fts, strict=True)

        model_signature._validate_pandas_df(pd.DataFrame([[2, -5], [6, 8]], columns=["a", "b"]), fts)

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT8),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([[2, -5], [6, 8]], columns=["a", "b"]), fts)

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.INT64),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([[2, None], [6, 8]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([[2, None], [6, 8.0]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(pd.DataFrame([[2.5, 5], [6.8, 8]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._validate_pandas_df(pd.DataFrame([5, 6], columns=["a"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([5, 6], columns=["a"]), fts[:1])

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["c", "d"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a scalar feature while list data is provided.",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts
            )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT8, shape=(2,)),
            model_signature.FeatureSpec("b", model_signature.DataType.UINT64, shape=(2,)),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[[1, 2], [2, 6]], [[2, 3], [2, 6]]], columns=["a", "b"]), fts)
        model_signature._validate_pandas_df(
            pd.DataFrame([[[1, 2], [2, 6]], [[2, 3], [2, 6]]], columns=["a", "b"], index=[1, 2]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[[1, 257], [2, 6]], [[2, 3], [2, 6]]], columns=["a", "b"]), fts, strict=True
            )

        model_signature._validate_pandas_df(
            pd.DataFrame([[[1, 257], [2, 6]], [[2, 3], [2, 6]]], columns=["a", "b"]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[[1, 2], [2, -6]], [[2, 3], [2, 6]]], columns=["a", "b"]), fts, strict=True
            )

        model_signature._validate_pandas_df(
            pd.DataFrame([[[1, 2], [2, -6]], [[2, 3], [2, 6]]], columns=["a", "b"]), fts
        )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2.5, 6.8]], [2, None]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a array type feature while scalar data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame([[2, 2.5], [6, 6.8]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
            )

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2, 5]], [2, [6, 8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"], index=[1, 2]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
            )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2, 5])], [2, np.array([6, 8])]], columns=["a", "b"]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a array type feature while scalar data is provided.",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["b"]), fts[-1:]
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a array type feature while scalar data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["b"]), fts[-1:])

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(-1,)),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [2.5, 6.8, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2, 5]], [2, [6, 8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, None]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2, 5])], [2, np.array([6, 8])]], columns=["a", "b"]), fts
        )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2, 1)),
        ]

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(pd.DataFrame([[1, [[2.5], [6.8]]], [2, None]], columns=["a", "b"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8], [6.8]]]], columns=["a", "b"]), fts
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts
            )

        model_signature._validate_pandas_df(pd.DataFrame([[1, [[2], [5]]], [2, [[6], [8]]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([[2.5], [6.8]])], [2, np.array([[2.5], [6.8]])]], columns=["a", "b"]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([[2.5], [6.8]])], [2, np.array([[2.5], [6.8], [6.8]])]], columns=["a", "b"]),
                fts,
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature shape [\\(\\)0-9,\\s-]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
            )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([[2], [5]])], [2, np.array([[6], [8]])]], columns=["a", "b"]), fts
        )

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.STRING)]
        model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["a"]), fts)
        model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["a"], index=[2, 5, 6, 8]), fts)

        model_signature._validate_pandas_df(pd.DataFrame(["a", "b", None, "d"], columns=["a"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(
                pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"]), fts
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a scalar feature while list data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [[1, 2]]}), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a scalar feature while array data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [np.array([1, 2])]}), fts)

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.BYTES)]
        model_signature._validate_pandas_df(
            pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"], index=[2, 5, 6, 8]), fts
        )
        model_signature._validate_pandas_df(
            pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]] + [None], columns=["a"]), fts
        )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["a"]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a scalar feature while list data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [[1, 2]]}), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature is a scalar feature while array data is provided.",
        ):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [np.array([1, 2])]}), fts)

        ftgs = [
            model_signature.FeatureGroupSpec(
                "a",
                [
                    model_signature.FeatureSpec("b", model_signature.DataType.INT64),
                    model_signature.FeatureSpec("c", model_signature.DataType.INT64),
                ],
            )
        ]
        model_signature._validate_pandas_df(
            pd.DataFrame({"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]}, columns=["a"]), ftgs
        )

        ftgs = [
            model_signature.FeatureGroupSpec(
                "a",
                [
                    model_signature.FeatureSpec("b", model_signature.DataType.INT64),
                    model_signature.FeatureSpec("c", model_signature.DataType.INT64, shape=(2,)),
                ],
                shape=(-1,),
            )
        ]
        model_signature._validate_pandas_df(
            pd.DataFrame({"a": [[{"b": 1, "c": [2, 3]}, {"b": 4, "c": [5, 6]}], [{"b": 7, "c": [8, 9]}]]}), ftgs
        )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.STRING, shape=(-1,)),
        ]
        model_signature._validate_pandas_df(
            pd.DataFrame({"a": [["a", "b", "c"], ["d", "e", "f"], None]}),
            fts,
        )

    def test_validate_data_with_features(self) -> None:
        fts = [
            model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
            model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
        ]

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Empty data is found."
        ):
            model_signature._convert_and_validate_local_data(np.array([]), fts)

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            model_signature._convert_and_validate_local_data(np.array(5), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._convert_and_validate_local_data(np.array([[2.5, 5], [6.8, 8]]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Un-supported type <class 'list'> provided.",
        ):
            model_signature._convert_and_validate_local_data([], fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Inconsistent type of object found in data",
        ):
            model_signature._convert_and_validate_local_data([1, [1, 1]], fts)

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Ill-shaped list data"
        ):
            model_signature._convert_and_validate_local_data([[1], [1, 1]], fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._convert_and_validate_local_data([[2.1, 5.0], [6.8, 8.0]], fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            model_signature._convert_and_validate_local_data(pd.DataFrame([[2.5, 5], [6.8, 8]]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Data does not have the same number of features as signature",
        ):
            model_signature._convert_and_validate_local_data(pd.DataFrame([5, 6]), fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Data does not have the same number of features as signature.",
        ):
            model_signature._convert_and_validate_local_data(np.array([5, 6]), fts)

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._convert_and_validate_local_data(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)

        df = model_signature._convert_and_validate_local_data(np.array([5, 6]), fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data(pd.DataFrame([5, 6]), fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data([5, 6], fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data(np.array([[2, 5], [6, 8]]), fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])

        df = model_signature._convert_and_validate_local_data(pd.DataFrame([[2, 5], [6, 8]]), fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])

        df = model_signature._convert_and_validate_local_data(
            pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.INT64),
            ],
        )
        self.assertListEqual(df.columns.to_list(), ["a", "b"])

        df = model_signature._convert_and_validate_local_data([[2, 5], [6, 8]], fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])

        ftgs = [
            model_signature.FeatureGroupSpec(
                "input_feature_0",
                [
                    model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                    model_signature.FeatureSpec("b", model_signature.DataType.INT64),
                ],
            )
        ]

        df = model_signature._convert_and_validate_local_data(
            pd.DataFrame({0: [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}), ftgs
        )

        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])


if __name__ == "__main__":
    absltest.main()
