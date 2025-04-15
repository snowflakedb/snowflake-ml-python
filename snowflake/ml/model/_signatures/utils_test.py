from typing import Any

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import core, utils
from snowflake.ml.test_utils import exception_utils


class ModelSignatureMiscTest(absltest.TestCase):
    def testrename_features(self) -> None:
        utils.rename_features([])

        fts = [core.FeatureSpec("a", core.DataType.INT64)]
        self.assertListEqual(
            utils.rename_features(fts, ["b"]),
            [core.FeatureSpec("b", core.DataType.INT64)],
        )

        fts = [core.FeatureSpec("a", core.DataType.INT64, shape=(2,))]
        self.assertListEqual(
            utils.rename_features(fts, ["b"]),
            [core.FeatureSpec("b", core.DataType.INT64, shape=(2,))],
        )

        fts = [core.FeatureSpec("a", core.DataType.INT64, shape=(2,))]
        utils.rename_features(fts)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="\\d+ feature names are provided, while there are \\d+ features.",
        ):
            fts = [core.FeatureSpec("a", core.DataType.INT64, shape=(2,))]
            utils.rename_features(fts, ["b", "c"])

    def testrename_pandas_df(self) -> None:
        fts = [
            core.FeatureSpec("input_feature_0", core.DataType.INT64),
            core.FeatureSpec("input_feature_1", core.DataType.INT64),
        ]

        df = pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"])

        pd.testing.assert_frame_equal(df, utils.rename_pandas_df(df, fts))

        df = pd.DataFrame([[2, 5], [6, 8]])

        pd.testing.assert_frame_equal(df, utils.rename_pandas_df(df, fts), check_names=False)
        pd.testing.assert_index_equal(
            pd.Index(["input_feature_0", "input_feature_1"]), right=utils.rename_pandas_df(df, fts).columns
        )

    def test_infer_list(self) -> None:
        data1 = [1, 2, 3]
        expected_spec1 = core.FeatureSpec(name="test_feature", dtype=core.DataType.INT64, shape=(3,))
        self.assertEqual(utils.infer_list("test_feature", data1), expected_spec1)

        data2 = [[1, 2], [3, 4]]
        expected_spec2 = core.FeatureSpec(name="test_feature", dtype=core.DataType.INT64, shape=(2, 2))
        self.assertEqual(utils.infer_list("test_feature", data2), expected_spec2)

        data3 = [{"a": 3, "b": 4}]
        expected_spec3 = core.FeatureGroupSpec(
            name="test_feature",
            specs=[core.FeatureSpec("a", core.DataType.INT64), core.FeatureSpec("b", core.DataType.INT64)],
            shape=(-1,),
        )
        self.assertEqual(utils.infer_list("test_feature", data3), expected_spec3)

        data4: list[Any] = []
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Empty list is found.",
        ):
            utils.infer_list("test_feature", data4)

    def test_infer_dict(self) -> None:
        data = {"a": 1, "b": "2"}
        expected_spec = core.FeatureGroupSpec(
            name="test_feature",
            specs=[
                core.FeatureSpec(name="a", dtype=core.DataType.INT64),
                core.FeatureSpec(name="b", dtype=core.DataType.STRING),
            ],
        )
        self.assertEqual(utils.infer_dict("test_feature", data), expected_spec)

        data = {"a": [1, 2], "b": [3, 4]}
        expected_spec = core.FeatureGroupSpec(
            name="test_feature",
            specs=[
                core.FeatureSpec(name="a", dtype=core.DataType.INT64, shape=(2,)),
                core.FeatureSpec(name="b", dtype=core.DataType.INT64, shape=(2,)),
            ],
        )
        self.assertEqual(utils.infer_dict("test_feature", data), expected_spec)

        data = {"a": [{"a": 3, "b": 4}], "b": {"a": 3, "b": 4}}
        expected_spec = core.FeatureGroupSpec(
            name="test_feature",
            specs=[
                core.FeatureGroupSpec(
                    name="a",
                    specs=[
                        core.FeatureSpec(name="a", dtype=core.DataType.INT64),
                        core.FeatureSpec(name="b", dtype=core.DataType.INT64),
                    ],
                    shape=(-1,),
                ),
                core.FeatureGroupSpec(
                    name="b",
                    specs=[
                        core.FeatureSpec(name="a", dtype=core.DataType.INT64),
                        core.FeatureSpec(name="b", dtype=core.DataType.INT64),
                    ],
                ),
            ],
        )
        self.assertEqual(utils.infer_dict("test_feature", data), expected_spec)

        data = {}
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Empty dictionary is found.",
        ):
            utils.infer_dict("test_feature", data)


if __name__ == "__main__":
    absltest.main()
