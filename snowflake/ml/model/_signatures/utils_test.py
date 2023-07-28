import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import core, utils


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

        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    absltest.main()
