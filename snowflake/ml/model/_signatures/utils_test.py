from typing import Any

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import core, utils
from snowflake.ml.test_utils import exception_utils


class HuggingFacePipelineSignatureAutoInferTest(absltest.TestCase):
    def test_video_classification_signature_auto_infer(self) -> None:
        """Test that video-classification task produces correct signature."""
        sig = utils.huggingface_pipeline_signature_auto_infer(
            task="video-classification",
            params={},
        )

        self.assertIsNotNone(sig)

        # Build expected signature for comparison
        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="video", dtype=core.DataType.BYTES),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="labels",
                    specs=[
                        core.FeatureSpec(name="label", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
                    ],
                    shape=(-1,),
                ),
            ],
        )

        assert sig is not None

        # Compare inputs
        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        self.assertEqual(sig.inputs[0], expected_sig.inputs[0])

        # Compare outputs using equality (which checks name, specs, and shape)
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])


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
