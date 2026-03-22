import datetime
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

    def test_image_feature_extraction_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-feature-extraction", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="images", dtype=core.DataType.BYTES),
            ],
            outputs=[
                core.FeatureSpec(name="feature_extraction", dtype=core.DataType.DOUBLE, shape=(-1,)),
            ],
        )

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        self.assertEqual(sig.inputs[0], expected_sig.inputs[0])
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])

    def test_image_to_text_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-to-text", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="images", dtype=core.DataType.BYTES),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="outputs",
                    specs=[
                        core.FeatureSpec(name="generated_text", dtype=core.DataType.STRING),
                    ],
                    shape=(-1,),
                ),
            ],
        )

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        self.assertEqual(sig.inputs[0], expected_sig.inputs[0])
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])

    def test_object_detection_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="object-detection", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="images", dtype=core.DataType.BYTES),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="detections",
                    specs=[
                        core.FeatureSpec(name="label", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
                        core.FeatureGroupSpec(
                            name="box",
                            specs=[
                                core.FeatureSpec(name="xmin", dtype=core.DataType.INT64),
                                core.FeatureSpec(name="ymin", dtype=core.DataType.INT64),
                                core.FeatureSpec(name="xmax", dtype=core.DataType.INT64),
                                core.FeatureSpec(name="ymax", dtype=core.DataType.INT64),
                            ],
                        ),
                    ],
                    shape=(-1,),
                ),
            ],
        )

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        self.assertEqual(sig.inputs[0], expected_sig.inputs[0])
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


class InferParamTest(absltest.TestCase):
    """Tests for utils.infer_param."""

    def test_scalars(self) -> None:
        cases = [
            ("int_p", 42, core.DataType.INT64),
            ("float_p", 3.14, core.DataType.DOUBLE),
            ("str_p", "hello", core.DataType.STRING),
            ("bool_p", True, core.DataType.BOOL),
            ("bytes_p", b"data", core.DataType.BYTES),
            ("dt_p", datetime.datetime(2024, 1, 1), core.DataType.TIMESTAMP_NTZ),
        ]
        for name, value, expected_dtype in cases:
            param = utils.infer_param(name, value)
            self.assertEqual(param.name, name)
            self.assertEqual(param.dtype, expected_dtype)
            self.assertEqual(param.default_value, value)
            self.assertIsNone(param.shape)

    def test_list(self) -> None:
        param = utils.infer_param("weights", [1.0, 2.0, 3.0])
        self.assertEqual(param.name, "weights")
        self.assertEqual(param.dtype, core.DataType.DOUBLE)
        self.assertEqual(param.default_value, [1.0, 2.0, 3.0])
        self.assertEqual(param.shape, (-1,))

    def test_nested_list(self) -> None:
        param = utils.infer_param("matrix", [[1, 2], [3, 4]])
        self.assertEqual(param.dtype, core.DataType.INT64)
        self.assertEqual(param.shape, (-1, -1))

    def test_mixed_numeric_widens(self) -> None:
        """int+float list widens to DOUBLE via numpy, matching feature inference."""
        param = utils.infer_param("nums", [1, 0.7, 2])
        self.assertEqual(param.dtype, core.DataType.DOUBLE)
        self.assertEqual(param.shape, (-1,))

    def test_bool_list(self) -> None:
        param = utils.infer_param("flags", [True, False, True])
        self.assertEqual(param.dtype, core.DataType.BOOL)
        self.assertEqual(param.shape, (-1,))

    def test_none_raises(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Cannot infer ParamSpec dtype.*from None value",
        ):
            utils.infer_param("bad", None)

    def test_empty_list_raises(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Cannot infer ParamSpec dtype.*from an empty list",
        ):
            utils.infer_param("bad", [])

    def test_unsupported_scalar_type_raises(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Cannot infer ParamSpec dtype from value of type set",
        ):
            utils.infer_param("bad", {1, 2, 3})

    def test_dict_raises(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Dict values are not yet supported.",
        ):
            utils.infer_param("bad", {"a": 1})

    def test_dict_in_list_raises(self) -> None:
        """Dicts in lists produce object dtype -> rejected by convert_list_to_ndarray."""
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Ragged nested or Unsupported list-like data",
        ):
            utils.infer_param("bad", [{"a": 1}])

    def test_mixed_incompatible_raises(self) -> None:
        """Truly incompatible types produce object dtype -> rejected by convert_list_to_ndarray."""
        cases = [
            [datetime.datetime(2024, 1, 1), 42],
            [1, None],
        ]
        for value in cases:
            with exception_utils.assert_snowml_exceptions(
                self,
                expected_original_error_type=ValueError,
                expected_regex="Ragged nested or Unsupported list-like data",
            ):
                utils.infer_param("bad", value)

    def test_ragged_list_raises(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Ragged nested or Unsupported list-like data",
        ):
            utils.infer_param("bad", [[1, 2], [3]])


if __name__ == "__main__":
    absltest.main()
