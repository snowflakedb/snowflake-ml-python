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

    def test_document_question_answering_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="document-question-answering", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="image", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="question", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="answers",
                    specs=[
                        core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
                        core.FeatureSpec(name="start", dtype=core.DataType.INT64),
                        core.FeatureSpec(name="end", dtype=core.DataType.INT64),
                        core.FeatureSpec(name="answer", dtype=core.DataType.STRING),
                    ],
                    shape=(-1,),
                ),
            ],
        )

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        for i in range(len(sig.inputs)):
            self.assertEqual(sig.inputs[i], expected_sig.inputs[i])
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])

    def test_visual_question_answering_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="visual-question-answering", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="image", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="question", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureGroupSpec(
                    name="answers",
                    specs=[
                        core.FeatureSpec(name="answer", dtype=core.DataType.STRING),
                        core.FeatureSpec(name="score", dtype=core.DataType.DOUBLE),
                    ],
                    shape=(-1,),
                ),
            ],
        )

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        for i in range(len(sig.inputs)):
            self.assertEqual(sig.inputs[i], expected_sig.inputs[i])
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

    def test_zero_shot_image_classification_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-image-classification", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="images", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="candidate_labels", dtype=core.DataType.STRING, shape=(-1,)),
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

        self.assertEqual(len(sig.inputs), len(expected_sig.inputs))
        for i in range(len(sig.inputs)):
            self.assertEqual(sig.inputs[i], expected_sig.inputs[i])
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])

    def test_zero_shot_object_detection_signature_auto_infer(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-object-detection", params={})
        self.assertIsNotNone(sig)
        assert sig is not None

        expected_sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="images", dtype=core.DataType.BYTES),
                core.FeatureSpec(name="candidate_labels", dtype=core.DataType.STRING, shape=(-1,)),
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
        for i in range(len(sig.inputs)):
            self.assertEqual(sig.inputs[i], expected_sig.inputs[i])
        self.assertEqual(len(sig.outputs), len(expected_sig.outputs))
        self.assertEqual(sig.outputs[0], expected_sig.outputs[0])


class HuggingFaceParamSpecTest(absltest.TestCase):
    """Tests that applicable tasks include ParamSpecs in their auto-inferred signatures."""

    def _get_param_names(self, task: str, **extra_kw: Any) -> set[str]:
        sig = utils.huggingface_pipeline_signature_auto_infer(task=task, params={}, **extra_kw)
        self.assertIsNotNone(sig)
        assert sig is not None
        return {p.name for p in sig.params}

    def test_fill_mask_params(self) -> None:
        self.assertEqual(self._get_param_names("fill-mask"), {"targets", "top_k"})

    def test_fill_mask_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="fill-mask", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertIsNone(defaults["targets"])
        self.assertIsNone(defaults["top_k"])

    def test_fill_mask_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="fill-mask", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- question-answering ---

    def test_question_answering_params(self) -> None:
        self.assertEqual(
            self._get_param_names("question-answering"),
            {
                "top_k",
                "doc_stride",
                "max_answer_len",
                "max_seq_len",
                "max_question_len",
                "handle_impossible_answer",
                "align_to_words",
            },
        )

    def test_question_answering_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="question-answering", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["top_k"], 1)
        self.assertEqual(defaults["doc_stride"], 128)
        self.assertEqual(defaults["max_answer_len"], 15)
        self.assertEqual(defaults["max_seq_len"], 384)
        self.assertEqual(defaults["max_question_len"], 64)
        self.assertFalse(defaults["handle_impossible_answer"])
        self.assertTrue(defaults["align_to_words"])

    def test_question_answering_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="question-answering", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    def test_question_answering_both_variants_have_same_params(self) -> None:
        flat_sig = utils.huggingface_pipeline_signature_auto_infer(task="question-answering", params={})
        list_sig = utils.huggingface_pipeline_signature_auto_infer(task="question-answering", params={"top_k": 3})
        assert flat_sig is not None and list_sig is not None
        self.assertEqual({p.name for p in flat_sig.params}, {p.name for p in list_sig.params})

    # --- table-question-answering ---

    def test_table_question_answering_params(self) -> None:
        self.assertEqual(
            self._get_param_names("table-question-answering"),
            {"sequential", "padding", "truncation"},
        )

    def test_table_question_answering_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="table-question-answering", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertFalse(defaults["sequential"])
        self.assertIsNone(defaults["padding"])
        self.assertIsNone(defaults["truncation"])

    def test_table_question_answering_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="table-question-answering", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- text-classification / sentiment-analysis ---

    def test_text_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("text-classification"),
            {"top_k", "function_to_apply"},
        )

    def test_sentiment_analysis_params(self) -> None:
        self.assertEqual(
            self._get_param_names("sentiment-analysis"),
            {"top_k", "function_to_apply"},
        )

    def test_text_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="text-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["top_k"], 1)
        self.assertIsNone(defaults["function_to_apply"])

    def test_text_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="text-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    def test_text_classification_both_variants_have_same_params(self) -> None:
        flat_sig = utils.huggingface_pipeline_signature_auto_infer(task="text-classification", params={})
        list_sig = utils.huggingface_pipeline_signature_auto_infer(task="text-classification", params={"top_k": 5})
        assert flat_sig is not None and list_sig is not None
        self.assertEqual({p.name for p in flat_sig.params}, {p.name for p in list_sig.params})

    # --- image-classification ---

    def test_image_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("image-classification"),
            {"top_k", "function_to_apply", "timeout"},
        )

    def test_image_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["top_k"], 5)
        self.assertIsNone(defaults["function_to_apply"])
        self.assertIsNone(defaults["timeout"])

    def test_image_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- video-classification ---

    def test_video_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("video-classification"),
            {"top_k", "num_frames", "frame_sampling_rate", "function_to_apply"},
        )

    def test_video_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="video-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["top_k"], 5)
        self.assertIsNone(defaults["num_frames"])
        self.assertEqual(defaults["frame_sampling_rate"], 1)
        self.assertEqual(defaults["function_to_apply"], "softmax")

    def test_video_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="video-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- visual-question-answering ---

    def test_visual_question_answering_params(self) -> None:
        self.assertEqual(
            self._get_param_names("visual-question-answering"),
            {"top_k", "timeout"},
        )

    def test_visual_question_answering_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="visual-question-answering", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["top_k"], 5)
        self.assertIsNone(defaults["timeout"])

    def test_visual_question_answering_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="visual-question-answering", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- image-feature-extraction ---

    def test_image_feature_extraction_params(self) -> None:
        self.assertEqual(
            self._get_param_names("image-feature-extraction"),
            {"timeout"},
        )

    def test_image_feature_extraction_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-feature-extraction", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertIsNone(defaults["timeout"])

    def test_image_feature_extraction_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="image-feature-extraction", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- object-detection ---

    def test_object_detection_params(self) -> None:
        self.assertEqual(
            self._get_param_names("object-detection"),
            {"threshold", "timeout"},
        )

    def test_object_detection_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="object-detection", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["threshold"], 0.5)
        self.assertIsNone(defaults["timeout"])

    def test_object_detection_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="object-detection", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- zero-shot-image-classification ---

    def test_zero_shot_image_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("zero-shot-image-classification"),
            {"hypothesis_template", "timeout"},
        )

    def test_zero_shot_image_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-image-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["hypothesis_template"], "This is a photo of {}.")
        self.assertIsNone(defaults["timeout"])

    def test_zero_shot_image_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-image-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- zero-shot-object-detection ---

    def test_zero_shot_object_detection_params(self) -> None:
        self.assertEqual(
            self._get_param_names("zero-shot-object-detection"),
            {"threshold", "top_k", "timeout"},
        )

    def test_zero_shot_object_detection_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-object-detection", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["threshold"], 0.1)
        self.assertIsNone(defaults["top_k"])
        self.assertIsNone(defaults["timeout"])

    def test_zero_shot_object_detection_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-object-detection", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- zero-shot-classification ---

    def test_zero_shot_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("zero-shot-classification"),
            {"hypothesis_template", "multi_label"},
        )

    def test_zero_shot_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["hypothesis_template"], "This example is {}.")
        self.assertFalse(defaults["multi_label"])

    def test_zero_shot_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))

    # --- zero-shot-audio-classification ---

    def test_zero_shot_audio_classification_params(self) -> None:
        self.assertEqual(
            self._get_param_names("zero-shot-audio-classification"),
            {"hypothesis_template"},
        )

    def test_zero_shot_audio_classification_param_defaults(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-audio-classification", params={})
        assert sig is not None
        defaults = {p.name: p.default_value for p in sig.params}
        self.assertEqual(defaults["hypothesis_template"], "This is a sound of {}.")

    def test_zero_shot_audio_classification_param_names_dont_collide_with_inputs(self) -> None:
        sig = utils.huggingface_pipeline_signature_auto_infer(task="zero-shot-audio-classification", params={})
        assert sig is not None
        input_names = {spec.name.upper() for spec in sig.inputs}
        param_names = {p.name.upper() for p in sig.params}
        self.assertTrue(input_names.isdisjoint(param_names))


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
