import json
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from snowflake.ml.model._packager.model_meta import model_meta


class UtilTest(absltest.TestCase):
    def test_add_explain_method_signature(self) -> None:
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="feature2"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="output1")],
        )

        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict": predict_sig}
        )
        new_meta = handlers_utils.add_explain_method_signature(
            model_meta=meta,
            explain_method="explain",
            target_method="predict",
        )

        self.assertIn("explain", new_meta.signatures)
        explain_sig = new_meta.signatures["explain"]
        self.assertEqual(explain_sig.inputs, predict_sig.inputs)

        for input_feature in predict_sig.inputs:
            self.assertIn(
                model_signature.FeatureSpec(
                    dtype=model_signature.DataType.DOUBLE, name=f"{input_feature.name}_explanation"
                ),
                explain_sig.outputs,
            )

    def test_convert_explanations_to_2D_df_multi_value_string_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = ["2", "3"]
        explanation_list = np.array(
            [[[0.3, -0.3], [0.5, -0.5]], [[0.2, -0.2], [0.4, -0.4]], [[0.1, -0.1], [0.6, -0.6]]]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{"2": 0.3, "3": -0.3}, {"2": 0.5, "3": -0.5}]],
                1: [json.dumps(v) for v in [{"2": 0.2, "3": -0.2}, {"2": 0.4, "3": -0.4}]],
                2: [json.dumps(v) for v in [{"2": 0.1, "3": -0.1}, {"2": 0.6, "3": -0.6}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_multi_value_int_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = [2, 3]
        explanation_list = np.array(
            [[[0.3, -0.3], [0.5, -0.5]], [[0.2, -0.2], [0.4, -0.4]], [[0.1, -0.1], [0.6, -0.6]]]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{2: 0.3, 3: -0.3}, {2: 0.5, 3: -0.5}]],
                1: [json.dumps(v) for v in [{2: 0.2, 3: -0.2}, {2: 0.4, 3: -0.4}]],
                2: [json.dumps(v) for v in [{2: 0.1, 3: -0.1}, {2: 0.6, 3: -0.6}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_single_value(self) -> None:
        model = mock.MagicMock()
        explanation_list = np.array([[0.3, 0.5], [0.2, 0.4], [-0.1, -0.6]])
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [0.3, 0.5],
                1: [0.2, 0.4],
                2: [-0.1, -0.6],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_multi_value_no_class_attr(self) -> None:
        model = mock.MagicMock(spec=[])
        explanation_list = np.array(
            [[[0.3, -0.3], [0.5, -0.5]], [[0.2, -0.2], [0.4, -0.4]], [[0.1, -0.1], [0.6, -0.6]]]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{0: 0.3, 1: -0.3}, {0: 0.5, 1: -0.5}]],
                1: [json.dumps(v) for v in [{0: 0.2, 1: -0.2}, {0: 0.4, 1: -0.4}]],
                2: [json.dumps(v) for v in [{0: 0.1, 1: -0.1}, {0: 0.6, 1: -0.6}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)


if __name__ == "__main__":
    absltest.main()
