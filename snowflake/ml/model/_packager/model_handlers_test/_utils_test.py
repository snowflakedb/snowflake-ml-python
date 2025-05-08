import json
from typing import cast
from unittest import mock

import catboost
import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.ml.test_utils import exception_utils


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

    def test_add_inferred_explain_method_signature(self) -> None:
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature2"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="output1")],
        )
        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict": predict_sig}
        )

        def explain_fn(data: type_hints.SupportedDataType) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "a": [0.3, 0.5],
                    "b": [0.3, 0.5],
                    "c": [0.3, 0.5],
                    "d": ["some_string", "some_string"],  # multiclass explanations are formatted as string
                }
            )

        new_meta = handlers_utils.add_inferred_explain_method_signature(
            model_meta=meta,
            explain_method="explain",
            target_method="predict",
            background_data=pd.DataFrame(
                {
                    "feature1": ["a", "b", "c"],
                    "feature2": [10.0, 20.0, 30.0],
                }
            ),
            explain_fn=explain_fn,
            output_feature_names=["feature1_a", "feature1_b", "feature1_c", "feature2"],
        )

        self.assertIn("explain", new_meta.signatures)
        explain_sig = new_meta.signatures["explain"]
        self.assertEqual(explain_sig.inputs, predict_sig.inputs)

        for double_feature in ["feature1_a", "feature1_b", "feature1_c"]:
            self.assertIn(
                model_signature.FeatureSpec(
                    dtype=model_signature.DataType.DOUBLE, name=f"{double_feature}_explanation"
                ),
                explain_sig.outputs,
            )
        self.assertIn(
            model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="feature2_explanation"),
            explain_sig.outputs,
        )

    def test_convert_explanations_to_2D_df_multi_value_string_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = ["2", "3", "4"]
        explanation_list = np.array(
            [
                [[0.3, -0.2, -0.1], [0.5, -0.2, -0.3]],
                [[0.2, -0.15, -0.05], [0.4, -0.2, -0.2]],
                [[-0.05, 0.1, -0.05], [-0.6, -0.6, 1.2]],
            ]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{"2": 0.3, "3": -0.2, "4": -0.1}, {"2": 0.5, "3": -0.2, "4": -0.3}]],
                1: [json.dumps(v) for v in [{"2": 0.2, "3": -0.15, "4": -0.05}, {"2": 0.4, "3": -0.2, "4": -0.2}]],
                2: [json.dumps(v) for v in [{"2": -0.05, "3": 0.1, "4": -0.05}, {"2": -0.6, "3": -0.6, "4": 1.2}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_binary_string_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = ["2", "3"]
        explanation_list = np.array(
            [
                [[0.3, -0.2], [0.5, -0.2]],
                [[0.2, -0.15], [0.4, -0.2]],
                [[-0.05, 0.1], [-0.6, -0.6]],
            ]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [-0.2, -0.2],
                1: [-0.15, -0.2],
                2: [0.1, -0.6],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_multi_value_int_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = [2, 3, 4]
        explanation_list = np.array(
            [
                [[0.3, -0.2, -0.1], [0.5, -0.2, -0.3]],
                [[0.2, -0.15, -0.05], [0.4, -0.2, -0.2]],
                [[-0.05, 0.1, -0.05], [-0.6, -0.6, 1.2]],
            ]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{2: 0.3, 3: -0.2, 4: -0.1}, {2: 0.5, 3: -0.2, 4: -0.3}]],
                1: [json.dumps(v) for v in [{2: 0.2, 3: -0.15, 4: -0.05}, {2: 0.4, 3: -0.2, 4: -0.2}]],
                2: [json.dumps(v) for v in [{2: -0.05, 3: 0.1, 4: -0.05}, {2: -0.6, 3: -0.6, 4: 1.2}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_binary_int_labels(self) -> None:
        model = mock.MagicMock()
        model.classes_ = [2, 3]
        explanation_list = np.array(
            [
                [[0.3, -0.2], [0.5, -0.2]],
                [[0.2, -0.15], [0.4, -0.2]],
                [[-0.05, 0.1], [-0.6, -0.6]],
            ]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [-0.2, -0.2],
                1: [-0.15, -0.2],
                2: [0.1, -0.6],
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
            [
                [[0.3, -0.3, 0.1], [0.5, -0.5, 0.1]],
                [[0.2, -0.2, 0.1], [0.4, -0.4, 0.1]],
                [[0.1, -0.1, 0.1], [0.6, -0.6, 0.1]],
            ]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [json.dumps(v) for v in [{0: 0.3, 1: -0.3, 2: 0.1}, {0: 0.5, 1: -0.5, 2: 0.1}]],
                1: [json.dumps(v) for v in [{0: 0.2, 1: -0.2, 2: 0.1}, {0: 0.4, 1: -0.4, 2: 0.1}]],
                2: [json.dumps(v) for v in [{0: 0.1, 1: -0.1, 2: 0.1}, {0: 0.6, 1: -0.6, 2: 0.1}]],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_convert_explanations_to_2D_df_binary_no_class_attr(self) -> None:
        model = mock.MagicMock(spec=[])
        explanation_list = np.array(
            [[[0.3, -0.3], [0.5, -0.5]], [[0.2, -0.2], [0.4, -0.4]], [[0.1, -0.1], [0.6, -0.6]]]
        )
        explanations_df = handlers_utils.convert_explanations_to_2D_df(model, explanation_list)
        expected_df = pd.DataFrame.from_dict(
            {
                0: [-0.3, -0.5],
                1: [-0.2, -0.4],
                2: [-0.1, -0.6],
            },
            orient="index",
        )
        pd.testing.assert_frame_equal(explanations_df, expected_df)

    def test_validate_model_task(self) -> None:

        task_list = list(type_hints.Task)
        for task in task_list:
            for inferred_task in task_list:
                expected_task = inferred_task if inferred_task != type_hints.Task.UNKNOWN else task
                self.assertEqual(
                    expected_task,
                    handlers_utils.validate_model_task(task, inferred_task),
                )
                if inferred_task != type_hints.Task.UNKNOWN:
                    if task == type_hints.Task.UNKNOWN:
                        with self.assertLogs(level="INFO") as cm:
                            handlers_utils.validate_model_task(task, inferred_task)
                            assert len(cm.output) == 1, "expecting only 1 log"
                            log = cm.output[0]
                            self.assertEqual(
                                f"INFO:absl:Inferred Task: {inferred_task.name} is used as "
                                f"task for this model version",
                                log,
                            )
                    elif inferred_task != task:
                        with self.assertWarnsRegex(
                            UserWarning,
                            f"Inferred Task: {inferred_task.name} is used as task for "
                            f"this model version and passed argument Task: {task.name} is ignored",
                        ):
                            handlers_utils.validate_model_task(task, inferred_task)

    def test_validate_signature_with_signature_and_sample_data(self) -> None:
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.UINT16, name="feature2"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.UINT16, name="output1")],
        )

        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict": predict_sig}
        )
        sample_data = pd.DataFrame({"feature1": [10, 20, 30], "feature2": [10, 20, 30]})
        model = catboost.CatBoostRegressor()
        predict_fun = mock.MagicMock()

        # check the function is called Once only for inputs
        with mock.patch(
            "snowflake.ml.model.model_signature._convert_and_validate_local_data"
        ) as mock_validate_local_data:
            meta = handlers_utils.validate_signature(model, meta, ["predict"], sample_data, predict_fun)
            mock_validate_local_data.assert_called_once()

        # comparing unsigned signature against signed values throws error
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Feature type [^\\s]* is not met by all elements",
        ):
            sample_data = pd.DataFrame({"feature1": [10, 20, 30], "feature2": [-10, 20, 30]})
            meta = handlers_utils.validate_signature(model, meta, ["predict"], sample_data, predict_fun)

    def test_validate_signature_with_only_signature(self) -> None:
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="feature2"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.UINT16, name="output1")],
        )
        predict_fun = mock.MagicMock()
        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict": predict_sig}
        )
        model = catboost.CatBoostRegressor()

        # test with correct signature
        with mock.patch(
            "snowflake.ml.model._packager.model_handlers._utils.validate_target_methods"
        ) as mock_validate_target_methods:
            handlers_utils.validate_signature(model, meta, [], None, predict_fun)
            mock_validate_target_methods.assert_called_once()

        # test with wrong signature. 'predict_not_callable' is not callable from model
        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict_not_callable": predict_sig}
        )
        with self.assertRaisesRegex(
            ValueError, "Target method predict_not_callable is not callable or does not exist in the model."
        ):
            handlers_utils.validate_signature(model, meta, [], None, predict_fun)

    def test_validate_signature_with_only_sample_data(self) -> None:
        # metadata with no signatures
        meta = model_meta.ModelMetadata(name="name", env=model_env.ModelEnv(), model_type="custom")

        # sample input data
        data = {"feature1": [10, 20, 30], "feature2": [10.0, 20.0, 30.0], "feature3": ["a", "b", "c"]}
        sample_data = pd.DataFrame(data)

        model = catboost.CatBoostRegressor()

        # mocking predict function calls
        predict_fun = mock.MagicMock()
        predict_fun.side_effect = lambda x, y: pd.DataFrame({"output1": [1, 2, 3]})

        meta = handlers_utils.validate_signature(model, meta, ["predict"], sample_data, predict_fun)
        self.assertEqual(
            1,
            len(meta.signatures.keys()),
        )
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature2"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="feature3"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output1")],
        )
        self.assertEqual(predict_sig, meta.signatures.get("predict"))

    def test_get_truncated_sample_data(self) -> None:

        # when data_size > 10 rows
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)))
        self.assertEqual(10, cast(pd.DataFrame, handlers_utils.get_truncated_sample_data(df, 10)).shape[0])

        # when data_size = 10 rows
        df = pd.DataFrame(np.random.randint(0, 100, size=(10, 3)))
        self.assertEqual(10, cast(pd.DataFrame, handlers_utils.get_truncated_sample_data(df, 10)).shape[0])

        # when data_size < 10 rows
        df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3)))
        self.assertEqual(5, cast(pd.DataFrame, handlers_utils.get_truncated_sample_data(df, 10)).shape[0])


if __name__ == "__main__":
    absltest.main()
