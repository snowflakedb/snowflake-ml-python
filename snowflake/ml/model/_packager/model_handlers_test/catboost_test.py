import os
import tempfile
import warnings
from typing import Any
from unittest import mock

import catboost
import numpy as np
import pandas as pd
import shap
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers_test import test_utils


class CatBoostHandlerTest(absltest.TestCase):
    def test_catboost_classifier_explain_disabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=classifier,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, catboost.CatBoostClassifier)
                np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, catboost.CatBoost)
            np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
            np.testing.assert_allclose(pk.model.predict_proba(cal_X_test), y_pred_proba)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)
            self.assertEqual(pk.meta.task, model_types.Task.TABULAR_BINARY_CLASSIFICATION)

    def test_catboost_explainablity_enabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)
        explanations = shap.TreeExplainer(classifier)(cal_X_test).values

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}

            # check for warnings if sample_input_data is not provided while saving the model
            with self.assertWarnsRegex(
                UserWarning, "sample_input_data should be provided for better explainability results"
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_default_explain")).save(
                    name="model1_default_explain",
                    model=classifier,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.CatBoostModelSaveOptions(enable_explainability=True),
                )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_default_explain"))
                pk.load(as_custom_model=True)
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
                np.testing.assert_allclose(explain_method(cal_X_test), explanations)

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=True),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)
            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(explain_method(cal_X_test), explanations)

            # test calling saving background_data when sample_input_data is present
            with mock.patch(
                "snowflake.ml.model._packager.model_handlers._utils.save_background_data"
            ) as save_background_data:
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_explain_enabled")).save(
                    name="model1_no_sig_explain_enabled",
                    model=classifier,
                    sample_input_data=cal_X_test,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.CatBoostModelSaveOptions(enable_explainability=True),
                )
                save_background_data.assert_called_once()
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_explain_enabled"))
            pk.load(as_custom_model=True)
            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(explain_method(cal_X_test), explanations)

    def test_catboost_multiclass_explainablity_enabled(self) -> None:
        cal_data = datasets.load_iris()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)
        explanations = shap.TreeExplainer(classifier)(cal_X_test).values

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=True),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(predict_method(cal_X_test), y_pred)
                np.testing.assert_allclose(
                    test_utils.convert2D_json_to_3D(explain_method(cal_X_test).to_numpy()), explanations
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=True),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred)
            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)
            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(
                test_utils.convert2D_json_to_3D(explain_method(cal_X_test).to_numpy()), explanations
            )


class CatBoostHandlerParamsTest(absltest.TestCase):
    """Param forwarding for CatBoost. Covers shape-preserving documented kwargs only."""

    @staticmethod
    def _train_classifier(iterations: int = 10) -> tuple[catboost.CatBoostClassifier, pd.DataFrame, pd.DataFrame]:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, _ = model_selection.train_test_split(cal_X, cal_y, random_state=0)
        classifier = catboost.CatBoostClassifier(iterations=iterations, thread_count=1, verbose=False, random_seed=0)
        classifier.fit(cal_X_train, cal_y_train)
        return classifier, cal_X_train, cal_X_test

    def test_catboost_documented_kwargs_forwarded(self) -> None:
        """Shape-preserving kwargs flow to CatBoost.predict / predict_proba."""
        classifier, _, cal_X_test = self._train_classifier()
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)

        # prediction_type only on predict; on predict_proba it would change shape.
        common_params = [
            model_signature.ParamSpec("ntree_start", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("ntree_end", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("thread_count", model_signature.DataType.INT64, default_value=-1),
            model_signature.ParamSpec("verbose", model_signature.DataType.BOOL, default_value=False),
            model_signature.ParamSpec("task_type", model_signature.DataType.STRING, default_value="CPU"),
        ]
        predict_params = common_params + [
            model_signature.ParamSpec("prediction_type", model_signature.DataType.STRING, default_value="Class"),
        ]
        sigs = {
            "predict": model_signature.ModelSignature(
                inputs=model_signature.infer_signature(cal_X_test, y_pred).inputs,
                outputs=model_signature.infer_signature(cal_X_test, y_pred).outputs,
                params=predict_params,
            ),
            "predict_proba": model_signature.ModelSignature(
                inputs=model_signature.infer_signature(cal_X_test, y_pred_proba).inputs,
                outputs=model_signature.infer_signature(cal_X_test, y_pred_proba).outputs,
                params=common_params,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=sigs,
                options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            predict_proba_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            assert callable(predict_proba_method)

            # Default (no kwargs) matches direct calls on the raw model.
            np.testing.assert_allclose(predict_method(cal_X_test).to_numpy(), np.expand_dims(y_pred, axis=1))
            np.testing.assert_allclose(predict_proba_method(cal_X_test).to_numpy(), y_pred_proba)

            # ntree_end and prediction_type are value-changing; the rest are config-only but must
            # still be accepted without error.
            predict_cases: list[tuple[str, Any]] = [
                ("ntree_start", 0),
                ("ntree_end", 3),
                ("thread_count", 1),
                ("verbose", False),
                ("task_type", "CPU"),
                ("prediction_type", "RawFormulaVal"),
            ]
            for kwarg_name, kwarg_value in predict_cases:
                with self.subTest(method="predict", kwarg=kwarg_name):
                    res = predict_method(cal_X_test, **{kwarg_name: kwarg_value})
                    expected = np.asarray(classifier.predict(cal_X_test, **{kwarg_name: kwarg_value}))
                    if expected.ndim == 1:
                        expected = np.expand_dims(expected, axis=1)
                    np.testing.assert_allclose(res.to_numpy(), expected)

            for kwarg_name, kwarg_value in predict_cases:
                # prediction_type isn't on predict_proba's signature (output is always probabilities).
                if kwarg_name == "prediction_type":
                    continue
                with self.subTest(method="predict_proba", kwarg=kwarg_name):
                    res = predict_proba_method(cal_X_test, **{kwarg_name: kwarg_value})
                    expected = classifier.predict_proba(cal_X_test, **{kwarg_name: kwarg_value})
                    np.testing.assert_allclose(res.to_numpy(), expected)

            # Confirm ntree_end=3 actually changes the output.
            res_default = predict_proba_method(cal_X_test)
            res_few_trees = predict_proba_method(cal_X_test, ntree_end=3)
            self.assertFalse(
                np.allclose(res_default.to_numpy(), res_few_trees.to_numpy()),
                "ntree_end=3 predict_proba should differ from the full-iteration default",
            )

    def test_catboost_handler_class_forwards_kwargs_unconditionally(self) -> None:
        """Handler forwards kwargs without filtering by signature.params (handler-level only)."""
        # mv.run rejects undeclared params client-side; this test covers the Python layer only.
        classifier, _, cal_X_test = self._train_classifier()
        y_pred = classifier.predict(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            sig = model_signature.infer_signature(cal_X_test, y_pred)
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures={"predict": sig},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Known kwarg, undeclared in signature — value change proves it reached CatBoost.
            res = predict_method(cal_X_test, ntree_end=3)
            expected = np.expand_dims(classifier.predict(cal_X_test, ntree_end=3), axis=1)
            np.testing.assert_allclose(res.to_numpy(), expected)

    def test_catboost_unknown_kwarg_raises(self) -> None:
        """CatBoost rejects unknown kwargs."""
        classifier, _, cal_X_test = self._train_classifier()
        y_pred = classifier.predict(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            sig = model_signature.infer_signature(cal_X_test, y_pred)
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures={"predict": sig},
                options=model_types.CatBoostModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            with self.assertRaises(TypeError):
                predict_method(cal_X_test, totally_made_up_kwarg=42)


if __name__ == "__main__":
    absltest.main()
