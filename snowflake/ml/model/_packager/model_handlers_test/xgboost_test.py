import os
import tempfile
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import shap
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers_test import test_utils


class XgboostHandlerTest(absltest.TestCase):
    def test_xgb_booster_explainability_disabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        classifier = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = classifier.predict(xgboost.DMatrix(data=cal_X_test))
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=classifier,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.XGBModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=False),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, xgboost.Booster)
                np.testing.assert_allclose(pk.model.predict(xgboost.DMatrix(data=cal_X_test)), y_pred)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
                # test task is set even without explain
                self.assertEqual(pk.meta.task, model_types.Task.TABULAR_BINARY_CLASSIFICATION)

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, xgboost.Booster)
            np.testing.assert_allclose(pk.model.predict(xgboost.DMatrix(data=cal_X_test)), y_pred)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

    def test_xgb_explainability_disabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        classifier = xgboost.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
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
                    options=model_types.XGBModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.XGBModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, xgboost.XGBClassifier)
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
                options=model_types.XGBModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, xgboost.XGBClassifier)
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

    def test_xgb_explainablity_enabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        classifier = xgboost.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)
        explanations = shap.TreeExplainer(classifier)(cal_X_test).values
        with tempfile.TemporaryDirectory() as tmpdir:

            # check for warnings if sample_input_data is not provided while saving the model
            with self.assertWarnsRegex(
                UserWarning, "sample_input_data should be provided for better explainability results"
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=classifier,
                    signatures={"predict": model_signature.infer_signature(cal_X_test, y_pred)},
                    metadata={"author": "halu", "version": "1"},
                    task=model_types.Task.UNKNOWN,
                    options=model_types.XGBModelSaveOptions(),
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
                np.testing.assert_allclose(explain_method(cal_X_test), explanations)

            with mock.patch(
                "snowflake.ml.model._packager.model_handlers._utils.save_background_data"
            ) as save_background_data:
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                    name="model1_no_sig",
                    model=classifier,
                    sample_input_data=cal_X_test,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.XGBModelSaveOptions(),
                )
                save_background_data.assert_called_once()

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
            assert pk.meta
            # correctly inferred even when unknown
            self.assertEqual(pk.meta.task, model_types.Task.TABULAR_BINARY_CLASSIFICATION)

    def test_xgb_explainablity_multiclass(self) -> None:
        cal_data = datasets.load_iris()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        classifier = xgboost.XGBClassifier(reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        explanations = shap.TreeExplainer(classifier)(cal_X_test).values
        with tempfile.TemporaryDirectory() as tmpdir:

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures={"predict": model_signature.infer_signature(cal_X_test, y_pred)},
                metadata={"author": "halu", "version": "1"},
                options=model_types.XGBModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
                np.testing.assert_allclose(
                    test_utils.convert2D_json_to_3D(explain_method(cal_X_test).to_numpy()), explanations
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.XGBModelSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(
                test_utils.convert2D_json_to_3D(explain_method(cal_X_test).to_numpy()), explanations
            )


class XgboostVersionHandlingTest(absltest.TestCase):
    """Tests for XGBoost version-specific handling."""

    def test_gpu_params_xgboost_before_3_1_0(self) -> None:
        """Test that XGBoost < 3.1.0 uses legacy gpu_hist tree_method."""
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = xgboost.XGBClassifier(n_estimators=10, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "test", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=False),
            )

            # Test with XGBoost version < 3.1.0
            with mock.patch.object(xgboost, "__version__", "2.1.4"):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(options={"use_gpu": True})

                assert pk.model is not None
                assert isinstance(pk.model, xgboost.XGBModel)
                # For XGBModel, get_params() returns the parameters
                params = pk.model.get_params()
                self.assertEqual(params.get("tree_method"), "gpu_hist")
                self.assertEqual(params.get("predictor"), "gpu_predictor")

    def test_gpu_params_xgboost_3_1_0_and_later(self) -> None:
        """Test that XGBoost >= 3.1.0 uses device=cuda for GPU acceleration."""
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = xgboost.XGBClassifier(n_estimators=10, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "test", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=False),
            )

            # Test with XGBoost version >= 3.1.0
            with mock.patch.object(xgboost, "__version__", "3.1.0"):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(options={"use_gpu": True})

                assert pk.model is not None
                assert isinstance(pk.model, xgboost.XGBModel)
                params = pk.model.get_params()
                self.assertEqual(params.get("tree_method"), "hist")
                self.assertEqual(params.get("device"), "cuda")

    def test_shap_incompatible_with_xgboost_3_1_raises_error(self) -> None:
        """Test that SHAP < 0.50.0 with XGBoost >= 3.1.0 raises RuntimeError."""
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = xgboost.XGBClassifier(n_estimators=10, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "test", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=True),
            )

            # Test with XGBoost >= 3.1.0 and SHAP < 0.50.0
            with mock.patch.object(xgboost, "__version__", "3.1.0"):
                with mock.patch.object(shap, "__version__", "0.46.0"):
                    pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                    pk.load(as_custom_model=True)

                    explain_method = getattr(pk.model, "explain", None)
                    assert callable(explain_method)

                    with self.assertRaises(RuntimeError) as context:
                        explain_method(cal_X_test)

                    self.assertIn("SHAP version 0.46.0 is incompatible", str(context.exception))
                    self.assertIn("XGBoost version 3.1.0", str(context.exception))
                    self.assertIn("SHAP >= 0.50.0", str(context.exception))

    def test_shap_compatible_with_xgboost_before_3_1(self) -> None:
        """Test that any SHAP version works with XGBoost < 3.1.0."""
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = xgboost.XGBClassifier(n_estimators=10, max_depth=3, n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)
        expected_explanations = shap.TreeExplainer(classifier)(cal_X_test).values

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "test", "version": "1"},
                options=model_types.XGBModelSaveOptions(enable_explainability=True),
            )

            # Test with XGBoost < 3.1.0 and old SHAP - should work fine
            with mock.patch.object(xgboost, "__version__", "2.1.4"):
                with mock.patch.object(shap, "__version__", "0.46.0"):
                    pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                    pk.load(as_custom_model=True)

                    explain_method = getattr(pk.model, "explain", None)
                    assert callable(explain_method)

                    # Should not raise error
                    result = explain_method(cal_X_test)
                    np.testing.assert_allclose(result, expected_explanations)


if __name__ == "__main__":
    absltest.main()
