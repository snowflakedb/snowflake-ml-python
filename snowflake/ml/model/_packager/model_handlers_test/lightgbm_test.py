import os
import tempfile
import warnings
from typing import Any
from unittest import mock

import lightgbm
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager


class LightGBMHandlerTest(absltest.TestCase):
    def test_lightgbm_booster_explainability_disabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train(
            {"objective": "binary", "num_threads": 1}, lightgbm.Dataset(cal_X_train, label=cal_y_train)
        )
        y_pred = regressor.predict(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regressor,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.LGBMModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regressor,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, lightgbm.Booster)
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
                model=regressor,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, lightgbm.Booster)
            np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
            self.assertEqual(pk.meta.task, model_types.Task.TABULAR_BINARY_CLASSIFICATION)

    def test_lightgbm_booster_explainablity_enabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train(
            {"objective": "binary", "num_threads": 1}, lightgbm.Dataset(cal_X_train, label=cal_y_train)
        )
        y_pred = regressor.predict(cal_X_test)
        explanations: npt.NDArray[Any] = shap.TreeExplainer(regressor)(cal_X_test).values
        if explanations.ndim == 3 and explanations.shape[2] == 2:
            explanations = np.apply_along_axis(lambda arr: arr[1], -1, explanations)

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}

            # check for warnings if sample_input_data is not provided while saving the model
            with self.assertWarnsRegex(
                UserWarning, "sample_input_data should be provided for better explainability results"
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regressor,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.LGBMModelSaveOptions(),
                )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, lightgbm.Booster)
                np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))
                np.testing.assert_allclose(explain_method(cal_X_test), explanations)

            # test calling saving background_data when sample_input_data is present
            with mock.patch(
                "snowflake.ml.model._packager.model_handlers._utils.save_background_data"
            ) as save_background_data:
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                    name="model1_no_sig",
                    model=regressor,
                    sample_input_data=cal_X_test,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.LGBMModelSaveOptions(),
                )
                save_background_data.assert_called_once()

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta

            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(explain_method(cal_X_test), explanations)

    def test_lightgbm_classifier_explainability_disabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier(n_jobs=1)
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
                    options=model_types.LGBMModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, lightgbm.LGBMClassifier)
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
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, lightgbm.LGBMClassifier)
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

    def test_lightgbm_classifier_explainablity_enabled(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, _ = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier(n_jobs=1)
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)
        explanations: npt.NDArray[Any] = shap.TreeExplainer(classifier)(cal_X_test).values
        if explanations.ndim == 3 and explanations.shape[2] == 2:
            explanations = np.apply_along_axis(lambda arr: arr[1], -1, explanations)

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.LGBMModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, lightgbm.LGBMClassifier)
                np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
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
                options=model_types.LGBMModelSaveOptions(),
            )

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

            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(explain_method(cal_X_test), explanations)


class LightGBMHandlerParamsTest(absltest.TestCase):
    """Param forwarding for LightGBM. Covers shape-preserving documented kwargs only."""

    @staticmethod
    def _load_data() -> tuple[pd.DataFrame, pd.Series]:
        cal_data = datasets.load_breast_cancer()
        # Underscored names; LightGBM normalizes whitespace and validate_features=True is strict.
        feature_names = [name.replace(" ", "_") for name in cal_data.feature_names]
        cal_X = pd.DataFrame(cal_data.data, columns=feature_names)
        cal_y = pd.Series(cal_data.target)
        return cal_X, cal_y

    @classmethod
    def _train_booster(cls, num_boost_round: int = 20) -> tuple[lightgbm.Booster, pd.DataFrame, pd.DataFrame]:
        cal_X, cal_y = cls._load_data()
        cal_X_train, cal_X_test, cal_y_train, _ = model_selection.train_test_split(cal_X, cal_y, random_state=0)
        booster = lightgbm.train(
            {"objective": "binary", "num_threads": 1},
            lightgbm.Dataset(cal_X_train, label=cal_y_train),
            num_boost_round=num_boost_round,
        )
        return booster, cal_X_train, cal_X_test

    @classmethod
    def _train_classifier(cls, n_estimators: int = 20) -> tuple[lightgbm.LGBMClassifier, pd.DataFrame, pd.DataFrame]:
        cal_X, cal_y = cls._load_data()
        cal_X_train, cal_X_test, cal_y_train, _ = model_selection.train_test_split(cal_X, cal_y, random_state=0)
        classifier = lightgbm.LGBMClassifier(n_jobs=1, n_estimators=n_estimators, random_state=0)
        classifier.fit(cal_X_train, cal_y_train)
        return classifier, cal_X_train, cal_X_test

    def test_lightgbm_booster_documented_kwargs_forwarded(self) -> None:
        """Shape-preserving kwargs flow to Booster.predict."""
        booster, _, cal_X_test = self._train_booster()
        y_pred = booster.predict(cal_X_test)

        # Declare a ParamSpec for every documented kwarg the user might call with.
        sig = model_signature.ModelSignature(
            inputs=model_signature.infer_signature(cal_X_test, y_pred).inputs,
            outputs=model_signature.infer_signature(cal_X_test, y_pred).outputs,
            params=[
                model_signature.ParamSpec("start_iteration", model_signature.DataType.INT64, default_value=0),
                model_signature.ParamSpec("num_iteration", model_signature.DataType.INT64, default_value=0),
                model_signature.ParamSpec("raw_score", model_signature.DataType.BOOL, default_value=False),
                model_signature.ParamSpec("validate_features", model_signature.DataType.BOOL, default_value=False),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=booster,
                signatures={"predict": sig},
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Default (no kwargs) matches Booster.predict(X).
            res_default = predict_method(cal_X_test)
            np.testing.assert_allclose(res_default.to_numpy().flatten(), y_pred)

            kwarg_cases: list[tuple[str, Any]] = [
                ("start_iteration", 5),
                ("num_iteration", 3),
                ("raw_score", True),
                ("validate_features", True),
            ]
            for kwarg_name, kwarg_value in kwarg_cases:
                with self.subTest(kwarg=kwarg_name):
                    res = predict_method(cal_X_test, **{kwarg_name: kwarg_value})
                    expected = booster.predict(cal_X_test, **{kwarg_name: kwarg_value})
                    np.testing.assert_allclose(res.to_numpy().flatten(), expected)

            # Confirm at least one kwarg actually changes the output.
            res_few_trees = predict_method(cal_X_test, num_iteration=1)
            self.assertFalse(
                np.allclose(res_default.to_numpy(), res_few_trees.to_numpy()),
                "num_iteration=1 prediction should differ from the full-iteration default",
            )

    def test_lightgbm_classifier_documented_kwargs_forwarded(self) -> None:
        """Shape-preserving kwargs flow to LGBMClassifier.predict and predict_proba."""
        classifier, _, cal_X_test = self._train_classifier()
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)

        predict_params = [
            model_signature.ParamSpec("start_iteration", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("num_iteration", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("raw_score", model_signature.DataType.BOOL, default_value=False),
            model_signature.ParamSpec("validate_features", model_signature.DataType.BOOL, default_value=False),
        ]
        # raw_score changes predict_proba shape from (n,2) to (n,) for binary — skip on predict_proba.
        predict_proba_params = [
            model_signature.ParamSpec("start_iteration", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("num_iteration", model_signature.DataType.INT64, default_value=0),
            model_signature.ParamSpec("validate_features", model_signature.DataType.BOOL, default_value=False),
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
                params=predict_proba_params,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=sigs,
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            predict_proba_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            assert callable(predict_proba_method)

            np.testing.assert_allclose(predict_method(cal_X_test).to_numpy().flatten(), y_pred)
            np.testing.assert_allclose(predict_proba_method(cal_X_test).to_numpy(), y_pred_proba)

            predict_cases: list[tuple[str, Any]] = [
                ("start_iteration", 5),
                ("num_iteration", 3),
                ("raw_score", True),
                ("validate_features", True),
            ]
            for kwarg_name, kwarg_value in predict_cases:
                with self.subTest(method="predict", kwarg=kwarg_name):
                    res = predict_method(cal_X_test, **{kwarg_name: kwarg_value})
                    expected = classifier.predict(cal_X_test, **{kwarg_name: kwarg_value})
                    np.testing.assert_allclose(res.to_numpy().flatten(), expected)

            for kwarg_name, kwarg_value in predict_cases:
                if kwarg_name == "raw_score":
                    continue
                with self.subTest(method="predict_proba", kwarg=kwarg_name):
                    res = predict_proba_method(cal_X_test, **{kwarg_name: kwarg_value})
                    expected = classifier.predict_proba(cal_X_test, **{kwarg_name: kwarg_value})
                    np.testing.assert_allclose(res.to_numpy(), expected)

            self.assertFalse(
                np.allclose(
                    predict_proba_method(cal_X_test).to_numpy(),
                    predict_proba_method(cal_X_test, num_iteration=1).to_numpy(),
                ),
                "num_iteration=1 predict_proba should differ from the full-iteration default",
            )

    def test_lightgbm_handler_class_forwards_kwargs_unconditionally(self) -> None:
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
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Known kwarg, undeclared in signature — value change proves it reached LightGBM.
            res_known = predict_method(cal_X_test, num_iteration=1)
            np.testing.assert_allclose(res_known.to_numpy().flatten(), classifier.predict(cal_X_test, num_iteration=1))

            # Unknown kwarg — LightGBM's **kwargs silently absorbs it; output equals default.
            res_unknown = predict_method(cal_X_test, totally_made_up_kwarg=42)
            np.testing.assert_allclose(res_unknown.to_numpy().flatten(), classifier.predict(cal_X_test))

    def test_lightgbm_kwarg_errors_propagate(self) -> None:
        """LightGBM errors on bad kwargs reach the caller."""
        classifier, _, cal_X_test = self._train_classifier()
        y_pred = classifier.predict(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            sig = model_signature.infer_signature(cal_X_test, y_pred)
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures={"predict": sig},
                options=model_types.LGBMModelSaveOptions(enable_explainability=False),
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # X passed twice — predict raises.
            with self.assertRaises(TypeError):
                predict_method(cal_X_test, X=cal_X_test)


if __name__ == "__main__":
    absltest.main()
