import os
import tempfile
import warnings
from typing import Optional
from unittest import mock

import numpy as np
import pandas as pd
import shap
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, multioutput
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers.sklearn import (
    _apply_transforms_up_to_last_step,
    _unpack_container_runtime_pipeline,
)


class SKLearnHandlerTest(absltest.TestCase):
    def test_skl_multiple_output_proba_no_explain(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict_proba": model_signature.infer_signature(iris_X_df, model.predict_proba(iris_X_df))}
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
                options=model_types.SKLModelSaveOptions(),
            )

            orig_res = model.predict_proba(iris_X_df[-10:])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, multioutput.MultiOutputClassifier)
            loaded_res = pk.model.predict_proba(iris_X_df[-10:])
            np.testing.assert_allclose(np.hstack(orig_res), np.hstack(loaded_res))

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            udf_res = predict_method(iris_X_df[-10:])
            np.testing.assert_allclose(
                np.hstack(orig_res), np.hstack([np.array(udf_res[col].to_list()) for col in udf_res])
            )

            with self.assertRaises(ValueError):
                model_packager.ModelPackager(local_dir_path=os.path.join(tmpdir, "model1_no_sig_bad")).save(
                    name="model1_no_sig_bad",
                    model=model,
                    sample_input_data=iris_X_df,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions(
                        {
                            "target_methods": ["random"],
                        }
                    ),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=model,
                sample_input_data=iris_X_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(
                    {
                        "enable_explainability": False,
                    }
                ),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, multioutput.MultiOutputClassifier)
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])), np.hstack(pk.model.predict_proba(iris_X_df[-10:]))
            )
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), pk.model.predict(iris_X_df[-10:]))
            self.assertEqual(s["predict_proba"], pk.meta.signatures["predict_proba"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta

            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            udf_res = predict_method(iris_X_df[-10:])
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])),
                np.hstack([np.array(udf_res[col].to_list()) for col in udf_res]),
            )

            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), predict_method(iris_X_df[-10:]).to_numpy())

    def test_skl_unsupported_explain(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict_proba": model_signature.infer_signature(iris_X_df, model.predict_proba(iris_X_df))}
            with self.assertRaisesRegex(
                ValueError,
                "Sample input data is required to enable explainability.",
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    conda_dependencies=["scikit-learn"],
                    options=model_types.SKLModelSaveOptions(enable_explainability=True),
                )

            with self.assertRaisesRegex(
                ValueError, "Explainability for this model is not supported. Please set `enable_explainability=False`"
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                    name="model1_no_sig",
                    model=model,
                    sample_input_data=iris_X_df,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions(enable_explainability=True),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=model,
                sample_input_data=iris_X_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, multioutput.MultiOutputClassifier)
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])), np.hstack(pk.model.predict_proba(iris_X_df[-10:]))
            )
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), pk.model.predict(iris_X_df[-10:]))
            self.assertEqual(s["predict_proba"], pk.meta.signatures["predict_proba"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta

            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            udf_res = predict_method(iris_X_df[-10:])
            np.testing.assert_allclose(
                np.hstack(model.predict_proba(iris_X_df[-10:])),
                np.hstack([np.array(udf_res[col].to_list()) for col in udf_res]),
            )

            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(model.predict(iris_X_df[-10:]), predict_method(iris_X_df[-10:]).to_numpy())

            assert not hasattr(pk.model, "explain")

    def test_skl_no_explain(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(iris_X_df, regr.predict(iris_X_df))}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions(enable_explainability=False),
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(enable_explainability=False),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, linear_model.LinearRegression)
                np.testing.assert_allclose(np.array([-0.08254936]), pk.model.predict(iris_X_df[:1]))

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=regr,
                sample_input_data=iris_X_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(enable_explainability=False),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, linear_model.LinearRegression)
            np.testing.assert_allclose(np.array([-0.08254936]), pk.model.predict(iris_X_df[:1]))
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))

    def test_skl_explain(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)

        explanations = shap.Explainer(regr, iris_X_df)(iris_X_df).values
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(iris_X_df, regr.predict(iris_X_df))}
            with self.assertRaisesRegex(
                ValueError,
                "Sample input data is required to enable explainability.",
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regr,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions(enable_explainability=True),
                )

            # test calling saving background_data when sample_input_data is present
            with mock.patch(
                "snowflake.ml.model._packager.model_handlers._utils.save_background_data"
            ) as save_background_data:
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1_no_sig",
                    model=regr,
                    sample_input_data=iris_X_df,
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SKLModelSaveOptions(),
                )
                save_background_data.assert_called_once()

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1_no_sig",
                model=regr,
                sample_input_data=iris_X_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))
                np.testing.assert_allclose(explain_method(iris_X_df), explanations)

    def test_skl_explain_with_np(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)

        explanations = shap.Explainer(regr, iris_X_df)(iris_X_df).values
        with tempfile.TemporaryDirectory() as tmpdir:

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1_no_sig",
                model=regr,
                sample_input_data=iris_X_df.values,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))
                np.testing.assert_allclose(explain_method(iris_X_df), explanations)

    def test_skl_no_default_explain_without_background_data(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(iris_X_df, regr.predict(iris_X_df))}

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                self.assertEqual(explain_method, None)

    def test_skl_default_explain_sklearn_pipeline(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        pipe = Pipeline([("regr", regr)])

        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        pipe.fit(iris_X_df, iris_y)

        explanations = shap.Explainer(regr, iris_X_df)(iris_X_df).values
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=pipe,
                sample_input_data=iris_X_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X_df[:1]))
                np.testing.assert_allclose(explain_method(iris_X_df), explanations)

    def test_skl_default_explain_sklearn_pipeline_with_np(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        pipe = Pipeline([("regr", regr)])
        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        pipe.fit(iris_X, iris_y)

        explanations = shap.Explainer(regr, iris_X)(iris_X).values
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=pipe,
                sample_input_data=iris_X,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(predict_method)
                assert callable(explain_method)
                np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(iris_X[:1]))
                np.testing.assert_allclose(explain_method(iris_X), explanations)

    def test_skl_object_no_explainable_method(self) -> None:
        iris_X, _ = datasets.load_iris(return_X_y=True)
        scaler = StandardScaler()
        scaler.fit(iris_X)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "transform_model")).save(
                name="transform_model",
                model=scaler,
                sample_input_data=iris_X,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "transform_model"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                transform_method = getattr(pk.model, "transform", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(transform_method)
                self.assertEqual(explain_method, None)

    def test_skl_object_no_explainable_method_set_enable(self) -> None:
        iris_X, _ = datasets.load_iris(return_X_y=True)
        scaler = StandardScaler()
        scaler.fit(iris_X)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError,
                "Signature for target method None is missing or no method to explain.",
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="transform_model",
                    model=scaler,
                    sample_input_data=iris_X,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

    def test_skl_pipeline_object_no_explainable_method(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        scaler = StandardScaler()
        pipe = Pipeline([("scaler", scaler)])
        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        pipe.fit(iris_X, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "pipeline_transform_model")).save(
                name="pipeline_transform_model",
                model=pipe,
                sample_input_data=iris_X,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "pipeline_transform_model"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                transform_method = getattr(pk.model, "transform", None)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(transform_method)
                self.assertEqual(explain_method, None)

    def test_skl_with_cr_estimator(self) -> None:
        class SecondMockEstimator:
            ...

        class MockEstimator:
            @property
            def _sklearn_estimator(self) -> SecondMockEstimator:
                return SecondMockEstimator()

        skl_pipeline = Pipeline(steps=[("mock", MockEstimator())])
        oss_pipeline = _unpack_container_runtime_pipeline(skl_pipeline)

        assert isinstance(oss_pipeline.steps[0][1], SecondMockEstimator)

    def test_skl_pipeline_categorical_explain(self) -> None:
        input_cols = ["category_abc", "category_xyz"]
        output_cols = ["prediction"]
        pandas_df = pd.DataFrame(
            data={
                "category_abc": ["a", "b", "c"],
                "category_xyz": ["x", "y", "z"],
                "prediction": [1, 2, 3],
            }
        )
        X = pandas_df[input_cols]
        y = pandas_df[output_cols]

        expected_explain_output_cols = [
            "category_abc_a_explanation",
            "category_abc_b_explanation",
            "category_abc_c_explanation",
            "category_xyz_x_explanation",
            "category_xyz_y_explanation",
            "category_xyz_z_explanation",
        ]

        regr = linear_model.LinearRegression()
        onehot = OneHotEncoder()
        pipe = Pipeline([("onehot", onehot), ("regr", regr)])
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "pipeline_onehot_model")).save(
                name="pipeline_onehot_model",
                model=pipe,
                sample_input_data=X,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SKLModelSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "pipeline_onehot_model"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                explain_method = getattr(pk.model, "explain", None)
                assert callable(explain_method)

                transformed = onehot.transform(X)
                expected_explanations = shap.Explainer(regr, transformed)(transformed).values
                np.testing.assert_allclose(pk.model.explain(X), expected_explanations)  # type: ignore[union-attr]

                assert "explain" in pk.meta.signatures
                explain_signature = pk.meta.signatures["explain"]
                for name in input_cols:
                    self.assertIn(
                        model_signature.FeatureSpec(name, model_signature.DataType.STRING), explain_signature.inputs
                    )
                for name in expected_explain_output_cols:
                    self.assertIn(
                        model_signature.FeatureSpec(name, model_signature.DataType.DOUBLE), explain_signature.outputs
                    )


class SklearnHelperFunctionsTest(absltest.TestCase):
    def test_apply_transforms_up_to_last_step(self) -> None:
        class MockEstimator:
            def predict(self, X: model_types.SupportedDataType) -> model_types.SupportedDataType:
                return X / 2  # type: ignore[operator]

        class MockTransformer:
            def transform(self, X: model_types.SupportedDataType) -> model_types.SupportedDataType:
                return X + 1  # type: ignore[operator]

            def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> Optional[list[str]]:
                if input_features is None:
                    return None
                return [f"{f}_transformed" for f in input_features]

        X = np.array([[1, 2, 3], [4, 5, 6]])

        pipe = Pipeline(steps=[("transformer", MockTransformer()), ("estimator", MockEstimator())])
        transformed = _apply_transforms_up_to_last_step(pipe, X, ["a", "b", "c"])
        self.assertEqual(transformed.columns.tolist(), ["a_transformed", "b_transformed", "c_transformed"])
        np.testing.assert_allclose(transformed, X + 1)

        class BadTransformer:
            pass

        bad_pipe = Pipeline(steps=[("bad_transformer", BadTransformer()), ("estimator", MockEstimator())])
        with self.assertRaisesRegex(
            ValueError,
            "Step 'bad_transformer' does not have a 'transform' method.",
        ):
            _apply_transforms_up_to_last_step(bad_pipe, X)


if __name__ == "__main__":
    absltest.main()
