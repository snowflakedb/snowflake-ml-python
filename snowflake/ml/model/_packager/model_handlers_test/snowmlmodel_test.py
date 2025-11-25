import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import shap
from absl.testing import absltest
from sklearn import datasets
from sklearn.model_selection import RepeatedStratifiedKFold

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils
from snowflake.ml.modeling.impute import SimpleImputer  # type: ignore[attr-defined]
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.modeling.model_selection import (
    GridSearchCV,  # type: ignore[attr-defined]
)
from snowflake.ml.modeling.pipeline import Pipeline  # type: ignore[attr-defined]
from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    OneHotEncoder,
    StandardScaler,
)
from snowflake.ml.modeling.xgboost import (  # type: ignore[attr-defined]
    XGBClassifier,
    XGBRegressor,
)


class SnowMLModelHandlerTest(absltest.TestCase):
    def test_snowml_all_input_no_explain(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}

            with self.assertWarnsRegex(UserWarning, "Model signature will automatically be inferred during fitting"):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regr,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": False},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
                options={"enable_explainability": False},
                task=model_types.Task.TABULAR_REGRESSION,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])
                # correctly set when specified
                self.assertEqual(pk.meta.task, model_types.Task.TABULAR_REGRESSION)

    def test_snowml_signature_partial_input(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
                options={"enable_explainability": False},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_signature_drop_input_cols(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(
            input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS, drop_input_cols=True
        )
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
                options={"enable_explainability": False},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_xgboost_explain_default(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS, n_jobs=1)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        explanations = shap.TreeExplainer(regr.to_xgboost())(df[INPUT_COLUMNS]).values

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
                task=model_types.Task.TABULAR_BINARY_CLASSIFICATION,  # incorrect type but should be inferred properly
                options=model_types.SNOWModelSaveOptions(enable_explainability=True),
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
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])
                np.testing.assert_allclose(explanations, explain_method(df[INPUT_COLUMNS]).values)
                # correctly set even when incorrect
                self.assertEqual(pk.meta.task, model_types.Task.TABULAR_REGRESSION)

    def test_snowml_all_input_with_explain(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]
        explanations = shap.Explainer(regr.to_sklearn(), df[INPUT_COLUMNS])(df[INPUT_COLUMNS]).values

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                sample_input_data=df[INPUT_COLUMNS],
                metadata={"author": "halu", "version": "1"},
                options={"enable_explainability": True},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])
                explain_method = getattr(pk.model, "explain", None)
                assert callable(explain_method)
                np.testing.assert_allclose(explanations, explain_method(df[INPUT_COLUMNS]).values)

    def test_snowml_preprocessing(self) -> None:
        input_cols = ["F"]
        output_cols = ["F"]
        pandas_df = pd.DataFrame(data={"F": [1, 2, 3]})

        scaler = StandardScaler(input_cols=input_cols, output_cols=output_cols)
        scaler.fit(pandas_df)

        predictions = scaler.transform(pandas_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=scaler,
                sample_input_data=pandas_df,
                metadata={"author": "halu", "version": "1"},
                options=model_types.SNOWModelSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "transform", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predictions, predict_method(pandas_df))

    def test_snowml_pipeline_explainability(self) -> None:
        df = pd.DataFrame(
            {
                "num1": np.random.rand(100),
                "num2": np.random.rand(100),
                "num3": np.random.rand(100),
                "num4": np.random.rand(100),
                "cat1": np.random.choice(["a", "b", "c"], 100),
                "cat2": np.random.choice(["x", "y", "z"], 100),
                "label": np.random.randint(0, 2, 100),
            }
        )

        # No Snowflake connection, disable telemetry for this test
        Pipeline._send_pipeline_configuration_telemetry = lambda _: None

        NUMERIC_FEATURE_COLS = ["num1", "num2", "num3", "num4"]
        CATEGORICAL_COLS = ["cat1", "cat2"]
        LABEL_COLS = ["label"]

        pre_processing = Pipeline(
            steps=[
                ("impute", SimpleImputer(input_cols=NUMERIC_FEATURE_COLS, output_cols=NUMERIC_FEATURE_COLS)),
                ("scaler", StandardScaler(input_cols=NUMERIC_FEATURE_COLS, output_cols=NUMERIC_FEATURE_COLS)),
                (
                    "onehot",
                    OneHotEncoder(
                        input_cols=CATEGORICAL_COLS,
                        output_cols=CATEGORICAL_COLS,
                        drop_input_cols=True,
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )
        model_pipeline = Pipeline(
            steps=[
                ("pre_processing", pre_processing),
                (
                    "grid_search_xgb",
                    GridSearchCV(
                        estimator=XGBClassifier(n_jobs=1),
                        param_grid={
                            "n_estimators": [150],
                            "learning_rate": [0.1],
                            "max_depth": [5],
                            "subsample": [1],
                        },
                        n_jobs=1,
                        scoring="accuracy",
                        label_cols=LABEL_COLS,
                        output_cols=["OUTPUT_LABEL"],
                        drop_input_cols=True,
                        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1),
                    ),
                ),
            ]
        )
        model_pipeline.fit(df)
        predictions = model_pipeline.predict(df)

        transformed_df = df
        for _, step in model_pipeline.steps[:-1]:
            transformed_df = step.transform(transformed_df)
        predictor = model_pipeline.steps[-1][1]
        explainer = shap.Explainer(predictor.predict_proba, transformed_df)
        explanations = handlers_utils.convert_explanations_to_2D_df(predictor, explainer(transformed_df).values)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model_pipeline,
                sample_input_data=df,
                metadata={"author": "halu", "version": "1"},
                options={"enable_explainability": True},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predictions, predict_method(df))
            explain_method = getattr(pk.model, "explain", None)
            assert callable(explain_method)
            np.testing.assert_allclose(explanations, explain_method(df), atol=0.05)
            # explain method should have the output match transformed data, e.g. onehot-encoded categorical data
            # 11 = 4 numerical + 3 onehot * 2 categorical + 1 label
            assert len(pk.meta.signatures["explain"].outputs) == 11


if __name__ == "__main__":
    absltest.main()
