from typing import Optional

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, svm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistrySklearnModelDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters({"pip_requirements": None}, {"pip_requirements": ["scikit-learn"]})  # type: ignore[misc]
    def test_sklearn(self, pip_requirements: Optional[list[str]]) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        svc = svm.LinearSVC()
        svc.fit(iris_X, iris_y)

        self._test_registry_model_deployment(
            model=svc,
            sample_input_data=iris_X,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(svc.predict(iris_X), columns=res.columns),
                        rtol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            pip_requirements=pip_requirements,
            options={"enable_explainability": False},
        )

    @absltest.skipIf(True, "Temporarily quarantined until Inference server release")
    def test_sklearn_pipeline_wide_input_1(self) -> None:
        n_samples = 10
        n_features = 650
        data = {}

        for i in range(n_features):
            if i % 3 == 0:
                col_name = f'"z_feature_{i:03d}"'
                data[col_name] = np.random.randint(0, 10, n_samples)
            elif i % 3 == 1:
                col_name = f'"FEATURE_{i:03d}"'
                data[col_name] = np.random.choice(["A", "B", "C"], n_samples)
            else:
                col_name = f"a_feature_{i:03d}"
                data[col_name] = np.random.normal(0, 1, n_samples)

        train_df = pd.DataFrame(data)
        y_train = np.random.choice([0, 1], n_samples)

        string_cols = [f'"FEATURE_{i:03d}"' for i in range(n_features) if i % 3 == 1]
        numerical_cols = [col for col in train_df.columns if col not in string_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), string_cols),
                ("num", "passthrough", numerical_cols),
            ],
            remainder="drop",
        )

        classifier = RandomForestClassifier(n_estimators=10)
        model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

        model.fit(train_df, y_train)

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=train_df,
            prediction_assert_fns={
                "predict": (
                    train_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(model.predict(train_df), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    @absltest.skipIf(True, "Temporarily quarantined until Inference server release")
    def test_sklearn_pipeline_wide_input_2(self) -> None:
        n_samples = 10
        numerical_cols = [f"num_{i:03d}" for i in range(500)]
        data = {
            "z": np.random.choice(["type_a", "type_b", "type_c"], n_samples),
            "a": np.random.choice(["group_x", "group_y", "group_z"], n_samples),
            "b": np.random.choice(["status_1", "status_2"], n_samples),
            "y": np.random.choice(["code_1", "code_2"], n_samples),
        }

        for col in numerical_cols:
            data[col] = np.random.normal(0, 1, n_samples)

        train_df = pd.DataFrame(data)
        y_train = np.random.choice([0, 1], n_samples)

        multilabel_cols = ["z", "y"]
        ohe_cols_in = ["a", "b"]

        vectorizers = []
        for cn in multilabel_cols:
            vectorizers.append((f"{cn}_COUNTER", CountVectorizer(), cn))

        transformers = ColumnTransformer(
            transformers=[
                (
                    "OHE",
                    OneHotEncoder(
                        categories="auto",
                        max_categories=10,
                        handle_unknown="ignore",
                    ),
                    ohe_cols_in,
                ),
            ]
            + vectorizers,
            remainder="passthrough",
        )

        estimator = SGDClassifier()
        sklearn_pipeline = Pipeline([("preprocessing", transformers), ("estimator", estimator)])
        sklearn_pipeline.fit(train_df, y_train)

        self._test_registry_model_deployment(
            model=sklearn_pipeline,
            sample_input_data=train_df,
            prediction_assert_fns={
                "predict": (
                    train_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(sklearn_pipeline.predict(train_df), columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"enable_explainability": False},
        )


if __name__ == "__main__":
    absltest.main()
