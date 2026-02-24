from importlib import metadata as importlib_metadata

import mlflow
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal import env
from snowflake.ml.model._signatures import numpy_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryMLFlowModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_mlflow_model_deploy_sklearn_df(
        self,
    ) -> None:
        db = datasets.load_diabetes(as_frame=True)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
                metadata={"author": "halu", "version": "1"},
                conda_env={
                    "dependencies": [f"python=={env.PYTHON_VERSION}"]
                    + list(
                        map(
                            lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                            [
                                "mlflow",
                                "cloudpickle",
                                "numpy",
                                "scikit-learn",
                                "scipy",
                                "typing-extensions",
                            ],
                        )
                    ),
                    "name": "mlflow-env",
                },
            )

            run_id = run.info.run_id

        self._test_registry_model(
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
            prediction_assert_fns={
                "predict": (
                    X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(predictions, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"relax_version": False},
        )

    def test_mlflow_model_deploy_sklearn(self) -> None:
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
                metadata={"author": "halu", "version": "1"},
                conda_env={
                    "dependencies": [f"python=={env.PYTHON_VERSION}"]
                    + list(
                        map(
                            lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                            [
                                "mlflow",
                                "cloudpickle",
                                "numpy",
                                "scikit-learn",
                                "scipy",
                                "typing-extensions",
                            ],
                        )
                    ),
                    "name": "mlflow-env",
                },
            )

            run_id = run.info.run_id

        X_test_df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df([X_test])

        self._test_registry_model(
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
            prediction_assert_fns={
                "predict": (
                    X_test_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(predictions, columns=res.columns),
                        check_dtype=False,
                    ),
                ),
            },
            options={"relax_version": False},
        )

    def test_echo_model(self) -> None:
        class EchoModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):  # type: ignore[no-untyped-def]
                return model_input

        with mlflow.start_run() as run:
            python_model = EchoModel()

            input_df = pd.DataFrame(
                {
                    "col_str": ["hello", "world"],
                    "col_int": [1, 2],
                    "col_float": [1.5, 2.5],
                    "col_list": [[1, 2, 3], [4, 5, 6]],
                    "col_list_of_lists": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    "col_dict": [{"a": 1}, {"b": 2}],
                    "col_list_of_dicts": [[{"a": 1}, {"b": 2}], [{"c": 3}, {"d": 4}]],
                }
            )

            signature = mlflow.models.infer_signature(
                model_input=input_df,
                model_output=input_df,
            )

            mlflow.pyfunc.log_model(
                artifact_path="echo_model",
                python_model=python_model,
                input_example=input_df,
                conda_env={
                    "dependencies": [f"python=={env.PYTHON_VERSION}"]
                    + list(
                        map(
                            lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                            [
                                "mlflow",
                                "cloudpickle",
                                "pandas",
                                "numpy",
                            ],
                        )
                    ),
                    "name": "mlflow-env",
                },
                signature=signature,
            )

            run_id = run.info.run_id

        def check_echo_result(res: pd.DataFrame) -> None:
            assert list(res.columns) == list(
                input_df.columns
            ), f"Column mismatch: {list(res.columns)} != {list(input_df.columns)}"
            assert len(res) == len(input_df), f"Row count mismatch: {len(res)} != {len(input_df)}"
            for col in input_df.columns:
                for i in range(len(input_df)):
                    expected = input_df[col].iloc[i]
                    actual = res[col].iloc[i]
                    assert actual == expected, f"Mismatch at [{i}, {col}]: {actual!r} != {expected!r}"

        self._test_registry_model(
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/echo_model"),
            prediction_assert_fns={
                "predict": (
                    input_df,
                    check_echo_result,
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
