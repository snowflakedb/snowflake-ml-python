from importlib import metadata as importlib_metadata

import mlflow
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal import env
from snowflake.ml.model._signatures import numpy_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryMLFlowModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_mlflow_model_deploy_sklearn_df(
        self,
        registry_test_fn: str,
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

        getattr(self, registry_test_fn)(
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


if __name__ == "__main__":
    absltest.main()
