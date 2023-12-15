from importlib import metadata as importlib_metadata

import mlflow
import numpy as np
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
                    lambda res: np.testing.assert_allclose(np.expand_dims(predictions, axis=1), res.to_numpy()),
                ),
            },
        )

    def test_mlflow_model_deploy_sklearn(
        self,
    ) -> None:
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
                    lambda res: np.testing.assert_allclose(np.expand_dims(predictions, axis=1), res.to_numpy()),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
