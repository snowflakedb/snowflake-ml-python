import uuid
from importlib import metadata as importlib_metadata

import mlflow
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal import env
from snowflake.ml.model._signatures import numpy_handler, snowpark_handler
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestMlflowBatchInferenceInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"cpu_requests": "4", "memory_requests": "4Gi"},
        # todo: add tests for gpu
    )
    @absltest.skip("Test disabled for https://snowflakecomputing.atlassian.net/browse/SNOW-2369887")
    def test_mlflow(
        self,
        cpu_requests: str,
        memory_requests: str,
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
        # Convert integer column names to strings for Snowpark compatibility
        X_test_df.columns = X_test_df.columns.astype(str)
        x_df_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self.session,
            X_test_df,
        )

        # Rename columns using withColumnRenamed in a loop
        for old_name in x_df_sp.columns:
            clean_old_name = old_name.replace('"', "")
            new_name = f'"input_feature_{clean_old_name}"'
            x_df_sp = x_df_sp.withColumnRenamed(old_name, new_name)

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
            input_spec=x_df_sp,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            num_workers=1,
            service_name=f"batch_inference_{name}",
            replicas=2,
            function_name="predict",
        )


if __name__ == "__main__":
    absltest.main()
