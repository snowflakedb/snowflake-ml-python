import os
import tempfile
import uuid

import mlflow
import numpy as np
import pandas as pd
import torch
from absl.testing import absltest
from sklearn import datasets, ensemble, model_selection

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager


class MLFlowHandlerTest(absltest.TestCase):
    def test_mlflow_model(self) -> None:
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
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.8.13",
                        "pip<=23.0.1",
                        {
                            "pip": [
                                "mlflow<3,>=2.3",
                                "cloudpickle==2.0.0",
                                "numpy==1.23.4",
                                "psutil==5.9.0",
                                "scikit-learn==1.2.2",
                                "scipy==1.9.3",
                                "typing-extensions==4.5.0",
                            ]
                        },
                    ],
                    "name": "mlflow-env",
                },
                signature=signature,
                metadata={"author": "halu", "version": "1"},
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.save(name="model1", model=mlflow_pyfunc_model, options={"relax_version": False})
            assert pk.model
            assert pk.meta

            self.assertEqual(pk.meta.env.python_version, "3.8")
            self.assertDictEqual(pk.meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                pk.meta.signatures,
                {
                    "predict": model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec(
                                name="input_feature_0", dtype=model_signature.DataType.DOUBLE, shape=(10,)
                            )
                        ],
                        outputs=[
                            model_signature.FeatureSpec(name="output_feature_0", dtype=model_signature.DataType.DOUBLE)
                        ],
                    )
                },
            )
            self.assertListEqual(
                sorted(pk.meta.env.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", pk.meta.env.conda_dependencies)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(pk.model.metadata.run_id, run_id)

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_again")).save(
                name="model1_again", model=mlflow_pyfunc_model, options={"relax_version": False}
            )

            self.assertEqual(pk.meta.env.python_version, "3.8")
            self.assertDictEqual(pk.meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                pk.meta.signatures,
                {
                    "predict": model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec(
                                name="input_feature_0", dtype=model_signature.DataType.DOUBLE, shape=(10,)
                            )
                        ],
                        outputs=[
                            model_signature.FeatureSpec(name="output_feature_0", dtype=model_signature.DataType.DOUBLE)
                        ],
                    )
                },
            )
            self.assertListEqual(
                sorted(pk.meta.env.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", pk.meta.env.conda_dependencies)

            np.testing.assert_allclose(predictions, pk.model.predict(X_test))

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            X_df = pd.DataFrame(X_test)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_df).to_numpy())

    def test_mlflow_model_df_inputs(self) -> None:
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
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1", model=mlflow_pyfunc_model, options={"relax_version": False}
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(pk.model.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, pk.model.predict(X_test))

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_test).to_numpy())

    def test_mlflow_model_bad_case(self) -> None:
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
            )

            run_id = run.info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model", dst_path=tmpdir)
            mlflow_pyfunc_model = mlflow.pyfunc.load_model(local_path)
            mlflow_pyfunc_model.metadata.run_id = uuid.uuid4().hex.lower()

            with self.assertRaisesRegex(ValueError, "Cannot load MLFlow model artifacts."):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=mlflow_pyfunc_model,
                    options={"ignore_mlflow_dependencies": True},
                )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.save(
                name="model1",
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_dependencies": True, "relax_version": False},
            )
            assert pk.model
            assert pk.meta

            self.assertEmpty(pk.meta.env.pip_requirements)

            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=mlflow_pyfunc_model,
                    options={
                        "model_uri": local_path,
                        "ignore_mlflow_dependencies": True,
                        "relax_version": False,
                        "enable_explainability": True,
                    },
                )

            with self.assertRaisesRegex(ValueError, "Cannot load MLFlow model dependencies."):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1", model=mlflow_pyfunc_model, options={"relax_version": False}
                )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
            pk.save(
                name="model2",
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_metadata": True, "relax_version": False},
            )
            assert pk.model
            assert pk.meta

            self.assertIsNone(pk.meta.metadata)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(pk.model.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, pk.model.predict(X_test))

            model_packager.ModelPackager(os.path.join(tmpdir, "model2_again")).save(
                name="model2_again",
                model=pk.model,
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_test).to_numpy())

    def test_mlflow_model_pytorch(self) -> None:
        net = torch.nn.Linear(6, 1)
        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        X = torch.randn(6)
        y = torch.randn(1)

        epochs = 5
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = net(X)

            loss = loss_function(outputs, y)
            loss.backward()

            optimizer.step()

        with mlflow.start_run():
            signature = mlflow.models.infer_signature(X.numpy(), net(X).detach().numpy())
            model_info = mlflow.pytorch.log_model(net, "model", signature=signature)

        pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
        input_x = torch.randn(6).numpy()
        predictions = pytorch_pyfunc.predict(input_x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1", model=pytorch_pyfunc, options={"relax_version": False}
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)

            np.testing.assert_allclose(predictions, pk.model.predict(input_x))

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(
                np.expand_dims(predictions, axis=1), predict_method(pd.DataFrame(input_x)).to_numpy()
            )


if __name__ == "__main__":
    absltest.main()
