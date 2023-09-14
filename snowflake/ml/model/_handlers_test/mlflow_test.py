import os
import tempfile
import uuid

import mlflow
import numpy as np
import pandas as pd
import torch
from absl.testing import absltest
from sklearn import datasets, ensemble, model_selection

from snowflake.ml.model import _model as model_api, model_signature


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
            saved_meta = model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
            )

            self.assertEqual(saved_meta.python_version, "3.8.13")
            self.assertDictEqual(saved_meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                saved_meta.signatures,
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
                sorted(saved_meta.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", saved_meta.conda_dependencies)

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            _ = model_api._save(
                name="model1_again",
                local_dir_path=os.path.join(tmpdir, "model1_again"),
                model=m,
            )

            self.assertEqual(meta.python_version, "3.8.13")
            self.assertDictEqual(meta.metadata, {"author": "halu", "version": "1"})
            self.assertDictEqual(
                meta.signatures,
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
                sorted(meta.pip_requirements),
                sorted(
                    [
                        "mlflow<3,>=2.3",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ]
                ),
            )
            self.assertIn("pip<=23.0.1", meta.conda_dependencies)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
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
            _ = model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
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
                _ = model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=mlflow_pyfunc_model,
                    options={"ignore_mlflow_dependencies": True},
                )

            saved_meta = model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_dependencies": True},
            )

            self.assertEmpty(saved_meta.pip_requirements)

            with self.assertRaisesRegex(ValueError, "Cannot load MLFlow model dependencies."):
                _ = model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=mlflow_pyfunc_model,
                )

            saved_meta = model_api._save(
                name="model2",
                local_dir_path=os.path.join(tmpdir, "model2"),
                model=mlflow_pyfunc_model,
                options={"model_uri": local_path, "ignore_mlflow_metadata": True},
            )

            self.assertIsNone(saved_meta.metadata)

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model2"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(m.metadata.run_id, run_id)

            np.testing.assert_allclose(predictions, m.predict(X_test))

            _ = model_api._save(
                name="model2_again",
                local_dir_path=os.path.join(tmpdir, "model2_again"),
                model=m,
            )

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model2"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
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
            _ = model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=pytorch_pyfunc,
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
            assert isinstance(m, mlflow.pyfunc.PyFuncModel)

            np.testing.assert_allclose(predictions, m.predict(input_x))

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(
                np.expand_dims(predictions, axis=1), predict_method(pd.DataFrame(input_x)).to_numpy()
            )


if __name__ == "__main__":
    absltest.main()
