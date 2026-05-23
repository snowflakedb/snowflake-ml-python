import os
import tempfile
import uuid
import warnings
from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import torch
from absl.testing import absltest
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml.model import (
    model_signature,
    target_platform,
    type_hints as model_types,
)
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers import mlflow as mlflow_handler


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
                        "python=3.11.9",
                        "pip<=24.0",
                        {
                            "pip": [
                                "mlflow<4,>=3.3",
                                "cloudpickle==3.0.0",
                                "numpy==2.1.3",
                                "psutil==7.0.0",
                                "scikit-learn==1.6.1",
                                "scipy==1.14.1",
                                "typing-extensions==4.12.2",
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

            self.assertEqual(pk.meta.env.python_version, "3.11")
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
                        "mlflow<4,>=3.3",
                        "numpy==2.1.3",
                        "psutil==7.0.0",
                        "scikit-learn==1.6.1",
                        "scipy==1.14.1",
                        "typing-extensions==4.12.2",
                    ]
                ),
            )
            self.assertIn("pip<=24.0", pk.meta.env.conda_dependencies)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)
            self.assertNotEqual(pk.model.metadata.run_id, run_id)

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_again")).save(
                name="model1_again", model=mlflow_pyfunc_model, options={"relax_version": False}
            )

            self.assertEqual(pk.meta.env.python_version, "3.11")
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
                        "mlflow<4,>=3.3",
                        "numpy==2.1.3",
                        "psutil==7.0.0",
                        "scikit-learn==1.6.1",
                        "scipy==1.14.1",
                        "typing-extensions==4.12.2",
                    ]
                ),
            )
            self.assertIn("pip<=24.0", pk.meta.env.conda_dependencies)

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
                options=model_types.KerasSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.expand_dims(predictions, axis=1), predict_method(X_test).to_numpy())

    def test_mlflow_model_from_save_model(self) -> None:
        """Models created via mlflow.*.save_model() and loaded from a local path should
        be automatically re-logged and packagable without needing model_uri."""
        db = datasets.load_diabetes(as_frame=True)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        signature = mlflow.models.signature.infer_signature(X_test, predictions)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "saved_model")
            mlflow.sklearn.save_model(rf, save_path, signature=signature)

            saved_model = mlflow.pyfunc.load_model(save_path)

            # Without model_uri, packaging should succeed via automatic re-logging.
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_no_uri"))
            pk.save(
                name="pkg_no_uri",
                model=saved_model,
                options={"ignore_mlflow_dependencies": True, "relax_version": False},
            )
            assert pk.model
            assert pk.meta

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_no_uri"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)

            np.testing.assert_allclose(predictions, pk.model.predict(X_test))

            # With model_uri pointing to the local directory, packaging should also succeed.
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_with_uri"))
            pk.save(
                name="pkg_with_uri",
                model=saved_model,
                options={
                    "model_uri": save_path,
                    "ignore_mlflow_dependencies": True,
                    "relax_version": False,
                },
            )
            assert pk.model
            assert pk.meta

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_with_uri"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)

            np.testing.assert_allclose(predictions, pk.model.predict(X_test))

    def test_mlflow_python_model_from_save_model(self) -> None:
        """PythonModel subclasses saved via mlflow.pyfunc.save_model() and loaded
        from a local path should be automatically re-logged and packagable."""

        class EchoModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):  # type: ignore[no-untyped-def]
                return model_input

        input_df = pd.DataFrame(
            {
                "col_str": ["hello", "world"],
                "col_int": [1, 2],
                "col_float": [1.5, 2.5],
            }
        )
        signature = mlflow.models.infer_signature(model_input=input_df, model_output=input_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "saved_echo_model")
            mlflow.pyfunc.save_model(
                path=save_path,
                python_model=EchoModel(),
                signature=signature,
            )

            saved_model = mlflow.pyfunc.load_model(save_path)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_no_uri"))
            pk.save(
                name="pkg_no_uri",
                model=saved_model,
                options={"ignore_mlflow_dependencies": True, "relax_version": False},
            )
            assert pk.model
            assert pk.meta

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_no_uri"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)

            result = pk.model.predict(input_df)
            pd.testing.assert_frame_equal(result, input_df, check_dtype=False)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_with_uri"))
            pk.save(
                name="pkg_with_uri",
                model=saved_model,
                options={
                    "model_uri": save_path,
                    "ignore_mlflow_dependencies": True,
                    "relax_version": False,
                },
            )
            assert pk.model
            assert pk.meta

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "pkg_with_uri"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, mlflow.pyfunc.PyFuncModel)

            result = pk.model.predict(input_df)
            pd.testing.assert_frame_equal(result, input_df, check_dtype=False)

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

    def test_mlflow_model_pip_deps_warehouse_error(self) -> None:
        """When an MLflow model has pip deps and explicitly targets WAREHOUSE
        without artifact_repository_map, save_model should raise an error."""
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.11.9",
                        "pip<=24.0",
                        {"pip": ["mlflow<4,>=3.3", "scikit-learn==1.6.1"]},
                    ],
                    "name": "mlflow-env",
                },
                signature=signature,
            )
            run_id = run.info.run_id

        mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Explicitly targeting WAREHOUSE with pip deps and no artifact_repository_map should error
            with self.assertRaises(exceptions.SnowflakeMLException):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model_wh"))
                pk.save(
                    name="model_wh",
                    model=mlflow_pyfunc_model,
                    target_platforms=[target_platform.TargetPlatform.WAREHOUSE],
                    options={"relax_version": False},
                )

            # Targeting both WAREHOUSE and SPCS should also error
            with self.assertRaises(exceptions.SnowflakeMLException):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model_both"))
                pk.save(
                    name="model_both",
                    model=mlflow_pyfunc_model,
                    target_platforms=[
                        target_platform.TargetPlatform.WAREHOUSE,
                        target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                    ],
                    options={"relax_version": False},
                )

    def test_mlflow_model_pip_deps_no_target_platforms_warning(self) -> None:
        """When an MLflow model has pip deps and target_platforms is None (default),
        save_model should emit a warning."""
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.11.9",
                        "pip<=24.0",
                        {"pip": ["mlflow<4,>=3.3", "scikit-learn==1.6.1"]},
                    ],
                    "name": "mlflow-env",
                },
                signature=signature,
            )
            run_id = run.info.run_id

        mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with tempfile.TemporaryDirectory() as tmpdir:
            # target_platforms=None with pip deps and no artifact_repository_map should warn
            with self.assertWarnsRegex(UserWarning, "pip requirements that require `artifact_repository_map`"):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model_none"))
                pk.save(
                    name="model_none",
                    model=mlflow_pyfunc_model,
                    target_platforms=None,
                    options={"relax_version": False},
                )

    def test_mlflow_model_pip_deps_no_warning_or_error(self) -> None:
        """No warning or error when targeting SPCS-only or when artifact_repository_map is provided."""
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                conda_env={
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.11.9",
                        "pip<=24.0",
                        {"pip": ["mlflow<4,>=3.3", "scikit-learn==1.6.1"]},
                    ],
                    "name": "mlflow-env",
                },
                signature=signature,
            )
            run_id = run.info.run_id

        mlflow_pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Targeting SPCS-only should not trigger warning or error
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model_spcs"))
                pk.save(
                    name="model_spcs",
                    model=mlflow_pyfunc_model,
                    target_platforms=[target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
                    options={"relax_version": False},
                )
            handler_warnings = [w for w in caught if "artifact_repository_map" in str(w.message)]
            self.assertEmpty(handler_warnings)

            # With artifact_repository_map provided should not trigger warning or error
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model_repo"))
                pk.save(
                    name="model_repo",
                    model=mlflow_pyfunc_model,
                    target_platforms=[target_platform.TargetPlatform.WAREHOUSE],
                    artifact_repository_map={"pip": "my_db.my_schema.my_repo"},
                    options={"relax_version": False},
                )
            handler_warnings = [w for w in caught if "artifact_repository_map" in str(w.message)]
            self.assertEmpty(handler_warnings)


class MLFlowHandlerParamsTest(absltest.TestCase):
    """Test cases for MLFlow handler support for model signature params."""

    INPUT_DF = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    def _package_and_load(
        self,
        tmpdir: str,
        python_model: mlflow.pyfunc.PythonModel,
        mlflow_sig: mlflow.models.ModelSignature,
        snowml_sig: model_signature.ModelSignature,
    ) -> model_packager.ModelPackager:
        """Save the PythonModel via MLflow, then via ModelPackager, and load as custom model."""
        save_path = os.path.join(tmpdir, "saved_model")
        mlflow.pyfunc.save_model(path=save_path, python_model=python_model, signature=mlflow_sig)

        mlflow_pyfunc_model = mlflow.pyfunc.load_model(save_path)

        pkg_path = os.path.join(tmpdir, "packaged")
        model_packager.ModelPackager(pkg_path).save(
            name="model",
            model=mlflow_pyfunc_model,
            signatures={"predict": snowml_sig},
            options={
                "model_uri": save_path,
                "ignore_mlflow_dependencies": True,
                "relax_version": False,
            },
        )

        pk = model_packager.ModelPackager(pkg_path)
        pk.load(as_custom_model=True)
        assert pk.model is not None
        assert pk.meta is not None
        return pk

    def test_mlflow_python_model_with_param_forwarding(self) -> None:
        """A single scalar param declared in the ModelSignature is forwarded into MLflow's params= dict."""

        class ScalingModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                scale = (params or {}).get("scale", 1.0)
                return model_input * scale

        mlflow_sig = mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            outputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            params=mlflow.types.ParamSchema([mlflow.types.ParamSpec("scale", mlflow.types.DataType.double, 1.0)]),
        )

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pk = self._package_and_load(tmpdir, ScalingModel(), mlflow_sig, snowml_sig)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Without kwargs — uses the MLflow ParamSchema default (scale=1.0).
            res_default = predict_method(self.INPUT_DF)
            np.testing.assert_allclose(res_default["feature"].to_numpy(), self.INPUT_DF["feature"].to_numpy())

            # With kwargs — value is wrapped into MLflow's ``params`` dict and reaches the PythonModel.
            res_scaled = predict_method(self.INPUT_DF, scale=2.0)
            np.testing.assert_allclose(
                res_scaled["feature"].to_numpy(),
                self.INPUT_DF["feature"].to_numpy() * 2.0,
            )

    def test_mlflow_python_model_with_mixed_scalar_param_types(self) -> None:
        """Multiple scalar params of different MLflow dtypes (double, long, boolean, string) all reach the model."""

        class EchoParamsModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                params = params or {}
                offset = params["offset"] if params.get("invert", False) is False else -params["offset"]
                value = model_input["feature"] * params["scale"] + offset
                label = f"{params['prefix']}:{int(params['repeat'])}"
                return pd.DataFrame({"value": value, "label": [label] * len(value)})

        mlflow_sig = mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            outputs=mlflow.types.Schema(
                [
                    mlflow.types.ColSpec(mlflow.types.DataType.double, "value"),
                    mlflow.types.ColSpec(mlflow.types.DataType.string, "label"),
                ]
            ),
            params=mlflow.types.ParamSchema(
                [
                    mlflow.types.ParamSpec("scale", mlflow.types.DataType.double, 1.0),
                    mlflow.types.ParamSpec("offset", mlflow.types.DataType.double, 0.0),
                    mlflow.types.ParamSpec("repeat", mlflow.types.DataType.long, 1),
                    mlflow.types.ParamSpec("invert", mlflow.types.DataType.boolean, False),
                    mlflow.types.ParamSpec("prefix", mlflow.types.DataType.string, "row"),
                ]
            ),
        )

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[
                model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.DOUBLE),
                model_signature.FeatureSpec(name="label", dtype=model_signature.DataType.STRING),
            ],
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
                model_signature.ParamSpec(name="offset", dtype=model_signature.DataType.DOUBLE, default_value=0.0),
                model_signature.ParamSpec(name="repeat", dtype=model_signature.DataType.INT64, default_value=1),
                model_signature.ParamSpec(name="invert", dtype=model_signature.DataType.BOOL, default_value=False),
                model_signature.ParamSpec(name="prefix", dtype=model_signature.DataType.STRING, default_value="row"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pk = self._package_and_load(tmpdir, EchoParamsModel(), mlflow_sig, snowml_sig)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Defaults: scale=1.0, offset=0.0, invert=False, repeat=1, prefix="row".
            res_default = predict_method(self.INPUT_DF)
            np.testing.assert_allclose(res_default["value"].to_numpy(), self.INPUT_DF["feature"].to_numpy())
            self.assertListEqual(res_default["label"].tolist(), ["row:1"] * len(self.INPUT_DF))

            # All param types forwarded together; ``invert=True`` flips offset sign.
            res_custom = predict_method(
                self.INPUT_DF,
                scale=2.0,
                offset=0.5,
                repeat=3,
                invert=True,
                prefix="custom",
            )
            np.testing.assert_allclose(
                res_custom["value"].to_numpy(),
                self.INPUT_DF["feature"].to_numpy() * 2.0 - 0.5,
            )
            self.assertListEqual(res_custom["label"].tolist(), ["custom:3"] * len(self.INPUT_DF))

    def test_mlflow_python_model_with_array_shaped_param(self) -> None:
        """A list-shaped param (variable-length array of doubles) is forwarded as a list."""

        class WeightedSumModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                weights = (params or {}).get("weights", [1.0])
                weight_sum = float(sum(weights))
                feature = model_input["feature"].to_numpy()
                return pd.DataFrame({"feature": feature * weight_sum})

        mlflow_sig = mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            outputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            params=mlflow.types.ParamSchema(
                [mlflow.types.ParamSpec("weights", mlflow.types.DataType.double, [1.0], (-1,))]
            ),
        )

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            params=[
                model_signature.ParamSpec(
                    name="weights",
                    dtype=model_signature.DataType.DOUBLE,
                    default_value=[1.0],
                    shape=(-1,),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pk = self._package_and_load(tmpdir, WeightedSumModel(), mlflow_sig, snowml_sig)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            # Default weights=[1.0] → sum=1.0 → output equals input.
            res_default = predict_method(self.INPUT_DF)
            np.testing.assert_allclose(res_default["feature"].to_numpy(), self.INPUT_DF["feature"].to_numpy())

            # Override with a longer list of weights; sum=6.0 reaches the model.
            res_custom = predict_method(self.INPUT_DF, weights=[1.0, 2.0, 3.0])
            np.testing.assert_allclose(
                res_custom["feature"].to_numpy(),
                self.INPUT_DF["feature"].to_numpy() * 6.0,
            )

    def test_mlflow_python_model_with_none_default_param(self) -> None:
        """A snowml ParamSpec with default_value=None round-trips through save/load and forwards values when passed."""

        class OptionalThresholdModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                threshold = (params or {}).get("threshold", 0.0)
                feature = model_input["feature"].to_numpy()
                return pd.DataFrame({"feature": feature + float(threshold)})

        # MLflow ParamSchema needs a concrete default for type inference; snowml allows None
        # to indicate "no semantic default".
        mlflow_sig = mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            outputs=mlflow.types.Schema([mlflow.types.ColSpec(mlflow.types.DataType.double, "feature")]),
            params=mlflow.types.ParamSchema([mlflow.types.ParamSpec("threshold", mlflow.types.DataType.double, 0.0)]),
        )

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            params=[
                model_signature.ParamSpec(name="threshold", dtype=model_signature.DataType.DOUBLE, default_value=None),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pk = self._package_and_load(tmpdir, OptionalThresholdModel(), mlflow_sig, snowml_sig)
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)

            assert pk.meta is not None
            loaded_sig = pk.meta.signatures["predict"]
            self.assertEqual(len(loaded_sig.params), 1)
            param = loaded_sig.params[0]
            assert isinstance(param, model_signature.ParamSpec)
            self.assertEqual(param.name, "threshold")
            self.assertIsNone(param.default_value)

            # Omitting the kwarg falls back to MLflow's concrete default (0.0).
            res_default = predict_method(self.INPUT_DF)
            np.testing.assert_allclose(res_default["feature"].to_numpy(), self.INPUT_DF["feature"].to_numpy())

            # Explicit value reaches the model.
            res_custom = predict_method(self.INPUT_DF, threshold=10.0)
            np.testing.assert_allclose(
                res_custom["feature"].to_numpy(),
                self.INPUT_DF["feature"].to_numpy() + 10.0,
            )

    def test_handler_forwards_dict_kwarg_through_params(self) -> None:
        """Wrapper-level test: a dict-typed kwarg is bundled verbatim into MLflow's ``params`` dict.

        This bypasses MLflow's runtime signature enforcement (which in MLflow 2.x cannot represent
        ``Object``/dict params) and directly verifies the contract of our handler change: any
        ``**method_kwargs`` reach ``raw_model.predict(X, params=...)`` unchanged, regardless of
        each value's Python type. Once the conda env moves to MLflow 3.x, an end-to-end variant
        with a real ``ParamSchema([ParamSpec("config", Object([...]), ...)])`` can replace this.
        """
        raw_model_mock = mock.MagicMock(spec=mlflow.pyfunc.PyFuncModel)
        raw_model_mock.predict.return_value = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            params=[
                model_signature.ParamGroupSpec(
                    name="config",
                    specs=[
                        model_signature.ParamSpec(
                            name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0
                        ),
                        model_signature.ParamSpec(
                            name="label", dtype=model_signature.DataType.STRING, default_value="default"
                        ),
                    ],
                    default_value={"scale": 1.0, "label": "default"},
                ),
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
            ],
        )

        model_meta_mock = mock.MagicMock()
        model_meta_mock.signatures = {"predict": snowml_sig}

        custom = mlflow_handler.MLFlowHandler.convert_as_custom_model(raw_model_mock, model_meta_mock)
        predict_method = getattr(custom, "predict", None)
        assert callable(predict_method)

        # No kwargs → wrapper preserves the legacy ``raw_model.predict(X)`` call shape (no ``params=``).
        predict_method(self.INPUT_DF)
        no_kwargs_call = raw_model_mock.predict.call_args_list[-1]
        self.assertNotIn("params", no_kwargs_call.kwargs)
        pd.testing.assert_frame_equal(no_kwargs_call.args[0], self.INPUT_DF)

        # Mixed kwargs (dict + scalar) → both reach ``params=`` unchanged, preserving types.
        config_value = {"scale": 3.0, "label": "custom"}
        predict_method(self.INPUT_DF, config=config_value, temperature=2.0)
        mixed_call = raw_model_mock.predict.call_args_list[-1]
        pd.testing.assert_frame_equal(mixed_call.args[0], self.INPUT_DF)
        self.assertEqual(mixed_call.kwargs["params"], {"config": config_value, "temperature": 2.0})
        # The dict value must be the exact object the caller passed, not a coerced copy.
        self.assertIs(mixed_call.kwargs["params"]["config"], config_value)

    def test_handler_drops_none_valued_kwargs_before_forwarding(self) -> None:
        """Wrapper-level test: None-valued kwargs are filtered before ``raw_model.predict``.

        snowml ``ParamSpec(default_value=None)`` produces ``method_kwargs[name] = None`` when
        the caller omits the param. Filtering routes that into MLflow's "key absent -> schema
        default" path; MLflow has no runtime semantic for None in a typed ``ParamSchema``.
        """
        raw_model_mock = mock.MagicMock(spec=mlflow.pyfunc.PyFuncModel)
        raw_model_mock.predict.return_value = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

        snowml_sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.DOUBLE)],
            params=[
                model_signature.ParamSpec(name="scale", dtype=model_signature.DataType.DOUBLE, default_value=1.0),
                model_signature.ParamSpec(name="threshold", dtype=model_signature.DataType.DOUBLE, default_value=None),
            ],
        )

        model_meta_mock = mock.MagicMock()
        model_meta_mock.signatures = {"predict": snowml_sig}

        custom = mlflow_handler.MLFlowHandler.convert_as_custom_model(raw_model_mock, model_meta_mock)
        predict_method = getattr(custom, "predict", None)
        assert callable(predict_method)

        # Mixed concrete + ``None``: only the concrete value reaches MLflow.
        predict_method(self.INPUT_DF, scale=2.0, threshold=None)
        mixed_call = raw_model_mock.predict.call_args_list[-1]
        self.assertEqual(mixed_call.kwargs["params"], {"scale": 2.0})
        self.assertNotIn("threshold", mixed_call.kwargs["params"])

        # All-``None``: wrapper falls back to the legacy ``raw_model.predict(X)`` call shape so
        # MLflow's ``ParamSchema`` defaults apply uniformly (no empty ``params={}`` is forwarded).
        predict_method(self.INPUT_DF, scale=None, threshold=None)
        all_none_call = raw_model_mock.predict.call_args_list[-1]
        self.assertNotIn("params", all_none_call.kwargs)
        pd.testing.assert_frame_equal(all_none_call.args[0], self.INPUT_DF)


if __name__ == "__main__":
    absltest.main()
