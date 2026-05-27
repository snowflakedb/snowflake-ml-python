import tempfile
from importlib import metadata as importlib_metadata

import mlflow
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal import env
from snowflake.ml.model import model_signature
from snowflake.ml.model._signatures import numpy_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryMLFlowModelInteg(registry_model_test_base.RegistryModelTestBase):
    _SKLEARN_CONDA_ENV: dict = {
        "dependencies": [f"python=={env.PYTHON_VERSION}"]
        + list(
            map(
                lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                ["mlflow", "cloudpickle", "numpy", "scikit-learn", "scipy", "typing-extensions"],
            )
        ),
        "name": "mlflow-env",
    }

    @parameterized.named_parameters(  # type: ignore[misc]
        ("log_model_df_input", True, False, False),
        ("log_model_numpy_input", False, False, False),
        ("save_model_no_uri", True, True, False),
        ("save_model_with_uri", True, True, True),
    )
    def test_mlflow_sklearn_model(self, as_frame: bool, use_save_model: bool, pass_model_uri: bool) -> None:
        db = datasets.load_diabetes(as_frame=as_frame)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        signature = mlflow.models.signature.infer_signature(X_test, predictions)

        test_input = X_test if as_frame else numpy_handler.SeqOfNumpyArrayHandler.convert_to_df([X_test])

        with tempfile.TemporaryDirectory() as tmpdir:
            if use_save_model:
                save_path = f"{tmpdir}/saved_sklearn_model"
                mlflow.sklearn.save_model(rf, save_path, signature=signature, conda_env=self._SKLEARN_CONDA_ENV)
                model = mlflow.pyfunc.load_model(save_path)
                options: dict = {"ignore_mlflow_dependencies": True, "relax_version": False}
                if pass_model_uri:
                    options["model_uri"] = save_path
            else:
                with mlflow.start_run() as run:
                    mlflow.sklearn.log_model(
                        rf,
                        "model",
                        signature=signature,
                        metadata={"author": "halu", "version": "1"},
                        conda_env=self._SKLEARN_CONDA_ENV,
                    )
                model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
                options = {"relax_version": False}

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(predictions, columns=res.columns),
                            check_dtype=False,
                        ),
                    ),
                },
                options=options,
            )

    @staticmethod
    def _create_echo_model_fixtures() -> tuple[
        "mlflow.pyfunc.PythonModel",
        pd.DataFrame,
        "mlflow.models.ModelSignature",
        dict,
    ]:
        """Return an EchoModel instance, sample input DataFrame, inferred signature, and conda env."""

        class EchoModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):  # type: ignore[no-untyped-def]
                return model_input

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

        signature = mlflow.models.infer_signature(model_input=input_df, model_output=input_df)

        conda_env = {
            "dependencies": [f"python=={env.PYTHON_VERSION}"]
            + list(
                map(
                    lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                    ["mlflow", "cloudpickle", "pandas", "numpy"],
                )
            ),
            "name": "mlflow-env",
        }

        return EchoModel(), input_df, signature, conda_env

    @staticmethod
    def _check_echo_result(input_df: pd.DataFrame, res: pd.DataFrame) -> None:
        assert list(res.columns) == list(
            input_df.columns
        ), f"Column mismatch: {list(res.columns)} != {list(input_df.columns)}"
        assert len(res) == len(input_df), f"Row count mismatch: {len(res)} != {len(input_df)}"
        for col in input_df.columns:
            for i in range(len(input_df)):
                expected = input_df[col].iloc[i]
                actual = res[col].iloc[i]
                assert actual == expected, f"Mismatch at [{i}, {col}]: {actual!r} != {expected!r}"

    @parameterized.named_parameters(  # type: ignore[misc]
        ("log_model", False, False),
        ("save_model_no_uri", True, False),
        ("save_model_with_uri", True, True),
    )
    def test_echo_model(self, use_save_model: bool, pass_model_uri: bool) -> None:
        echo_model, input_df, signature, conda_env = self._create_echo_model_fixtures()

        with tempfile.TemporaryDirectory() as tmpdir:
            if use_save_model:
                save_path = f"{tmpdir}/saved_echo_model"
                mlflow.pyfunc.save_model(
                    path=save_path,
                    python_model=echo_model,
                    input_example=input_df,
                    conda_env=conda_env,
                    signature=signature,
                )
                model = mlflow.pyfunc.load_model(save_path)
                options: dict = {"ignore_mlflow_dependencies": True, "relax_version": False}
                if pass_model_uri:
                    options["model_uri"] = save_path
            else:
                with mlflow.start_run() as run:
                    mlflow.pyfunc.log_model(
                        artifact_path="echo_model",
                        python_model=echo_model,
                        input_example=input_df,
                        conda_env=conda_env,
                        signature=signature,
                    )
                model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/echo_model")
                options = {}

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        input_df,
                        lambda res: self._check_echo_result(input_df, res),
                    ),
                },
                options=options or None,
            )

    def test_python_model_with_params(self) -> None:
        """End-to-end smoke that user-declared ParamSpec values reach the wrapped MLflow PythonModel at runtime.

        Covers varied param shapes in a single registry round-trip (each integ test is multi-minute):

        * Mixed scalar dtypes — ``scale`` (double), ``repeat`` (long), ``invert`` (boolean),
          ``prefix`` (string), ``repeat`` exercises the wrapper-level ``ParamSpec.dtype``
          cast to widen it to ``np.int64`` before MLflow sees it.
        * Array param with variable shape — ``weights`` (list of doubles, ``shape=(-1,)``).
        * ``None`` default on the snowml side — ``threshold`` (MLflow side carries a concrete
          ``0.0`` default since MLflow validates types; snowml side records ``None``). The MLflow
          handler drops ``None``-valued kwargs before forwarding to ``PyFuncModel.predict``, so in
          the default call MLflow's own ``ParamSchema`` default (``0.0``) reaches the wrapped
          model. The metadata round-trip and the ``None``-omission contract are both exercised
          here.

        Dict-shaped (MLflow ``Object`` → snowml ``ParamGroupSpec``) params are not covered:
        MLflow 2.x's public ``ParamSpec`` API rejects ``Object`` dtype at construction, and its
        ``_enforce_params_schema`` strips any param not declared in the schema before the
        ``PythonModel`` sees it. Wrapper-level dict forwarding is covered by the unit test
        ``MLFlowHandlerParamsTest.test_handler_forwards_dict_kwarg_through_params``; once the env
        moves to MLflow 3.x this integ test can grow an end-to-end dict variant.
        """

        class VariedParamsModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
                params = params or {}
                scale = float(params["scale"])
                weights_sum = float(sum(params["weights"]))
                # MLflow guarantees ``threshold`` is present (the handler drops ``None``-valued
                # kwargs and MLflow backfills missing keys with the ``ParamSchema`` default).
                offset = float(params["threshold"])
                multiplier = -1.0 if params["invert"] else 1.0
                feature = model_input["feature"].to_numpy()
                value = multiplier * (feature * scale * weights_sum + offset)
                label = f"{params['prefix']}:{int(params['repeat'])}"
                return pd.DataFrame({"value": value, "label": [label] * len(value)})

        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

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
                    mlflow.types.ParamSpec("repeat", mlflow.types.DataType.long, 1),
                    mlflow.types.ParamSpec("invert", mlflow.types.DataType.boolean, False),
                    mlflow.types.ParamSpec("prefix", mlflow.types.DataType.string, "row"),
                    mlflow.types.ParamSpec("weights", mlflow.types.DataType.double, [1.0], (-1,)),
                    # MLflow ParamSchema requires a concrete default for type inference; the snowml
                    # side below records ``None`` as the "no default" sentinel for the same param.
                    mlflow.types.ParamSpec("threshold", mlflow.types.DataType.double, 0.0),
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
                model_signature.ParamSpec(name="repeat", dtype=model_signature.DataType.INT64, default_value=1),
                model_signature.ParamSpec(name="invert", dtype=model_signature.DataType.BOOL, default_value=False),
                model_signature.ParamSpec(name="prefix", dtype=model_signature.DataType.STRING, default_value="row"),
                model_signature.ParamSpec(
                    name="weights",
                    dtype=model_signature.DataType.DOUBLE,
                    default_value=[1.0],
                    shape=(-1,),
                ),
                # snowml-side ``None`` default. The MLflow handler drops ``None``-valued kwargs
                # before forwarding, so MLflow's ``ParamSchema`` default (``0.0`` above) reaches
                # the wrapped model when the caller omits the param.
                model_signature.ParamSpec(name="threshold", dtype=model_signature.DataType.DOUBLE, default_value=None),
            ],
        )

        conda_env = {
            "dependencies": [f"python=={env.PYTHON_VERSION}"]
            + list(
                map(
                    lambda pkg: f"{pkg}=={importlib_metadata.distribution(pkg).version}",
                    ["mlflow", "cloudpickle", "pandas", "numpy"],
                )
            ),
            "name": "mlflow-env",
        }

        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path="varied_params_model",
                python_model=VariedParamsModel(),
                input_example=input_df,
                conda_env=conda_env,
                signature=mlflow_sig,
            )
            model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/varied_params_model")

        feature = input_df["feature"].to_numpy()
        # Defaults: scale=1.0, repeat=1, invert=False, prefix="row", weights=[1.0] (sum=1.0),
        # threshold=0.0 (from MLflow schema since snowml side records None).
        expected_default_value = feature * 1.0 * 1.0 + 0.0
        # Overrides: scale=2.0, repeat=3, invert=True, prefix="run", weights=[1.0, 2.0, 3.0]
        # (sum=6.0), threshold=0.5. Sign flips because invert=True.
        custom_weights = [1.0, 2.0, 3.0]
        expected_custom_value = -1.0 * (feature * 2.0 * sum(custom_weights) + 0.5)

        # Snowflake NUMBER columns can come back as ``decimal.Decimal`` boxed in an object-dtype
        # column, which ``np.testing.assert_allclose`` cannot ``isfinite``. Coerce to float first.
        def check_default(res: pd.DataFrame) -> None:
            np.testing.assert_allclose(res["value"].astype(float).to_numpy(), expected_default_value)
            assert list(res["label"].tolist()) == ["row:1"] * len(input_df), res["label"].tolist()

        def check_custom(res: pd.DataFrame) -> None:
            np.testing.assert_allclose(res["value"].astype(float).to_numpy(), expected_custom_value)
            assert list(res["label"].tolist()) == ["run:3"] * len(input_df), res["label"].tolist()

        self._test_registry_model(
            model=model,
            signatures={"predict": snowml_sig},
            prediction_assert_fns={
                "predict": (input_df, check_default),
            },
            params_assert_fns={
                "predict": (
                    input_df,
                    {
                        "scale": 2.0,
                        "repeat": 3,
                        "invert": True,
                        "prefix": "run",
                        "weights": custom_weights,
                        "threshold": 0.5,
                    },
                    check_custom,
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
