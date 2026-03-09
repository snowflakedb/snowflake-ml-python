import tempfile
from importlib import metadata as importlib_metadata

import mlflow
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, model_selection

from snowflake.ml._internal import env
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


if __name__ == "__main__":
    absltest.main()
