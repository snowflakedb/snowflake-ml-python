import importlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml._internal import file_utils
from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.test_utils import exception_utils


class DemoModelWithManyArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(os.path.join(context.path("bias"), "bias1"), encoding="utf-8") as f:
            v1 = int(f.read())
        with open(os.path.join(context.path("bias"), "bias2"), encoding="utf-8") as f:
            v2 = int(f.read())
        self.bias = v1 + v2

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class ModelLoadHygieneTest(absltest.TestCase):
    def test_model_load_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "fake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                sys.path.insert(0, src_path)

                from fake.fake_module import p

                self.assertEqual(p.__file__, py_file_path)

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

                model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                    name="model1",
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                    code_paths=[os.path.join(src_path, "fake")],
                )

                model_packager.ModelPackager(os.path.join(workspace, "model1")).load()
                from fake.fake_module import p

                self.assertEqual(p.__file__, os.path.join(workspace, "model1", "code", "fake", "fake_module", "p.py"))

                importlib.reload(p)
                self.assertEqual(p.__file__, py_file_path)
                sys.path.remove(src_path)

    def test_model_save_validation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "snowflake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                        name="model1",
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[os.path.join(src_path, "snowflake")],
                    )

            with tempfile.TemporaryDirectory() as src_path:
                py_file_path = os.path.join(src_path, "snowflake.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                        name="model1",
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[py_file_path],
                    )

    def test_zipimport_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            zipped_snowml_path = os.path.join(workspace, "snowml.zip")
            file_utils.zip_python_package(zipped_snowml_path, "snowflake.ml")

            sys.path.append(zipped_snowml_path)
            try:
                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                    name="model1",
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                    options={"embed_local_ml_library": True, "_legacy_save": True},
                )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(
                            workspace, "model1", "code", "snowflake", "ml", "model", "_packager", "model_packager.py"
                        )
                    )
                )
            finally:
                sys.path.remove(zipped_snowml_path)


class ModelPackagerTest(absltest.TestCase):
    def test_save_validation_1(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            pk = model_packager.ModelPackager(os.path.join(workspace, "model1"))

            with exception_utils.assert_snowml_exceptions(
                self,
                expected_original_error_type=ValueError,
                expected_regex="Signatures and sample_input both cannot be specified at the same time.",
            ):
                pk.save(
                    name="model1",
                    model=linear_model.LinearRegression(),
                    sample_input=d,
                    signatures={"predict": model_signature.ModelSignature(inputs=[], outputs=[])},
                )

            with exception_utils.assert_snowml_exceptions(
                self,
                expected_original_error_type=ValueError,
                expected_regex=(
                    "Signatures and sample_input both cannot be None at the same time for this kind of model."
                ),
            ):
                pk.save(
                    name="model1",
                    model=linear_model.LinearRegression(),
                )

    def test_save_validation_2(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, LinearRegression)
            np.testing.assert_allclose(predictions, desired=pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

    def test_bad_save_model(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            os.mkdir(os.path.join(workspace, "bias"))
            with open(os.path.join(workspace, "bias", "bias1"), "w", encoding="utf-8") as f:
                f.write("25")
            with open(os.path.join(workspace, "bias", "bias2"), "w", encoding="utf-8") as f:
                f.write("68")
            lm = DemoModelWithManyArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(workspace, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = {"predict": model_signature.infer_signature(d, lm.predict(d))}

            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                    name="model1",
                    model=lm,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_packager.ModelPackager(os.path.join(workspace, "model1")).save(
                name="model1",
                model=lm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
                python_version="3.5.2",
            )

            pk = model_packager.ModelPackager(os.path.join(workspace, "model1"))
            pk.load(meta_only=True)

            with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                pk = model_packager.ModelPackager(os.path.join(workspace, "model1"))
                pk.load()


if __name__ == "__main__":
    absltest.main()
