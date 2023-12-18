import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)


class SnowMLModelHandlerTest(absltest.TestCase):
    def test_snowml_all_input(self) -> None:
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
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertWarnsRegex(UserWarning, "Model signature will automatically be inferred during fitting"):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=regr,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                )
            with self.assertWarnsRegex(UserWarning, "Model signature will automatically be inferred during fitting"):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                    name="model1_no_sig",
                    model=regr,
                    sample_input=df[INPUT_COLUMNS],
                    metadata={"author": "halu", "version": "1"},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_signature_partial_input(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH"]
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

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

    def test_snowml_signature_drop_input_cols(self) -> None:
        iris = datasets.load_iris()

        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LinearRegression(
            input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS, drop_input_cols=True
        )
        regr.fit(df)

        predictions = regr.predict(df[:1])[[OUTPUT_COLUMNS]]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=regr,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, LinearRegression)
                np.testing.assert_allclose(predictions, pk.model.predict(df[:1])[[OUTPUT_COLUMNS]])

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])


if __name__ == "__main__":
    absltest.main()
