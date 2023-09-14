import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.model import _model as model_api, model_signature
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
            with self.assertRaises(ValueError):
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api._save(
                name="model1_no_sig",
                local_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df[INPUT_COLUMNS],
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[-0.08254936]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])

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
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertRaises(ValueError):
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api._save(
                name="model1_no_sig",
                local_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[0.17150434]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            # Compare the Model Signature without indexing
            self.assertItemsEqual(s["predict"].to_dict(), meta.signatures["predict"].to_dict())

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[0.17150434]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])

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
            s = {"predict": model_signature.infer_signature(df[INPUT_COLUMNS], regr.predict(df)[[OUTPUT_COLUMNS]])}
            with self.assertRaises(ValueError):
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=regr,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=regr,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m: LinearRegression
                m, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
                np.testing.assert_allclose(predictions, m.predict(df[:1])[[OUTPUT_COLUMNS]])
                m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predictions, predict_method(df[:1])[[OUTPUT_COLUMNS]])

            model_api._save(
                name="model1_no_sig",
                local_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regr,
                sample_input=df,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            np.testing.assert_allclose(np.array([[-0.08254936]]), m.predict(df[:1])[[OUTPUT_COLUMNS]])
            s = regr.model_signatures
            # Compare the Model Signature without indexing
            self.assertItemsEqual(s["predict"].to_dict(), meta.signatures["predict"].to_dict())

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(np.array([[-0.08254936]]), predict_method(df[:1])[[OUTPUT_COLUMNS]])


if __name__ == "__main__":
    absltest.main()
