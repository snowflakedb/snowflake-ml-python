import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import _model as model_api, model_signature


class XgboostHandlerTest(absltest.TestCase):
    def test_xgb_booster(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(xgboost.DMatrix(data=cal_X_test))
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=regressor,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=regressor,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, xgboost.Booster)
                np.testing.assert_allclose(m.predict(xgboost.DMatrix(data=cal_X_test)), y_pred)
                m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            model_api._save(
                name="model1_no_sig",
                local_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            assert isinstance(m, xgboost.Booster)
            np.testing.assert_allclose(m.predict(xgboost.DMatrix(data=cal_X_test)), y_pred)
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

    def test_xgb(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        y_pred = regressor.predict(cal_X_test)
        y_pred_proba = regressor.predict_proba(cal_X_test)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(tmpdir, "model1"),
                    model=regressor,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir, "model1"),
                model=regressor,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                m, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))
                assert isinstance(m, xgboost.XGBClassifier)
                np.testing.assert_allclose(m.predict(cal_X_test), y_pred)
                m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), as_custom_model=True)
                predict_method = getattr(m_udf, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            model_api._save(
                name="model1_no_sig",
                local_dir_path=os.path.join(tmpdir, "model1_no_sig"),
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
            )

            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"))
            assert isinstance(m, xgboost.XGBClassifier)
            np.testing.assert_allclose(m.predict(cal_X_test), y_pred)
            np.testing.assert_allclose(m.predict_proba(cal_X_test), y_pred_proba)
            self.assertEqual(s["predict"], meta.signatures["predict"])

            m_udf, _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1_no_sig"), as_custom_model=True)
            predict_method = getattr(m_udf, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            predict_method = getattr(m_udf, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)


if __name__ == "__main__":
    absltest.main()
