import os
import tempfile
import warnings

import catboost
import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager


class CatBoostHandlerTest(absltest.TestCase):
    def test_catboost_classifier(self) -> None:
        cal_data = datasets.load_breast_cancer()
        cal_X = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
        cal_y = pd.Series(cal_data.target)
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = catboost.CatBoostClassifier()
        classifier.fit(cal_X_train, cal_y_train)
        y_pred = classifier.predict(cal_X_test)
        y_pred_proba = classifier.predict_proba(cal_X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            s = {"predict": model_signature.infer_signature(cal_X_test, y_pred)}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=classifier,
                    signatures={**s, "another_predict": s["predict"]},
                    metadata={"author": "halu", "version": "1"},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=classifier,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, catboost.CatBoostClassifier)
                np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "predict", None)
                assert callable(predict_method)
                np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=classifier,
                sample_input_data=cal_X_test,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, catboost.CatBoost)
            np.testing.assert_allclose(pk.model.predict(cal_X_test), y_pred)
            np.testing.assert_allclose(pk.model.predict_proba(cal_X_test), y_pred_proba)
            self.assertEqual(s["predict"], pk.meta.signatures["predict"])

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            predict_method = getattr(pk.model, "predict", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), np.expand_dims(y_pred, axis=1))

            predict_method = getattr(pk.model, "predict_proba", None)
            assert callable(predict_method)
            np.testing.assert_allclose(predict_method(cal_X_test), y_pred_proba)


if __name__ == "__main__":
    absltest.main()
