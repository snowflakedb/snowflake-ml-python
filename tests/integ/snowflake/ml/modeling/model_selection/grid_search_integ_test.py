import numbers
from typing import Any
from unittest import mock

import inflection
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SkPCA
from sklearn.ensemble import (
    IsolationForest as SkIsolationForest,
    RandomForestClassifier as SkRandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV as SkGridSearchCV
from sklearn.svm import SVC as SkSVC, SVR as SkSVR
from xgboost import XGBClassifier as SkXGBClassifier

from snowflake.ml.modeling.decomposition import PCA
from snowflake.ml.modeling.ensemble import IsolationForest, RandomForestClassifier
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.modeling.svm import SVC, SVR
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


def _load_iris_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    input_df_pandas = load_iris(as_frame=True).frame
    input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
    input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

    input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
    label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]

    return input_df_pandas, input_cols, label_col


class GridSearchCVTest(parameterized.TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

        pd_data, input_col, label_col = _load_iris_data()
        self._input_df_pandas = pd_data
        self._input_cols = input_col
        self._label_col = label_col
        self._input_df = self._session.create_dataframe(self._input_df_pandas)

    def tearDown(self):
        self._session.close()

    def _compare_cv_results(self, cv_result_1: dict[str, Any], cv_result_2: dict[str, Any]) -> None:
        # compare the keys
        self.assertEqual(cv_result_1.keys(), cv_result_2.keys())
        # compare the values
        for k, v in cv_result_1.items():
            if isinstance(v, np.ndarray):
                if k.startswith("param_"):  # compare the masked array
                    np.ma.allequal(v, cv_result_2[k])
                elif k == "params":  # compare the parameter combination
                    self.assertEqual(v.tolist(), cv_result_2[k])
                elif k.endswith("test_score"):  # compare the test score
                    np.testing.assert_allclose(v, cv_result_2[k], rtol=1.0e-1, atol=1.0e-2)
                # Do not compare the fit time

    def _compare_global_variables(
        self,
        sk_obj: SkGridSearchCV,
        sklearn_reg: SkGridSearchCV,
    ) -> None:
        # the result of SnowML grid search cv should behave the same as sklearn's
        if hasattr(sk_obj, "refit_time_"):
            # if refit = False, there is no attribute called refit_time_
            assert isinstance(sk_obj.refit_time_, float)
        if hasattr(sk_obj, "best_score_"):
            # if refit = callable and no best_score specified, then this attribute is empty
            np.testing.assert_allclose(
                sk_obj.best_score_,
                sklearn_reg.best_score_,
                rtol=1.0e-4,
            )
        self.assertEqual(sk_obj.multimetric_, sklearn_reg.multimetric_)
        self.assertEqual(sk_obj.best_index_, sklearn_reg.best_index_)
        if hasattr(sk_obj, "n_splits_"):  # n_splits_ is only available in RandomSearchCV
            self.assertEqual(sk_obj.n_splits_, sklearn_reg.n_splits_)
        if hasattr(sk_obj, "best_estimator_"):
            for variable_name in sk_obj.best_estimator_.__dict__.keys():
                if variable_name not in ("n_jobs", "estimator", "estimator_", "estimators_", "_Booster"):
                    target_element = getattr(sk_obj.best_estimator_, variable_name)
                    if isinstance(target_element, np.ndarray):
                        if getattr(sk_obj.best_estimator_, variable_name).dtype == "object":
                            self.assertEqual(
                                getattr(sk_obj.best_estimator_, variable_name).tolist(),
                                getattr(sklearn_reg.best_estimator_, variable_name).tolist(),
                            )
                        else:
                            np.testing.assert_allclose(
                                target_element,
                                getattr(sklearn_reg.best_estimator_, variable_name),
                                rtol=1.0e-5,
                                atol=1.0e-4,
                            )
                    elif isinstance(target_element, (str, bool, tuple, dict)):
                        self.assertEqual(
                            target_element,
                            getattr(sklearn_reg.best_estimator_, variable_name),
                        )
                    elif isinstance(target_element, numbers.Number):
                        np.testing.assert_allclose(
                            target_element,
                            getattr(sklearn_reg.best_estimator_, variable_name),
                            rtol=1.0e-5,
                            atol=1.0e-4,
                        )
                    elif isinstance(target_element, type(None)):
                        assert isinstance(getattr(sklearn_reg.best_estimator_, variable_name), type(None))
                    else:
                        raise ValueError(variable_name, target_element)
            self.assertEqual(sk_obj.n_features_in_, sklearn_reg.n_features_in_)
        if hasattr(sk_obj, "feature_names_in_") and hasattr(
            sklearn_reg, "feature_names_in_"
        ):  # feature_names_in_ variable is only available when `best_estimator_` is defined
            self.assertEqual(sk_obj.feature_names_in_.tolist(), sklearn_reg.feature_names_in_.tolist())
        if hasattr(sk_obj, "classes_"):
            np.testing.assert_allclose(sk_obj.classes_, sklearn_reg.classes_)
        self._compare_cv_results(sk_obj.cv_results_, sklearn_reg.cv_results_)
        if not sk_obj.multimetric_:
            self.assertEqual(sk_obj.best_params_, sklearn_reg.best_params_)

    @parameterized.parameters(
        {
            "enable_efficient_memory_usage": False,
        },
        {
            "enable_efficient_memory_usage": True,
        },
    )
    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_fit_and_compare_results(self, mock_is_single_node, enable_efficient_memory_usage) -> None:
        mock_is_single_node.return_value = True  # falls back to HPO implementation
        from snowflake.ml.modeling._internal.snowpark_implementations import (
            distributed_hpo_trainer,
        )

        distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = enable_efficient_memory_usage

        sklearn_reg = SkGridSearchCV(estimator=SkSVR(), param_grid={"C": [1, 10], "kernel": ("linear", "rbf")})
        reg = GridSearchCV(estimator=SVR(), param_grid={"C": [1, 10], "kernel": ("linear", "rbf")})
        reg.set_input_cols(self._input_cols)
        output_cols = ["OUTPUT_" + c for c in self._label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(self._label_col)

        reg.fit(self._input_df)
        sklearn_reg.fit(X=self._input_df_pandas[self._input_cols], y=self._input_df_pandas[self._label_col].squeeze())

        actual_arr = reg.predict(self._input_df).to_pandas().sort_values(by="INDEX")[output_cols].to_numpy()
        sklearn_numpy_arr = sklearn_reg.predict(self._input_df_pandas[self._input_cols])

        # the result of SnowML grid search cv should behave the same as sklearn's
        assert reg._sklearn_object.best_params_ == sklearn_reg.best_params_
        np.testing.assert_allclose(reg._sklearn_object.best_score_, sklearn_reg.best_score_, rtol=1.0e-1, atol=1.0e-2)
        self._compare_cv_results(reg._sklearn_object.cv_results_, sklearn_reg.cv_results_)

        np.testing.assert_allclose(actual_arr.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)

        # Test on fitting on snowpark Dataframe, and predict on pandas dataframe
        actual_arr_pd = reg.predict(self._input_df.to_pandas()).sort_values(by="INDEX")[output_cols].to_numpy()
        np.testing.assert_allclose(actual_arr_pd.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)

    @parameterized.parameters(
        {
            "is_single_node": False,
            "skmodel": SkRandomForestClassifier,
            "model": RandomForestClassifier,
            "params": {"n_estimators": [50, 200], "min_samples_split": [1.0, 2, 3], "max_depth": [3, 8]},
            "kwargs": dict(),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkRandomForestClassifier,
            "model": RandomForestClassifier,
            "params": {"n_estimators": [50, 200], "min_samples_split": [1.0, 2, 3], "max_depth": [3, 8]},
            "kwargs": dict(return_train_score=True),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkSVC,
            "model": SVC,
            "params": {"kernel": ("linear", "rbf"), "C": [1, 10, 80]},
            "kwargs": dict(),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkSVC,
            "model": SVC,
            "params": {"kernel": ("linear", "rbf"), "C": [1, 10, 80]},
            "kwargs": dict(return_train_score=True),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkXGBClassifier,
            "model": XGBClassifier,
            "params": {"max_depth": [2, 6], "learning_rate": [0.1, 0.01]},
            "kwargs": dict(scoring=["accuracy", "f1_macro"], refit="f1_macro"),
            "estimator_kwargs": dict(seed=42),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkXGBClassifier,
            "model": XGBClassifier,
            "params": {"max_depth": [2, 6], "learning_rate": [0.1, 0.01]},
            "kwargs": dict(scoring=["accuracy", "f1_macro"], refit="f1_macro", return_train_score=True),
            "estimator_kwargs": dict(seed=42),
            "enable_efficient_memory_usage": False,
        },
        {
            "is_single_node": False,
            "skmodel": SkRandomForestClassifier,
            "model": RandomForestClassifier,
            "params": {"n_estimators": [50, 200], "min_samples_split": [1.0, 2, 3], "max_depth": [3, 8]},
            "kwargs": dict(),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": True,
        },
        {
            "is_single_node": False,
            "skmodel": SkRandomForestClassifier,
            "model": RandomForestClassifier,
            "params": {"n_estimators": [50, 200], "min_samples_split": [1.0, 2, 3], "max_depth": [3, 8]},
            "kwargs": dict(return_train_score=True),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": True,
        },
        {
            "is_single_node": False,
            "skmodel": SkSVC,
            "model": SVC,
            "params": {"kernel": ("linear", "rbf"), "C": [1, 10, 80]},
            "kwargs": dict(),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": True,
        },
        {
            "is_single_node": False,
            "skmodel": SkSVC,
            "model": SVC,
            "params": {"kernel": ("linear", "rbf"), "C": [1, 10, 80]},
            "kwargs": dict(return_train_score=True),
            "estimator_kwargs": dict(random_state=0),
            "enable_efficient_memory_usage": True,
        },
        {
            "is_single_node": False,
            "skmodel": SkXGBClassifier,
            "model": XGBClassifier,
            "params": {"max_depth": [2, 6], "learning_rate": [0.1, 0.01]},
            "kwargs": dict(scoring=["accuracy", "f1_macro"], refit="f1_macro"),
            "estimator_kwargs": dict(seed=42),
            "enable_efficient_memory_usage": True,
        },
        {
            "is_single_node": False,
            "skmodel": SkXGBClassifier,
            "model": XGBClassifier,
            "params": {"max_depth": [2, 6], "learning_rate": [0.1, 0.01]},
            "kwargs": dict(scoring=["accuracy", "f1_macro"], refit="f1_macro", return_train_score=True),
            "estimator_kwargs": dict(seed=42),
            "enable_efficient_memory_usage": True,
        },
    )
    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_fit_and_compare_results_distributed(
        self,
        mock_is_single_node,
        is_single_node,
        skmodel,
        model,
        params,
        kwargs,
        estimator_kwargs,
        enable_efficient_memory_usage,
    ) -> None:
        mock_is_single_node.return_value = is_single_node
        from snowflake.ml.modeling._internal.snowpark_implementations import (
            distributed_hpo_trainer,
        )

        distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = enable_efficient_memory_usage

        sklearn_reg = SkGridSearchCV(estimator=skmodel(**estimator_kwargs), param_grid=params, cv=3, **kwargs)
        reg = GridSearchCV(estimator=model(**estimator_kwargs), param_grid=params, cv=3, **kwargs)
        reg.set_input_cols(self._input_cols)
        output_cols = ["OUTPUT_" + c for c in self._label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(self._label_col)

        reg.fit(self._input_df)
        sklearn_reg.fit(X=self._input_df_pandas[self._input_cols], y=self._input_df_pandas[self._label_col].squeeze())
        sk_obj = reg.to_sklearn()

        # the result of SnowML grid search cv should behave the same as sklearn's
        self._compare_global_variables(sk_obj, sklearn_reg)

        actual_arr = reg.predict(self._input_df).to_pandas().sort_values(by="INDEX")[output_cols].to_numpy()
        sklearn_numpy_arr = sklearn_reg.predict(self._input_df_pandas[self._input_cols])
        np.testing.assert_allclose(actual_arr.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)

        # Test on fitting on snowpark Dataframe, and predict on pandas dataframe
        actual_arr_pd = reg.predict(self._input_df.to_pandas()).sort_values(by="INDEX")[output_cols].to_numpy()
        np.testing.assert_allclose(actual_arr_pd.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)

        # Test score
        actual_score = reg.score(self._input_df)
        sklearn_score = sklearn_reg.score(
            self._input_df_pandas[self._input_cols], self._input_df_pandas[self._label_col]
        )
        np.testing.assert_allclose(actual_score, sklearn_score, rtol=1.0e-1, atol=1.0e-2)

        actual_score = reg.score(self._input_df_pandas)
        np.testing.assert_allclose(actual_score, sklearn_score, rtol=1.0e-1, atol=1.0e-2)

        # n_features_in_ is available because `refit` is set to `True`.
        self.assertEqual(sk_obj.n_features_in_, sklearn_reg.n_features_in_)

        # classes are available because these are classifier models
        for idx, class_ in enumerate(sk_obj.classes_):
            self.assertEqual(class_, sklearn_reg.classes_[idx])

        # Test predict_proba
        if hasattr(reg, "predict_proba"):
            actual_inference_result = (
                reg.predict_proba(self._input_df, output_cols_prefix="OUTPUT_").to_pandas().sort_values(by="INDEX")
            )
            actual_output_cols = [c for c in actual_inference_result.columns if c.find("OUTPUT_") >= 0]
            actual_inference_result = actual_inference_result[actual_output_cols].to_numpy()
            sklearn_predict_prob_array = sklearn_reg.predict_proba(self._input_df_pandas[self._input_cols])
            np.testing.assert_allclose(actual_inference_result.flatten(), sklearn_predict_prob_array.flatten())

            actual_pandas_result = reg.predict_proba(
                self._input_df_pandas[self._input_cols], output_cols_prefix="OUTPUT_"
            )
            actual_pandas_result = actual_pandas_result[actual_output_cols].to_numpy()
            np.testing.assert_allclose(
                actual_pandas_result.flatten(), sklearn_predict_prob_array.flatten(), rtol=1.0e-1, atol=1.0e-2
            )

        # Test predict_log_proba
        if hasattr(reg, "predict_log_proba"):
            actual_log_proba_result = reg.predict_log_proba(self._input_df).to_pandas().sort_values(by="INDEX")
            actual_output_cols = [c for c in actual_log_proba_result.columns if c.find("PREDICT_LOG_PROBA_") >= 0]
            actual_log_proba_result = actual_log_proba_result[actual_output_cols].to_numpy()
            sklearn_log_prob_array = sklearn_reg.predict_log_proba(self._input_df_pandas[self._input_cols])
            np.testing.assert_allclose(actual_log_proba_result.flatten(), sklearn_log_prob_array.flatten())

            actual_pandas_result = reg.predict_log_proba(self._input_df_pandas[self._input_cols])
            actual_pandas_result = actual_pandas_result[actual_output_cols].to_numpy()
            np.testing.assert_allclose(actual_pandas_result.flatten(), sklearn_log_prob_array.flatten())

        # Test decision function
        if hasattr(reg, "decision_function"):
            actual_decision_function = (
                reg.decision_function(self._input_df, output_cols_prefix="OUTPUT_").to_pandas().sort_values(by="INDEX")
            )
            actual_output_cols = [c for c in actual_decision_function.columns if c.find("OUTPUT_") >= 0]
            actual_decision_function_result = actual_decision_function[actual_output_cols].to_numpy()
            sklearn_decision_function = sklearn_reg.decision_function(self._input_df_pandas[self._input_cols])
            np.testing.assert_allclose(
                actual_decision_function_result, sklearn_decision_function, rtol=1.0e-1, atol=1.0e-2
            )

            actual_pandas_result = reg.decision_function(
                self._input_df_pandas[self._input_cols], output_cols_prefix="OUTPUT_"
            )
            actual_pandas_result = actual_pandas_result[actual_output_cols].to_numpy()
            np.testing.assert_allclose(
                actual_pandas_result.flatten(), sklearn_decision_function.flatten(), rtol=1.0e-1, atol=1.0e-2
            )

    @parameterized.parameters(
        {
            "enable_efficient_memory_usage": False,
        },
        {
            "enable_efficient_memory_usage": True,
        },
    )
    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_transform(self, mock_is_single_node, enable_efficient_memory_usage) -> None:
        mock_is_single_node.return_value = False
        from snowflake.ml.modeling._internal.snowpark_implementations import (
            distributed_hpo_trainer,
        )

        distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = enable_efficient_memory_usage

        params = {"n_components": range(1, 3)}
        sk_pca = SkPCA()
        sklearn_reg = SkGridSearchCV(sk_pca, params, cv=3)

        pca = PCA()
        reg = GridSearchCV(estimator=pca, param_grid=params, cv=3)
        reg.set_input_cols(self._input_cols)
        output_cols = ["OUTPUT_" + c for c in self._label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(self._label_col)

        reg.fit(self._input_df)
        sklearn_reg.fit(X=self._input_df_pandas[self._input_cols], y=self._input_df_pandas[self._label_col].squeeze())

        # the result of SnowML grid search cv should behave the same as sklearn's
        sk_obj = reg.to_sklearn()
        self._compare_global_variables(sk_obj, sklearn_reg)

        transformed = reg.transform(self._input_df).to_pandas().sort_values(by="INDEX")
        sk_transformed = sklearn_reg.transform(self._input_df_pandas[self._input_cols])

        actual_output_cols = [c for c in transformed.columns if c.find("OUTPUT_") >= 0]
        transformed = transformed[actual_output_cols].astype("float64").to_numpy()

        np.testing.assert_allclose(transformed, sk_transformed, rtol=1.0e-1, atol=1.0e-2)

        transformed = reg.transform(self._input_df_pandas[self._input_cols])
        transformed = transformed[actual_output_cols].to_numpy()
        np.testing.assert_allclose(transformed, sk_transformed, rtol=1.0e-1, atol=1.0e-2)

    def test_not_fitted_exception(self) -> None:
        param_grid = {"max_depth": [2, 6], "learning_rate": [0.1, 0.01]}
        reg = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid)

        with self.assertRaises(RuntimeError, msg="Estimator not fitted before accessing property model_signatures!"):
            reg.predict(self._input_df)

        with self.assertRaises(
            RuntimeError, msg="Estimator GridSearchCV not fitted before calling predict_proba method."
        ):
            reg.predict_proba(self._input_df)

    @parameterized.parameters(
        {
            "enable_efficient_memory_usage": False,
        },
        {
            "enable_efficient_memory_usage": True,
        },
    )
    def test_score_samples(self, enable_efficient_memory_usage) -> None:
        from snowflake.ml.modeling._internal.snowpark_implementations import (
            distributed_hpo_trainer,
        )

        distributed_hpo_trainer.ENABLE_EFFICIENT_MEMORY_USAGE = enable_efficient_memory_usage
        param_grid = {"max_features": [1, 2]}
        sklearn_reg = SkGridSearchCV(
            estimator=SkIsolationForest(random_state=0), param_grid=param_grid, scoring="accuracy"
        )
        reg = GridSearchCV(estimator=IsolationForest(random_state=0), param_grid=param_grid, scoring="accuracy")
        reg.set_input_cols(self._input_cols)
        output_cols = ["OUTPUT_" + c for c in self._label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(self._label_col)

        reg.fit(self._input_df)
        sklearn_reg.fit(X=self._input_df_pandas[self._input_cols], y=self._input_df_pandas[self._label_col].squeeze())

        # Test score_samples
        actual_score_samples_result = reg.score_samples(self._input_df).to_pandas().sort_values(by="INDEX")
        actual_output_cols = [c for c in actual_score_samples_result.columns if c.find("SCORE_SAMPLES_") >= 0]
        actual_score_samples_result = actual_score_samples_result[actual_output_cols].to_numpy()
        sklearn_score_samples_array = sklearn_reg.score_samples(self._input_df_pandas[self._input_cols])
        np.testing.assert_allclose(actual_score_samples_result.flatten(), sklearn_score_samples_array.flatten())

        actual_pandas_result = reg.score_samples(self._input_df_pandas[self._input_cols])
        actual_pandas_result = actual_pandas_result[actual_output_cols].to_numpy()
        np.testing.assert_allclose(actual_pandas_result.flatten(), sklearn_score_samples_array.flatten())


if __name__ == "__main__":
    absltest.main()
