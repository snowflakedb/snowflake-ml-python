"""
The main purpose of this file is to use Linear Regression,
to match all kinds of input and output for GridSearchCV/RandomSearchCV.
"""

import inspect
from typing import Any, Union
from unittest import mock

import cloudpickle as cp
import inflection
import numpy as np
import numpy.typing as npt
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV as SkGridSearchCV, KFold
from sklearn.model_selection._split import BaseCrossValidator

from snowflake.ml.modeling.linear_model import (  # type: ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.modeling.model_selection import (  # type: ignore[attr-defined]
    GridSearchCV,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


def _load_iris_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    input_df_pandas = load_iris(as_frame=True).frame
    input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
    input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

    input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
    label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]

    return input_df_pandas, input_cols, label_col


def matrix_scorer(clf: SkLinearRegression, X: npt.NDArray[Any], y: npt.NDArray[np.int_]) -> dict[str, float]:
    y_pred = clf.predict(X)
    return {
        "mean_absolute_error": mean_absolute_error(y, y_pred),
        "mean_squared_error": mean_squared_error(y, y_pred),
        "r2_score": r2_score(y, y_pred),
    }


def refit_strategy(cv_results: dict[str, Any]) -> Any:
    precision_threshold = 0.3
    cv_results_ = pd.DataFrame(cv_results)
    high_precision_cv_results = cv_results_[cv_results_["mean_test_score"] > precision_threshold]
    return high_precision_cv_results["mean_test_score"].idxmin()


cp.register_pickle_by_value(inspect.getmodule(matrix_scorer))
cp.register_pickle_by_value(inspect.getmodule(refit_strategy))


class HPOCorrectness(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

        pd_data, input_col, label_col = _load_iris_data()
        self._input_df_pandas = pd_data
        self._input_cols = input_col
        self._label_col = label_col
        self._input_df = self._session.create_dataframe(self._input_df_pandas)

    def tearDown(self) -> None:
        self._session.close()

    def _compare_cv_results(self, cv_result_1: dict[str, Any], cv_result_2: dict[str, Any]) -> None:
        # compare the keys
        self.assertEqual(cv_result_1.keys(), cv_result_2.keys())
        # compare the values
        for k, v in cv_result_1.items():
            if isinstance(v, np.ndarray):
                if k.startswith("param_"):  # compare the masked array
                    np.ma.allequal(v, cv_result_2[k])  # type: ignore[no-untyped-call]
                elif k == "params":  # compare the parameter combination
                    self.assertEqual(v.tolist(), cv_result_2[k])
                elif k.endswith("test_score"):  # compare the test score
                    np.testing.assert_allclose(v, cv_result_2[k], rtol=1.0e-5)
                # Do not compare the fit time

    def _compare_global_variables(self, sk_obj: SkGridSearchCV, sklearn_reg: SkGridSearchCV) -> None:
        # the result of SnowML grid search cv should behave the same as sklearn's
        if hasattr(sk_obj, "refit_time_"):
            # if refit = False, there is no attribute called refit_time_
            assert isinstance(sk_obj.refit_time_, float)
        if hasattr(sk_obj, "best_score_"):
            # if refit = callable and no best_score specified, then this attribute is empty
            np.testing.assert_allclose(sk_obj.best_score_, sklearn_reg.best_score_, rtol=1.0e-5)
        self.assertEqual(sk_obj.multimetric_, sklearn_reg.multimetric_)
        self.assertEqual(sk_obj.best_index_, sklearn_reg.best_index_)
        if hasattr(sk_obj, "n_splits_"):  # n_splits_ is only available in RandomSearchCV
            self.assertEqual(sk_obj.n_splits_, sklearn_reg.n_splits_)
        if hasattr(sk_obj, "best_estimator_"):
            for variable_name in sk_obj.best_estimator_.__dict__.keys():
                if variable_name != "n_jobs":
                    if isinstance(getattr(sk_obj.best_estimator_, variable_name), np.ndarray):
                        if getattr(sk_obj.best_estimator_, variable_name).dtype == "object":
                            self.assertEqual(
                                getattr(sk_obj.best_estimator_, variable_name).tolist(),
                                getattr(sklearn_reg.best_estimator_, variable_name).tolist(),
                            )
                        else:
                            np.testing.assert_allclose(
                                getattr(sk_obj.best_estimator_, variable_name),
                                getattr(sklearn_reg.best_estimator_, variable_name),
                                rtol=1.0e-5,
                            )
                    else:
                        np.testing.assert_allclose(
                            getattr(sk_obj.best_estimator_, variable_name),
                            getattr(sklearn_reg.best_estimator_, variable_name),
                            rtol=1.0e-5,
                        )
            self.assertEqual(sk_obj.n_features_in_, sklearn_reg.n_features_in_)
        if hasattr(sk_obj, "feature_names_in_") and hasattr(
            sklearn_reg, "feature_names_in_"
        ):  # feature_names_in_ variable is only available when `best_estimator_` is defined
            self.assertEqual(sk_obj.feature_names_in_.tolist(), sklearn_reg.feature_names_in_.tolist())
        if hasattr(sk_obj, "classes_"):
            self.assertEqual(sk_obj.classes_, sklearn_reg.classes_)
        self._compare_cv_results(sk_obj.cv_results_, sklearn_reg.cv_results_)
        if not sk_obj.multimetric_:
            self.assertEqual(sk_obj.best_params_, sklearn_reg.best_params_)

    @parameterized.parameters(  # type: ignore[misc]
        # Standard Sklearn sample
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(),
        },
        # param_grid: list of dictionary
        {
            "is_single_node": False,
            "params": [
                {"fit_intercept": [True], "positive": [True, False]},
                {"fit_intercept": [False], "positive": [True, False]},
            ],
            "cv": 5,
            "kwargs": dict(),
        },
        # cv: CV splitter
        {
            "is_single_node": False,
            "params": [
                {"fit_intercept": [True], "positive": [True, False]},
                {"fit_intercept": [False], "positive": [True, False]},
            ],
            "cv": KFold(5),
            "kwargs": dict(),
        },
        # cv: iterator with numpy array
        {
            "is_single_node": False,
            "params": [
                {"fit_intercept": [True], "positive": [True, False]},
                {"fit_intercept": [False], "positive": [True, False]},
            ],
            "cv": [
                (
                    np.array([i for i in range(30, 150)]),
                    np.array([i for i in range(30)]),
                ),
                (
                    np.array([i for i in range(30)] + [i for i in range(60, 150)]),
                    np.array([i for i in range(30, 60)]),
                ),
                (
                    np.array([i for i in range(60)] + [i for i in range(90, 150)]),
                    np.array([i for i in range(60, 90)]),
                ),
                (
                    np.array([i for i in range(90)] + [i for i in range(120, 150)]),
                    np.array([i for i in range(90, 120)]),
                ),
                (
                    np.array([i for i in range(120)]),
                    np.array([i for i in range(120, 150)]),
                ),
            ],
            "kwargs": dict(),
        },
        # cv: iterator with list
        {
            "is_single_node": False,
            "params": [
                {"fit_intercept": [True], "positive": [True, False]},
                {"fit_intercept": [False], "positive": [True, False]},
            ],
            "cv": [
                (
                    [i for i in range(30, 150)],
                    [i for i in range(30)],
                ),
                (
                    [i for i in range(30)] + [i for i in range(60, 150)],
                    [i for i in range(30, 60)],
                ),
                (
                    [i for i in range(60)] + [i for i in range(90, 150)],
                    [i for i in range(60, 90)],
                ),
                (
                    [i for i in range(90)] + [i for i in range(120, 150)],
                    [i for i in range(90, 120)],
                ),
                (
                    [i for i in range(120)],
                    [i for i in range(120, 150)],
                ),
            ],
            "kwargs": dict(),
        },
        # scoring: single score with a string
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(scoring="neg_mean_absolute_error"),
        },
        # scoring: single score with a callable
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(scoring=make_scorer(mean_absolute_error)),
        },
        # scoring: multiple scores with list
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(
                scoring=["neg_mean_absolute_error", "neg_mean_squared_error"],
                refit="neg_mean_squared_error",
                return_train_score=True,
            ),
        },
        #  scoring: multiple scores with tuple
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(
                scoring=("neg_mean_absolute_error", "neg_mean_squared_error"),
                refit="neg_mean_squared_error",
                return_train_score=True,
            ),
        },
        # scoring: multiple scores with custom function
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(scoring=matrix_scorer, refit="mean_squared_error"),
        },
        # scoring: multiple scores with dictionary of callable
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(
                scoring={
                    "mean_absolute_error": make_scorer(mean_absolute_error),
                    "mean_squared_error": make_scorer(mean_squared_error),
                },
                refit="mean_squared_error",
                return_train_score=True,
            ),
        },
        # refit: True
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(refit=False),
        },
        # refit: str
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(refit="neg_mean_squared_error"),
        },
        # refit: callable
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(refit=refit_strategy, error_score=0.98),
        },
        # error_score: float
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(error_score=0),
        },
        # error_score: 'raise'
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(
                scoring=["neg_mean_absolute_error", "neg_mean_squared_error"],
                refit="neg_mean_squared_error",
                return_train_score=True,
            ),
        },
        # return_train_score: True
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(return_train_score=True),
        },
        # n_jobs = -1
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(n_jobs=-1),
        },
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(n_jobs=3),
        },
        # pre_dispatch: ignored
        {
            "is_single_node": False,
            "params": {"fit_intercept": [True, False], "positive": [True, False]},
            "cv": 5,
            "kwargs": dict(pre_dispatch=5),
        },
    )
    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_fit_and_compare_results(
        self,
        mock_is_single_node: mock.MagicMock,
        is_single_node: bool,
        params: Union[dict[str, Any], list[dict[str, Any]]],
        cv: Union[int, BaseCrossValidator, list[tuple[Union[list[int], npt.NDArray[np.int_]]]]],
        kwargs: dict[str, Any],
    ) -> None:
        mock_is_single_node.return_value = is_single_node

        reg = GridSearchCV(estimator=LinearRegression(), param_grid=params, cv=cv, **kwargs)
        sklearn_reg = SkGridSearchCV(estimator=SkLinearRegression(), param_grid=params, cv=cv, **kwargs)
        reg.set_input_cols(self._input_cols)
        output_cols = ["OUTPUT_" + c for c in self._label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(self._label_col)

        reg.fit(self._input_df)
        sklearn_reg.fit(X=self._input_df_pandas[self._input_cols], y=self._input_df_pandas[self._label_col].squeeze())
        sk_obj = reg.to_sklearn()

        self._compare_global_variables(sk_obj, sklearn_reg)


if __name__ == "__main__":
    absltest.main()
