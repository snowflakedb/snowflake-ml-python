from typing import List, Tuple

import inflection
import pandas as pd
from absl.testing import absltest
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

from snowflake.ml.modeling.model_selection import (  # type: ignore[attr-defined]
    GridSearchCV,
    RandomizedSearchCV,
)


def _load_iris_data() -> Tuple[pd.DataFrame, List[str], List[str]]:
    input_df_pandas = load_iris(as_frame=True).frame
    input_df_pandas.columns = [f'"{inflection.parameterize(c, "_")}"' for c in input_df_pandas.columns]
    input_df_pandas['"index"'] = input_df_pandas.reset_index().index

    input_cols = [c for c in input_df_pandas.columns if not c.startswith('"target"')]
    label_col = [c for c in input_df_pandas.columns if c.startswith('"target"')]

    return input_df_pandas, input_cols, label_col


class CheckSklearnInferenceCornerCases(absltest.TestCase):
    """sklearn_inference function is the base helper function to execute all the pandas dataframe input
    This test is to cover corner cases that implemented within sklearn_inference, including
    - output_cols dimension mismatch
    - double quoted column names

    Args:
        absltest (_type_): default test
    """

    def setUp(self) -> None:
        pd_data, input_col, label_col = _load_iris_data()
        self._input_df_pandas = pd_data
        self._input_cols = input_col
        self._label_col = label_col

    def test_sklearn_inference_gridsearch(self) -> None:
        reg = GridSearchCV(
            estimator=LinearRegression(),
            input_cols=self._input_cols,
            param_grid={"fit_intercept": [True, False], "positive": [True, False]},
        )
        reg.set_label_cols(self._label_col)
        reg.set_drop_input_cols(True)
        reg.fit(self._input_df_pandas)
        # In predict function, the pandas dataframe's column name is actually wrong (["1"])
        # it would raise error

        with self.assertRaises(ValueError):
            reg.predict(pd.DataFrame({"1": []}))

        #
        test_pd = self._input_df_pandas
        test_pd.columns = [
            '"sepal_length_cm"',
            "sepal_width_cm",  # mismatched quotation style from input
            '"petal_length_cm"',
            '"petal_width_cm"',
            '"target"',
            '"index"',
        ]

        with self.assertRaises(ValueError):
            reg.predict(test_pd)

    def test_sklearn_inference_randomizedsearch(self) -> None:
        reg = RandomizedSearchCV(
            estimator=LinearRegression(),
            input_cols=self._input_cols,
            param_distributions={"fit_intercept": [True, False], "positive": [True, False]},
        )
        reg.set_input_cols(self._input_cols)
        reg.set_label_cols(self._label_col)
        reg.set_drop_input_cols(True)
        reg.fit(self._input_df_pandas)
        # In predict function, the pandas dataframe's column name is actually wrong (["1"])
        # it would raise error
        with self.assertRaises(ValueError):
            reg.predict(pd.DataFrame({"1": []}))

        # in the pandas dataframe's column name, some of them are single quoted
        # some of them are double quoted
        test_pd = self._input_df_pandas
        test_pd.columns = [
            '"sepal_length_cm"',
            "sepal_width_cm",  # mismatched quotation style from input
            '"petal_length_cm"',
            '"petal_width_cm"',
            '"target"',
            '"index"',
        ]

        with self.assertRaises(ValueError):
            reg.predict(test_pd)


if __name__ == "__main__":
    absltest.main()
