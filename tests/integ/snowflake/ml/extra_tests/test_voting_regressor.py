#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import inflection
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.ensemble import VotingRegressor as SkVotingRegressor
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    SGDRegressor as SkSGDRegressor,
)

from snowflake.ml.sklearn.ensemble import VotingRegressor
from snowflake.ml.sklearn.linear_model import LinearRegression, SGDRegressor
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class VotingRegressorTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df = self._session.create_dataframe(input_df_pandas)

        sklearn_reg = SkVotingRegressor(
            estimators=[("e1", SkLinearRegression()), ("e2", SkSGDRegressor(random_state=0))]
        )
        reg = VotingRegressor(estimators=[("e1", LinearRegression()), ("e2", SGDRegressor(random_state=0))])
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)

        reg.fit(input_df)
        sklearn_reg.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_col].squeeze())

        actual_arr = reg.predict(input_df).to_pandas().sort_values(by="INDEX")[output_cols].astype("float64").to_numpy()
        sklearn_numpy_arr = sklearn_reg.predict(input_df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)

    def test_fit_and_compare_results_pandas(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        # input_df = self._session.create_dataframe(input_df_pandas)

        sklearn_reg = SkVotingRegressor(
            estimators=[("e1", SkLinearRegression()), ("e2", SkSGDRegressor(random_state=0))]
        )
        reg = VotingRegressor(estimators=[("e1", LinearRegression()), ("e2", SGDRegressor(random_state=0))])
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)

        reg.fit(input_df_pandas)
        sklearn_reg.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_col].squeeze())

        actual_arr = reg.predict(input_df_pandas).sort_values(by="INDEX")[output_cols].astype("float64").to_numpy()
        sklearn_numpy_arr = sklearn_reg.predict(input_df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    main()
