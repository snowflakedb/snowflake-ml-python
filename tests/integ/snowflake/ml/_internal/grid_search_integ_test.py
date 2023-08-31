#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import inflection
import numpy as np
import pytest
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV as SkGridSearchCV
from sklearn.svm import SVR as SkSVR

from snowflake.ml.modeling.model_selection._internal import GridSearchCV
from snowflake.ml.modeling.svm import SVR
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


@pytest.mark.pip_incompatible
class GridSearchCVTest(TestCase):
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

        sklearn_reg = SkGridSearchCV(estimator=SkSVR(), param_grid={"C": [1, 10], "kernel": ("linear", "rbf")})
        reg = GridSearchCV(estimator=SVR(), param_grid={"C": [1, 10], "kernel": ("linear", "rbf")})
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)

        reg.fit(input_df)
        sklearn_reg.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_col].squeeze())

        actual_arr = reg.predict(input_df).to_pandas().sort_values(by="INDEX")[output_cols].astype("float64").to_numpy()
        sklearn_numpy_arr = sklearn_reg.predict(input_df_pandas[input_cols])

        assert reg._sklearn_object.best_params_ == sklearn_reg.best_params_

        np.testing.assert_allclose(actual_arr.flatten(), sklearn_numpy_arr.flatten(), rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    main()
