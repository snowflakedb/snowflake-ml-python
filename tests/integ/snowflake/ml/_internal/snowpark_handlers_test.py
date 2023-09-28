from typing import List, Tuple

import inflection
import numpy as np
import pandas as pd
import pytest
from absl.testing import absltest, parameterized
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression

import snowflake.snowpark.session
from snowflake.ml.modeling._internal.snowpark_handlers import (
    SklearnWrapperProvider,
    SnowparkHandlers,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark._internal.utils import is_in_stored_procedure


@pytest.mark.pip_incompatible
class SnowparkHandlersTest(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = (
            Session.builder.configs(SnowflakeLoginOptions()).create()
            if not is_in_stored_procedure()
            else snowflake.snowpark.session._get_active_session()
        )
        self._handlers = SnowparkHandlers(
            class_name="test", subproject="subproject", wrapper_provider=SklearnWrapperProvider()
        )

    def tearDown(self) -> None:
        if not is_in_stored_procedure():
            self._session.close()

    def _get_test_dataset(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Constructs input dataset to be used in the integration test.

        Args:
            sklearn_obj: SKLearn object under tests. If the sklearn_obj supports multioutput, then this method will
            add extra label columns to test multioutput functionality.
            add_sample_weight_col: If true and addiptional column named "SAMPLE_WEIGHT" will be added to the dataset
            representing the weight of each sample.

        Returns:
            A tuple containing pandas dataframe, list of input columns names, and list of label column names.
        """
        input_df_pandas = load_diabetes(as_frame=True).frame

        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        # Predict UDF processes and returns data in random order.
        # Add INDEX column so that output can be sorted by that column
        # to compare results with local sklearn predict.
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

        input_cols = [
            c
            for c in input_df_pandas.columns
            if not c.startswith("TARGET") and not c.startswith("SAMPLE_WEIGHT") and not c.startswith("INDEX")
        ]
        label_cols = [c for c in input_df_pandas.columns if c.startswith("TARGET")]

        return (input_df_pandas, input_cols, label_cols)

    def test_fit_snowpark(self) -> None:
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        input_df = self._session.create_dataframe(input_df_pandas)

        sklearn_estimator = SkLinearRegression()

        fit_estimator = self._handlers.fit_snowpark(
            dataset=input_df,
            session=self._session,
            estimator=sklearn_estimator,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            input_cols=input_cols,
            label_cols=label_cols,
            sample_weight_col=None,
        )

        pandas_fit_estimator = sklearn_estimator.fit(
            X=input_df_pandas[input_cols], y=input_df_pandas[label_cols].squeeze()
        )

        # Confirm that sproc was stored in session._FIT_WRAPPER_SPROCS for reuse.
        assert "SklearnWrapperProvider" in self._session._FIT_WRAPPER_SPROCS

        fit_estimator = self._handlers.fit_snowpark(
            dataset=input_df,
            session=self._session,
            estimator=sklearn_estimator,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            input_cols=input_cols,
            label_cols=label_cols,
            sample_weight_col=None,
        )
        np.testing.assert_allclose(fit_estimator.coef_, pandas_fit_estimator.coef_)

    def test_batch_inference(self) -> None:
        sklearn_estimator = SkLinearRegression()
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        input_df = self._session.create_dataframe(input_df_pandas)

        fit_estimator = sklearn_estimator.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_cols].squeeze())

        output_cols = ["OUTPUT_" + c for c in label_cols]

        predictions = self._handlers.batch_inference(
            dataset=input_df,
            session=self._session,
            estimator=fit_estimator,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            inference_method="predict",
            input_cols=input_cols,
            pass_through_columns=list(set(input_df.columns) - set(output_cols)),
            expected_output_cols_list=output_cols,
            expected_output_cols_type="INT",
        )

        sklearn_numpy_arr = fit_estimator.predict(input_df_pandas[input_cols])
        sf_numpy_arr = predictions.to_pandas().sort_values(by="INDEX")[output_cols].to_numpy().flatten()

        np.testing.assert_allclose(sklearn_numpy_arr, sf_numpy_arr, rtol=1.0e-1, atol=1.0e-2)

    def test_score_snowpark(self) -> None:
        sklearn_estimator = SkLinearRegression()
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        input_df = self._session.create_dataframe(input_df_pandas)

        fit_estimator = sklearn_estimator.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_cols].squeeze())

        score = self._handlers.score_snowpark(
            dataset=input_df,
            session=self._session,
            estimator=fit_estimator,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            score_sproc_imports=["sklearn"],
            input_cols=input_cols,
            label_cols=label_cols,
            sample_weight_col=None,
        )

        sklearn_score = fit_estimator.score(input_df_pandas[input_cols], input_df_pandas[label_cols].squeeze())

        np.testing.assert_allclose(score, sklearn_score)


if __name__ == "__main__":
    absltest.main()
