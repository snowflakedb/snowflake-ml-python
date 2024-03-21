from typing import List, Tuple

import inflection
import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression

from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_handlers import (
    SnowparkTransformHandlers,
)
from tests.integ.snowflake.ml.test_utils import common_test_base


class SnowparkHandlersTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        sklearn_estimator = SkLinearRegression()

        self.input_df_pandas, self.input_cols, self.label_cols = self._get_test_dataset()
        self.fit_estimator = sklearn_estimator.fit(
            X=self.input_df_pandas[self.input_cols], y=self.input_df_pandas[self.label_cols].squeeze()
        )
        self.input_df = self.session.create_dataframe(self.input_df_pandas)

        self._handlers = SnowparkTransformHandlers(
            dataset=self.input_df, estimator=self.fit_estimator, class_name="test", subproject="subproject"
        )

    def _get_test_dataset(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Constructs input dataset to be used in the integration test.

        Args:
            sklearn_obj: SKLearn object under tests. If the sklearn_obj supports multioutput, then this method will
            add extra label columns to test multioutput functionality.
            add_sample_weight_col: If true and additional column named "SAMPLE_WEIGHT" will be added to the dataset
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

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["inflection"])
    def test_batch_inference(self) -> None:

        output_cols = ["OUTPUT_" + c for c in self.label_cols]

        predictions = self._handlers.batch_inference(
            session=self.session,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            inference_method="predict",
            input_cols=self.input_cols,
            drop_input_cols=False,
            expected_output_cols=output_cols,
            expected_output_cols_type="int",
        )

        sklearn_numpy_arr = self.fit_estimator.predict(self.input_df_pandas[self.input_cols])
        sf_numpy_arr = predictions.to_pandas().sort_values(by="INDEX")[output_cols].to_numpy().flatten()

        np.testing.assert_allclose(sklearn_numpy_arr, sf_numpy_arr, rtol=1.0e-1, atol=1.0e-2)

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["inflection"])
    def test_score_snowpark(self) -> None:

        score = self._handlers.score(
            session=self.session,
            dependencies=["snowflake-snowpark-python", "numpy", "scikit-learn", "cloudpickle"],
            score_sproc_imports=["sklearn"],
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=None,
        )

        sklearn_score = self.fit_estimator.score(
            self.input_df_pandas[self.input_cols], self.input_df_pandas[self.label_cols].squeeze()
        )

        np.testing.assert_allclose(score, sklearn_score)


if __name__ == "__main__":
    absltest.main()
