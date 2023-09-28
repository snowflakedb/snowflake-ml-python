#
# This code is auto-generated using the sklearn_wrapper_template.py_template template.
# Do not modify the auto-generated code(except automatic reformatting by precommit hooks).
#
import inflection
import numpy as np
import pytest
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SkIterativeImputer
from sklearn.linear_model import LinearRegression as SkLinearRegression

from snowflake.ml.modeling.impute import IterativeImputer
from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


@pytest.mark.pip_incompatible
class IterativeImputerTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_cols = ["TARGET"]
        # Add index column
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        # Introduce null into data
        np.random.seed(0)
        mask = np.random.choice([True, False], size=input_df_pandas[input_cols].shape, p=[0.05, 0.95])
        input_df_pandas[input_cols][mask] = np.nan
        # Create snowpark df
        input_df = self._session.create_dataframe(input_df_pandas)

        sklearn_reg = SkIterativeImputer(random_state=0, max_iter=2000, estimator=SkLinearRegression())
        sklearn_reg.fit(input_df_pandas[input_cols])
        sklearn_numpy_arr = sklearn_reg.transform(input_df_pandas[input_cols])

        reg = IterativeImputer(random_state=0, max_iter=2000, estimator=LinearRegression())
        reg.set_input_cols(input_cols)
        reg.set_output_cols(input_cols)
        reg.set_label_cols(label_cols)

        reg.fit(input_df)
        actual_arr = (
            reg.transform(input_df).to_pandas().sort_values(by="INDEX")[input_cols].astype("float64").to_numpy()
        )

        self.assertFalse(np.any(np.isnan(actual_arr)))
        if not np.allclose(actual_arr, sklearn_numpy_arr, rtol=1.0e-1, atol=1.0e-2):
            has_diff = ~np.isclose(actual_arr, sklearn_numpy_arr, rtol=1.0e-1, atol=1.0e-2)
            print(f"Num differences: {has_diff.sum()}")
            print(f"Actual values: {actual_arr.take(has_diff.nonzero())}")
            print(f"SK values: {sklearn_numpy_arr.take(has_diff.nonzero())}")
            raise AssertionError("Results didn't match for IterativeImputer")


if __name__ == "__main__":
    main()
