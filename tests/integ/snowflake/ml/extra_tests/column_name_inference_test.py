import inflection
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression

from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class ColumnNameInferenceTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def _get_test_dataset(self):
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        return (input_df_pandas, input_cols, label_col)

    def _test_column_name_inference(
        self, use_snowpark_interface: bool = True, use_passthrough_cols: bool = True
    ) -> None:
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        sklearn_reg = SkLinearRegression()
        if use_passthrough_cols:
            input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
            reg = LinearRegression(label_cols=label_cols, passthrough_cols=["INDEX"])
        else:
            reg = LinearRegression(label_cols=label_cols)

        if use_snowpark_interface:
            input_df = self._session.create_dataframe(input_df_pandas)
            reg.fit(input_df)
        else:
            reg.fit(input_df_pandas)

        sklearn_reg.fit(X=input_df_pandas[input_cols], y=input_df_pandas[label_cols])

        actual_results = reg.predict(input_df_pandas)
        if use_passthrough_cols:
            actual_results = actual_results.sort_values(by="INDEX")[reg.get_output_cols()].to_numpy()
        else:
            actual_results = actual_results[reg.get_output_cols()].to_numpy()

        sklearn_results = sklearn_reg.predict(input_df_pandas[input_cols])

        np.testing.assert_array_equal(reg.get_input_cols(), input_cols)
        np.testing.assert_allclose(actual_results.flatten(), sklearn_results.flatten(), rtol=1.0e-3, atol=1.0e-3)

    def test_snowpark_interface_with_passthrough_cols(self):
        self._test_column_name_inference(use_snowpark_interface=True, use_passthrough_cols=True)

    def test_snowpark_interface_with_out_passthrough_cols(self):
        self._test_column_name_inference(use_snowpark_interface=True, use_passthrough_cols=False)

    def test_pandas_interface_with_passthrough_cols(self):
        self._test_column_name_inference(use_snowpark_interface=False, use_passthrough_cols=True)

    def test_pandas_interface_with_out_passthrough_cols(self):
        self._test_column_name_inference(use_snowpark_interface=False, use_passthrough_cols=False)


if __name__ == "__main__":
    main()
