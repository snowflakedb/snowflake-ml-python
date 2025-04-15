import inflection
import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor as SkBaggingRegressor

from snowflake.ml.modeling.ensemble import BaggingRegressor
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session


class BatchInferenceWithNJobsTest(TestCase):
    def _get_test_dataset(self) -> tuple[pd.DataFrame, DataFrame, list[str], list[str]]:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET") and not c.startswith("INDEX")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]

        input_df = self._session.create_dataframe(input_df_pandas)
        return (input_df_pandas, input_df, input_cols, label_col)

    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_if_output_col_name_overlaps_input_cols(self) -> None:
        input_df_pandas, input_df, input_cols, label_cols = self._get_test_dataset()

        snowml_classifier = BaggingRegressor(
            input_cols=input_cols,
            label_cols=label_cols,
            random_state=0,
            n_jobs=4,
        )
        native_classifier = SkBaggingRegressor(
            random_state=0,
            n_jobs=4,
        )
        snowml_classifier.fit(input_df)
        native_classifier.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        training_predictions = (
            snowml_classifier.predict(input_df)
            .to_pandas()
            .sort_values(by="INDEX")[snowml_classifier.get_output_cols()]
            .astype("float64")
            .to_numpy()
        )
        native_predictions = native_classifier.predict(input_df_pandas[input_cols])
        np.testing.assert_allclose(
            training_predictions.flatten(), native_predictions.flatten(), rtol=1.0e-1, atol=1.0e-2
        )


if __name__ == "__main__":
    main()
