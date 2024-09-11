import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from xgboost import XGBClassifier as SKXGBClassifier

from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class BatchInferenceWithNanDataTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def _get_test_dataset(self):
        # Create a DataFrame with 5 columns of random numbers
        df = pd.DataFrame(np.random.rand(100, 5), columns=["A", "B", "C", "D", "E"])

        # Randomly assign around 50% of the values to be NaN
        np.random.seed(0)
        df = df.mask(np.random.random(df.shape) < 0.5)
        # make sure the first row contains NaN value
        assert pd.isnull(df.iloc[0]).any()

        # Add a 'target' column with random 0s and 1s
        df["TARGET"] = np.random.choice([0.0, 1.0], size=len(df))
        # Normalize column names
        input_cols = [c for c in df.columns if not c.startswith("TARGET")]
        label_col = [c for c in df.columns if c.startswith("TARGET")]
        df["INDEX"] = df.reset_index().index
        return (df, input_cols, label_col)

    def test_nan_data(self) -> None:
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        input_df_sp = self._session.create_dataframe(input_df_pandas)
        kwargs = {"seed": 123, "eval_metric": "auc", "n_jobs": -1, "booster": "gbtree"}
        classifier = XGBClassifier(input_cols=input_cols, label_cols=label_cols, **kwargs)
        native_classifier = SKXGBClassifier(**kwargs)

        classifier.fit(input_df_sp)
        native_classifier.fit(
            input_df_pandas[input_cols],
            input_df_pandas[label_cols],
        )
        training_predictions = classifier.predict_proba(input_df_sp).to_pandas().sort_values(by="INDEX")
        PREDICT_PROBA_COLS = []
        for c in training_predictions.columns:
            if "PREDICT_PROBA_" in c:
                PREDICT_PROBA_COLS.append(c)

        training_predictions = training_predictions[PREDICT_PROBA_COLS].to_numpy()
        native_predictions = native_classifier.predict_proba(input_df_pandas[input_cols])
        np.testing.assert_allclose(
            training_predictions.flatten(), native_predictions.flatten(), rtol=1.0e-1, atol=1.0e-2
        )

    def test_if_output_col_name_overlaps_input_cols(self) -> None:
        # Add one corner case that
        # if output_cols overlaps the name in input_cols, will there be error?
        input_df_pandas, input_cols, label_cols = self._get_test_dataset()
        input_df_sp = self._session.create_dataframe(input_df_pandas)
        kwargs = {"seed": 123, "eval_metric": "auc", "n_jobs": -1, "booster": "gbtree"}
        classifier = XGBClassifier(input_cols=input_cols, label_cols=label_cols, output_cols=input_cols[0], **kwargs)
        native_classifier = SKXGBClassifier(**kwargs)

        classifier.fit(input_df_sp)
        native_classifier.fit(
            input_df_pandas[input_cols],
            input_df_pandas[label_cols],
        )
        training_predictions = (
            classifier.predict(input_df_sp).to_pandas().sort_values(by="INDEX")[input_cols[0]].to_numpy()
        )
        native_predictions = native_classifier.predict(input_df_pandas[input_cols])
        np.testing.assert_allclose(
            training_predictions.flatten(), native_predictions.flatten(), rtol=1.0e-1, atol=1.0e-2
        )


if __name__ == "__main__":
    main()
