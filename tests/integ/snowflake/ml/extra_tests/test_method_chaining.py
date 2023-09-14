import inflection
import pytest
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_iris

from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


@pytest.mark.pip_incompatible
class TestMethodChaining(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_multiple_infer_method_calls_on_the_same_df(self) -> None:
        input_df_pandas = load_iris(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET") and not c.startswith("INDEX")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        estimator = LogisticRegression(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, random_state=0
        ).fit(input_df)

        out = estimator.predict(input_df)
        out = estimator.predict_proba(out)
        out.collect()


if __name__ == "__main__":
    main()
