from typing import List, Tuple

import inflection
import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression

from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session, functions, types


class DecimalTypeTest(TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def _get_test_dataset(self) -> Tuple[pd.DataFrame, DataFrame, List[str], List[str]]:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df = self._session.create_dataframe(input_df_pandas)
        # casting every columns as decimal type
        fields = input_df.schema.fields
        selected_cols = []
        for field in fields:
            src = field.column_identifier.quoted_name
            dest = types.DecimalType(15, 10)
            selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
        input_df = input_df.select(selected_cols)
        return (input_df_pandas, input_df, input_cols, label_col)

    def test_decimal_type(self) -> None:
        input_df_pandas, input_df, input_cols, label_cols = self._get_test_dataset()

        sklearn_reg = SkLinearRegression()
        reg = LinearRegression(input_cols=input_cols, label_cols=label_cols)

        sklearn_reg.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        reg.fit(input_df)

        actual_results = reg.predict(input_df_pandas)[reg.get_output_cols()].to_numpy()
        sklearn_results = sklearn_reg.predict(input_df_pandas[input_cols])

        np.testing.assert_allclose(actual_results.flatten(), sklearn_results.flatten(), rtol=1.0e-3, atol=1.0e-3)


if __name__ == "__main__":
    main()
