import numpy as np
import pandas as pd
from absl.testing import parameterized
from absl.testing.absltest import main

from snowflake import snowpark
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.INTEGER] * 4 + [utils.DataType.FLOAT]
_BINARY_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=2,
)
_Y_TRUE_COL = _SCHEMA[1]
_Y_PRED_COL = _SCHEMA[2]
_SAMPLE_WEIGHT_COL = _SCHEMA[5]


class MetricsUtilsTest(parameterized.TestCase):
    """Test metrics utils."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.product(
        df=(_BINARY_DATA,),
        sample_weight_col_name=(None, _SAMPLE_WEIGHT_COL),
        sample_score_col_name=(_Y_PRED_COL, _Y_TRUE_COL),
        normalize=(False, True),
    )
    def test_weighted_sum(self, df, sample_weight_col_name, sample_score_col_name, normalize) -> None:
        pandas_df = pd.DataFrame(df, columns=_SCHEMA)
        input_df = self._session.create_dataframe(pandas_df)

        snowpark_weight_col = input_df[sample_weight_col_name] if sample_weight_col_name else None
        actual_sum = metrics_utils.weighted_sum(
            df=input_df,
            sample_score_column=input_df[sample_score_col_name],
            sample_weight_column=snowpark_weight_col,
            normalize=normalize,
            statement_params=None,
        )

        sample_weight = pandas_df[sample_weight_col_name].to_numpy() if sample_weight_col_name else None
        score_column = pandas_df[sample_score_col_name].to_numpy()
        if normalize:
            expected_sum = np.average(score_column, weights=sample_weight)
        else:
            if sample_weight_col_name:
                expected_sum = np.dot(score_column, sample_weight)
            else:
                expected_sum = np.sum(score_column)

        np.testing.assert_approx_equal(actual_sum, expected_sum)


if __name__ == "__main__":
    main()
