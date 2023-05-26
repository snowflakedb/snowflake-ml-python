#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn import metrics as sklearn_metrics

from snowflake import snowpark
from snowflake.ml import metrics as snowml_metrics
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.framework import utils

_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=100,
    types=[utils.DataType.INTEGER, utils.DataType.INTEGER, utils.DataType.FLOAT],
    low=0,
    high=20,
)


class AccuracyScoreTest(TestCase):
    """Test accuracy score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_accuracy_score(self) -> None:
        input_df = self._session.create_dataframe(_DATA, schema=_SCHEMA)
        pandas_df = input_df.to_pandas()

        score = snowml_metrics.accuracy_score(
            df=input_df, y_true_col_name=_SCHEMA[1], y_pred_col_name=_SCHEMA[2], normalize=False
        )
        score_sklearn = sklearn_metrics.accuracy_score(pandas_df[_SCHEMA[1]], pandas_df[_SCHEMA[2]], normalize=False)
        np.testing.assert_allclose(score, score_sklearn)

    def test_accuracy_score_sample_weight(self) -> None:
        input_df = self._session.create_dataframe(_DATA, schema=_SCHEMA)
        pandas_df = input_df.to_pandas()

        score = snowml_metrics.accuracy_score(
            df=input_df,
            y_true_col_name=_SCHEMA[1],
            y_pred_col_name=_SCHEMA[2],
            sample_weight_col_name=_SCHEMA[3],
            normalize=False,
        )
        score_sklearn = sklearn_metrics.accuracy_score(
            pandas_df[_SCHEMA[1]],
            pandas_df[_SCHEMA[2]],
            sample_weight=pandas_df[_SCHEMA[3]].to_numpy(),
            normalize=False,
        )
        np.testing.assert_allclose(score, score_sklearn)

    def test_accuracy_score_normalized(self) -> None:
        input_df = self._session.create_dataframe(_DATA, schema=_SCHEMA)
        pandas_df = input_df.to_pandas()

        score = snowml_metrics.accuracy_score(
            df=input_df, y_true_col_name=_SCHEMA[1], y_pred_col_name=_SCHEMA[2], normalize=True
        )
        score_sklearn = sklearn_metrics.accuracy_score(pandas_df[_SCHEMA[1]], pandas_df[_SCHEMA[2]], normalize=True)
        np.testing.assert_allclose(score, score_sklearn)

    def test_accuracy_score_sample_weight_normalized(self) -> None:
        input_df = self._session.create_dataframe(_DATA, schema=_SCHEMA)
        pandas_df = input_df.to_pandas()

        score = snowml_metrics.accuracy_score(
            df=input_df,
            y_true_col_name=_SCHEMA[1],
            y_pred_col_name=_SCHEMA[2],
            sample_weight_col_name=_SCHEMA[3],
            normalize=True,
        )
        score_sklearn = sklearn_metrics.accuracy_score(
            pandas_df[_SCHEMA[1]], pandas_df[_SCHEMA[2]], sample_weight=pandas_df[_SCHEMA[3]].to_numpy(), normalize=True
        )
        np.testing.assert_allclose(score, score_sklearn)


if __name__ == "__main__":
    main()
