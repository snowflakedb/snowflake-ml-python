#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.metrics import r2_score as SKr2_score

from snowflake.ml.metrics import regression
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Row, Session


class R2ScoreTest(TestCase):
    """Test R2 score."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_r2_score(self) -> None:
        input_df = self._session.create_dataframe(
            [
                Row(-1.0, -1.5),
                Row(8.3, 7.6),
                Row(2.0, 2.5),
                Row(3.5, 4.7),
                Row(2.5, 1.5),
                Row(4.0, 3.8),
            ],
            schema=["col1", "col2"],
        )

        r2 = regression.r2_score(df=input_df, y_true_col_name="col1", y_pred_col_name="col2")
        pandas_df = input_df.to_pandas()
        SKr2 = SKr2_score(pandas_df["COL1"], pandas_df["COL2"])
        assert np.allclose(r2, SKr2)


if __name__ == "__main__":
    main()
