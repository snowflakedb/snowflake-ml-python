#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import unittest

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.utils import connection_params


@unittest.skip("not PrPr")
class MonitorTest(absltest.TestCase):
    """Test Covariance matrix."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_compare_udfs(self) -> None:
        from snowflake.ml.modeling.metrics import monitor

        inputDf = self._session.create_dataframe(
            [
                snowpark.Row(-2, -5),
                snowpark.Row(8, 7),
            ],
            schema=["COL1", "COL2"],
        )
        self._session.udf.register(
            lambda x, y: x + y,
            return_type=snowpark.types.IntegerType(),
            input_types=[snowpark.types.IntegerType(), snowpark.types.IntegerType()],
            name="add1",
        )
        self._session.udf.register(
            lambda x, y: x + y + 1,
            return_type=snowpark.types.IntegerType(),
            input_types=[snowpark.types.IntegerType(), snowpark.types.IntegerType()],
            name="add2",
        )
        res = monitor.compare_udfs_outputs("add1", "add2", inputDf)
        pdf = res.to_pandas()
        assert pdf.iloc[0][0] == -7 and pdf.iloc[0][1] == -6

        resBucketize = monitor.compare_udfs_outputs("add1", "add2", inputDf, {"min": 0, "max": 20, "size": 2})
        pdfBucketize = resBucketize.to_pandas()
        assert pdfBucketize.iloc[0][1] == 1 and pdfBucketize.iloc[0][2] == 1

        # test invalid bucket_config arg, should fail
        err = None
        try:
            monitor.compare_udfs_outputs("add1", "add2", inputDf, {"min": 0})
        except Exception as e:
            err = e
        assert err is not None
        err = None
        try:
            monitor.compare_udfs_outputs("add1", "add2", inputDf, {"max": 0, "Size": 2})
        except Exception as e:
            err = e
        assert err is not None
        err = None
        try:
            monitor.compare_udfs_outputs("add1", "add2", inputDf, {"MIN": 0, "max": 20, "size": 2, "no": 1})
        except Exception as e:
            err = e
        assert err is not None

    def test_get_basic_stats(self) -> None:
        from snowflake.ml.modeling.metrics import monitor

        inputDf = self._session.create_dataframe(
            [
                snowpark.Row(-2, -5),
                snowpark.Row(8, 7),
                snowpark.Row(100, 98),
            ],
            schema=["MODEL1", "MODEL2"],
        )
        d1, d2 = monitor.get_basic_stats(inputDf)
        assert d1["HLL"] == d2["HLL"] == 3
        assert d1["MIN"] == -2 and d2["MIN"] == -5
        assert d1["MAX"] == 100 and d2["MAX"] == 98


if __name__ == "__main__":
    absltest.main()
