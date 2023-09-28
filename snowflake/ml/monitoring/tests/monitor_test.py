#!/usr/bin/env python3
import math, unittest
from typing import Any, List

import numpy as np
from absl.testing import absltest
from sklearn.ensemble import RandomForestClassifier

from snowflake import snowpark
from snowflake.ml.utils import connection_params
from snowflake.snowpark import functions, types


def rel_entropy(x: float, y: float) -> float:
    if np.isnan(x) or np.isnan(y):
        return np.NAN
    elif x > 0 and y > 0:
        return x * math.log2(x / y)
    elif x == 0 and y >= 0:
        return 0
    else:
        return np.inf


# This is the official JS algorithm
def JS_helper(p1: List[float], q1: List[float]) -> Any:
    p = np.asarray(p1)
    q = np.asarray(q1)
    m = (p + q) / 2.0
    tmp = np.column_stack((p, m))
    left = np.array([rel_entropy(x, y) for x, y in tmp])
    tmp = np.column_stack((q, m))
    right = np.array([rel_entropy(x, y) for x, y in tmp])
    left_sum = np.sum(left)
    right_sum = np.sum(right)
    js = left_sum + right_sum
    return np.sqrt(js / 2.0)


@unittest.skip("not PrPr")
class MonitorTest(absltest.TestCase):
    """Test Covariance matrix."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_compare_udfs(self) -> None:
        from snowflake.ml.monitoring import monitor

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
            replace=True,
        )
        self._session.udf.register(
            lambda x, y: x + y + 1,
            return_type=snowpark.types.IntegerType(),
            input_types=[snowpark.types.IntegerType(), snowpark.types.IntegerType()],
            name="add2",
            replace=True,
        )
        res = monitor.compare_udfs_outputs("add1", "add2", inputDf)
        pdf = res.to_pandas()
        assert pdf.iloc[0][0] == -7 and pdf.iloc[0][1] == -6

        resBucketize = monitor.compare_udfs_outputs("add1", "add2", inputDf, {"min": 0, "max": 20, "size": 2})
        pdfBucketize = resBucketize.to_pandas()
        assert pdfBucketize.iloc[0][1] == 1 and pdfBucketize.iloc[0][2] == 1

    def test_get_basic_stats(self) -> None:
        from snowflake.ml.monitoring import monitor

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

    def test_jensenshannon(self) -> None:
        from snowflake.ml.monitoring import monitor

        df1 = self._session.create_dataframe(
            [
                snowpark.Row(-3),
                snowpark.Row(-2),
                snowpark.Row(8),
                snowpark.Row(100),
            ],
            schema=["col1"],
        )

        df2 = self._session.create_dataframe(
            [
                snowpark.Row(-2),
                snowpark.Row(8),
                snowpark.Row(100),
                snowpark.Row(140),
            ],
            schema=["col2"],
        )

        df3 = self._session.create_dataframe(
            [
                snowpark.Row(-3),
                snowpark.Row(-2),
                snowpark.Row(8),
                snowpark.Row(8),
                snowpark.Row(8),
                snowpark.Row(100),
            ],
            schema=["col1"],
        )

        js = monitor.jensenshannon(df1, "col1", df2, "col2")
        assert abs(js - JS_helper([0.125, 0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25, 0.125])) <= 1e-5
        js = monitor.jensenshannon(df1, "col1", df3, "col1")
        assert abs(js - JS_helper([0.25, 0.25, 0.25, 0.25], [1.0 / 6, 1.0 / 6, 0.5, 1.0 / 6])) <= 1e-5

    def test_shap(self) -> None:
        X_train = np.random.randint(1, 90, (4, 5))
        y_train = np.random.randint(0, 3, (4, 1))

        clf = RandomForestClassifier(max_depth=3, random_state=0)
        clf.fit(X_train, y_train)

        test_sample = np.array([[3, 2, 1, 4, 5]])

        inputDf = self._session.create_dataframe(
            [snowpark.Row(3, 2, 1, 4, 5)],
            schema=["COL1", "COL2", "COL3", "COL4", "COL5"],
        )

        from snowflake.ml.monitoring.shap import ShapExplainer

        sf_explainer = ShapExplainer(self._session, clf.predict, X_train)
        shapdf2 = sf_explainer.get_shap(inputDf)
        shapdf2_1 = sf_explainer(inputDf)
        assert shapdf2_1 is not None
        v2 = shapdf2.collect()[0].as_dict(True)["SHAP"]
        v2 = v2.replace("\n", "").strip("[] ").split(",")

        import shap

        shap_explainer1 = shap.Explainer(clf.predict, X_train)
        shap_values1 = shap_explainer1(test_sample)

        self._session.add_packages("numpy", "shap")

        def get_shap(input: list) -> list:  # type: ignore[type-arg]
            shap_explainer = shap.Explainer(clf.predict, X_train)
            shap_values = shap_explainer(np.array([input]))
            return shap_values.values.tolist()  # type: ignore[no-any-return]

        shapudf = self._session.udf.register(get_shap, input_types=[types.ArrayType()], return_type=types.ArrayType())

        shapdf1 = inputDf.select(
            functions.array_construct("COL1", "COL2", "COL3", "COL4", "COL5").alias("INPUT")
        ).select(functions.get(shapudf("INPUT"), 0).alias("SHAP"))
        v1 = shapdf1.collect()[0].as_dict(True)["SHAP"]
        v1 = v1.replace("\n", "").strip("[] ").split(",")

        assert abs(float(v1[0]) - shap_values1.values[0][0]) <= 1e-5
        assert abs(float(v1[1]) - shap_values1.values[0][1]) <= 1e-5
        assert abs(float(v2[0]) - shap_values1.values[0][0]) <= 1e-5
        assert abs(float(v2[1]) - shap_values1.values[0][1]) <= 1e-5


if __name__ == "__main__":
    absltest.main()
