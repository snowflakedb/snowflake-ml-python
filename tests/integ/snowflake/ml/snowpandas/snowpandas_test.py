#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
import uuid
from typing import List

import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn import (
    datasets,
    ensemble,
    linear_model,
    model_selection,
    multioutput,
    pipeline,
    preprocessing,
)

from snowflake.ml import snowpandas
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark.modin import pandas as snow_pd
from snowflake.snowpark.modin.pandas.frontend import snow_dataframe
from tests.integ.snowflake.ml.modeling.framework import utils

_ROWS = 100
_TYPES = [utils.DataType.FLOAT] * 4
_DATA, _SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=10,
)
_X, _X_SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=10,
)
_Y, _Y_SCHEMA = utils.gen_fuzz_data(
    rows=_ROWS,
    types=[utils.DataType.FLOAT],
    low=0,
    high=5,
)
_X_TEST, _X_SCHEMA_TEST = utils.gen_fuzz_data(
    rows=_ROWS,
    types=_TYPES,
    low=0,
    high=10,
)
_TEMP_TABLE = "_SNOWPANDAS_TEMP"


class SnowpandasTest(TestCase):
    """Test SnowPandas patching."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def _convert_to_snowpandas(self, df: pd.DataFrame) -> snow_dataframe.SnowparkPandasDataFrame:
        table_name = f"{_TEMP_TABLE}_{str(uuid.uuid1()).replace('-', '')}"
        self._session.create_dataframe(df).write.mode("overwrite").save_as_table(table_name, table_type="temp")
        return snow_pd.read_snowflake(table_name)  # type: ignore[no-any-return]

    def test_non_native(self) -> None:
        pd_X_train, pd_y_train = pd.DataFrame(_X, columns=_X_SCHEMA), pd.DataFrame(_Y, columns=_Y_SCHEMA)
        pd_X_test = pd.DataFrame(_X_TEST, columns=_X_SCHEMA_TEST)
        snow_X_train, snow_y_train = self._convert_to_snowpandas(pd_X_train), self._convert_to_snowpandas(pd_y_train)
        snow_X_test = self._convert_to_snowpandas(pd_X_test)

        model = linear_model.LinearRegression()
        model.fit(pd_X_train, pd_y_train)
        res = model.predict(pd_X_test)

        snowpandas.init()

        model1 = linear_model.LinearRegression()
        model1.fit(snow_X_train, snow_y_train)
        res1 = model1.predict(snow_X_test)
        np.testing.assert_almost_equal(res1.to_numpy(), res)

        model2 = linear_model.LinearRegression()
        model2.fit(pd_X_train, pd_y_train)
        res2 = model2.predict(pd_X_test)
        np.testing.assert_almost_equal(res2, res)

        model3 = linear_model.LinearRegression()
        model3.fit(snow_X_train, snow_y_train)
        res3 = model3.predict(pd_X_test)
        np.testing.assert_almost_equal(res3, res)

        model4 = linear_model.LinearRegression()
        model4.fit(pd_X_train, pd_y_train)
        res4 = model4.predict(snow_X_test)
        np.testing.assert_almost_equal(res4.to_numpy(), res)

    def test_classifier(self) -> None:
        X, y = datasets.make_classification(
            n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False
        )
        X_train, X_test, y_train, _ = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        pd_X_train, pd_y_train = pd.DataFrame(
            X_train, columns=[f"X{i}" for i in range(X_train.shape[1])]
        ), pd.DataFrame(y_train, columns=["Y"])
        pd_X_test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
        snow_X_train, snow_y_train = self._convert_to_snowpandas(pd_X_train), self._convert_to_snowpandas(pd_y_train)
        snow_X_test = self._convert_to_snowpandas(pd_X_test)

        model = ensemble.RandomForestClassifier(max_depth=2, random_state=0).fit(pd_X_train, pd_y_train)
        res = model.predict(pd_X_test)

        snowpandas.init()

        model1 = ensemble.RandomForestClassifier(max_depth=2, random_state=0).fit(snow_X_train, snow_y_train)
        res1 = model1.predict(snow_X_test)
        np.testing.assert_almost_equal(res1.to_numpy().flatten(), res)

    def test_multioutput(self) -> None:
        X, y = datasets.make_multilabel_classification(n_classes=3, random_state=0)
        X_train, X_test, y_train, _ = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        pd_X_train, pd_y_train = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X.shape[1])]), pd.DataFrame(
            y_train, columns=[f"Y{i}" for i in range(y.shape[1])]
        )
        pd_X_test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
        snow_X_train, snow_y_train = self._convert_to_snowpandas(pd_X_train), self._convert_to_snowpandas(pd_y_train)
        snow_X_test = self._convert_to_snowpandas(pd_X_test)

        model = multioutput.MultiOutputClassifier(linear_model.LogisticRegression()).fit(pd_X_train, pd_y_train)
        res = model.predict(pd_X_test)

        snowpandas.init()

        model1 = multioutput.MultiOutputClassifier(linear_model.LogisticRegression()).fit(snow_X_train, snow_y_train)
        res1 = model1.predict(snow_X_test)
        np.testing.assert_almost_equal(res1.to_numpy(), res)

    def test_different_sessions(self) -> None:
        X, y = datasets.make_regression(n_samples=1000, n_features=4, n_informative=2, random_state=0, shuffle=False)
        X_train, X_test, y_train, _ = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        pd_X_train, pd_y_train = pd.DataFrame(
            X_train, columns=[f"X{i}" for i in range(X_train.shape[1])]
        ), pd.DataFrame(y_train, columns=["Y"])
        pd_X_test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
        snow_X_train, snow_y_train = self._convert_to_snowpandas(pd_X_train), self._convert_to_snowpandas(pd_y_train)
        snow_X_test = self._convert_to_snowpandas(pd_X_test)

        model = linear_model.LinearRegression()
        model.fit(pd_X_train, pd_y_train)
        res = model.predict(pd_X_test)

        snowpandas.init()

        model1 = linear_model.LinearRegression()
        model1.fit(snow_X_train, snow_y_train)
        self._session.sql(f"remove {self._session.get_session_stage()}/{model1._snowflake_model_file}").collect()
        res1 = model1.predict(snow_X_test)
        np.testing.assert_almost_equal(res1.to_numpy(), res)

    def test_pipeline(self) -> None:
        iris = datasets.load_iris(as_frame=True)
        pd_X, pd_y = iris.data, iris.target
        pd_X_train, pd_X_test, pd_y_train, _ = model_selection.train_test_split(
            pd_X, pd_y, test_size=0.2, random_state=42
        )
        snow_X_train, snow_X_test, snow_y_train = (
            snow_pd.DataFrame(pd_X_train),
            snow_pd.DataFrame(pd_X_test),
            snow_pd.DataFrame(pd_y_train),
        )

        orig_pipeline = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", linear_model.LogisticRegression(random_state=42)),
            ]
        ).fit(pd_X_train, pd_y_train)
        pd_y_pred = orig_pipeline.predict(pd_X_test)

        snowpandas.init()

        pipeline1 = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", linear_model.LogisticRegression(random_state=42)),
            ]
        ).fit(snow_X_train, snow_y_train)
        snow_y_pred1 = pipeline1.predict(snow_X_test)
        np.testing.assert_equal(snow_y_pred1.to_numpy().flatten(), pd_y_pred.flatten())

        pipeline2 = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", linear_model.LogisticRegression(random_state=42)),
            ]
        ).fit(pd_X_train, pd_y_train)
        snow_y_pred2 = pipeline2.predict(pd_X_test)
        np.testing.assert_equal(snow_y_pred2.flatten(), pd_y_pred.flatten())

        pipeline3 = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", linear_model.LogisticRegression(random_state=42)),
            ]
        ).fit(snow_X_train, snow_y_train)
        snow_y_pred3 = pipeline3.predict(pd_X_test)
        np.testing.assert_equal(snow_y_pred3.flatten(), pd_y_pred.flatten())

        pipeline4 = pipeline.Pipeline(
            [
                ("scaler", preprocessing.StandardScaler()),
                ("classifier", linear_model.LogisticRegression(random_state=42)),
            ]
        ).fit(pd_X_train, pd_y_train)
        snow_y_pred4 = pipeline4.predict(snow_X_test)
        np.testing.assert_equal(snow_y_pred4.to_numpy().flatten(), pd_y_pred.flatten())


if __name__ == "__main__":
    main()
