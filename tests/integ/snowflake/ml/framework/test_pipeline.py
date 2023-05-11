#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import importlib
import os
import pickle
import sys
import tempfile
from typing import List, Union

import cloudpickle
import inflection
import joblib
import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import (
    LinearRegression as SklearnLinearRegression,
    SGDClassifier as SklearnSGDClassifier,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler as SklearnMinMaxScaler,
    StandardScaler as SklearnStandardScaler,
)

from snowflake.ml.framework.pipeline import Pipeline
from snowflake.ml.preprocessing import (  # type: ignore[attr-defined]
    MinMaxScaler,
    StandardScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session
from tests.integ.snowflake.ml.framework import utils as framework_utils
from tests.integ.snowflake.ml.framework.utils import (
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


# TODO(snandamuri): Replace these dummy classes with actual estimators when estimator are landed.
class SnowmlLinearRegression:
    def __init__(self, input_cols: List[str], output_cols: List[str], label_cols: List[str], session: Session) -> None:
        self.model = SklearnLinearRegression()
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.label_cols = label_cols
        self._session = session

    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "SnowmlLinearRegression":
        if isinstance(dataset, DataFrame):
            pandas_df = dataset.to_pandas()
        else:
            pandas_df = dataset
        self.model.fit(pandas_df[self.input_cols], pandas_df[self.label_cols])
        return self

    def predict(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        if isinstance(dataset, DataFrame):
            pandas_df = dataset.to_pandas()
            result = self.model.predict(pandas_df[self.input_cols])
            pandas_df[self.output_cols] = result.reshape(pandas_df.shape[0], len(self.output_cols))
            return self._session.create_dataframe(pandas_df)
        else:
            pandas_df = dataset
            result = self.model.predict(pandas_df[self.input_cols])
            pandas_df[self.output_cols] = result.reshape(pandas_df.shape[0], len(self.output_cols))
            return pandas_df


class SnowmlSGDClassifier:
    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        label_cols: List[str],
        session: Session,
        random_state: int = 0,
    ) -> None:
        self.model = SklearnSGDClassifier(random_state=random_state, loss="log_loss")
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.label_cols = label_cols
        self._session = session

    def fit(self, dataset: DataFrame) -> "SnowmlSGDClassifier":
        pandas_df = dataset.to_pandas()
        self.model.fit(pandas_df[self.input_cols], pandas_df[self.label_cols])
        return self

    def predict(self, dataset: DataFrame) -> DataFrame:
        pandas_df = dataset.to_pandas()
        result = self.model.predict(pandas_df[self.input_cols])
        pandas_df[self.output_cols] = result.reshape(pandas_df.shape[0], len(self.output_cols))
        return self._session.create_dataframe(pandas_df)

    def predict_proba(self, dataset: DataFrame) -> DataFrame:
        pandas_df = dataset.to_pandas()
        proba = self.model.predict_proba(pandas_df[self.input_cols])
        columns = [f"CLASS_{c}" for c in range(0, proba.shape[1])]
        return self._session.create_dataframe(pd.DataFrame(proba, columns=columns))

    def predict_log_proba(self, dataset: DataFrame) -> DataFrame:
        pandas_df = dataset.to_pandas()
        log_proba = self.model.predict_log_proba(pandas_df[self.input_cols])
        columns = [f"CLASS_{c}" for c in range(0, log_proba.shape[1])]
        return self._session.create_dataframe(pd.DataFrame(log_proba, columns=columns))

    def score(self, dataset: DataFrame) -> DataFrame:
        pandas_df = dataset.to_pandas()
        score = self.model.score(pandas_df[self.input_cols], pandas_df[self.label_cols])
        return self._session.create_dataframe(pd.DataFrame({"score": [score]}))


class TestPipeline(TestCase):
    """Test Pipeline."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_single_step(self) -> None:
        """
        Test Pipeline with a single step.

        Raises
        ------
        AssertionError
            If the queries of the transformed dataframes with and without Pipeline are not identical.
        """
        input_col, output_col = NUMERIC_COLS[0], "output"
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        scaler = MinMaxScaler().set_input_cols(input_col).set_output_cols(output_col)
        pipeline = Pipeline([("scaler", scaler)])
        pipeline.fit(df)
        transformed_df = pipeline.transform(df)

        expected_df = scaler.fit(df).transform(df)
        assert transformed_df.queries["queries"][-1] == expected_df.queries["queries"][-1]

    def test_multiple_steps(self) -> None:
        """
        Test Pipeline with multiple steps.

        Raises
        ------
        AssertionError
            If the queries of the transformed dataframes with and without Pipeline are not identical.
        """
        input_col, output_col1, output_col2 = NUMERIC_COLS[0], "output1", "output2"
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        mms = MinMaxScaler().set_input_cols(input_col).set_output_cols(output_col1)
        ss = StandardScaler().set_input_cols(output_col1).set_output_cols(output_col2)
        pipeline = Pipeline([("mms", mms), ("ss", ss)])
        pipeline.fit(df)
        transformed_df = pipeline.transform(df)

        df1 = mms.fit(df).transform(df)
        df2 = ss.fit(df1).transform(df1)
        assert transformed_df.queries["queries"][-1] == df2.queries["queries"][-1]

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn pipeline.
        """
        data, schema = DATA, SCHEMA
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        pipeline_output_cols = ["OUT1", "OUT2"]

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        ss = StandardScaler(input_cols=input_cols, output_cols=output_cols)
        mms = MinMaxScaler(input_cols=output_cols, output_cols=pipeline_output_cols)
        pipeline = Pipeline([("ss", ss), ("mms", mms)])
        pipeline.fit(df1)
        filepath = os.path.join(tempfile.gettempdir(), "test_pipeline.pkl")
        self._to_be_deleted_files.append(filepath)
        pipeline_dump_cloudpickle = cloudpickle.dumps(pipeline)
        pipeline_dump_pickle = pickle.dumps(pipeline)
        joblib.dump(pipeline, filepath)

        self._session.close()

        # transform in session 2
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        importlib.reload(sys.modules["snowflake.ml.framework.pipeline"])

        # cloudpickle
        pipeline_load_cloudpickle = cloudpickle.loads(pipeline_dump_cloudpickle)
        transformed_df_cloudpickle = pipeline_load_cloudpickle.transform(df2[input_cols_extended])
        actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()

        # pickle
        pipeline_load_pickle = pickle.loads(pipeline_dump_pickle)
        transformed_df_pickle = pipeline_load_pickle.transform(df2[input_cols_extended])
        actual_arr_pickle = transformed_df_pickle.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()

        # joblib
        pipeline_load_joblib = joblib.load(filepath)
        transformed_df_joblib = pipeline_load_joblib.transform(df2[input_cols_extended])
        actual_arr_joblib = transformed_df_joblib.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()

        # sklearn
        skpipeline = SkPipeline([("ss", SklearnStandardScaler()), ("mms", SklearnMinMaxScaler())])
        skpipeline.fit(df_pandas[input_cols])
        sklearn_arr = skpipeline.transform(df_pandas[input_cols])

        assert np.allclose(actual_arr_cloudpickle, sklearn_arr)
        assert np.allclose(actual_arr_pickle, sklearn_arr)
        assert np.allclose(actual_arr_joblib, sklearn_arr)

    def test_pipeline_with_regression_estimators(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        mms = MinMaxScaler()
        mms.set_input_cols(["AGE"])
        mms.set_output_cols(["AGE"])
        ss = StandardScaler()
        ss.set_input_cols(["AGE"])
        ss.set_output_cols(["AGE"])

        estimator = SnowmlLinearRegression(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, session=self._session
        )

        pipeline = Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])

        assert not hasattr(pipeline, "transform")
        assert not hasattr(pipeline, "fit_transform")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "fit_predict")
        assert not hasattr(pipeline, "predict_proba")
        assert not hasattr(pipeline, "predict_log_proba")
        assert not hasattr(pipeline, "score")

        # fit and predict
        pipeline.fit(input_df)
        output_df = pipeline.predict(input_df)
        actual_results = output_df.to_pandas()[output_cols].to_numpy()

        # Do the same with SKLearn
        age_col_transform = SkPipeline(steps=[("mms", SklearnMinMaxScaler()), ("ss", SklearnStandardScaler())])
        skpipeline = SkPipeline(
            steps=[
                (
                    "preprocessor",
                    SkColumnTransformer(
                        transformers=[("age_col_transform", age_col_transform, ["AGE"])], remainder="passthrough"
                    ),
                ),
                ("estimator", SklearnLinearRegression()),
            ]
        )

        skpipeline.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        sk_predict_results = skpipeline.predict(input_df_pandas[input_cols])

        assert np.allclose(actual_results, sk_predict_results)

    def test_pipeline_with_classifier_estimators(self) -> None:
        input_df_pandas = load_iris(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        estimator = SnowmlSGDClassifier(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, session=self._session, random_state=0
        )

        pipeline = Pipeline(steps=[("estimator", estimator)])

        assert not hasattr(pipeline, "transform")
        assert not hasattr(pipeline, "fit_transform")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "fit_predict")
        assert hasattr(pipeline, "predict_proba")
        assert hasattr(pipeline, "predict_log_proba")
        assert hasattr(pipeline, "score")

        # fit and predict
        pipeline.fit(input_df)
        output_df = pipeline.predict(input_df)
        actual_results = output_df.to_pandas()[output_cols].to_numpy().flatten()
        actual_proba = pipeline.predict_proba(input_df).to_pandas().to_numpy()
        actual_log_proba = pipeline.predict_log_proba(input_df).to_pandas().to_numpy()
        actual_score = pipeline.score(input_df).to_pandas().iat[0, 0]

        # Do the same with SKLearn
        skpipeline = SkPipeline(steps=[("estimator", SklearnSGDClassifier(random_state=0, loss="log_loss"))])

        skpipeline.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        sk_predict_results = skpipeline.predict(input_df_pandas[input_cols])
        sk_proba = skpipeline.predict_proba(input_df_pandas[input_cols])
        sk_log_proba = skpipeline.predict_log_proba(input_df_pandas[input_cols])
        sk_score = skpipeline.score(input_df_pandas[input_cols], input_df_pandas[label_cols])

        assert np.allclose(actual_results, sk_predict_results)
        assert np.allclose(actual_proba, sk_proba)
        assert np.allclose(actual_log_proba, sk_log_proba)
        assert np.allclose(actual_score, sk_score)

    def test_pipeline_transform_with_pandas_dataframe(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]

        mms = MinMaxScaler(input_cols=input_cols, output_cols=input_cols)
        ss = StandardScaler(input_cols=input_cols, output_cols=input_cols)
        pipeline = Pipeline(steps=[("mms", mms), ("ss", ss)])

        pipeline.fit(input_df)

        snow_df_output = pipeline.transform(input_df).to_pandas()
        pandas_df_output = pipeline.transform(input_df_pandas)

        assert pandas_df_output.columns.shape == snow_df_output.columns.shape
        assert np.allclose(snow_df_output[pandas_df_output.columns].to_numpy(), pandas_df_output.to_numpy())

        snow_df_output_2 = pipeline.transform(input_df[input_cols]).to_pandas()
        pandas_df_output_2 = pipeline.transform(input_df_pandas[input_cols])

        assert pandas_df_output_2.columns.shape == snow_df_output_2.columns.shape
        assert np.allclose(snow_df_output_2[pandas_df_output_2.columns].to_numpy(), pandas_df_output_2.to_numpy())

    def test_pipeline_with_regression_estimators_pandas_dataframe(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        mms = MinMaxScaler()
        mms.set_input_cols(["AGE"])
        mms.set_output_cols(["AGE"])
        ss = StandardScaler()
        ss.set_input_cols(["AGE"])
        ss.set_output_cols(["AGE"])

        estimator = SnowmlLinearRegression(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, session=self._session
        )

        pipeline = Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])

        self.assertFalse(hasattr(pipeline, "transform"))
        self.assertFalse(hasattr(pipeline, "fit_transform"))
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "fit_predict"))
        self.assertFalse(hasattr(pipeline, "predict_proba"))
        self.assertFalse(hasattr(pipeline, "predict_log_proba"))
        self.assertFalse(hasattr(pipeline, "score"))

        # fit and predict
        pipeline.fit(input_df_pandas)
        output_df = pipeline.predict(input_df_pandas)

        actual_results = output_df[output_cols].to_numpy()

        # Do the same with SKLearn
        age_col_transform = SkPipeline(steps=[("mms", SklearnMinMaxScaler()), ("ss", SklearnStandardScaler())])
        skpipeline = SkPipeline(
            steps=[
                (
                    "preprocessor",
                    SkColumnTransformer(
                        transformers=[("age_col_transform", age_col_transform, ["AGE"])], remainder="passthrough"
                    ),
                ),
                ("estimator", SklearnLinearRegression()),
            ]
        )

        skpipeline.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        sk_predict_results = skpipeline.predict(input_df_pandas[input_cols])

        np.testing.assert_allclose(actual_results, sk_predict_results)


if __name__ == "__main__":
    main()
