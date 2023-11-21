#!/usr/bin/env python3
import importlib
import os
import pickle
import sys
import tempfile
from typing import List

import cloudpickle
import inflection
import joblib
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import (
    LinearRegression as SklearnLinearRegression,
    LogisticRegression as SklearnLogisticRegression,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler as SklearnMinMaxScaler,
    StandardScaler as SklearnStandardScaler,
)

from snowflake.ml.model.model_signature import DataType, FeatureSpec, ModelSignature
from snowflake.ml.modeling import pipeline as snowml_pipeline
from snowflake.ml.modeling.linear_model import (
    LinearRegression as SnowmlLinearRegression,
    LogisticRegression as SnowmlLogisticRegression,
)
from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    MinMaxScaler,
    StandardScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


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
        pipeline = snowml_pipeline.Pipeline([("scaler", scaler)])
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
        input_col, output_col1, output_col2 = NUMERIC_COLS[0], "OUTPUT1", "OUTPUT2"
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        mms = MinMaxScaler().set_input_cols(input_col).set_output_cols(output_col1)
        ss = StandardScaler().set_input_cols(output_col1).set_output_cols(output_col2)
        pipeline = snowml_pipeline.Pipeline([("mms", mms), ("ss", ss)])
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
        pipeline = snowml_pipeline.Pipeline([("ss", ss), ("mms", mms)])
        pipeline.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            pipeline_dump_cloudpickle = cloudpickle.dumps(pipeline)
            pipeline_dump_pickle = pickle.dumps(pipeline)
            joblib.dump(pipeline, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.pipeline"])

            # cloudpickle
            pipeline_load_cloudpickle = cloudpickle.loads(pipeline_dump_cloudpickle)
            transformed_df_cloudpickle = pipeline_load_cloudpickle.transform(df2[input_cols_extended])
            actual_arr_cloudpickle = (
                transformed_df_cloudpickle.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()
            )

            # pickle
            pipeline_load_pickle = pickle.loads(pipeline_dump_pickle)
            transformed_df_pickle = pipeline_load_pickle.transform(df2[input_cols_extended])
            actual_arr_pickle = transformed_df_pickle.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()

            # joblib
            pipeline_load_joblib = joblib.load(file.name)
            transformed_df_joblib = pipeline_load_joblib.transform(df2[input_cols_extended])
            actual_arr_joblib = transformed_df_joblib.sort(id_col)[pipeline_output_cols].to_pandas().to_numpy()

            # sklearn
            skpipeline = SkPipeline([("ss", SklearnStandardScaler()), ("mms", SklearnMinMaxScaler())])
            skpipeline.fit(df_pandas[input_cols])
            sklearn_arr = skpipeline.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)

    def test_pipeline_with_regression_estimators(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET") and not c.startswith("INDEX")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        mms = MinMaxScaler()
        mms.set_input_cols(["AGE"])
        mms.set_output_cols(["AGE"])
        ss = StandardScaler()
        ss.set_input_cols(["AGE"])
        ss.set_output_cols(["AGE"])

        estimator = SnowmlLinearRegression(input_cols=input_cols, output_cols=output_cols, label_cols=label_cols)

        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])

        assert not hasattr(pipeline, "transform")
        assert not hasattr(pipeline, "fit_transform")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "fit_predict")
        assert not hasattr(pipeline, "predict_proba")
        assert not hasattr(pipeline, "predict_log_proba")
        assert hasattr(pipeline, "score")

        # fit and predict
        pipeline.fit(input_df)
        output_df = pipeline.predict(input_df)
        actual_results = output_df.to_pandas().sort_values(by="INDEX")[output_cols].to_numpy()

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

        np.testing.assert_allclose(actual_results, sk_predict_results, rtol=1.0e-1, atol=1.0e-2)

    def test_pipeline_with_classifier_estimators(self) -> None:
        input_df_pandas = load_iris(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET") and not c.startswith("INDEX")]
        label_cols = ["TARGET"]
        output_cols = ["OUTPUT"]

        estimator = SnowmlLogisticRegression(
            input_cols=input_cols, output_cols=output_cols, label_cols=label_cols, random_state=0
        )

        pipeline = snowml_pipeline.Pipeline(steps=[("estimator", estimator)])

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
        actual_results = output_df.to_pandas().sort_values(by="INDEX")[output_cols].astype(float).to_numpy().flatten()
        actual_proba = pipeline.predict_proba(input_df).to_pandas().sort_values(by="INDEX")
        actual_proba = actual_proba[[c for c in actual_proba.columns if c.find("PREDICT_PROBA_") >= 0]].to_numpy()
        actual_log_proba = pipeline.predict_log_proba(input_df).to_pandas().sort_values(by="INDEX")
        actual_log_proba = actual_log_proba[
            [c for c in actual_log_proba.columns if c.find("PREDICT_LOG_PROBA_") >= 0]
        ].to_numpy()
        actual_score = pipeline.score(input_df)

        # Do the same with SKLearn
        skpipeline = SkPipeline(steps=[("estimator", SklearnLogisticRegression(random_state=0))])

        skpipeline.fit(input_df_pandas[input_cols], input_df_pandas[label_cols])
        sk_predict_results = skpipeline.predict(input_df_pandas[input_cols])
        sk_proba = skpipeline.predict_proba(input_df_pandas[input_cols])
        sk_log_proba = skpipeline.predict_log_proba(input_df_pandas[input_cols])
        sk_score = skpipeline.score(input_df_pandas[input_cols], input_df_pandas[label_cols])

        np.testing.assert_allclose(actual_results, sk_predict_results)
        np.testing.assert_allclose(actual_proba, sk_proba, rtol=1.0e-1, atol=1.0e-2)
        np.testing.assert_allclose(actual_log_proba, sk_log_proba, rtol=1.0e-1, atol=1.0e-2)
        np.testing.assert_allclose(actual_score, sk_score)

    def test_pipeline_transform_with_pandas_dataframe(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]

        mms = MinMaxScaler(input_cols=input_cols, output_cols=input_cols)
        ss = StandardScaler(input_cols=input_cols, output_cols=input_cols)
        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss)])

        pipeline.fit(input_df)

        snow_df_output = pipeline.transform(input_df).to_pandas()
        pandas_df_output = pipeline.transform(input_df_pandas)

        assert pandas_df_output.columns.shape == snow_df_output.columns.shape
        np.testing.assert_allclose(snow_df_output[pandas_df_output.columns].to_numpy(), pandas_df_output.to_numpy())

        snow_df_output_2 = pipeline.transform(input_df[input_cols]).to_pandas()
        pandas_df_output_2 = pipeline.transform(input_df_pandas[input_cols])

        assert pandas_df_output_2.columns.shape == snow_df_output_2.columns.shape
        np.testing.assert_allclose(
            snow_df_output_2[pandas_df_output_2.columns].to_numpy(), pandas_df_output_2.to_numpy()
        )

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

        estimator = SnowmlLinearRegression(input_cols=input_cols, output_cols=output_cols, label_cols=label_cols)

        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])

        self.assertFalse(hasattr(pipeline, "transform"))
        self.assertFalse(hasattr(pipeline, "fit_transform"))
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "fit_predict"))
        self.assertFalse(hasattr(pipeline, "predict_proba"))
        self.assertFalse(hasattr(pipeline, "predict_log_proba"))
        self.assertTrue(hasattr(pipeline, "score"))

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

    def test_pipeline_signature_quoted_columns_pandas(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # Normalize column names
        input_df_pandas.columns = [f'"{inflection.parameterize(c, "_")}"' for c in input_df_pandas.columns]

        input_cols = [c for c in input_df_pandas.columns if not c.startswith('"target"')]
        label_cols = ['"target"']
        output_cols = '"output"'

        mms = MinMaxScaler()
        mms.set_input_cols(['"age"'])
        mms.set_output_cols(['"age"'])
        ss = StandardScaler()
        ss.set_input_cols(['"age"'])
        ss.set_output_cols(['"age"'])

        estimator = SnowmlLinearRegression(input_cols=input_cols, output_cols=output_cols, label_cols=label_cols)

        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])
        pipeline.fit(input_df_pandas)

        model_signatures = pipeline.model_signatures

        expected_model_signatures = {
            "predict": ModelSignature(
                inputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols],
                outputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols]
                + [FeatureSpec(name=output_cols, dtype=DataType.DOUBLE)],
            )
        }
        self.assertEqual(model_signatures["predict"].to_dict(), expected_model_signatures["predict"].to_dict())

    def test_pipeline_signature_snowpark(self) -> None:
        input_df_pandas = load_diabetes(as_frame=True).frame
        # If the pandas dataframe columns are not quoted, they will be quoted after create_dataframe.
        input_df_pandas.columns = [inflection.parameterize(c, "_") for c in input_df_pandas.columns]

        input_df = self._session.create_dataframe(input_df_pandas)

        input_cols = [c for c in input_df.columns if not c.startswith('"target"')]
        label_cols = ['"target"']
        output_cols = "OUTPUT"

        mms = MinMaxScaler()
        mms.set_input_cols(['"age"'])
        mms.set_output_cols(['"age"'])
        ss = StandardScaler()
        ss.set_input_cols(['"age"'])
        ss.set_output_cols(['"age"'])

        estimator = SnowmlLinearRegression(input_cols=input_cols, output_cols=output_cols, label_cols=label_cols)

        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])

        pipeline.fit(input_df)

        model_signatures = pipeline.model_signatures

        expected_model_signatures = {
            "predict": ModelSignature(
                inputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols],
                outputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols]
                + [FeatureSpec(name=output_cols, dtype=DataType.DOUBLE)],
            )
        }

        self.assertEqual(model_signatures["predict"].to_dict(), expected_model_signatures["predict"].to_dict())

    def test_pipeline_signature(self) -> None:
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

        estimator = SnowmlLinearRegression(input_cols=input_cols, output_cols=output_cols, label_cols=label_cols)

        pipeline = snowml_pipeline.Pipeline(steps=[("mms", mms), ("ss", ss), ("estimator", estimator)])
        pipeline.fit(input_df_pandas)

        model_signatures = pipeline.model_signatures

        expected_model_signatures = {
            "predict": ModelSignature(
                inputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols],
                outputs=[FeatureSpec(name=c, dtype=DataType.DOUBLE) for c in input_cols]
                + [FeatureSpec(name="OUTPUT", dtype=DataType.DOUBLE)],
            )
        }
        self.assertEqual(model_signatures["predict"].to_dict(), expected_model_signatures["predict"].to_dict())


if __name__ == "__main__":
    main()
