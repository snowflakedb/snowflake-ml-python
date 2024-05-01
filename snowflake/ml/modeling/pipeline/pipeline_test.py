import os

import pandas as pd
from absl.testing import absltest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression as sklearn_LR
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.preprocessing import MinMaxScaler as sklearn_MMS

from snowflake.ml.modeling.lightgbm import LGBMClassifier
from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.modeling.pipeline.pipeline import IN_ML_RUNTIME_ENV_VAR, Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.snowpark import DataFrame


class PipelineTest(absltest.TestCase):
    def setUp(self) -> None:
        self.dataframe_snowpark = absltest.mock.MagicMock(spec=DataFrame)
        self.simple_pipeline = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col2"]),
                ),
            ]
        )

        self.pipeline_two_steps_no_estimator = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1"]),
                ),
            ]
            * 2
        )

        self.pipeline_two_steps_with_estimator = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1"]),
                ),
                ("model", LinearRegression(label_cols=["col3"])),
            ]
        )
        return super().setUp()

    def test_dataset_can_be_trained_in_ml_runtime(self) -> None:
        """Test that the pipeline can only be trained in ml runtime if correct dataset type and
        environment variables present.
        """

        assert self.simple_pipeline._can_be_trained_in_ml_runtime(dataset=pd.DataFrame()) is False
        assert self.simple_pipeline._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark) is False

        os.environ[IN_ML_RUNTIME_ENV_VAR] = "True"
        assert self.simple_pipeline._can_be_trained_in_ml_runtime(dataset=pd.DataFrame()) is False
        assert self.simple_pipeline._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark) is True

        del os.environ[IN_ML_RUNTIME_ENV_VAR]

    def test_pipeline_can_be_trained_in_ml_runtime(self) -> None:
        """Test that the pipeline can be trained in the ml runtime if it has the correct configuration
        of steps.
        """
        os.environ[IN_ML_RUNTIME_ENV_VAR] = "True"

        assert self.simple_pipeline._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark) is True

        pipeline_three_steps = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1"]),
                ),
            ]
            * 3
        )

        assert pipeline_three_steps._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark) is False

        assert (
            self.pipeline_two_steps_no_estimator._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark) is False
        )

        assert (
            self.pipeline_two_steps_with_estimator._can_be_trained_in_ml_runtime(dataset=self.dataframe_snowpark)
            is True
        )

        del os.environ[IN_ML_RUNTIME_ENV_VAR]

    def test_wrap_transformer_in_column_transformer(self):
        input_cols = ["col1"]
        transformer = MinMaxScaler(input_cols=input_cols, output_cols=["col2"])
        transformer_name = "MMS"

        wrapped_transformer = Pipeline._wrap_transformer_in_column_transformer(transformer_name, transformer)
        assert isinstance(wrapped_transformer, ColumnTransformer)
        assert len(wrapped_transformer.transformers) == 1

        inner_transformer = wrapped_transformer.transformers[0]
        assert inner_transformer[0] == transformer_name
        assert isinstance(inner_transformer[1], sklearn_MMS)
        assert inner_transformer[2] == input_cols

    def test_create_unfitted_sklearn_object(self) -> None:
        sklearn_pipeline = self.pipeline_two_steps_with_estimator._create_unfitted_sklearn_object()
        assert isinstance(sklearn_pipeline, sklearn_Pipeline)
        for transformer_step in sklearn_pipeline.steps[:-1]:
            assert isinstance(transformer_step[1], ColumnTransformer)

        pipeline_three_steps_with_estimator = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1"]),
                ),
                (
                    "MMS2",
                    MinMaxScaler(),
                ),
                ("model", LinearRegression()),
            ]
        )
        sklearn_pipeline = pipeline_three_steps_with_estimator._create_unfitted_sklearn_object()
        assert isinstance(sklearn_pipeline, sklearn_Pipeline)

        skl_pipeline_steps = sklearn_pipeline.steps
        assert isinstance(skl_pipeline_steps[0][1], ColumnTransformer)
        assert isinstance(skl_pipeline_steps[1][1], sklearn_MMS)
        assert isinstance(skl_pipeline_steps[2][1], sklearn_LR)

    def test_get_native_object(self) -> None:
        sklearn_type = LinearRegression(input_cols=["col1, col2"], label_cols=["col3"])
        Pipeline._get_native_object(sklearn_type)

        xgb_type = XGBRegressor(input_cols=["col1, col2"], label_cols=["col3"])
        Pipeline._get_native_object(xgb_type)

        lgbm_type = LGBMClassifier(input_cols=["col1, col2"], label_cols=["col3"])
        Pipeline._get_native_object(lgbm_type)

        with self.assertRaises(ValueError):
            Pipeline._get_native_object(pd.DataFrame())

    def test_get_label_cols(self) -> None:
        assert self.simple_pipeline._get_label_cols() == []

        assert self.pipeline_two_steps_no_estimator._get_label_cols() == []

        assert len(self.pipeline_two_steps_with_estimator._get_label_cols()) == 1

    def test_is_pipeline_modifying_label_or_sample_weight(self) -> None:
        """Tests whether the pipeline modifies either the label or sample weight columns."""
        assert self.simple_pipeline._is_pipeline_modifying_label_or_sample_weight() is False

        pipeline_modifying_label = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1_out"]),
                ),
                (
                    "MMS2",
                    MinMaxScaler(input_cols=["col3"], output_cols=["col3_out"]),
                ),
                ("model", LinearRegression(input_cols=["col1"], label_cols=["col3"])),
            ]
        )
        assert pipeline_modifying_label._is_pipeline_modifying_label_or_sample_weight() is True

        pipeline_modifying_sample_weight = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1_out"]),
                ),
                (
                    "MMS2",
                    MinMaxScaler(input_cols=["col3"], output_cols=["col3_out"]),
                ),
                ("model", LinearRegression(input_cols=["col1"], sample_weight_col="col3")),
            ]
        )
        assert pipeline_modifying_sample_weight._is_pipeline_modifying_label_or_sample_weight() is True

    def test_is_convertible_to_sklearn_object(self) -> None:
        assert self.simple_pipeline._is_convertible_to_sklearn_object() is True
        assert self.pipeline_two_steps_with_estimator._is_convertible_to_sklearn_object() is True

        pipeline_second_step_uses_input_cols = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1_out"]),
                ),
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col_2"], output_cols=["col_2_out"]),
                ),
            ]
        )
        assert pipeline_second_step_uses_input_cols._is_convertible_to_sklearn_object() is False

        pipeline_modifying_label = Pipeline(
            steps=[
                (
                    "MMS",
                    MinMaxScaler(input_cols=["col1"], output_cols=["col1_out"]),
                ),
                (
                    "MMS2",
                    MinMaxScaler(input_cols=["col3"], output_cols=["col3_out"]),
                ),
                ("model", LinearRegression(input_cols=["col1"], label_cols=["col3"])),
            ]
        )
        assert pipeline_second_step_uses_input_cols._is_convertible_to_sklearn_object() is False

        pipeline_inner_step_is_not_convertible = Pipeline(steps=[("pipeline", pipeline_modifying_label)])
        assert pipeline_inner_step_is_not_convertible._is_convertible_to_sklearn_object() is False

    def test_to_sklearn(self) -> None:
        """Tests behavior for converting the pipeline to an sklearn pipeline"""
        assert isinstance(self.simple_pipeline.to_sklearn(), sklearn_Pipeline)

    def tearDown(self) -> None:
        os.environ.pop(IN_ML_RUNTIME_ENV_VAR, None)
        return super().tearDown()


if __name__ == "__main__":
    absltest.main()
