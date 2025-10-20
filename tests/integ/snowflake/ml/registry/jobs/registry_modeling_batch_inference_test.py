import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets

from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import MinMaxScaler, OneHotEncoder
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.snowpark import functions as F, types as T
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestModelingBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_snowml_logistic_regression_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"

        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(iris_X)

        test_features = iris_X[INPUT_COLUMNS]

        # Generate expected predictions using the original model
        model_output = regr.predict(iris_X)[[OUTPUT_COLUMNS]]

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regr,
            sample_input_data=test_features,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_snowml_xgboost_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"

        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(iris_X)

        test_features = iris_X[INPUT_COLUMNS]

        # Generate expected predictions using the original model
        model_output = regr.predict(iris_X)[[OUTPUT_COLUMNS]]

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regr,
            sample_input_data=test_features,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_snowml_lightgbm_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"

        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        regr.fit(iris_X)

        test_features = iris_X[INPUT_COLUMNS]

        # Generate expected predictions using the original model
        model_output = regr.predict(iris_X)[[OUTPUT_COLUMNS]]

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=regr,
            sample_input_data=test_features,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1, "cpu_requests": None},
        {"num_workers": 2, "replicas": 2, "cpu_requests": "4"},
    )
    def test_snowml_pipeline_batch_inference(
        self,
        replicas: int,
        cpu_requests: str,
        num_workers: int,
    ) -> None:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        def add_simple_category(df: pd.DataFrame) -> pd.DataFrame:
            bins = (-1, 4, 5, 6, 10)
            group_names = ["Unknown", "1_quartile", "2_quartile", "3_quartile"]
            categories = pd.cut(df.SEPALLENGTH, bins, labels=group_names)
            df["SIMPLE"] = categories
            return df

        # Add string to the dataset
        df_cat = add_simple_category(df)
        iris_df = self.session.create_dataframe(df_cat)

        fields = iris_df.schema.fields
        # Map DoubleType to DecimalType
        selected_cols = []
        count = 0
        for field in fields:
            src = field.column_identifier.quoted_name
            if isinstance(field.datatype, T.DoubleType) and count == 0:
                dest = T.DecimalType(15, 10)
                selected_cols.append(F.cast(F.col(src), dest).alias(src))
                count += 1
            else:
                selected_cols.append(F.col(src))
        iris_df = iris_df.select(selected_cols)

        numeric_features = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        categorical_features = ["SIMPLE"]
        numeric_features_output = [x + "_O" for x in numeric_features]
        label_cols = "TARGET"

        pipeline = Pipeline(
            steps=[
                (
                    "OHEHOT",
                    OneHotEncoder(input_cols=categorical_features, output_cols="CAT_OUTPUT", drop_input_cols=True),
                ),
                (
                    "SCALER",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numeric_features,
                        output_cols=numeric_features_output,
                        drop_input_cols=True,
                    ),
                ),
                (
                    "CLASSIFIER",
                    LogisticRegression(label_cols=label_cols),
                ),
            ]
        )
        pipeline.fit(iris_df)

        # Prepare test features (without label column)
        test_features = df_cat.drop(columns=[label_cols])

        # Convert categorical columns to strings to match Snowpark behavior
        for col in test_features.select_dtypes(include=["category"]).columns:
            test_features[col] = test_features[col].astype(str)

        # Generate expected predictions using the original model
        model_output = pipeline.predict(test_features)

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=pipeline,
            sample_input_data=test_features,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=cpu_requests,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="predict",
            expected_predictions=expected_predictions,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"num_workers": 1, "replicas": 1},
    )
    def test_snowml_transformers_only_pipeline_batch_inference(
        self,
        replicas: int,
        num_workers: int,
    ) -> None:
        iris = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"])
        df.columns = [s.replace(" (CM)", "").replace(" ", "") for s in df.columns.str.upper()]

        def add_simple_category(df: pd.DataFrame) -> pd.DataFrame:
            bins = (-1, 4, 5, 6, 10)
            group_names = ["Unknown", "1_quartile", "2_quartile", "3_quartile"]
            categories = pd.cut(df.SEPALLENGTH, bins, labels=group_names)
            df["SIMPLE"] = categories
            return df

        # Add string to the dataset
        df_cat = add_simple_category(df)
        iris_df = self.session.create_dataframe(df_cat)

        fields = iris_df.schema.fields
        # Map DoubleType to DecimalType
        selected_cols = []
        count = 0
        for field in fields:
            src = field.column_identifier.quoted_name
            if isinstance(field.datatype, T.DoubleType) and count == 0:
                dest = T.DecimalType(15, 10)
                selected_cols.append(F.cast(F.col(src), dest).alias(src))
                count += 1
            else:
                selected_cols.append(F.col(src))
        iris_df = iris_df.select(selected_cols)

        numeric_features = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        categorical_features = ["SIMPLE"]
        numeric_features_output = [x + "_O" for x in numeric_features]

        pipeline = Pipeline(
            steps=[
                (
                    "OHEHOT",
                    OneHotEncoder(input_cols=categorical_features, output_cols="CAT_OUTPUT", drop_input_cols=True),
                ),
                (
                    "SCALER",
                    MinMaxScaler(
                        clip=True,
                        input_cols=numeric_features,
                        output_cols=numeric_features_output,
                        drop_input_cols=True,
                    ),
                ),
            ]
        )
        pipeline.fit(iris_df)

        test_features = df_cat[categorical_features + numeric_features]

        # Convert categorical columns to strings to match Snowpark behavior
        for col in test_features.select_dtypes(include=["category"]).columns:
            test_features[col] = test_features[col].astype(str)

        # Generate expected predictions using the original model
        model_output = pipeline.transform(test_features)

        # Prepare input data and expected predictions using common function
        input_spec, expected_predictions = self._prepare_batch_inference_data(test_features, model_output)

        service_name, output_stage_location = self._prepare_service_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=pipeline,
            sample_input_data=test_features,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            num_workers=num_workers,
            service_name=service_name,
            replicas=replicas,
            function_name="transform",
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
