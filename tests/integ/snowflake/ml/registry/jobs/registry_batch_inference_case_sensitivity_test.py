import uuid

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestModel(custom_model.CustomModel):
    """Simple model for case sensitivity testing."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": [1] * len(input)})


class RegistryBatchInferenceCaseSensitivityTest(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    def test_case_sensitive_1(self) -> None:
        model = TestModel(custom_model.ModelContext())

        # Model signature
        sample_input_data = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
            schema=[
                '"feature1"',
                '"Feature2"',
                '"FEATURE3"',
                '"feature 4"',
                '"feature_5"',
                '"feature-6"',
                '"feature7"',
            ],
        )

        # Actual input data
        input_spec = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
            schema=[
                '"FEATURE1"',
                '"FEATURE2"',
                '"FEATURE3"',
                '"FEATURE 4"',
                '"FEATURE_5"',
                '"FEATURE-6"',
                '"feature7"',
            ],
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_data,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=None,
            num_workers=1,
            service_name=f"case_sensitivity_1_{name}",
            replicas=1,
            options={"method_options": {"predict": {"case_sensitive": True}}},
        )

    def test_case_sensitive_2(self) -> None:
        model = TestModel(custom_model.ModelContext())

        # Model signature
        sample_input_data = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            schema=['"FEATURE1"', '"FEATURE2"', '"FEATURE3"', '"FEATURE 4"', '"FEATURE_5"', '"FEATURE-6"'],
        )

        # Actual input data
        input_spec = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            schema=['"feature1"', '"Feature2"', '"FEATURE3"', '"feature 4"', '"feature_5"', '"feature-6"'],
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_data,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=None,
            num_workers=1,
            service_name=f"case_sensitivity_2_{name}",
            replicas=1,
            options={"method_options": {"predict": {"case_sensitive": True}}},
        )

    def test_insensitive_model_input_signature(self) -> None:
        model = TestModel(custom_model.ModelContext())

        # Model signature
        sample_input_data = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            schema=["FEATURE1", "FEATURE2", "FEATURE3", "FEATURE_4"],
        )

        # Actual input data
        input_spec = self.session.create_dataframe(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            schema=['"feature1"', '"Feature2"', "FEATURE3", '"feature_4"'],
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_data,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=None,
            num_workers=1,
            service_name=f"case_sensitivity_3_{name}",
            replicas=1,
            options={"method_options": {"predict": {"case_sensitive": False}}},
        )

    def test_column_reordering(self) -> None:
        """Test that columns are properly reordered even with case differences."""
        model = TestModel(custom_model.ModelContext())

        # Model signature expects specific order: feature1, feature2 (lowercase)
        sample_input_data = self.session.create_dataframe([[1, 2], [3, 4]], schema=['"FEATURE1"', '"FEATURE2"'])

        # Actual input data has columns in different order and case: FEATURE2, FEATURE1 (uppercase, reversed)
        input_spec = self.session.create_dataframe([[2, 1], [4, 3], [6, 5]], schema=['"feature2"', '"FEATURE1"'])

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_data,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=None,
            num_workers=1,
            service_name=f"reorder_{name}",
            replicas=1,
            options={"method_options": {"predict": {"case_sensitive": False}}},
        )

    def test_extra_columns(self) -> None:
        """Test case insensitive matching when input data has extra columns."""
        model = TestModel(custom_model.ModelContext())

        # Model signature expects only two lowercase columns
        sample_input_data = self.session.create_dataframe([[1, 2], [3, 4]], schema=['"feature1"', '"feature2"'])

        # Actual input data has extra columns and different case
        input_spec = self.session.create_dataframe(
            [[1, 2, "extra1", 10], [3, 4, "extra2", 11]],
            schema=['"FEATURE1"', '"FEATURE2"', '"EXTRA_COL1"', '"EXTRA_COL2"'],
        )

        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sample_input_data,
            X=input_spec,
            output_stage_location=output_stage_location,
            cpu_requests=None,
            num_workers=1,
            service_name=f"extra_cols_{name}",
            replicas=1,
            options={"method_options": {"predict": {"case_sensitive": True}}},
        )


if __name__ == "__main__":
    absltest.main()
