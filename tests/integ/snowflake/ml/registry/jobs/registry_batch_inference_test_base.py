import inspect
import json
import logging
import os
import uuid
from typing import Any, Optional

import pandas as pd
import pytest
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake import snowpark
from snowflake.ml import jobs
from snowflake.ml.model import (
    JobSpec,
    ModelVersion,
    OutputSpec,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)


@pytest.mark.spcs_deployment_image
class RegistryBatchInferenceTestBase(common_test_base.CommonTestBase):
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _TEST_SPCS_WH = "REGTEST_ML_SMALL"
    _INDEX_COL = "INDEX"

    BUILDER_IMAGE_PATH = os.getenv("BUILDER_IMAGE_PATH", None)
    BASE_CPU_IMAGE_PATH = os.getenv("BASE_CPU_IMAGE_PATH", None)
    BASE_GPU_IMAGE_PATH = os.getenv("BASE_GPU_IMAGE_PATH", None)

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        # Get login options BEFORE session creation (which clears password for security)
        login_options = connection_params.SnowflakeLoginOptions()

        # Capture password from login options before session creation clears it
        pat_token = login_options.get("password")

        # Now create session (this will clear password in session._conn._lower_case_parameters)
        super().setUp()

        # Set log level to INFO so that service logs are visible
        logging.basicConfig(level=logging.INFO)

        # Read private_key_path from session connection parameters (after session creation)
        conn_params = self.session._conn._lower_case_parameters
        private_key_path = conn_params.get("private_key_path")

        if private_key_path:
            # Try to load private key for JWT authentication
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=backends.default_backend()
                )
            self.pat_token = None
        elif pat_token:
            # Use PAT token from password parameter
            self.private_key = None
            self.pat_token = pat_token
        else:
            # No authentication credentials available
            self.private_key = None
            self.pat_token = None
            raise ValueError("No authentication credentials found: neither private_key_path nor password parameter set")

        self.snowflake_account_url = self.session._conn._lower_case_parameters.get("host", None)
        if self.snowflake_account_url:
            self.snowflake_account_url = f"https://{self.snowflake_account_url}"

        self._run_id = uuid.uuid4().hex[:2]
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = "PUBLIC"
        self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "image_repo"
        ).upper()
        self._test_stage = "TEST_STAGE"

        self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_stage(self._test_stage, sse_encrypted=True)  # SSE encryption for Azure
        self._db_manager.create_image_repo(self._test_image_repo)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _test_registry_batch_inference(
        self,
        model: model_types.SupportedModelType,
        input_spec: snowpark.DataFrame,
        output_stage_location: str,
        service_name: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        gpu_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        replicas: Optional[int] = 1,
        force_rebuild: bool = True,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        use_default_repo: bool = False,
        function_name: Optional[str] = None,
        expected_predictions: Optional[pd.DataFrame] = None,
        blocking: bool = True,
    ) -> jobs.MLJob[Any]:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            options=options,
            signatures=signatures,
        )

        return self._deploy_batch_inference(
            mv,
            input_spec=input_spec,
            output_stage_location=output_stage_location,
            service_name=service_name,
            gpu_requests=gpu_requests,
            cpu_requests=cpu_requests,
            memory_requests=memory_requests,
            service_compute_pool=service_compute_pool,
            num_workers=num_workers,
            replicas=replicas,
            function_name=function_name,
            expected_predictions=expected_predictions,
            blocking=blocking,
            use_default_repo=use_default_repo,
        )

    def _deploy_batch_inference(
        self,
        mv: ModelVersion,
        input_spec: snowpark.DataFrame,
        output_stage_location: str,
        service_name: str,
        gpu_requests: Optional[str] = None,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        service_compute_pool: Optional[str] = None,
        num_workers: Optional[int] = None,
        replicas: int = 1,
        function_name: Optional[str] = None,
        expected_predictions: Optional[pd.DataFrame] = None,
        blocking: bool = True,
        use_default_repo: bool = False,
    ) -> jobs.MLJob[Any]:
        if self.BUILDER_IMAGE_PATH and self.BASE_CPU_IMAGE_PATH and self.BASE_GPU_IMAGE_PATH:
            with_image_override = True
        elif not self.BUILDER_IMAGE_PATH and not self.BASE_CPU_IMAGE_PATH and not self.BASE_GPU_IMAGE_PATH:
            with_image_override = False
        else:
            raise ValueError(
                "Please set or unset BUILDER_IMAGE_PATH, BASE_CPU_IMAGE_PATH, and BASE_GPU_IMAGE_PATH at the same time."
            )

        if service_name is None:
            service_name = f"service_{inspect.stack()[1].function}_{self._run_id}"
        if service_compute_pool is None:
            service_compute_pool = self._TEST_CPU_COMPUTE_POOL if gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

        if with_image_override:
            """
            self._deploy_model_with_image_override(
                mv,
                service_name=service_name,
                service_compute_pool=sql_identifier.SqlIdentifier(service_compute_pool),
                gpu_requests=gpu_requests,
                num_workers=num_workers,
                max_instances=max_instances,
                max_batch_rows=max_batch_rows,
                force_rebuild=False,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
            )
            """
            # TODO: implement this
            pass
        else:
            job = mv.run_batch(
                compute_pool=service_compute_pool,
                input_spec=input_spec,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(
                    image_repo=(
                        None
                        if use_default_repo
                        else ".".join([self._test_db, self._test_schema, self._test_image_repo])
                    ),
                    job_name=service_name,
                    num_workers=num_workers,
                    gpu_requests=gpu_requests,
                    cpu_requests=cpu_requests,
                    memory_requests=memory_requests,
                    replicas=replicas,
                    function_name=function_name,
                ),
            )
            if blocking:
                job.wait()
            else:
                return job

        self.assertEqual(job.status, "DONE")

        success_file_path = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file_path}").collect()
        self.assertGreater(len(list_results), 0, f"Batch job did not produce success file at: {success_file_path}")

        # todo: add more logic to validate the outcome
        df = self.session.read.option("on_error", "CONTINUE").parquet(output_stage_location)
        self.assertEqual(
            df.count(),
            input_spec.count(),
            f"Output row count ({df.count()}) does not match input row count ({input_spec.count()})",
        )

        # Compare expected and actual output if provided
        if expected_predictions is not None:
            # Convert Snowpark DataFrame to pandas for comparison
            actual_output = df.to_pandas()

            # Apply json.loads to parse string representations of arrays
            for col in actual_output.columns:
                if col.startswith("output_feature_"):
                    actual_output[col] = actual_output[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

            # Sort both dataframes by the index column for consistent comparison
            self.assertTrue(self._INDEX_COL in expected_predictions.columns)
            self.assertTrue(self._INDEX_COL in actual_output.columns)
            expected_predictions = expected_predictions.sort_values(self._INDEX_COL).reset_index(drop=True)
            actual_output = actual_output.sort_values(self._INDEX_COL).reset_index(drop=True)

            # Order columns consistently
            expected_columns = sorted(expected_predictions.columns)
            actual_columns = sorted(actual_output.columns)

            # Ensure both dataframes have the same columns
            self.assertEqual(
                set(expected_columns),
                set(actual_columns),
                f"Expected columns {expected_columns} do not match actual columns {actual_columns}",
            )

            # Reorder columns to match
            actual_output = actual_output[expected_columns]

            # Compare the dataframes
            pd.testing.assert_frame_equal(
                expected_predictions,
                actual_output,
                check_dtype=False,
                check_exact=False,
                rtol=1e-3,
                atol=1e-6,
            )

        return mv

    def _prepare_batch_inference_data(
        self,
        input_pandas_df: pd.DataFrame,
        model_output: pd.DataFrame,
    ) -> tuple[snowpark.DataFrame, pd.DataFrame]:
        """Prepare input data with an index column and expected predictions.

        Args:
            input_pandas_df: Input data as pandas DataFrame
            model_output: Model predictions as pandas DataFrame

        Returns:
            Tuple of (input_spec, expected_predictions)
        """
        # Create input data with an index column for deterministic ordering
        input_with_index = input_pandas_df.copy()
        input_with_index[self._INDEX_COL] = range(len(input_pandas_df))

        # Convert to Snowpark DataFrame
        input_spec = self.session.create_dataframe(input_with_index)

        # Generate expected predictions by concatenating input data with model output
        # Reset both indices to ensure proper alignment
        expected_predictions = input_with_index.reset_index(drop=True)
        model_output_reset = model_output.reset_index(drop=True)
        expected_predictions = pd.concat([expected_predictions, model_output_reset], axis=1)

        # Sort columns to match the actual output order
        expected_predictions = expected_predictions.reindex(columns=sorted(expected_predictions.columns))

        return input_spec, expected_predictions

    def _prepare_service_name_and_stage_for_batch_inference(self) -> tuple[str, str]:
        """Prepare batch inference setup by generating unique identifiers and output stage location.

        Creates a unique name based on UUID and constructs the corresponding output stage
        location path for batch inference operations.

        Returns:
            tuple[str, str]: A tuple containing:
                - service_name: Unique identifier with underscores (replacing hyphens from UUID)
                - output_stage_location: Full stage path for batch inference output files
        """
        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        service_name = f"BATCH_INFERENCE_{name}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{service_name}/output/"

        return service_name, output_stage_location
