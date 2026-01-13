import inspect
import json
import uuid
from typing import Any, Optional

import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.jobs import job
from snowflake.ml.model import (
    JobSpec,
    ModelVersion,
    OutputSpec,
    model_signature,
    type_hints as model_types,
)
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils


class RegistryBatchInferenceTestBase(registry_spcs_test_base.RegistrySPCSTestBase):
    _INDEX_COL = "INDEX"

    def _create_stage(self) -> None:
        self._db_manager.create_stage(self._test_stage, sse_encrypted=True)  # SSE encryption for Azure

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
        prediction_assert_fn: Optional[Any] = None,
        column_handling: Optional[dict[str, Any]] = None,
    ) -> job.MLJob[Any]:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"
        options = options or {}
        options["embed_local_ml_library"] = True
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
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
            prediction_assert_fn=prediction_assert_fn,
            column_handling=column_handling,
        )

    def _deploy_batch_inference_with_image_override(
        self,
        mv: ModelVersion,
        *,
        input_spec: snowpark.DataFrame,
        output_stage_location: str,
        job_name: str,
        service_compute_pool: str,
        gpu_requests: Optional[str] = None,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        replicas: int = 1,
        force_rebuild: bool = True,
        function_name: Optional[str] = None,
    ) -> job.MLJob[Any]:
        """Deploy batch inference job with image override."""
        # Determine if we're using GPU based on gpu_requests
        is_gpu = gpu_requests is not None
        image_path = self.BASE_BATCH_GPU_IMAGE_PATH if is_gpu else self.BASE_BATCH_CPU_IMAGE_PATH
        assert image_path is not None, "Base image path must be set for batch inference image override deployment."
        database, schema, job_id = self._get_fully_qualified_service_or_job_name(job_name)
        compute_pool = sql_identifier.SqlIdentifier(service_compute_pool)
        mv._service_ops._model_deployment_spec.clear()

        self._add_common_model_deployment_spec_options(
            mv=mv,
            database=database,
            schema=schema,
            force_rebuild=force_rebuild,
        )

        # Prepare input data
        input_stage_location = f"{output_stage_location.rstrip('/')}/_temporary/"
        try:
            input_spec.write.copy_into_location(location=input_stage_location, file_format_type="parquet", header=True)
        except Exception as e:
            raise RuntimeError(f"Failed to process input_spec: {e}")

        # Add job spec
        mv._service_ops._model_deployment_spec.add_job_spec(
            job_database_name=database,
            job_schema_name=schema,
            job_name=job_id,
            inference_compute_pool_name=compute_pool,
            num_workers=num_workers,
            max_batch_rows=None,
            input_stage_location=input_stage_location,
            input_file_pattern="*",
            output_stage_location=output_stage_location,
            completion_filename="_SUCCESS",
            function_name=mv._get_function_info(function_name=function_name)["target_method"],
            warehouse=sql_identifier.SqlIdentifier(self._TEST_SPCS_WH),
            cpu=cpu_requests,
            memory=memory_requests,
            gpu=gpu_requests,
            replicas=replicas,
        )

        _, async_job = self._deploy_override_model(
            mv=mv,
            database=database,
            schema=schema,
            inference_image=image_path,
            is_batch_inference=True,
        )

        # Wait for deployment to complete
        async_job.result()

        # Return the job object
        return job.MLJob(
            id=sql_identifier.get_fully_qualified_name(database, schema, job_id),
            session=self.session,
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
        prediction_assert_fn: Optional[Any] = None,
        column_handling: Optional[dict[str, Any]] = None,
    ) -> job.MLJob[Any]:
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
            # Use image override deployment
            job = self._deploy_batch_inference_with_image_override(
                mv,
                input_spec=input_spec,
                output_stage_location=output_stage_location,
                job_name=service_name,
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
                service_compute_pool=service_compute_pool,
                num_workers=num_workers,
                replicas=replicas,
                function_name=function_name,
            )
        else:
            # Use normal deployment
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
                column_handling=column_handling,
            )

        if not blocking:
            return job

        job.wait()
        if job.status != "DONE":
            logs = job.get_logs(limit=100)
            msg = f"Job status is {job.status}, expected DONE.\n\nLast 100 lines of job logs:\n{logs}"
        else:
            msg = None
        self.assertEqual(job.status, "DONE", msg)

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

        # Apply custom validation function if provided
        if prediction_assert_fn is not None:
            actual_output = df.to_pandas()
            prediction_assert_fn(actual_output)

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

        return job

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

    def _prepare_service_name_and_stage_for_batch_inference(self) -> tuple[str, str, str]:
        """Prepare batch inference setup by generating unique identifiers and stage locations.

        Creates a unique name based on UUID and constructs the corresponding stage
        location paths for batch inference operations.

        Returns:
            tuple[str, str, str]: A tuple containing:
                - service_name: Unique identifier with underscores (replacing hyphens from UUID)
                - output_stage_location: Full stage path for batch inference output files
                - input_files_stage_location: Full stage path for input files (e.g., images, videos)
        """
        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        service_name = f"BATCH_INFERENCE_{name}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{service_name}/output/"
        input_files_stage_location = (
            f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{service_name}/input_files/"
        )

        return service_name, output_stage_location, input_files_stage_location
