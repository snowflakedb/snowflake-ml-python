import ast
import inspect
import json
import uuid
from typing import Any, Callable, Optional

import pandas as pd
from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.jobs import job
from snowflake.ml.model import (
    InputSpec,
    JobSpec,
    ModelVersion,
    OutputSpec,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._client.ops.service_ops import ServiceOperator
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils


def create_openai_chat_completion_output_validator(
    expected_phrases: list[str],
    test_case: absltest.TestCase,
) -> Callable[[pd.DataFrame], None]:
    """Create a validator function that checks if OpenAI chat completion output contains expected phrases.

    Args:
        expected_phrases: List of phrases that should appear in the output (case-insensitive).
        test_case: The test case instance for assertions.

    Returns:
        A validation function that takes a DataFrame and asserts expected phrases are present.
    """

    def validator(output_df: pd.DataFrame) -> None:
        # Extract content from the 'id' column which contains the choices array
        # Structure: id -> list of choices -> message -> content
        all_content = []

        # Look for 'id' or 'ID' column which contains the actual choices
        id_col = None
        for col in output_df.columns:
            if col.lower() == "id":
                id_col = col
                break

        if id_col is not None:
            for val in output_df[id_col]:
                if val is None:
                    continue

                # Parse JSON string if needed
                test_case.assertIsInstance(val, str)
                val = json.loads(val)

                # val should be a list of choice objects
                test_case.assertIsInstance(val, list)
                for choice in val:
                    test_case.assertIsInstance(choice, dict)
                    message = choice.get("message", {})
                    test_case.assertIsInstance(message, dict)
                    content = message.get("content", "")
                    test_case.assertIsInstance(content, str)
                    if content:
                        all_content.append(str(content))

        output_text = " ".join(all_content).lower()

        # If no content found, show helpful debug info
        if not output_text.strip():
            test_case.fail(f"No content found. Columns: {list(output_df.columns)}. DataFrame:\n{output_df.to_string()}")

        for phrase in expected_phrases:
            test_case.assertIn(
                phrase.lower(),
                output_text,
                f"Expected phrase '{phrase}' not found in output. Output text: {output_text[:1000]}...",
            )

    return validator


class RegistryBatchInferenceTestBase(registry_spcs_test_base.RegistrySPCSTestBase):
    _INDEX_COL = "INDEX"

    def _create_stage(self) -> None:
        self._db_manager.create_stage(self._test_stage, sse_encrypted=True)  # SSE encryption for Azure

    def _test_registry_batch_inference(
        self,
        model: model_types.SupportedModelType,
        X: snowpark.DataFrame,
        *,
        compute_pool: Optional[str] = None,
        input_spec: Optional[InputSpec] = None,
        output_spec: OutputSpec,
        job_spec: Optional[JobSpec] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        expected_predictions: Optional[pd.DataFrame] = None,
        blocking: bool = True,
        prediction_assert_fn: Optional[Any] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
        assert_container_count: Optional[int] = None,
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
            X=X,
            compute_pool=compute_pool,
            input_spec=input_spec,
            output_spec=output_spec,
            job_spec=job_spec,
            expected_predictions=expected_predictions,
            blocking=blocking,
            prediction_assert_fn=prediction_assert_fn,
            inference_engine_options=inference_engine_options,
            assert_container_count=assert_container_count,
        )

    def _deploy_batch_inference_with_image_override(
        self,
        mv: ModelVersion,
        *,
        X: snowpark.DataFrame,
        compute_pool: str,
        input_spec: Optional[InputSpec] = None,
        output_spec: OutputSpec,
        job_spec: Optional[JobSpec] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
    ) -> job.MLJob[Any]:
        """Deploy batch inference job with image override."""
        job_spec = job_spec or JobSpec()
        input_spec = input_spec or InputSpec()
        output_stage_location = output_spec.stage_location

        # Determine if we're using GPU based on gpu_requests
        is_gpu = job_spec.gpu_requests is not None
        image_path = self.BASE_BATCH_GPU_IMAGE_PATH if is_gpu else self.BASE_BATCH_CPU_IMAGE_PATH
        assert image_path is not None, "Base image path must be set for batch inference image override deployment."
        assert (
            self.BUILDER_IMAGE_PATH is not None
        ), "Builder image path must be set for batch inference image override deployment."
        assert (
            self.PROXY_IMAGE_PATH is not None
        ), "Proxy image path must be set for batch inference image override deployment."
        assert (
            self.VLLM_IMAGE_PATH is not None
        ), "vLLM image path must be set for batch inference image override deployment."
        assert (
            self.RAY_ORCHESTRATOR_PATH is not None
        ), "Ray orchestrator image path must be set for batch inference image override deployment."

        job_name = job_spec.job_name or f"BATCH_INFERENCE_{str(uuid.uuid4()).replace('-', '_').upper()}"
        database, schema, job_id = self._get_fully_qualified_service_or_job_name(job_name)
        compute_pool_id = sql_identifier.SqlIdentifier(compute_pool)
        mv._service_ops._model_deployment_spec.clear()

        self._add_common_model_deployment_spec_options(
            mv=mv,
            database=database,
            schema=schema,
            force_rebuild=job_spec.force_rebuild,
        )

        # Prepare input data
        input_stage_location = f"{output_stage_location.rstrip('/')}/_temporary/"
        try:
            X.write.copy_into_location(location=input_stage_location, file_format_type="parquet", header=True)
        except Exception as e:
            raise RuntimeError(f"Failed to process input data: {e}")

        # Add job spec
        mv._service_ops._model_deployment_spec.add_job_spec(
            job_database_name=database,
            job_schema_name=schema,
            job_name=job_id,
            inference_compute_pool_name=compute_pool_id,
            num_workers=job_spec.num_workers,
            max_batch_rows=None,
            input_stage_location=input_stage_location,
            input_file_pattern="*",
            output_stage_location=output_stage_location,
            completion_filename="_SUCCESS",
            function_name=mv._get_function_info(function_name=job_spec.function_name)["target_method"],
            warehouse=sql_identifier.SqlIdentifier(self._TEST_SPCS_WH),
            cpu=job_spec.cpu_requests,
            memory=job_spec.memory_requests,
            gpu=job_spec.gpu_requests,
            replicas=job_spec.replicas or 1,
            column_handling=ServiceOperator._encode_column_handling(input_spec.column_handling),
            params=ServiceOperator._encode_params(input_spec.params),
        )

        # Add inference engine spec if options are provided
        if inference_engine_options is not None:
            if "engine" not in inference_engine_options:
                raise ValueError("'engine' field is required in inference_engine_options")
            mv._service_ops._model_deployment_spec.add_inference_engine_spec(
                inference_engine=inference_engine_options["engine"],
                inference_engine_args=inference_engine_options.get("engine_args_override"),
            )

        # TODO: set batch inference base image here too as session parameter
        # instead of in the spec (in _deploy_override_model)

        # Set session parameters for batch inference image overrides
        batch_image_overrides: dict[str, Optional[str]] = {
            "SPCS_MODEL_RAY_ORCHESTRATOR_CONTAINER_URL": self.RAY_ORCHESTRATOR_PATH,
            "SPCS_MODEL_INFERENCE_PROXY_CONTAINER_URL": self.PROXY_IMAGE_PATH,
            "SPCS_MODEL_INFERENCE_ENGINE_CONTAINER_URLS": f'{{"vllm": "{self.VLLM_IMAGE_PATH}"}}',
            "SPCS_MODEL_INFERENCE_ENGINE_SUPPORTED_TASKS": (
                '{"vllm": ["text-generation", "image-text-to-text", ' '"video-text-to-text", "audio-text-to-text"]}'
            ),
        }

        for key, value in batch_image_overrides.items():
            self.session.sql(f"ALTER SESSION SET {key} = '{value}'").collect()

        try:
            _, async_job = self._deploy_override_model(
                mv=mv,
                database=database,
                schema=schema,
                inference_image=image_path,
                is_batch_inference=True,
            )

            # Wait for deployment to complete
            async_job.result()
        finally:
            # Unset session parameters after deployment
            for key in batch_image_overrides:
                self.session.sql(f"ALTER SESSION UNSET {key}").collect()

        # Return the job object
        return job.MLJob(
            id=sql_identifier.get_fully_qualified_name(database, schema, job_id),
            session=self.session,
        )

    def _with_image_override(self) -> bool:
        image_paths = [
            self.BUILDER_IMAGE_PATH,
            self.BASE_BATCH_CPU_IMAGE_PATH,
            self.BASE_BATCH_GPU_IMAGE_PATH,
            self.RAY_ORCHESTRATOR_PATH,
            self.PROXY_IMAGE_PATH,
            self.VLLM_IMAGE_PATH,
        ]

        all_set = all(path is not None for path in image_paths)
        all_unset = all(path is None for path in image_paths)

        if not all_set and not all_unset:
            raise ValueError(
                "Please set or unset all batch inference image override environment variables at the same time: "
                "BUILDER_IMAGE_PATH, BASE_BATCH_CPU_IMAGE_PATH, BASE_BATCH_GPU_IMAGE_PATH, "
                "RAY_ORCHESTRATOR_PATH, PROXY_IMAGE_PATH, and VLLM_IMAGE_PATH."
            )

        return all_set

    def _deploy_batch_inference(
        self,
        mv: ModelVersion,
        X: snowpark.DataFrame,
        *,
        compute_pool: Optional[str] = None,
        input_spec: Optional[InputSpec] = None,
        output_spec: OutputSpec,
        job_spec: Optional[JobSpec] = None,
        expected_predictions: Optional[pd.DataFrame] = None,
        blocking: bool = True,
        prediction_assert_fn: Optional[Any] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
        assert_container_count: Optional[int] = None,
    ) -> job.MLJob[Any]:

        with_image_override = self._with_image_override()

        # Resolve compute pool if not provided
        job_spec = job_spec or JobSpec()
        if compute_pool is None:
            compute_pool = self._TEST_CPU_COMPUTE_POOL if job_spec.gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

        if with_image_override:
            # Use image override deployment
            batch_job = self._deploy_batch_inference_with_image_override(
                mv,
                X=X,
                compute_pool=compute_pool,
                input_spec=input_spec,
                output_spec=output_spec,
                job_spec=job_spec,
                inference_engine_options=inference_engine_options,
            )
        else:
            # Use normal deployment
            # Merge image_repo into job_spec if not already set
            if job_spec.image_repo is None:
                job_spec = JobSpec(
                    **{
                        **job_spec.model_dump(),
                        "image_repo": ".".join([self._test_db, self._test_schema, self._test_image_repo]),
                    }
                )

            batch_job = mv.run_batch(
                X,
                compute_pool=compute_pool,
                input_spec=input_spec,
                output_spec=output_spec,
                job_spec=job_spec,
                inference_engine_options=inference_engine_options,
            )

        if not blocking:
            return batch_job

        output_stage_location = output_spec.stage_location

        batch_job.wait()
        if batch_job.status != "DONE":
            logs = batch_job.get_logs(limit=100)
            msg = f"Job status is {batch_job.status}, expected DONE.\n\nLast 100 lines of job logs:\n{logs}"
        else:
            msg = None
        self.assertEqual(batch_job.status, "DONE", msg)

        # Validate container count if specified
        if assert_container_count is not None:
            containers = batch_job._service_spec["spec"]["containers"]
            self.assertEqual(
                len(containers),
                assert_container_count,
                f"Expected {assert_container_count} containers but got {len(containers)}: {containers}",
            )

        success_file_path = output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file_path}").collect()
        self.assertGreater(len(list_results), 0, f"Batch job did not produce success file at: {success_file_path}")

        # todo: add more logic to validate the outcome
        df = self.session.read.option("on_error", "CONTINUE").parquet(output_stage_location)
        self.assertEqual(
            df.count(),
            X.count(),
            f"Output row count ({df.count()}) does not match input row count ({X.count()})",
        )

        # Apply custom validation function if provided
        if prediction_assert_fn is not None:
            actual_output = df.to_pandas()
            prediction_assert_fn(actual_output)

        # Compare expected and actual output if provided
        if expected_predictions is not None:
            # Convert Snowpark DataFrame to pandas for comparison
            actual_output = df.to_pandas()

            # Apply json.loads to parse string representations of arrays/objects
            # Snowflake may return nested arrays as JSON strings when reading from parquet
            def try_parse_json(x: Any) -> Any:
                if isinstance(x, str):
                    x_stripped = x.strip()
                    if x_stripped.startswith(("[", "{")):
                        try:
                            return json.loads(x)
                        except json.JSONDecodeError:
                            pass
                        # Fallback to ast.literal_eval for Python-style syntax (handles trailing commas)
                        try:
                            return ast.literal_eval(x)
                        except (ValueError, SyntaxError):
                            pass
                return x

            for col in actual_output.columns:
                actual_output[col] = actual_output[col].apply(try_parse_json)

            # Also apply to expected_predictions as it may contain data from Snowflake
            for col in expected_predictions.columns:
                expected_predictions[col] = expected_predictions[col].apply(try_parse_json)

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

        return batch_job

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

    def _prepare_job_name_and_stage_for_batch_inference(self) -> tuple[str, str, str]:
        """Prepare job name and stage locations for batch inference.

        Creates a unique job name and constructs the corresponding stage location paths.

        Returns:
            tuple[str, str, str]: A tuple containing:
                - job_name: Unique job name identifier
                - output_stage_location: Full stage path for batch inference output files
                - input_files_stage_location: Full stage path for input files (e.g., images, videos)
        """
        job_name = f"BATCH_INFERENCE_{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{job_name}/output/"
        input_files_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{job_name}/input_files/"

        return job_name, output_stage_location, input_files_stage_location
