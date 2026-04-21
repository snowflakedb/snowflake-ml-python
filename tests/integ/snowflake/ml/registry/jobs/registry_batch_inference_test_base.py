import ast
import inspect
import json
import logging
import uuid
from typing import Any, Callable, Literal, Optional

import pandas as pd
from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.jobs import job
from snowflake.ml.model import ModelVersion, model_signature, type_hints as model_types
from snowflake.ml.model.batch import InputSpec, JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry import registry_spcs_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils

logger = logging.getLogger(__name__)

# Which env-var bundle ``_has_image_override`` consults. Subclasses (e.g. pip-only batch tests)
# set ``_BATCH_IMAGE_OVERRIDE_MODE`` instead of overriding ``_has_image_override``.
BatchImageOverrideMode = Literal["full", "pip_only_batch"]


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
    _BATCH_IMAGE_OVERRIDE_MODE: BatchImageOverrideMode = "full"

    def _get_batch_image_override_session_params(self) -> dict[str, str]:
        overrides: dict[str, Optional[str]] = {
            "SPCS_MODEL_BUILD_CONTAINER_URL": self.BUILDER_IMAGE_PATH,
            "SPCS_MODEL_BASE_CPU_BATCH_INFERENCE_CONTAINER_URL": self.BASE_BATCH_CPU_IMAGE_PATH,
            "SPCS_MODEL_BASE_GPU_BATCH_INFERENCE_CONTAINER_URL": self.BASE_BATCH_GPU_IMAGE_PATH,
            "SPCS_MODEL_RAY_ORCHESTRATOR_CONTAINER_URL": self.RAY_ORCHESTRATOR_PATH,
            "SPCS_MODEL_INFERENCE_PROXY_CONTAINER_URL": self.PROXY_IMAGE_PATH,
            "SPCS_MODEL_INFERENCE_ENGINE_VLLM_URL": self.VLLM_IMAGE_PATH,
        }
        return {k: v for k, v in overrides.items() if v is not None}

    def setUp(self) -> None:
        super().setUp()
        if self._has_image_override():
            for key, value in self._get_batch_image_override_session_params().items():
                self.session.sql(f"ALTER SESSION SET {key} = '{value}'").collect()

    def tearDown(self) -> None:
        if self._has_image_override():
            for key in self._get_batch_image_override_session_params():
                self.session.sql(f"ALTER SESSION UNSET {key}").collect()
        super().tearDown()

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
        target_platforms: Optional[list[str]] = None,
        skip_row_count_check: bool = False,
        model_name: Optional[str] = None,
        version_name: Optional[str] = None,
        python_version: Optional[str] = None,
        conda_dependencies: Optional[list[str]] = None,
    ) -> job.MLJob[Any]:
        # If conda_dependencies is not explicitly provided, add the default snowpark-python dependency.
        # Pass an empty list to skip conda dependencies (for pip-only tests).
        if conda_dependencies is None:
            conda_dependencies = [
                test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python")
            ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = model_name or f"model_{inspect.stack()[1].function}"
        version = version_name or f"ver_{self._run_id}"
        options = options or {}
        options.setdefault("embed_local_ml_library", True)
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=target_platforms or ["SNOWPARK_CONTAINER_SERVICES"],
            options=options,
            signatures=signatures,
            python_version=python_version,
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
            skip_row_count_check=skip_row_count_check,
        )

    def _has_image_override(self, mode: Optional[BatchImageOverrideMode] = None) -> bool:
        """Return whether session image overrides are enabled for the given mode.

        Args:
            mode: Which bundle of ``*_IMAGE_*PATH`` / ``*_PATH`` env vars must be set together.
                If None, uses ``self._BATCH_IMAGE_OVERRIDE_MODE`` (``\"full\"`` on this base class;
                pip-only tests set ``\"pip_only_batch\"``). If not None, that mode is used instead of
                the class default.

        Returns:
            True if the bundle is fully set, False if fully unset.

        Raises:
            ValueError: If the chosen mode's env vars are only partially set, or if ``mode`` (or
                ``self._BATCH_IMAGE_OVERRIDE_MODE`` when ``mode`` is None) is not a known value.
        """
        effective_mode = mode if mode is not None else self._BATCH_IMAGE_OVERRIDE_MODE

        if effective_mode == "full":
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

        if effective_mode == "pip_only_batch":
            image_paths = [
                self.BUILDER_IMAGE_PATH,
                self.BASE_BATCH_CPU_IMAGE_PATH,
                self.BASE_BATCH_GPU_IMAGE_PATH,
                self.MODEL_LOGGER_PATH,
            ]
            if all(image_paths):
                return True
            if not any(image_paths):
                return False
            raise ValueError(
                "Please set or unset BUILDER_IMAGE_PATH, BASE_BATCH_CPU_IMAGE_PATH, "
                "BASE_BATCH_GPU_IMAGE_PATH, and MODEL_LOGGER_PATH at the same time."
            )

        raise ValueError(f"Unknown batch image override mode: {effective_mode!r}")

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
        skip_row_count_check: bool = False,
    ) -> job.MLJob[Any]:
        # Resolve compute pool if not provided
        job_spec = job_spec or JobSpec()
        if compute_pool is None:
            compute_pool = self._TEST_CPU_COMPUTE_POOL if job_spec.gpu_requests is None else self._TEST_GPU_COMPUTE_POOL

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

        try:
            batch_job.wait(timeout=1800)
        except TimeoutError:
            logger.warning(f"Batch job timed out after 30 minutes. Status: {batch_job.status}")
        if batch_job.status != "DONE":
            logs = batch_job.get_logs(limit=100)
            msg = (
                f"Job {batch_job.id} status is {batch_job.status}, expected DONE.\n\n"
                f"Last 100 lines of job logs:\n{logs}"
            )

            # Also fetch logs from proxy and model-inference containers if there are multiple containers
            containers = batch_job._service_spec.get("spec", {}).get("containers", [])
            container_names = {c["name"] for c in containers}
            for container_name in ("proxy", "model-inference"):
                if len(containers) > 1 and container_name in container_names:
                    try:
                        from snowflake.ml.jobs.job import _get_logs

                        container_logs = _get_logs(self.session, batch_job.id, limit=100, container_name=container_name)
                        msg += f"\n\nLast 100 lines of {container_name} logs:\n{container_logs}"
                    except Exception as e:
                        msg += f"\n\nFailed to get {container_name} logs: {e}"
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
        if not skip_row_count_check:
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
