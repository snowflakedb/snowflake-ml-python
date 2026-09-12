"""Deploy a Model Garden LLM."""

import json
import logging
import time
import uuid
from typing import Any

import pandas as pd
from absl.testing import absltest
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization

from snowflake.ml._internal.utils import connection_params
from snowflake.ml.model import inference_engine
from snowflake.ml.model._client.model import (
    batch_inference_job_specs,
    model_version_impl,
)
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager

logger = logging.getLogger(__name__)

_MODEL_GARDEN_MODEL_NAME = '"TINYLLAMA-1.1B-CHAT-V1.0-20260902"'
_FAILED_SERVICE_STATUSES = frozenset({"FAILED", "FAILED_CLEANING_UP", "INTERNAL_ERROR"})
_GPU_REQUESTS = "1"
_CPU_REQUESTS = "4"
_MEMORY_REQUESTS = "16Gi"
_NODE_COUNT = 2


def _maybe_json_load(value: Any) -> Any:
    """Parse JSON objects/arrays stored as strings (common in parquet VARIANT columns)."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


class TestRegistryModelGardenDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    def setUp(self) -> None:
        login_options = connection_params.SnowflakeLoginOptions()
        pat_token = login_options.get("password")
        common_test_base.CommonTestBase.setUp(self)
        logging.basicConfig(level=logging.INFO)

        conn_params = self.session._conn._lower_case_parameters
        private_key_path = conn_params.get("private_key_path")
        if private_key_path:
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=backends.default_backend()
                )
            self.pat_token = None
        elif pat_token:
            self.private_key = None
            self.pat_token = pat_token
        else:
            self.private_key = None
            self.pat_token = None
            raise ValueError("No authentication credentials found: neither private_key_path nor password parameter set")

        self.snowflake_account_url = self.session._conn._lower_case_parameters.get("host", None)
        if self.snowflake_account_url:
            self.snowflake_account_url = f"https://{self.snowflake_account_url}"

        self._run_id = uuid.uuid4().hex[:4]
        self._service_short_name = f"MG_TINYLLAMA_{self._run_id}".upper()
        current_role = str(self.session.sql("SELECT CURRENT_ROLE() AS R").collect()[0]["R"]).strip('"')
        self._created_ephemeral_db = current_role.upper() != "PUBLIC"

        if not self.session.get_current_warehouse():
            self.session.sql(f"USE WAREHOUSE {self._TEST_SPCS_WH}").collect()

        self._db_manager = db_manager.DBManager(self.session)
        if self._created_ephemeral_db:
            self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
            self._test_schema = "PUBLIC"
            self._test_image_repo = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                self._run_id, "image_repo"
            ).upper()
            self._test_stage = "TEST_STAGE"
            self._db_manager.create_database(self._test_db)
            self._create_stage()
            self._db_manager.create_image_repo(self._test_image_repo)
            self._db_manager.cleanup_databases(expire_hours=6)
        else:
            current_db = self.session.get_current_database()
            current_schema = self.session.get_current_schema()
            self._test_db = str(current_db).strip('"') if current_db else "HAYU"
            self._test_schema = str(current_schema).strip('"') if current_schema else "PUBLIC"
            self._test_stage = f"MG_STAGE_{self._run_id}".upper()
            self._create_stage()
            logger.info("PUBLIC role: using existing %s.%s (no CREATE DATABASE)", self._test_db, self._test_schema)

        self.registry = registry.Registry(self.session, database_name=self._test_db, schema_name=self._test_schema)
        if self._has_image_override():
            for key, value in self._get_image_override_session_params().items():
                self.session.sql(f"ALTER SESSION SET {key} = '{value}'").collect()

    def tearDown(self) -> None:
        if self._has_image_override():
            for key in self._get_image_override_session_params():
                self.session.sql(f"ALTER SESSION UNSET {key}").collect()
        if getattr(self, "_created_ephemeral_db", True):
            super().tearDown()
            return
        try:
            self.session.sql(
                f"DROP SERVICE IF EXISTS {self._test_db}.{self._test_schema}.{self._service_short_name}"
            ).collect()
        except Exception as e:
            logger.warning("Could not drop PUBLIC-role trial service: %s", e)
        try:
            self.session.sql(f"DROP STAGE IF EXISTS {self._test_db}.{self._test_schema}.{self._test_stage}").collect()
        except Exception as e:
            logger.warning("Could not drop PUBLIC-role trial stage: %s", e)
        common_test_base.CommonTestBase.tearDown(self)

    def _snowflake_models_names(self) -> set[str]:
        rows = self.session.sql("SHOW MODELS IN SNOWFLAKE.MODELS").collect()
        names = set()
        for row in rows:
            row_dict = row.as_dict()
            raw_name = row_dict.get("name")
            if raw_name is None:
                raw_name = row_dict.get("NAME")
            if raw_name is None:
                continue
            names.add(str(raw_name).strip('"'))
        return names

    def _resolve_model_garden_model(self) -> str | None:
        if _MODEL_GARDEN_MODEL_NAME.strip('"') in self._snowflake_models_names():
            return _MODEL_GARDEN_MODEL_NAME
        return None

    def _wait_for_named_service_status(
        self,
        service_short_name: str,
        *,
        expected_status: str,
        timeout_s: float = 1800.0,
        poll_interval_s: float = 10.0,
    ) -> str:
        """Poll SHOW SERVICES for this test's service until ``expected_status`` or failure."""
        deadline = time.time() + timeout_s
        last_status = "<unknown>"
        escaped_name = service_short_name.replace("'", "''")
        query = f"SHOW SERVICES LIKE '{escaped_name}' IN SCHEMA {self._test_db}.{self._test_schema}"
        while time.time() < deadline:
            rows = self.session.sql(query).collect()
            if not rows:
                last_status = "PENDING"
            else:
                row_dict = rows[0].as_dict()
                last_status = str(row_dict.get("status") or row_dict.get("STATUS") or "<unknown>")
            logger.info("Service %s status=%s", service_short_name, last_status)
            if last_status == expected_status:
                return last_status
            if last_status in _FAILED_SERVICE_STATUSES:
                raise AssertionError(f"Inference service reached {last_status!r} while waiting for {expected_status}.")
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Service {service_short_name} did not reach {expected_status} within "
            f"{timeout_s:.0f}s (last_status={last_status!r})."
        )

    def _retry_on_transient_inference(self, fn: Any) -> Any:
        max_retries = 3
        retry_delays = [30, 60, 90]
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "connection refused" in error_str.lower() or "502" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delays[attempt]
                        logger.warning(
                            "Inference failed (attempt %s/%s): %s. Retrying in %s seconds...",
                            attempt + 1,
                            max_retries,
                            e,
                            wait_time,
                        )
                        time.sleep(wait_time)
                        continue
                raise
        raise last_error  # pragma: no cover

    def _invoke_service_sql(self, service_name: str) -> Any:
        query = f"""
            SELECT {service_name}!__CALL__(
              [{{'role': 'user', 'content': [{{'type': 'text', 'text': 'Say hello in one short sentence.'}}]}}],
              0.2::FLOAT,
              64,
              [],
              1,
              FALSE,
              1.0::FLOAT,
              0.0::FLOAT,
              0.0::FLOAT,
              NULL
            ) AS RESULT
        """

        def _run() -> Any:
            rows = self.session.sql(query).collect()
            self.assertGreater(len(rows), 0)
            row_dict = rows[0].as_dict()
            result = row_dict.get("RESULT")
            if result is None:
                result = row_dict.get("result")
            return result

        return self._retry_on_transient_inference(_run)

    def _openai_call_input(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                    "temperature": 0.2,
                    "max_completion_tokens": 64,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "response_format": None,
                }
            ]
        )

    def _model_garden_model_version(self, model_name: str) -> model_version_impl.ModelVersion:
        model_garden_registry = registry.Registry(self.session, database_name="SNOWFLAKE", schema_name="MODELS")
        return model_garden_registry.get_model(model_name).default

    def _invoke_mv_run(self, service_name: str, model_name: str) -> pd.DataFrame:
        mv = self._model_garden_model_version(model_name)
        return self._retry_on_transient_inference(
            lambda: mv.run(self._openai_call_input(), function_name="__call__", service_name=service_name)
        )

    def _enable_ai_complete_session_params(self) -> None:
        for stmt in (
            "ALTER SESSION SET ENABLE_SPCS_SERVICE_FUNCTIONS_IN_AISQL = true",
            "ALTER SESSION SET SPCS_MODEL_INFERENCE_SERVER_ENABLE_AI_COMPLETE = true",
        ):
            try:
                self.session.sql(stmt).collect()
            except Exception as e:
                logger.warning("Could not set session parameter (%s): %s", stmt, e)

    def _invoke_ai_complete(self, service_name: str) -> Any:
        query = f"SELECT AI_COMPLETE('{service_name}', 'Say hello in one short sentence.') AS RESULT"

        def _run() -> Any:
            rows = self.session.sql(query).collect()
            self.assertGreater(len(rows), 0)
            row_dict = rows[0].as_dict()
            result = row_dict.get("RESULT")
            if result is None:
                result = row_dict.get("result")
            return result

        return self._retry_on_transient_inference(_run)

    def _assert_chat_completion(self, result: Any) -> None:
        payload = result
        if isinstance(result, pd.DataFrame):
            self.assertGreaterEqual(len(result), 1)
            payload = result.iloc[0].to_dict()
        payload = _maybe_json_load(payload)
        self.assertIsInstance(payload, dict)
        decoded_payload = {key: _maybe_json_load(value) for key, value in payload.items()}
        if "choices" not in decoded_payload:
            for value in decoded_payload.values():
                nested = _maybe_json_load(value)
                if isinstance(nested, dict) and "choices" in nested:
                    decoded_payload = {key: _maybe_json_load(inner) for key, inner in nested.items()}
                    break
        choices = _maybe_json_load(decoded_payload.get("choices"))
        self.assertIsInstance(choices, list)
        self.assertGreaterEqual(len(choices), 1)
        first_choice = _maybe_json_load(choices[0])
        self.assertIsInstance(first_choice, dict)
        message = _maybe_json_load(first_choice.get("message"))
        self.assertIsInstance(message, dict)
        content = message.get("content")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def _assert_ai_complete_text(self, result: Any) -> None:
        if isinstance(result, dict):
            self._assert_chat_completion(result)
            return
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def _require_model_garden_model(self) -> str:
        model_name = self._resolve_model_garden_model()
        if model_name is None:
            self.skipTest(f"No {_MODEL_GARDEN_MODEL_NAME} model in SNOWFLAKE.MODELS; skipping.")
        return model_name

    def test_model_garden_service(self) -> None:
        model_name = self._require_model_garden_model()

        model_fqn = f"SNOWFLAKE.MODELS.{model_name}"
        logger.info("Using Model Garden model %s", model_fqn)

        self._enable_ai_complete_session_params()

        service_short_name = self._service_short_name
        service_name = f"{self._test_db}.{self._test_schema}.{service_short_name}"
        mv = self._model_garden_model_version(model_name)
        mv.create_service(
            service_name=service_name,
            service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
            gpu_requests=_GPU_REQUESTS,
            cpu_requests=_CPU_REQUESTS,
            memory_requests=_MEMORY_REQUESTS,
            ingress_enabled=True,
            min_instances=_NODE_COUNT,
            max_instances=_NODE_COUNT,
            num_workers=1,
            autocapture=False,
            inference_engine_options={
                "engine": inference_engine.InferenceEngine.VLLM,
                "engine_args_override": ["--tensor-parallel-size=1"],
            },
        )
        self._wait_for_named_service_status(service_short_name, expected_status="RUNNING")

        # Each invocation path and the lifecycle check run independently so that one broken path still
        # reports the results of the others against the same running service.
        with self.subTest(invocation="service_function_sql"):
            self._assert_chat_completion(self._invoke_service_sql(service_name))
            logger.info("subTest passed: service_function_sql")

        with self.subTest(invocation="model_version_run"):
            self._assert_chat_completion(self._invoke_mv_run(service_name, model_name))
            logger.info("subTest passed: model_version_run")

        with self.subTest(invocation="ai_complete"):
            self._assert_ai_complete_text(self._invoke_ai_complete(service_name))
            logger.info("subTest passed: ai_complete")

        with self.subTest(lifecycle="suspend_resume"):
            self.session.sql(f"ALTER SERVICE {service_name} SUSPEND").collect()
            self._wait_for_named_service_status(service_short_name, expected_status="SUSPENDED")

            self.session.sql(f"ALTER SERVICE {service_name} RESUME").collect()
            self._wait_for_named_service_status(service_short_name, expected_status="RUNNING")

            self._assert_chat_completion(self._invoke_service_sql(service_name))
            logger.info("subTest passed: suspend_resume")

    def test_model_garden_batch_inference_job(self) -> None:
        model_name = self._require_model_garden_model()
        logger.info("Using Model Garden model SNOWFLAKE.MODELS.%s", model_name)

        mv = self._model_garden_model_version(model_name)
        job_short_name = f"MG_BATCH_{self._run_id}".upper()
        job_name = f"{self._test_db}.{self._test_schema}.{job_short_name}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{job_short_name}/output/"
        input_df = self.session.create_dataframe(self._openai_call_input())

        batch_job = mv.run_batch(
            input_df,
            compute_pool=self._TEST_GPU_COMPUTE_POOL,
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            resources_spec=batch_inference_job_specs.ResourcesSpec(
                cpu_requests=_CPU_REQUESTS,
                memory_requests=_MEMORY_REQUESTS,
                gpu_requests=_GPU_REQUESTS,
            ),
            inference_spec=batch_inference_job_specs.InferenceSpec(
                engine_options=batch_inference_job_specs.EngineOptions(
                    engine=inference_engine.InferenceEngine.VLLM,
                    engine_args_override=["--tensor-parallel-size=1"],
                )
            ),
            function_name="__call__",
            job_name=job_name,
            replicas=_NODE_COUNT,
        )

        try:
            batch_job.wait(timeout=1800)
        except TimeoutError:
            logger.warning("Batch job timed out after 30 minutes. Status: %s", batch_job.status)

        if batch_job.status != "DONE":
            logs = batch_job.get_logs(limit=100)
            self.fail(f"Job {batch_job.id} status is {batch_job.status}, expected DONE.\n\n{logs}")

        resolved_job_name = batch_job.id.split(".")[-1].strip('"')
        job_output_stage_location = output_stage_location.rstrip("/") + "/" + resolved_job_name + "/"
        success_file_path = job_output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file_path}").collect()
        self.assertGreater(len(list_results), 0, f"Batch job did not produce success file at: {success_file_path}")

        output_df = self.session.read.option("on_error", "CONTINUE").parquet(job_output_stage_location)
        self.assertEqual(output_df.count(), input_df.count())
        self._assert_chat_completion(output_df.to_pandas())
        logger.info("test_model_garden_batch_inference_job passed")


class TestOpenAIChatCompletionAssert(absltest.TestCase):
    """Parquet batch rows often store VARIANT fields as JSON strings."""

    def test_dataframe_choices_json_string(self) -> None:
        TestRegistryModelGardenDeploymentInteg._assert_chat_completion(
            self,
            pd.DataFrame(
                [
                    {
                        "choices": json.dumps([{"message": {"role": "assistant", "content": "Hello there."}}]),
                    }
                ]
            ),
        )

    def test_dataframe_wrapped_json_column(self) -> None:
        TestRegistryModelGardenDeploymentInteg._assert_chat_completion(
            self,
            pd.DataFrame(
                [
                    {
                        "messages": "[]",
                        "__CALL__": json.dumps({"choices": [{"message": {"content": "Hello there."}}]}),
                    }
                ]
            ),
        )

    def test_parsed_openai_dict(self) -> None:
        TestRegistryModelGardenDeploymentInteg._assert_chat_completion(
            self,
            {"choices": [{"message": {"role": "assistant", "content": "Hello there."}}]},
        )


if __name__ == "__main__":
    absltest.main()
