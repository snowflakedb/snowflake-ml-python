"""
Test the /ai_complete endpoint for AI_COMPLETE.
"""

import http
import json
import logging
import os
import tempfile
from typing import Any, Optional

import pandas as pd
import requests
from absl.testing import absltest
from retrying import retry

from snowflake.ml.model import ModelVersion
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


class TestAICompleteEndpointInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Integration tests for the /ai_complete endpoint."""

    # Class-level state populated on first setUp call
    _endpoint: Optional[str] = None
    _model_version: Optional[ModelVersion] = None
    _service_name: Optional[str] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    def setUp(self) -> None:
        super().setUp()
        # The AI_COMPLETE feature requires a proxy image with the endpoint code.
        # Override proxy image to one that has /ai_complete support.

        self.session.sql("ALTER SESSION SET SPCS_MODEL_INFERENCE_SERVER_ENABLE_AI_COMPLETE=true;").collect()
        self.session.sql("ALTER SESSION SET ENABLE_SPCS_SERVICE_FUNCTIONS_IN_AISQL=true;").collect()

        # Deploy once, reuse across all tests
        if TestAICompleteEndpointInteg._endpoint is None:
            self._deploy_test_service()

    def tearDown(self) -> None:
        # Don't call super().tearDown() per-test — it drops the DB which kills the shared service.
        # Cleanup happens in tearDownClass after all tests complete.
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # Drop the test DB (and service) after all tests are done
        if hasattr(cls, "_db_manager") and hasattr(cls, "_test_db"):
            cls._db_manager.drop_database(cls._test_db)
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint

    def _deploy_test_service(self) -> None:
        """Deploy a vLLM-backed text-generation model for all AI_COMPLETE tests."""
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            download_snapshot=False,
        )

        # Minimal input to satisfy model logging validation
        input_data = pd.DataFrame.from_records(
            [
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7,
                    "max_completion_tokens": 50,
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

        def check_res(res: pd.DataFrame) -> None:
            self.assertGreater(len(res), 0)

        model_version = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={"__call__": (input_data, check_res)},
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(InferenceEngine.VLLM),
        )

        TestAICompleteEndpointInteg._model_version = model_version
        TestAICompleteEndpointInteg._endpoint = self._ensure_ingress_url(model_version)
        TestAICompleteEndpointInteg._service_name = model_version.list_services().loc[0, "name"]
        logger.info(f"AI_COMPLETE test service deployed: {self._service_name} at {self._endpoint}")

    # ─── Helper Methods ───────────────────────────────────────────────

    @staticmethod
    def _retry_if_status_retriable(result: requests.Response) -> bool:
        """Retry on transient ingress errors (403 = 'could not find service' during propagation)."""
        return result.status_code in [
            http.HTTPStatus.UNAUTHORIZED,
            http.HTTPStatus.FORBIDDEN,
            http.HTTPStatus.TOO_MANY_REQUESTS,
            http.HTTPStatus.SERVICE_UNAVAILABLE,
            http.HTTPStatus.GATEWAY_TIMEOUT,
        ]

    def _make_ai_complete_request(self, payload: dict[str, Any]) -> requests.Response:
        """POST to /ai_complete with authentication."""
        auth_handler = self._get_auth_for_inference(self._endpoint)
        return retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=30000,
            retry_on_result=self._retry_if_status_retriable,
        )(requests.post)(
            f"https://{self._endpoint}/ai_complete",
            json=payload,
            auth=auth_handler,
            timeout=120,
        )

    # Used for malformed request tests.
    def _make_raw_request(self, body: str) -> requests.Response:
        """POST raw string body to /ai_complete (for malformed request tests)."""
        auth_handler = self._get_auth_for_inference(self._endpoint)
        return retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=30000,
            retry_on_result=self._retry_if_status_retriable,
        )(requests.post)(
            f"https://{self._endpoint}/ai_complete",
            data=body,
            headers={"Content-Type": "application/json"},
            auth=auth_handler,
            timeout=30,
        )

    def _build_row(
        self,
        row_idx: int,
        prompt: str,
        *,
        model_name: str = "test-model",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        provisioned_throughput_id: Any = None,
    ) -> list:
        """Build a single AI_COMPLETE-format row: [rowIdx, model, messages, ptId, options]."""
        if messages is None:
            messages = [{"role": "user", "content": {"template": "{0}", "args": [prompt]}}]

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if response_format is not None:
            options["response_format"] = response_format

        return [row_idx, model_name, messages, provisioned_throughput_id, options]

    def _assert_success_row(self, row: list, expected_idx: int) -> dict[str, Any]:
        """Assert row is [idx, {value: {...}, error: null}] and return value dict."""
        self.assertEqual(
            row[0],
            expected_idx,
            f"Row index mismatch: expected {expected_idx}, got {row[0]}",
        )
        result = row[1]
        self.assertIsNone(
            result["error"],
            f"Expected no error for row {expected_idx}, got: {result['error']}",
        )
        self.assertIsNotNone(result["value"], f"Expected value for row {expected_idx}, got None")
        return result["value"]

    def _assert_error_row(self, row: list, expected_idx: int) -> str:
        """Assert row is [idx, {value: null, error: "..."}] and return error string."""
        self.assertEqual(
            row[0],
            expected_idx,
            f"Row index mismatch: expected {expected_idx}, got {row[0]}",
        )
        result = row[1]
        self.assertIsNone(result["value"], f"Expected null value for error row {expected_idx}")
        self.assertIsNotNone(result["error"], f"Expected error for row {expected_idx}, got None")
        return result["error"]

    # ─── Test Cases ───────────────────────────────────────────────────

    def test_model_name_in_response_matches_request(self) -> None:
        """Verify the 'model' field in the response value echoes the model name from the request row."""
        model_name = "my-custom-model-name"
        payload = {"data": [self._build_row(0, "What is 2+2?", model_name=model_name)]}

        response = self._make_ai_complete_request(payload)
        if response.status_code != 200:
            # Log response for debugging
            logger.error(
                f"ai_complete returned {response.status_code}: {response.text[:500]}. "
                f"Request payload: {json.dumps(payload)[:500]}"
            )
        self.assertEqual(response.status_code, 200, f"Response body: {response.text[:500]}")

        data = response.json()["data"]
        self.assertLen(data, 1)

        value = self._assert_success_row(data[0], expected_idx=0)
        self.assertEqual(value["model"], model_name)

    def test_row_level_error(self) -> None:
        """Trigger a per-row error and verify the error field is populated."""
        # Empty messages array should cause vLLM to reject the request
        payload = {
            "data": [
                [0, "model-name", [], None, {}],
            ]
        }

        response = self._make_ai_complete_request(payload)
        # Response is always 200 (per-row errors go in the data envelope)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        self.assertLen(data, 1)
        error_msg = self._assert_error_row(data[0], expected_idx=0)
        self.assertGreater(len(error_msg), 0)

    def test_malformed_request_missing_columns_returns_400(self) -> None:
        """Send a row with fewer than 5 columns - expect HTTP 400."""
        payload = {
            "data": [
                [
                    0,
                    "model-name",
                    [{"role": "user", "content": {"template": "hi", "args": []}}],
                ]
                # Only 3 columns, need 5
            ]
        }

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 400)

    def test_malformed_request_invalid_json_returns_400(self) -> None:
        """Send garbage body - expect HTTP 400."""
        response = self._make_raw_request("this is not json{{{")
        self.assertEqual(response.status_code, 400)

    def test_empty_data_array_returns_400(self) -> None:
        """Send {"data": []} - expect HTTP 400 per proxy design (empty is error)."""
        payload = {"data": []}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 400)

    def test_all_rows_succeed_batch(self) -> None:
        """Multi-row batch where all rows get valid completions."""
        prompts = [
            "What color is the sky?",
            "Name a fruit.",
            "What is 1+1?",
            "Say hello in French.",
            "Name a planet.",
        ]
        payload = {"data": [self._build_row(i, prompt) for i, prompt in enumerate(prompts)]}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        self.assertLen(data, len(prompts))

        for i, row in enumerate(data):
            value = self._assert_success_row(row, expected_idx=i)
            self.assertIn("choices", value)
            self.assertGreater(len(value["choices"][0]["messages"]), 0)
            self.assertIn("usage", value)

    def test_mixed_response_format(self) -> None:
        """Some rows have response_format, some don't. Verify structured_output vs choices."""
        schema = {
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
        }

        payload = {
            "data": [
                # Row 0: no response_format -> choices
                self._build_row(0, "What is 2+2?"),
                # Row 1: with response_format -> structured_output
                self._build_row(
                    1,
                    "What color is the sky? Respond with JSON containing a 'color' field.",
                    response_format={"type": "json", "schema": schema},
                ),
                # Row 2: no response_format -> choices
                self._build_row(2, "Say hello."),
            ]
        }

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        self.assertLen(data, 3)

        # Row 0: choices (no response_format)
        value0 = self._assert_success_row(data[0], expected_idx=0)
        self.assertIn("choices", value0)
        self.assertNotIn("structured_output", value0)

        # Row 1: structured_output (response_format specified)
        value1 = self._assert_success_row(data[1], expected_idx=1)
        self.assertIn("structured_output", value1)
        self.assertNotIn("choices", value1)

        # Row 2: choices (no response_format)
        value2 = self._assert_success_row(data[2], expected_idx=2)
        self.assertIn("choices", value2)
        self.assertNotIn("structured_output", value2)

    def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversation: system + user + assistant + user messages."""
        messages = [
            {
                "role": "system",
                "content": {
                    "template": "{0}",
                    "args": ["You are a helpful assistant."],
                },
            },
            {
                "role": "user",
                "content": {
                    "template": "{0}",
                    "args": ["What is the capital of France?"],
                },
            },
            {
                "role": "assistant",
                "content": {
                    "template": "{0}",
                    "args": ["The capital of France is Paris."],
                },
            },
            {
                "role": "user",
                "content": {"template": "{0}", "args": ["What is its population?"]},
            },
        ]

        payload = {"data": [[0, "test-model", messages, None, {"temperature": 0.5}]]}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        self.assertLen(data, 1)
        value = self._assert_success_row(data[0], expected_idx=0)
        self.assertIn("choices", value)
        self.assertGreater(len(value["choices"][0]["messages"]), 0)

    def test_temperature_options_variation(self) -> None:
        """Different temperatures across rows - all should produce valid responses."""
        temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
        payload = {
            "data": [self._build_row(i, "Say a random word.", temperature=temp) for i, temp in enumerate(temperatures)]
        }

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        self.assertLen(data, len(temperatures))

        for i, row in enumerate(data):
            value = self._assert_success_row(row, expected_idx=i)
            self.assertIn("choices", value)
            self.assertGreater(
                len(value["choices"][0]["messages"]),
                0,
                f"Empty response for temperature={temperatures[i]}",
            )

    def test_max_tokens_limits_output(self) -> None:
        """Send with max_tokens=5 and verify output is short."""
        payload = {"data": [self._build_row(0, "Write a very long story about dragons.", max_tokens=5)]}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        value = self._assert_success_row(data[0], expected_idx=0)
        self.assertIn("usage", value)
        if value["usage"] is not None:
            self.assertLessEqual(value["usage"]["completion_tokens"], 5)

    def test_usage_fields_present(self) -> None:
        """Verify response contains usage with prompt_tokens, completion_tokens, total_tokens."""
        payload = {"data": [self._build_row(0, "Hello, how are you?")]}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        value = self._assert_success_row(data[0], expected_idx=0)
        self.assertIn("usage", value)
        self.assertIsNotNone(value["usage"])
        self.assertGreater(value["usage"]["prompt_tokens"], 0)
        self.assertGreater(value["usage"]["completion_tokens"], 0)
        self.assertGreater(value["usage"]["total_tokens"], 0)
        self.assertEqual(
            value["usage"]["total_tokens"],
            value["usage"]["prompt_tokens"] + value["usage"]["completion_tokens"],
        )

    def test_template_resolution_multiple_args(self) -> None:
        """Verify template with multiple args is resolved correctly."""
        messages = [
            {
                "role": "user",
                "content": {
                    "template": "Translate the word '{0}' to {1}. Reply with only the translation.",
                    "args": ["hello", "French"],
                },
            }
        ]

        payload = {"data": [[0, "test-model", messages, None, {"temperature": 0.3}]]}

        response = self._make_ai_complete_request(payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()["data"]
        value = self._assert_success_row(data[0], expected_idx=0)
        self.assertIn("choices", value)
        self.assertGreater(len(value["choices"][0]["messages"]), 0)

    # ─── SQL E2E Tests ────────────────────────────────────────────────────────
    # Test cases cover the use cases from the following public docs:
    # https://docs.snowflake.com/en/sql-reference/functions/ai_complete-single-string
    # https://docs.snowflake.com/en/sql-reference/functions/ai_complete-prompt-object (string prompts only)

    def test_sql_basic_completion(self) -> None:
        service_name = self._service_name
        result = self.session.sql(f"SELECT AI_COMPLETE('{service_name}', 'What is 2+2?')").collect()

        self.assertLen(result, 1)
        response_text = result[0][0]
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 0)

    def test_sql_prompt_with_concat(self) -> None:
        """SQL AI_COMPLETE with CONCAT-based prompt from table columns."""
        service_name = self._service_name
        self.session.sql("CREATE OR REPLACE TEMPORARY TABLE words (word VARCHAR, language VARCHAR)").collect()
        self.session.sql("INSERT INTO words VALUES ('hello', 'French'), ('goodbye', 'Spanish')").collect()

        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                '{service_name}',
                CONCAT('Translate the word ', word, ' to ', language, '. Reply with only the translation.')
            )
            FROM words
            ORDER BY word
        """
        ).collect()
        self.assertLen(result, 2)
        for row in result:
            self.assertIsNotNone(row[0])
            self.assertGreater(len(row[0]), 0)

    def test_sql_with_parameters(self) -> None:
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                model => '{service_name}',
                prompt => 'Say hello.',
                model_parameters => {{'temperature': 0.7, 'max_tokens': 50}}
            )
        """
        ).collect()

        self.assertLen(result, 1)
        response_text = result[0][0]
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 0)

    def test_sql_show_details(self) -> None:
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                model => '{service_name}',
                prompt => 'Hello',
                model_parameters => {{'max_tokens': 10}},
                show_details => true
            )
        """
        ).collect()

        self.assertLen(result, 1)
        response_value = result[0][0]
        self.assertIsNotNone(response_value)

        parsed = json.loads(response_value)

        self.assertIn("choices", parsed)
        self.assertIn("model", parsed)
        self.assertEqual(parsed["model"], service_name)
        self.assertIn("usage", parsed)

        self.assertIn("completion_tokens", parsed["usage"])
        self.assertIn("prompt_tokens", parsed["usage"])
        self.assertIn("total_tokens", parsed["usage"])

    def test_sql_structured_output_sql_definition(self) -> None:
        service_name = self._service_name
        prompt = (
            "Extract structured data from this customer interaction note: "
            "Customer Sarah Jones complained about the mobile app crashing during checkout. "
            "She tried to purchase 3 items: a red XL jacket ($89.99), blue running shoes ($129.50), "
            "and a fitness tracker ($199.00). The app crashed after she entered her shipping address "
            "at 123 Main St, Portland OR, 97201. She has been a premium member since January 2024."
        )
        response_format = (
            "TYPE OBJECT(note OBJECT(items_count NUMBER, " "price ARRAY(STRING), address STRING, member_date STRING))"
        )
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                model => '{service_name}',
                prompt => '{prompt}',
                model_parameters => {{'temperature': 0, 'max_tokens': 100}},
                response_format => {response_format}
            )
        """
        ).collect()

        self.assertLen(result, 1)
        response_value = result[0][0]
        self.assertIsNotNone(response_value)

        parsed = json.loads(response_value)

        self.assertIn("note", parsed)
        self.assertIn("items_count", parsed["note"])
        self.assertIn("price", parsed["note"])
        self.assertIn("address", parsed["note"])

    def test_sql_structured_output_json_schema(self) -> None:
        service_name = self._service_name
        prompt = (
            "Extract structured data from this customer interaction note: "
            "Customer Sarah Jones complained about the mobile app crashing during checkout. "
            "She tried to purchase 3 items: a red XL jacket ($89.99), blue running shoes ($129.50), "
            "and a fitness tracker ($199.00). The app crashed after she entered her shipping address "
            "at 123 Main St, Portland OR, 97201. She has been a premium member since January 2024."
        )
        schema = (
            "{'type':'json','schema':{'type':'object','properties':"
            "{'note':{'type':'object','properties':"
            "{'items_count':{'type':'number'},"
            "'price':{'type':'array','items':{'type':'string'}},"
            "'address':{'type':'string'},"
            "'member_date':{'type':'string'}},"
            "'required':['items_count','price','address','member_date']}}}}"
        )
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
            model => '{service_name}',
            prompt => '{prompt}',
            model_parameters => {{'temperature': 0, 'max_tokens': 4096}},
            response_format => {schema},
            show_details => true
        )
        """
        ).collect()

        self.assertLen(result, 1)
        response_value = result[0][0]
        self.assertIsNotNone(response_value)

        parsed = json.loads(response_value)

        # With show_details, structured output is nested in the response envelope
        self.assertIn("structured_output", parsed)
        self.assertIn("usage", parsed)
        raw_message = parsed["structured_output"][0]["raw_message"]
        self.assertIn("note", raw_message)
        self.assertIn("items_count", raw_message["note"])
        self.assertIn("price", raw_message["note"])
        self.assertIn("address", raw_message["note"])
        self.assertIn("member_date", raw_message["note"])

    def test_sql_max_tokens_limits_output(self) -> None:
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                model => '{service_name}',
                prompt => 'Write a very long story about dragons.',
                model_parameters => {{'max_tokens': 5}},
                show_details => true
            )
        """
        ).collect()
        self.assertLen(result, 1)
        parsed = json.loads(result[0][0]) if isinstance(result[0][0], str) else result[0][0]
        self.assertIn("usage", parsed)
        self.assertLessEqual(parsed["usage"]["completion_tokens"], 5)

    def test_sql_prompt_object(self) -> None:
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                '{service_name}',
                PROMPT('Translate the word {{0}} to {{1}}. Reply with only the translation.', 'hello', 'French')
            )
        """
        ).collect()
        self.assertLen(result, 1)
        response_text = result[0][0]
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 0)

    def test_sql_message_array_with_options(self) -> None:
        """SQL AI_COMPLETE with inline message array and options object."""
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                '{service_name}',
                [{{'role':'user','content':'What is 2+2?'}}],
                {{'temperature': 0.7, 'max_tokens': 50}}
            )
        """
        ).collect()

        self.assertLen(result, 1)
        response_text = result[0][0]
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 0)

    def test_sql_array_construct_object_construct(self) -> None:
        """SQL AI_COMPLETE with ARRAY_CONSTRUCT(OBJECT_CONSTRUCT(...)) message syntax."""
        service_name = self._service_name
        result = self.session.sql(
            f"""
            SELECT AI_COMPLETE(
                '{service_name}',
                ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('role','user','content','Say hello'))
            )
        """
        ).collect()

        self.assertLen(result, 1)
        response_text = result[0][0]
        self.assertIsNotNone(response_text)
        self.assertGreater(len(response_text), 0)

    def test_sql_ai_complete_on_table_column(self) -> None:
        service_name = self._service_name
        self.session.sql("CREATE OR REPLACE TEMPORARY TABLE reviews (id INT, review_text VARCHAR)").collect()
        self.session.sql(
            "INSERT INTO reviews VALUES "
            "(1, 'Great product, fast shipping!'), "
            "(2, 'Terrible quality, broke after one day.'), "
            "(3, 'Average experience, nothing special.')"
        ).collect()

        result = self.session.sql(
            f"""
            SELECT id, AI_COMPLETE(
                '{service_name}',
                PROMPT(
                    'Classify this review as positive, negative, or neutral. Reply with one word only: {{0}}',
                    review_text
                )
            ) AS sentiment
            FROM reviews
            ORDER BY id
        """
        ).collect()

        self.assertLen(result, 3)
        for row in result:
            self.assertIsNotNone(row[1])
            self.assertGreater(len(row[1]), 0)


if __name__ == "__main__":
    absltest.main()
