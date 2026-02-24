"""Base class for vLLM inference server with runtime parameters (ParamSpec) tests.

This module provides common functionality for testing vLLM models with ParamSpec parameters.
"""

import os
import tempfile
from typing import Any

import pandas as pd
import requests

from snowflake.ml.model import inference_engine
from snowflake.ml.model.models import huggingface
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class RegistryVLLMParamsTestBase(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Base class for vLLM inference with runtime parameters tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_hf_endpoint = None
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        # Get HF token if available (used for gated models)
        cls.hf_token = os.getenv("HF_TOKEN", None)
        # Unset HF_ENDPOINT to avoid artifactory errors
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir is not None:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            del os.environ["TRANSFORMERS_CACHE"]
        if cls._original_hf_home is not None:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            del os.environ["HF_HOME"]
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint

    def _create_vllm_model(self) -> huggingface.TransformersPipeline:
        """Create a HuggingFace model for vLLM testing."""
        return huggingface.TransformersPipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            compute_pool_for_log=None,
        )

    def _get_vllm_inference_engine_options(self) -> dict[str, Any]:
        """Get inference engine options for vLLM."""
        return {"engine": inference_engine.InferenceEngine.VLLM}

    def _create_test_input(self, prompt: str = "What is the capital of France?") -> pd.DataFrame:
        """Create test input with only the messages column.

        When using OPENAI_CHAT_WITH_PARAMS_SIGNATURE, the input DataFrame only contains
        the 'messages' column. Parameters are supplied separately via:
        - params argument in mv.run()
        - top-level 'params' key in split/records REST formats
        - columns added to flat format REST API (handled by base class or manually)

        Args:
            prompt: The user prompt to include in the messages.

        Returns:
            DataFrame with a single 'messages' column containing the chat messages.
        """
        return pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        },
                    ],
                }
            ],
        )

    def _validate_openai_response(self, res: pd.DataFrame, expected_num_choices: int = 1) -> None:
        """Validate the OpenAI-format response from vLLM."""
        pd.testing.assert_index_equal(
            res.columns,
            pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
            check_order=False,
        )

        self.assertEqual(len(res), 1, "Expected single response row")

        for row in res["choices"]:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), expected_num_choices, f"Expected {expected_num_choices} choices")
            self.assertIn("message", row[0])
            self.assertIn("content", row[0]["message"])
            self.assertGreater(len(row[0]["message"]["content"]), 0, "Response content should not be empty")

    def _validate_openai_response_with_content(
        self,
        res: pd.DataFrame,
        expected_phrases: list[str],
        expected_num_choices: int = 1,
    ) -> None:
        """Validate OpenAI-format response structure and check for expected phrases in content.

        Args:
            res: DataFrame containing the OpenAI-format response.
            expected_phrases: List of phrases that should appear in the output (case-insensitive).
            expected_num_choices: Expected number of choices in the response.
        """
        # First validate structure
        self._validate_openai_response(res, expected_num_choices)

        # Extract all content from choices
        all_content = []
        for row in res["choices"]:
            for choice in row:
                message = choice.get("message", {})
                content = message.get("content", "")
                if content:
                    all_content.append(str(content))

        output_text = " ".join(all_content).lower()

        # If no content found, show helpful debug info
        if not output_text.strip():
            self.fail(f"No content found. Columns: {list(res.columns)}. DataFrame:\n{res.to_string()}")

        # Check for expected phrases
        for phrase in expected_phrases:
            self.assertIn(
                phrase.lower(),
                output_text,
                f"Expected phrase '{phrase}' not found in output. Output text: {output_text[:1000]}...",
            )

    def _make_rest_api_request(
        self,
        endpoint: str,
        request_payload: dict[str, Any],
        target_method: str,
    ) -> requests.Response:
        """Make a REST API request with a custom payload format."""
        auth_handler = self._get_auth_for_inference(endpoint)
        return requests.post(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=request_payload,
            auth=auth_handler,
            timeout=60,
        )

    def _parse_response_data(self, response: requests.Response) -> pd.DataFrame:
        """Parse REST API response into a DataFrame."""
        response.raise_for_status()
        response_data = response.json()["data"]
        # Response format: [[row_idx, {col1: val1, ...}], ...]
        return pd.DataFrame([x[1] for x in response_data])

    def _create_flat_format_payload_with_params(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        max_completion_tokens: int = 250,
        n: int = 1,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> dict[str, Any]:
        """Create flat format (external data format) payload with params as columns.

        For flat format REST API calls, params are included as columns in the data array.
        Format: {"data": [[row_idx, messages, temp, max_tokens, stop, n, stream, top_p, freq_pen, pres_pen]]}

        Args:
            messages: List of chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_completion_tokens: Maximum number of tokens to generate.
            n: Number of completions to generate.
            top_p: Nucleus sampling probability.
            frequency_penalty: Frequency penalty for token repetition.
            presence_penalty: Presence penalty for token repetition.

        Returns:
            Dict containing the flat format payload for REST API.
        """
        return {
            "data": [
                [
                    0,  # row index
                    messages,
                    temperature,
                    max_completion_tokens,
                    None,  # stop
                    n,
                    False,  # stream
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                ]
            ]
        }
