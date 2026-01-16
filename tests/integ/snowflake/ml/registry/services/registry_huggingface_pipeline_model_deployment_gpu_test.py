import os
import tempfile
from typing import Any, Optional

import pandas as pd
import pytest
from absl.testing import absltest, parameterized

from snowflake.ml.model import inference_engine, openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.models import huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryHuggingFacePipelineDeploymentGPUModelInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir.name
        # Get HF token if available (used for gated models)
        self.hf_token = os.getenv("HF_TOKEN", None)
        # Unset HF_ENDPOINT to avoid artifactory errors
        # TODO: Remove this once artifactory is fixed
        if "HF_ENDPOINT" in os.environ:
            self._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self._original_cache_dir
        self.cache_dir.cleanup()
        if self._original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = self._original_hf_endpoint

    def _get_inference_engine_options_for_inference_engine(
        self,
        inference_engine_type: str,
        base_inference_engine_options: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Helper method to generate inference_engine_options based on inference engine type.

        Args:
            inference_engine_type: Inference engine type - either "Default" (Python) or "vLLM"
            base_inference_engine_options: Base inference engine options to merge with inference engine-specific options

        Returns:
            Dictionary of inference engine options or None for Default backend
        """
        inference_engine_options = base_inference_engine_options.copy() if base_inference_engine_options else {}

        if inference_engine_type == "vLLM":
            inference_engine_options["engine"] = inference_engine.InferenceEngine.VLLM
        elif inference_engine_type != "Default":
            raise ValueError(f"Unknown inference engine type: {inference_engine_type}. Must be 'Default' or 'vLLM'")

        # Return None for Default backend if no other options are set
        if inference_engine_type == "Default" and not inference_engine_options:
            return None

        return inference_engine_options if inference_engine_options else None

    def _test_with_model_logging(
        self,
        model_name: str,
        inference_engine: str = "Default",
        requires_token: bool = False,
        task: str = "text-generation",
        base_inference_engine_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Helper method to test with model logging.

        Tests both single-row and batch inference to ensure the model handles
        multiple records correctly.

        Args:
            model_name: HuggingFace model identifier
            inference_engine: Inference engine type - either "Default" (Python) or "vLLM" (default: "Default")
            requires_token: Whether the model is gated and requires HF token
        """
        # Skip test if token is required but not available
        if requires_token and not self.hf_token:
            self.skipTest(f"Skipping test for gated model {model_name} - HF_TOKEN not available")

        # TODO: Remove this once the proxy image that can handle content parts protocol
        # is available in system repository.
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        model = huggingface_pipeline.HuggingFacePipelineModel(
            task=task,
            model=model_name,
            download_snapshot=False,
            token=self.hf_token if requires_token else None,
        )

        x_df_single = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Complete the sentence.",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",  # noqa: E501
                                },
                            ],
                        },
                    ],
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": 3,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                }
            ],
        )

        def check_single_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])
                self.assertGreater(len(row[0]["message"]["content"]), 0)

        # Deploy model and test with single row
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (
                    x_df_single,
                    check_single_res,
                ),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            use_model_logging=True,
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(
                inference_engine,
                base_inference_engine_options,
            ),
        )

        test_prompts = [
            "What is the capital of France?",
            "Write a short poem about the ocean.",
            "Explain what machine learning is in one sentence.",
            "What are the three primary colors?",
            "How do you say hello in Spanish?",
            "What is 15 multiplied by 7?",
            "Name a famous scientist.",
            "What is the boiling point of water?",
            "Describe a sunset in a few words.",
            "What programming language is used for data science?",
        ]

        x_df_batch = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.7,
                    "max_completion_tokens": 200,
                    "stop": None,
                    "n": 2,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                }
                for prompt in test_prompts
            ],
        )

        def check_batch_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            self.assertEqual(len(res), len(test_prompts))

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 2)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])
                self.assertGreater(len(row[0]["message"]["content"]), 0)

        service_name = mv.list_services().loc[0, "name"]
        endpoint = self._ensure_ingress_url(mv)
        jwt_token_generator = self._get_jwt_token_generator()

        res_service = mv.run(x_df_batch, function_name="__call__", service_name=service_name)
        check_batch_res(res_service)

        res_api = self._inference_using_rest_api(
            x_df_batch,
            endpoint=endpoint,
            jwt_token_generator=jwt_token_generator,
            target_method="__call__",
        )
        check_batch_res(res_api)

    def _test_text_generation(
        self,
        pip_requirements: Optional[list[str]],
        use_default_repo: bool,
        inference_engine: str = "Default",
    ) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            max_length=200,
        )

        x = [
            [
                {"role": "system", "content": "Complete the sentence."},
                {
                    "role": "user",
                    "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                },
            ]
        ]

        x_df = pd.DataFrame([x], columns=["inputs"])

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertIn("generated_text", row[0])

        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (
                    x_df,
                    check_res,
                ),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            pip_requirements=pip_requirements,
            use_default_repo=use_default_repo,
            inference_engine_options=self._get_inference_engine_options_for_inference_engine(inference_engine),
        )

    @parameterized.product(  # type: ignore[misc]
        pip_requirements=[None, ["transformers"]],
    )
    def test_text_generation(
        self,
        pip_requirements: Optional[list[str]],
    ) -> None:
        self._test_text_generation(pip_requirements, use_default_repo=False)

    def test_text_generation_with_default_repo(
        self,
    ) -> None:
        self._test_text_generation(None, use_default_repo=True)

    def test_text_generation_with_model_logging_tiny_llama(self) -> None:
        """Test text generation with TinyLlama model."""
        self._test_with_model_logging(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            requires_token=False,
        )

    @parameterized.parameters(  # type: ignore[misc]
        "Default",
        "vLLM",
    )
    @pytest.mark.conda_incompatible
    def test_text_generation_with_model_logging_qwen(self, inference_engine: str) -> None:
        """Test text generation with Qwen model."""
        self._test_with_model_logging(
            model_name="Qwen/Qwen3-1.7B",
            inference_engine=inference_engine,
            requires_token=False,
        )

    @pytest.mark.conda_incompatible
    def test_text_generation_with_model_logging_gemma(self) -> None:
        """Test text generation with Gemma model (gated - requires HF token) using vLLM."""
        self._test_with_model_logging(
            model_name="google/gemma-3-1b-it",
            inference_engine="vLLM",
            requires_token=True,  # This is a gated model
        )

    @pytest.mark.conda_incompatible
    def test_image_text_to_text_with_model_logging_gemma(self) -> None:
        """Test image text to text with Gemma model (gated - requires HF token) using vLLM."""
        self._test_with_model_logging(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            task="image-text-to-text",
            inference_engine="vLLM",
            base_inference_engine_options={
                "engine_args_override": [
                    "--gpu-memory-utilization=0.9",
                    "--max-model-len=1024",
                ],
            },
        )

    @pytest.mark.conda_incompatible
    def test_create_service_signature_validation(self) -> None:
        """Test mv.create_service API flow with openai chat signature validation."""
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B",
        )

        # Define test data in OpenAI chat format
        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {"role": "system", "content": "Complete the sentence."},
                        {
                            "role": "user",
                            "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                        },
                    ],
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": 3,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])
                self.assertGreater(len(row[0]["message"]["content"]), 0)

        # Test with use_model_logging=False and OPENAI_CHAT_SIGNATURE
        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (
                    x_df,
                    check_res,
                ),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            gpu_requests="1",
            use_model_logging=False,
            inference_engine_options=self._get_inference_engine_options_for_inference_engine("vLLM"),
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )


if __name__ == "__main__":
    absltest.main()
