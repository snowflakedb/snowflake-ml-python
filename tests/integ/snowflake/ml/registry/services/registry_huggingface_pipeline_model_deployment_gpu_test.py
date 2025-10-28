import os
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

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

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self._original_cache_dir
        self.cache_dir.cleanup()

    def _test_text_generation_with_model_logging(
        self,
        model_name: str,
        requires_token: bool = False,
    ) -> None:
        """Helper method to test text generation with model logging.

        Args:
            model_name: HuggingFace model identifier
            requires_token: Whether the model is gated and requires HF token
        """
        # Skip test if token is required but not available
        if requires_token and not self.hf_token:
            self.skipTest(f"Skipping test for gated model {model_name} - HF_TOKEN not available")

        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model=model_name,
            download_snapshot=False,
            token=self.hf_token if requires_token else None,
        )

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
            use_model_logging=True,
        )

    def _test_text_generation(
        self,
        pip_requirements: Optional[list[str]],
        use_default_repo: bool,
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
        self._test_text_generation_with_model_logging(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            requires_token=False,
        )

    def test_text_generation_with_model_logging_qwen(self) -> None:
        """Test text generation with Qwen model."""
        self._test_text_generation_with_model_logging(
            model_name="Qwen/Qwen3-1.7B",
            requires_token=False,
        )

    def test_text_generation_with_model_logging_gemma(self) -> None:
        """Test text generation with Gemma model (gated - requires HF token)."""
        self._test_text_generation_with_model_logging(
            model_name="google/gemma-3-1b-it",
            requires_token=True,  # This is a gated model
        )


if __name__ == "__main__":
    absltest.main()
