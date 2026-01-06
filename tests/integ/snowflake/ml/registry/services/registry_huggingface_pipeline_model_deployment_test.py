import os
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import compute_pool, openai_signatures
from snowflake.ml.model.models import huggingface, huggingface_pipeline
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryHuggingFacePipelineDeploymentModelInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir.name

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

    @parameterized.product(  # type: ignore[misc]
        pip_requirements=[None, ["transformers", "torch==2.6.0"]],
    )
    # TODO: Remove torch dependency when the version is available in go/conda
    def test_text_generation(
        self,
        pip_requirements: Optional[list[str]],
    ) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            max_length=200,
        )

        NUM_CHOICES = 3
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
                    "n": NUM_CHOICES,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1,
                }
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), NUM_CHOICES)
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
            options={},
            pip_requirements=pip_requirements,
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )

    def test_token_classification_with_model_logging(self) -> None:
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="token-classification",
            model="dslim/bert-base-NER",
            download_snapshot=False,
        )

        x_df = pd.DataFrame(
            [
                ["My name is Izumi and I live in Tokyo, Japan."],
            ],
            columns=["inputs"],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertIn("entity", row[0])
                self.assertIn("score", row[0])
                self.assertIn("index", row[0])
                self.assertIn("word", row[0])
                self.assertIn("start", row[0])
                self.assertIn("end", row[0])

        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (
                    x_df,
                    check_res,
                ),
            },
            use_model_logging=True,
        )

    @parameterized.product(  # type: ignore[misc]
        compute_pool_for_log=[
            compute_pool.DEFAULT_CPU_COMPUTE_POOL,
            compute_pool.DEFAULT_GPU_COMPUTE_POOL,
            None,
        ],
    )
    def test_remote_log_model(self, compute_pool_for_log: Optional[str]) -> None:
        if compute_pool_for_log is compute_pool.DEFAULT_CPU_COMPUTE_POOL:
            # test the default behavior, do not pass compute_pool_for_log
            model = huggingface.TransformersPipeline(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
            )
        else:
            # test
            # 1. the remote logging behavior, pass compute_pool_for_log
            # 2. the local mode behavior, pass None
            model = huggingface.TransformersPipeline(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
                compute_pool_for_log=compute_pool_for_log,
            )
        mv = self.registry.log_model(
            model=model,
            model_name="tinyllama_remote_log",
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )

        assert mv is not None


if __name__ == "__main__":
    absltest.main()
